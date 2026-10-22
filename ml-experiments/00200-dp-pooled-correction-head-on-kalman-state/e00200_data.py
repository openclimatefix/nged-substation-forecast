"""Rows for experiment 200's correction head, built from arm D's stored forecasts.

A row is one (substation, run, valid time) of a run's day-ahead window. Every input is known at the
run's origin: the errors (actual minus arm D mean) up to 3.5 hours before the origin, never the
target's own error. `error_grid` puts each substation's shortest-lead error on a half-hourly grid,
so that the injection of the known-answer gate can edit the grid and rebuild the inputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 200
SLUG: Final[str] = "dp-pooled-correction-head-on-kalman-state"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
SRC_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00155-dp-bg-stage2-assembly-night-offset-and-error-correction"
    / "work"
    / "dp_preds"
)
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LAG_HOURS: Final[float] = 3.5
N_RECENT: Final[int] = 12
"""Half-hours of recent errors, the last 6 hours before the cut-off."""
PROFILE_DAYS: Final[int] = 14
WEEKS: Final[int] = 4
"""Same weekday and half-hour errors of the last 4 weeks that are known at the cut-off."""
LEAD_MIN: Final[float] = 12.0
LEAD_MAX: Final[float] = 48.0
SCORE_LEAD_MIN: Final[float] = 15.5
SCORE_LEAD_MAX: Final[float] = 39.0
R_KAPPA: Final[float] = 0.75
STEP: Final[pl.Duration] = pl.duration(minutes=30)  # ty: ignore[invalid-assignment]


@dataclass
class Site:
    """One substation's arm D rows and its half-hourly error grid."""

    time_series_id: int
    rows: pl.DataFrame
    t0: np.datetime64
    error_grid: np.ndarray


def load_site(*, time_series_id: int) -> Site:
    """Read arm D (seed 0, all periods) for one substation and build its error grid."""
    frames = [
        pl.read_parquet(path)
        for path in sorted(SRC_DIR.glob("d_*_D*.parquet"))
        if "spread" not in path.name
    ]
    rows = (
        pl.concat(frames)
        .filter((pl.col("arm") == "d") & (pl.col("time_series_id") == time_series_id))
        .filter(pl.col("seed") == 0)
        .unique(subset=["power_fcst_init_time", "valid_time"])
        .sort("power_fcst_init_time", "valid_time")
    )
    short = (
        rows.sort("valid_time", "lead_hours")
        .unique(subset="valid_time", keep="first")
        .select("valid_time", error=pl.col("actual") - pl.col("q50"))
        .sort("valid_time")
    )
    t0 = short["valid_time"].min()
    n = int((short["valid_time"].max() - t0).total_seconds() // 1800) + 1  # ty: ignore[unsupported-operator]
    grid = np.full(n, np.nan, dtype=np.float32)
    idx = ((short["valid_time"] - t0).dt.total_seconds() // 1800).to_numpy().astype(np.int64)
    grid[idx] = short["error"].to_numpy()
    return Site(
        time_series_id=time_series_id,
        rows=rows,
        t0=np.datetime64(t0.replace(tzinfo=None), "us"),  # ty: ignore[missing-argument, no-matching-overload, unknown-argument, unresolved-attribute]
        error_grid=grid,
    )


def build_features(*, site: Site, error_grid: np.ndarray) -> dict[str, np.ndarray]:
    """Features and targets for the site's rows with leads in the training window.

    `error_grid` is passed in so that the gate can pass an injected copy.
    """
    rows = site.rows.filter(pl.col("lead_hours").is_between(LEAD_MIN, LEAD_MAX))
    valid = rows["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    init = (
        rows["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    )
    half = np.timedelta64(30, "m")
    idx_v = ((valid - site.t0) // half).astype(np.int64)
    idx_k = ((init - np.timedelta64(int(LAG_HOURS * 60), "m") - site.t0) // half).astype(np.int64)
    n = len(error_grid)

    recent = np.full((len(rows), N_RECENT), np.nan, dtype=np.float32)
    for j in range(N_RECENT):
        pos = idx_k - j
        ok = (pos >= 0) & (pos < n)
        recent[ok, j] = error_grid[pos[ok]]

    k_min = np.ceil((idx_v - idx_k) / 48.0).astype(np.int64)
    profile_sum = np.zeros(len(rows), dtype=np.float64)
    profile_n = np.zeros(len(rows), dtype=np.float64)
    for j in range(PROFILE_DAYS):
        pos = idx_v - 48 * (k_min + j)
        ok = (pos >= 0) & (pos < n)
        val = np.where(ok, error_grid[np.clip(pos, 0, n - 1)], np.nan)
        good = ~np.isnan(val)
        profile_sum += np.where(good, val, 0.0)
        profile_n += good
    profile = np.where(profile_n > 0, profile_sum / np.maximum(profile_n, 1), 0.0).astype(
        np.float32
    )

    k_week = np.ceil((idx_v - idx_k) / 336.0).astype(np.int64)
    week_sum = np.zeros(len(rows), dtype=np.float64)
    week_n = np.zeros(len(rows), dtype=np.float64)
    for j in range(WEEKS):
        pos = idx_v - 336 * (k_week + j)
        ok = (pos >= 0) & (pos < n)
        val = np.where(ok, error_grid[np.clip(pos, 0, n - 1)], np.nan)
        good = ~np.isnan(val)
        week_sum += np.where(good, val, 0.0)
        week_n += good
    weekly = np.where(week_n > 0, week_sum / np.maximum(week_n, 1), 0.0).astype(np.float32)

    acc = np.zeros(len(rows))
    cnt = np.zeros(len(rows))
    for j in range(48):
        pos = idx_k - j
        ok = (pos >= 0) & (pos < n)
        val = np.where(ok, error_grid[np.clip(pos, 0, n - 1)], np.nan)
        acc += np.where(np.isnan(val), 0.0, val)
        cnt += ~np.isnan(val)
    day_recent = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0).astype(np.float32)

    actual = rows["actual"].to_numpy()
    mean = rows["mean"].to_numpy()
    target = (actual - rows["q50"].to_numpy()).astype(np.float32)
    # The target is actual minus mean at the (possibly injected) grid value of the row's own time.
    injected_delta = np.where(
        (idx_v >= 0) & (idx_v < n), error_grid[np.clip(idx_v, 0, n - 1)], np.nan
    )
    base_delta = site.error_grid[np.clip(idx_v, 0, len(site.error_grid) - 1)]
    shift = np.nan_to_num(injected_delta - base_delta, nan=0.0)
    target = target + shift.astype(np.float32)

    quantiles = np.stack([rows[q].to_numpy() - rows["q50"].to_numpy() for q in QUANTILES], axis=1)
    hod = ((valid - valid.astype("datetime64[D]")) // half).astype(np.int64)
    dow = ((valid.astype("datetime64[D]").astype(np.int64) + 3) % 7).astype(np.int64)
    return {
        "recent": np.nan_to_num(recent, nan=0.0),
        "recent_ok": (~np.isnan(recent)).mean(axis=1).astype(np.float32),
        "profile": profile,
        "weekly": weekly,
        "weekly_n": (week_n / WEEKS).astype(np.float32),
        "profile_n": (profile_n / PROFILE_DAYS).astype(np.float32),
        "day_recent": day_recent,
        "lead": rows["lead_hours"].to_numpy().astype(np.float32),
        "hod": hod,
        "dow": dow,
        "mean": mean.astype(np.float32),
        "target": target,
        "q_offsets": quantiles.astype(np.float32),
        "valid": valid,
        "init": init,
        "actual": actual.astype(np.float32),
        "q50": rows["q50"].to_numpy().astype(np.float32),
        "quantiles": rows.select(QUANTILES).to_numpy().astype(np.float32),
        "switching": rows["switching_mask"].to_numpy(),
    }
