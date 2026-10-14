"""Paths, series, grids, and the recent-power inputs of experiment 00008.

The experiment asks whether summaries of recent power help the stage 2 calibrator of experiment
00005 (`docs/ml-experiments/00008-lag-summaries-in-stage2-calibrator-demand/result.md`). It reads
00005's stage 1 statistics and stage 1a predictions without refitting them, so every arm sees the
same stage 1 inputs. Everything runs on one dense half-hourly grid per series, so a lag is an index
offset and a forecast origin is an index. The newest observation a forecast initialised at `T` may
use is the half-hour before `T`.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 8
SLUG: Final[str] = "lag-summaries-in-stage2-calibrator-demand"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
CONFIG: Final[str] = "from2023"
"""Experiment 00005's primary training window, from 2023-09."""

SMOKE: Final[bool] = os.environ.get("E00008_SMOKE") == "1"
"""A smoke test uses two series and writes under `E00008_WORK_DIR`."""

WORK_DIR: Final[Path] = (
    Path(os.environ["E00008_WORK_DIR"])
    if SMOKE
    else ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
)
"""Intermediate tables; private and never committed."""
E00005_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00005-lag-placement-stage1b-vs-stage2-demand" / "work"
)
"""Experiment 00005's work folder, read only: stage 1 statistics and stage 1a predictions."""
STATS_DIR: Final[Path] = E00005_WORK_DIR / "stats"

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
DROPPED_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
SUBSTATION_TYPES: Final[frozenset[str]] = frozenset({"Primary", "BSP", "GSP"})

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP_MINUTES: Final[int] = 30
STEPS_PER_DAY: Final[int] = 48
STEPS_PER_WEEK: Final[int] = 7 * STEPS_PER_DAY
MIN_OBSERVED: Final[int] = 24
"""A summary window needs at least this many observed half-hours of its 48, or it is null."""

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("1-2 d", 24.0, 48.0),
    ("3-6 d", 48.0, 144.0),
    ("7-10 d", 144.0, 240.0),
    ("11-15 d", 240.0, 10_000.0),
)

LAG_STEPS: Final[tuple[int, ...]] = (48, 96, 336, 672)
LAG_NAMES: Final[tuple[str, ...]] = ("lag_24h", "lag_48h", "lag_168h", "lag_336h")
RESIDUAL_COLUMNS: Final[tuple[str, ...]] = (
    "residual_origin",
    "residual_origin_1h",
    "residual_origin_3h",
    "residual_same_time_of_day",
    "residual_same_time_of_week",
)
POWER_LAG_COLUMNS: Final[tuple[str, ...]] = tuple(f"power_{name}" for name in LAG_NAMES)
ANOMALY_COLUMNS: Final[tuple[str, ...]] = tuple(f"anomaly_{name}" for name in LAG_NAMES)
SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "last_24h_min",
    "last_24h_mean",
    "last_24h_max",
    "same_weekday_min",
    "same_weekday_mean",
    "same_weekday_max",
)
RECENT_COLUMNS: Final[tuple[str, ...]] = (
    *RESIDUAL_COLUMNS,
    *POWER_LAG_COLUMNS,
    *ANOMALY_COLUMNS,
    *SUMMARY_COLUMNS,
)
"""Every recent-power input of any arm, in the order `recent_inputs` returns them."""

CALIBRATOR_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 12,
}
CALIBRATOR_ROUNDS: Final[int] = 300


def series_table(*, full: bool = False) -> pl.DataFrame:
    """Return the 20 substation series with label, substation type, and effective capacity.

    A smoke test returns two series unless `full` is set.

    Args:
        full: Return all 20 series even in a smoke test.

    Returns:
        One row per series with `time_series_id`, `site` (D1 to D22), `substation_type`, and
        `effective_capacity_mw`, sorted by the site's number.

    Raises:
        ValueError: If the table does not hold 20 Primary, BSP, or GSP series.
    """
    table = pl.read_parquet(SERIES_PATH)
    metadata = pl.read_parquet(
        study_dir_for(study="quick_wins_fixed_pipeline_1156")
        / "frames"
        / "combined"
        / "fold_a"
        / "metadata.parquet",
        columns=["time_series_id", "substation_type"],
    )
    kept = (
        table.filter(~pl.col("site").is_in(DROPPED_SITES))
        .join(metadata, on="time_series_id", how="left", validate="1:1")
        .with_columns(pl.col("substation_type").cast(pl.String))
        .select("time_series_id", "site", "substation_type", "effective_capacity_mw")
        .sort(pl.col("site").str.slice(1).cast(pl.Int32))
    )
    if SMOKE and not full:
        return kept.head(2)
    if kept.height != 20 or not set(kept["substation_type"].to_list()) <= SUBSTATION_TYPES:
        msg = "expected the 20 Primary, BSP, and GSP series"
        raise ValueError(msg)
    return kept


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


def grid_index(*, times: pl.Series) -> np.ndarray:
    """Return the grid index of each UTC timestamp in a Polars series."""
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def load_power(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each series' observed power over effective capacity on the grid, NaN where missing.

    Args:
        series: `series_table()`.

    Returns:
        Each site label's float32 array on the grid.
    """
    grid = pl.DataFrame({"valid_time": grid_times()})
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            observed.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", pl.col("power") / row["effective_capacity_mw"]
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        out[row["site"]] = joined["power"].to_numpy().astype(np.float32)
    return out


def read_era5_pred(*, seed: int, sites: list[str]) -> dict[str, np.ndarray]:
    """Read experiment 00005's stage 1a predictions at ERA5 weather, each series on the grid.

    Args:
        seed: The fitting seed.
        sites: The site labels to return.

    Returns:
        Each requested site label's array on the grid.
    """
    path = E00005_WORK_DIR / "era5_pred" / f"{CONFIG}_seed{seed}.npy"
    stored = dict(zip(series_table(full=True)["site"], np.load(path), strict=True))
    return {site: stored[site] for site in sites}


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid."""
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def day_summaries(*, power: np.ndarray) -> np.ndarray:
    """Return the minimum, mean, and maximum of the 48 half-hours ending at each grid index.

    Missing half-hours are ignored, and a window with fewer than `MIN_OBSERVED` observed half-hours
    gives NaN. Windows that start
    before the grid are padded with missing half-hours.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.

    Returns:
        A float32 array of shape (grid size, 3), the columns being minimum, mean, and maximum.
    """
    padded = np.concatenate([np.full(STEPS_PER_DAY - 1, np.nan, dtype=np.float32), power])
    windows = sliding_window_view(padded, STEPS_PER_DAY)
    seen = np.isfinite(windows)
    count = seen.sum(axis=1)
    total = np.where(seen, windows, 0.0).sum(axis=1, dtype=np.float64)
    low = np.where(seen, windows, np.inf).min(axis=1)
    high = np.where(seen, windows, -np.inf).max(axis=1)
    empty = count < MIN_OBSERVED
    return np.stack(
        [
            np.where(empty, np.nan, low),
            np.where(empty, np.nan, total / np.maximum(count, 1)),
            np.where(empty, np.nan, high),
        ],
        axis=1,
    ).astype(np.float32)


def recent_inputs(
    *, power: np.ndarray, residual: np.ndarray, target: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Return every recent-power input for each case, using only observations before the origin.

    The first 13 columns are experiment 00005's recent-origin inputs, computed as it computed them.
    The last 6 are the summaries. The newest observation a forecast initialised at grid index
    `origin` may use is at `origin - 1`. A power lag or lag anomaly is null where the lag is no
    longer than the lead, which is `target - origin`.

    The last-24-hours summary covers the 48 half-hours ending at `origin - 1`. The same-weekday
    summary covers the 48 half-hours of the latest UTC calendar day on the target's weekday that
    ends at or before `origin - 1`, which is at least one week before the target's day.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        residual: Observed power minus stage 1a at the observed ERA5 weather, on the grid.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, len(RECENT_COLUMNS)), columns in `RECENT_COLUMNS` order.

    Raises:
        ValueError: If a case's valid time is not after its origin.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    last = origin - 1
    gap = target - last
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - STEPS_PER_WEEK * np.ceil(gap / STEPS_PER_WEEK).astype(int)
    lead = target - origin
    columns = [
        take(values=residual, index=last),
        take(values=residual, index=last - 2),
        take(values=residual, index=last - 6),
        take(values=residual, index=day),
        take(values=residual, index=week),
    ]
    usable = [lag > lead for lag in LAG_STEPS]
    columns += [
        np.where(ok, take(values=power, index=target - lag), np.nan)
        for lag, ok in zip(LAG_STEPS, usable, strict=True)
    ]
    columns += [
        np.where(ok, take(values=residual, index=target - lag), np.nan)
        for lag, ok in zip(LAG_STEPS, usable, strict=True)
    ]
    summaries = day_summaries(power=power)
    # The grid starts at 00:00 UTC, so a multiple of 48 is a UTC midnight.
    day_start = (target // STEPS_PER_DAY) * STEPS_PER_DAY
    weeks_back = np.maximum(1, np.ceil((day_start + STEPS_PER_DAY - 1 - last) / STEPS_PER_WEEK))
    weekday_end = day_start - STEPS_PER_WEEK * weeks_back.astype(int) + STEPS_PER_DAY - 1
    if (weekday_end > last).any():
        msg = "the same-weekday window must end before the origin"
        raise ValueError(msg)
    for end in (last, weekday_end):
        block = np.full((end.size, 3), np.nan, dtype=np.float32)
        inside = (end >= 0) & (end < power.size)
        block[inside] = summaries[end[inside]]
        columns += [block[:, i] for i in range(3)]
    return np.stack(columns, axis=1).astype(np.float32)


def pinball_quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)
