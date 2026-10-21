r"""Build the three residual panels of experiment 130 from monthly fits of the convex demand model.

For each of the 16 Primaries the convex model without drift (`gamk`'s base, fitted PV capacity) is
fitted on a 24-month window at 00:00 UTC on the 1st and on the 16th of each month, and each fit
forecasts the half-hours up to the next cut of the same series (past weather, so these are lab
residuals). The fits see uninjected power only. Three panels follow, each with the layout of
experiment 82's panel (`expected` and `residual` in MW, per series, on the half-hourly grid):

- `frozen`: observed power minus the convex forecast (arm B and arm D).
- `innovation`: the one-step innovations of the Kalman filter on the frozen residual (arm C).
- Each of the two, for the model refitted on the 1st and for the model refitted on the 16th.

Stages, in order (each writes files to the private folder):

- `fit`: the 1st-of-month and 16th-of-month fits, one pickle per series and offset.
- `hyper`: the filter's four hyperparameters per series, fitted on odd-month frozen residuals.
- `panel`: the frozen and innovation panels for both offsets.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00130-*/e00130_gam_panel.py fit hyper panel
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from math import log, pi
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
from e00130_common import GRID_LENGTH, GRID_START, SERIES_IDS, TUNING_MONTHS, WORK_DIR, stamp_months
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    MIN_PRIOR_SPREAD,
    VARIANCE_FLOOR,
    FilterHyperparameters,
    fit_site_hyperparameters,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import GamFit, GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PART_4_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
"""Experiment 106's folder: the shared feature cache and the tuned penalties and filter prior."""
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent
    / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
GAM_DIR: Final[Path] = WORK_DIR / "gam"
RIDGE: Final[float] = 0.1
WINDOW_MONTHS: Final[int] = 24
SEASON_YEARS: Final[range] = range(2019, 2027)
MIN_TRAINING_DAYS: Final[int] = 90
FIRST_CUT_YEAR_MONTH: Final[tuple[int, int]] = (2021, 12)
LAST_CUT: Final[datetime] = datetime(2026, 6, 16, tzinfo=UTC)
OFFSETS: Final[tuple[int, ...]] = (1, 16)
"""Day of the month on which each of the two models refits."""
N_WORKERS: Final[int] = 8
BAND_COUNT: Final[int] = 4


def site_of_series() -> dict[int, tuple[str, float]]:
    """Map each detector series id to its Primary's label and effective capacity in MW."""
    table = pl.read_parquet(SERIES_PATH)
    out = {
        int(r["time_series_id"]): (r["site"], float(r["effective_capacity_mw"]))
        for r in table.iter_rows(named=True)
    }
    missing = [s for s in SERIES_IDS if s not in out]
    if missing:
        msg = f"series without a label: {missing}"
        raise ValueError(msg)
    return {s: out[s] for s in SERIES_IDS}


def load_cache() -> FeatureCache:
    """Return experiment 106's shared feature cache."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=PART_4_DIR / "cache")


def cuts_for(*, offset_day: int) -> list[datetime]:
    """The cuts at 00:00 UTC on `offset_day` of each month from 2021-12 to 2026-06."""
    out = []
    year, month = FIRST_CUT_YEAR_MONTH
    while (cut := datetime(year, month, offset_day, tzinfo=UTC)) <= LAST_CUT:
        out.append(cut)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def base_spec() -> GamSpec:
    """The convex model without drift, with the fitted PV capacity and experiment 106's penalties."""
    tuned = json.loads((PART_4_DIR / "work" / "tuned.json").read_text())
    return GamSpec(
        ridge=RIDGE,
        drift="none",
        step_penalty=tuned["step_penalty"],
        capacity="fitted",
        final_knot="free",
    )


def fit_path(*, series: int, offset_day: int) -> Path:
    """The file that holds one series' fits for one refit day."""
    return GAM_DIR / f"fits_{series}_{offset_day}.pkl"


def fit_job(job: tuple[int, int]) -> str:
    """Fit every cut of one series and refit day, forecast to the next cut, and store both."""
    series, offset_day = job
    torch.set_num_threads(1)
    site, _ = site_of_series()[series]
    cache = load_cache()
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    needed = MIN_TRAINING_DAYS * 48
    cuts = cuts_for(offset_day=offset_day)
    naive = np.array([np.datetime64(c.replace(tzinfo=None), "us") for c in cuts])
    spec = base_spec()
    forecast = np.full(GRID_LENGTH, np.nan)
    fitted, lost = 0, 0
    for k, cut in enumerate(cuts):
        end = cache.end_times
        if int((usable & (end <= naive[k])).sum()) < needed:
            continue
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=TrainingWindow(months=WINDOW_MONTHS),
            season_years=SEASON_YEARS,
        )
        stop = naive[k + 1] if k + 1 < len(cuts) else np.datetime64("2026-07-01T00:00", "us")
        rows = np.flatnonzero((end > naive[k]) & (end <= stop) & np.isfinite(features.y))
        if not len(rows):
            continue
        fitted += 1
        try:
            points = predict_gam(cache=cache, fit=fit, index=rows)
        except ValueError:
            points = np.full(len(rows), np.nan)
            lost += len(rows)
        stamps = ((end[rows] - np.datetime64("2022-01-01T00:00", "us")) // np.timedelta64(30, "m"))
        keep = (stamps >= 0) & (stamps < GRID_LENGTH)
        forecast[stamps[keep].astype(int)] = points[keep] * _capacity(series=series) / 100.0
    GAM_DIR.mkdir(parents=True, exist_ok=True)
    np.save(GAM_DIR / f"forecast_{series}_{offset_day}.npy", forecast)
    return f"series {series} ({site}) day {offset_day}: {fitted} fits, {lost} rows without inputs"


def _capacity(*, series: int) -> float:
    return site_of_series()[series][1]


def stage_fit() -> None:
    """Fit all series and both refit days in parallel."""
    jobs = [(s, d) for s in SERIES_IDS for d in OFFSETS]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for line in pool.map(fit_job, jobs):
            print(line, flush=True)


def observed(*, series: int) -> np.ndarray:
    """Observed power in MW on the grid, from the feature cache (equal to experiment 82's panel)."""
    site, capacity = site_of_series()[series]
    cache = load_cache()
    y = cache.sites[site].y
    stamps = (
        (cache.end_times - np.datetime64("2022-01-01T00:00", "us")) // np.timedelta64(30, "m")
    ).astype(int)
    keep = (stamps >= 0) & (stamps < GRID_LENGTH) & np.isfinite(y)
    out = np.full(GRID_LENGTH, np.nan)
    out[stamps[keep]] = y[keep] * capacity / 100.0
    return out


def frozen_panel(*, offset_day: int) -> dict[str, dict[int, np.ndarray]]:
    """`expected`, `residual`, and `power` in MW per series for the model refitted on `offset_day`."""
    out: dict[str, dict[int, np.ndarray]] = {"power": {}, "expected": {}, "residual": {}}
    for series in SERIES_IDS:
        expected = np.load(GAM_DIR / f"forecast_{series}_{offset_day}.npy")
        power = observed(series=series)
        out["power"][series] = power
        out["expected"][series] = expected
        out["residual"][series] = power - expected
    return out


def hour_band(*, n: int = GRID_LENGTH) -> np.ndarray:
    """The hour band of each grid stamp, as in experiment 111's calendar (0 to 3)."""
    cache = load_cache()
    band = cache.calendar.band
    stamps = (
        (cache.end_times - np.datetime64("2022-01-01T00:00", "us")) // np.timedelta64(30, "m")
    ).astype(int)
    out = np.full(n, -1, dtype=int)
    keep = (stamps >= 0) & (stamps < n)
    out[stamps[keep]] = band[keep]
    return out


def innovations(
    *, residual: np.ndarray, band: np.ndarray, hyper: FilterHyperparameters
) -> np.ndarray:
    """One-step innovations of the two-state filter (level plus transient), per hour band.

    The filter is `studies.dp_assimilation`'s, run on one band's readings at a time; the
    innovation is the reading minus the state predicted from the earlier readings, so it never
    contains the reading's own update. The innovation is NaN where there is no reading.
    """
    out = np.full(len(residual), np.nan)
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    for b in range(BAND_COUNT):
        positions = np.flatnonzero((band == b) & np.isfinite(residual))
        level, anomaly = 0.0, 0.0
        p_ll, p_lz, p_zz = INITIAL_LEVEL_VARIANCE, 0.0, stationary
        previous = int(positions[0]) if len(positions) else 0
        for position in positions:
            gap = int(position) - previous
            previous = int(position)
            if gap:
                decay = phi**gap
                anomaly *= decay
                p_ll += q_step * gap
                p_lz *= decay
                p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
            variance = p_ll + 2.0 * p_lz + p_zz + noise
            innovation = residual[position] - level - anomaly
            out[position] = innovation
            gain_l = (p_ll + p_lz) / variance
            gain_z = (p_lz + p_zz) / variance
            level += gain_l * innovation
            anomaly += gain_z * innovation
            p_ll, p_lz, p_zz = (
                max(p_ll - gain_l * gain_l * variance, VARIANCE_FLOOR),
                p_lz - gain_l * gain_z * variance,
                max(p_zz - gain_z * gain_z * variance, VARIANCE_FLOOR),
            )
    return out


def stage_hyper() -> None:
    """Fit each series' filter on its odd-month frozen residual of the 1st-of-month model."""
    tuned = json.loads((PART_4_DIR / "work" / "tuned.json").read_text())["filter_hyper"]["a0"]
    stacked = np.array(list(tuned.values()))
    prior_mean = stacked.mean(axis=0)
    prior_spread = np.maximum(stacked.std(axis=0, ddof=1), MIN_PRIOR_SPREAD)
    band = hour_band()
    odd = np.isin(stamp_months(), list(TUNING_MONTHS))
    panel = frozen_panel(offset_day=1)
    out = {}
    for series, (site, capacity) in site_of_series().items():
        if not np.isfinite(panel["residual"][series]).any():
            print(series, site, "no fits; skipped", flush=True)
            continue
        points = np.where(odd, panel["residual"][series] / capacity * 100.0, np.nan)
        result = fit_site_hyperparameters(
            residual=points,
            band=band,
            prior_mean=prior_mean,
            prior_spread=prior_spread,
            start=np.array(tuned[site]) if site in tuned else prior_mean,
        )
        out[str(series)] = result.as_array().tolist()
        print(series, site, np.round(result.as_array(), 3), flush=True)
    (GAM_DIR / "hyper.json").write_text(json.dumps(out))


def load_hyper() -> dict[int, FilterHyperparameters]:
    """The stored filter hyperparameters per series."""
    raw = json.loads((GAM_DIR / "hyper.json").read_text())
    return {int(s): FilterHyperparameters.from_array(values=np.array(v)) for s, v in raw.items()}


def stage_panel() -> None:
    """Write the frozen and innovation residual panels for both refit days."""
    band = hour_band()
    hyper = load_hyper()
    caps = {s: c for s, (_, c) in site_of_series().items()}
    for offset_day in OFFSETS:
        panel = frozen_panel(offset_day=offset_day)
        innov = {
            s: innovations(
                residual=panel["residual"][s] / caps[s] * 100.0, band=band, hyper=hyper[s]
            )
            * caps[s]
            / 100.0
            for s in SERIES_IDS
            if s in hyper
        }
        with (GAM_DIR / f"panel_{offset_day}.pkl").open("wb") as handle:
            pickle.dump({"frozen": panel, "innovation": innov}, handle)
        print(f"panel day {offset_day} written", flush=True)


def main() -> None:
    """Run the requested stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="+", choices=["fit", "hyper", "panel"])
    args = parser.parse_args()
    for stage in args.stages:
        {"fit": stage_fit, "hyper": stage_hyper, "panel": stage_panel}[stage]()


if __name__ == "__main__":
    main()
