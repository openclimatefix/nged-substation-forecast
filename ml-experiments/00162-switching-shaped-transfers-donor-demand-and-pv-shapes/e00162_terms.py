"""Split experiment 130's convex-model forecast into a solar term and a demand term, per series.

For each of the 16 Primaries the convex model without drift (experiment 130's `base_spec`, refitted
here on the 1st of every month with a 24-month window) forecasts the half-hours up to the next cut.
The forecast is a sum of columns of a linear model. The capacity columns carry the fitted solar
output (negative, because net power is demand minus solar); all other columns are the demand terms.
Both sums are stored in MW on the half-hourly grid, with their total, so the total can be checked
against experiment 130's stored forecast.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python e00162_terms.py [--series 8 --limit 3]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch
from e00162_common import GRID_LENGTH, SERIES_IDS, WORK_DIR
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

PART_4_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
)
SERIES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "studies/per_study/stage1_pilot/demand/inputs/series.parquet"
)
TERMS_DIR: Final[Path] = WORK_DIR / "terms"
GAM_130_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual" / "gam"
)
RIDGE: Final[float] = 0.1
WINDOW_MONTHS: Final[int] = 24
SEASON_YEARS: Final[range] = range(2019, 2027)
MIN_TRAINING_DAYS: Final[int] = 90
FIRST_CUT: Final[tuple[int, int]] = (2021, 12)
LAST_CUT: Final[datetime] = datetime(2026, 6, 16, tzinfo=UTC)
N_WORKERS: Final[int] = 4
GRID_ORIGIN: Final[np.datetime64] = np.datetime64("2022-01-01T00:00", "us")


def site_of_series() -> dict[int, tuple[str, float]]:
    """Map each series id to its Primary's label and effective capacity in MW."""
    table = pl.read_parquet(SERIES_PATH)
    out = {
        int(r["time_series_id"]): (r["site"], float(r["effective_capacity_mw"]))
        for r in table.iter_rows(named=True)
    }
    return {s: out[s] for s in SERIES_IDS}


def load_cache() -> FeatureCache:
    """Return experiment 106's shared feature cache."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=PART_4_DIR / "cache")


def cuts() -> list[datetime]:
    """The cuts at 00:00 UTC on the 1st of each month from 2021-12 to 2026-06."""
    out = []
    year, month = FIRST_CUT
    while (cut := datetime(year, month, 1, tzinfo=UTC)) <= LAST_CUT:
        out.append(cut)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def base_spec() -> GamSpec:
    """The convex model without drift, fitted PV capacity, and experiment 106's penalties."""
    tuned = json.loads((PART_4_DIR / "work" / "tuned.json").read_text())
    return GamSpec(
        ridge=RIDGE,
        drift="none",
        step_penalty=tuned["step_penalty"],
        capacity="fitted",
        final_knot="free",
    )


def series_job(series: int) -> str:
    """Fit every cut of one series and store its solar, demand, and total terms in MW."""
    path = TERMS_DIR / f"terms_{series}.npz"
    if path.exists():
        return f"series {series}: exists"
    torch.set_num_threads(1)
    site, capacity_mw = site_of_series()[series]
    cache = load_cache()
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    end = cache.end_times
    all_cuts = cuts()
    naive = np.array([np.datetime64(c.replace(tzinfo=None), "us") for c in all_cuts])
    spec = base_spec()
    solar = np.full(GRID_LENGTH, np.nan)
    demand = np.full(GRID_LENGTH, np.nan)
    total = np.full(GRID_LENGTH, np.nan)
    fitted = 0
    for k, cut in enumerate(all_cuts):
        if int((usable & (end <= naive[k])).sum()) < MIN_TRAINING_DAYS * 48:
            continue
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=TrainingWindow(months=WINDOW_MONTHS),
            season_years=SEASON_YEARS,
        )
        stop = naive[k + 1] if k + 1 < len(all_cuts) else np.datetime64("2026-07-01T00:00", "us")
        rows = np.flatnonzero((end > naive[k]) & (end <= stop) & np.isfinite(features.y))
        if not len(rows):
            continue
        fitted += 1
        capacity_columns = np.zeros(fit.layout.n_columns, dtype=bool)
        capacity_columns[fit.layout.slices["capacity"]] = True
        try:
            parts = {
                "solar": predict_gam(cache=cache, fit=fit, index=rows, columns=capacity_columns),
                "demand": predict_gam(cache=cache, fit=fit, index=rows, columns=~capacity_columns),
                "total": predict_gam(cache=cache, fit=fit, index=rows),
            }
        except ValueError:
            continue
        stamps = ((end[rows] - GRID_ORIGIN) // np.timedelta64(30, "m")).astype(int)
        keep = (stamps >= 0) & (stamps < GRID_LENGTH)
        scale = capacity_mw / 100.0
        solar[stamps[keep]] = parts["solar"][keep] * scale
        demand[stamps[keep]] = parts["demand"][keep] * scale
        total[stamps[keep]] = parts["total"][keep] * scale
    TERMS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, solar=solar, demand=demand, total=total)
    tmp.rename(path)
    return f"series {series} ({site}): {fitted} fits"


def main() -> None:
    """Fit all series in a pool of workers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", type=int, default=0)
    args = parser.parse_args()
    jobs = [args.series] if args.series else list(SERIES_IDS)
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(jobs))) as pool:
        for line in pool.map(series_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
