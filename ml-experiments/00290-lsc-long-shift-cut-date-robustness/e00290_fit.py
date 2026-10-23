"""Repeat experiment 274's 30-day, 25% transfer at many fit-cut dates and cut offsets.

For each (substation, end date) job, a transfer of 25% of the substation's own power is added for
the 30 days before the end date. The 24-month GAM_ONLY is fitted on the clean and on the injected
series at cuts from the end date to 72 hours after it, and the bias (injected minus clean forecast,
in points of effective capacity) is read over fixed windows after the last cut.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        uv run --no-sync python e00290_fit.py --workers 3
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_ens_weather import replace_site
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
    naive_utc,
)
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00290-lsc-long-shift-cut-date-robustness"
ROWS_DIR: Final[Path] = EXPERIMENT_DIR / "rows"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SITES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D5", "D10", "D11", "D12", "D19", "D21", "D22")
SEASON_YEARS: Final[range] = range(2019, 2027)
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as in the GAM-K settings of experiments 252 and 274."""
WINDOW: Final[TrainingWindow] = TrainingWindow(months=24)


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after (or before) `moment`'s month."""
    index = moment.year * 12 + moment.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def target_rows(*, cache: FeatureCache, site: str, cut: datetime, stop: datetime) -> np.ndarray:
    """Return the scored grid rows from `cut` to `stop`: a reading and finite model inputs."""
    features = cache.sites[site]
    first, last = naive_utc(moment=cut), naive_utc(moment=stop)
    keep = (
        (cache.end_times > first)
        & (cache.end_times <= last)
        & features.present
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    return np.flatnonzero(keep)


def forecast(*, cache: FeatureCache, site: str, cut: datetime, index: np.ndarray) -> np.ndarray:
    """Fit GAM_ONLY with the cut's rows (24-month window) and predict the rows `index`."""
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[GamSpec(ridge=RIDGE, drift="none", capacity="fitted", final_knot="free")],
        cut=cut,
        window=WINDOW,
        season_years=SEASON_YEARS,
    )
    return predict_gam(cache=cache, fit=fit, index=index)


END_DATES: Final[tuple[datetime, ...]] = tuple(
    datetime(y, m, d, tzinfo=UTC)
    for y, m, d in (
        (2023, 1, 15),
        (2023, 3, 1),
        (2023, 4, 20),
        (2023, 6, 10),
        (2023, 7, 31),
        (2023, 9, 15),
        (2023, 10, 31),
        (2023, 12, 20),
        (2024, 2, 10),
        (2024, 4, 5),
        (2024, 6, 25),
        (2024, 8, 30),
    )
)
CUT_OFFSET_HOURS: Final[tuple[int, ...]] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 48, 72)
SIZE: Final[float] = 0.25
DURATION: Final[timedelta] = timedelta(days=30)
# Fixed target windows, in days after the end date, all later than the last cut (72 hours).
WINDOWS_DAYS: Final[dict[str, tuple[float, float]]] = {
    "days_3_to_10": (3.0, 10.0),
    "days_10_to_17": (10.0, 17.0),
    "days_3_to_33": (3.0, 33.0),
}
AFTER_48H: Final[tuple[float, float]] = (0.0, 2.0)


def run_job(*, cache: FeatureCache, site: str, end: datetime) -> pl.DataFrame:
    """Return one row per (cut offset, target window) for one substation and end date."""
    features = cache.sites[site]
    onset = end - DURATION
    stamps = cache.end_times
    active = (stamps > naive_utc(moment=onset)) & (stamps <= naive_utc(moment=end))
    delta = np.where(active, SIZE * features.y, 0.0)
    injected = replace_site(cache=cache, site=site, y=features.y + delta)
    mean_transfer = float(np.nanmean(delta[active & np.isfinite(delta)]))
    records: list[dict] = []
    for hours in CUT_OFFSET_HOURS:
        cut = end + timedelta(hours=hours)
        targets = dict(WINDOWS_DAYS)
        if hours == 0:
            targets["after_48h"] = AFTER_48H
        cut_forecasts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (first_day, last_day) in targets.items():
            index = target_rows(
                cache=cache,
                site=site,
                cut=max(cut, end + timedelta(days=first_day)),
                stop=end + timedelta(days=last_day),
            )
            if len(index) == 0:
                continue
            clean = forecast(cache=cache, site=site, cut=cut, index=index)
            moved = forecast(cache=injected, site=site, cut=cut, index=index)
            cut_forecasts[name] = (clean, moved)
            truth = features.y[index]
            records.append(
                {
                    "site": site,
                    "end": end,
                    "cut_offset_hours": hours,
                    "window": name,
                    "n_rows": len(index),
                    "mean_bias": float(np.mean(moved - clean)),
                    "mean_transfer": mean_transfer,
                    "mean_error_clean_vs_nra": float(np.mean(clean - truth)),
                    "mean_error_injected_vs_nra": float(np.mean(moved - truth)),
                    "n_train_rows": int(
                        (
                            (stamps <= naive_utc(moment=cut))
                            & (stamps > naive_utc(moment=add_months(moment=cut, months=-24)))
                            & features.present
                            & np.isfinite(features.y)
                        ).sum()
                    ),
                }
            )
    return pl.DataFrame(records)


_CACHE: FeatureCache | None = None


def _job(job: tuple[str, datetime]) -> str:
    torch.set_num_threads(1)
    site, end = job
    target = ROWS_DIR / f"{site}_{end:%Y%m%d}.parquet"
    if target.exists():
        return f"{target.name}: already done"
    assert _CACHE is not None
    rows = run_job(cache=_CACHE, site=site, end=end)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    rows.write_parquet(temporary)
    temporary.rename(target)
    return f"{target.name}: {rows.height} rows"


def main() -> None:
    """Run every (substation, end date) job in parallel; finished jobs are skipped on rerun."""
    global _CACHE  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    jobs = [(site, end) for site in SITES for end in END_DATES]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
