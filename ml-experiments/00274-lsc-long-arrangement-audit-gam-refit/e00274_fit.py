"""Inject a load transfer into the training data, refit monthly, and record how the forecast moves.

For each (substation, onset date) job, every scenario (size, sign, duration) is fitted at every
monthly cut from the month before the onset to 12 months after the end of the transfer. Each fit
uses only rows before its cut and predicts the whole following month with the actual weather.
The bias is the difference between the forecast trained on the injected series and the forecast
trained on the clean series at the same cut; the clean series is the true normal running
arrangement (NRA).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        uv run --no-sync python e00274_fit.py --workers 3
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

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

EXPERIMENT_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00274-lsc-long-arrangement-audit-gam-refit"
ROWS_DIR: Final[Path] = EXPERIMENT_DIR / "rows"
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SITES: Final[tuple[str, ...]] = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D10",
    "D11",
    "D12",
    "D19",
    "D21",
    "D22",
)
ONSETS: Final[tuple[datetime, ...]] = (
    datetime(2022, 12, 1, tzinfo=UTC),
    datetime(2023, 6, 1, tzinfo=UTC),
    datetime(2023, 12, 1, tzinfo=UTC),
    datetime(2024, 6, 1, tzinfo=UTC),
)
SIZES: Final[tuple[float, ...]] = (0.10, 0.25, 0.40)
DURATION_DAYS: Final[tuple[int | None, ...]] = (7, 14, 30, 60, 90, None)
"""Durations in days; `None` is a permanent step that lasts to the end of the data."""
DONOR_SIZE: Final[float] = 0.25
"""The donor case (transfer subtracted) runs at this size only."""
CONTROL_DURATION_DAYS: Final[int] = 30
MONTHS_AFTER_END: Final[int] = 12
AFTER_WINDOW: Final[timedelta] = timedelta(hours=48)
LAST_CUT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
SEASON_YEARS: Final[range] = range(2019, 2027)
RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty, as in the GAM-K settings of experiment 252."""
WINDOWS: Final[dict[str, TrainingWindow]] = {
    "24m": TrainingWindow(months=24),
    "12m": TrainingWindow(months=12),
    "all": TrainingWindow(),
}
ShapeType = Literal["share", "fixed"]


@dataclass(frozen=True)
class Scenario:
    """One injected transfer.

    Attributes:
        size: The share of the receiver's own power that moves; 0 for the control.
        sign: +1 when the series receives the transfer, -1 when it donates.
        duration_days: The transfer's length in days, or `None` for a permanent step.
        shape: `share` moves `size` times the receiver's own power at each half-hour; `fixed`
            moves `size` times the receiver's mean power over the year before the onset at every
            half-hour (a block of customers whose profile is flat).
    """

    size: float
    sign: int
    duration_days: int | None
    shape: ShapeType = "share"


def scenarios() -> list[Scenario]:
    """Return every scenario of a job: the receiver grid, the donor case, and the zero control."""
    grid = [Scenario(size, 1, days) for size in SIZES for days in DURATION_DAYS]
    donors = [Scenario(DONOR_SIZE, -1, days) for days in DURATION_DAYS]
    fixed = [Scenario(DONOR_SIZE, 1, days, "fixed") for days in DURATION_DAYS]
    return [*grid, *donors, *fixed, Scenario(0.0, 1, CONTROL_DURATION_DAYS)]


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after (or before) the first-of-month `moment`."""
    index = moment.year * 12 + moment.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def end_of(*, onset: datetime, scenario: Scenario) -> datetime | None:
    """Return the first instant after the transfer, or `None` for a permanent step."""
    if scenario.duration_days is None:
        return None
    return onset + timedelta(days=scenario.duration_days)


def cuts_of(*, onset: datetime, end: datetime | None) -> list[datetime]:
    """Return the cuts: monthly from the month before the onset to 12 months after the end.

    A fit at the end of the transfer is added, so that the 48 hours after the end are forecast by
    a fit that trained on the whole transfer whatever the calendar alignment.
    """
    stop = (
        LAST_CUT if end is None else min(LAST_CUT, add_months(moment=end, months=MONTHS_AFTER_END))
    )
    cuts = []
    cut = add_months(moment=onset, months=-1)
    while cut <= stop:
        cuts.append(cut)
        cut = add_months(moment=cut, months=1)
    if end is not None and end <= LAST_CUT:
        cuts.append(end)
    return sorted(set(cuts))


def phases_of(*, times: np.ndarray, onset: datetime, end: datetime | None) -> np.ndarray:
    """Label each target row (by the UTC end of its half-hour) with its phase.

    Returns:
        Strings `BEFORE`, `DURING`, `AFTER_48H`, or `LATER`, one per row.
    """
    onset64, after = naive_utc(moment=onset), np.full(len(times), "DURING", dtype="<U9")
    after[times <= onset64] = "BEFORE"
    if end is not None:
        end64 = naive_utc(moment=end)
        window = np.timedelta64(int(AFTER_WINDOW.total_seconds()), "s")
        after[times > end64] = "LATER"
        after[(times > end64) & (times <= end64 + window)] = "AFTER_48H"
    return after


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


def forecast(
    *, cache: FeatureCache, site: str, cut: datetime, window: TrainingWindow, index: np.ndarray
) -> np.ndarray:
    """Fit GAM_ONLY with the cut's rows and predict the rows `index`."""
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[GamSpec(ridge=RIDGE, drift="none", capacity="fitted", final_knot="free")],
        cut=cut,
        window=window,
        season_years=SEASON_YEARS,
    )
    return predict_gam(cache=cache, fit=fit, index=index)


def run_job(*, cache: FeatureCache, site: str, onset: datetime) -> pl.DataFrame:
    """Run every scenario of one (substation, onset) job and return the per-phase bias rows."""
    features = cache.sites[site]
    clean_forecasts: dict[tuple[str, datetime, datetime], np.ndarray] = {}
    records: list[dict] = []
    for scenario in scenarios():
        end = end_of(onset=onset, scenario=scenario)
        delta = np.zeros_like(features.y)
        stamps = cache.end_times
        active = stamps > naive_utc(moment=onset)
        if end is not None:
            active = active & (stamps <= naive_utc(moment=end))
        if scenario.shape == "share":
            delta[active] = scenario.sign * scenario.size * features.y[active]
        else:
            before = (stamps <= naive_utc(moment=onset)) & (
                stamps > naive_utc(moment=add_months(moment=onset, months=-12))
            )
            level = float(np.nanmean(features.y[before]))
            delta[active] = scenario.sign * scenario.size * level
        injected = replace_site(cache=cache, site=site, y=features.y + delta)
        cuts = cuts_of(onset=onset, end=end)
        stops = [*cuts[1:], add_months(moment=cuts[-1], months=1)]
        for cut, stop in zip(cuts, stops, strict=True):
            index = target_rows(cache=cache, site=site, cut=cut, stop=stop)
            if len(index) == 0:
                continue
            phase = phases_of(times=cache.end_times[index], onset=onset, end=end)
            for name, window in WINDOWS.items():
                key = (name, cut, stop)
                if key not in clean_forecasts:
                    clean_forecasts[key] = forecast(
                        cache=cache, site=site, cut=cut, window=window, index=index
                    )
                clean = clean_forecasts[key]
                moved = forecast(cache=injected, site=site, cut=cut, window=window, index=index)
                truth = features.y[index]
                for label in np.unique(phase):
                    rows = phase == label
                    records.append(
                        {
                            "site": site,
                            "onset": onset,
                            "size": scenario.size,
                            "sign": scenario.sign,
                            "duration_days": -1
                            if scenario.duration_days is None
                            else scenario.duration_days,
                            "window": name,
                            "cut": cut,
                            "phase": str(label),
                            "n_rows": int(rows.sum()),
                            "mean_bias": float(np.mean(moved[rows] - clean[rows])) * scenario.sign,
                            "mean_transfer": float(np.mean(delta[index][rows])) * scenario.sign,
                            "mean_level": float(np.mean(truth[rows])),
                            "mae_clean_vs_nra": float(np.mean(np.abs(clean[rows] - truth[rows]))),
                            "mae_injected_vs_nra": float(
                                np.mean(np.abs(moved[rows] - truth[rows]))
                            ),
                            "mean_error_clean_vs_nra": float(np.mean(clean[rows] - truth[rows])),
                            "mean_error_injected_vs_nra": float(np.mean(moved[rows] - truth[rows])),
                            "shape": scenario.shape,
                            "mean_error_injected_vs_observed": float(
                                np.mean(moved[rows] - truth[rows] - delta[index][rows])
                            ),
                        }
                    )
    return pl.DataFrame(records)


_CACHE: FeatureCache | None = None


def _job(job: tuple[str, datetime]) -> str:
    torch.set_num_threads(1)
    site, onset = job
    target = ROWS_DIR / f"{site}_{onset:%Y%m}.parquet"
    if target.exists():
        return f"{target.name}: already done"
    assert _CACHE is not None
    rows = run_job(cache=_CACHE, site=site, onset=onset)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    rows.write_parquet(temporary)
    temporary.rename(target)
    return f"{target.name}: {rows.height} rows"


def main() -> None:
    """Run every (substation, onset) job in parallel; a finished job's file is skipped on rerun."""
    global _CACHE  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sites", nargs="+", default=list(SITES))
    args = parser.parse_args()
    ROWS_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE = load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)
    jobs = [(site, onset) for site in args.sites for onset in ONSETS]
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
