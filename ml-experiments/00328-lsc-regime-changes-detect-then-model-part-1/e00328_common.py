# ruff: noqa: E501
"""Shared constants and helpers of experiment 328 (detect permanent regime changes, then model them).

Every earlier experiment's files are read and never written. This experiment writes only under its
own data folder.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00295_common as c295  # ty: ignore[unresolved-import]
import numpy as np
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import TrainingWindow
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 328
SLUG: Final[str] = "lsc-regime-changes-detect-then-model-part-1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
D8_BREAK: Final[np.datetime64] = np.datetime64("2023-08-15", "D")
FIRST_CUT: Final[datetime] = datetime(2020, 9, 1, tzinfo=UTC)
LAST_CUT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
WINDOW_MONTHS: Final[int] = 24
CUSUM_H_GRID: Final[tuple[float, ...]] = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
CUSUM_K: Final[float] = 0.5
CUSUM_TRAILING_DAYS: Final[int] = 180
CUSUM_MIN_DAYS: Final[int] = 90
REFRACTORY_DAYS: Final[int] = 120
HISTOGRAM_BETA_GRID: Final[tuple[float, ...]] = (32.0, 64.0, 128.0, 256.0, 512.0)
HISTOGRAM_MIN_WEEKS: Final[int] = 4
HISTOGRAM_MATCH_DAYS: Final[int] = 21
DETECT_WITHIN_DAYS: Final[int] = 60
STEP_SIZES: Final[tuple[float, ...]] = (0.10, 0.25)
RAMP_DAYS: Final[int] = 30
SPAWN_WORKERS: Final[int] = 6


def month_start(*, year: int, month: int) -> datetime:
    """Return the first instant of a month in UTC."""
    return datetime(year, month, 1, tzinfo=UTC)


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after the first-of-month `moment`."""
    index = moment.year * 12 + moment.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def monthly_cuts() -> list[datetime]:
    """Return every first-of-month forecast origin from `FIRST_CUT` to `LAST_CUT`."""
    cuts, cut = [], FIRST_CUT
    while cut <= LAST_CUT:
        cuts.append(cut)
        cut = add_months(moment=cut, months=1)
    return cuts


def window() -> TrainingWindow:
    """Return the rolling training window of the rolling lagless GAM."""
    return TrainingWindow(months=WINDOW_MONTHS)


def out_of_sample_residual(
    *, cache: FeatureCache, site: str, cut: datetime, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the lagless GAM on the 24 months before `cut`, return the month after `cut`.

    Args:
        cache: The feature cache.
        site: The Primary's label.
        cut: The first of the month, the forecast origin.
        rows: The grid rows with a reading and finite model inputs.

    Returns:
        The row indices of the month after `cut`, the readings, and the residual (reading minus
        forecast) in points of effective capacity. Empty arrays when the month has no rows or the
        Primary has no training rows.
    """
    stamps = cache.end_times
    start, stop = naive_utc(moment=cut), naive_utc(moment=add_months(moment=cut, months=1))
    target = rows[(stamps[rows] > start) & (stamps[rows] <= stop)]
    empty = np.array([], dtype=float)
    if len(target) == 0:
        return target, empty, empty
    try:
        pred = c295.fit_and_predict(cache=cache, site=site, cut=cut, window=window(), index=target)
    except ValueError:  # no training rows before this cut
        return target[:0], empty, empty
    y = cache.sites[site].y[target]
    return target, y, y - pred


def day_of(*, end_times: np.ndarray) -> np.ndarray:
    """Return the UTC date of each half-hour, counting the half-hour by its start."""
    return (end_times - np.timedelta64(1, "s")).astype("datetime64[D]")


def n_workers() -> int:
    """Return the worker count, capped at the slot's 6 cores."""
    return min(SPAWN_WORKERS, os.cpu_count() or 1)
