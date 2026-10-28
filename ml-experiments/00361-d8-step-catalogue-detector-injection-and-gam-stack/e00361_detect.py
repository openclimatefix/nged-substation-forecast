# ruff: noqa: E501
"""A plain as-of step detector for half-hourly power, and the cache loader it runs on.

The baseline of a half-hour is the median of the same half-hour of the day over the 2 to 8 days
before and after it (only the days that exist before a cut, when a cut is given). The residual is
the power minus the baseline. A negative segment is a run in which the 6-hour centred rolling
median of the residual stays at or below minus the threshold, with short gaps merged. A second pass
uses experiment 159's rule for multi-day drops (24-hour median, gaps under 12 hours merged, at
least 2 days). The last half-window before a cut is never flagged, because a centred window there
would be half empty.
"""

import os

os.environ.setdefault("E00252_SOURCE", "mva")
os.environ.setdefault("E00252_ARM", "a1")

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

HALF_HOURS_PER_DAY: Final[int] = 48
THRESHOLD_MW: Final[float] = 1.5
"""Experiment 159's depth threshold."""
SMOOTH_WINDOW: Final[int] = 12
"""The residual is smoothed with a centred rolling median of 6 hours."""
MERGE_GAP: Final[int] = 12
"""Runs separated by under 6 hours are merged."""
MIN_STEP: Final[int] = 12
"""A step lasts at least 6 hours."""
MIN_DROP: Final[int] = 2 * HALF_HOURS_PER_DAY
"""A multi-day drop lasts at least 2 days."""
BASELINE_DAYS: Final[tuple[int, ...]] = (-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8)
MIN_BASELINE_VALUES: Final[int] = 6


@dataclass(frozen=True)
class Segments:
    """Detected segments, as half-hour index ranges `[start, end)` on the input array.

    Attributes:
        start: First half-hour index of each segment.
        end: One past the last index of each segment.
        level_mw: Median residual in the segment (negative for a fall in power).
    """

    start: np.ndarray
    end: np.ndarray
    level_mw: np.ndarray

    def __len__(self) -> int:
        """Return the number of segments."""
        return len(self.start)


def baseline(*, power_mw: np.ndarray) -> np.ndarray:
    """Return the median of the same half-hour on the 2 to 8 days before and after each index."""
    n = len(power_mw)
    stack = np.full((len(BASELINE_DAYS), n), np.nan)
    for row, day in enumerate(BASELINE_DAYS):
        shift = day * HALF_HOURS_PER_DAY
        if shift < 0:
            stack[row, -shift:] = power_mw[: n + shift]
        elif shift < n:
            stack[row, : n - shift] = power_mw[shift:]
    count = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        median = np.nanmedian(stack, axis=0)
    return np.where(count >= MIN_BASELINE_VALUES, median, np.nan)


def smoothed_residual(
    *, power_mw: np.ndarray, window: int = SMOOTH_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Return the residual and its centred rolling median over `window` half-hours."""
    residual = power_mw - baseline(power_mw=power_mw)
    series = pl.Series(np.where(np.isfinite(residual), residual, np.nan)).fill_nan(None)
    rolled = series.rolling_median(window_size=window, min_samples=window // 2, center=True)
    return residual, rolled.fill_null(np.nan).to_numpy()


def _runs(*, flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.concatenate([[False], flag, [False]]).astype(np.int8)
    change = np.diff(padded)
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


def detect(
    *,
    power_mw: np.ndarray,
    threshold_mw: float = THRESHOLD_MW,
    negative: bool = True,
    window: int = SMOOTH_WINDOW,
    merge_gap: int = MERGE_GAP,
    min_length: int = MIN_STEP,
) -> Segments:
    """Return the segments of at least 6 hours where power sits `threshold_mw` below (or above) normal.

    Args:
        power_mw: Half-hourly power in MW on a regular grid, NaN where missing. Pass only the
            rows before a fit cut to detect as of that cut.
        threshold_mw: The depth the smoothed residual must reach.
        negative: Detect falls in power when `True`, rises when `False`.
        window: Half-hours in the centred rolling median of the residual.
        merge_gap: Runs closer than this many half-hours are merged.
        min_length: The fewest half-hours a segment may last.

    Returns:
        The segments, in time order.
    """
    residual, smooth = smoothed_residual(power_mw=power_mw, window=window)
    sign = 1.0 if negative else -1.0
    with np.errstate(invalid="ignore"):
        flag = sign * smooth <= -threshold_mw
    start, end = _runs(flag=flag)
    keep_start, keep_end = [], []
    for s, e in zip(start, end, strict=True):
        if keep_end and s - keep_end[-1] < merge_gap:
            keep_end[-1] = e
        else:
            keep_start.append(s)
            keep_end.append(e)
    n = len(power_mw)
    segments = [(s, e) for s, e in zip(keep_start, keep_end, strict=True) if e - s >= min_length]
    segments = [(s, e) for s, e in segments if e <= n - window // 2]
    starts = np.array([s for s, _ in segments], dtype=np.int64)
    ends = np.array([e for _, e in segments], dtype=np.int64)
    levels = np.array([float(np.nanmedian(residual[s:e])) for s, e in segments])
    return Segments(start=starts, end=ends, level_mw=levels)


def detect_drops(*, power_mw: np.ndarray, threshold_mw: float = THRESHOLD_MW) -> Segments:
    """Return the multi-day drops: experiment 159's rule (24-hour median, gaps under 12 hours)."""
    return detect(
        power_mw=power_mw,
        threshold_mw=threshold_mw,
        window=2 * HALF_HOURS_PER_DAY,
        merge_gap=24,
        min_length=MIN_DROP,
    )


def load_power_mw(*, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the site's half-hourly end times and power in MW from experiment 252's cache."""
    import e00252_common as common  # ty: ignore[unresolved-import]

    cache = common.load_cache()
    features = cache.sites[site]
    capacity = float(features.table.effective_capacity_mw)
    return cache.end_times, features.y * capacity / 100.0
