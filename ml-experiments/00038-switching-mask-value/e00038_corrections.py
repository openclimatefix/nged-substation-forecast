"""The stack's error correction with some history half-hours masked out.

The rule is experiment 00019's: the correction at an origin is the mean of `y - f` over the window
that ends `newest` half-hours of data age before the origin, with at least 3 days present. Masked
half-hours are left out of the mean. With under 3 days left the last correction that had 3 days is
carried forward (zero before the first).
"""

from collections.abc import Callable
from typing import Final

import numpy as np
from e00038_intervals import BLOCK, Interval, known_at

MIN_COUNT: Final[int] = 3 * BLOCK
NEWEST_AGE: Final[int] = 8
"""The newest usable observation is 8 half-hours (4 hours) before the origin, in 00019's rule."""
MaskFor = Callable[[int, int], np.ndarray]
"""`mask(newest, lo)` returns a bool array over stamps `lo..newest` that is True where excluded."""


def correction_series(
    *,
    residual: np.ndarray,
    newest: np.ndarray,
    window: int,
    masked: MaskFor | None = None,
    open_start: Callable[[int], int | None] | None = None,
) -> np.ndarray:
    """Return the correction at each origin, whose newest observation stamp is in `newest`.

    Args:
        residual: `y - f` per stamp, NaN where missing.
        newest: Stamp of the newest usable observation at each origin.
        window: Window length in half-hours.
        masked: Optional mask provider; None gives the unmasked rule.
        open_start: Optional `start(newest)` giving the onset of an interval still open at
            `newest` (or None). While one is open, the correction becomes the masked correction
            plus the mean residual over the last 24 hours of the open interval minus the masked
            correction, i.e. it follows the event's current residual level.

    Returns:
        One correction per origin, in the units of `residual`.
    """
    out = np.zeros(len(newest))
    last = 0.0
    for i, n_stamp in enumerate(newest):
        n = int(n_stamp)
        lo = max(n - window + 1, 0)
        values = residual[lo : n + 1]
        keep = np.isfinite(values)
        if masked is not None:
            keep &= ~masked(n, lo)
        if keep.sum() >= MIN_COUNT:
            last = float(values[keep].mean())
        out[i] = last
        if open_start is not None and (start := open_start(n)) is not None:
            recent = residual[max(start, n - BLOCK + 1) : n + 1]
            if np.isfinite(recent).any():
                out[i] = float(np.nanmean(recent))
    return out


def static_mask(*, mask: np.ndarray) -> MaskFor:
    """Return a provider that reads a fixed bool array (the oracle and random arms)."""
    return lambda n, lo: mask[lo : n + 1]


def detected_mask(*, intervals: list[Interval], series: int, window: int) -> MaskFor:
    """Return a provider of the intervals on `series` that an origin with `newest = n` can know."""
    own = [iv for iv in intervals if iv.series == series]

    def provider(n: int, lo: int) -> np.ndarray:
        out = np.zeros(n - lo + 1, dtype=bool)
        for _, start, stop in known_at(intervals=own, newest=n, window=window):
            out[max(start, lo) - lo : max(stop, lo) - lo] = True
        return out

    return provider


def open_interval_start(
    *, intervals: list[Interval], series: int, window: int
) -> Callable[[int], int | None]:
    """Return `start(n)`: the latest onset of an interval on `series` still open at stamp `n`."""
    own = [iv for iv in intervals if iv.series == series]

    def start(n: int) -> int | None:
        open_onsets = [iv.onset for iv in own if iv.onset + window + 2 <= n < iv.known_from]
        return max(open_onsets) if open_onsets else None

    return start
