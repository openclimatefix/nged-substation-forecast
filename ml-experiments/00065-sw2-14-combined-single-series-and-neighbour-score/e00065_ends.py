"""The PE end rule of experiment 61: an interval ends at the strongest opposite step that follows.

Nothing here reads a switching log. The alarms stay as they are; only each interval's end moves.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Final

import numpy as np
from e00065_common import GRID_LENGTH
from e00065_detectors import DetectorInputs, refit_stamps
from e00065_intervals import BLOCK, Interval

RETURN_Z_MIN: Final[float] = 1.5
"""A return step must reach this many noise units in the opposite direction to count."""
MIN_DURATION: Final[int] = BLOCK
"""Shortest duration searched, in half-hours (24 hours)."""
END_SEARCH_STAMPS: Final[int] = 14 * BLOCK
"""The end is searched for in the 14 days after the onset."""


@cache
def near_refit() -> np.ndarray:
    """True within 24 hours of a monthly refit, where the residual model's jump can mimic a step."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        mask[max(stamp - BLOCK + 1, 0) : stamp + BLOCK] = True
    return mask


def end_only_intervals(*, intervals: list[Interval], inputs: DetectorInputs) -> list[Interval]:
    """Replace each interval's end by the strongest return step in the next 14 days (PE).

    A partner interval's step peaks up to one stamp from the alarm, so its direction is the sign of
    its own relative step. When no stamp reaches `RETURN_Z_MIN` in the opposite direction, the
    interval keeps its tracked end.
    """
    out: list[Interval] = []
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(iv.relative_step) if iv.role == "partner" else np.sign(z[iv.onset])
        stop = min(iv.onset + END_SEARCH_STAMPS + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = near_refit()[iv.onset + MIN_DURATION : stop]
        usable = np.where(np.isfinite(ret) & (ret >= RETURN_Z_MIN) & ~blocked, ret, -np.inf)
        if usable.size == 0 or not np.isfinite(usable.max()):
            out.append(iv)
            continue
        end = int(iv.onset + MIN_DURATION + np.argmax(usable))
        out.append(replace(iv, end=end, known_from=end + BLOCK))
    return out
