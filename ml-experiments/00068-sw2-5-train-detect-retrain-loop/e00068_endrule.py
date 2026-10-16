"""PE, the end rule of experiment 61: each interval ends at the strongest opposite step.

D0M's alarms are kept. The end of each alarm's interval is the stamp, between 24 hours and 14 days
after the onset, where the 24-hour step back towards the starting level is largest, provided that
step is at least `RETURN_Z_MIN` noise units. Stamps within 24 hours of a monthly refit are skipped.
When no stamp qualifies, the interval keeps the end that D0M's level tracker gave it.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Final

import numpy as np
from e00068_common import GRID_LENGTH
from e00068_detectors import DetectorInputs, refit_stamps
from e00068_intervals import BLOCK, Interval

RETURN_Z_MIN: Final[float] = 1.5
"""A return step must reach this many noise units in the opposite direction to count."""
MIN_DURATION: Final[int] = BLOCK
"""Shortest duration searched, in half-hours (24 hours)."""
END_SEARCH_STAMPS: Final[int] = 14 * BLOCK
"""The end search stops 14 days after the onset."""


@cache
def near_refit() -> np.ndarray:
    """True within 24 hours of a monthly refit, where the residual model's jump can mimic a step."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        mask[max(stamp - BLOCK + 1, 0) : stamp + BLOCK] = True
    return mask


def end_only_intervals(*, intervals: list[Interval], inputs: DetectorInputs) -> list[Interval]:
    """Replace each interval's end by the strongest return step in the next 14 days (PE)."""
    out: list[Interval] = []
    blocked_all = near_refit()
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(z[iv.onset])
        stop = min(iv.onset + END_SEARCH_STAMPS + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = blocked_all[iv.onset + MIN_DURATION : stop]
        usable = np.where(np.isfinite(ret) & (ret >= RETURN_Z_MIN) & ~blocked, ret, -np.inf)
        if usable.size == 0 or not np.isfinite(usable.max()):
            out.append(iv)
            continue
        end = int(iv.onset + MIN_DURATION + np.argmax(usable))
        out.append(replace(iv, end=end, known_from=end + BLOCK))
    return out
