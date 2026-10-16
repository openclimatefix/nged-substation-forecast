"""The end rule: replace each alarm interval's end by the strongest opposite step within a cap.

The alarms stay as they are. Nothing here reads a switching log.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Final

import numpy as np
from e00066_common import GRID_LENGTH
from e00066_detectors import REFIT_PERIOD_WINDOW, DetectorInputs, refit_stamps
from e00066_intervals import BLOCK, Interval

RETURN_Z_MIN: Final[float] = 1.5
"""A return step must reach this many noise units in the opposite direction to count."""
MIN_DURATION: Final[int] = BLOCK
"""Shortest duration searched, in half-hours (24 hours)."""
CAPS_DAYS: Final[tuple[int, ...]] = (14, 30, 60)


@cache
def near_refit() -> np.ndarray:
    """True within 24 hours of a monthly refit, where the residual model's jump can mimic a step."""
    mask = np.zeros(GRID_LENGTH, dtype=bool)
    for stamp in refit_stamps():
        mask[max(stamp - REFIT_PERIOD_WINDOW + 1, 0) : stamp + REFIT_PERIOD_WINDOW] = True
    return mask


def end_rule_intervals(
    *, intervals: list[Interval], inputs: DetectorInputs, cap_days: int
) -> list[Interval]:
    """Replace each interval's end by the strongest return step in the next `cap_days` days.

    A return step is a stamp from 24 hours to `cap_days` after the onset where the step statistic
    runs opposite to the onset step by at least `RETURN_Z_MIN` noise units, outside the 24 hours
    around a monthly refit. The statistic is the one of `inputs`, so a 4-hour detector searches
    its own 4-hour steps. When no stamp qualifies, the interval keeps its tracked end.
    """
    out: list[Interval] = []
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(z[iv.onset])
        stop = min(iv.onset + cap_days * BLOCK + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = near_refit()[iv.onset + MIN_DURATION : stop]
        usable = np.where(np.isfinite(ret) & (ret >= RETURN_Z_MIN) & ~blocked, ret, -np.inf)
        if usable.size == 0 or not np.isfinite(usable.max()):
            out.append(iv)
            continue
        end = int(iv.onset + MIN_DURATION + np.argmax(usable))
        out.append(replace(iv, end=end, known_from=end + BLOCK))
    return out
