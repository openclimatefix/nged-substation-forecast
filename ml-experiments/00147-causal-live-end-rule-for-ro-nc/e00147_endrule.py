"""The end rule: replace each alarm interval's end by the strongest opposite step within a cap.

The alarms stay as they are. Nothing here reads a switching log.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache
from typing import Final

import numpy as np
import polars as pl
from e00147_common import GRID_LENGTH
from e00147_detectors import REFIT_PERIOD_WINDOW, DetectorInputs, refit_stamps
from e00147_intervals import BLOCK, Interval

RETURN_Z_MIN: Final[float] = 1.5
"""A return step must reach this many noise units in the opposite direction to count."""
MIN_DURATION: Final[int] = BLOCK
"""Shortest duration searched, in half-hours (24 hours)."""
CAPS_DAYS: Final[tuple[int, ...]] = (14, 30, 60)


def blocked_by_refit(*, inputs: DetectorInputs) -> np.ndarray:
    """Stamps where a return step is ignored: near a refit, unless `inputs` never crosses one."""
    if not inputs.suppress_refits:
        return np.zeros(GRID_LENGTH, dtype=bool)
    return near_refit()


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
    blocked_mask = blocked_by_refit(inputs=inputs)
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(z[iv.onset])
        stop = min(iv.onset + cap_days * BLOCK + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = blocked_mask[iv.onset + MIN_DURATION : stop]
        usable = np.where(np.isfinite(ret) & (ret >= RETURN_Z_MIN) & ~blocked, ret, -np.inf)
        if usable.size == 0 or not np.isfinite(usable.max()):
            out.append(iv)
            continue
        end = int(iv.onset + MIN_DURATION + np.argmax(usable))
        out.append(replace(iv, end=end, known_from=end + BLOCK))
    return out


FIRST_SHARE: Final[float] = 0.5
"""A first-return step must reach this share of the onset step's size (post hoc arm)."""


def first_return_intervals(
    *, intervals: list[Interval], inputs: DetectorInputs, cap_days: int
) -> list[Interval]:
    """Post hoc: end each interval at the first strong return step instead of the strongest one.

    The first stamp from 24 hours to `cap_days` after the onset where the opposite step reaches
    `FIRST_SHARE` of the onset step (and `RETURN_Z_MIN` noise units) fixes where the return begins.
    The end is then the stamp of the largest opposite step within one detector window after that
    stamp. When no stamp qualifies, the interval keeps its tracked end.
    """
    out: list[Interval] = []
    blocked_mask = blocked_by_refit(inputs=inputs)
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(z[iv.onset])
        stop = min(iv.onset + cap_days * BLOCK + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = blocked_mask[iv.onset + MIN_DURATION : stop]
        need = max(RETURN_Z_MIN, FIRST_SHARE * abs(z[iv.onset]))
        usable = np.where(np.isfinite(ret) & (ret >= need) & ~blocked, ret, -np.inf)
        hits = np.flatnonzero(np.isfinite(usable))
        if hits.size == 0:
            out.append(iv)
            continue
        first = int(hits[0])
        best = first + int(np.argmax(usable[first : first + inputs.window + 1]))
        end = int(iv.onset + MIN_DURATION + best)
        out.append(replace(iv, end=end, known_from=end + BLOCK))
    return out


RULES: Final[tuple[str, ...]] = ("R0", "R1", "R2", "R3")
"""R0 is the offline rule, and R1 to R3 call the end with no data from after the call."""
PEAK_FALL: Final[float] = 0.8
"""R2 calls the end once the opposite step falls below this share of the largest seen so far."""
HOLD_STAMPS: Final[int] = 12
"""R3 calls the end after the opposite step stays above the threshold for this many stamps."""


def live_end_intervals(
    *, intervals: list[Interval], inputs: DetectorInputs, rule: str, cap_days: int
) -> list[Interval]:
    """End each interval by one of the rules, with `known_from` the stamp at which it is called.

    The step statistic at stamp t averages the `inputs.window` stamps from t, so it can be computed
    at t + window. The first return step is found as in `first_return_intervals`. Then:

    - R0 (offline): the end is the largest opposite step within one window after the first return
      step, which is known at the first return step plus two windows.
    - R1: the end is the first return step, called at that stamp plus one window.
    - R2: the end is the largest opposite step so far, called when the opposite step falls below
      `PEAK_FALL` of that largest step, or one window after the first return step.
    - R3: the end is the first stamp from which the opposite step stays above the threshold for
      `HOLD_STAMPS` stamps, called at the last of them plus one window.

    When no stamp qualifies, the interval keeps its tracked end and the stamp it was known from.
    """
    out: list[Interval] = []
    blocked_mask = blocked_by_refit(inputs=inputs)
    window = inputs.window
    for iv in intervals:
        z = inputs.shift[iv.series] / inputs.scale[iv.series]
        sign = np.sign(z[iv.onset])
        stop = min(iv.onset + cap_days * BLOCK + 1, GRID_LENGTH)
        ret = -sign * z[iv.onset + MIN_DURATION : stop]
        blocked = blocked_mask[iv.onset + MIN_DURATION : stop]
        need = max(RETURN_Z_MIN, FIRST_SHARE * abs(z[iv.onset]))
        qualifies = np.isfinite(ret) & (ret >= need) & ~blocked
        usable = np.where(qualifies, ret, -np.inf)
        hits = np.flatnonzero(qualifies)
        base = iv.onset + MIN_DURATION
        if hits.size == 0:
            out.append(iv)
            continue
        first = int(hits[0])
        if rule == "R0":
            best = first + int(np.argmax(usable[first : first + window + 1]))
            end, known = base + best, base + first + 2 * window
        elif rule == "R1":
            end, known = base + first, base + first + window
        elif rule == "R2":
            peak, peak_at, known = -np.inf, first, base + first + 2 * window
            for k in range(window + 1):
                if first + k >= ret.size:
                    break
                value = ret[first + k] if np.isfinite(ret[first + k]) else -np.inf
                if value > peak:
                    peak, peak_at = value, first + k
                elif k > 0 and value < PEAK_FALL * peak:
                    known = base + first + k + window
                    break
            end = base + peak_at
        elif rule == "R3":
            held = [
                j
                for j in hits
                if j + HOLD_STAMPS <= qualifies.size and qualifies[j : j + HOLD_STAMPS].all()
            ]
            if not held:
                out.append(iv)
                continue
            end = base + int(held[0])
            known = end + HOLD_STAMPS - 1 + window
        else:
            raise ValueError(rule)
        out.append(replace(iv, end=int(end), known_from=int(known)))
    return out


MERGE_STAMPS: Final[tuple[int, ...]] = (12, 48, 96, 192)
"""Merge tolerances in half-hours: M12 is 6 hours and M48 is 24 hours (pre-registered), and M96 and
M192, 2 and 4 days, were added after M12 and M48 gave identical results (post hoc)."""
RETURN_GRACE: Final[int] = BLOCK
"""An opposite-sign alarm this soon after a called end is still the return step of that event."""


def filter_alarms(*, table: pl.DataFrame, live: list[Interval], tolerance: int) -> pl.DataFrame:
    """Drop alarms that repeat or close an earlier event on the same series, using only the past.

    An alarm is dropped when an earlier kept alarm on its series fired within `tolerance`
    half-hours, or when its step is opposite to a kept interval of its series that is still open
    at the alarm's stamp (its end has not been called) or was called less than `RETURN_GRACE`
    before. `live` holds the intervals that the end rule gives for the unfiltered `table`; each is
    matched to its alarm by series and onset.
    """
    by_alarm = {(iv.series, iv.onset): iv for iv in live if iv.role == "alarm"}
    kept: list[dict] = []
    for row in table.sort("stamp").iter_rows(named=True):
        series, stamp = row["series"], row["stamp"]
        sign = np.sign(row["step_mw"])
        drop = False
        for earlier in kept:
            if earlier["series"] != series:
                continue
            if stamp - earlier["stamp"] <= tolerance:
                drop = True
                break
            interval = by_alarm.get((series, earlier["stamp"]))
            opposite = np.sign(earlier["step_mw"]) != sign
            if interval is not None and opposite and stamp < interval.known_from + RETURN_GRACE:
                drop = True
                break
        if not drop:
            kept.append(row)
    return pl.DataFrame(kept, schema=table.schema)
