"""The pre-registered shape rules of experiment 175, as pure functions on residual arrays.

A residual array is one value per half-hour on a continuous grid, `NaN` where a row is missing.
Every threshold below is fixed in the plan (`result.md`) before any day is looked at.
"""

from dataclasses import dataclass
from datetime import date
from typing import Final, Literal

import numpy as np

ShapeType = Literal["a_block", "d_spike", "b_offset", "c_ramp_or_midday", "mixed"]

JUMP_FACTOR: Final[float] = 3.0
"""A block edge needs a jump and a shift of more than this many day-before spreads."""
NEIGHBOUR_FACTOR: Final[float] = 2.0
"""A neighbour's opposite step needs more than this many of the neighbour's own spreads."""
SHARP_SHARE: Final[float] = 0.5
"""The 1-hour jump must be at least this share of the 4-hour shift (a step, not a ramp)."""
JUMP_READINGS: Final[int] = 2
"""Readings on each side of the 1-hour jump test."""
SHIFT_READINGS: Final[int] = 8
"""Readings on each side of the 4-hour persistence test."""
MIN_SIDE_READINGS: Final[int] = 4
"""Valid readings needed on each side of the persistence test."""
SPIKE_FACTOR: Final[float] = 4.0
SPIKE_MAX_READINGS: Final[int] = 3
SPIKE_MIN_SHARE: Final[float] = 0.5
OFFSET_FACTOR: Final[float] = 1.0
"""The day's median residual must exceed this many day-before spreads for class (b)."""
OFFSET_SIGN_SHARE: Final[float] = 0.75
MIDDAY_HOURS: Final[tuple[float, float]] = (10.0, 16.0)
MIDDAY_MIN_SHARE: Final[float] = 0.5
RAMP_SMOOTH_READINGS: Final[int] = 6
RAMP_TOLERANCE_FACTOR: Final[float] = 0.5
RAMP_MIN_RANGE_FACTOR: Final[float] = 2.0
SPREAD_FLOOR_QUANTILE: Final[float] = 0.10
SPREAD_LOOKBACK_DAYS: Final[int] = 7
RETURN_MIN_SIZE_SHARE: Final[float] = 0.5
NEIGHBOUR_WINDOW_READINGS: Final[int] = 4
"""Two hours either side of the substation's edge."""
MAD_TO_STD: Final[float] = 1.4826


@dataclass(frozen=True)
class Edge:
    """A sharp, persistent step in a residual series.

    Attributes:
        position: Grid position of the first reading after the step.
        shift: Median of the 8 readings after minus median of the 8 before.
        strength: `abs(shift)` over the spread used for the test.
    """

    position: int
    shift: float
    strength: float


def spread(*, values: np.ndarray) -> float:
    """Return 1.4826 times the median absolute deviation of the valid values, or NaN."""
    valid = values[np.isfinite(values)]
    if valid.size < 24:
        return float("nan")
    return float(MAD_TO_STD * np.median(np.abs(valid - np.median(valid))))


def day_before_spreads(
    *, residual: np.ndarray, local_dates: np.ndarray, days: list[date]
) -> dict[date, float]:
    """Return the spread `s` for each requested day, taken from the previous local day.

    A previous day with fewer than 24 valid readings falls back to the nearest earlier day within
    `SPREAD_LOOKBACK_DAYS` that has enough, then to the floor. The floor is the 10th percentile of
    the series' daily spreads over the grid.
    """
    unique = np.unique(local_dates)
    daily = {day: spread(values=residual[local_dates == day]) for day in unique}
    valid = np.array([v for v in daily.values() if np.isfinite(v)])
    floor = float(np.quantile(valid, SPREAD_FLOOR_QUANTILE))
    out: dict[date, float] = {}
    for day in days:
        value = float("nan")
        for back in range(1, SPREAD_LOOKBACK_DAYS + 1):
            candidate = daily.get(np.datetime64(day) - np.timedelta64(back, "D"), float("nan"))
            if np.isfinite(candidate):
                value = candidate
                break
        out[day] = max(value, floor) if np.isfinite(value) else floor
    return out


def _median_before_after(
    *, residual: np.ndarray, position: int, width: int, minimum: int
) -> tuple[float, float]:
    """Return the medians of the `width` readings before and from `position`, NaN if too few."""
    before = residual[max(position - width, 0) : position]
    after = residual[position : position + width]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if before.size < minimum or after.size < minimum:
        return float("nan"), float("nan")
    return float(np.median(before)), float(np.median(after))


def step_candidates(*, residual: np.ndarray, positions: np.ndarray, scale: float) -> list[Edge]:
    """Return the edges among `positions` whose jump and shift both exceed `JUMP_FACTOR * scale`.

    The jump compares the 2 readings either side of the position (within 1 hour). The shift
    compares the 8 readings either side (at least 4 valid each) and must have the jump's sign.
    Consecutive qualifying positions are merged into the one with the largest shift.
    """
    threshold = JUMP_FACTOR * scale
    passing: list[Edge] = []
    for position in positions:
        jump_before, jump_after = _median_before_after(
            residual=residual, position=int(position), width=JUMP_READINGS, minimum=1
        )
        shift_before, shift_after = _median_before_after(
            residual=residual,
            position=int(position),
            width=SHIFT_READINGS,
            minimum=MIN_SIDE_READINGS,
        )
        jump = jump_after - jump_before
        shift = shift_after - shift_before
        if (
            np.isfinite(jump)
            and np.isfinite(shift)
            and abs(jump) > threshold
            and abs(shift) > threshold
            and jump * shift > 0
            and abs(jump) >= SHARP_SHARE * abs(shift)
        ):
            passing.append(Edge(int(position), shift, abs(shift) / scale))
    merged: list[Edge] = []
    for edge in passing:
        if merged and edge.position - merged[-1].position <= JUMP_READINGS:
            if edge.strength > merged[-1].strength:
                merged[-1] = edge
        else:
            merged.append(edge)
    return merged


def _is_spike(*, day: np.ndarray, scale: float) -> bool:
    valid = day[np.isfinite(day)]
    if valid.size == 0:
        return False
    deviation = np.abs(valid - np.median(valid))
    far = deviation > SPIKE_FACTOR * scale
    return bool(
        1 <= far.sum() <= SPIKE_MAX_READINGS
        and deviation[far].sum() >= SPIKE_MIN_SHARE * deviation.sum()
    )


def _is_offset(*, day: np.ndarray, scale: float) -> bool:
    valid = day[np.isfinite(day)]
    if valid.size == 0:
        return False
    median = np.median(valid)
    same_sign = np.mean(np.sign(valid) == np.sign(median))
    return bool(abs(median) > OFFSET_FACTOR * scale and same_sign >= OFFSET_SIGN_SHARE)


def _is_ramp_or_midday(*, day: np.ndarray, hours: np.ndarray, scale: float) -> bool:
    valid = np.isfinite(day)
    if valid.sum() < 2:
        return False
    squared = np.where(valid, day, 0.0) ** 2
    midday = (hours >= MIDDAY_HOURS[0]) & (hours < MIDDAY_HOURS[1])
    if squared.sum() > 0 and squared[midday].sum() >= MIDDAY_MIN_SHARE * squared.sum():
        return True
    values = day[valid]
    if values.size < RAMP_SMOOTH_READINGS:
        return False
    kernel = np.lib.stride_tricks.sliding_window_view(values, RAMP_SMOOTH_READINGS)
    smooth = np.median(kernel, axis=1)
    steps = np.diff(smooth)
    tolerance = RAMP_TOLERANCE_FACTOR * scale
    monotone = bool(np.all(steps >= -tolerance) or np.all(steps <= tolerance))
    return monotone and float(smooth.max() - smooth.min()) > RAMP_MIN_RANGE_FACTOR * scale


def classify_day(
    *, day: np.ndarray, hours: np.ndarray, edges: list[Edge], scale: float
) -> ShapeType:
    """Return the first matching class in the order a, d, b, c, else `mixed`.

    Args:
        day: The day's residual (observed minus forecast), `NaN` where a row is missing.
        hours: Local hour of day of each of the day's half-hours.
        edges: The day's block edges from `step_candidates`.
        scale: The day-before spread `s`.
    """
    if edges:
        return "a_block"
    if _is_spike(day=day, scale=scale):
        return "d_spike"
    if _is_offset(day=day, scale=scale):
        return "b_offset"
    if _is_ramp_or_midday(day=day, hours=hours, scale=scale):
        return "c_ramp_or_midday"
    return "mixed"


def returns_within_day(*, primary: Edge, edges: list[Edge]) -> bool:
    """Return whether an opposite edge of at least half the primary's size follows it."""
    return any(
        edge.position > primary.position
        and edge.shift * primary.shift < 0
        and abs(edge.shift) >= RETURN_MIN_SIZE_SHARE * abs(primary.shift)
        for edge in edges
    )


def neighbour_opposite_step(
    *, residual: np.ndarray, edge_position: int, scale: float, sign: float
) -> bool:
    """Return whether a neighbour has an opposite-signed step within 2 hours of `edge_position`.

    The neighbour's step statistic is the 8-reading shift, and must exceed `NEIGHBOUR_FACTOR *
    scale` with the sign opposite to `sign`.
    """
    for position in range(
        edge_position - NEIGHBOUR_WINDOW_READINGS, edge_position + NEIGHBOUR_WINDOW_READINGS + 1
    ):
        before, after = _median_before_after(
            residual=residual, position=position, width=SHIFT_READINGS, minimum=MIN_SIDE_READINGS
        )
        shift = after - before
        if np.isfinite(shift) and abs(shift) > NEIGHBOUR_FACTOR * scale and shift * sign < 0:
            return True
    return False


def persists(
    *, residual: np.ndarray, local_dates: np.ndarray, day: date, shift: float
) -> bool | None:
    """Return whether the 7 later days' median level sits at least half a step from the 7 earlier.

    Returns `None` when fewer than 3 days on either side have 24 valid readings.
    """

    def daily_median(offset: int) -> float:
        values = residual[local_dates == np.datetime64(day) + np.timedelta64(offset, "D")]
        valid = values[np.isfinite(values)]
        return float(np.median(valid)) if valid.size >= 24 else float("nan")

    after = np.array([daily_median(k) for k in range(1, 8)])
    before = np.array([daily_median(-k) for k in range(1, 8)])
    if np.isfinite(after).sum() < 3 or np.isfinite(before).sum() < 3:
        return None
    difference = float(np.nanmedian(after) - np.nanmedian(before))
    return difference * shift > 0 and abs(difference) >= RETURN_MIN_SIZE_SHARE * abs(shift)
