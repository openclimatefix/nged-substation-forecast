"""The three monitors of experiment 72: an L0 change-point fit, a rolling median, and HYST2.

**Every monitor takes one farm's signal (a value per day, NaN on a day without one) and returns the
alarms it raises, each a day index and a size.** An alarm is a downward change of at least
`MIN_STEP` in the level. The three monitors are offline: a day's level may read days after it.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np

from studies.capacity_posthoc import hysteresis_level

MIN_STEP: Final[float] = 0.05
"""The smallest downward change in level, as a share of the level, that raises an alarm."""

MIN_SEGMENT_VALID_DAYS: Final[int] = 14
"""A segment of the change-point fit holds at least this many valid days."""

WINSOR_HALF_WINDOW: Final[int] = 7
"""The winsorising window is the valid day and this many valid days either side."""

WINSOR_ROBUST_SDS: Final[float] = 3.0
"""Values are clipped to this many robust standard deviations from the local median."""

ROLLING_MEDIAN_VALID_DAYS: Final[int] = 28
"""The rolling median reads this many valid days, centred on the day."""

MAD_TO_SD: Final[float] = 1.4826
"""Scale from a median absolute deviation to a standard deviation of a normal distribution."""


@dataclass(frozen=True)
class Alarm:
    """One downward change that a monitor reports.

    Attributes:
        day: The index of the first day of the lower level.
        size: The share of the level lost, positive.
    """

    day: int
    size: float


def robust_sd(*, values: np.ndarray) -> float:
    """Return the robust standard deviation of the finite values (median absolute deviation).

    Args:
        values: A one-dimensional array with NaN for null.

    Returns:
        The robust standard deviation, 0 when fewer than two values are finite.
    """
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        return 0.0
    return float(MAD_TO_SD * np.median(np.abs(finite - np.median(finite))))


def winsorise(*, values: np.ndarray, sd: float) -> np.ndarray:
    """Clip each value to `WINSOR_ROBUST_SDS` robust standard deviations of its local median.

    Args:
        values: The valid-day values, one-dimensional and finite.
        sd: The farm's robust standard deviation.

    Returns:
        The clipped values.
    """
    out = np.empty_like(values)
    for index in range(len(values)):
        window = values[max(0, index - WINSOR_HALF_WINDOW) : index + WINSOR_HALF_WINDOW + 1]
        centre = np.median(window)
        out[index] = np.clip(values[index], centre - WINSOR_ROBUST_SDS * sd, centre + WINSOR_ROBUST_SDS * sd)
    return out


def optimal_partition(*, values: np.ndarray, penalty: float, min_length: int) -> list[int]:
    """Return the start indices of the segments that minimise squared error plus a penalty each.

    Args:
        values: A one-dimensional array of finite values.
        penalty: The cost of each segment after the first.
        min_length: The fewest values a segment holds.

    Returns:
        The sorted start index of every segment; always begins with 0. A single segment is `[0]`
        when there are fewer than twice `min_length` values.
    """
    count = len(values)
    if count < 2 * min_length:
        return [0]
    total = np.concatenate([[0.0], np.cumsum(values)])
    squares = np.concatenate([[0.0], np.cumsum(values**2)])
    best = np.full(count + 1, np.inf)
    previous = np.zeros(count + 1, dtype=int)
    best[0] = -penalty
    for end in range(min_length, count + 1):
        starts = np.arange(0, end - min_length + 1)
        length = end - starts
        segment_sum = total[end] - total[starts]
        cost = (squares[end] - squares[starts]) - segment_sum**2 / length
        candidate = best[starts] + cost + penalty
        choice = int(np.argmin(candidate))
        best[end] = candidate[choice]
        previous[end] = starts[choice]
    starts_found: list[int] = []
    end = count
    while end > 0:
        start = int(previous[end])
        starts_found.append(start)
        end = start
    return sorted(starts_found)


def change_point_alarms(*, signal: np.ndarray, penalty_scale: float) -> list[Alarm]:
    """Return the downward changes of at least `MIN_STEP` found by the L0 change-point fit.

    The penalty is `penalty_scale` times the robust variance times the logarithm of the number of
    valid days.

    Args:
        signal: One farm's daily signal (logarithmic ratio), NaN on a day without one.
        penalty_scale: The multiplier that the tuning draw sets.

    Returns:
        The alarms, in day order.
    """
    days = np.flatnonzero(np.isfinite(signal))
    if len(days) < 2 * MIN_SEGMENT_VALID_DAYS:
        return []
    values = signal[days]
    sd = robust_sd(values=values)
    if sd == 0.0:
        return []
    clipped = winsorise(values=values, sd=sd)
    penalty = penalty_scale * sd**2 * np.log(len(days))
    starts = optimal_partition(values=clipped, penalty=penalty, min_length=MIN_SEGMENT_VALID_DAYS)
    levels = [
        float(np.mean(clipped[start:stop]))
        for start, stop in zip(starts, [*starts[1:], len(clipped)], strict=True)
    ]
    alarms = []
    for index in range(1, len(starts)):
        share = 1.0 - np.exp(levels[index] - levels[index - 1])
        if share >= MIN_STEP:
            alarms.append(Alarm(day=int(days[starts[index]]), size=float(share)))
    return alarms


def fitted_level(*, signal: np.ndarray, penalty_scale: float) -> np.ndarray:
    """Return the change-point fit's level on every day, held between valid days.

    Args:
        signal: One farm's daily signal.
        penalty_scale: The penalty multiplier.

    Returns:
        An array with the signal's length, NaN before the first valid day.
    """
    days = np.flatnonzero(np.isfinite(signal))
    out = np.full(len(signal), np.nan)
    if len(days) < 2 * MIN_SEGMENT_VALID_DAYS:
        return out
    values = signal[days]
    sd = robust_sd(values=values)
    clipped = winsorise(values=values, sd=sd)
    penalty = penalty_scale * sd**2 * np.log(len(days))
    starts = optimal_partition(values=clipped, penalty=penalty, min_length=MIN_SEGMENT_VALID_DAYS)
    stops = [*starts[1:], len(clipped)]
    for start, stop in zip(starts, stops, strict=True):
        first = days[start]
        last = days[stop] if stop < len(days) else len(signal)
        out[first:last] = np.mean(clipped[start:stop])
    return out


def rolling_median_alarms(*, signal: np.ndarray, threshold: float) -> list[Alarm]:
    """Return an alarm where the centred rolling median first drops below minus `threshold`.

    An alarm ends when the median rises back above half the threshold, so one long loss raises one
    alarm.

    Args:
        signal: One farm's daily signal.
        threshold: The share below the farm's median level that raises an alarm.

    Returns:
        The alarms, in day order.
    """
    days = np.flatnonzero(np.isfinite(signal))
    if len(days) < ROLLING_MEDIAN_VALID_DAYS:
        return []
    values = signal[days]
    half = ROLLING_MEDIAN_VALID_DAYS // 2
    level = np.array(
        [np.median(values[max(0, i - half) : i + half + 1]) for i in range(len(values))]
    )
    baseline = np.median(values)
    drop = 1.0 - np.exp(level - baseline)
    alarms = []
    alarmed = False
    for index in range(len(values)):
        if not alarmed and drop[index] >= threshold:
            alarmed = True
            alarms.append(Alarm(day=int(days[index]), size=float(drop[index])))
        elif alarmed and drop[index] < threshold / 2:
            alarmed = False
    return alarms


def hysteresis_alarms(*, ratio: np.ndarray, drop: float) -> list[Alarm]:
    """Return an alarm where the hysteresis rule starts to follow the ratio downward.

    Args:
        ratio: One farm's deseasonalised level over its reference level (1 is normal), as
            `studies.capacity_seasonal.hyst2_ratio` returns it.
        drop: The share below 1 that counts as a drop.

    Returns:
        The alarms, in day order.
    """
    level = hysteresis_level(ratio=ratio, drop=drop)
    alarms = []
    for day in range(1, len(level)):
        if level[day - 1] == 1.0 and level[day] < 1.0 - drop:
            alarms.append(Alarm(day=day, size=float(1.0 - level[day])))
    return alarms
