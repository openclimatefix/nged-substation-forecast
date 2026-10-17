"""Injected losses, the week-shuffled placebo, and the scoring rules of experiment 72.

**An injection multiplies one farm's real power by `1 - size` from an onset day for a number of
days.** The injected farm alone is scored. The other farms' signals do not change, because the
neighbour medians come from the unmodified record (`e00072_signal.CleanBaseline`).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from e00072_detectors import (
    Alarm,
    change_point_alarms,
    hysteresis_alarms,
    rolling_median_alarms,
)

MethodType = Literal["changepoint", "rolling_median", "hyst"]
"""The three monitors."""

METHODS: Final[tuple[MethodType, ...]] = ("changepoint", "rolling_median", "hyst")

SeasonType = Literal["apr_sep", "oct_mar"]
"""The half-year in which a loss starts."""

SIZES: Final[tuple[float, ...]] = (0.0, 0.03, 0.10, 0.25)
"""Injected sizes; 0 is the control that measures how often a window holds an alarm by chance."""

DURATIONS: Final[tuple[int, ...]] = (30, 90)
"""Temporary losses last this many days."""

PERMANENT_SIZES: Final[tuple[float, ...]] = (0.10, 0.25)
"""Permanent losses, which last until the end of the record."""

DETECTION_WINDOW_DAYS: Final[int] = 28
"""A loss is detected by a downward alarm from its first day to this many days after."""

EARLY_WINDOW_DAYS: Final[int] = 14
"""The post hoc sensitivity window starts this many days before a loss's first day."""

MIN_SPACING_DAYS: Final[int] = 150
"""Losses planted in one run start at least this many days apart, and after their footprint."""

PLACEMENT_TRIES: Final[int] = 2000
"""Random onsets tried before a draw gives up placing a run's losses."""

DAYS_PER_YEAR: Final[float] = 365.25

AlarmFunction = Callable[[np.ndarray], list[Alarm]]
"""A monitor with its one tuned parameter fixed, from a farm's signal to its alarms."""


@dataclass(frozen=True)
class Loss:
    """One planted loss.

    Attributes:
        size: The share of power lost; 0 is a control with no change.
        first_day: The index of the first day.
        days: The duration in days; `None` for a loss that lasts to the end of the record.
        season: The half-year of `first_day`.
    """

    size: float
    first_day: int
    days: int | None
    season: SeasonType


def make_alarm_function(*, method: MethodType, parameter: float) -> AlarmFunction:
    """Return a monitor with its tuned parameter fixed.

    Args:
        method: Which monitor.
        parameter: The change-point penalty multiplier, or the rolling-median threshold, or the
            hysteresis drop.

    Returns:
        A function from a farm's signal to its alarms.
    """
    if method == "changepoint":
        return lambda signal: change_point_alarms(signal=signal, penalty_scale=parameter)
    if method == "rolling_median":
        return lambda signal: rolling_median_alarms(signal=signal, threshold=parameter)
    return lambda signal: hysteresis_alarms(ratio=np.exp(signal), drop=parameter)


def season_of(*, dates: np.ndarray, day: int) -> SeasonType:
    """Return the half-year that a day index falls in."""
    month = int(dates[day].astype("datetime64[M]").astype(int) % 12) + 1
    return "apr_sep" if 4 <= month <= 9 else "oct_mar"


def place_losses(
    *,
    rng: np.random.Generator,
    dates: np.ndarray,
    valid_range: tuple[int, int],
    allowed: np.ndarray,
    specs: Sequence[tuple[float, int | None]],
    season: SeasonType,
) -> list[Loss] | None:
    """Draw onsets for a run's losses, apart from each other and inside one half-year.

    A loss that lasts to the end of the record must be the only loss in its run.

    Args:
        rng: The random generator.
        dates: One calendar day per index.
        valid_range: The first and last (exclusive) day index of the farm's record.
        allowed: A boolean per day; False where a loss may not start (near farm A's low months).
        specs: Each loss's size and duration in days (`None` for permanent).
        season: The half-year every onset falls in.

    Returns:
        The losses in onset order, or `None` when no placement was found.
    """
    first, last = valid_range
    candidates = np.array(
        [
            day
            for day in range(first + 60, last - 120)
            if allowed[day] and season_of(dates=dates, day=day) == season
        ]
    )
    if len(candidates) < len(specs):
        return None
    for _ in range(PLACEMENT_TRIES):
        onsets = sorted(int(day) for day in rng.choice(candidates, size=len(specs), replace=False))
        shuffled = [specs[i] for i in rng.permutation(len(specs))]
        ends = [onset + (days or 0) for onset, (_, days) in zip(onsets, shuffled, strict=True)]
        gaps = [onsets[i + 1] - ends[i] for i in range(len(onsets) - 1)]
        if all(gap >= MIN_SPACING_DAYS for gap in gaps):
            return [
                Loss(size=size, first_day=onset, days=days, season=season)
                for onset, (size, days) in zip(onsets, shuffled, strict=True)
            ]
    return None


def inject(*, power_mw: np.ndarray, losses: Sequence[Loss]) -> np.ndarray:
    """Return a farm's hourly power with the losses applied.

    Args:
        power_mw: The farm's hourly power, shape (days, 24).
        losses: The planted losses.

    Returns:
        A copy of the power, scaled by `1 - size` over each loss.
    """
    out = power_mw.copy()
    for loss in losses:
        stop = len(out) if loss.days is None else loss.first_day + loss.days
        out[loss.first_day : stop] *= 1.0 - loss.size
    return out


def loss_stop(*, loss: Loss, total_days: int) -> int:
    """Return the first day index after a loss ends."""
    return total_days if loss.days is None else loss.first_day + loss.days


def detected(*, alarms: Sequence[Alarm], loss: Loss, early_days: int = 0) -> Alarm | None:
    """Return the first alarm in the loss's detection window, or `None`.

    Args:
        alarms: The farm's alarms.
        loss: The planted loss.
        early_days: How many days before the loss's first day the window starts; 0 is the
            pre-registered window.

    Returns:
        The first alarm from `early_days` before the first day to `DETECTION_WINDOW_DAYS` after.
    """
    for alarm in alarms:
        if loss.first_day - early_days <= alarm.day <= loss.first_day + DETECTION_WINDOW_DAYS:
            return alarm
    return None


def farm_years(*, signal: np.ndarray) -> float:
    """Return the years from a farm's first signal day to its last."""
    days = np.flatnonzero(np.isfinite(signal))
    return 0.0 if len(days) < 2 else float((days[-1] - days[0]) / DAYS_PER_YEAR)


def week_shuffle(*, signal: np.ndarray, dates: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute whole weeks of a signal among the years that share a week of the year.

    Args:
        signal: One farm's daily signal.
        dates: One calendar day per index, as `datetime64[D]`.
        rng: The random generator.

    Returns:
        A signal of the same length. Only the days from the farm's first to its last valid day are
        permuted, and the days outside stay missing. Blocks of different lengths (the record's
        partial ends and the year's last long week) are permuted only among blocks of their own
        length.
    """
    finite = np.flatnonzero(np.isfinite(signal))
    if len(finite) == 0:
        return signal.copy()
    first, last = int(finite[0]), int(finite[-1]) + 1
    inner_dates = dates[first:last]
    years = inner_dates.astype("datetime64[Y]").astype(int)
    day_of_year = (
        inner_dates - inner_dates.astype("datetime64[Y]").astype("datetime64[D]")
    ).astype(int)
    week = np.minimum(day_of_year // 7, 51)
    blocks: dict[tuple[int, int], list[int]] = {}
    for index, (year, week_of_year) in enumerate(zip(years, week, strict=True)):
        blocks.setdefault((int(year), int(week_of_year)), []).append(index + first)
    groups: dict[tuple[int, int], list[list[int]]] = {}
    for (_, week_of_year), indices in blocks.items():
        groups.setdefault((week_of_year, len(indices)), []).append(indices)
    out = signal.copy()
    for members in groups.values():
        permuted = rng.permutation(len(members))
        for target, source in enumerate(permuted):
            out[members[target]] = signal[members[source]]
    return out


def tuned_parameter(
    *,
    rates: Mapping[float, float],
    target: float,
) -> float:
    """Return the most sensitive parameter whose false-alarm rate is at most the target.

    Args:
        rates: The false alarms per farm-year at each candidate parameter.
        target: The largest false-alarm rate allowed.

    Returns:
        The parameter with the highest rate not above `target`; the one with the lowest rate when
        none meets it.
    """
    meeting = {parameter: rate for parameter, rate in rates.items() if rate <= target}
    if not meeting:
        return min(rates, key=lambda parameter: rates[parameter])
    return max(meeting, key=lambda parameter: meeting[parameter])
