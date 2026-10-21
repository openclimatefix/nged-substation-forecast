"""The day-of-week arm of experiment 128 (KDS-dow): a shrunken same-weekday correction on KDS.

**The correction is a ridge-shrunken, exponentially weighted mean of the KDS filter's one-step
innovations at the same weekday and hour band of earlier weeks.** The innovation of a reading is the
observed residual minus the filter's prediction just before the reading. For a target on local day
`t` in hour band `b`, the day-band mean innovation `m(t', b)` is taken on the days `t' = t - 7k`
(`k = 1, 2, ...`), and the correction is

    sum_k forgetting**(k - 1) * m(t - 7k, b) / (sum_k forgetting**(k - 1) * present + kappa)

where `present` is 1 for a day-band with at least one reading. With `kappa` large the correction is
zero, which is the shared set of hour bands that KDS already uses for every weekday. A weekday's
correction leaves zero only if earlier weeks at that weekday and band agree. `forgetting = 0` is the
"copy what happened 7 days ago" memory.

The constants `forgetting` and `kappa` are fitted per substation on the tuning months by the sum of
squared one-step innovations left after the correction. Zero correction is a candidate, and wins
when no pair improves the score. A forecast reads only readings at or before its cutoff, and a
target day looks back at most `MAX_WEEKS` weeks.

The module also holds the pure last-week reference (`last_week_means`): the forecast is the actual
value 7 days (336 half-hours) before the target, if that value was read at or before the cutoff.
"""

from dataclasses import dataclass
from math import inf
from typing import Final

import e00128_filter as joint
import numpy as np

HALF_HOURS_PER_WEEK: Final[int] = 336
DAYS_PER_WEEK: Final[int] = 7
MAX_WEEKS: Final[int] = 26
"""How many weeks back a forecast looks. The largest forgetting factor weights week 26 by 0.003."""
FORGETTING_GRID: Final[tuple[float, ...]] = (0.0, 0.3, 0.5, 0.7, 0.8)
"""Candidate weekly forgetting factors; 0 keeps only last week."""
KAPPA_GRID: Final[tuple[float, ...]] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
"""Candidate shrinkage strengths, in day-band means (an effective count of weeks)."""


@dataclass(frozen=True)
class WeekdayConstants:
    """One substation's constants.

    Attributes:
        forgetting: The weight of each further week back, relative to the week before it.
        kappa: The shrinkage strength; `inf` switches the correction off.
        reduction: The fitted drop in the tuning months' sum of squared innovations, in points
            squared, 0 if the correction is off.
    """

    forgetting: float
    kappa: float
    reduction: float

    @property
    def is_off(self) -> bool:
        """Whether the correction is zero everywhere."""
        return self.kappa == inf


OFF: Final[WeekdayConstants] = WeekdayConstants(forgetting=0.0, kappa=inf, reduction=0.0)
"""The constants of a correction that is zero everywhere."""


def _day_band_sums(
    *, day: np.ndarray, band: np.ndarray, value: np.ndarray, first_day: int, n_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sum and count of values per (day, band), both shape (n_days, 4)."""
    sums = np.zeros((n_days, joint.N_BANDS))
    counts = np.zeros((n_days, joint.N_BANDS))
    np.add.at(sums, (day - first_day, band), value)
    np.add.at(counts, (day - first_day, band), 1.0)
    return sums, counts


def _weekly_memory(
    *, sums: np.ndarray, counts: np.ndarray, forgetting: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted sum of earlier same-weekday day-band means and the weights' sum.

    Row `t` of the first array is `sum_k forgetting**(k-1) * mean[t - 7k]` over days with a
    reading, and row `t` of the second is the sum of the same weights. Both have shape (n_days, 4).
    """
    n_days = len(sums)
    padded = -(-n_days // DAYS_PER_WEEK) * DAYS_PER_WEEK
    present = np.zeros((padded, joint.N_BANDS))
    mean = np.zeros((padded, joint.N_BANDS))
    present[:n_days] = counts > 0
    mean[:n_days] = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    total = np.zeros((padded, joint.N_BANDS))
    weight = np.zeros((padded, joint.N_BANDS))
    for start in range(0, padded - DAYS_PER_WEEK, DAYS_PER_WEEK):
        this_week = slice(start, start + DAYS_PER_WEEK)
        next_week = slice(start + DAYS_PER_WEEK, start + 2 * DAYS_PER_WEEK)
        total[next_week] = mean[this_week] * present[this_week] + forgetting * total[this_week]
        weight[next_week] = present[this_week] + forgetting * weight[this_week]
    return total[:n_days], weight[:n_days]


def fit_weekday_constants(
    *, trace: joint.JointTrace, band: np.ndarray, tuning_mask: np.ndarray
) -> WeekdayConstants:
    """Fit a substation's `forgetting` and `kappa` on the tuning months.

    Args:
        trace: The KDS filter's trace over the whole stream.
        band: The hour band of every grid row.
        tuning_mask: True on the grid rows that count in the tuning score.

    Returns:
        The pair with the lowest sum of squared innovations left on the tuning readings, or `OFF`
        if no pair lowers it.
    """
    if len(trace.positions) == 0:
        return OFF
    reading_band = band[trace.positions].astype(np.int64)
    first_day, n_days = int(trace.day[0]), int(trace.day[-1] - trace.day[0]) + 1
    sums, counts = _day_band_sums(
        day=trace.day, band=reading_band, value=trace.innovation, first_day=first_day, n_days=n_days
    )
    counted = tuning_mask[trace.positions]
    tuned_sums, tuned_counts = _day_band_sums(
        day=trace.day[counted],
        band=reading_band[counted],
        value=trace.innovation[counted],
        first_day=first_day,
        n_days=n_days,
    )
    best = OFF
    for forgetting in FORGETTING_GRID:
        total, weight = _weekly_memory(sums=sums, counts=counts, forgetting=forgetting)
        for kappa in KAPPA_GRID:
            correction = total / (weight + kappa)
            reduction = float(np.sum(2.0 * correction * tuned_sums - tuned_counts * correction**2))
            if reduction > best.reduction:
                best = WeekdayConstants(forgetting=forgetting, kappa=kappa, reduction=reduction)
    return best


def weekday_term(
    *,
    trace: joint.JointTrace,
    band: np.ndarray,
    constants: WeekdayConstants,
    cutoff: int,
    target_day: np.ndarray,
    target_band: np.ndarray,
) -> np.ndarray:
    """Return the correction of each target, from readings at or before the cutoff only.

    Args:
        trace: The KDS filter's trace over the whole stream.
        band: The hour band of every grid row.
        constants: The substation's fitted constants.
        cutoff: The newest grid row the forecast may see.
        target_day: The local day of each target (`e00128_common.local_day_index`).
        target_band: The hour band of each target.

    Returns:
        The correction of each target, in points; zeros if the correction is off or no reading is
        old enough.
    """
    zeros = np.zeros(len(target_day))
    seen = int(np.searchsorted(trace.positions, cutoff, side="right"))
    if constants.is_off or seen == 0 or len(target_day) == 0:
        return zeros
    first_day = max(int(target_day.min()) - DAYS_PER_WEEK * MAX_WEEKS, int(trace.day[0]))
    last_day = int(target_day.max())
    start = int(np.searchsorted(trace.day[:seen], first_day, side="left"))
    if start >= seen:
        return zeros
    sums, counts = _day_band_sums(
        day=trace.day[start:seen],
        band=band[trace.positions[start:seen]].astype(np.int64),
        value=trace.innovation[start:seen],
        first_day=first_day,
        n_days=last_day - first_day + 1,
    )
    total, weight = _weekly_memory(sums=sums, counts=counts, forgetting=constants.forgetting)
    rows = target_day.astype(np.int64) - first_day
    return total[rows, target_band] / (weight[rows, target_band] + constants.kappa)


def last_week_means(*, actual: np.ndarray, target: np.ndarray, cutoff: int) -> np.ndarray:
    """Return the reference forecast: the actual 336 half-hours before each target.

    Args:
        actual: The observed value on every grid row, `NaN` where there is no reading.
        target: The grid row of each target.
        cutoff: The newest grid row the forecast may see.

    Returns:
        The forecast of each target; `NaN` where last week's value is after the cutoff, before the
        grid, or unread.
    """
    source = target - HALF_HOURS_PER_WEEK
    usable = (source >= 0) & (source <= cutoff)
    out = np.full(len(target), np.nan)
    out[usable] = actual[source[usable]]
    return out
