"""Same-weekday error profiles on top of GAM-K, for experiment 142.

**Every arm is K0's level and transient plus a profile shift.** For a run with origin on UTC day
`O`, the target is the UTC day `T = O + 1`. A source day is a target day `S` in `[T - 15, T - 2]`:
its D+1 block came from the run of day `S - 1`, and the day `S` ended before `O` began, so it is
fully observed. Each source day's block is K0's median forecast minus the reading, by local
half-hour, with the day's mean removed (as in experiment 115's recipe R).

The all-days profile `A` averages the usable source days of the last 14. The weekday profile `W`
averages the usable source days of the same group as `T` (the same weekday, or the same day type)
among `[T - 7 n, T - 2]`. The shift is `-kappa * (A + lambda * (W - A))`, with `W = A` where `W`
has no usable day.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import e00142_arms as arms
import e00142_site as site_module
import numpy as np
from studies.dp_calendar import bank_holiday_names

PROFILE_DAYS: Final[int] = 14
MIN_PROFILE_DAYS: Final[int] = 3
MIN_DAY_READINGS: Final[int] = 40
FIRST_SOURCE_BACK: Final[int] = 2
"""The newest source day is `T - 2`: the target day `T - 1` is the origin's own day."""
PLACEBO_OFFSET_DAYS: Final[int] = 3
"""The placebo reads the weekday that falls this many days after the target's."""
DAY_TYPES: Final[tuple[int, ...]] = (0, 1, 1, 1, 2, 3, 4)
"""The day type of Monday (0) to Sunday (6): Monday, Tuesday to Thursday, Friday, Saturday, or
Sunday."""
CHRISTMAS_RUN_DAYS: Final[tuple[int, ...]] = (24, 27, 28, 29, 30, 31)
EPOCH: Final[np.datetime64] = np.datetime64("1970-01-01", "D")


def holiday_like_days(*, first: date, last: date) -> set[date]:
    """Return the English bank holidays, Easter Saturday and Sunday, and 24 and 27-31 December."""
    out = set(bank_holiday_names())
    for day, name in bank_holiday_names().items():
        if name == "Good Friday":
            out |= {day + timedelta(days=1), day + timedelta(days=2)}
    day = first
    while day <= last:
        if day.month == 12 and day.day in CHRISTMAS_RUN_DAYS:
            out.add(day)
        day += timedelta(days=1)
    return out


@dataclass(frozen=True)
class ProfileSpec:
    """How a profile shift is built.

    Attributes:
        kappa: The shrink applied to the profile.
        weeks: The number of weeks `n` of same-group days, or `None` for the all-days profile alone.
        weight: The weight lambda of the group profile against the all-days profile.
        group: `weekday`, `day_type`, or `placebo` (the weekday that falls 3 days after the target).
        holiday_rule: True to leave holiday-like days out of every profile and give a holiday-like
            target no correction.
    """

    kappa: float
    weeks: int | None = None
    weight: float = 0.0
    group: str = "weekday"
    holiday_rule: bool = True


def day_blocks(
    *, data: site_module.SiteData, k0_q50: np.ndarray, actual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each run's centred D+1 error block by local half-hour, and its target day number.

    Args:
        data: The site data.
        k0_q50: K0's median forecast at every row.
        actual: The reading at every row (`NaN` where there is none); may be an injected copy.

    Returns:
        Blocks of shape (runs, 2, 48): the sum and the count of each local half-hour's centred
        errors (a clock-change day holds one half-hour twice), zero for an unusable run, and the UTC
        day number (days since 1970-01-01) of each run's target day.
    """
    d1 = data.d1_rows()
    error = k0_q50 - actual
    blocks = np.zeros((len(data.runs), 2, 48))
    origin_days = data.run_origins.astype("datetime64[D]")
    for run in range(len(data.runs)):
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        rows = rows[d1[rows]]
        if len(rows) != 48:
            continue
        block = error[rows]
        if int(np.isfinite(block).sum()) < MIN_DAY_READINGS:
            continue
        centred = block - np.nanmean(block)
        finite = np.isfinite(centred)
        np.add.at(blocks[run, 0], data.row_half_hour[rows][finite], centred[finite])
        np.add.at(blocks[run, 1], data.row_half_hour[rows][finite], 1.0)
    targets = (origin_days + np.timedelta64(1, "D") - EPOCH).astype(np.int64)
    return blocks, targets


def _group_of(*, day_number: int, group: str) -> int:
    weekday = (day_number + 3) % 7  # 1970-01-01 was a Thursday; Monday is 0
    if group == "day_type":
        return DAY_TYPES[weekday]
    return weekday


def profile_shift_by_run(
    *, blocks: np.ndarray, targets: np.ndarray, spec: ProfileSpec
) -> np.ndarray:
    """Return each run's shift by local half-hour, shape (runs, 48): `-kappa` times the profile.

    A run whose all-days profile has fewer than `MIN_PROFILE_DAYS` usable days gets zeros, and so
    does a holiday-like target when the holiday rule is on.
    """
    day_of_block = {int(day): index for index, day in enumerate(targets)}
    first = date(1970, 1, 1)
    holiday = (
        holiday_like_days(
            first=first + timedelta(days=int(targets.min()) - 30),
            last=first + timedelta(days=int(targets.max()) + 30),
        )
        if spec.holiday_rule
        else set()
    )
    banned = {(d - first).days for d in holiday}
    out = np.zeros((len(targets), 48))
    for run, target in enumerate(targets.tolist()):
        if target in banned:
            continue

        def usable(day: int) -> np.ndarray | None:
            index = day_of_block.get(day)
            if index is None or day in banned:
                return None
            block = blocks[index]
            return block if block[1].sum() > 0 else None

        all_days = [
            b
            for back in range(FIRST_SOURCE_BACK, PROFILE_DAYS + FIRST_SOURCE_BACK)
            if (b := usable(target - back)) is not None
        ]
        if len(all_days) < MIN_PROFILE_DAYS:
            continue
        mean_all = _mean_by_half_hour(blocks=all_days)
        profile = mean_all
        if spec.weeks is not None and spec.weight > 0.0:
            if spec.group == "placebo":
                sources = [target - 7 * j + PLACEBO_OFFSET_DAYS for j in range(1, spec.weeks + 1)]
            else:
                wanted = _group_of(day_number=target, group=spec.group)
                sources = [
                    d
                    for d in range(target - 7 * spec.weeks, target - FIRST_SOURCE_BACK + 1)
                    if _group_of(day_number=d, group=spec.group) == wanted
                ]
            group_blocks = [b for d in sources if (b := usable(d)) is not None]
            if group_blocks:
                profile = mean_all + spec.weight * (
                    _mean_by_half_hour(blocks=group_blocks) - mean_all
                )
        out[run] = -spec.kappa * profile
    return out


def _mean_by_half_hour(*, blocks: list[np.ndarray]) -> np.ndarray:
    total = np.sum([b[0] for b in blocks], axis=0)
    count = np.sum([b[1] for b in blocks], axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def shift_at_rows(*, data: site_module.SiteData, by_run: np.ndarray) -> np.ndarray:
    """Spread each run's per-half-hour shift to the run's rows (every lead)."""
    return by_run[data.run_of_row, data.row_half_hour]


def profile_correction(
    *,
    data: site_module.SiteData,
    k0: arms.Correction,
    k0_q50: np.ndarray,
    spec: ProfileSpec,
    actual: np.ndarray | None = None,
) -> arms.Correction:
    """Return K0's level and transient plus the profile shift at every row."""
    blocks, targets = day_blocks(
        data=data, k0_q50=k0_q50, actual=data.actual if actual is None else actual
    )
    by_run = profile_shift_by_run(blocks=blocks, targets=targets, spec=spec)
    return arms.Correction(
        level=k0.level, anomaly=k0.anomaly, shape=shift_at_rows(data=data, by_run=by_run)
    )
