"""The pre-registered holiday set and the day categories experiment 143 scores.

**The holiday set is fixed in the plan before any fit.** It is every England and Wales bank holiday
(including the substitute days the `holidays` package lists as observed) and every day from 24
December to 2 January. Days are local (Europe/London) dates. The categories below are disjoint, so
every scored row has exactly one.
"""

from datetime import date, timedelta
from typing import Final

import numpy as np
from studies.dp_calendar import bank_holiday_names

CHRISTMAS_FIRST: Final[tuple[int, int]] = (12, 24)
"""24 December: the first day of the Christmas window."""
CHRISTMAS_LAST: Final[tuple[int, int]] = (1, 2)
"""2 January: the last day of the Christmas window."""
AFTER_DAYS: Final[int] = 5
"""How many days after the end of a holiday cluster count as leakage days."""
CLUSTER_GAP_DAYS: Final[int] = 3
"""Holiday days at most this far apart (bank holiday, weekend, bank holiday) are one cluster."""
EPOCH: Final[np.datetime64] = np.datetime64("1970-01-01", "D")

CATEGORIES: Final[tuple[str, ...]] = (
    "christmas_window",
    "bank_holiday",
    "after_christmas",
    "after_bank_holiday",
    "other",
)
"""Disjoint day categories: the Christmas window (24 December to 2 January), a bank holiday outside
it, days 1 to 5 after the end of the Christmas window, days 1 to 5 after the end of any other
holiday cluster, and every remaining day."""
EXTRA_GROUPS: Final[dict[str, str]] = {
    "holiday": "christmas_window|bank_holiday",
    "after_holiday": "after_christmas|after_bank_holiday",
}


def in_christmas_window(*, day: date) -> bool:
    """Return True for 24 December to 2 January."""
    return (day.month == CHRISTMAS_FIRST[0] and day.day >= CHRISTMAS_FIRST[1]) or (
        day.month == CHRISTMAS_LAST[0] and day.day <= CHRISTMAS_LAST[1]
    )


def holiday_days(*, first: date, last: date) -> set[date]:
    """Return the holiday set within `first` to `last`, both included."""
    bank = {d for d in bank_holiday_names() if first <= d <= last}
    window = {
        first + timedelta(days=i)
        for i in range((last - first).days + 1)
        if in_christmas_window(day=first + timedelta(days=i))
    }
    return bank | window


def day_numbers(*, days: set[date]) -> np.ndarray:
    """Return the sorted days since 1970-01-01."""
    return np.array(sorted((np.datetime64(d, "D") - EPOCH).astype(np.int64) for d in days))


def clusters(*, holiday: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted holiday day numbers into clusters; return each cluster's first and last day."""
    out: list[tuple[int, int]] = []
    for day in holiday:
        if out and day - out[-1][1] <= CLUSTER_GAP_DAYS:
            out[-1] = (out[-1][0], int(day))
        else:
            out.append((int(day), int(day)))
    return out


def categorise(*, local_day: np.ndarray, holiday: np.ndarray) -> np.ndarray:
    """Label each local day number with its category.

    Args:
        local_day: Days since 1970-01-01, one per row.
        holiday: The sorted holiday day numbers.

    Returns:
        A string array of `CATEGORIES`.
    """
    out = np.full(local_day.shape, "other", dtype="<U20")
    for _first, last in clusters(holiday=holiday):
        window = in_christmas_window(day=(EPOCH + np.timedelta64(last, "D")).astype(date))
        after = (local_day > last) & (local_day <= last + AFTER_DAYS)
        out[after] = "after_christmas" if window else "after_bank_holiday"
    is_holiday = np.isin(local_day, holiday)
    window_days = np.array(
        [
            in_christmas_window(day=(EPOCH + np.timedelta64(int(d), "D")).astype(date))
            for d in holiday
        ]
    )
    in_window = np.isin(local_day, holiday[window_days])
    out[is_holiday] = "bank_holiday"
    out[in_window] = "christmas_window"
    return out


def skip_days(*, holiday: np.ndarray, also_day_after: bool) -> np.ndarray:
    """Return the local day numbers whose readings the filter ignores.

    Args:
        holiday: The sorted holiday day numbers.
        also_day_after: True to add each non-holiday day that follows a holiday day.

    Returns:
        The sorted day numbers.
    """
    days = {int(d) for d in holiday}
    if also_day_after:
        days |= {int(d) + 1 for d in holiday}
    return np.array(sorted(days))


def core_days(*, holiday: np.ndarray) -> np.ndarray:
    """Return the holiday days minus 27 December to 2 January that are not bank holidays.

    This is the set the `K-skip-core` arm skips: bank holidays and 24 to 26 December, so that
    27 to 30 December targets can be compared between an arm that skips Christmas Day and an arm
    that also skips 27 to 30 December.
    """
    bank = {int((np.datetime64(d, "D") - EPOCH).astype(np.int64)) for d in bank_holiday_names()}
    keep = []
    for day in holiday:
        date_ = (EPOCH + np.timedelta64(int(day), "D")).astype(date)
        if int(day) in bank or (date_.month == CHRISTMAS_FIRST[0] and date_.day <= 26):
            keep.append(int(day))
    return np.array(keep)
