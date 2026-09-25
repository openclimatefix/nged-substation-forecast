"""Read ECMWF IFS HRES from Open-Meteo's Single Runs archive: which run and lead serves an hour.

The archive holds one 00 UTC run a day with every hourly lead from 0 to 240 hours. Day `N` of a
lead-day comparison reads the 00 UTC run issued `N` days before the target hour's own day, at the
lead the hour falls on, the rule the ENS and GEFS arms of the matched-lead comparison follow.

**A solar hour is labelled by its end, so its own day is the day of the instant an hour earlier.**
The lead of a solar hour at day `N` is `24 * N + 1` to `24 * N + 24` hours, and the lead of a wind
hour, which is read at its label, is `24 * N` to `24 * N + 23` hours. The run ends at lead 240, so
day 10 cannot be served: `last_servable_day` says which day is the last.

**Radiation is a mean over the hour ending at its label, and is occasionally served as -1 W/m2.**
`clip_radiation` clips it at zero.

**Hourly values at longer leads are interpolated.** IFS HRES publishes hourly steps to lead 90
hours, 3-hourly steps to lead 144, and 6-hourly steps to lead 240, and the archive interpolates or
disaggregates the coarser steps to hourly. `native_step_hours` returns the step a lead lies on.
"""

from typing import Final, Literal

import polars as pl

DomainType = Literal["solar", "wind"]

LAST_LEAD_HOURS: Final[int] = 240
"""The last lead of a run in the archive."""

LAST_HOURLY_NATIVE_LEAD_HOURS: Final[int] = 90
"""The last lead IFS HRES publishes every hour."""

LAST_THREE_HOURLY_NATIVE_LEAD_HOURS: Final[int] = 144
"""The last lead IFS HRES publishes every 3 hours; it publishes every 6 hours beyond it."""

HOURS_PER_DAY: Final[int] = 24


def _instant(*, time: pl.Expr, domain: DomainType) -> pl.Expr:
    """Return the instant a target hour describes: an hour before a solar label, else the label."""
    return time - pl.duration(hours=1) if domain == "solar" else time


def served_init_time(*, time: pl.Expr, day: int, domain: DomainType) -> pl.Expr:
    """Return the start of the 00 UTC run that serves a target hour at a lead day.

    Args:
        time: The target hour's label, timezone-aware.
        day: The lead day: the run is issued this many days before the hour's own day, and day 0
            is the run of the hour's own day.
        domain: `solar` or `wind`.

    Returns:
        The run's start, timezone-aware.
    """
    return _instant(time=time, domain=domain).dt.truncate("1d") - pl.duration(days=day)


def served_lead_hours(*, time: pl.Expr, day: int, domain: DomainType) -> pl.Expr:
    """Return the lead in whole hours at which a target hour reads its serving run.

    Args:
        time: The target hour's label, timezone-aware.
        day: The lead day, as in `served_init_time`.
        domain: `solar` or `wind`.

    Returns:
        `24 * day + 1` to `24 * day + 24` for solar, and `24 * day` to `24 * day + 23` for wind.
    """
    init = served_init_time(time=time, day=day, domain=domain)
    return (time - init).dt.total_hours().cast(pl.Int32)


def last_servable_day(*, domain: DomainType) -> int:
    """Return the last lead day whose every target hour falls within the run's 240 hours.

    Args:
        domain: `solar` or `wind`.

    Returns:
        9 for both technologies: day 10 would need leads 241 to 264 (solar) or 240 to 263 (wind).
    """
    last_offset = HOURS_PER_DAY if domain == "solar" else HOURS_PER_DAY - 1
    return (LAST_LEAD_HOURS - last_offset) // HOURS_PER_DAY


def native_step_hours(*, lead_hours: int) -> int:
    """Return the width in hours of the IFS HRES output step a lead lies on.

    Args:
        lead_hours: A lead of the run, 0 to 240.

    Returns:
        1 up to lead 90, 3 up to lead 144, and 6 beyond.

    Raises:
        ValueError: If `lead_hours` is outside the run.
    """
    if not 0 <= lead_hours <= LAST_LEAD_HOURS:
        msg = f"{lead_hours} is not a lead of a run of {LAST_LEAD_HOURS} hours"
        raise ValueError(msg)
    if lead_hours <= LAST_HOURLY_NATIVE_LEAD_HOURS:
        return 1
    return 3 if lead_hours <= LAST_THREE_HOURLY_NATIVE_LEAD_HOURS else 6


def clip_radiation(*, radiation: pl.Expr) -> pl.Expr:
    """Clip a radiation value at zero, which the archive occasionally serves as -1 W/m2."""
    return radiation.clip(lower_bound=0.0)
