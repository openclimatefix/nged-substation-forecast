"""Calendar and weather input columns of the differentiable-physics (DP) series, in local time.

Every function takes UTC timestamps that mark the **end** of a half-hour, as NGED's cleaned power
does, and works from the half-hour's **start** in Europe/London time, so a half-hour that ends at
00:00 local belongs to the day it began in. The columns are the inputs of the convex model in
`studies.dp_gam`: the day type and half-hour of day that index the daily profile, the four hour
bands that the calendar, heating, and drift terms are separate in, named holiday periods, a tent
around Christmas, day-of-year harmonics, trailing means of temperature, and solar elevation.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import holidays
import numpy as np
import polars as pl

LOCAL_TIME_ZONE: Final[str] = "Europe/London"
"""The time zone that day types, half-hours of day, and bands are read in."""

HALF_HOURS_PER_DAY: Final[int] = 48
"""The number of half-hour slots in a local day (daylight-saving days use 46 or 50 of them)."""

DAY_TYPES: Final[tuple[str, ...]] = ("weekday", "saturday", "sunday", "bank_holiday")
"""The day types of the daily profile; the index is the position in this tuple."""

BAND_NAMES: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening")
"""The four hour bands that cover the whole day, by the local hour a half-hour starts in."""

BAND_FIRST_HOURS: Final[tuple[int, ...]] = (0, 6, 11, 16)
"""The first local hour of each band: night 00:00 to 05:59, morning 06:00 to 10:59, midday 11:00 to
15:59, and evening 16:00 to 23:59."""

HOLIDAY_PERIODS: Final[tuple[str, ...]] = (
    "New Year's Day",
    "Good Friday",
    "Easter Monday",
    "May Day",
    "Spring Bank Holiday",
    "Late Summer Bank Holiday",
    "Christmas Day",
    "Boxing Day",
    "one-off bank holiday",
    "Easter Saturday",
    "Easter Sunday",
    "bridge day",
    "Christmas run",
)
"""The named holiday periods that carry a coefficient per hour band.

The first eight are the recurring English bank holidays. A bank holiday that happens once in the
data (the Platinum Jubilee, the Queen's funeral, the Coronation) is `one-off bank holiday`.
`Christmas run` is 24 December and 27 to 31 December. A name earlier in the tuple takes precedence
on a day that fits two, so a bank holiday on 27 December is a bank holiday and not part of the run.
"""

CHRISTMAS_WINDOW_DAYS: Final[int] = 14
"""The tent and signed tent around Christmas Day are zero further than this many days away."""

HEATING_THRESHOLD_C: Final[float] = 15.5
"""The temperature below which demand rises; the centre of the series brief's prior."""

SMOOTHING_HOURS: Final[tuple[int, ...]] = (6, 24, 72, 168)
"""The trailing windows, in hours, over which temperature is averaged to give thermal inertia."""

LIGHTING_KNOTS_DEG: Final[tuple[float, ...]] = (-6.0, 0.0, 6.0, 12.0)
"""The solar elevations at which the lighting ramps end; each ramp is 6 degrees wide."""

RAMP_WIDTH_DEG: Final[float] = 6.0
"""The width of one lighting ramp in solar elevation."""

HOLIDAY_YEARS: Final[range] = range(2018, 2029)
"""The years for which bank holidays are looked up."""

_ONE_OFF_NAMES: Final[frozenset[str]] = frozenset(
    {
        "Platinum Jubilee of Elizabeth II",
        "State Funeral of Queen Elizabeth II",
        "Coronation of Charles III",
        "Wedding of Prince William and Catherine Middleton",
        "Diamond Jubilee of Elizabeth II",
        "Silver Jubilee of Elizabeth II",
    }
)

_OBSERVED_SUFFIX: Final[str] = " (observed)"

SATURDAY: Final[int] = 5
"""`datetime.date.weekday()` of a Saturday; Sunday is one more."""
SUNDAY: Final[int] = 6
FIRST_SEASON_MONTH: Final[int] = 7
"""A season-year runs from July to June."""
FIRST_RECURRING_COUNT: Final[int] = 8
"""The first entries of `HOLIDAY_PERIODS` are the recurring bank holidays."""
CHRISTMAS_RUN_DAYS: Final[tuple[int, ...]] = (24, 27, 28, 29, 30, 31)
"""The days of December that belong to the Christmas run."""


@dataclass(frozen=True)
class CalendarColumns:
    """The calendar inputs of every half-hour in a series of timestamps.

    Attributes:
        half_hour: The local half-hour of day the half-hour starts in, 0 to 47, int16.
        day_type: The index into `DAY_TYPES`, int8.
        band: The index into `BAND_NAMES`, int8.
        holiday_period: The index into `HOLIDAY_PERIODS`, or -1 on an ordinary day, int8.
        tent: The tent around Christmas Day, 1 on the day and 0 beyond the window, float64.
        signed_tent: The tent times the signed distance to Christmas Day over the window, float64.
        harmonics: Four columns, the sine and cosine of one and of two times the angle of the day
            of the year, float64.
        local_date: The local date the half-hour starts on, as `datetime64[D]`.
        season_year: The year in which the July-to-June season-year begins, int16.
        day_of_year: The local day of the year, int16.
        quarter: The calendar quarter of the year, 0 to 3, int8.
    """

    half_hour: np.ndarray
    day_type: np.ndarray
    band: np.ndarray
    holiday_period: np.ndarray
    tent: np.ndarray
    signed_tent: np.ndarray
    harmonics: np.ndarray
    local_date: np.ndarray
    season_year: np.ndarray
    day_of_year: np.ndarray
    quarter: np.ndarray


def bank_holiday_names() -> dict[date, str]:
    """Return the English bank holidays, each date with its period name.

    Returns:
        A map from date to a name in `HOLIDAY_PERIODS` (`one-off bank holiday` when the name is
        not one of the eight recurring holidays).
    """
    england = holidays.UnitedKingdom(subdiv="ENG", years=HOLIDAY_YEARS)
    names: dict[date, str] = {}
    for day, raw_name in england.items():
        name = raw_name.removesuffix(_OBSERVED_SUFFIX)
        names[day] = (
            name if name in HOLIDAY_PERIODS[:FIRST_RECURRING_COUNT] else "one-off bank holiday"
        )
    return names


def _easter_weekend(*, good_fridays: list[date]) -> dict[date, str]:
    """Return Easter Saturday and Easter Sunday for each Good Friday."""
    weekend: dict[date, str] = {}
    for friday in good_fridays:
        weekend[friday + timedelta(days=1)] = "Easter Saturday"
        weekend[friday + timedelta(days=2)] = "Easter Sunday"
    return weekend


def _period_of_dates(*, days: list[date]) -> np.ndarray:
    """Return the holiday-period index of each date, -1 on an ordinary day."""
    bank = bank_holiday_names()
    good_fridays = [d for d, name in bank.items() if name == "Good Friday"]
    easter = _easter_weekend(good_fridays=good_fridays)
    index = {name: i for i, name in enumerate(HOLIDAY_PERIODS)}
    out = np.full(len(days), -1, dtype=np.int8)
    for position, day in enumerate(days):
        if day in bank:
            out[position] = index[bank[day]]
        elif day in easter:
            out[position] = index[easter[day]]
        elif _is_bridge_day(day=day, bank=bank):
            out[position] = index["bridge day"]
        elif day.month == 12 and day.day in CHRISTMAS_RUN_DAYS:
            out[position] = index["Christmas run"]
    return out


def _is_bridge_day(*, day: date, bank: dict[date, str]) -> bool:
    """Return True for a working day between a bank holiday and a weekend."""
    if day.weekday() >= SATURDAY or day in bank:
        return False
    before = day - timedelta(days=1)
    after = day + timedelta(days=1)
    return (before in bank and after.weekday() >= SATURDAY) or (
        after in bank and before.weekday() >= SATURDAY
    )


def calendar_columns(*, end_times: pl.Series) -> CalendarColumns:
    """Return the calendar inputs of the half-hours that end at `end_times`.

    Args:
        end_times: UTC datetimes marking the end of each half-hour.

    Returns:
        The columns, one entry per timestamp, in the order of `end_times`.
    """
    local = end_times.dt.offset_by("-30m").dt.convert_time_zone(LOCAL_TIME_ZONE)
    hour = local.dt.hour().to_numpy().astype(np.int16)
    minute = local.dt.minute().to_numpy().astype(np.int16)
    half_hour = hour * 2 + minute // 30
    band = (
        (hour >= BAND_FIRST_HOURS[1]).astype(np.int8)
        + (hour >= BAND_FIRST_HOURS[2]).astype(np.int8)
        + (hour >= BAND_FIRST_HOURS[3]).astype(np.int8)
    )
    dates = local.dt.date()
    unique_dates = dates.unique().sort()
    unique_list = unique_dates.to_list()
    period_of_unique = _period_of_dates(days=unique_list)
    bank = bank_holiday_names()
    is_bank_unique = np.array([d in bank for d in unique_list])
    weekday_unique = np.array([d.weekday() for d in unique_list])
    lookup_frame = pl.DataFrame(
        {
            "local_date": unique_dates,
            "period": period_of_unique,
            "is_bank": is_bank_unique,
            "weekday": weekday_unique,
        }
    )
    joined = pl.DataFrame({"local_date": dates}).join(
        lookup_frame, on="local_date", how="left", validate="m:1", maintain_order="left"
    )
    weekday = joined["weekday"].to_numpy()
    day_type = np.where(
        joined["is_bank"].to_numpy(),
        3,
        np.where(weekday == SATURDAY, 1, np.where(weekday == SUNDAY, 2, 0)),
    ).astype(np.int8)
    day_of_year = dates.dt.ordinal_day().to_numpy().astype(np.int16)
    angle = 2.0 * np.pi * day_of_year / 365.25
    harmonics = np.stack(
        [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)], axis=1
    )
    signed_days = _days_to_christmas(dates=dates)
    inside = np.abs(signed_days) <= CHRISTMAS_WINDOW_DAYS
    tent = np.where(inside, 1.0 - np.abs(signed_days) / (CHRISTMAS_WINDOW_DAYS + 1), 0.0)
    signed_tent = np.where(inside, signed_days / (CHRISTMAS_WINDOW_DAYS + 1), 0.0) * tent
    month = dates.dt.month().to_numpy()
    season_year = (dates.dt.year().to_numpy() - (month < FIRST_SEASON_MONTH)).astype(np.int16)
    return CalendarColumns(
        half_hour=half_hour,
        day_type=day_type,
        band=band,
        holiday_period=joined["period"].to_numpy().astype(np.int8),
        tent=tent,
        signed_tent=signed_tent,
        harmonics=harmonics,
        local_date=dates.to_numpy().astype("datetime64[D]"),
        season_year=season_year,
        day_of_year=day_of_year,
        quarter=((month - 1) // 3).astype(np.int8),
    )


def _days_to_christmas(*, dates: pl.Series) -> np.ndarray:
    """Return the signed number of days from each date to the nearest 25 December."""
    year = dates.dt.year().to_numpy()
    days = dates.to_numpy().astype("datetime64[D]")
    candidates = []
    for offset in (-1, 0, 1):
        christmas = np.array([f"{y + offset}-12-25" for y in year], dtype="datetime64[D]")
        candidates.append((days - christmas).astype(np.int64))
    stacked = np.stack(candidates, axis=1)
    nearest = np.argmin(np.abs(stacked), axis=1)
    return stacked[np.arange(len(year)), nearest].astype(np.float64)


def trailing_mean_temperatures(*, series: pl.DataFrame) -> pl.DataFrame:
    """Return the trailing means of temperature over each of `SMOOTHING_HOURS`.

    A trailing mean at stamp `T` averages the readings stamped `T - k hours` (exclusive) to `T`,
    so it uses no reading later than `T`. The first `k` hours of the series are null.

    Args:
        series: Columns `valid_time` (UTC, evenly spaced with a step that divides an hour, no
            gaps) and `temperature_c`.

    Returns:
        Columns `valid_time` and `temperature_c` (the reading itself), plus `temperature_c_k6`,
        `temperature_c_k24`, and so on, sorted by `valid_time`.

    Raises:
        ValueError: If the stamps are not evenly spaced by a step that divides an hour.
    """
    ordered = series.sort("valid_time").select("valid_time", "temperature_c")
    steps = ordered["valid_time"].diff().drop_nulls().unique()
    step_minutes = steps[0].total_seconds() / 60.0 if len(steps) == 1 else 0.0
    if step_minutes <= 0.0 or 60.0 % step_minutes != 0.0:
        msg = "the temperature series must have one evenly spaced step that divides an hour"
        raise ValueError(msg)
    per_hour = round(60.0 / step_minutes)
    return ordered.with_columns(
        [
            pl.col("temperature_c")
            .rolling_mean(window_size=k * per_hour, min_samples=k * per_hour)
            .alias(f"temperature_c_k{k}")
            for k in SMOOTHING_HOURS
        ]
    )


def lighting_ramps(*, elevation_deg: np.ndarray) -> np.ndarray:
    """Return four saturating ramps of how dark it is, from the sun's elevation.

    Ramp `j` is 0 when the sun is higher than the knot `LIGHTING_KNOTS_DEG[j]`, rises linearly
    to 1 over `RAMP_WIDTH_DEG` degrees below the knot, and stays at 1 after that. Every ramp is 0
    at midday, so the ramps do not overlap with the solar term.

    Args:
        elevation_deg: The sun's elevation above the horizon, in degrees.

    Returns:
        An array of shape (len(elevation_deg), 4).
    """
    knots = np.array(LIGHTING_KNOTS_DEG)
    return np.clip((knots[None, :] - elevation_deg[:, None]) / RAMP_WIDTH_DEG, 0.0, 1.0)
