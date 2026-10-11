"""Features the production feature engineer does not build, computed from its output.

Every function takes the engineered frame and adds columns that depend only on that row's own
valid time, location, and forecast weather, so no column reads data from after the forecast's
initialisation time. The columns are the ones a quick-win arm tries: solar geometry and the
clear-sky index for PV, wind components and a power-curve proxy for wind, degree-hours for demand,
and bank-holiday and ordinal calendar columns.

The bank holidays are those of England and Wales for 2024 to 2026, typed in because the `holidays`
package is not a dependency. They cover the whole training and validation period of every fold.
"""

from datetime import date
from typing import Final

import numpy as np
import patito as pt
import polars as pl
from contracts.power_schemas import TimeSeriesMetadata
from studies.solar import cos_zenith, zenith

LOCAL_TIMEZONE: Final[str] = "Europe/London"
"""The timezone behaviour follows, as in the production feature engineer."""

BANK_HOLIDAYS: Final[tuple[date, ...]] = (
    date(2024, 1, 1),
    date(2024, 3, 29),
    date(2024, 4, 1),
    date(2024, 5, 6),
    date(2024, 5, 27),
    date(2024, 8, 26),
    date(2024, 12, 25),
    date(2024, 12, 26),
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 5, 5),
    date(2025, 5, 26),
    date(2025, 8, 25),
    date(2025, 12, 25),
    date(2025, 12, 26),
    date(2026, 1, 1),
    date(2026, 4, 3),
    date(2026, 4, 6),
    date(2026, 5, 4),
    date(2026, 5, 25),
    date(2026, 8, 31),
    date(2026, 12, 25),
    date(2026, 12, 28),
)
"""England and Wales bank holidays, 2024 to 2026, with the substitute days for Boxing Day 2026."""

WIND_CUT_IN_M_S: Final[float] = 3.0
WIND_RATED_M_S: Final[float] = 12.5
WIND_CUT_OUT_M_S: Final[float] = 25.0
"""Speeds of a generic farm-level power curve: zero below cut-in and above cut-out, a normalised
cube between cut-in and rated, and flat from rated to cut-out."""

HEATING_BASE_C: Final[float] = 15.5
"""Air temperature below which a degree-hour of heating demand accrues."""

MIN_COS_ZENITH_FOR_INDEX: Final[float] = 0.1
"""Below this sun height the clear-sky index is null, since the ratio blows up there."""

MAX_CLEAR_SKY_INDEX: Final[float] = 1.5
"""Upper clip of the clear-sky index, which cloud enhancement lifts a little above 1."""

EXTRA_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "cos_zenith",
    "clear_sky_index",
    "wind_u_100m",
    "wind_v_100m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_power_proxy",
    "heating_degree_hours",
    "is_bank_holiday",
    "days_to_nearest_bank_holiday",
    "half_hour_of_day",
    "day_of_year",
)
"""The columns `add_extra_features` adds."""


def haurwitz_clear_sky_ghi(*, cos_zenith_values: pl.Expr) -> pl.Expr:
    """Return the Haurwitz clear-sky global horizontal irradiance, in W m⁻², from cos(zenith).

    Args:
        cos_zenith_values: The cosine of the solar zenith angle, zero at and below the horizon.

    Returns:
        1098 cos(z) exp(-0.057 / cos(z)) where the sun is up, and 0 elsewhere.
    """
    return (
        pl.when(cos_zenith_values > 0)
        .then(1098.0 * cos_zenith_values * (-0.057 / cos_zenith_values).exp())
        .otherwise(0.0)
    )


def wind_power_proxy(*, speed: pl.Expr) -> pl.Expr:
    """Return a generic farm-level wind power curve, between 0 and 1, of a wind speed in m s⁻¹.

    Args:
        speed: Wind speed at hub height.

    Returns:
        Zero below cut-in and above cut-out, a normalised cube from cut-in to rated, and 1 from
        rated up to cut-out.
    """
    cube = (speed**3 - WIND_CUT_IN_M_S**3) / (WIND_RATED_M_S**3 - WIND_CUT_IN_M_S**3)
    return (
        pl.when(speed < WIND_CUT_IN_M_S)
        .then(0.0)
        .when(speed < WIND_RATED_M_S)
        .then(cube)
        .when(speed <= WIND_CUT_OUT_M_S)
        .then(1.0)
        .otherwise(0.0)
    )


def _cos_zenith_by_series_and_time(
    *, frame: pl.DataFrame, metadata: pt.DataFrame[TimeSeriesMetadata]
) -> pl.DataFrame:
    """Compute cos(zenith) once per distinct (series, valid time), at the midpoint of the half-hour.

    A valid time is the end of its half-hour, so the sun that matters sits 15 minutes earlier.
    """
    parts = []
    locations = metadata.select("time_series_id", "latitude", "longitude").iter_rows()
    for series_id, latitude, longitude in locations:
        stamps = frame.filter(pl.col("time_series_id") == series_id)["valid_time"].unique().sort()
        if stamps.len() == 0:
            continue
        angle = zenith(
            stamps=stamps.dt.offset_by("-15m"), latitude=float(latitude), longitude=float(longitude)
        )
        parts.append(
            pl.DataFrame(
                {
                    "time_series_id": np.full(stamps.len(), series_id, dtype=np.int32),
                    "valid_time": stamps,
                    "cos_zenith": cos_zenith(zenith_deg=angle),
                }
            )
        )
    return pl.concat(parts)


def add_extra_features(
    *, frame: pl.DataFrame, metadata: pt.DataFrame[TimeSeriesMetadata]
) -> pl.DataFrame:
    """Add the columns in `EXTRA_FEATURE_NAMES` to an engineered frame.

    Args:
        frame: Engineered features with `time_series_id`, `valid_time`, and the weather columns
            `downward_short_wave_radiation_flux_surface`, `temperature_2m`, and the wind speed and
            direction at 10 m and 100 m.
        metadata: The series' `latitude` and `longitude`.

    Returns:
        The frame with the extra columns appended, in `EXTRA_FEATURE_NAMES` order.
    """
    local = pl.col("valid_time").dt.convert_time_zone(LOCAL_TIMEZONE)
    holiday_dates = pl.Series("holiday", list(BANK_HOLIDAYS), dtype=pl.Date)
    holiday_ordinals = holiday_dates.dt.epoch("d").to_list()

    with_sun = frame.join(
        _cos_zenith_by_series_and_time(frame=frame, metadata=metadata),
        on=["time_series_id", "valid_time"],
        how="left",
    )
    clear_sky = haurwitz_clear_sky_ghi(cos_zenith_values=pl.col("cos_zenith"))
    local_ordinal = local.dt.date().dt.epoch("d").cast(pl.Int32)
    nearest_gap = pl.min_horizontal(
        *[(local_ordinal - int(holiday)).abs() for holiday in holiday_ordinals]
    )
    return with_sun.with_columns(
        clear_sky_index=pl.when(pl.col("cos_zenith") >= MIN_COS_ZENITH_FOR_INDEX)
        .then(
            (pl.col("downward_short_wave_radiation_flux_surface") / clear_sky).clip(
                0.0, MAX_CLEAR_SKY_INDEX
            )
        )
        .otherwise(None),
        wind_u_100m=-pl.col("wind_speed_100m") * pl.col("wind_direction_100m").radians().sin(),
        wind_v_100m=-pl.col("wind_speed_100m") * pl.col("wind_direction_100m").radians().cos(),
        wind_u_10m=-pl.col("wind_speed_10m") * pl.col("wind_direction_10m").radians().sin(),
        wind_v_10m=-pl.col("wind_speed_10m") * pl.col("wind_direction_10m").radians().cos(),
        wind_power_proxy=wind_power_proxy(speed=pl.col("wind_speed_100m")),
        heating_degree_hours=(HEATING_BASE_C - pl.col("temperature_2m")).clip(lower_bound=0.0),
        is_bank_holiday=local.dt.date().is_in(holiday_dates.implode()).cast(pl.Int8),
        days_to_nearest_bank_holiday=nearest_gap,
        half_hour_of_day=(local.dt.hour() * 2 + local.dt.minute() // 30).cast(pl.Int16),
        day_of_year=local.dt.ordinal_day().cast(pl.Int16),
    ).select(*frame.columns, *EXTRA_FEATURE_NAMES)
