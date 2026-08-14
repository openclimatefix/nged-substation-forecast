"""Shared NWP test data for the root integration tests.

The several ``live_forecasts`` / CV / metrics integration tests each build a synthetic ``Nwp``
frame; the physical-unit constants below were byte-identical across all of them, so they live
here, as does ``half_hours``, which every one of them needs to place valid times correctly
relative to the publication delay. The per-test *writers* (which differ in init-times, cells, and
ensemble members) stay local to each test file. Importable by bare name via the
``pythonpath = ["tests"]`` pytest setting.
"""

from datetime import datetime, timedelta
from typing import Final

import polars as pl
from weather_utils import NWP_PUBLICATION_DELAY_HOURS

NWP_CONTINUOUS_COL_VALUES: Final[dict[str, float]] = {
    "temperature_2m": 15.0,
    "dew_point_temperature_2m": 10.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 180.0,
    "wind_speed_100m": 8.0,
    "wind_direction_100m": 180.0,
    "pressure_surface": 101_000.0,
    "pressure_reduced_to_mean_sea_level": 101_500.0,
    "geopotential_height_500hpa": 5_500.0,
    "downward_long_wave_radiation_flux_surface": 300.0,
    "downward_short_wave_radiation_flux_surface": 200.0,
    "precipitation_surface": 0.001,
}
"""Physically plausible Float32 constants, one per continuous ``Nwp`` variable."""


def half_hours(day: datetime) -> pl.Series:
    """Half-hourly valid times inside a 00Z run's forecast window, after the publication delay.

    The start is derived from ``NWP_PUBLICATION_DELAY_HOURS`` rather than hard-coded: bulk mode
    stamps ``power_fcst_init_time = init_time + NWP_PUBLICATION_DELAY_HOURS`` and drops every
    earlier valid time as a hindcast, so a fixed window silently empties the fixture — and the
    asset then fails with "no usable rows" — if the delay ever grows.
    """
    first = day.replace(hour=0) + timedelta(hours=NWP_PUBLICATION_DELAY_HOURS + 1)
    return pl.datetime_range(
        first, first + timedelta(hours=2), interval="30m", time_zone="UTC", eager=True
    )
