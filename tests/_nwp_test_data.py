"""Shared NWP test data for the root integration tests.

The several ``live_forecasts`` / CV / metrics integration tests each build a synthetic ``Nwp``
frame; the physical-unit constants below were byte-identical across all of them, so they live
here. The per-test *writers* (which differ in init-times, cells, and ensemble members) stay local
to each test file. Importable by bare name via the ``pythonpath = ["tests"]`` pytest setting.
"""

from typing import Final

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
