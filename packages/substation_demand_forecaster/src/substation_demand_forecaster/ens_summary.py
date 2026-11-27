"""The 51-member ENS summary the step-5 neural network reads: member mean and spread per half-hour.

Each member is interpolated to half-hours on its own, wind speed is derived from each member's
interpolated components, and only then are the member mean and the population standard deviation
taken. Averaging the u and v components before taking the speed is wrong by up to 4.2 m/s, so the
order of the steps matters.
"""

from typing import Final

import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

from substation_demand_forecaster.bundle import PUBLICATION_DELAY, SUMMARY_HORIZON

WEATHER_VARIABLES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
"""The seven weather variables, in the neural network's column order."""

SOURCE_COLUMNS: Final[dict[str, str]] = {
    "shortwave": "downward_short_wave_radiation_flux_surface",
    "longwave": "downward_long_wave_radiation_flux_surface",
    "temperature_2m": "temperature_2m",
    "dew_point_temperature_2m": "dew_point_temperature_2m",
    "pressure_surface": "pressure_surface",
    "precipitation": "precipitation_surface",
}
"""The `nwp` column behind each weather variable that the table serves directly."""

WIND_COMPONENTS: Final[tuple[str, str]] = ("wind_u_10m", "wind_v_10m")

SUMMARY_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{statistic}_{variable}" for statistic in ("mean", "std") for variable in WEATHER_VARIABLES
)
"""The 14 summary columns: the seven means, then the seven standard deviations."""


def summarise_runs(nwp: pl.LazyFrame, primaries: pl.DataFrame) -> pl.LazyFrame:
    """Return the ENS member mean and population standard deviation of each run, per Primary.

    Precipitation is returned as stored; the input matrix applies `log1p` to it.

    Args:
        nwp: Rows of the `nwp` table with `h3_index`, `init_time`, `ensemble_member`,
            `valid_time`, and the `SOURCE_COLUMNS` and `WIND_COMPONENTS` values, for one or more
            runs of one weather model.
        primaries: `Bundle.primaries`; each Primary takes the weather of its H3 cell.

    Returns:
        One row per (`nwp_init_time`, `time_series_id`, `valid_time`) for valid times after the
        run's native origin (initialisation time plus `PUBLICATION_DELAY`) up to `SUMMARY_HORIZON`
        after it: `nwp_init_time` and `valid_time` (Datetime, UTC), `time_series_id` (Int32), and
        the `SUMMARY_COLUMNS` (Float32).
    """
    native = (
        nwp.select(
            "h3_index",
            "init_time",
            "ensemble_member",
            "valid_time",
            *SOURCE_COLUMNS.values(),
            *WIND_COMPONENTS,
        )
        .join(
            primaries.lazy().select("time_series_id", "h3_index"),
            on="h3_index",
            how="inner",
        )
        # `_upsample_nwp_to_half_hourly` raises on any column that is neither a weather variable
        # nor one of its resampling keys, so the cell identifier goes and the run's key is renamed.
        .drop("h3_index")
        .rename({"init_time": "nwp_init_time"})
    )
    half_hourly = _upsample_nwp_to_half_hourly(
        native,
        solar_coordinates=primaries.lazy().select("time_series_id", "latitude", "longitude"),
        # One run with an unusual radiation lead leaves its radiation null rather than failing.
        strict_radiation_schedule=False,
    )
    origin = pl.col("nwp_init_time") + PUBLICATION_DELAY
    u, v = (pl.col(component) for component in WIND_COMPONENTS)
    return (
        half_hourly.rename({source: name for name, source in SOURCE_COLUMNS.items()})
        .with_columns(wind_speed_10m=(u**2 + v**2).sqrt())
        .filter(pl.col("valid_time") > origin, pl.col("valid_time") <= origin + SUMMARY_HORIZON)
        .group_by("nwp_init_time", "time_series_id", "valid_time")
        .agg(
            *(pl.col(variable).mean().alias(f"mean_{variable}") for variable in WEATHER_VARIABLES),
            *(
                pl.col(variable).std(ddof=0).alias(f"std_{variable}")
                for variable in WEATHER_VARIABLES
            ),
        )
        .cast(dict.fromkeys(SUMMARY_COLUMNS, pl.Float32))
        .sort("nwp_init_time", "time_series_id", "valid_time")
    )
