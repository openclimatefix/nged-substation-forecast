from datetime import datetime

import patito as pt
import polars as pl

from contracts.power_schemas import LIST_OF_TIME_SERIES_TYPES

from .common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype

_FEATURE_DTYPE = pt.Field(dtype=pl.Float32, allow_missing=True)


class AllFeatures(pt.Model):
    """Final joined dataset ready for XGBoost.

    Weather features are kept in their physical units (e.g., degrees Celsius, m/s)
    to ensure precision during interpolation and feature engineering.
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    time_series_id: int = _get_time_series_id_dtype()
    time_series_type: str = pt.Field(dtype=pl.Enum(LIST_OF_TIME_SERIES_TYPES))
    ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)

    power: float = pt.Field(dtype=pl.Float32)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)

    # Weather features
    temperature_2m: float | None = _FEATURE_DTYPE
    dew_point_temperature_2m: float | None = _FEATURE_DTYPE
    wind_speed_10m: float | None = _FEATURE_DTYPE
    wind_direction_10m: float | None = _FEATURE_DTYPE
    wind_speed_100m: float | None = _FEATURE_DTYPE
    wind_direction_100m: float | None = _FEATURE_DTYPE
    pressure_surface: float | None = _FEATURE_DTYPE
    pressure_reduced_to_mean_sea_level: float | None = _FEATURE_DTYPE
    geopotential_height_500hpa: float | None = _FEATURE_DTYPE
    downward_long_wave_radiation_flux_surface: float | None = _FEATURE_DTYPE
    downward_short_wave_radiation_flux_surface: float | None = _FEATURE_DTYPE
    precipitation_surface: float | None = _FEATURE_DTYPE
    categorical_precipitation_type_surface: int | None = _FEATURE_DTYPE

    # Derived weather features
    windchill: float | None = _FEATURE_DTYPE

    # Lagged features
    latest_weekly_lagged_power: float | None = _FEATURE_DTYPE
    latest_weekly_lagged_temperature_2m: float | None = _FEATURE_DTYPE
    latest_weekly_lagged_wind_speed_10m: float | None = _FEATURE_DTYPE
    latest_weekly_lagged_downward_short_wave_radiation_flux_surface: float | None = _FEATURE_DTYPE

    # Temperature trends
    temperature_2m_6h_ago: float | None = _FEATURE_DTYPE
    temperature_2m_trend_6h: float | None = _FEATURE_DTYPE

    # Temporal features. `local` means "in the local timezone", e.g. "Europe/London". We use `local`
    # as the main input feature, because it's the local time that mostly drives demand.
    local_utc_offset: int | None = pt.Field(dtype=pl.Int8, allow_missing=True)
    local_time_of_day_sin: float | None = _FEATURE_DTYPE
    local_time_of_day_cos: float | None = _FEATURE_DTYPE
    local_time_of_year_sin: float | None = _FEATURE_DTYPE
    local_time_of_year_cos: float | None = _FEATURE_DTYPE
    local_day_of_week_sin: float | None = _FEATURE_DTYPE
    local_day_of_week_cos: float | None = _FEATURE_DTYPE
    local_day_of_week: str | None = pt.Field(
        dtype=pl.Enum(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    )


class Metrics(pt.Model):
    """Evaluation metrics for power forecasts."""

    time_series_id: int = _get_time_series_id_dtype()
    power_fcst_model_name: str = pt.Field(dtype=pl.Categorical)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)
    mae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    rmse: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    nmae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
