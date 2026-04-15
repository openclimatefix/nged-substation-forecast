from datetime import datetime

import patito as pt
import polars as pl
from pydantic import BaseModel

from .common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype


class InferenceParams(BaseModel):
    """Parameters for ML model inference."""

    # The time that we create our power forecast. This might be called `t0` in some other OCF
    # projects. When running backtests, we cannot use any NWPs available after `forecast_time`
    # (i.e. init_time + delay > forecast_time).
    power_fcst_init_time: datetime

    power_fcst_model_name: str


class XGBoostInputFeatures(pt.Model):
    """Final joined dataset ready for XGBoost.

    Weather features are kept in their physical units (e.g., degrees Celsius, m/s)
    to ensure precision during interpolation and feature engineering.
    """

    valid_time: datetime = pt.Field(dtype=UTC_DATETIME_DTYPE)
    time_series_id: int = _get_time_series_id_dtype()
    time_series_type: str = pt.Field(dtype=pl.Categorical)
    ensemble_member: int | None = pt.Field(dtype=pl.UInt8, allow_missing=True)
    power: float = pt.Field(dtype=pl.Float32)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)

    # Power lags
    latest_available_weekly_power_lag: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather features
    temperature_2m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    dew_point_temperature_2m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    # PHYSICAL WIND FEATURES:
    # These are calculated from interpolated U/V components in the forecasting
    # pipeline, ensuring physically realistic wind speed and direction.
    wind_speed_10m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_direction_10m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_speed_100m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    wind_direction_100m: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    pressure_surface: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    pressure_reduced_to_mean_sea_level: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    geopotential_height_500hpa: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    downward_long_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    downward_short_wave_radiation_flux_surface: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    precipitation_surface: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    categorical_precipitation_type_surface: int | None = pt.Field(
        dtype=pl.UInt8, allow_missing=True
    )

    # Physical features
    windchill: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Weather lags/trends
    temperature_2m_lag_7d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    downward_short_wave_radiation_flux_surface_lag_7d: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    temperature_2m_lag_14d: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    downward_short_wave_radiation_flux_surface_lag_14d: float | None = pt.Field(
        dtype=pl.Float32, allow_missing=True
    )
    temperature_2m_6h_ago: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    temp_trend_6h: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)

    # Temporal features
    hour_sin: float = pt.Field(dtype=pl.Float32)
    hour_cos: float = pt.Field(dtype=pl.Float32)
    day_of_year_sin: float = pt.Field(dtype=pl.Float32)
    day_of_year_cos: float = pt.Field(dtype=pl.Float32)
    day_of_week: int = pt.Field(dtype=pl.Int8)


class Metrics(pt.Model):
    """Evaluation metrics for power forecasts."""

    time_series_id: int = _get_time_series_id_dtype()
    power_fcst_model_name: str = pt.Field(dtype=pl.Categorical)
    lead_time_hours: float = pt.Field(dtype=pl.Float32)
    mae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    rmse: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
    nmae: float | None = pt.Field(dtype=pl.Float32, allow_missing=True)
