"""Feature engineering for XGBoost forecasting."""

import logging
from datetime import timedelta

import polars as pl

from ml_core.scaling import uint8_to_physical_unit
from xgboost_forecaster.scaling import load_scaling_params

log = logging.getLogger(__name__)

# TODO: All the functions in this file should be moved into a common package, so other ML models can
# also use these features.

# TODO: All the functions in this file should use `pt.DataFrame[...]` type hints for both the input
# and output.


def add_physical_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add wind speed, direction, and windchill.

    Args:
        df: LazyFrame with NWP variables.

    Returns:
        LazyFrame with added physical features.
    """
    schema_names = df.collect_schema().names()
    params = load_scaling_params()
    descale_cols = ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]

    # Check if columns exist
    existing_cols = [c for c in descale_cols if c in schema_names]
    if not existing_cols:
        return df

    descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(existing_cols)))

    # Apply descaling
    df = df.with_columns([pl.col(c).alias(f"{c}_phys") for c in existing_cols]).with_columns(
        descale_exprs
    )

    # Calculate windchill if both temperature and wind speed are present
    if "temperature_2m" in existing_cols and "wind_speed_10m" in existing_cols:
        # V is wind speed in km/h
        v_kmh = pl.col("wind_speed_10m_phys") * 3.6
        temp = pl.col("temperature_2m_phys")

        df = df.with_columns(
            windchill=(
                13.12 + 0.6215 * temp - 11.37 * (v_kmh**0.16) + 0.3965 * temp * (v_kmh**0.16)
            ).cast(pl.Float32)
        )

    return df


def add_weather_features(
    weather: pl.LazyFrame, history: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    """Add lags and trends to weather data.

    Args:
        weather: Current weather forecast.
        history: Historical weather data (optional, used for lags).

    Returns:
        LazyFrame with added weather features.
    """
    schema_names = weather.collect_schema().names()
    if "temperature_2m" not in schema_names:
        return weather

    weather = add_physical_features(weather).sort("init_time")
    if history is not None:
        history = add_physical_features(history)
        full_weather = pl.concat([history, weather], how="diagonal").sort("init_time")
    else:
        full_weather = weather

    def _add_lag_asof(
        df: pl.LazyFrame, source_df: pl.LazyFrame, offset: timedelta, suffix: str
    ) -> pl.LazyFrame:
        by_cols = ["target_valid_time"]
        if "h3_index" in schema_names:
            by_cols.append("h3_index")
        if "ensemble_member" in schema_names:
            by_cols.append("ensemble_member")

        source_cols = ["valid_time", "init_time", "temperature_2m"]
        if "h3_index" in schema_names:
            source_cols.append("h3_index")
        if "ensemble_member" in schema_names:
            source_cols.append("ensemble_member")
        if "downward_short_wave_radiation_flux_surface" in schema_names:
            source_cols.append("downward_short_wave_radiation_flux_surface")

        right = source_df.select(source_cols).rename(
            {"valid_time": "target_valid_time", "temperature_2m": f"temperature_2m_{suffix}"}
        )
        if "downward_short_wave_radiation_flux_surface" in schema_names:
            right = right.rename(
                {"downward_short_wave_radiation_flux_surface": f"sw_radiation_{suffix}"}
            )

        left = df.with_columns(target_valid_time=pl.col("valid_time") - offset)

        return left.join_asof(right, on="init_time", by=by_cols, strategy="backward").drop(
            "target_valid_time"
        )

    weather = _add_lag_asof(weather, full_weather, timedelta(days=7), "lag_7d")
    weather = _add_lag_asof(weather, full_weather, timedelta(days=14), "lag_14d")
    weather = _add_lag_asof(weather, full_weather, timedelta(hours=6), "6h_ago")

    return weather.with_columns(
        temp_trend_6h=(
            pl.col("temperature_2m").cast(pl.Float32)
            - pl.col("temperature_2m_6h_ago").cast(pl.Float32)
        ).cast(pl.Float32)
    )
