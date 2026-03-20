"""Feature engineering for XGBoost forecasting."""

from datetime import timedelta

import numpy as np
import polars as pl

from xgboost_forecaster.scaling import load_scaling_params, uint8_to_physical_unit

# TODO: All the functions in this file should be moved into a common package, so other ML models can
# also use these features.

# TODO: All the functions in this file should use `pt.DataFrame[...]` type hints for both the input
# and output.


def add_temporal_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add cyclical and standard temporal features.

    Args:
        df: DataFrame with a 'valid_time' column.

    Returns:
        DataFrame with added temporal features.
    """
    return df.with_columns(
        # Cyclical Hour (24h)
        hour_sin=(pl.col("valid_time").dt.hour() * 2 * np.pi / 24).sin().cast(pl.Float32),
        hour_cos=(pl.col("valid_time").dt.hour() * 2 * np.pi / 24).cos().cast(pl.Float32),
        # Cyclical Day of Year (365.25d)
        day_of_year_sin=(pl.col("valid_time").dt.ordinal_day() * 2 * np.pi / 365.25)
        .sin()
        .cast(pl.Float32),
        day_of_year_cos=(pl.col("valid_time").dt.ordinal_day() * 2 * np.pi / 365.25)
        .cos()
        .cast(pl.Float32),
        # Day of week (0-6)
        day_of_week=pl.col("valid_time").dt.weekday().cast(pl.Int8),
        # TODO: Try adding `day_of_week_sin` and `day_of_week_cos`.
    )


def add_physical_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add wind speed, direction, and windchill.

    Args:
        df: DataFrame with NWP variables.

    Returns:
        DataFrame with added physical features.
    """
    params = load_scaling_params()
    descale_cols = ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]
    descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(descale_cols)))
    phys_df = df.select(["valid_time"] + descale_cols).with_columns(descale_exprs)

    # Windchill formula: 13.12 + 0.6215*T - 11.37*V^0.16 + 0.3965*T*V^0.16
    # V is wind speed in km/h
    v_kmh = pl.col("wind_speed_10m") * 3.6
    temp = pl.col("temperature_2m")

    phys_df = phys_df.with_columns(
        windchill=(
            13.12 + 0.6215 * temp - 11.37 * (v_kmh**0.16) + 0.3965 * temp * (v_kmh**0.16)
        ).cast(pl.Float32)
    )

    # Join back to original df
    return df.with_columns(
        [
            phys_df["wind_speed_10m"].alias("wind_speed_10m_phys").cast(pl.Float32),
            phys_df["wind_direction_10m"].alias("wind_direction_10m_phys").cast(pl.Float32),
            phys_df["windchill"].cast(pl.Float32),
        ]
    )


def add_weather_features(
    weather: pl.DataFrame, history: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Add lags and trends to weather data.

    Args:
        weather: Current weather forecast.
        history: Historical weather data (optional, used for lags).

    Returns:
        DataFrame with added weather features.
    """
    if history is not None:
        full_weather = (
            pl.concat(
                [
                    history.select(pl.all().exclude("ensemble_member")),
                    weather.select(pl.all().exclude("ensemble_member")),
                ],
                how="diagonal",
            )
            .unique("valid_time")
            .sort("valid_time")
        )
    else:
        full_weather = weather

    weather = add_physical_features(weather)
    if history is not None:
        full_weather = add_physical_features(full_weather)

    def _get_lagged_view(df: pl.DataFrame, offset: timedelta, suffix: str) -> pl.DataFrame:
        lagged = df.select(
            [
                (pl.col("valid_time") + offset).alias("valid_time"),
                pl.col("temperature_2m").alias(f"temperature_2m_{suffix}"),
                pl.col("downward_short_wave_radiation_flux_surface").alias(
                    f"sw_radiation_{suffix}"
                ),
            ]
            + ([pl.col("ensemble_member")] if "ensemble_member" in df.columns else [])
        )
        return lagged

    weather_lag_7d = _get_lagged_view(full_weather, timedelta(days=7), "lag_7d")
    weather_lag_14d = _get_lagged_view(full_weather, timedelta(days=14), "lag_14d")
    weather_trend_6h = full_weather.select(
        [
            (pl.col("valid_time") + timedelta(hours=6)).alias("valid_time"),
            pl.col("temperature_2m").alias("temperature_2m_6h_ago"),
        ]
        + ([pl.col("ensemble_member")] if "ensemble_member" in full_weather.columns else [])
    )

    join_on = ["valid_time"]
    if "ensemble_member" in weather.columns and "ensemble_member" in weather_lag_7d.columns:
        join_on.append("ensemble_member")

    weather = (
        weather.join(weather_lag_7d, on=join_on, how="left")
        .join(weather_lag_14d, on=join_on, how="left")
        .join(weather_trend_6h, on=join_on, how="left")
    )

    return weather.with_columns(
        temp_trend_6h=(
            pl.col("temperature_2m").cast(pl.Float32)
            - pl.col("temperature_2m_6h_ago").cast(pl.Float32)
        ).cast(pl.Float32)
    )
