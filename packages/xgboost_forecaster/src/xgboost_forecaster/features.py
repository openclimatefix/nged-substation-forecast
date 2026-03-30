"""Feature engineering for XGBoost forecasting."""

from datetime import timedelta

import polars as pl

from xgboost_forecaster.scaling import load_scaling_params, uint8_to_physical_unit

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
    params = load_scaling_params()
    descale_cols = ["temperature_2m", "wind_speed_10m", "wind_direction_10m"]

    # Check if columns exist
    existing_cols = [c for c in descale_cols if c in df.collect_schema().names()]
    if not existing_cols:
        return df

    descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(existing_cols)))

    # Apply descaling
    df = df.with_columns([pl.col(c).alias(f"{c}_phys") for c in existing_cols]).with_columns(
        descale_exprs
    )

    # Calculate windchill if both temperature and wind speed are present
    if (
        "temperature_2m_phys" in df.collect_schema().names()
        and "wind_speed_10m_phys" in df.collect_schema().names()
    ):
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
    # Check if required columns exist
    if "temperature_2m" not in weather.collect_schema().names():
        return weather

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

    def _get_lagged_view(df: pl.LazyFrame, offset: timedelta, suffix: str) -> pl.LazyFrame:
        cols = [
            (pl.col("valid_time") + offset).alias("valid_time"),
            pl.col("temperature_2m").alias(f"temperature_2m_{suffix}"),
        ]
        if "downward_short_wave_radiation_flux_surface" in df.collect_schema().names():
            cols.append(
                pl.col("downward_short_wave_radiation_flux_surface").alias(f"sw_radiation_{suffix}")
            )

        if "ensemble_member" in df.collect_schema().names():
            cols.append(pl.col("ensemble_member"))

        lagged = df.select(cols)
        return lagged

    weather_lag_7d = _get_lagged_view(full_weather, timedelta(days=7), "lag_7d")
    weather_lag_14d = _get_lagged_view(full_weather, timedelta(days=14), "lag_14d")

    weather_trend_6h_cols = [
        (pl.col("valid_time") + timedelta(hours=6)).alias("valid_time"),
        pl.col("temperature_2m").alias("temperature_2m_6h_ago"),
    ]
    if "ensemble_member" in full_weather.collect_schema().names():
        weather_trend_6h_cols.append(pl.col("ensemble_member"))

    weather_trend_6h = full_weather.select(weather_trend_6h_cols)

    join_on = ["valid_time"]
    if (
        "ensemble_member" in weather.collect_schema().names()
        and "ensemble_member" in weather_lag_7d.collect_schema().names()
    ):
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
