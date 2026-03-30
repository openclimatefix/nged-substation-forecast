"""Feature engineering for XGBoost forecasting."""

import logging
from datetime import timedelta

import polars as pl

from contracts.data_schemas import NwpColumns
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
    descale_cols = [
        NwpColumns.TEMPERATURE_2M,
        NwpColumns.WIND_SPEED_10M,
        NwpColumns.WIND_DIRECTION_10M,
    ]

    # Check if columns exist
    existing_cols = [c for c in descale_cols if c in schema_names]
    if not existing_cols:
        return df

    descale_exprs = uint8_to_physical_unit(params.filter(pl.col("col_name").is_in(existing_cols)))

    # Apply descaling directly to the original columns
    df = df.with_columns(descale_exprs)

    # Calculate windchill if both temperature and wind speed are present
    if NwpColumns.TEMPERATURE_2M in existing_cols and NwpColumns.WIND_SPEED_10M in existing_cols:
        # V is wind speed in km/h
        v_kmh = pl.col(NwpColumns.WIND_SPEED_10M) * 3.6
        temp = pl.col(NwpColumns.TEMPERATURE_2M)

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
    if NwpColumns.TEMPERATURE_2M not in schema_names:
        return weather

    weather = add_physical_features(weather).sort(NwpColumns.INIT_TIME)
    if history is not None:
        history = add_physical_features(history)
        full_weather = pl.concat([history, weather], how="diagonal").sort(NwpColumns.INIT_TIME)
    else:
        full_weather = weather

    # Check if we have enough history for the 14-day lag
    sorted_weather = full_weather.sort(NwpColumns.VALID_TIME)
    min_time_df = sorted_weather.head(1).select(NwpColumns.VALID_TIME).collect()
    max_time_df = sorted_weather.tail(1).select(NwpColumns.VALID_TIME).collect()

    if (
        isinstance(min_time_df, pl.DataFrame)
        and isinstance(max_time_df, pl.DataFrame)
        and min_time_df.height > 0
        and max_time_df.height > 0
    ):
        min_time = min_time_df.item(0, NwpColumns.VALID_TIME)
        max_time = max_time_df.item(0, NwpColumns.VALID_TIME)

        if min_time is not None and max_time is not None:
            if max_time - min_time < timedelta(days=14):
                log.warning(
                    "Provided weather data does not cover the required 14-day lag range. "
                    "Weather lag features will be null for the first 14 days of the forecast."
                )

    def _add_lag_asof(
        df: pl.LazyFrame, source_df: pl.LazyFrame, offset: timedelta, suffix: str
    ) -> pl.LazyFrame:
        by_cols = ["target_valid_time"]
        if NwpColumns.H3_INDEX in schema_names:
            by_cols.append(NwpColumns.H3_INDEX)
        if NwpColumns.ENSEMBLE_MEMBER in schema_names:
            by_cols.append(NwpColumns.ENSEMBLE_MEMBER)

        source_cols = [NwpColumns.VALID_TIME, NwpColumns.INIT_TIME, NwpColumns.TEMPERATURE_2M]
        if NwpColumns.H3_INDEX in schema_names:
            source_cols.append(NwpColumns.H3_INDEX)
        if NwpColumns.ENSEMBLE_MEMBER in schema_names:
            source_cols.append(NwpColumns.ENSEMBLE_MEMBER)
        if NwpColumns.SW_RADIATION in schema_names:
            source_cols.append(NwpColumns.SW_RADIATION)

        right = source_df.select(source_cols).rename(
            {
                NwpColumns.VALID_TIME: "target_valid_time",
                NwpColumns.TEMPERATURE_2M: f"{NwpColumns.TEMPERATURE_2M}_{suffix}",
            }
        )
        if NwpColumns.SW_RADIATION in schema_names:
            right = right.rename({NwpColumns.SW_RADIATION: f"sw_radiation_{suffix}"})

        left = df.with_columns(target_valid_time=pl.col(NwpColumns.VALID_TIME) - offset)

        return left.join_asof(right, on=NwpColumns.INIT_TIME, by=by_cols, strategy="backward").drop(
            "target_valid_time"
        )

    weather = _add_lag_asof(weather, full_weather, timedelta(days=7), "lag_7d")
    weather = _add_lag_asof(weather, full_weather, timedelta(days=14), "lag_14d")
    weather = _add_lag_asof(weather, full_weather, timedelta(hours=6), "6h_ago")

    return weather.with_columns(
        temp_trend_6h=(
            pl.col(NwpColumns.TEMPERATURE_2M).cast(pl.Float32)
            - pl.col(f"{NwpColumns.TEMPERATURE_2M}_6h_ago").cast(pl.Float32)
        ).cast(pl.Float32)
    )
