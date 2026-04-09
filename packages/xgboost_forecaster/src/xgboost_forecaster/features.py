"""Feature engineering for XGBoost forecasting."""

import logging
from datetime import timedelta

import polars as pl

from contracts.data_schemas import NwpColumns

log = logging.getLogger(__name__)


def add_autoregressive_lags(
    df: pl.LazyFrame, flows_30m: pl.LazyFrame, telemetry_delay_hours: int = 24
) -> pl.LazyFrame:
    """Add autoregressive lags to the feature matrix.

    This function calculates the required lag dynamically to strictly prevent
    lookahead bias, ensuring that the model only uses power flow data that
    would have been available at the time the forecast was made.

    Args:
        df: The input LazyFrame (schema: XGBoostInputFeatures).
        flows_30m: Historical power flows downsampled to 30m.
        telemetry_delay_hours: Delay in hours for telemetry availability.

    Returns:
        LazyFrame with added lag features (schema: XGBoostInputFeatures).
    """
    # 1. Calculate the required lag dynamically to strictly prevent lookahead bias
    df = (
        df.with_columns(
            lead_time_days=(pl.col("valid_time") - pl.col("init_time")).dt.total_seconds()
            / (24 * 3600)
        )
        .with_columns(
            lag_days=pl.max_horizontal(
                pl.lit(1),
                ((pl.col("lead_time_days") + telemetry_delay_hours / 24.0) / 7.0)
                .ceil()
                .cast(pl.Int32),
            )
            * 7
        )
        .with_columns(
            target_lag_time=pl.col("valid_time") - pl.duration(days=1) * pl.col("lag_days")
        )
    )

    # 2. Join flows_30m on ["time_series_id", "target_lag_time"] to extract the exact
    # latest_available_weekly_power_lag without needing pre-calculated lag_7d or lag_14d columns.
    lag_df = flows_30m.select(
        pl.col("time_series_id"),
        pl.col("end_time").alias("target_lag_time"),
        pl.col("value").alias("latest_available_weekly_power_lag"),
    )

    df = df.join(lag_df, on=["time_series_id", "target_lag_time"], how="left")

    return df


def add_weather_features(
    weather: pl.LazyFrame, history: pl.LazyFrame | None = None
) -> pl.LazyFrame:
    """Add lags and trends to weather data.

    Args:
        weather: Current weather forecast (schema: ProcessedNwp).
        history: Historical weather data (optional, used for lags).

    Returns:
        LazyFrame with added weather features (schema: ProcessedNwp).
    """
    schema_names = weather.collect_schema().names()
    if NwpColumns.TEMPERATURE_2M not in schema_names:
        raise ValueError(
            f"Required weather column '{NwpColumns.TEMPERATURE_2M}' is missing from the input "
            f"LazyFrame. Available columns: {schema_names}"
        )

    # Add windchill if both temperature and wind speed are present
    if NwpColumns.TEMPERATURE_2M in schema_names and NwpColumns.WIND_SPEED_10M in schema_names:
        v_kmh = pl.col(NwpColumns.WIND_SPEED_10M) * 3.6
        temp = pl.col(NwpColumns.TEMPERATURE_2M)

        weather = weather.with_columns(
            windchill=(
                13.12 + 0.6215 * temp - 11.37 * (v_kmh**0.16) + 0.3965 * temp * (v_kmh**0.16)
            ).cast(pl.Float32)
        )

    weather = weather.sort(NwpColumns.INIT_TIME)
    if history is not None:
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
        """Add a lagged weather feature using an exact point-in-time join.

        This function uses a backward `join_asof` on `init_time` with `target_valid_time`
        to fetch historical weather forecasts without lookahead bias. It simulates the exact
        knowledge state at `init_time` by finding the most recent forecast that was valid
        at `target_valid_time` (which is `valid_time - offset`).

        This prevents data leakage by ensuring the model only sees weather forecasts that
        were actually available at the time the forecast was made.
        """
        by_cols = ["target_valid_time"]
        if NwpColumns.H3_INDEX in schema_names:
            by_cols.append(NwpColumns.H3_INDEX)

        source_schema = source_df.collect_schema().names()
        actual_temp_col = NwpColumns.TEMPERATURE_2M
        actual_sw_col = NwpColumns.SW_RADIATION

        source_cols = [NwpColumns.VALID_TIME, NwpColumns.INIT_TIME, actual_temp_col]
        if NwpColumns.H3_INDEX in schema_names:
            source_cols.append(NwpColumns.H3_INDEX)
        if actual_sw_col in source_schema:
            source_cols.append(actual_sw_col)

        # Filter source_df to only include ensemble_member == 0 if it exists.
        # This ensures that lag features are consistent across all ensemble members
        # and reduces the memory footprint of the join.
        if NwpColumns.ENSEMBLE_MEMBER in source_schema:
            source_df = source_df.filter(pl.col(NwpColumns.ENSEMBLE_MEMBER) == 0)

        right = source_df.select(source_cols).rename(
            {
                NwpColumns.VALID_TIME: "target_valid_time",
                actual_temp_col: f"{NwpColumns.TEMPERATURE_2M}_{suffix}",
            }
        )
        if actual_sw_col in source_schema:
            right = right.rename({actual_sw_col: f"{NwpColumns.SW_RADIATION}_{suffix}"})

        left = df.with_columns(target_valid_time=pl.col(NwpColumns.VALID_TIME) - offset)

        # We explicitly sort by the group keys (by_cols) and the join key (init_time)
        # to ensure the data is correctly ordered for the asof join.
        # We pass check_sortedness=False to suppress a false-positive warning from Polars
        # that can occur even when the data is correctly sorted.
        return (
            left.sort(by_cols + [NwpColumns.INIT_TIME])
            .join_asof(
                right.sort(by_cols + [NwpColumns.INIT_TIME]),
                on=NwpColumns.INIT_TIME,
                by=by_cols,
                strategy="backward",
                check_sortedness=False,
            )
            .drop("target_valid_time")
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


def add_time_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add lead_time_hours and nwp_init_hour features.

    Args:
        df: The input LazyFrame (schema: XGBoostInputFeatures).

    Returns:
        LazyFrame with added time features (schema: XGBoostInputFeatures).
    """

    return df.with_columns(
        lead_time_hours=(
            pl.col(NwpColumns.VALID_TIME) - pl.col(NwpColumns.INIT_TIME)
        ).dt.total_minutes()
        / 60.0,
        nwp_init_hour=pl.col(NwpColumns.INIT_TIME).dt.hour().cast(pl.Int32),
    )
