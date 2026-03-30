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
    """Add windchill and rename weather variables to indicate their uint8 scaling.

    Clever Optimization:
    Weather variables are kept in their 0-255 scaled representation (e.g., `temperature_2m_uint8_scaled`)
    to save memory and computation. XGBoost is invariant to monotonic transformations, so this does not
    affect model performance. For SHAP analysis or EDA, use `descale_for_analysis` to convert them back
    to physical units.
    """
    schema_names = df.collect_schema().names()
    params = load_scaling_params()

    scaling_cols_df = params.select("col_name")
    scaling_cols = scaling_cols_df.to_series().to_list()

    existing_cols = [c for c in scaling_cols if c in schema_names]
    if not existing_cols:
        return df

    # Calculate windchill if both temperature and wind speed are present
    if NwpColumns.TEMPERATURE_2M in existing_cols and NwpColumns.WIND_SPEED_10M in existing_cols:
        # Temporarily descale just for windchill calculation
        descale_exprs = uint8_to_physical_unit(
            params.filter(
                pl.col("col_name").is_in([NwpColumns.TEMPERATURE_2M, NwpColumns.WIND_SPEED_10M])
            )
        )

        temp_exprs = [
            expr.alias(f"temp_descale_{expr.meta.output_name()}") for expr in descale_exprs
        ]
        df = df.with_columns(temp_exprs)

        v_kmh = pl.col(f"temp_descale_{NwpColumns.WIND_SPEED_10M}") * 3.6
        temp = pl.col(f"temp_descale_{NwpColumns.TEMPERATURE_2M}")

        df = df.with_columns(
            windchill=(
                13.12 + 0.6215 * temp - 11.37 * (v_kmh**0.16) + 0.3965 * temp * (v_kmh**0.16)
            ).cast(pl.Float32)
        ).drop(
            [
                f"temp_descale_{NwpColumns.TEMPERATURE_2M}",
                f"temp_descale_{NwpColumns.WIND_SPEED_10M}",
            ]
        )

    # Rename all scaled columns to *_uint8_scaled
    rename_mapping = {c: f"{c}_uint8_scaled" for c in existing_cols}
    df = df.rename(rename_mapping)

    return df


def descale_for_analysis(df: pl.LazyFrame) -> pl.LazyFrame:
    """Descale *_uint8_scaled columns back to physical units for SHAP/EDA.

    This utility function is designed for data scientists and engineers who need to
    inspect the model's features in their original physical units (e.g., degrees Celsius, m/s).
    It automatically detects any column ending in `_uint8_scaled`, looks up the correct
    scaling parameters, and converts the values back to their physical representation.

    Example Usage:
        ```python
        # Load the feature dataset
        features_df = pl.scan_parquet("data/features/*.parquet")

        # Descale the weather features for exploratory data analysis
        descaled_df = descale_for_analysis(features_df)

        # Now you can plot temperature in Celsius instead of 0-255!
        descaled_df.select("temperature_2m").collect().plot()
        ```

    Args:
        df: A Polars LazyFrame containing `_uint8_scaled` columns.

    Returns:
        A Polars LazyFrame with the scaled columns replaced by their physical unit equivalents.
    """
    schema_names = df.collect_schema().names()
    params = load_scaling_params()

    descale_exprs = []
    cols_to_drop = []
    for col in schema_names:
        if col.endswith("_uint8_scaled"):
            base_col = col.replace("_uint8_scaled", "")
            # Find the matching scaling param by checking if base_col ends with the param's col_name
            for row in params.iter_rows(named=True):
                param_col = row["col_name"]
                if base_col.endswith(param_col):
                    b_min = row["buffered_min"]
                    b_range = row["buffered_range"]
                    expr = ((pl.col(col).cast(pl.Float32) / 255 * b_range) + b_min).alias(base_col)
                    descale_exprs.append(expr)
                    cols_to_drop.append(col)
                    break

    if descale_exprs:
        df = df.with_columns(descale_exprs).drop(cols_to_drop)

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
        if NwpColumns.ENSEMBLE_MEMBER in schema_names:
            by_cols.append(NwpColumns.ENSEMBLE_MEMBER)

        # We need to use the renamed columns if they exist
        temp_col = f"{NwpColumns.TEMPERATURE_2M}_uint8_scaled"
        sw_col = f"{NwpColumns.SW_RADIATION}_uint8_scaled"

        source_schema = source_df.collect_schema().names()
        actual_temp_col = temp_col if temp_col in source_schema else NwpColumns.TEMPERATURE_2M
        actual_sw_col = sw_col if sw_col in source_schema else NwpColumns.SW_RADIATION

        source_cols = [NwpColumns.VALID_TIME, NwpColumns.INIT_TIME, actual_temp_col]
        if NwpColumns.H3_INDEX in schema_names:
            source_cols.append(NwpColumns.H3_INDEX)
        if NwpColumns.ENSEMBLE_MEMBER in schema_names:
            source_cols.append(NwpColumns.ENSEMBLE_MEMBER)
        if actual_sw_col in source_schema:
            source_cols.append(actual_sw_col)

        right = source_df.select(source_cols).rename(
            {
                NwpColumns.VALID_TIME: "target_valid_time",
                actual_temp_col: f"{NwpColumns.TEMPERATURE_2M}_{suffix}",
            }
        )
        if actual_sw_col in source_schema:
            right = right.rename({actual_sw_col: f"sw_radiation_{suffix}"})

        left = df.with_columns(target_valid_time=pl.col(NwpColumns.VALID_TIME) - offset)

        return left.join_asof(right, on=NwpColumns.INIT_TIME, by=by_cols, strategy="backward").drop(
            "target_valid_time"
        )

    weather = _add_lag_asof(weather, full_weather, timedelta(days=7), "lag_7d")
    weather = _add_lag_asof(weather, full_weather, timedelta(days=14), "lag_14d")
    weather = _add_lag_asof(weather, full_weather, timedelta(hours=6), "6h_ago")

    # We need to use the renamed column for the trend calculation
    temp_col = f"{NwpColumns.TEMPERATURE_2M}_uint8_scaled"
    weather_schema = weather.collect_schema().names()
    actual_temp_col = temp_col if temp_col in weather_schema else NwpColumns.TEMPERATURE_2M

    return weather.with_columns(
        temp_trend_6h=(
            pl.col(actual_temp_col).cast(pl.Float32)
            - pl.col(f"{NwpColumns.TEMPERATURE_2M}_6h_ago").cast(pl.Float32)
        ).cast(pl.Float32)
    )
