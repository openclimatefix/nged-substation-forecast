import math

import polars as pl

# Static features registry
# These are basic features that don't require parameterization.
STATIC_FEATURE_REGISTRY: dict[str, pl.Expr] = {
    "windchill": (
        13.12
        + 0.6215 * pl.col("temperature_2m")
        - 11.37 * ((pl.col("wind_speed_10m") * 3.6) ** 0.16)
        + 0.3965 * pl.col("temperature_2m") * ((pl.col("wind_speed_10m") * 3.6) ** 0.16)
    ).alias("windchill"),
    "temperature_2m_trend_6h": (pl.col("temperature_2m") - pl.col("temperature_2m_6h_ago")).alias(
        "temperature_2m_trend_6h"
    ),
}


def calculate_lead_time(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Calculates the lead time in hours if 'init_time' is present in the schema."""
    if "init_time" in lf.collect_schema().names():
        return lf.with_columns(
            lead_time_hours=(
                (pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600
            ).cast(pl.Float32)
        )
    return lf


def apply_lag_feature(lf: pl.LazyFrame, base_col: str, lag_hours: int) -> pl.LazyFrame:
    """
    Applies a lag feature using a time-aware lazy self-join.

    Why a lazy join instead of shift()? Using shift() assumes the data is perfectly
    contiguous without missing rows. If a row is missing, shift() will pull data from
    the wrong time period, silently corrupting the feature. A time-aware join guarantees
    mathematical safety: if the lagged time doesn't exist, it correctly returns null.

    Args:
        lf: The LazyFrame to apply the lag to.
        base_col: The column to lag (e.g., 'power').
        lag_hours: The number of hours to lag.

    Returns:
        A LazyFrame with the new lagged column.
    """
    # Create a dataframe with the target time we want to join against
    lf_with_target_time = lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
    )

    # The right side of the join only needs the keys and the base column
    right_lf = lf.select(
        pl.col("time_series_id"),
        pl.col("valid_time"),
        pl.col(base_col).alias(f"{base_col}_lag_{lag_hours}h"),
    )

    # Join target_lf with right_lf
    result = lf_with_target_time.join(
        right_lf,
        left_on=["time_series_id", "target_time"],
        right_on=["time_series_id", "valid_time"],
        how="left",
    ).drop("target_time")

    return result


def apply_rolling_mean_feature(lf: pl.LazyFrame, base_col: str, window_hours: int) -> pl.LazyFrame:
    """
    Applies a rolling mean feature using time-aware rolling aggregations.

    Why dynamically check for ensemble_member? If we group by time_series_id alone
    when ensemble members are present, the rolling window will mix data across different
    ensemble members, causing data contamination. Grouping by ensemble_member (if present)
    prevents this.

    Args:
        lf: The LazyFrame to apply the rolling mean to.
        base_col: The column to calculate the rolling mean for.
        window_hours: The window size in hours.

    Returns:
        A LazyFrame with the new rolling mean column.
    """
    schema_names = lf.collect_schema().names()
    group_by_cols = ["time_series_id"]
    if "ensemble_member" in schema_names:
        group_by_cols.append("ensemble_member")

    rolled = lf.rolling(
        index_column="valid_time", period=f"{window_hours}h", group_by=group_by_cols
    ).agg(pl.col(base_col).mean().alias(f"{base_col}_rolling_mean_{window_hours}h"))

    # Join the result back to the original lf
    join_keys = group_by_cols + ["valid_time"]
    return lf.join(rolled, on=join_keys, how="left")


def apply_latest_weekly_lag_feature(
    lf: pl.LazyFrame, source_lf: pl.LazyFrame, base_col: str, new_col: str, join_cols: list[str]
) -> pl.LazyFrame:
    """
    Applies a dynamic lag feature based on the lead time.

    Why dynamic lags? If we are forecasting 10 days ahead, a 7-day lag is useless because
    we won't have the actual data 7 days from now. We need to use the *latest available*
    weekly lag (e.g., 14 days ago). This function calculates the appropriate multiple of
    168 hours (1 week) based on the lead time.

    Args:
        lf: The main LazyFrame containing the target times and lead times.
        source_lf: The LazyFrame containing the historical data to pull from.
        base_col: The column in source_lf to lag (e.g., 'power').
        new_col: The name of the resulting lagged column.
        join_cols: Additional columns to join on (e.g., ['time_series_id', 'ensemble_member']).

    Returns:
        A LazyFrame with the new dynamically lagged column.
    """
    schema_names = lf.collect_schema().names()

    if "lead_time_hours" in schema_names:
        # Calculate the required lag in hours.
        # If lead_time is 0-167, lag is 168.
        # If lead_time is 168-335, lag is 336.
        lag_hours = (((pl.col("lead_time_hours") / 168).floor() + 1) * 168).cast(pl.Int64)
    else:
        # Default to 1 week if no lead time is available
        lag_hours = pl.lit(168).cast(pl.Int64)

    # Calculate the target time we need to look up in the source data
    lf_with_target = lf.with_columns(
        target_time=pl.col("valid_time") - pl.duration(hours=lag_hours)
    )

    # Prepare the right side of the join
    right_lf = source_lf.select(
        *join_cols,
        pl.col("valid_time"),
        pl.col(base_col).alias(new_col),
    )

    # Perform the join
    result = lf_with_target.join(
        right_lf,
        left_on=join_cols + ["target_time"],
        right_on=join_cols + ["valid_time"],
        how="left",
    ).drop("target_time")

    return result


def apply_local_time_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Applies local time features (time of day, day of week, time of year) to the LazyFrame.

    Why local time? Energy consumption patterns are driven by human behavior, which follows
    local time (including Daylight Saving Time), not UTC. A 9 AM peak in winter (UTC) is
    different from a 9 AM peak in summer (UTC+1).

    Args:
        lf: The LazyFrame containing a 'valid_time' column in UTC.

    Returns:
        A LazyFrame with new local time features.
    """
    # Convert valid_time to Europe/London timezone
    # We first ensure it's treated as UTC, then convert to the target timezone
    lf = lf.with_columns(
        local_time=pl.col("valid_time")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/London")
    )

    # Calculate local UTC offset in hours
    # base_utc_offset + dst_offset gives the total offset from UTC
    lf = lf.with_columns(
        local_utc_offset=(
            pl.col("local_time").dt.base_utc_offset() + pl.col("local_time").dt.dst_offset()
        ).dt.total_seconds()
        / 3600
    )

    # Calculate local time of day (0-24)
    local_hour_float = pl.col("local_time").dt.hour() + pl.col("local_time").dt.minute() / 60.0

    # Calculate local time of year (0-1)
    # Using 366 to safely handle leap years without complex logic
    local_year_fraction = pl.col("local_time").dt.ordinal_day() / 366.0

    # Calculate local day of week (1-7)
    local_weekday = pl.col("local_time").dt.weekday()

    weekday_map = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday",
    }

    lf = lf.with_columns(
        local_time_of_day_sin=(local_hour_float / 24.0 * 2 * math.pi).sin().cast(pl.Float32),
        local_time_of_day_cos=(local_hour_float / 24.0 * 2 * math.pi).cos().cast(pl.Float32),
        local_time_of_year_sin=(local_year_fraction * 2 * math.pi).sin().cast(pl.Float32),
        local_time_of_year_cos=(local_year_fraction * 2 * math.pi).cos().cast(pl.Float32),
        local_day_of_week_sin=(local_weekday / 7.0 * 2 * math.pi).sin().cast(pl.Float32),
        local_day_of_week_cos=(local_weekday / 7.0 * 2 * math.pi).cos().cast(pl.Float32),
        local_day_of_week=local_weekday.replace_strict(weekday_map, return_dtype=pl.String).cast(
            pl.Enum(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        ),
    )

    # Drop the temporary local_time column
    return lf.drop("local_time")
