import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures

# Static features registry
# These are basic features that don't require parameterization.
STATIC_FEATURE_REGISTRY: dict[str, pl.Expr] = {
    # Example static features (these would typically be more complex in reality)
    # For now, we'll just define some placeholders or simple expressions
    # Assuming 'valid_time' is available and is a datetime
    "local_time_of_day_sin": (
        pl.col("valid_time").dt.hour().cast(pl.Float32) / 24.0 * 2 * 3.14159
    ).alias("local_time_of_day_sin"),  # Simplified
    "local_day_of_week": pl.col("valid_time")
    .dt.weekday()
    .cast(pl.String)
    .alias("local_day_of_week"),  # Simplified
}


def apply_lag_feature(
    lf: pt.LazyFrame[AllFeatures], base_col: str, lag_hours: int
) -> pt.LazyFrame[AllFeatures]:
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


def apply_rolling_mean_feature(
    lf: pt.LazyFrame[AllFeatures], base_col: str, window_hours: int
) -> pt.LazyFrame[AllFeatures]:
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
