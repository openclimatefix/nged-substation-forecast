import polars as pl
import patito as pt
import dagster
from contracts.data_schemas import PowerTimeSeries


def sort_data(df: pl.DataFrame) -> pl.DataFrame:
    """Sorts the DataFrame by 'period_end_time'."""
    return df.sort("period_end_time")


def calculate_rolling_variance(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates rolling variance over a 6-hour window."""
    rolling_df = df.rolling(index_column="period_end_time", period="6h").agg(
        rolling_variance=pl.col("power").var()
    )
    return df.join(rolling_df, on="period_end_time")


def validate_data(df: pl.DataFrame) -> pt.DataFrame[PowerTimeSeries]:
    """Validates the cleaned DataFrame."""
    if df.is_empty():
        raise ValueError("All rows were removed after filtering by variance threshold.")
    return PowerTimeSeries.validate(df)


def clean_power_data(
    df: pl.DataFrame,
    time_series_id: int | None = None,
    variance_thresholds: dict[int, float] | None = None,
    default_threshold: float = 0.1,
) -> pt.DataFrame[PowerTimeSeries]:
    # 1. Sort
    df = sort_data(df)

    # 2. Drop initial nulls
    df = df.drop_nulls(subset=["power"])
    if df.is_empty():
        raise ValueError("All rows were removed after dropping nulls.")

    # Determine threshold
    threshold = default_threshold
    if variance_thresholds and time_series_id is not None and time_series_id in variance_thresholds:
        threshold = variance_thresholds[time_series_id]

    # 3. Identify 'bad pre-amble data'
    df = calculate_rolling_variance(df)

    # Find the first index where rolling variance exceeds the threshold
    valid_rows = df.filter(pl.col("rolling_variance") > threshold)

    if valid_rows.is_empty():
        raise ValueError("All rows were removed after filtering by variance threshold.")

    first_valid_time = valid_rows.select("period_end_time").min().item()

    # Slice the DataFrame from that index onwards.
    df = df.filter(pl.col("period_end_time") >= first_valid_time)

    # Drop the rolling_variance column
    df = df.drop("rolling_variance")

    # Drop nulls again
    df = df.with_columns(power=pl.col("power").fill_nan(None)).drop_nulls(subset=["power"])
    dagster.get_dagster_logger().info(f"DF after dropping nulls: {df}")

    # 4. Validate
    return validate_data(df)
