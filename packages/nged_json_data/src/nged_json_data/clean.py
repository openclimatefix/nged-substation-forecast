import polars as pl
import patito as pt
import dagster
from contracts.data_schemas import PowerTimeSeries


def sort_data(df: pl.DataFrame) -> pl.DataFrame:
    """Sorts the DataFrame by 'end_time'."""
    return df.sort("end_time")


def calculate_rolling_variance(df: pl.DataFrame) -> pl.DataFrame:
    """Calculates rolling variance over a 6-hour window."""
    rolling_df = df.rolling(index_column="end_time", period="6h").agg(
        rolling_variance=pl.col("value").var()
    )
    return df.join(rolling_df, on="end_time")


def apply_variance_threshold(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Nullifies values where rolling variance is below the threshold."""
    dagster.get_dagster_logger().info(f"DF before threshold: {df}")
    df = df.with_columns(
        value=pl.when(
            (pl.col("rolling_variance").is_null()) | (pl.col("rolling_variance") <= threshold)
        )
        .then(None)
        .otherwise(pl.col("value"))
    )
    dagster.get_dagster_logger().info(f"DF after threshold: {df}")
    return df.drop("rolling_variance")


def validate_data(df: pl.DataFrame) -> pt.DataFrame[PowerTimeSeries]:
    """Validates the cleaned DataFrame."""
    if df.is_empty():
        raise ValueError("All rows were removed after filtering by variance threshold.")
    return PowerTimeSeries.validate(df)


def clean_power_data(
    df: pl.DataFrame,
    substation_number: int | None = None,
    variance_thresholds: dict[int, float] | None = None,
    default_threshold: float = 0.1,
) -> pt.DataFrame[PowerTimeSeries]:
    # 1. Sort
    df = sort_data(df)

    # 2. Drop initial nulls
    df = df.drop_nulls(subset=["value"])
    if df.is_empty():
        raise ValueError("All rows were removed after dropping nulls.")

    # Determine threshold
    threshold = default_threshold
    if (
        variance_thresholds
        and substation_number is not None
        and substation_number in variance_thresholds
    ):
        threshold = variance_thresholds[substation_number]

    # 3. Identify 'bad pre-amble data'
    df = calculate_rolling_variance(df)
    df = apply_variance_threshold(df, threshold)

    # Drop nulls again
    df = df.with_columns(value=pl.col("value").fill_nan(None)).drop_nulls(subset=["value"])
    dagster.get_dagster_logger().info(f"DF after dropping nulls: {df}")

    # 4. Validate
    return validate_data(df)
