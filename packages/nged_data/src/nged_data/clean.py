import polars as pl
import patito as pt
import dagster
from contracts.data_schemas import PowerTimeSeries


from typing import cast


def sort_data(df: pt.DataFrame[PowerTimeSeries]) -> pt.DataFrame[PowerTimeSeries]:
    """Sorts the DataFrame by 'period_end_time'."""
    return cast(pt.DataFrame[PowerTimeSeries], df.sort("period_end_time"))


def calculate_rolling_variance(df: pt.DataFrame[PowerTimeSeries]) -> pl.DataFrame:
    """Calculates rolling variance over a 6-hour window per time_series_id."""
    # 6 hours = 12 periods of 30 minutes
    return df.with_columns(
        rolling_variance=pl.col("power").rolling_var(window_size=12).over("time_series_id")
    )


def validate_data(df: pl.DataFrame) -> pt.DataFrame[PowerTimeSeries]:
    """Validates the cleaned DataFrame."""
    if df.is_empty():
        raise ValueError("All rows were removed after filtering by variance threshold.")
    return PowerTimeSeries.validate(df)


def clean_power_time_series(
    df: pl.DataFrame,
    stuck_std_threshold: float,
    min_mw_threshold: float,
    max_mw_threshold: float,
    time_series_id: int | None = None,
    variance_thresholds: dict[int, float] | None = None,
    default_threshold: float = 0.1,
) -> pt.DataFrame[PowerTimeSeries]:
    # 1. Validate input
    df = PowerTimeSeries.validate(df)

    # 1.5 Remove duplicates
    df = cast(
        pt.DataFrame[PowerTimeSeries], df.unique(subset=["time_series_id", "period_end_time"])
    )

    # 2. Sort
    df = sort_data(df)

    # 3. Drop initial nulls
    df = df.drop_nulls(subset=["power"])
    if df.is_empty():
        raise ValueError("All rows were removed after dropping nulls.")
    df = PowerTimeSeries.validate(df)

    # Determine threshold
    threshold = default_threshold
    if variance_thresholds and time_series_id is not None and time_series_id in variance_thresholds:
        threshold = variance_thresholds[time_series_id]

    # 4. Identify 'bad pre-amble data'
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

    # 5. Identify 'stuck' and 'insane' values
    # Stuck sensors: Rolling std dev < stuck_std_threshold over 48-period (24-hour) window.
    # Insane power: power < min_mw_threshold or power > max_mw_threshold.

    # Note: The original cleaning.py used "substation_number" for rolling_std.
    # Assuming "time_series_id" is the equivalent here.

    stuck_mask = (
        pl.col("power").rolling_std(48).fill_null(0).over("time_series_id") < stuck_std_threshold
    )
    insane_mask = (pl.col("power") < min_mw_threshold) | (pl.col("power") > max_mw_threshold)

    bad_value_mask = stuck_mask | insane_mask

    df = df.with_columns(
        pl.when(bad_value_mask).then(pl.lit(None)).otherwise(pl.col("power")).alias("power")
    )

    # Drop nulls again
    df = df.with_columns(power=pl.col("power").fill_nan(None)).drop_nulls(subset=["power"])
    dagster.get_dagster_logger().info(f"DF after dropping nulls: {df}")

    # 6. Validate
    return validate_data(df)
