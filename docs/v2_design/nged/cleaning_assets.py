import dagster as dg
import patito as pt
import polars as pl
from contracts.data_schemas import PowerTimeSeries
from utils import create_dagster_type_from_patito_model

PowerTimeSeriesType = create_dagster_type_from_patito_model(PowerTimeSeries)


@dg.asset(
    partitions_def=dg.DailyPartitionsDefinition(start_date="2020-01-01"),
    # Depends on BOTH ingestion streams. Dagster will ensure they are ready.
    deps=["archive_raw_power_time_series", "live_raw_power_time_series"],
    dagster_type=PowerTimeSeriesType,
    metadata={
        "delta_write_mode": "overwrite",
        "delta_partition_by": ["time_series_id"],
        "delta_partition_time_col": "period_end_time",
        # CRITICAL: Tell the IOManager we need a 1-day lookback to compute rolling features!
        "delta_read_lookback_days": 1,
    },
)
def cleaned_power_time_series(
    context: dg.AssetExecutionContext,
    # The IOManager will load the data from the unified 'raw_power_time_series' table
    raw_power_time_series: pt.DataFrame[PowerTimeSeries],
) -> pt.DataFrame[PowerTimeSeries]:
    """
    Applies data quality checks to the raw power data.

    WHY DAILY PARTITIONS? We want to be able to backfill the 4-year archive
    concurrently across many workers. Daily partitions allow this.

    WHY LOOKBACK? To calculate a 24-hour rolling standard deviation for the very
    first timestamp in a daily partition, we need the previous day's data. The
    IOManager handles fetching this extra data automatically based on metadata.
    """
    # 1. Apply cleaning logic (e.g., stuck sensors, insane values)
    # Note: Bad values are replaced with `null` to preserve the 30-min temporal grid.
    cleaned_df = apply_cleaning_rules(raw_power_time_series)

    # 2. Filter out the lookback period before returning, so we only overwrite
    # the exact partition window we are responsible for.
    start, end = context.asset_partitions_time_window
    cleaned_df = cleaned_df.filter(pl.col("period_end_time").is_between(start, end, closed="left"))

    # 3. Validate the output still matches the schema (which allows nulls)
    validated_df = PowerTimeSeries.validate(cleaned_df)

    # 4. Return. The IOManager handles the Delta OVERWRITE.
    return pt.DataFrame[PowerTimeSeries](validated_df)
