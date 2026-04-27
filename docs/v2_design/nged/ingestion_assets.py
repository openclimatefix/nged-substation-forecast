import dagster as dg
import patito as pt
import polars as pl
from contracts.data_schemas import PowerTimeSeries
from utils import create_dagster_type_from_patito_model

PowerTimeSeriesType = create_dagster_type_from_patito_model(PowerTimeSeries)


@dg.asset(
    dagster_type=PowerTimeSeriesType,
    metadata={
        # Both archive and live write to the SAME physical Delta table
        "delta_table_name": "raw_power_time_series",
        "delta_write_mode": "incremental_append",
        "delta_append_keys": ["time_series_id", "period_end_time"],
    },
)
def archive_raw_power_time_series(
    context: dg.AssetExecutionContext,
) -> pt.DataFrame[PowerTimeSeries]:
    """
    Unpartitioned asset. Runs ONCE when a researcher initializes the project.
    Reads all per-substation JSON files in the archive directory.

    WHY UNPARTITIONED? The archive is partitioned by entity (substation), not time.
    Forcing this into a time-partitioned asset would require opening every file
    thousands of times. We load it once, and the IOManager appends it safely.
    """
    # Jack's comment: Can we reduce duplication between this function and the next one?

    # 1. Locate all archive JSON files
    # Jack's comment: Download the files concurrently.
    json_files = get_all_archive_json_files(context)

    # 2. Parse and combine
    dfs = [parse_single_json(f) for f in json_files]

    # Jack's comments:
    # - Let's have two functions:
    #    - nged_json_to_power_time_series and
    #    - nged_json_to_time_series_metadata
    # Each will validate the dataframes one-by-one

    combined_df = pl.concat(dfs)

    # 4. Return. The IO Manager handles the Delta incremental append.
    return pt.DataFrame[PowerTimeSeries](combined_df)


@dg.asset(
    partitions_def=dg.TimeWindowPartitionsDefinition(
        cron_schedule="0 */6 * * *", start="2026-01-01-00:00", fmt="%Y-%m-%d-%H:%M"
    ),
    dagster_type=PowerTimeSeriesType,
    metadata={
        # Both archive and live write to the SAME physical Delta table
        "delta_table_name": "raw_power_time_series",
        "delta_write_mode": "incremental_append",
        "delta_append_keys": ["time_series_id", "period_end_time"],
    },
)
def live_raw_power_time_series(context: dg.AssetExecutionContext) -> pt.DataFrame[PowerTimeSeries]:
    """
    Runs every 6 hours. Reads the rolling 2-week JSONs provided by NGED.

    WHY INCREMENTAL APPEND? Because NGED provides a rolling 2 weeks, we will see
    the same data multiple times. The IOManager's High Water Mark logic ensures
    we only append the strictly new rows.
    """
    # 1. Locate JSON files for the current 6-hour partition
    json_files = get_json_files_for_partition(context)

    # 2. Parse and combine
    dfs = [parse_single_json(f) for f in json_files]
    combined_df = pl.concat(dfs)

    # 3. Validate against Patito schema
    validated_df = PowerTimeSeries.validate(combined_df)

    # 4. Return. The IO Manager handles the Delta incremental append.
    return pt.DataFrame[PowerTimeSeries](validated_df)
