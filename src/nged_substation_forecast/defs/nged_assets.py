"""Dagster assets for NGED data."""

import dagster as dg
import polars as pl
import patito as pt
from contracts.data_schemas import (
    PowerTimeSeries,
)
from contracts.settings import Settings
from dagster import (
    AssetExecutionContext,
    ResourceParam,
    asset,
)
from nged_json_data import (
    append_to_delta,
    clean_power_data,
    load_nged_json,
)


@asset(group_name="NGED_JSON")
def nged_json_archive_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """One-off historical backfill of NGED JSON data."""
    json_dir = settings.nged_data_path / "json" / "archive"

    for json_file in json_dir.glob("*.json"):
        metadata_df, time_series_df = load_nged_json(json_file)

        # Clean power data
        substation_number = metadata_df.get_column("substation_number").item()
        cleaned_df = clean_power_data(
            time_series_df,
            substation_number=substation_number,
            variance_thresholds=settings.data_quality.variance_thresholds,
        )

        # Append to delta
        append_to_delta(cleaned_df, settings.nged_data_path / "delta" / "json_data")

    context.log.info("Finished processing archive JSON data.")


@asset(
    partitions_def=dg.TimeWindowPartitionsDefinition(
        cron_schedule="0 */6 * * *",
        start="2026-01-26-00:00",
        fmt="%Y-%m-%d-%H:%M",
        timezone="UTC",
    ),
    group_name="NGED_JSON",
    op_tags={"dagster/concurrency_key": "nged_json_ingestion"},
)
def nged_json_live_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """Live updates of NGED JSON data."""
    # Assuming there's a directory with JSON files for the current partition
    partition_date = context.partition_time_window.start
    json_dir = settings.nged_data_path / "json" / "live" / partition_date.strftime("%Y-%m-%d-%H")

    cleaned_dfs = []
    for json_file in json_dir.glob("*.json"):
        metadata_df, time_series_df = load_nged_json(json_file)

        # Clean power data
        substation_number = metadata_df.get_column("substation_number").item()
        cleaned_df = clean_power_data(
            time_series_df,
            substation_number=substation_number,
            variance_thresholds=settings.data_quality.variance_thresholds,
        )
        cleaned_dfs.append(cleaned_df)

    if cleaned_dfs:
        # Combine all cleaned dataframes and append in a single operation
        combined_df = pl.concat(cleaned_dfs)
        append_to_delta(
            pt.DataFrame[PowerTimeSeries](combined_df),
            settings.nged_data_path / "delta" / "json_data",
        )

    context.log.info(f"Finished processing live JSON data for {partition_date}.")
