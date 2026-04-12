"""Dagster assets for NGED data."""

from pathlib import Path

import dagster as dg
import patito as pt
import polars as pl
from contracts.data_schemas import (
    PowerTimeSeries,
)
from contracts.settings import Settings
from dagster import (
    AssetExecutionContext,
    ResourceParam,
    asset,
)
from nged_data import (
    append_to_delta,
    load_nged_json,
    upsert_metadata,
)


@asset(group_name="NGED_JSON")
def nged_json_archive_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """One-off historical backfill of NGED JSON data."""
    json_dir = settings.nged_data_path / "json" / "archive"

    for json_file in json_dir.glob("*.json"):
        metadata_df, time_series_df = load_nged_json(json_file)

        # Upsert metadata
        upsert_metadata(metadata_df, settings.nged_data_path / "metadata" / "json_metadata")

        # Validate raw data
        validated_df = PowerTimeSeries.validate(time_series_df)

        # Append to delta
        validated_df = validated_df.unique(subset=["time_series_id", "period_end_time"])
        append_to_delta(
            pt.DataFrame[PowerTimeSeries](validated_df),
            settings.nged_data_path / "delta" / "raw_power_time_series",
        )

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

        # Upsert metadata
        upsert_metadata(metadata_df, settings.nged_data_path / "metadata" / "json_metadata")

        # Validate raw data
        validated_df = PowerTimeSeries.validate(time_series_df)
        cleaned_dfs.append(validated_df)

    if cleaned_dfs:
        # Combine all validated dataframes
        combined_df = pl.concat(cleaned_dfs).unique(subset=["time_series_id", "period_end_time"])

        delta_path = settings.nged_data_path / "delta" / "raw_power_time_series"

        # 1. Read the existing raw_power_time_series Delta table.
        if delta_path.exists():
            # 2. Find the maximum period_end_time in the existing data.
            # Read only period_end_time for efficiency.
            existing_df = pl.read_delta(str(delta_path), columns=["period_end_time"])

            if not existing_df.is_empty():
                max_timestamp = existing_df["period_end_time"].max()

                # 3. Filter the incoming combined_df to only include rows with period_end_time strictly greater than the maximum timestamp found.
                combined_df = combined_df.filter(pl.col("period_end_time") > max_timestamp)

        # 4. Append only the filtered, new data to the Delta table.
        # 5. Handle the case where the Delta table is empty (implicitly handled by the if delta_path.exists() and if not existing_df.is_empty() checks).
        if not combined_df.is_empty():
            append_to_delta(
                pt.DataFrame[PowerTimeSeries](combined_df),
                delta_path,
            )

    context.log.info(f"Finished processing live JSON data for {partition_date}.")


@asset(group_name="NGED_JSON")
def nged_sharepoint_json_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """Ingest the 33 NGED JSON files provided via SharePoint."""
    # Hardcoded path for the specific SharePoint drop
    json_dir = Path("data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/")

    if not json_dir.exists():
        context.log.warning(f"Directory {json_dir} does not exist. Skipping ingestion.")
        return

    for json_file in json_dir.glob("TimeSeries_*.json"):
        if "cleaned" in json_file.name:
            continue
        context.log.info(f"Processing {json_file.name}")
        metadata_df, time_series_df = load_nged_json(json_file)

        # Upsert metadata to Parquet
        upsert_metadata(
            metadata_df, settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
        )

        # Validate raw data
        try:
            validated_df = PowerTimeSeries.validate(time_series_df)
        except ValueError as e:
            context.log.warning(f"Skipping {json_file.name} due to: {e}")
            continue

        # Append to Delta table
        validated_df = validated_df.unique(subset=["time_series_id", "period_end_time"])
        append_to_delta(
            pt.DataFrame[PowerTimeSeries](validated_df),
            settings.nged_data_path / "delta" / "raw_power_time_series",
        )

    context.log.info("Finished processing SharePoint JSON data.")
