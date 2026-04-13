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

from nged_substation_forecast.exceptions import NGEDDataValidationError
from nged_substation_forecast.utils import filter_new_delta_records


def process_nged_json_file(
    json_file: Path,
    metadata_path: Path,
    context: AssetExecutionContext,
) -> pt.DataFrame[PowerTimeSeries]:
    """Processes a single NGED JSON file: loads, upserts metadata, and validates.

    This helper function centralizes the ingestion logic for NGED JSON files,
    ensuring consistent metadata handling and strict data validation.
    If validation fails, it logs the error and raises an NGEDDataValidationError
    to halt the pipeline explicitly on schema mismatch and provide clear
    debugging context.
    """
    context.log.info(f"Processing {json_file.name}")
    metadata_df, time_series_df = load_nged_json(json_file)

    # Upsert metadata
    upsert_metadata(metadata_df, metadata_path)

    # Validate raw data
    try:
        validated_df = PowerTimeSeries.validate(time_series_df)
    except ValueError as e:
        context.log.error(f"Validation failed for {json_file.name}: {e}")
        # Raise a specific validation error to halt the pipeline explicitly on
        # schema mismatch and provide clear debugging context.
        raise NGEDDataValidationError(f"Failed to validate {json_file.name}: {e}") from e

    return pt.DataFrame[PowerTimeSeries](validated_df)


@asset(group_name="NGED_JSON")
def nged_json_archive_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """One-off historical backfill of NGED JSON data."""
    json_dir = settings.nged_data_path / "json" / "archive"
    metadata_path = settings.nged_data_path / "metadata" / "json_metadata"
    delta_path = settings.nged_data_path / "delta" / "raw_power_time_series"

    for json_file in json_dir.glob("*.json"):
        validated_df = process_nged_json_file(json_file, metadata_path, context)

        # Append to delta
        validated_df = pt.DataFrame[PowerTimeSeries](
            validated_df.unique(subset=["time_series_id", "period_end_time"])
        )
        append_to_delta(
            validated_df,
            delta_path,
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
    metadata_path = settings.nged_data_path / "metadata" / "json_metadata"
    delta_path = settings.nged_data_path / "delta" / "raw_power_time_series"

    cleaned_dfs = []
    for json_file in json_dir.glob("*.json"):
        validated_df = process_nged_json_file(json_file, metadata_path, context)
        cleaned_dfs.append(validated_df)

    if cleaned_dfs:
        # Combine all validated dataframes
        combined_df = pl.concat(cleaned_dfs).unique(subset=["time_series_id", "period_end_time"])

        # Use the new utility function to ensure idempotency
        combined_df = filter_new_delta_records(
            pt.DataFrame[PowerTimeSeries](combined_df), delta_path
        )

        # Append only the filtered, new data to the Delta table.
        if not combined_df.is_empty():
            append_to_delta(
                pt.DataFrame[PowerTimeSeries](combined_df),
                delta_path,
            )

    context.log.info(f"Finished processing live JSON data for {partition_date}.")


@asset(group_name="NGED_JSON")
def nged_sharepoint_json_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """Ingest the 33 NGED JSON files provided via SharePoint."""
    # Hardcoded path for the specific SharePoint drop.
    # Note that `nged_sharepoint_json_asset` will be deleted soon. This asset only exists for
    # temporary testing. So it's fine to use a hard-coded path for now.
    json_dir = Path("data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/")
    metadata_path = settings.nged_data_path / "parquet" / "time_series_metadata.parquet"
    delta_path = settings.nged_data_path / "delta" / "raw_power_time_series"

    if not json_dir.exists():
        context.log.warning(f"Directory {json_dir} does not exist. Skipping ingestion.")
        return

    for json_file in json_dir.glob("TimeSeries_*.json"):
        if "cleaned" in json_file.name:
            continue

        validated_df = process_nged_json_file(json_file, metadata_path, context)

        # Append to Delta table
        validated_df = pt.DataFrame[PowerTimeSeries](
            validated_df.unique(subset=["time_series_id", "period_end_time"])
        )
        append_to_delta(
            validated_df,
            delta_path,
        )

    context.log.info("Finished processing SharePoint JSON data.")
