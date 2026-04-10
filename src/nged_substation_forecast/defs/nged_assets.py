"""Dagster assets for NGED data."""

import dagster as dg
import polars as pl
import patito as pt
from pathlib import Path
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
    clean_power_data,
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

        # Clean power data
        time_series_id = int(metadata_df.get_column("time_series_id").item())
        cleaned_df = clean_power_data(
            time_series_df,
            time_series_id=time_series_id,
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

        # Upsert metadata
        upsert_metadata(metadata_df, settings.nged_data_path / "metadata" / "json_metadata")

        # Clean power data
        time_series_id = int(metadata_df.get_column("time_series_id").item())
        cleaned_df = clean_power_data(
            time_series_df,
            time_series_id=time_series_id,
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


@asset(group_name="NGED_JSON")
def nged_sharepoint_json_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """Ingest the 33 NGED JSON files provided via SharePoint."""
    # Hardcoded path for the specific SharePoint drop
    json_dir = Path("data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/")

    if not json_dir.exists():
        context.log.warning(f"Directory {json_dir} does not exist. Skipping ingestion.")
        return

    for json_file in json_dir.glob("*.json"):
        context.log.info(f"Processing {json_file.name}")
        metadata_df, time_series_df = load_nged_json(json_file)

        # Upsert metadata to Parquet
        upsert_metadata(metadata_df, settings.nged_data_path / "metadata" / "json_metadata")

        # Clean power data
        time_series_id = int(metadata_df.get_column("time_series_id").item())
        cleaned_df = clean_power_data(
            time_series_df,
            time_series_id=time_series_id,
            variance_thresholds=settings.data_quality.variance_thresholds,
        )

        # Append to Delta table
        append_to_delta(cleaned_df, settings.nged_data_path / "delta" / "json_data")

    context.log.info("Finished processing SharePoint JSON data.")


@asset(group_name="NGED_JSON")
def substation_metadata(settings: ResourceParam[Settings]) -> pl.DataFrame:
    """Load substation metadata from Parquet."""
    metadata_path = settings.nged_data_path / "parquet" / "substation_metadata.parquet"
    return pl.read_parquet(metadata_path)
