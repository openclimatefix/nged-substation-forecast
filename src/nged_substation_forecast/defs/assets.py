from contracts.settings import Settings
from dagster import asset
from nged_data.storage import (
    append_time_series_to_delta_table,
    load_new_data_from_nged_s3,
    upsert_metadata,
)


@asset
def power_time_series_and_metadata() -> None:
    """
    Ingests raw telemetry and metadata from NGED S3 into our local storage.

    This asset acts as the entry point for NGED data into our system. It fetches
    the latest available data from the external S3 bucket and appends it to our
    local Delta table for time series data, while upserting the latest metadata.
    This raw data will later be consumed by downstream cleaning assets to prepare
    it for forecasting models.

    WHY UNPARTITIONED? Because NGED's JSON files are published roughly every 5 hours, and so
    the start time changes every day. And because we don't want people to have to spin up
    thousands of Dagster runs (one per partition) when first backfilling. It's much more efficient
    to just check what's available on NGED's S3 bucket and append to our local Delta table.
    """
    settings = Settings()
    delta_path = settings.nged_data_path / "power_time_series.delta"
    metadata_path = settings.nged_data_path / "metadata.parquet"

    # Fetch new data from S3, using the existing delta table to determine what's new
    new_metadata, new_time_series = load_new_data_from_nged_s3(delta_path)

    append_time_series_to_delta_table(new_time_series, delta_path)
    upsert_metadata(new_metadata, metadata_path)
