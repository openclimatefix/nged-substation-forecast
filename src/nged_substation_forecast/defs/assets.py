from dagster import asset
from contracts.settings import Settings
from nged_data.storage import (
    load_new_data_from_nged_s3,
    append_time_series_to_delta_table,
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
    """
    settings = Settings()
    
    # Define paths for local storage
    delta_path = settings.nged_data_path / "power_time_series.delta"
    metadata_path = settings.nged_data_path / "metadata.parquet"
    
    # Fetch new data from S3, using the existing delta table to determine what's new
    new_metadata, new_time_series = load_new_data_from_nged_s3(delta_path)
    
    # Append the new time series data to our local Delta table
    append_time_series_to_delta_table(new_time_series, delta_path)
    
    # Upsert the new metadata to our local Parquet file
    upsert_metadata(new_metadata, metadata_path)
