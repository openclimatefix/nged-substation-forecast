from datetime import datetime
from typing import Final, cast

import polars as pl
from contracts.settings import Settings
from dagster import AssetExecutionContext, MetadataValue, asset
from nged_data.storage import (
    append_time_series_to_delta_table,
    download_and_parse_files,
    get_new_file_listing,
    select_new_rows,
    upsert_metadata,
)

_DATETIME_FMT: Final[str] = "%Y-%m-%d %H:%M"


@asset
def power_time_series_and_metadata(context: AssetExecutionContext) -> None:
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

    # Fetch new data from S3, using the existing delta table to determine what's new.
    # We are deliberately keeping the code simple for now, but may move the S3 store
    # to a Dagster ConfigurableResource in the future.
    store = settings.get_nged_s3_store()
    paths_df = get_new_file_listing(store)
    new_paths_df = select_new_rows(paths_df, delta_path)

    context.add_output_metadata(
        {
            "all_timeseries_files_on_nged_s3": MetadataValue.json(
                _formatted_summary_of_dataframe(paths_df)
            ),
            "filtered_timeseries_files": MetadataValue.json(
                _formatted_summary_of_dataframe(new_paths_df)
            ),
        }
    )

    new_metadata_and_time_series = download_and_parse_files(store, new_paths_df)
    if new_metadata_and_time_series:
        new_metadata, new_time_series = new_metadata_and_time_series

        # Save new data:
        # TODO: Modify these two functions so they return a summary of what they did, so we can log
        # this.
        append_time_series_to_delta_table(new_time_series, delta_path)
        upsert_metadata(new_metadata, metadata_path)

        context.add_output_metadata(
            {
                "new_power_time_series_rows_downloaded": MetadataValue.json(
                    _formatted_summary_of_dataframe(new_time_series, time_col="time")
                )
            }
        )
    else:
        context.add_output_metadata({"new_power_time_series_rows_downloaded": 0})


def _formatted_summary_of_dataframe(
    df: pl.DataFrame, time_col: str = "end_time"
) -> dict[str, str | int]:
    summary: dict[str, str | int] = {"len": len(df)}
    if len(df) > 0:
        summary.update(
            {
                "start_date": cast(datetime, df[time_col].min()).strftime(_DATETIME_FMT),
                "end_date": cast(datetime, df[time_col].max()).strftime(_DATETIME_FMT),
            }
        )
    return summary
