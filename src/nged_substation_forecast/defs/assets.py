from datetime import datetime
from typing import Any, cast

import patito as pt
from contracts.power_schemas import PowerTimeSeries
from contracts.settings import Settings
from dagster import AssetExecutionContext, MetadataValue, TableRecord, asset
from nged_data.storage import (
    NoNewData,
    _ProcessedFileListing,
    download_and_parse_files,
    list_timeseries_json_files,
    remove_small_files_from_listing,
    select_new_rows,
    upsert_metadata,
)


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
    list_of_all_json_files = list_timeseries_json_files(store)
    list_of_large_json_files = remove_small_files_from_listing(list_of_all_json_files)
    list_of_new_json_files = select_new_rows(list_of_large_json_files, delta_path)

    # Log statistics to be shown in Dagster's UI.
    _log_paths_stats(
        context, list_of_all_json_files, list_of_large_json_files, list_of_new_json_files
    )

    try:
        new_metadata, new_power_ts = download_and_parse_files(store, list_of_new_json_files)
    except NoNewData:
        context.add_output_metadata({"Rows of new power TS downloaded": 0, "New metadata rows": 0})
        return

    # Save TimeSeriesMetadata:
    upsert_metadata_stats = upsert_metadata(new_metadata, metadata_path)
    context.add_output_metadata(upsert_metadata_stats)

    # Save PowerTimeSeries:
    new_power_ts_deduped = select_new_rows(new_power_ts, delta_path)
    if not new_power_ts_deduped.is_empty():
        # FIXME: mkdir won't work when delta_path is on S3!
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        new_power_ts_deduped.write_delta(
            delta_path, mode="append", delta_write_options={"partition_by": "time_series_id"}
        )

    _log_power_timeseries_stats(context, new_power_ts, new_power_ts_deduped)


def _log_paths_stats(
    context: AssetExecutionContext,
    list_of_all_json_files: pt.DataFrame[_ProcessedFileListing],
    list_of_large_json_files: pt.DataFrame[_ProcessedFileListing],
    list_of_new_json_files: pt.DataFrame[_ProcessedFileListing],
) -> None:
    table = [
        TableRecord(_summarise_file_listing(list_of_all_json_files, "All JSON files on S3")),
        TableRecord(_summarise_file_listing(list_of_large_json_files, "Files larger than 1kB")),
        TableRecord(_summarise_file_listing(list_of_new_json_files, "Files with new data")),
    ]
    context.add_output_metadata({"nged_s3_paths": MetadataValue.table(table)})


# TODO: Think about reducing duplication and tidying. Some thoughts:
# - Should the _summarise_* functions be methods on the data contract class?
# - Can we reduce duplication between _summarise_paths_df and _summarise_power_ts?
# - Should we use a TypedDict for these summaries? That sets defaults to N/A?


def _summarise_file_listing(
    paths_df: pt.DataFrame[_ProcessedFileListing], stage_name: str
) -> dict[str, str | int]:
    summary: dict[str, str | int] = {
        "Stage": stage_name,
        "Files": len(paths_df),
        "Start Date": "N/A",
        "End Date": "N/A",
        "TimeSeriesIDs": "N/A",
        "Min file size": "N/A",
        "Max file size": "N/A",
    }
    if len(paths_df) > 0:
        summary.update(
            {
                "Start Date": _format_datetime(paths_df["start_time"].min()),
                "End Date": _format_datetime(paths_df["end_time"].max()),
                # TODO: We can't list *all* time_series_ids when we're handling 1,000s of IDs!
                "TimeSeriesIDs": str(paths_df["time_series_id"].unique().sort().to_list()),
                "Min file size": paths_df["filesize_bytes"].min(),
                "Max file size": paths_df["filesize_bytes"].max(),
            }
        )
    return summary


def _log_power_timeseries_stats(
    context: AssetExecutionContext,
    new_power_ts_downloaded: pt.DataFrame[PowerTimeSeries],
    new_power_ts_deduped: pt.DataFrame[PowerTimeSeries],
) -> None:
    table = [
        TableRecord(_summarise_power_ts(new_power_ts_downloaded, "Downloaded timeseries")),
        TableRecord(_summarise_power_ts(new_power_ts_deduped, "De-duped rows appended to disk")),
    ]
    context.add_output_metadata({"PowerTimeSeries": MetadataValue.table(table)})


def _summarise_power_ts(
    time_series_df: pt.DataFrame[PowerTimeSeries], stage_name: str
) -> dict[str, str | int]:
    summary: dict[str, str | int] = {
        "Stage": stage_name,
        "Rows": len(time_series_df),
        "Start Date": "N/A",
        "End Date": "N/A",
        "TimeSeriesIDs": "N/A",
    }
    if len(time_series_df) > 0:
        summary.update(
            {
                "Start Date": _format_datetime(time_series_df["time"].min()),
                "End Date": _format_datetime(time_series_df["time"].max()),
                # TODO: We can't list *all* time_series_ids when we're handling 1,000s of IDs!
                "TimeSeriesIDs": str(time_series_df["time_series_id"].unique().sort().to_list()),
            }
        )
    return summary


def _format_datetime(dt: Any) -> str:
    return cast(datetime, dt).strftime("%Y-%m-%d %H:%M")
