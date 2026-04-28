import logging
from pathlib import Path
from typing import Final, cast, overload

import obstore
import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import PROJECT_ROOT, Settings

from nged_data.read_nged_json import (
    _extract_power_time_series,
    _extract_time_series_metadata,
)

log = logging.getLogger(__name__)


class _NgedJsonFileListing(pt.Model):
    path: str
    time_series_id: int = pt.Field(dtype=PowerTimeSeries.dtypes["time_series_id"])
    end_time: int = pt.Field(
        dtype=PowerTimeSeries.dtypes["time"],
        description=(
            "The end of the time window recorded by the time series data in the JSON file,"
            " according to the Unix epoch in the path"
        ),
    )


def get_new_file_listing(store: obstore.store.S3Store) -> pt.DataFrame[_NgedJsonFileListing]:
    """List all the timeseries JSON files in NGED's S3 bucket.

    The paths are assumed to be of the form:
    timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
    """
    paths: list[str] = []
    for chunk in store.list(prefix="timeseries"):
        # `list()` returns the file listing in chunks of `chunk_size=50` items per chunk.
        paths_for_chunk = [item["path"] for item in chunk if item["path"].endswith(".json")]
        paths.extend(paths_for_chunk)

    # Create DataFrame of paths.
    # TODO: Also extract the start_time, as it's useful for logging.
    paths_df = pl.DataFrame({"path": paths}).with_columns(
        # Extract the end time:    ↓↓↓↓↓↓↓↓↓↓↓↓↓
        # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
        end_time=(
            pl.col("path")
            .str.extract(r"/(\d+)_(\d+)/", 2)  # Capture group 2: the digits after the underscore
            .cast(pl.Int64)
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
            .cast(UTC_DATETIME_DTYPE)  # Cast from time_unit="ms" to "us"
        ),
        # Extract the time series ID:                       ↓↓
        # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
        time_series_id=(pl.col("path").str.extract(r"TimeSeries_(\d+)", 1).cast(pl.Int32)),
    )
    paths_df = paths_df.sort("end_time")
    return _NgedJsonFileListing.validate(paths_df)


def download_and_parse_files(
    store: obstore.store.S3Store, paths_df: pt.DataFrame[_NgedJsonFileListing]
) -> None | tuple[pt.DataFrame[TimeSeriesMetadata], pt.DataFrame[PowerTimeSeries]]:
    """Load data end_time by end_time, in order, so more recent data overwrites older duplicates, if
    there are any duplicates.

    Returns None if there is no new data.
    """
    metadata_dfs = []
    power_time_series_dfs = []
    for end_time, df_for_end_time in paths_df.group_by("end_time", maintain_order=True):
        for path in df_for_end_time["path"]:
            # TODO: Use `store.get_async` to get all files for this group concurrently.
            result = store.get(path)
            json_bytes = bytes(result.bytes())
            df = pl.read_json(json_bytes)

            # Extract TimeSeriesMetadata from df:
            new_metadata_df = _extract_time_series_metadata(df)
            metadata_dfs.append(new_metadata_df)
            time_series_id: int = new_metadata_df["time_series_id"].item()

            # Extract PowerTimeSeries from df:
            try:
                new_time_series_df = _extract_power_time_series(
                    df=df, time_series_id=time_series_id
                )
            except pl.exceptions.InvalidOperationError as e:
                if "invalid dtype: expected 'Struct', got 'Null' for 'data'" in str(e):
                    log.warning(
                        f"The 'data' field is 'null' in {path=}. This is expected behaviour if"
                        " NGED's meter reported no values for the period covered by the JSON file."
                    )
                    # TODO: Maybe we need a way to stop our system from downloading files with no
                    # data in them, every time we run. Maybe filter out JSON files below a certain
                    # filesize???
                else:
                    raise
            else:
                power_time_series_dfs.append(new_time_series_df)

    log.info(
        f"{len(metadata_dfs)} new TimeSeriesMetadata DataFrames and {len(power_time_series_dfs)}"
        " new PowerTimeSeries dataframes extracted from NGED JSON data."
    )
    # Concatenate and return:
    if metadata_dfs and power_time_series_dfs:
        metadata_df = (
            pl.concat(metadata_dfs, how="diagonal")
            .unique(subset="time_series_id", keep="last")
            .sort("time_series_id")
        )
        time_series_df = (
            pl.concat(power_time_series_dfs)
            .unique(subset=["time_series_id", "time"], keep="last")
            .sort(by=PowerTimeSeries.columns_to_sort_by)
        )

        return TimeSeriesMetadata.validate(metadata_df), PowerTimeSeries.validate(time_series_df)
    else:
        return None


def get_nged_s3_store() -> obstore.store.S3Store:
    assert (PROJECT_ROOT / ".env").exists()
    settings = Settings()
    return obstore.store.S3Store.from_url(
        url=settings.nged_s3_bucket_url,
        config={
            "aws_access_key_id": settings.nged_s3_bucket_access_key,
            "aws_secret_access_key": settings.nged_s3_bucket_secret,
        },
    )


class _MaxTimePerTimeSeriesId(pt.Model):
    time_series_id: int = pt.Field(dtype=PowerTimeSeries.dtypes["time_series_id"])
    max_time: int = pt.Field(dtype=PowerTimeSeries.dtypes["time"])


# This overload tells type checkers that if you pass a `pt.DataFrame[PowerTimeSeries]` into
# `select_new_rows` then you get a `pt.DataFrame[PowerTimeSeries]` back.
@overload
def select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries],
    delta_path: Path,
) -> pt.DataFrame[PowerTimeSeries]: ...


# This overload tells type checkers that if you pass a `pt.DataFrame[_NgedJsonFileListing]` into
# `select_new_rows` then you get a `pt.DataFrame[_NgedJsonFileListing]` back.
@overload
def select_new_rows(
    time_series: pt.DataFrame[_NgedJsonFileListing],
    delta_path: Path,
) -> pt.DataFrame[_NgedJsonFileListing]: ...


def select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries | _NgedJsonFileListing],
    delta_path: Path,
) -> pt.DataFrame[PowerTimeSeries | _NgedJsonFileListing]:
    """
    Return rows in `time_series` that are more recent than the most recent
    data already in our Delta table, on a time_series_id by time_series_id basis.
    """

    if not delta_path.exists():
        log.info(f"{delta_path=} does not exist yet.")
        return time_series

    # Scan the existing delta table and find the max time per time_series_id
    max_times = cast(
        pl.DataFrame,  # Cast to pl.DataFrame to keep type checkers happy.
        pl.scan_delta(delta_path).group_by("time_series_id").agg(max_time=pl.max("time")).collect(),
    )

    log.info(
        f"Loaded max times for {max_times.height} time_series_ids from {delta_path}."
        f" Earliest time = {max_times['max_time'].min()}."
        f" Latest time = {max_times['max_time'].max()}"
    )

    _MaxTimePerTimeSeriesId.validate(max_times)

    # Check whether `time_series` is a `PowerTimeSeries` or a `_NgedJsonFileListing`
    if "time" in time_series.columns:
        pt_model = PowerTimeSeries
        time_col = "time"
        columns_to_sort_by = PowerTimeSeries.columns_to_sort_by
    elif "end_time" in time_series.columns:
        pt_model = _NgedJsonFileListing
        time_col = "end_time"
        columns_to_sort_by = "end_time"
    else:
        raise ValueError(
            "Expected `time_series` to have either a `time` column or an `end_time` column,"
            f" not {time_series.columns=}"
        )

    filtered_df = cast(
        pl.DataFrame,
        time_series.lazy()
        .join(max_times.lazy(), on="time_series_id", how="left")
        # If max_time is null for this time_series_id then this is a new time_series_id.
        .filter(pl.col("max_time").is_null() | (pl.col(time_col) > pl.col("max_time")))
        .drop("max_time")
        .sort(by=columns_to_sort_by)
        .collect(),
    )

    return pt.DataFrame(filtered_df).set_model(pt_model).validate()


def upsert_metadata(
    new_metadata: pt.DataFrame[TimeSeriesMetadata], metadata_path: Path
) -> pt.DataFrame[TimeSeriesMetadata]:
    """
    Upserts metadata to a Parquet file.

    This function assumes it is called by one thread at a time so no
    explicit locking is required.

    If the local Parquet file does not exist, it saves the new_metadata.
    If it exists, it merges the new_metadata with the existing metadata,
    keeping the latest version for each time_series_id, and updates the
    Parquet file if there are differences.

    Args:
        new_metadata: The new metadata DataFrame.
        metadata_path: The path to the Parquet file where we store our local version of the
        metadata.
    """
    COMPRESSION: Final[str] = "zstd"

    new_metadata = TimeSeriesMetadata.validate(new_metadata.sort("time_series_id"))

    # FIXME: `path.exists()` won't work when metadata_path is on S3!
    if not metadata_path.exists():
        log.info(f"Metadata file not found at {metadata_path}. Creating new file.")
        new_metadata.write_parquet(metadata_path, compression=COMPRESSION)
        return new_metadata

    # Read existing metadata
    existing_metadata = pl.read_parquet(metadata_path)
    TimeSeriesMetadata.validate(existing_metadata)

    # Compare metadata. `metadata_diff` contains all rows in `new_metadata` that do not have an
    # exact match in `existing_metadata`. Adapted from https://stackoverflow.com/a/79888719
    metadata_diff = new_metadata.filter(
        ~new_metadata.hash_rows().is_in(existing_metadata.hash_rows())
    )
    metadata_diff = TimeSeriesMetadata.validate(metadata_diff)

    if metadata_diff.is_empty():
        log.info("TimeSeriesMetadata is up to date.")
    else:
        log.info(
            f"New TimeSeriesMetadata available for {metadata_diff.height} timeseries_ids."
            f" Updating {metadata_path}."
        )

        # Merge metadata. Put new_metadata first so that unique(keep="first") keeps the new version
        merged_metadata = (
            pl.concat([new_metadata, existing_metadata])
            .unique(subset="time_series_id", keep="first")
            .sort("time_series_id")
        )

        TimeSeriesMetadata.validate(merged_metadata)

        merged_metadata.write_parquet(metadata_path, compression=COMPRESSION)

    return metadata_diff
