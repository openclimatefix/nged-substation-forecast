import logging
from pathlib import Path
from typing import cast, overload

import obstore
import patito as pt
import polars as pl
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import PROJECT_ROOT, Settings

from nged_data.read_nged_json import nged_json_to_metadata_df_and_time_series_df

log = logging.getLogger(__name__)


class _MaxTimePerTimeSeriesId(pt.Model):
    time_series_id: int = pt.Field(dtype=PowerTimeSeries.dtypes["time_series_id"])
    max_time: int = pt.Field(dtype=PowerTimeSeries.dtypes["time"])


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


def append_to_delta(power_time_series: pt.DataFrame[PowerTimeSeries], delta_path: Path) -> None:
    """
    Appends data to a Delta table, ensuring no duplicates based on (time_series_id, period_end_time).

    Args:
        power_time_series: The Patito DataFrame to append.
        delta_path: The path to the Delta table.
    """
    log.info(f"Preparing to append_to_delta at {delta_path}...")

    new_power_ts = _select_new_rows(power_time_series, delta_path)
    new_power_ts = new_power_ts.sort(by=PowerTimeSeries.columns_to_sort_by)

    log.info(
        f"Appending {new_power_ts.height:,d} rows of new PowerTimeSeries"
        f" (from {new_power_ts['time'].min()} to {new_power_ts['time'].max()}) to {delta_path=}"
    )

    PowerTimeSeries.validate(new_power_ts)

    if not new_power_ts.is_empty():
        delta_path.parent.mkdir(parents=True, exist_ok=True)
        new_power_ts.write_delta(
            delta_path, mode="append", delta_write_options={"partition_by": "time_series_id"}
        )


def load_new_data_from_nged_s3(
    delta_path: Path,
) -> tuple[pt.DataFrame[TimeSeriesMetadata], pt.DataFrame[PowerTimeSeries]]:
    store = get_nged_s3_store()

    # List all the JSON files on NGED's S3. The paths will be of the form:
    # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
    paths: list[str] = []
    for chunk in store.list(prefix="timeseries"):
        # `list()` returns the file listing in chunks of `chunk_size=50` items per chunk.
        paths_for_chunk = [item["path"] for item in chunk if item["path"].endswith(".json")]
        paths.extend(paths_for_chunk)

    # Create DataFrame of paths.
    paths_df = pl.DataFrame({"path": paths}).with_columns(
        # Extract the end time:    ↓↓↓↓↓↓↓↓↓↓↓↓↓
        # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
        end_time=(
            pl.col("path")
            .str.extract(r"/(\d+)_(\d+)/", 2)  # Capture group 2: the digits after the underscore
            .cast(pl.Int64)
            .cast(pl.Datetime("ms", time_zone="UTC"))
        ),
        # Extract the time series ID:                       ↓↓
        # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
        time_series_id=(pl.col("path").str.extract(r"TimeSeries_(\d+)", 1).cast(pl.Int32)),
    )
    paths_df = paths_df.sort("end_time")
    paths_df = _NgedJsonFileListing.validate(paths_df)

    paths_df = _select_new_rows(paths_df, delta_path)
    # TODO: All the code in this function above this line should probably in a separate function,
    # that will be wrapped in a Dagster ConfigurableResource. See:
    # https://github.com/openclimatefix/nged-substation-forecast/issues/115

    # Load data end_time by end_time, in order, so more recent data overwrites older duplicates, if
    # there are any duplicates.
    metadata_dfs = []
    power_time_series_dfs = []
    for _end_time, df_for_group in paths_df.group_by("end_time", maintain_order=True):
        for path in df_for_group["path"]:
            # TODO: Use `store.get_async` to get all files for this group concurrently.
            result = store.get(path)
            json_bytes = bytes(result.bytes())

            # TODO: Handle exception when JSON has null `data` field. James says:
            # "@Ben Willoughby can confirm, but I think that [null `data` fields] is the expected
            # behaviour for when no values were recorded by the monitor.
            new_metadata_df, new_time_series_df = nged_json_to_metadata_df_and_time_series_df(
                json_bytes
            )
            metadata_dfs.append(new_metadata_df)
            power_time_series_dfs.append(new_time_series_df)

    metadata_df = (
        pl.concat(metadata_dfs).unique(subset="time_series_id", keep="last").sort("time_series_id")
    )
    time_series_df = (
        pl.concat(power_time_series_dfs)
        .unique(subset=["time_series_id", "time"], keep="last")
        .sort(by=PowerTimeSeries.columns_to_sort_by)
    )

    return TimeSeriesMetadata.validate(metadata_df), PowerTimeSeries.validate(time_series_df)


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


@overload
def _select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries],
    delta_path: Path,
) -> pt.DataFrame[PowerTimeSeries]: ...


@overload
def _select_new_rows(
    time_series: pt.DataFrame[_NgedJsonFileListing],
    delta_path: Path,
) -> pt.DataFrame[_NgedJsonFileListing]: ...


def _select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries | _NgedJsonFileListing],
    delta_path: Path,
) -> pt.DataFrame[PowerTimeSeries | _NgedJsonFileListing]:
    """
    Filter `time_series` the find rows that are more recent than the most recent
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
    elif "end_time" in time_series.columns:
        pt_model = _NgedJsonFileListing
        time_col = "end_time"
    else:
        raise ValueError(
            "Expected `time_series` to have either a `time` column or an `end_time` column,"
            f" not {time_series.columns=}"
        )

    return cast(
        pt.DataFrame[pt_model],
        time_series.lazy()
        .join(max_times.lazy(), on="time_series_id", how="left")
        # If max_time is null for this time_series_id then this is a new time_series_id.
        .filter(pl.col("max_time").is_null() | (pl.col(time_col) > pl.col("max_time")))
        .drop("max_time")
        .collect(),
    )


def upsert_metadata(new_metadata: pt.DataFrame[TimeSeriesMetadata], metadata_path: Path) -> None:
    """
    Upserts metadata to a Parquet file.

    This function assumes it is called by the exclusive owner asset, so no
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
    new_metadata: pl.DataFrame = new_metadata.sort("time_series_id")
    TimeSeriesMetadata.validate(new_metadata)

    if not metadata_path.exists():
        log.info(f"Metadata file not found at {metadata_path}. Creating new file.")
        new_metadata.write_parquet(metadata_path)
        return

    # Read existing metadata
    existing_metadata = TimeSeriesMetadata.validate(pl.read_parquet(metadata_path))

    # Merge metadata
    # Put new_metadata first so that unique(keep="first") keeps the new version
    merged_metadata = TimeSeriesMetadata.validate(
        pl.concat([new_metadata, existing_metadata]).unique(subset="time_series_id", keep="first")
    ).sort("time_series_id")

    # Compare metadata
    if existing_metadata.equals(merged_metadata):
        log.info("Metadata is up to date.")
    else:
        log.info(f"Metadata update detected at {metadata_path}. Updating metadata file.")
        merged_metadata.write_parquet(metadata_path)
