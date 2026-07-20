import logging
from typing import Final, Sequence, TypedDict, overload

import obstore
import patito as pt
import polars as pl
from contracts._uri import (
    ObjectStoreOptions,
    delta_table_exists,
    if_local_path_then_make_parent_dir,
    object_exists,
)
from contracts.common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.typing_utils import typeddict_to_dict

from nged_data.read_nged_json import (
    _extract_power_time_series,
    _extract_time_series_metadata,
)

log = logging.getLogger(__name__)


class _RawFileListItem(TypedDict):
    path: str
    filesize_bytes: int


class _ProcessedFileListing(pt.Model):
    path: str
    filesize_bytes: int
    time_series_id: int = _get_time_series_id_dtype()
    start_time: int = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The start of the time window recorded by the time series data in the JSON file,"
            " according to the Unix epoch in the path"
        ),
    )
    end_time: int = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The end of the time window recorded by the time series data in the JSON file,"
            " according to the Unix epoch in the path"
        ),
    )


def list_timeseries_json_files(
    store: obstore.store.S3Store,
) -> pt.DataFrame[_ProcessedFileListing]:
    """List all the timeseries JSON files in NGED's S3 bucket.

    The paths are assumed to be of the form:
    timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
    """
    raw_file_listing: list[_RawFileListItem] = []
    total_objects = 0
    for chunk in store.list(prefix="timeseries"):
        # `list()` returns the file listing in chunks of `chunk_size=50` items per chunk.
        for object_meta in chunk:
            total_objects += 1
            if object_meta["path"].endswith(".json"):
                raw_file_listing.append(
                    _RawFileListItem(
                        path=object_meta["path"],
                        filesize_bytes=object_meta["size"],
                    ),
                )
    log.info(f"JSON files on NGED's S3: {len(raw_file_listing)} out of {total_objects=}")
    return _process_file_listing(raw_file_listing)


def _process_file_listing(
    raw_file_listing: list[_RawFileListItem],
) -> pt.DataFrame[_ProcessedFileListing]:
    """Create DataFrame of paths.

    Extracts the start_time, end_time, and time_series_id from the path string. The input paths
    should be of the form:

    timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
    """
    paths_df = (
        pl.DataFrame(raw_file_listing)
        .with_columns(
            # Extract:    start_time,    end_time,       time_series_id
            #            ↓↓↓↓↓↓↓↓↓↓↓↓↓ ↓↓↓↓↓↓↓↓↓↓↓↓↓            ↓↓
            # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json
            regex_captures=(
                pl.col("path").str.extract_groups(
                    r"/(?<start_time>\d+)_(?<end_time>\d+)/TimeSeries_(?<time_series_id>\d{1,2})"
                )
            )
        )
        .unnest("regex_captures")
        # Convert strings to datetimes and ints:
        .with_columns(
            pl.col(["start_time", "end_time"])
            .cast(pl.Int64)
            .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
            .cast(UTC_DATETIME_DTYPE),  # Cast from time_unit="ms" to "us"
            pl.col("time_series_id").cast(pl.Int32),
        )
        .sort(by="end_time")
    )
    return _ProcessedFileListing.validate(paths_df)


def remove_small_files_from_listing(
    file_listing: pt.DataFrame[_ProcessedFileListing],
    size_threshold_bytes: int = 1000,
) -> pt.DataFrame[_ProcessedFileListing]:
    """Remove files that are too small. This is used to remove NGED JSON files that have no `data`
    field.

    Typical files sizes for NGED JSON files:
        -  486 bytes: no `data` and no `WKT`.
        - 2011 bytes: 12 timesteps in `data` and no `WKT`.
        - 4328 bytes: no `data` but a large `WKT` string.
    """
    return file_listing.filter(pl.col("filesize_bytes") > size_threshold_bytes)


class NoNewData(Exception):
    pass


def download_and_parse_files(
    store: obstore.store.S3Store, paths_df: pt.DataFrame[_ProcessedFileListing]
) -> tuple[pt.DataFrame[TimeSeriesMetadata], pt.DataFrame[PowerTimeSeries]]:
    """Load data end_time by end_time, in order, so more recent data overwrites older duplicates, if
    there are any duplicates.

    Raises NoNewData if there is no new data.
    """
    metadata_dfs = []
    power_time_series_dfs = []
    for _end_time, df_for_end_time in paths_df.group_by("end_time", maintain_order=True):
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

    if len(metadata_dfs) == 0 or len(power_time_series_dfs) == 0:
        raise NoNewData

    # Concatenate and return:
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


class TimeSeriesCoverage(pt.Model):
    """Per-series observation-time span of the ``power_time_series`` Delta table.

    ``first_time``/``last_time`` are the earliest/latest observation ``time`` for each
    ``time_series_id``. A transient intermediate (never persisted): the freshness asset check
    reads ``last_time`` to detect staleness, ``select_new_rows`` reads ``last_time`` to find
    genuinely-new rows, and CV fold-eligibility (``eligible_time_series_ids``) reads both.
    """

    time_series_id: int = _get_time_series_id_dtype(unique=True)
    first_time: int = pt.Field(dtype=PowerTimeSeries.dtypes["time"])
    last_time: int = pt.Field(dtype=PowerTimeSeries.dtypes["time"])


def time_series_coverage(
    delta_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.DataFrame[TimeSeriesCoverage]:
    """Return the earliest/latest observation ``time`` on disk per ``time_series_id``.

    Returns an empty (but correctly typed) frame if the Delta table does not exist yet.
    ``min``/``max`` grouped by ``time_series_id`` are value aggregations, so they are safe from
    the Polars 32-bit row-count wraparound even on a very large table (see CLAUDE.md).

    Cost: a full two-column scan-and-aggregate, O(rows in the table). Projection pushdown drops
    the ``power`` column, but a group-wise ``min``/``max`` cannot be answered from Parquet
    row-group statistics (no engine on our stack does aggregate-from-statistics), so every
    ``time``/``time_series_id`` value is read; computing both bounds instead of one is ~20%
    more wall-clock and no extra memory (the shared scan dominates). The ``collect`` uses the
    streaming engine to keep peak memory bounded — the freshness check runs hourly on a small
    control-plane VM. Measured on a synthetic V2 table (2,500 series, half-hourly, partitioned
    by ``time_series_id``) for a year of history (43.8M rows): streaming ~0.21 s / ~190 MB peak,
    versus ~1.3 GB peak for the in-memory engine — same result, ~7x less memory. Cost scales
    linearly with accumulated history. If the scan ever becomes a problem, both bounds can
    instead be read from the Delta add-action ``min.time``/``max.time`` file statistics —
    metadata-only, O(files): ~0.02 s / <100 MB at the same scale — the same Delta-log-metadata
    trick used to count whole-table rows without scanning.

    `delta_path` is a local path or remote URI for the ``power_time_series`` Delta table;
    `storage_options` carries the object-store credentials/endpoint for a remote `delta_path`.
    """
    if not delta_table_exists(delta_path, storage_options):
        log.info(f"{delta_path=} does not exist yet; returning an empty coverage frame.")
        empty = pl.DataFrame(
            schema={name: TimeSeriesCoverage.dtypes[name] for name in TimeSeriesCoverage.columns}
        )
        return pt.DataFrame(empty).set_model(TimeSeriesCoverage).validate()

    coverage = (
        pl.scan_delta(delta_path, storage_options=typeddict_to_dict(storage_options))
        .group_by("time_series_id")
        .agg(first_time=pl.min("time"), last_time=pl.max("time"))
        # Streaming engine: bounds peak memory (~7x lower than in-memory at V2 scale) so the
        # hourly full-table aggregate stays comfortable on a small control-plane VM. See docstring.
        .collect(engine="streaming")
    )
    log.info(
        f"Found on-disk coverage for {coverage.height} time_series_ids from {delta_path}."
        f" {coverage['last_time'].min()=}. {coverage['last_time'].max()=}"
    )
    return pt.DataFrame(coverage).set_model(TimeSeriesCoverage).validate()


# This overload tells type checkers that if you pass a `pt.DataFrame[PowerTimeSeries]` into
# `select_new_rows` then you get a `pt.DataFrame[PowerTimeSeries]` back.
@overload
def select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries],
    delta_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.DataFrame[PowerTimeSeries]: ...


# This overload tells type checkers that if you pass a `pt.DataFrame[_ProcessedFileListing]` into
# `select_new_rows` then you get a `pt.DataFrame[_ProcessedFileListing]` back.
@overload
def select_new_rows(
    time_series: pt.DataFrame[_ProcessedFileListing],
    delta_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.DataFrame[_ProcessedFileListing]: ...


def select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries] | pt.DataFrame[_ProcessedFileListing],
    delta_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.DataFrame[PowerTimeSeries] | pt.DataFrame[_ProcessedFileListing]:
    """
    Return rows in `time_series` that are more recent than the most recent
    data already in our Delta table, on a time_series_id by time_series_id basis.

    `delta_path` is a local path or remote URI for the ``power_time_series`` Delta table;
    `storage_options` carries the object-store credentials/endpoint for a remote `delta_path`.
    """

    if not delta_table_exists(delta_path, storage_options):
        log.info(f"{delta_path=} does not exist yet.")
        return time_series

    # Scan the existing delta table for the most recent time per time_series_id.
    coverage = time_series_coverage(delta_path, storage_options)

    # Check whether `time_series` is a `PowerTimeSeries` or a `_ProcessedFileListing`
    if "time" in time_series.columns:
        pt_model = PowerTimeSeries
        time_col = "time"
        columns_to_sort_by = PowerTimeSeries.columns_to_sort_by
    elif "end_time" in time_series.columns:
        pt_model = _ProcessedFileListing
        time_col = "end_time"
        columns_to_sort_by = "end_time"
    else:
        raise ValueError(
            "Expected `time_series` to have either a `time` column or an `end_time` column,"
            f" not {time_series.columns=}"
        )

    # Strip the Patito model from `coverage` so Polars' cross-subclass join check accepts it, and
    # keep only `last_time` (the most recent time on disk per series) for the new-row filter.
    plain_last_times = pl.LazyFrame._from_pyldf(coverage.lazy()._ldf).select(
        "time_series_id", "last_time"
    )
    filtered_df = (
        time_series.lazy()
        .join(plain_last_times, on="time_series_id", how="left")
        # If last_time is null for this time_series_id then this is a new time_series_id.
        .filter(pl.col("last_time").is_null() | (pl.col(time_col) > pl.col("last_time")))
        .drop("last_time")
        .sort(by=columns_to_sort_by)
        .collect()
    )

    return pt.DataFrame(filtered_df).set_model(pt_model).validate()


class UpsertMetadataStats(TypedDict, total=False):
    metadata_n_new_TimeSeriesIDs: int
    metadata_n_updated_TimeSeriesIDs: int
    metadata_updated_TimeSeriesIDs: Sequence[int]


def upsert_metadata(
    new_metadata: pt.DataFrame[TimeSeriesMetadata],
    metadata_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> UpsertMetadataStats:
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
        metadata_path: Local path or remote URI of the Parquet file where we store our version
            of the metadata.
        storage_options: Object-store credentials/endpoint for a remote `metadata_path`;
            ``None``/empty for a local path.

    Returns stats about new metadata
    """
    COMPRESSION: Final[str] = "zstd"

    new_metadata = TimeSeriesMetadata.validate(new_metadata.sort("time_series_id"))

    if not object_exists(metadata_path, storage_options):
        log.info(f"Metadata file not found at {metadata_path}. Creating new file.")
        # write_parquet doesn't create missing parent directories, so a first-ever run against a
        # fresh local data root would fail here (this create branch runs before any Delta write
        # that would otherwise create the dir). Create the parent for a local metadata_path.
        if_local_path_then_make_parent_dir(metadata_path)
        new_metadata.write_parquet(
            metadata_path,
            compression=COMPRESSION,
            storage_options=typeddict_to_dict(storage_options),
        )
        return UpsertMetadataStats(
            metadata_n_new_TimeSeriesIDs=new_metadata.height,
            metadata_n_updated_TimeSeriesIDs=0,
        )

    # Read existing metadata
    existing_metadata = pl.read_parquet(
        metadata_path, storage_options=typeddict_to_dict(storage_options)
    )
    TimeSeriesMetadata.validate(existing_metadata)

    # Compare metadata. `metadata_diff` contains all rows in `new_metadata` that do not have an
    # exact match in `existing_metadata`. Adapted from https://stackoverflow.com/a/79888719
    metadata_diff = new_metadata.filter(
        ~new_metadata.hash_rows().is_in(existing_metadata.hash_rows().implode())
    )
    TimeSeriesMetadata.validate(metadata_diff)

    if metadata_diff.is_empty():
        log.info("TimeSeriesMetadata is up to date.")
        return UpsertMetadataStats(
            metadata_n_new_TimeSeriesIDs=0,
            metadata_n_updated_TimeSeriesIDs=0,
        )

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

    merged_metadata.write_parquet(
        metadata_path, compression=COMPRESSION, storage_options=typeddict_to_dict(storage_options)
    )

    # Compute stats
    new_ids = set(new_metadata["time_series_id"]) - set(existing_metadata["time_series_id"])
    updated_ids = list(
        set(metadata_diff["time_series_id"]).intersection(existing_metadata["time_series_id"])
    )
    return UpsertMetadataStats(
        metadata_n_new_TimeSeriesIDs=len(new_ids),
        metadata_n_updated_TimeSeriesIDs=len(updated_ids),
        metadata_updated_TimeSeriesIDs=sorted(updated_ids),
    )
