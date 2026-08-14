"""Reading NGED's telemetry JSON from S3 and writing power observations and metadata to storage."""

import logging
from collections.abc import Sequence
from typing import Final, NamedTuple, TypedDict, overload

import obstore
import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE, _get_time_series_id_dtype
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.typing_utils import typeddict_to_dict
from contracts.uri import (
    ObjectStoreOptions,
    delta_table_exists,
    if_local_path_then_make_parent_dir,
    object_exists,
)

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
            # timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json  # noqa: E501 — the arrows above must stay aligned with the real key.
            regex_captures=(
                pl.col("path").str.extract_groups(
                    r"/(?<start_time>\d+)_(?<end_time>\d+)/TimeSeries_(?<time_series_id>\d+)_"
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
    size_threshold_bytes: int = 520,
) -> pt.DataFrame[_ProcessedFileListing]:
    """Remove files too small to carry any readings.

    This is used to skip NGED JSON files that have no `data` field, so `download_and_parse_files`
    never has to fetch and parse them only to discard the result. It is an optimisation, not a
    correctness requirement: `download_and_parse_files` already tolerates a null `data` field.

    `size_threshold_bytes` defaults to 520, derived from the real files on NGED's S3: a WKT-less
    file with zero readings tops out at 488 bytes, and one with a single reading starts at 556
    bytes, so 520 sits in the gap between them. WKT-bearing (Primary substation) files run far
    larger — zero-reading examples measured between 4,405 and 20,148 bytes — so the WKT-less
    floor is the binding constraint on the threshold.

    That 68-byte gap comes from V1's 33 series, and it is narrow enough that V2's ~2,500 series
    want re-measuring before this default is trusted there. Two things would close it: a populated
    `information` field, which `TimeSeriesMetadata` records as always null in the V1 trial area,
    would push a zero-reading file above 520; and a substation name shorter than any in V1 would
    pull a one-reading file below it. Re-run the measurement rather than assume the gap survives.
    """
    n_files_before_filter = file_listing.height
    filtered = file_listing.filter(pl.col("filesize_bytes") > size_threshold_bytes)
    log.info(
        f"Files retained after the size filter: {filtered.height} out of {n_files_before_filter=}"
    )
    return filtered


class NoNewData(Exception):
    """Raised when a listing of NGED files yields no rows we have not already stored."""


class DownloadAndParseResult(NamedTuple):
    """Result of ``download_and_parse_files``.

    ``n_implausible_power_rows_dropped`` sums ``ExtractedPowerTimeSeries.n_dropped`` across every
    file in the batch — see ``PowerTimeSeries.drop_implausible_rows`` for what gets dropped and
    why.
    """

    metadata: pt.DataFrame[TimeSeriesMetadata]
    power_time_series: pt.DataFrame[PowerTimeSeries]
    n_implausible_power_rows_dropped: int


def download_and_parse_files(
    store: obstore.store.S3Store, paths_df: pt.DataFrame[_ProcessedFileListing]
) -> DownloadAndParseResult:
    """Load data end_time by end_time, in order.

    Loading in order means more recent data overwrites older duplicates, if there are any.

    Raises NoNewData if there is no new data.
    """
    metadata_dfs = []
    power_time_series_dfs = []
    n_implausible_power_rows_dropped = 0
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
                extracted = _extract_power_time_series(df=df, time_series_id=time_series_id)
            except pl.exceptions.InvalidOperationError as e:
                if "invalid dtype: expected 'Struct', got 'Null' for 'data'" in str(e):
                    log.warning(
                        f"The 'data' field is 'null' in {path=}. This is expected behaviour if"
                        " NGED's meter reported no values for the period covered by the JSON file."
                    )
                else:
                    raise
            else:
                power_time_series_dfs.append(extracted.dataframe)
                n_implausible_power_rows_dropped += extracted.n_dropped

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

    return DownloadAndParseResult(
        metadata=TimeSeriesMetadata.validate(metadata_df),
        power_time_series=PowerTimeSeries.validate(time_series_df),
        n_implausible_power_rows_dropped=n_implausible_power_rows_dropped,
    )


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
    the Polars 32-bit row-count wraparound even on a very large table (see
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/#data-handling>).

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
    """Return rows in `time_series` newer than what our Delta table already holds.

    The comparison is made on a time_series_id by time_series_id basis.

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
    """What the ``TimeSeriesMetadata`` upsert did, published as Dagster output metadata."""

    metadata_n_new_TimeSeriesIDs: int
    metadata_n_updated_TimeSeriesIDs: int
    metadata_updated_TimeSeriesIDs: Sequence[int]
    metadata_upsert_failed: str
    """Set by the asset when the whole upsert raised, so the power write went ahead without it."""


def upsert_metadata(
    new_metadata: pt.DataFrame[TimeSeriesMetadata],
    metadata_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> UpsertMetadataStats:
    """Upserts metadata to a Parquet file, keeping the newest version of each time series.

    This function assumes it is called by one thread at a time so no
    explicit locking is required.

    If the Parquet file does not exist, it saves the new_metadata. If it exists, it merges the
    new_metadata into it and rewrites the file only if something changed. The snapshot need not
    carry the same columns, or the same column order, as the stored roster, and rows are matched
    on ``time_series_id``. A series that ``new_metadata`` covers is replaced wholesale, so a field
    the snapshot has stopped carrying is **cleared** for that series. A series that
    ``new_metadata`` omits keeps its last stored values indefinitely. The roster therefore holds
    every time series we have ever seen, not only the ones in the latest snapshot.

    Args:
        new_metadata: The new metadata DataFrame.
        metadata_path: Local path or remote URI of the Parquet file where we store our version
            of the metadata.
        storage_options: Object-store credentials/endpoint for a remote `metadata_path`;
            ``None``/empty for a local path.

    Returns stats about new metadata
    """
    COMPRESSION: Final[str] = "zstd"

    # The annotation is not enforced at runtime and this is the package's only public entry point,
    # so check the caller's snapshot rather than trust it.
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

    existing_metadata = pl.read_parquet(
        metadata_path, storage_options=typeddict_to_dict(storage_options)
    )
    # The stored roster is outside this code's control — an older writer, a hand-edit, a truncated
    # upload — so an off-contract file must not be merged blind into the one we write back. As with
    # any raise from this function, the asset contains it rather than failing: it records
    # `metadata_upsert_failed` and lets the power write proceed (see `defs/assets.py`).
    TimeSeriesMetadata.validate(existing_metadata)

    # `how="diagonal"` because the snapshot and the stored roster can differ in both width and
    # column order, four TimeSeriesMetadata fields being `allow_missing`. Aligning them into one
    # frame also makes the `hash_rows` diff below insensitive to the stored column order, which
    # hashing the two frames separately is not.
    combined = pl.concat([new_metadata, existing_metadata], how="diagonal")
    new_rows = combined.head(new_metadata.height)
    stored_rows = combined.slice(new_metadata.height)

    # Compare metadata. `metadata_diff` contains all rows in `new_metadata` that do not have an
    # exact match in `existing_metadata`. Adapted from https://stackoverflow.com/a/79888719
    metadata_diff = new_rows.filter(~new_rows.hash_rows().is_in(stored_rows.hash_rows().implode()))
    # The first frame carrying the union of both inputs' columns: the concat adds to the snapshot's
    # rows any `allow_missing` field only the stored roster had. All four of those are nullable, so
    # this is a shape check on a frame neither validation above saw rather than a guard against a
    # known fault — the weakest of the four, and the first to reconsider if these get trimmed.
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
    merged_metadata = combined.unique(subset="time_series_id", keep="first").sort("time_series_id")

    # The last gate before the stored roster is overwritten. `unique` draws rows from both sides of
    # the concat, so this row set is one no validation above has seen.
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
