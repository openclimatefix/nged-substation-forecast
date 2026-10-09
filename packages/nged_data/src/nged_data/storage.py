"""Listing, downloading, and parsing NGED's telemetry JSON from S3.

Each file yields `TimeSeriesMetadata` describing one series and `PowerTimeSeries` power
observations from it. The metadata is upserted here, into a Parquet metadata table (see
`upsert_metadata`). The power observations are not. This module never writes `PowerTimeSeries`
rows to disk. It returns them to the caller, and the caller appends them to the
`power_time_series` Delta table. `select_new_rows` reads that same Delta table, to drop rows
already on disk, but does not write to it.

The module downloads each NGED file once. The downloaded-files list (`read_downloaded_files`,
`write_downloaded_files`) records the bucket listing that the ingest last processed in full, and
`select_files_not_yet_downloaded` keeps only the listed files that the downloaded-files list lacks.
"""

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

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
    is_remote_uri,
    object_exists,
)

from nged_data.read_nged_json import (
    ExtractedPowerTimeSeries,
    NoReadingsInFile,
    _extract_power_time_series,
    _extract_time_series_metadata,
)

log = logging.getLogger(__name__)


class _RawFileListItem(TypedDict):
    path: str
    filesize_bytes: int
    last_modified: datetime


class _ProcessedFileListing(pt.Model):
    path: str
    filesize_bytes: int
    last_modified: int = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "When NGED last wrote the object, from the bucket listing. A rewritten key keeps its"
            " path and changes this value."
        ),
    )
    time_series_id: int = _get_time_series_id_dtype()
    start_time: int = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The start of the time window recorded by the time series data in the JSON file,"
            " parsed from the millisecond Unix timestamp encoded in the path"
        ),
    )
    end_time: int = pt.Field(
        dtype=UTC_DATETIME_DTYPE,
        description=(
            "The end of the time window recorded by the time series data in the JSON file,"
            " parsed from the millisecond Unix timestamp encoded in the path"
        ),
    )


def list_timeseries_json_files(
    store: obstore.store.S3Store,
) -> pt.DataFrame[_ProcessedFileListing]:
    """List all the timeseries JSON files in NGED's S3 bucket.

    Each path encodes `start_time`, `end_time`, and `time_series_id`, and is assumed to be of the
    form:

        timeseries/1774512000000_1774533600000/TimeSeries_23_20260326T080000Z_20260326T140000Z.json

    `_process_file_listing` parses those three fields back out, and its aligned comment traces the
    regex against this same key.

    A key that does not match yields null captures and fails `_ProcessedFileListing.validate`. That
    validation failure aborts the listing for the whole bucket rather than skipping the one object.
    Aborting the whole listing is deliberately stricter than the null-`data` handling in
    `download_and_parse_files`, which logs the offending file and carries on. The difference is
    where the fault lies. A malformed reading originates upstream of our pipeline, at the meter or
    in the telemetry export. A key we cannot parse means NGED's naming convention has changed. Every
    `start_time`, `end_time`, and `time_series_id` this function returns is then suspect — including
    the values parsed from the keys that still match.
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
                        last_modified=object_meta["last_modified"],
                    ),
                )
    log.info(f"JSON files on NGED's S3: {len(raw_file_listing)} out of {total_objects=}")
    return _process_file_listing(raw_file_listing)


def _process_file_listing(
    raw_file_listing: list[_RawFileListItem],
) -> pt.DataFrame[_ProcessedFileListing]:
    """Create DataFrame of paths.

    Extracts `start_time`, `end_time`, and `time_series_id` from each path string, whose format
    `list_timeseries_json_files` documents. The aligned comment below traces the regex against a
    real key.
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


class _DownloadedFiles(pt.Model):
    """The two columns of a stored downloaded-files list.

    The list stores only these columns, so a later change to `_ProcessedFileListing`'s other
    columns cannot invalidate a list already on disk.
    """

    path: str
    last_modified: int = pt.Field(dtype=UTC_DATETIME_DTYPE)


class DownloadedFilesError(Exception):
    """Raised when the stored downloaded-files list cannot be read or fails validation.

    The ingest must stop on this error. Reading a damaged list as an empty one would download
    every file in NGED's bucket.
    """


def _empty_downloaded_files() -> pt.DataFrame[_DownloadedFiles]:
    empty = pl.DataFrame(
        schema={name: _DownloadedFiles.dtypes[name] for name in _DownloadedFiles.columns}
    )
    return pt.DataFrame(empty).set_model(_DownloadedFiles).validate()


def read_downloaded_files(
    downloaded_files_path: str,
    power_table_path: str,
    metadata_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.DataFrame[_DownloadedFiles]:
    """Read the bucket listing that the ingest last processed in full.

    The list describes files loaded into the `power_time_series` Delta table and the metadata
    parquet. If an operator moves either aside to rebuild it, the list would claim that files are
    loaded which the rebuilt state lacks. The list therefore counts as empty, and the next run
    downloads every file, whenever the list file, the power table, or the metadata parquet does not
    exist.

    Args:
        downloaded_files_path: Local path or remote URI of the downloaded-files parquet file.
        power_table_path: Local path or remote URI of the `power_time_series` Delta table.
        metadata_path: Local path or remote URI of the metadata parquet file.
        storage_options: Object-store credentials/endpoint for remote paths; ``None``/empty for
            local paths.

    Returns:
        The stored `path` and `last_modified` of each file, or an empty frame of the same type.

    Raises:
        DownloadedFilesError: if the list file exists but cannot be read or fails validation. A
            transient object-store error from an existence check is not wrapped, so the caller's
            retry guard can retry it.
    """
    if not (
        object_exists(downloaded_files_path, storage_options)
        and delta_table_exists(power_table_path, storage_options)
        and object_exists(metadata_path, storage_options)
    ):
        log.info(f"No usable downloaded-files list at {downloaded_files_path}; using an empty one.")
        return _empty_downloaded_files()
    try:
        stored = pl.read_parquet(
            downloaded_files_path, storage_options=typeddict_to_dict(storage_options)
        )
        return pt.DataFrame(_DownloadedFiles.validate(stored)).set_model(_DownloadedFiles)
    except (pl.exceptions.PolarsError, pt.exceptions.DataFrameValidationError, OSError) as exc:
        raise DownloadedFilesError(
            f"Could not read the downloaded-files list at {downloaded_files_path}. Delete the"
            " file to download every file in NGED's bucket again."
        ) from exc


def write_downloaded_files(
    downloaded_files_path: str,
    listing: pt.DataFrame[_ProcessedFileListing],
    storage_options: ObjectStoreOptions | None = None,
) -> None:
    """Replace the downloaded-files list with `listing`'s `path` and `last_modified` columns.

    A local write goes through a temporary file and `Path.replace`, so a killed process cannot leave
    a torn list. A torn list would stop the ingest, whereas a torn metadata parquet does not. An
    object-store write replaces the object in one request.

    Args:
        downloaded_files_path: Local path or remote URI of the downloaded-files parquet file.
        listing: The whole bucket listing that the run processed.
        storage_options: Object-store credentials/endpoint for a remote path; ``None``/empty for a
            local path.
    """
    downloaded_files = _DownloadedFiles.validate(
        listing.select("path", "last_modified").sort("path")
    )
    options = typeddict_to_dict(storage_options)
    if is_remote_uri(downloaded_files_path):
        downloaded_files.write_parquet(
            downloaded_files_path, compression="zstd", storage_options=options
        )
        return
    if_local_path_then_make_parent_dir(downloaded_files_path)
    temporary_path = f"{downloaded_files_path}.tmp"
    downloaded_files.write_parquet(temporary_path, compression="zstd")
    Path(temporary_path).replace(downloaded_files_path)


def select_files_not_yet_downloaded(
    file_listing: pt.DataFrame[_ProcessedFileListing],
    downloaded_files: pt.DataFrame[_DownloadedFiles],
) -> pt.DataFrame[_ProcessedFileListing]:
    """Keep the listed files whose `(path, last_modified)` the downloaded-files list lacks.

    A file whose path is known but whose `last_modified` changed was rewritten by NGED, so it is
    selected again. No time margin is needed: a file absent from an earlier listing is absent from
    the list, so a later listing selects it.

    Args:
        file_listing: The whole bucket listing.
        downloaded_files: The listing that the ingest last processed in full.

    Returns:
        The selected files in ascending `end_time` order.
    """
    # Strip the Patito model so Polars' cross-subclass join check accepts the right-hand frame.
    plain_downloaded_files = pl.DataFrame._from_pydf(downloaded_files._df)
    selected = file_listing.join(
        plain_downloaded_files, on=["path", "last_modified"], how="anti"
    ).sort("end_time", "path")
    return pt.DataFrame(selected).set_model(_ProcessedFileListing).validate()


class DownloadAndParseResult(NamedTuple):
    """Result of ``download_and_parse_files``.

    ``n_implausible_power_rows_dropped`` sums ``ExtractedPowerTimeSeries.n_dropped`` across every
    file in the batch — see ``PowerTimeSeries.drop_implausible_rows`` for what gets dropped and
    why.
    """

    metadata: pt.DataFrame[TimeSeriesMetadata]
    power_time_series: pt.DataFrame[PowerTimeSeries]
    n_implausible_power_rows_dropped: int


_DOWNLOAD_CHUNK_FILES: Final[int] = 500
"""How many files `download_and_parse_files` fetches before parsing them.

Chunks bound the raw JSON held in memory, because a full download fetches tens of thousands of
files."""

_MAX_REQUESTS_IN_FLIGHT: Final[int] = 32
"""How many requests `download_and_parse_files` has open on NGED's bucket at once."""


async def _fetch_all(store: obstore.store.S3Store, paths: Sequence[str]) -> list[bytes]:
    """Fetch every path concurrently, returning the bytes in the order of `paths`.

    The semaphore is created here, inside the coroutine, because a semaphore binds to the event
    loop that first uses it and each `asyncio.run` call starts a new loop.
    """
    semaphore = asyncio.Semaphore(_MAX_REQUESTS_IN_FLIGHT)

    async def fetch(path: str) -> bytes:
        async with semaphore:
            result = await store.get_async(path)
            return bytes(await result.bytes_async())

    return list(await asyncio.gather(*(fetch(path) for path in paths)))


def _parse_file(
    path: str, json_bytes: bytes
) -> tuple[pt.DataFrame[TimeSeriesMetadata], ExtractedPowerTimeSeries | None]:
    """Parse one NGED JSON file into its metadata and, if it has readings, its power rows.

    Args:
        path: The file's key in NGED's bucket, named in the error if parsing fails.
        json_bytes: The file's contents.

    Returns:
        The file's `TimeSeriesMetadata`, and its readings, or ``None`` when the `data` field is
        null or empty.

    Raises:
        ValueError: if the file is malformed or breaks the `TimeSeriesMetadata` contract. The
            message names `path`.
    """
    try:
        df = pl.read_json(json_bytes)
        metadata = _extract_time_series_metadata(df)
        time_series_id: int = metadata["time_series_id"].item()
        try:
            extracted = _extract_power_time_series(df=df, time_series_id=time_series_id)
        except NoReadingsInFile:
            log.warning(
                f"The 'data' field is null or empty in {path=}. This is expected behaviour if"
                " NGED's meter reported no values for the period covered by the JSON file."
            )
            return metadata, None
    except Exception as exc:
        raise ValueError(f"Could not parse the NGED file {path}") from exc
    return metadata, extracted


def download_and_parse_files(
    store: obstore.store.S3Store, paths_df: pt.DataFrame[_ProcessedFileListing]
) -> DownloadAndParseResult:
    """Download and parse each listed file, in ascending `end_time` order.

    Two files can cover overlapping periods for the same `time_series_id`. The function sorts its
    input by `end_time` itself, because an anti-join does not keep the listing's order. Processing
    in that order means the more recent file's readings overwrite the older file's duplicate rows,
    in the `unique(..., keep="last")` dedupes below.

    The function works through the sorted input in chunks of `_DOWNLOAD_CHUNK_FILES`. Per chunk it
    fetches all the files concurrently, at most `_MAX_REQUESTS_IN_FLIGHT` at a time, through
    `asyncio.run`, then parses them sequentially in input order. The function is synchronous and
    needs a thread with no running event loop, which a synchronous Dagster asset has.

    Args:
        store: The NGED S3 bucket to download each file from.
        paths_df: The files to process, typically the output of `select_files_not_yet_downloaded`.

    Returns:
        A `DownloadAndParseResult` bundling every file's parsed `TimeSeriesMetadata` and
        `PowerTimeSeries` rows, deduplicated across files — see `DownloadAndParseResult`'s
        docstring for what each field holds. When every file was data-less (`NoReadingsInFile`),
        the metadata is still returned and the power frame is empty.

    Raises:
        ValueError: if a file is malformed or breaks the `TimeSeriesMetadata` contract. The message
            names the file's path. The first failing request also raises, out of `asyncio.run`.
    """
    paths = paths_df.sort("end_time", "path")["path"].to_list()
    metadata_dfs: list[pt.DataFrame[TimeSeriesMetadata]] = []
    power_time_series_dfs: list[pt.DataFrame[PowerTimeSeries]] = []
    n_implausible_power_rows_dropped = 0
    for chunk_start in range(0, len(paths), _DOWNLOAD_CHUNK_FILES):
        chunk_paths = paths[chunk_start : chunk_start + _DOWNLOAD_CHUNK_FILES]
        chunk_bytes = asyncio.run(_fetch_all(store, chunk_paths))
        for path, json_bytes in zip(chunk_paths, chunk_bytes, strict=True):
            metadata, extracted = _parse_file(path, json_bytes)
            metadata_dfs.append(metadata)
            if extracted is not None:
                power_time_series_dfs.append(extracted.dataframe)
                n_implausible_power_rows_dropped += extracted.n_dropped

    log.info(
        f"{len(metadata_dfs)} new TimeSeriesMetadata DataFrames and {len(power_time_series_dfs)}"
        " new PowerTimeSeries dataframes extracted from NGED JSON data."
    )

    metadata_df = (
        pl.concat(metadata_dfs, how="diagonal")
        .unique(subset="time_series_id", keep="last")
        .sort("time_series_id")
    )
    if power_time_series_dfs:
        time_series_df = (
            pl.concat(power_time_series_dfs)
            .unique(subset=["time_series_id", "time"], keep="last")
            .sort(by=PowerTimeSeries.columns_to_sort_by)
        )
    else:
        time_series_df = pl.DataFrame(
            schema={name: PowerTimeSeries.dtypes[name] for name in PowerTimeSeries.columns}
        )

    return DownloadAndParseResult(
        metadata=TimeSeriesMetadata.validate(metadata_df),
        power_time_series=PowerTimeSeries.validate(time_series_df),
        n_implausible_power_rows_dropped=n_implausible_power_rows_dropped,
    )


class TimeSeriesCoverage(pt.Model):
    """Per-series observation-time span of the ``power_time_series`` Delta table.

    ``first_time``/``last_time`` are the earliest/latest observation ``time`` for each
    ``time_series_id``. This frame is a transient intermediate, never persisted. Two callers
    read the frame: the freshness asset check reads ``last_time`` to detect staleness, and
    cross-validation (CV) fold-eligibility (``eligible_time_series_ids``) reads both
    ``first_time`` and ``last_time``. A CV fold is one train/test split of the history, and a
    series is eligible for a fold only if its data covers that split.

    The freshness check reads this on-disk recency rather than the asset's materialisation
    timestamp. A materialisation-freshness policy would miss the failure the check exists to
    catch. When NGED's telemetry stalls, the ingest asset keeps materialising successfully on
    schedule, and writes nothing. The materialisation looks fresh; the newest observation on disk
    does not. Full reasoning:
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#warn-on-stale-power-data-with-a-dagster-asset-check>.
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
    the ``power`` column. A group-wise ``min``/``max`` cannot be answered from Parquet row-group
    statistics, because no engine on our stack does aggregate-from-statistics, so every
    ``time``/``time_series_id`` value is read. Computing both bounds instead of one takes ~20%
    more wall-clock time and no extra memory, because the shared scan dominates.

    The ``collect`` uses the streaming engine to keep peak memory bounded, because this scan runs
    hourly on a small control-plane VM, in the ``power_data_is_fresh`` asset check. The ingest does
    not run it: ``select_new_rows`` uses ``_existing_power_time_series_keys`` instead, a scan
    restricted to the reporting series' own history. The measurement used a synthetic V2 table:
    2,500 series, half-hourly,
    partitioned by ``time_series_id``, holding a year of history (43.8M rows). The streaming
    engine took ~0.21 s at ~190 MB peak. The in-memory engine peaked at ~1.3 GB for the same
    result, so streaming uses ~7x less memory.

    Cost scales linearly with accumulated history. If the scan ever becomes a problem, both
    bounds can instead be read from the Delta add-action ``min.time``/``max.time`` file
    statistics. Delta's transaction log records one add-action per data file, carrying that
    file's per-column minimum and maximum. That read is metadata-only, O(files): ~0.02 s and <100
    MB at the same scale. The statistics read is the same Delta-log-metadata trick used to count
    whole-table rows without scanning.

    `delta_path` is a local path or remote URI for the ``power_time_series`` Delta table;
    `storage_options` carries the object-store credentials/endpoint for a remote `delta_path`.
    """
    if not delta_table_exists(delta_path, storage_options):
        log.info(f"{delta_path=} does not exist yet; returning an empty coverage frame.")
        empty = pl.DataFrame(
            schema={name: TimeSeriesCoverage.dtypes[name] for name in TimeSeriesCoverage.columns}
        )
        return pt.DataFrame(empty).set_model(TimeSeriesCoverage).validate()

    return coverage_from_power(
        pt.LazyFrame.from_existing(
            pl.scan_delta(delta_path, storage_options=typeddict_to_dict(storage_options))
        ).set_model(PowerTimeSeries)
    )


def coverage_from_power(power: pt.LazyFrame[PowerTimeSeries]) -> pt.DataFrame[TimeSeriesCoverage]:
    """Return the earliest/latest observation ``time`` per ``time_series_id`` of a power frame.

    The aggregation behind `time_series_coverage`, split out so a caller can compute coverage of
    any scan of power, for example the unflagged rows of the cleaned table (`scan_cleaned_power`).

    Args:
        power: Lazy power observations; only `time_series_id` and `time` are read.

    Returns:
        One row per `time_series_id`, validated against `TimeSeriesCoverage`.
    """
    coverage = (
        pl.LazyFrame._from_pyldf(power._ldf)
        .group_by("time_series_id")
        .agg(first_time=pl.min("time"), last_time=pl.max("time"))
        # Streaming engine: bounds peak memory (~7x lower than in-memory at V2 scale) so the
        # hourly full-table aggregate stays comfortable on a small control-plane VM. See
        # `time_series_coverage`'s docstring.
        .collect(engine="streaming")
    )
    log.info(
        f"Found on-disk coverage for {coverage.height} time_series_ids."
        f" {coverage['last_time'].min()=}. {coverage['last_time'].max()=}"
    )
    return pt.DataFrame(coverage).set_model(TimeSeriesCoverage).validate()


def scan_cleaned_power(
    delta_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.LazyFrame[PowerTimeSeries]:
    """Scan the cleaned power table, keeping only the rows that no cleaning rule flagged.

    `scan_cleaned_power` is the one read path for every consumer of observed power except the ingest
    and its freshness check, so no consumer can forget the `drop_reason` filter. The raw table stays
    the record of what NGED delivered.

    Args:
        delta_path: Path or URI of the `cleaned_power_time_series` Delta table.
        storage_options: delta-rs object-store options for a remote `delta_path`.

    Returns:
        A lazy frame with the three `PowerTimeSeries` columns, whose `drop_reason` was null.
    """
    scan = pl.scan_delta(delta_path, storage_options=typeddict_to_dict(storage_options))
    unflagged = scan.filter(pl.col("drop_reason").is_null()).select(*PowerTimeSeries.columns)
    return pt.LazyFrame.from_existing(unflagged).set_model(PowerTimeSeries)


def _existing_power_time_series_keys(
    delta_path: str,
    storage_options: ObjectStoreOptions | None,
    time_series_ids: Sequence[int],
) -> pl.LazyFrame:
    """Return the `(time_series_id, time)` pairs already on disk for `time_series_ids`.

    `time_series_ids` holds the series present in the candidate frame `select_new_rows` is
    filtering. Restricting the scan to those series keeps the scan partition-pruned, reading just
    the reporting series' own history. `power_time_series` is partitioned by `time_series_id`
    (see `delta_store.power_time_series.write_power_time_series`). An unrestricted version of
    this scan would instead materialise every row in the table, to build the join's hash table.
    That whole-table materialisation is what `time_series_coverage`'s streaming aggregate avoids.
    The only caller, `select_new_rows`, already returns early via `delta_table_exists` before
    calling `_existing_power_time_series_keys`. This function therefore needs no empty-table
    branch of its own, unlike `time_series_coverage`.

    The measurement used the same synthetic V2 table that `time_series_coverage`'s docstring
    uses: 2,500 series, half-hourly, 1 year, 43.8M rows. At 1-20 reporting series the scan took
    ~0.04 s and negligible extra memory. Those few reporting series are the expected case,
    because NGED's files land a few at a time. Peak memory then rises to ~110 MB at 100 reporting
    series, ~585 MB at 500 reporting series, and ~2.75 GB at all 2,500 series. An hour where
    nearly every series reports at once therefore uses more memory here than
    `time_series_coverage`'s own whole-table scan. Two cases produce that hour: a bulk backfill,
    and recovery from an extended NGED outage. An anti-join has to materialise the actual rows to
    hash-join against, rather than collapsing each series to two values the way an aggregate
    does.
    """
    return (
        pl.scan_delta(delta_path, storage_options=typeddict_to_dict(storage_options))
        .filter(pl.col("time_series_id").is_in(time_series_ids))
        .select("time_series_id", "time")
    )


def select_new_rows(
    time_series: pt.DataFrame[PowerTimeSeries],
    delta_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> pt.DataFrame[PowerTimeSeries]:
    """Return rows in `time_series` genuinely missing from the Delta table.

    The filter is a genuine existence check: an anti-join on `(time_series_id, time)` against
    `_existing_power_time_series_keys`. A late file is therefore ingested even when a later reading
    for the same series is already on disk. So is a file that fills a gap earlier in a series'
    history. It also makes a re-download safe, because a rewritten file or a crash between the
    append and the downloaded-files list write can offer readings the table already holds. See
    `_existing_power_time_series_keys`'s docstring for the cost the existence check trades in
    return.

    When the Delta table does not exist yet, a call returns its input unchanged and scans nothing.

    Args:
        time_series: The parsed rows to check.
        delta_path: Local path or remote URI for the ``power_time_series`` Delta table.
        storage_options: Object-store credentials/endpoint for a remote `delta_path`.

    Returns:
        The rows of `time_series` that the Delta table lacks, in `PowerTimeSeries` sort order.
    """
    if not delta_table_exists(delta_path, storage_options):
        log.info(f"{delta_path=} does not exist yet.")
        return time_series

    reporting_ids = time_series["time_series_id"].unique().to_list()
    existing_keys = _existing_power_time_series_keys(delta_path, storage_options, reporting_ids)
    filtered_df = (
        time_series.lazy()
        .join(existing_keys, on=["time_series_id", "time"], how="anti")
        .sort(by=PowerTimeSeries.columns_to_sort_by)
        .collect()
    )
    return pt.DataFrame(filtered_df).set_model(PowerTimeSeries).validate()


class UpsertMetadataStats(TypedDict, total=False):
    """What the ``TimeSeriesMetadata`` upsert did, published as Dagster output metadata."""

    metadata_n_new_TimeSeriesIDs: int
    metadata_n_updated_TimeSeriesIDs: int
    metadata_updated_TimeSeriesIDs: Sequence[int]
    metadata_upsert_failed: str
    """Set by the asset when the whole upsert raised, so the power write went ahead without it.

    Read this field's presence as "the metadata table is stale until each affected series
    publishes its next file", not as a power-ingest failure — the power write is unaffected. See
    [Degraded input data](https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/#degraded-input-data-nwp-feed-down-or-telemetry-stalled),
    under "Reading a failed metadata table upsert", for the operational read of this field and what
    a stale metadata table costs while the failure persists.
    """


def upsert_metadata(
    new_metadata: pt.DataFrame[TimeSeriesMetadata],
    metadata_path: str,
    storage_options: ObjectStoreOptions | None = None,
) -> UpsertMetadataStats:
    """Upserts metadata to a Parquet file, keeping the newest version of each time series.

    If the Parquet file does not exist, it saves the new_metadata. If it exists, it merges the
    new_metadata into it and rewrites the file only if the incoming metadata differs from what is
    stored. ``new_metadata`` is a snapshot of the metadata for the series that reported this run.
    The snapshot need not carry the same columns, or the same column order, as the stored metadata
    table. Rows are matched on ``time_series_id``. A series that ``new_metadata`` covers is replaced
    wholesale, so a field the snapshot has stopped carrying is **cleared** for that series. A series
    that ``new_metadata`` omits keeps its last stored values indefinitely. The metadata table
    therefore holds every time series we have ever seen, not only the series in the latest snapshot.

    This function is not safe under concurrent callers: it assumes it is called by one thread at a
    time, and takes no lock.

    The rewrite is not atomic either. `write_parquet` overwrites the metadata table in place, with
    no write-to-temporary-file-and-rename. The metadata table therefore does not get the
    all-or-nothing commit that Delta gives the tables around it. See [principle 10, every write is
    atomic and idempotent](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#10-every-write-is-atomic-and-idempotent-and-every-failure-is-confined-to-one-partition).
    A crash or an out-of-memory kill part-way through a local write leaves a partial file.
    `pl.read_parquet` below is what rejects that partial file on the next run, before
    `TimeSeriesMetadata.validate` ever sees it. The error reads `ComputeError: parquet: File out of
    specification: The file must end with PAR1`. `validate` is the guard for the other case: a
    metadata table that reads back cleanly but is off-contract, from an older writer or a hand-edit.
    Either way the asset records `metadata_upsert_failed`, and the metadata table stays broken until
    an operator acts. A corrupt file is not a missing file, so the create branch below never runs
    again by itself.

    Deleting the file rebuilds it. `read_downloaded_files` counts the downloaded-files list as empty
    when the metadata table does not exist, so the next run downloads every file in NGED's bucket
    and the metadata table gets every series NGED publishes.

    Args:
        new_metadata: The new metadata DataFrame.
        metadata_path: Local path or remote URI of the Parquet file where we store our version
            of the metadata.
        storage_options: Object-store credentials/endpoint for a remote `metadata_path`;
            ``None``/empty for a local path.

    Returns:
        An `UpsertMetadataStats`. `metadata_n_new_TimeSeriesIDs` counts the `time_series_id`s in
        `new_metadata` that are new to the metadata table. `metadata_n_updated_TimeSeriesIDs` counts
        the `time_series_id`s already in the metadata table that changed.
        `metadata_updated_TimeSeriesIDs` holds the sorted list of changed `time_series_id`s. Both
        counts are zero and the id list is omitted when the parquet file was up to date already. The
        id list is also omitted on a first-ever write, when every id in `new_metadata` counts as new
        rather than updated.
    """
    COMPRESSION: Final[str] = "zstd"

    # The annotation is not enforced at runtime and this is the package's only public entry point,
    # so check the caller's snapshot rather than trust it.
    new_metadata = TimeSeriesMetadata.validate(new_metadata.sort("time_series_id"))

    if not object_exists(metadata_path, storage_options):
        log.info(f"Metadata file not found at {metadata_path}. Creating new file.")
        # write_parquet doesn't create missing parent directories, so a first-ever run against a
        # fresh local data root would fail here. This create branch runs before any Delta write that
        # would otherwise create the dir. Create the parent for a local metadata_path.
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
    # The stored metadata table is outside this code's control: it can come from an older writer, a
    # hand-edit, or a truncated upload. An off-contract file must therefore not be merged blind into
    # the metadata table we write back. As with any raise from this function, the asset contains it
    # rather than failing: it records `metadata_upsert_failed` and lets the power write proceed (see
    # `defs/assets.py`).
    TimeSeriesMetadata.validate(existing_metadata)

    # `how="diagonal"` because the snapshot and the stored metadata table can differ in both width
    # and column order: four TimeSeriesMetadata fields are `allow_missing`. Aligning the two frames
    # into one also makes the `hash_rows` diff below insensitive to the stored column order. Hashing
    # the two frames separately would not be.
    combined = pl.concat([new_metadata, existing_metadata], how="diagonal")
    new_rows = combined.head(new_metadata.height)
    stored_rows = combined.slice(new_metadata.height)

    # Compare metadata. `metadata_diff` contains all rows in `new_metadata` that do not have an
    # exact match in `existing_metadata`. Adapted from https://stackoverflow.com/a/79888719
    metadata_diff = new_rows.filter(~new_rows.hash_rows().is_in(stored_rows.hash_rows().implode()))
    # The first frame carrying the union of both inputs' columns: the concat adds to the snapshot's
    # rows any `allow_missing` field only the stored metadata table had. All four fields are
    # nullable, so this validate is a shape check on a frame neither validation above saw, not a
    # guard against a known fault. Of the four validate calls in this function this is the weakest,
    # and the first to reconsider if the validate calls get trimmed.
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

    # The last gate before the stored metadata table is overwritten. `unique` draws rows from both
    # sides of the concat, so no validation above has seen this row set.
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
