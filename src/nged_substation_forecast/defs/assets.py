import ast
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Final, Generic, Self, TypeVar

import patito as pt
import polars as pl
from contracts._uri import if_local_path_then_make_parent_dir
from contracts.geo_schemas import H3GridWeights
from contracts.power_schemas import PowerTimeSeries
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import (
    ECMWF_ENS_H3_RESOLUTION,
    NwpQualityReport,
    NwpRunCompletenessReport,
    assess_nwp_quality,
    assess_nwp_run_completeness,
)
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetCheckSpec,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    RetryRequested,
    TableColumn,
    TableMetadataValue,
    TableRecord,
    TableSchema,
    asset,
)
from delta_store.nwp import write_nwp
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe,
)
from dynamical_data.ecmwf_ens.download import (
    NwpRunNotYetAvailable,
    download_ecmwf_ens_data,
    open_ecmwf_ens_run,
)
from geo.great_britain.load import load_gb_boundary
from geo.h3 import compute_h3_grid_weights_for_boundary
from nged_data.storage import (
    NoNewData,
    UpsertMetadataStats,
    _ProcessedFileListing,
    download_and_parse_files,
    list_timeseries_json_files,
    remove_small_files_from_listing,
    select_new_rows,
    upsert_metadata,
)
from pydantic import BaseModel, computed_field, field_validator


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
    delta_path = settings.power_time_series_data_path
    metadata_path = settings.metadata_path
    storage_options = settings.storage_options

    # Fetch new data from S3, using the existing delta table to determine what's new.
    # We are deliberately keeping the code simple for now, but may move the S3 store
    # to a Dagster ConfigurableResource in the future.
    store = settings.get_nged_s3_store()
    list_of_all_json_files = list_timeseries_json_files(store)
    list_of_large_json_files = remove_small_files_from_listing(list_of_all_json_files)
    list_of_new_json_files = select_new_rows(list_of_large_json_files, delta_path, storage_options)

    # Log statistics to be shown in Dagster's UI.
    context.add_output_metadata(
        _FileListingSummary.make_table(
            "nged_s3_paths",
            {
                "All JSON files on S3": list_of_all_json_files,
                "Files larger than 1kB": list_of_large_json_files,
                "Files with new data": list_of_new_json_files,
            },
        )
    )

    try:
        downloaded = download_and_parse_files(store, list_of_new_json_files)
    except NoNewData:
        context.add_output_metadata(
            UpsertMetadataStats(metadata_n_new_TimeSeriesIDs=0, metadata_n_updated_TimeSeriesIDs=0)
        )
        return
    new_metadata, new_power_ts = downloaded.metadata, downloaded.power_time_series

    if downloaded.n_implausible_power_rows_dropped > 0:
        context.log.warning(
            f"Dropped {downloaded.n_implausible_power_rows_dropped} PowerTimeSeries row(s) with a"
            " malformed `time` during ingestion (outside the plausible datetime range or not"
            " aligned to :00/:30)."
        )
    context.add_output_metadata(
        {"n_implausible_power_rows_dropped": downloaded.n_implausible_power_rows_dropped}
    )

    # Save TimeSeriesMetadata:
    upsert_metadata_stats = upsert_metadata(new_metadata, metadata_path, storage_options)
    context.add_output_metadata(upsert_metadata_stats)

    # Save PowerTimeSeries:
    new_power_ts_deduped = select_new_rows(new_power_ts, delta_path, storage_options)
    if not new_power_ts_deduped.is_empty():
        if_local_path_then_make_parent_dir(delta_path)
        new_power_ts_deduped.write_delta(
            delta_path,
            mode="append",
            storage_options=typeddict_to_dict(storage_options),
            delta_write_options={"partition_by": "time_series_id"},
        )

    # Log statistics to be shown in Dagster's UI.
    context.add_output_metadata(
        _PowerTimeSeriesSummary.make_table(
            "PowerTimeSeries",
            {
                "Downloaded timeseries": new_power_ts,
                "De-duped rows appended to disk": new_power_ts_deduped,
            },
        )
    )


@asset
def h3_grid_weights(context: AssetExecutionContext) -> None:
    """
    Computes H3 grid weights for the Great Britain boundary.

    This asset calculates the fractional overlap of H3 cells with the GB boundary
    at various resolutions, which is used for spatial aggregation of weather data.
    """
    settings = Settings()
    boundary = load_gb_boundary()
    weights = compute_h3_grid_weights_for_boundary(
        boundary, nwp_grid_size_degrees=0.25, h3_res=ECMWF_ENS_H3_RESOLUTION
    )

    # Save to parquet
    h3_grid_weights_path = settings.h3_grid_weights_path
    if_local_path_then_make_parent_dir(h3_grid_weights_path)
    weights.write_parquet(
        h3_grid_weights_path, storage_options=typeddict_to_dict(settings.storage_options)
    )

    # Add metadata to Dagster context
    context.add_output_metadata(
        {
            "n_rows": len(weights),
            "path": h3_grid_weights_path,
        }
    )


ecmwf_ens_partitions = DailyPartitionsDefinition(
    start_date="2024-04-01", timezone="UTC", end_offset=1
)
"""One partition per day of ECMWF ENS 00Z runs. ``end_offset=1`` makes today's key exist before
its 00Z run has actually landed, matching Dynamical's publication lag; shared with
``ecmwf_ens_job``/``ecmwf_ens_schedule`` in ``defs/schedules.py``."""

_ECMWF_ENS_MAX_RETRIES: Final[int] = 8
"""Retries × ``_ECMWF_ENS_RETRY_DELAY_SECONDS`` ≈ 4h of coverage past the 08:30 UTC schedule
(``ecmwf_ens_schedule``), comfortably past Dynamical's typical publication time. Only applies to
``NwpRunNotYetAvailable``; a genuine bug fails immediately instead of retrying for hours."""

_ECMWF_ENS_RETRY_DELAY_SECONDS: Final[int] = 1800
"""How long to wait between retries of a not-yet-published ECMWF run."""

_NWP_QUALITY_CHECK_NAME: Final[str] = "nwp_has_no_unexpected_nulls"
"""Name of the per-run NWP data-quality check emitted by ``ecmwf_ens`` (see ``assess_nwp_quality``).

Computed in-asset from the frame already in memory rather than as a standalone ``@asset_check``
scanning the Delta table: the quality of a run is a property of the specific ingest we are holding,
so there is nothing to re-scan (and re-scanning the whole ~5.9B-row NWP table would hit Polars'
2**32 row-count ceiling). This differs from ``power_data_is_fresh``, whose freshness genuinely
drifts over time and so must re-read the table on a schedule."""

_NWP_COMPLETENESS_CHECK_NAME: Final[str] = "nwp_run_is_complete"
"""Name of the per-run NWP completeness check emitted by ``ecmwf_ens`` (see
``assess_nwp_run_completeness``).

Separate from ``nwp_has_no_unexpected_nulls`` because the two answer different questions with
different remedies: that one asks whether the rows we got are usable, this one asks whether we got
all the rows. Computed in-asset from the frame in memory, for the same reason."""

_NWP_NULL_SLICES_SCHEMA: Final[TableSchema] = TableSchema(
    columns=[
        TableColumn("variable", "string"),
        TableColumn("ensemble_member", "int"),
        TableColumn("valid_time", "string"),
        TableColumn("n_null_cells", "int"),
        TableColumn("n_total_cells", "int"),
    ]
)
"""Fixed schema for the affected-slices metadata table (so an empty table still renders)."""


@asset(
    partitions_def=ecmwf_ens_partitions,
    deps=["h3_grid_weights"],
    check_specs=[
        AssetCheckSpec(name=_NWP_QUALITY_CHECK_NAME, asset="ecmwf_ens", blocking=False),
        AssetCheckSpec(name=_NWP_COMPLETENESS_CHECK_NAME, asset="ecmwf_ens", blocking=False),
    ],
    # The `pool="ECMWF"` works in conjunction with the Dagster instance configuration
    # (e.g., in `dagster.yaml`) to limit the number of times this asset can be run
    # concurrently. This is crucial because downloading ECMWF data is memory-intensive.
    # See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
    pool="ECMWF",
)
def ecmwf_ens(context: AssetExecutionContext) -> MaterializeResult:
    """
    Downloads and processes ECMWF ensemble NWP data for a specific day.

    This asset fetches the 00Z NWP run for the partition date, converts it to a
    Polars DataFrame, and appends it to the Delta table through
    ``delta_store.nwp.write_nwp`` (Float32, significand-rounded).
    """
    settings = Settings()
    storage_options = settings.storage_options
    partition_date_str = context.partition_key
    nwp_init_time = datetime.strptime(partition_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    # Load dependencies
    h3_grid = pt.DataFrame(
        pl.read_parquet(
            settings.h3_grid_weights_path, storage_options=typeddict_to_dict(storage_options)
        )
    ).set_model(H3GridWeights)

    # Download and convert
    try:
        ds_lazy = open_ecmwf_ens_run(nwp_init_time=nwp_init_time, h3_grid=h3_grid)
    except NwpRunNotYetAvailable as exc:
        raise RetryRequested(
            max_retries=_ECMWF_ENS_MAX_RETRIES, seconds_to_wait=_ECMWF_ENS_RETRY_DELAY_SECONDS
        ) from exc
    context.log.info("Lazily opened Icechunk store.")

    ds = download_ecmwf_ens_data(ds_lazy)
    context.log.info("Downloaded Icechunk data.")

    nwp = convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)
    context.log.info(f"Converted NWP data to Polars. Columns: {nwp.columns}")

    nwp_data_path = settings.nwp_data_path
    if_local_path_then_make_parent_dir(nwp_data_path)
    write_nwp(nwp, nwp_data_path, storage_options)
    context.log.info(f"Saved NWP data to Delta table at {nwp_data_path}.")

    # Two non-fatal per-run checks. The first surfaces the tolerated scattered nulls (known upstream
    # ECMWF ENS corruption) that Nwp.validate deliberately let through. The second asks whether the
    # run is *whole*; a short run is the upstream provider misbehaving, so we keep the rows that did
    # arrive and WARN rather than discarding the run.
    quality = assess_nwp_quality(nwp)
    completeness = assess_nwp_run_completeness(
        nwp, expected_n_h3_cells=h3_grid["h3_index"].n_unique()
    )
    return MaterializeResult(
        metadata={
            "n_rows": len(nwp),
            "path": nwp_data_path,
            "init_time": str(nwp_init_time),
            **_nwp_run_shape_metadata(completeness),
        },
        check_results=[
            _nwp_quality_check_result(quality),
            _nwp_completeness_check_result(completeness),
        ],
    )


def _nwp_run_shape_metadata(report: NwpRunCompletenessReport) -> dict[str, MetadataValue]:
    """The run's observed shape, published on *every* materialisation so drift is visible in the
    Dagster UI timeline even on the runs where the completeness check passes."""
    return {
        "n_ensemble_members": MetadataValue.int(report.n_ensemble_members),
        "n_valid_times": MetadataValue.int(report.n_valid_times),
        "n_h3_cells": MetadataValue.int(report.n_h3_cells),
        # Text, not MetadataValue.timestamp: an empty run has no valid_time at all, and a key whose
        # metadata *type* changed between runs would break the Dagster UI's timeline plot.
        "valid_time_min": MetadataValue.text(_or_na(report.valid_time_min)),
        "valid_time_max": MetadataValue.text(_or_na(report.valid_time_max)),
    }


def _or_na(value: datetime | None) -> str:
    """Render an optional datetime for Dagster metadata, matching ``_BaseSummary``'s ``"N/A"``."""
    return "N/A" if value is None else str(value)


def _nwp_completeness_check_result(report: NwpRunCompletenessReport) -> AssetCheckResult:
    """Wrap an :class:`NwpRunCompletenessReport` into a WARN-severity Dagster check result."""
    return AssetCheckResult(
        check_name=_NWP_COMPLETENESS_CHECK_NAME,
        # WARN, never fail: an incomplete upstream run is absent input, not malformed input, and
        # partial NWP still forecasts far better than yesterday's run would. See
        # https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/.
        passed=report.is_complete,
        severity=AssetCheckSeverity.WARN,
        description=report.describe(),
        metadata={
            "n_ensemble_members": report.n_ensemble_members,
            "n_valid_times": report.n_valid_times,
            "n_h3_cells": report.n_h3_cells,
            "expected_n_h3_cells": report.expected_n_h3_cells,
            "n_rows": report.n_rows,
            "expected_n_rows": report.expected_n_rows,
            "missing_ensemble_members": list(report.missing_ensemble_members),
            "missing_lead_time_hours": list(report.missing_lead_time_hours),
            "h3_cell_shortfall": report.h3_cell_shortfall,
        },
    )


def _nwp_quality_check_result(report: NwpQualityReport) -> AssetCheckResult:
    """Wrap an :class:`NwpQualityReport` into a WARN-severity Dagster check result."""
    if report.is_healthy:
        description = "No unexpected nulls in the de-accumulated NWP variables."
    else:
        variables = ", ".join(report.affected_variables)
        description = (
            f"{report.n_null_cells} scattered null cell(s) beyond lead-0 in {variables} — known "
            "upstream ECMWF ENS corruption, tolerated. See "
            "https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/."
        )
    return AssetCheckResult(
        check_name=_NWP_QUALITY_CHECK_NAME,
        # WARN, never fail: the scatter is expected upstream corruption we deliberately ingest.
        passed=report.is_healthy,
        severity=AssetCheckSeverity.WARN,
        description=description,
        metadata={
            "n_scattered_null_cells": report.n_null_cells,
            "n_affected_slices": report.n_affected_slices,
            "affected_variables": list(report.affected_variables),
            "affected_slices": _nwp_null_slices_metadata(report.scattered),
        },
    )


_NWP_NULL_SLICES_TABLE_LIMIT: Final[int] = 100
"""Cap on rows rendered in the affected-slices metadata table.

A broadly-corrupt upstream run could touch thousands of (variable, member, valid_time) slices;
the exact totals live in the scalar metadata, so the table only needs the worst offenders to be
useful — bounding it keeps the Dagster event log from bloating on a bad day."""


def _nwp_null_slices_metadata(scattered: pl.DataFrame) -> TableMetadataValue:
    """Render the worst affected (variable, member, valid_time) slices as a Dagster metadata table.

    Capped at ``_NWP_NULL_SLICES_TABLE_LIMIT`` rows (most-null first); the full counts are in the
    scalar metadata alongside.
    """
    top = scattered.sort("n_null", descending=True).head(_NWP_NULL_SLICES_TABLE_LIMIT)
    records = [
        TableRecord(
            {
                "variable": row["variable"],
                "ensemble_member": row["ensemble_member"],
                "valid_time": str(row["valid_time"]),
                "n_null_cells": row["n_null"],
                "n_total_cells": row["n_total"],
            }
        )
        for row in top.iter_rows(named=True)
    ]
    return MetadataValue.table(records, schema=_NWP_NULL_SLICES_SCHEMA)


##############################################################################
# All the code below this line is just for outputting summary stats to Dagster
# TODO: Move the code below this line to a separate file.


T = TypeVar("T", bound=pt.Model)


class _BaseSummary(ABC, BaseModel, Generic[T]):
    """Create a Dagster table of summary statistics.

    The Generic[T] makes this superclass generic over pt.Models."""

    stage: str
    start_time: str = "N/A"
    end_time: str = "N/A"
    time_series_ids: str = "N/A"  # str representation of a list of ints

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def datetime_to_string(cls, v: Any) -> Any:
        return v.strftime("%Y-%m-%d %H:%M") if isinstance(v, datetime) else v

    @field_validator("time_series_ids", mode="before")
    @classmethod
    def unique_time_series_ids(cls, v: Any) -> Any:
        return str(v.unique().sort().to_list()) if isinstance(v, pl.Series) else v

    @computed_field
    @property
    def n_time_series_ids(self) -> int:
        return 0 if self.time_series_ids == "N/A" else len(ast.literal_eval(self.time_series_ids))

    @classmethod
    def make_table(
        cls, key: str, dataframes: dict[str, pt.DataFrame[T]]
    ) -> dict[str, TableMetadataValue]:
        table: list[TableRecord] = []
        for stage_name, df in dataframes.items():
            summary = cls.from_data_frame(stage_name, df)
            table_record = TableRecord(summary.model_dump())
            table.append(table_record)
        return {key: MetadataValue.table(table)}

    @classmethod
    @abstractmethod
    def from_data_frame(cls, stage_name: str, df: pt.DataFrame[T]) -> Self:
        pass


class _FileListingSummary(_BaseSummary[_ProcessedFileListing]):
    n_files: int
    min_file_size_bytes: int = 0
    max_file_size_bytes: int = 0

    @classmethod
    def from_data_frame(cls, stage_name: str, df: pt.DataFrame[_ProcessedFileListing]) -> Self:
        # The `ty: ignore` comments are because `ty` only looks at the types specified in the BaseModel.
        # `ty` doesn't know that we're casting the types in the `field_validator` methods.
        if len(df) > 0:
            return cls(
                stage=stage_name,
                n_files=len(df),
                start_time=df["start_time"].min(),
                end_time=df["end_time"].max(),
                # TODO: We can't list *all* time_series_ids when we're handling 1,000s of IDs!
                time_series_ids=df["time_series_id"],
                min_file_size_bytes=df["filesize_bytes"].min(),  # ty: ignore[invalid-argument-type]
                max_file_size_bytes=df["filesize_bytes"].max(),  # ty: ignore[invalid-argument-type]
            )
        else:
            return cls(stage=stage_name, n_files=0)


class _PowerTimeSeriesSummary(_BaseSummary[PowerTimeSeries]):
    n_rows: int

    @classmethod
    def from_data_frame(cls, stage_name: str, df: pt.DataFrame[PowerTimeSeries]) -> Self:
        if len(df) > 0:
            return cls(
                stage=stage_name,
                n_rows=len(df),
                start_time=df["time"].min(),
                end_time=df["time"].max(),
                # TODO: We can't list *all* time_series_ids when we're handling 1,000s of IDs!
                time_series_ids=df["time_series_id"],
            )
        else:
            return cls(stage=stage_name, n_rows=0)
