"""The ingestion Dagster assets.

NGED telemetry and metadata, the H3 grid weights, and the daily-partitioned ECMWF ENS download.
"""

import ast
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Final, Self

import patito as pt
import polars as pl
from contracts._uri import if_local_path_then_make_parent_dir
from contracts.geo_schemas import H3GridWeights
from contracts.power_schemas import PowerTimeSeries
from contracts.settings import Settings
from contracts.typing_utils import typeddict_to_dict
from contracts.weather_schemas import (
    ECMWF_ENS_H3_RESOLUTION,
    Nwp,
    NwpQualityReport,
    NwpRunCompletenessReport,
    NwpVariableWhollyMissing,
    assess_nwp_quality,
    assess_nwp_run_completeness,
)
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetCheckSpec,
    AssetExecutionContext,
    DagsterExecutionInterruptedError,
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
    ECMWF_ENS_INSTANTANEOUS_VARS,
    NwpRunNotYetAvailable,
    download_ecmwf_ens_data,
    open_ecmwf_ens_run,
)
from dynamical_data.ecmwf_ens.upstream_nulls import (
    UpstreamNullRate,
    assess_upstream_grid_point_nulls,
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

from nged_substation_forecast._sentry import report_asset_degradation, report_check_degradation
from nged_substation_forecast.defs._tags import PRODUCTION_LAYER_TAGS

_POWER_INGEST_MAX_RETRIES: Final[int] = 2
"""How many times to retry the NGED S3 read before letting the hourly ingest fail.

Deliberately small. The retry exists only to stop a transient object-store error from reporting a
fault that has already fixed itself; the data is never at risk, because this asset is unpartitioned,
re-lists NGED's bucket from scratch on every attempt, and the next hourly run back-fills whatever
this one missed. A longer budget would buy nothing the next hour does not already buy, and each
retry re-runs the whole listing, download and parse — which at V2 scale costs far more than the
delay does. A persistent outage still fails after the budget and still reports."""

_POWER_INGEST_RETRY_DELAY_SECONDS: Final[int] = 2
"""How long to wait between retries of a transient NGED S3 failure."""


@asset(tags=PRODUCTION_LAYER_TAGS)
def power_time_series_and_metadata(context: AssetExecutionContext) -> None:
    """Ingests raw telemetry and metadata from NGED S3 into our local storage.

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
    #
    # Everything that talks to NGED's bucket sits under one retry guard, so a transient object-store
    # error costs a short wait instead of a Sentry event for a fault that has already fixed itself.
    # The guard stops here rather than wrapping the whole body: a fault in the writes below is ours,
    # and retrying it would let `select_new_rows` dedupe the second attempt to a no-op and turn a
    # real failure into a green run.
    try:
        store = settings.get_nged_s3_store()
        list_of_all_json_files = list_timeseries_json_files(store)
        list_of_large_json_files = remove_small_files_from_listing(list_of_all_json_files)
        list_of_new_json_files = select_new_rows(
            list_of_large_json_files, delta_path, storage_options
        )

        # Log statistics to be shown in Dagster's UI.
        context.add_output_metadata(
            _FileListingSummary.make_table(
                "nged_s3_paths",
                {
                    "All JSON files on S3": list_of_all_json_files,
                    "Files above the size threshold": list_of_large_json_files,
                    "Files with new data": list_of_new_json_files,
                },
            )
        )

        downloaded = download_and_parse_files(store, list_of_new_json_files)
    except NoNewData:
        # An ordinary hour in which NGED published nothing new. Must be caught before the retry
        # guard below, or every such hour would retry.
        context.add_output_metadata(
            UpsertMetadataStats(metadata_n_new_TimeSeriesIDs=0, metadata_n_updated_TimeSeriesIDs=0)
        )
        return
    except BaseException as exc:
        # `BaseException` for the same reason as `checks.py::power_data_is_fresh`: obstore, delta-rs
        # and polars each define their own exception classes and a Rust panic is not an `Exception`,
        # so naming what must *propagate* is the only version that stays true as dependencies come
        # and go. That makes this deliberately liberal — a bug in our own code in here is retried
        # too, which costs the budget and then reports as it would have anyway.
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        context.log.warning(f"Could not read NGED's S3 bucket, requesting a retry: {exc!r}")
        raise RetryRequested(
            max_retries=_POWER_INGEST_MAX_RETRIES,
            seconds_to_wait=_POWER_INGEST_RETRY_DELAY_SECONDS,
        ) from exc
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

    # Save TimeSeriesMetadata. A roster failure must not stop the power write below; what that costs
    # is in https://openclimatefix.github.io/nged-substation-forecast/live_service/operations/
    try:
        upsert_metadata_stats = upsert_metadata(
            new_metadata=new_metadata, metadata_path=metadata_path, storage_options=storage_options
        )
    except BaseException as exc:
        # The same guard as the asset checks, for the same reason — see the comment in
        # `checks.py::power_data_is_fresh` for why `BaseException` and what it costs in tests.
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        context.log.exception(f"Could not upsert the TimeSeriesMetadata roster at {metadata_path}")
        report_asset_degradation(asset_name="power_time_series_and_metadata", exc=exc)
        upsert_metadata_stats = UpsertMetadataStats(metadata_upsert_failed=repr(exc))

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


@asset(tags=PRODUCTION_LAYER_TAGS)
def h3_grid_weights(context: AssetExecutionContext) -> None:
    """Computes H3 grid weights for the Great Britain boundary.

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
"""Retries × ``_ECMWF_ENS_RETRY_DELAY_SECONDS`` ≥ 4h of coverage past the 08:30 UTC schedule
(``ecmwf_ens_schedule``), comfortably past Dynamical's typical publication time — and past the
3h25m a measured republication took. Applies to ``NwpRunNotYetAvailable`` and
``NwpVariableWhollyMissing``, the two ways an upstream run says "not ready yet"; a genuine bug
fails immediately instead of retrying for hours.

"≥" rather than "≈" because only ``NwpRunNotYetAvailable`` is raised before the download.
``NwpVariableWhollyMissing`` comes from validation *after* it, so each of those retries also pays
for a full re-download (22.5s at best, minutes when the upstream fetch is slow) and re-takes the
``ECMWF`` concurrency pool slot. That is the price of not discarding the partition, and the
elapsed window is wider than the delays alone imply."""

_ECMWF_ENS_RETRY_DELAY_SECONDS: Final[int] = 1800
"""How long to wait between retries of a not-yet-published ECMWF run."""

_NWP_QUALITY_CHECK_NAME: Final[str] = "nwp_has_no_unexpected_nulls"
"""Name of the per-run NWP data-quality check emitted by ``ecmwf_ens`` (see ``assess_nwp_quality``).

Computed in-asset from the frame already in memory rather than as a standalone ``@asset_check``
scanning the Delta table: the quality of a run is a property of the specific ingest we are holding,
so there is nothing to re-scan (and re-scanning the whole ~5.9B-row NWP table would hit Polars'
2**32 row-count ceiling). This differs from ``power_data_is_fresh``, whose freshness genuinely
drifts over time and so must re-read the table on a schedule."""

_NWP_QUALITY_CHECK_DESCRIPTION: Final[str] = (
    "Nulls in the de-accumulated NWP variables (precipitation and the two radiation fluxes) beyond "
    "lead-0, counted at both stages of ingest and never mixed. The `nwp_grid_point` keys count the "
    "raw 0.25 degree grid Dynamical.org sent us, before aggregation: that is the provider signal — "
    "is the feed broken, and since when? The `h3_cell` keys count the cells we store after "
    "area-weighted aggregation: that is how much the model actually lost. The aggregation "
    "renormalises each cell over the grid points that supplied a value, so it absorbs most "
    "upstream scatter, and a corrupt run can have null grid points and no null cell. The two are "
    "not comparable as rates — different units over different populations — and only this check's "
    "`passed` follows the cells. See "
    "https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/."
)
"""Standing explanation shown in the Dagster UI's Checks view.

The per-run ``AssetCheckResult.description`` carries that run's numbers; this says what they
mean."""

_NWP_COMPLETENESS_CHECK_NAME: Final[str] = "nwp_run_is_complete"
"""Name of the per-run NWP completeness check emitted by ``ecmwf_ens`` (see
``assess_nwp_run_completeness``).

Separate from ``nwp_has_no_unexpected_nulls`` because the two answer different questions with
different remedies: that one asks whether the rows we got are usable, this one asks whether we got
all the rows. Computed in-asset from the frame in memory, for the same reason."""

_NWP_INSTANTANEOUS_CHECK_NAME: Final[str] = "nwp_instantaneous_variables_have_no_nulls"
"""Name of the per-run check on the instantaneous variables' raw-grid nulls.

Separate from ``nwp_has_no_unexpected_nulls`` because the two count populations whose nulls mean
opposite things, and so warrant opposite thresholds and different remedies: tolerated corruption to
read as a trend, against an anomaly to raise with Dynamical.org today."""

_NWP_INSTANTANEOUS_CHECK_DESCRIPTION: Final[str] = (
    "Nulls in the instantaneous NWP variables (temperature, dew point, the winds, the pressures, "
    "geopotential height) on the raw 0.25 degree grid Dynamical.org sent us. These are never "
    "legitimately null, so `passed` is false on a single null grid point — unlike "
    "`nwp_has_no_unexpected_nulls`, whose nulls are expected. A red result is a mail to "
    "Dynamical.org, not a re-run: the run has landed and no stored cell is affected, because the "
    "aggregation renormalises each H3 cell over the grid points that supplied a value and so "
    "absorbs scattered nulls before they reach one. That absorption is also why this check counts "
    "the raw grid: a null that does reach a cell never gets here, since `Nwp.validate` rejects "
    "the run first. See "
    "https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/."
)
"""Standing explanation shown in the Dagster UI's Checks view."""

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

_UPSTREAM_PER_VARIABLE_SCHEMA: Final[TableSchema] = TableSchema(
    columns=[
        TableColumn("variable", "string"),
        TableColumn("n_null_grid_points", "int"),
        TableColumn("n_affected_slices", "int"),
        TableColumn("n_total_grid_points", "int"),
    ]
)
"""Fixed schema for the per-variable raw-grid null table (so a clean run still renders one)."""


@asset(
    tags=PRODUCTION_LAYER_TAGS,
    partitions_def=ecmwf_ens_partitions,
    deps=["h3_grid_weights"],
    check_specs=[
        AssetCheckSpec(
            name=_NWP_QUALITY_CHECK_NAME,
            asset="ecmwf_ens",
            blocking=False,
            description=_NWP_QUALITY_CHECK_DESCRIPTION,
        ),
        AssetCheckSpec(
            name=_NWP_INSTANTANEOUS_CHECK_NAME,
            asset="ecmwf_ens",
            blocking=False,
            description=_NWP_INSTANTANEOUS_CHECK_DESCRIPTION,
        ),
        AssetCheckSpec(name=_NWP_COMPLETENESS_CHECK_NAME, asset="ecmwf_ens", blocking=False),
    ],
    # The `pool="ECMWF"` works in conjunction with the Dagster instance configuration
    # (e.g., in `dagster.yaml`) to limit the number of times this asset can be run
    # concurrently. This is crucial because downloading ECMWF data is memory-intensive.
    # See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
    pool="ECMWF",
)
def ecmwf_ens(context: AssetExecutionContext) -> MaterializeResult:
    """Downloads and processes ECMWF ensemble NWP data for a specific day.

    This asset fetches the 00Z NWP run for the partition date, converts it to a
    Polars DataFrame, and writes it to the Delta table through
    ``delta_store.nwp.write_nwp`` (Float32, significand-rounded), which replaces that
    ``(nwp_model_id, init_time)`` partition.

    Its ``nwp_has_no_unexpected_nulls`` check reports null counts for both the raw NWP grid and the
    stored H3 cells; that check's own description says which keys are which and why they differ.
    ``nwp_instantaneous_variables_have_no_nulls`` counts the raw grid again, over the variables that
    are never legitimately null, where a single null is worth acting on.
    """
    settings = Settings()
    storage_options = settings.storage_options
    partition_date_str = context.partition_key
    nwp_init_time = datetime.strptime(partition_date_str, "%Y-%m-%d").replace(tzinfo=UTC)

    # Load dependencies
    h3_grid = pt.DataFrame(
        pl.read_parquet(
            settings.h3_grid_weights_path, storage_options=typeddict_to_dict(storage_options)
        )
    ).set_model(H3GridWeights)

    # Download and convert. Both retryable failures mean "the upstream run is not ready yet", they
    # just say it at different points: the run is absent from the catalog, or it is present but a
    # weather variable is still wholesale empty. Dynamical.org publishes each run as ~40 separate
    # Icechunk commits, so an in-progress publication is genuinely readable and genuinely
    # incomplete, and a defective one gets republished — the 2026-08-09 00Z run was repaired 3h25m
    # after its first publication, inside this asset's four-hour retry budget. Every other error
    # still fails immediately.
    try:
        ds_lazy = open_ecmwf_ens_run(nwp_init_time=nwp_init_time, h3_grid=h3_grid)
        context.log.info("Lazily opened Icechunk store.")

        ds = download_ecmwf_ens_data(ds_lazy)
        context.log.info("Downloaded Icechunk data.")

        nwp = convert_nwp_xarray_dataset_to_polars_dataframe(ds=ds, h3_grid=h3_grid)
    except (NwpRunNotYetAvailable, NwpVariableWhollyMissing) as exc:
        context.log.warning(f"ECMWF ENS run not usable yet, requesting a retry: {exc}")
        raise RetryRequested(
            max_retries=_ECMWF_ENS_MAX_RETRIES, seconds_to_wait=_ECMWF_ENS_RETRY_DELAY_SECONDS
        ) from exc
    context.log.info(f"Converted NWP data to Polars. Columns: {nwp.columns}")

    # Three non-fatal per-run checks. The first surfaces the tolerated scattered nulls (known
    # upstream ECMWF ENS corruption) that Nwp.validate deliberately let through. The second counts
    # the same raw grid over the variables that are never legitimately null, where the aggregation
    # absorbs scattered corruption so completely that nothing downstream of it can see any. The
    # third asks whether the run is *whole*; a short run is the upstream provider misbehaving, so we
    # keep the rows that did arrive and WARN rather than discarding the run. Computed before the
    # write, not merely under a guard — rule 7 of
    # https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules
    try:
        quality = assess_nwp_quality(nwp)
        upstream = assess_upstream_grid_point_nulls(
            ds=ds, variables=Nwp.deaccumulated_var_names, exclude_lead_0=True
        )
        instantaneous = assess_upstream_grid_point_nulls(
            ds=ds, variables=ECMWF_ENS_INSTANTANEOUS_VARS, exclude_lead_0=False
        )
        completeness = assess_nwp_run_completeness(
            dataframe=nwp, expected_n_h3_cells=h3_grid["h3_index"].n_unique()
        )
        check_results = [
            _nwp_quality_check_result(report=quality, upstream=upstream),
            _nwp_instantaneous_check_result(instantaneous),
            _nwp_completeness_check_result(completeness),
        ]
        shape_metadata = _nwp_run_shape_metadata(completeness)
        upstream_metadata = _upstream_null_metadata(upstream)
    except BaseException as exc:
        # The same guard as `power_data_is_fresh` — see the comment there for why it catches
        # `BaseException`, what that costs when writing tests, and rule 7 for why a warning path
        # may never raise.
        if isinstance(exc, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
            raise  # A cancelled run must cancel.
        context.log.exception("Could not assess the ingested NWP run")
        # One event for one fault: all three checks share this guard, so they degrade together.
        # The tag names the quality check whichever one raised; the runbook says to expect that.
        report_check_degradation(check_name=_NWP_QUALITY_CHECK_NAME, exc=exc)
        check_results = [
            _degraded_nwp_check_result(check_name=_NWP_QUALITY_CHECK_NAME, exc=exc),
            _degraded_nwp_check_result(check_name=_NWP_INSTANTANEOUS_CHECK_NAME, exc=exc),
            _degraded_nwp_check_result(check_name=_NWP_COMPLETENESS_CHECK_NAME, exc=exc),
        ]
        shape_metadata = {}
        upstream_metadata = {}

    nwp_data_path = settings.nwp_data_path
    if_local_path_then_make_parent_dir(nwp_data_path)
    write_nwp(nwp=nwp, table_uri=nwp_data_path, storage_options=storage_options)
    context.log.info(f"Saved NWP data to Delta table at {nwp_data_path}.")

    return MaterializeResult(
        metadata={
            "n_rows": len(nwp),
            "path": nwp_data_path,
            "init_time": str(nwp_init_time),
            **shape_metadata,
            **upstream_metadata,
        },
        check_results=check_results,
    )


def _degraded_nwp_check_result(check_name: str, exc: BaseException) -> AssetCheckResult:
    """A WARN result for a per-run NWP check that could not be evaluated at all.

    Inside an asset declaring several ``AssetCheckSpec``s an unnamed result fails the step outright,
    so ``check_name`` is required — unlike in a standalone ``@asset_check``.
    """
    return AssetCheckResult(
        check_name=check_name,
        passed=False,
        severity=AssetCheckSeverity.WARN,
        description=f"Could not assess the ingested NWP run: {exc!r}",
    )


def _nwp_run_shape_metadata(report: NwpRunCompletenessReport) -> dict[str, MetadataValue]:
    """The run's observed shape.

    Published on every materialisation whose completeness assessment succeeded, including the ones
    where the check passes, so drift is visible in the Dagster UI timeline.
    """
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


def _nwp_quality_check_result(
    report: NwpQualityReport, upstream: UpstreamNullRate
) -> AssetCheckResult:
    """Wrap the two null reports for one run into a WARN-severity Dagster check result.

    ``passed`` follows the H3 cells alone. The upstream rate is a trend across runs, not a verdict
    on this one, and the archive has no threshold that separates a healthy feed from a degrading
    one — so it is published and plotted rather than gated. Escalating a badly-degraded run is
    <https://github.com/openclimatefix/nged-substation-forecast/issues/501>.
    """
    return AssetCheckResult(
        check_name=_NWP_QUALITY_CHECK_NAME,
        # WARN, never fail: these nulls are expected upstream corruption we deliberately ingest.
        # Only a variable empty in *every* slice is fatal, and Nwp.validate has already rejected
        # that before this runs.
        passed=report.is_healthy,
        severity=AssetCheckSeverity.WARN,
        description=_nwp_quality_description(report=report, upstream=upstream),
        metadata={
            "n_null_h3_cells": report.n_null_cells,
            "n_affected_h3_slices": report.n_affected_slices,
            # The split of `n_affected_h3_slices`, broken out because the two halves mean different
            # things and only one of them is measured well: a wholly-null slice reaches the cells
            # intact however they are aggregated, whereas the scattered remainder is only whatever
            # upstream corruption happened to take out every grid point of a cell. Both are
            # emitted so the operations runbook can name either as a number to read off this check.
            "n_whole_null_h3_slices": report.n_whole_null_slices,
            "n_scattered_h3_slices": report.n_scattered_slices,
            "affected_h3_variables": list(report.affected_variables),
            "affected_h3_slices": _nwp_null_slices_metadata(report.affected),
            "n_null_nwp_grid_points": upstream.n_null_nwp_grid_points,
            "n_total_nwp_grid_points": upstream.n_total_nwp_grid_points,
            "null_nwp_grid_point_fraction": upstream.null_nwp_grid_point_fraction,
            "n_affected_nwp_slices": upstream.n_affected_nwp_slices,
            "affected_nwp_variables": list(upstream.affected_nwp_variables),
            "per_nwp_variable": _upstream_per_variable_metadata(upstream.per_variable),
        },
    )


def _nwp_instantaneous_check_result(upstream: UpstreamNullRate) -> AssetCheckResult:
    """Wrap the instantaneous variables' raw-grid null count into a WARN Dagster check result.

    ``passed`` is a zero threshold, unlike ``nwp_has_no_unexpected_nulls``'s: these variables are
    never legitimately null, so one null grid point is worth an operator's attention. It still only
    WARNs, because by the time this runs the aggregation has already absorbed whatever it counts —
    the run is landed either way, and what is at stake is whether we ask Dynamical.org about it.
    """
    return AssetCheckResult(
        check_name=_NWP_INSTANTANEOUS_CHECK_NAME,
        passed=upstream.is_healthy,
        severity=AssetCheckSeverity.WARN,
        description=_nwp_instantaneous_description(upstream),
        metadata={
            "n_null_nwp_grid_points": upstream.n_null_nwp_grid_points,
            "n_total_nwp_grid_points": upstream.n_total_nwp_grid_points,
            "null_nwp_grid_point_fraction": upstream.null_nwp_grid_point_fraction,
            "n_affected_nwp_slices": upstream.n_affected_nwp_slices,
            "affected_nwp_variables": list(upstream.affected_nwp_variables),
            "per_nwp_variable": _upstream_per_variable_metadata(upstream.per_variable),
        },
    )


def _nwp_instantaneous_description(upstream: UpstreamNullRate) -> str:
    """Describe the instantaneous variables' raw-grid nulls for one run."""
    if upstream.is_healthy:
        return f"No nulls in {upstream.n_total_nwp_grid_points} instantaneous-variable grid points."
    return (
        f"{upstream.n_null_nwp_grid_points} of {upstream.n_total_nwp_grid_points} "
        f"instantaneous-variable grid point(s) null "
        f"({upstream.null_nwp_grid_point_fraction:.4%}) in "
        f"{', '.join(upstream.affected_nwp_variables)}, across "
        f"{upstream.n_affected_nwp_slices} (variable, member, step) slice(s)."
    )


def _upstream_per_variable_metadata(per_variable: pl.DataFrame) -> TableMetadataValue:
    """Render the per-variable raw-grid null counts as a Dagster metadata table.

    Uncapped, unlike the affected-slices table: there are thirteen weather variables in all, so this
    frame has no bad day on which it can grow.
    """
    records = [
        TableRecord(
            {
                "variable": row["variable"],
                "n_null_grid_points": row["n_null"],
                "n_affected_slices": row["n_affected_slices"],
                "n_total_grid_points": row["n_total"],
            }
        )
        for row in per_variable.iter_rows(named=True)
    ]
    return MetadataValue.table(records, schema=_UPSTREAM_PER_VARIABLE_SCHEMA)


def _nwp_quality_description(report: NwpQualityReport, upstream: UpstreamNullRate) -> str:
    """Describe both populations on every run, because the two routinely disagree.

    The aggregation absorbs most upstream scatter, so a run can arrive corrupt and still store no
    null cell; a description written from the cells alone would call that run clean.
    """
    grid_points = (
        f"Raw NWP grid: {upstream.n_null_nwp_grid_points} of {upstream.n_total_nwp_grid_points} "
        f"grid point(s) null beyond lead-0 ({upstream.null_nwp_grid_point_fraction:.4%})"
    )
    if not upstream.is_healthy:
        grid_points += (
            f" in {', '.join(upstream.affected_nwp_variables)}, across "
            f"{upstream.n_affected_nwp_slices} (variable, member, step) slice(s)"
        )
    cells = f"Stored H3 cells: {report.n_null_cells} null cell(s)"
    if not report.is_healthy:
        cells += (
            f" in {', '.join(report.affected_variables)}, across {report.n_scattered_slices} "
            f"partly-null and {report.n_whole_null_slices} wholly-null (member, valid_time) "
            "slice(s)"
        )
    # Only claim corruption when some was found: this string is the operator's summary of the run,
    # and a clean run described as tolerated corruption is the same false claim, inverted, that
    # reporting the cells alone used to make.
    verdict = (
        ""
        if upstream.is_healthy and report.is_healthy
        else (
            " Known upstream ECMWF ENS corruption, tolerated. See "
            "https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/."
        )
    )
    return f"{grid_points}. {cells}.{verdict}"


def _upstream_null_metadata(upstream: UpstreamNullRate) -> dict[str, MetadataValue]:
    """The upstream corruption rate, for the materialisation timeline.

    Published on every materialisation whose assessment succeeded, including the ones where the
    check passes, because the provider question this answers — is the feed degrading? — is about
    the trend across runs and is invisible in any single one.
    """
    return {
        "n_null_nwp_grid_points": MetadataValue.int(upstream.n_null_nwp_grid_points),
        "null_nwp_grid_point_fraction": MetadataValue.float(upstream.null_nwp_grid_point_fraction),
    }


_NWP_NULL_SLICES_TABLE_LIMIT: Final[int] = 100
"""Cap on rows rendered in the affected-slices metadata table.

A broadly-corrupt upstream run could touch thousands of (variable, member, valid_time) slices;
the exact totals live in the scalar metadata, so the table only needs the worst offenders to be
useful — bounding it keeps the Dagster event log from bloating on a bad day."""


def _nwp_null_slices_metadata(affected: pl.DataFrame) -> TableMetadataValue:
    """Render the worst affected (variable, member, valid_time) slices as a Dagster metadata table.

    Capped at ``_NWP_NULL_SLICES_TABLE_LIMIT`` rows; the full counts are in the scalar metadata
    alongside. Wholly-null slices sort first, then the most-null of the rest — that is the order an
    operator wants, because a wholly-null slice names a field that arrived missing. Sorting on
    ``n_null`` alone would not achieve it: slices need not have equal cell counts (a short run is
    tolerated), so a wholly-null slice of few cells can carry fewer nulls than a partly-null slice
    of many.
    """
    top = (
        affected.with_columns(is_whole_null=pl.col("n_null") == pl.col("n_total"))
        .sort(["is_whole_null", "n_null"], descending=True)
        .head(_NWP_NULL_SLICES_TABLE_LIMIT)
    )
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


class _BaseSummary[T: pt.Model](ABC, BaseModel):
    """Create a Dagster table of summary statistics.

    The type parameter ``T`` makes this superclass generic over ``pt.Model`` subclasses.
    """

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
        # The `ty: ignore` comments are because `ty` only looks at the types specified in the
        # BaseModel. `ty` doesn't know that we're casting the types in the `field_validator`
        # methods.
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
        return cls(stage=stage_name, n_rows=0)
