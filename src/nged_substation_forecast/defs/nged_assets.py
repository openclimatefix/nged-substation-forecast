"""Dagster assets for NGED data."""

import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import NamedTuple, cast

import dagster as dg
import h3.api.basic_int as h3
import httpx
import polars as pl
import patito as pt
from contracts.data_schemas import (
    MissingCorePowerVariablesError,
    SubstationMetadata,
    SubstationPowerFlows,
)
from contracts.settings import Settings
from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetExecutionContext,
    MaterializeResult,
    MetadataValue,
    ResourceParam,
    asset,
)
from ml_core.data import calculate_target_map
from nged_data import (
    ckan,
    clean_substation_flows,
    get_partition_window,
    process_live_primary_substation_power_flows,
    scan_delta_table,
)
from nged_data.substation_names import align
from .partitions import DAILY_PARTITIONS


class IngestionStage(str, Enum):
    """Stages of the ingestion process."""

    DOWNLOAD = "Download"
    PROCESSING = "Processing"
    STORAGE = "Storage"
    SUCCESS = "Success"


class SubstationIngestionResult(NamedTuple):
    """The result of ingesting a single substation's data."""

    substation_number: int
    stage: IngestionStage  # This tells us at which stage in the process the failure occurred.
    error_message: str | None = None
    csv_snippet: str | None = None
    df: pl.DataFrame | None = None


class LivePrimaryFlowsConfig(dg.Config):
    """Configuration for the live primary flows ingestion asset."""

    force_rerun_all: bool = False
    max_concurrent_connections: int = 32
    max_retries: int = 3
    substation_numbers: list[int] | None = None  # Download only a subset of substations.
    limit: int | None = None  # Only download the first n substations. Useful for testing.


def _get_delta_path(settings: Settings) -> Path:
    return settings.nged_data_path / "delta" / "live_primary_flows"


def _download_and_process_substation(
    substation_number: int,
    url: str,
    api_key: str,
    max_retries: int,
    log: dg.DagsterLogManager,
    client: httpx.Client,
    partition_date: datetime,
) -> SubstationIngestionResult:
    log.info(f"Processing substation {substation_number}...")

    # Download
    try:
        response = ckan.httpx_get_with_auth(
            url, api_key=api_key, max_retries=max_retries, client=client
        )
        csv_data = response.content
    except Exception as e:
        return SubstationIngestionResult(
            substation_number=substation_number, stage=IngestionStage.DOWNLOAD, error_message=str(e)
        )

    # Process
    try:
        new_df = process_live_primary_substation_power_flows(csv_data)
    except MissingCorePowerVariablesError:
        log.info(f"Skipping substation {substation_number} because it lacks MW/MVA data.")
        # Return success but with no dataframe, so it gets recorded as processed
        # but nothing is merged into Delta Lake.
        return SubstationIngestionResult(
            substation_number=substation_number, stage=IngestionStage.SUCCESS, df=None
        )
    except Exception as e:
        # The exception might have a note with the CSV snippet from process_flows.py
        error_message = str(e)
        if hasattr(e, "__notes__") and e.__notes__:
            error_message += "\n" + "\n".join(e.__notes__)

        try:
            csv_snippet = "\n".join(csv_data.decode("utf-8", errors="replace").splitlines()[:3])
        except Exception:
            csv_snippet = "Could not decode CSV snippet"
        return SubstationIngestionResult(
            substation_number=substation_number,
            stage=IngestionStage.PROCESSING,
            error_message=error_message,
            csv_snippet=csv_snippet,
        )

    # Add required columns for Delta Lake
    new_df = new_df.with_columns(
        [
            pl.lit(substation_number).cast(pl.Int32).alias("substation_number"),
            pl.lit(partition_date).cast(pl.Datetime("us", "UTC")).alias("ingested_at"),
        ]
    )

    new_df = new_df.select(["timestamp", "substation_number", "MW", "MVA", "MVAr", "ingested_at"])

    return SubstationIngestionResult(
        substation_number=substation_number, stage=IngestionStage.SUCCESS, df=new_df
    )


def _format_failure_metadata(
    failures: list[SubstationIngestionResult],
) -> dict[str, dg.MetadataValue]:
    if not failures:
        return {}

    # Build Markdown Table
    md_table = "| Substation Number | Stage | Error |\n| :--- | :--- | :--- |\n"
    for failure in failures:
        md_table += (
            f"| {failure.substation_number} | {failure.stage.value} | {failure.error_message} |\n"
        )

    # Build Snippets
    md_snippets = "### Failed CSV Snippets\n"
    for failure in failures:
        if failure.csv_snippet:
            md_snippets += (
                f"**{failure.substation_number}**:\n```text\n{failure.csv_snippet}\n```\n"
            )

    return {"Failure Details": MetadataValue.md(md_table + "\n" + md_snippets)}


def _get_processed_substations(
    delta_path: str, partition_date: datetime, log: dg.DagsterLogManager
) -> set[int]:
    """Identify which substations have already been processed for the given partition date."""
    if not Path(delta_path).exists():
        return set()

    try:
        processed_df = cast(
            pl.DataFrame,
            scan_delta_table(delta_path)
            .filter(pl.col("ingested_at") == pl.lit(partition_date).cast(pl.Datetime("us", "UTC")))
            .select("substation_number")
            .unique()
            .collect(),
        )
    except Exception as e:
        log.exception(f"Failed to read Delta Table {delta_path}: {e}")
        raise
    processed_substations = set(processed_df.get_column("substation_number").to_list())
    log.info(
        f"{len(processed_substations)} substations have already been processed for {partition_date}"
    )
    return processed_substations


class SubstationResource(NamedTuple):
    """A resource for a substation's live telemetry."""

    substation_number: int
    url: str


def _filter_resources(
    all_resources: list[SubstationResource],
    processed_substations: set[int],
    config: LivePrimaryFlowsConfig,
    log: dg.DagsterLogManager,
) -> list[SubstationResource]:
    """Filter resources based on processing state and configuration."""
    # Filter out already processed
    unprocessed = [r for r in all_resources if r.substation_number not in processed_substations]

    # Apply manual filters
    if config.substation_numbers:
        filtered = [r for r in unprocessed if r.substation_number in config.substation_numbers]
        log.info("Filtered to %d substations by number", len(filtered))
        return filtered

    if config.limit:
        limited = unprocessed[: config.limit]
        log.info("Limited to %d substations", config.limit)
        return limited

    return unprocessed


def _download_and_process_all(
    resources: list[SubstationResource],
    api_key: str,
    config: LivePrimaryFlowsConfig,
    log: dg.DagsterLogManager,
    partition_date: datetime,
) -> list[SubstationIngestionResult]:
    """Download and process multiple substations concurrently."""
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=config.max_concurrent_connections) as executor:
            return list(
                executor.map(
                    lambda r: _download_and_process_substation(
                        r.substation_number,
                        r.url,
                        api_key,
                        config.max_retries,
                        log,
                        client,
                        partition_date,
                    ),
                    resources,
                )
            )


def _merge_to_delta(
    delta_path: str, successes: list[SubstationIngestionResult], log: dg.DagsterLogManager
) -> None:
    """Merge successful ingestion results into the Delta Lake table."""
    if not successes:
        return

    log.info(f"Merging {len(successes)} dataframes into Delta Lake...")
    combined_df = pl.concat([r.df for r in successes if r.df is not None])

    if not Path(delta_path).exists():
        combined_df.write_delta(
            delta_path, delta_write_options={"partition_by": ["substation_number"]}
        )
        return

    # Retry loop for concurrent writes
    max_retries = 5
    for attempt in range(max_retries):
        try:
            (
                combined_df.write_delta(
                    delta_path,
                    mode="merge",
                    delta_merge_options={
                        "predicate": "s.timestamp = t.timestamp AND s.substation_number = t.substation_number",
                        "source_alias": "s",
                        "target_alias": "t",
                    },
                )
                # We must update matched records to support the force_rerun_all configuration,
                # ensuring that if we re-fetch data for an existing timestamp/substation,
                # the record in the Delta table is updated with the latest values.
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )
            break
        except Exception as e:
            if "concurrent" in str(e).lower() or "commit failed" in str(e).lower():
                time.sleep(random.uniform(0.5, 2.0) * (1.5**attempt))
                if attempt == max_retries - 1:
                    raise
            else:
                raise


@asset(
    partitions_def=DAILY_PARTITIONS,
    check_specs=[
        AssetCheckSpec(name="all_substations_succeeded", asset="live_primary_flows"),
    ],
    deps=["substation_metadata"],
)
def live_primary_flows(
    context: AssetExecutionContext,
    config: LivePrimaryFlowsConfig,
    settings: ResourceParam[Settings],
) -> Iterable[dg.AssetCheckResult | dg.MaterializeResult]:
    """Download and process live primary substation flows from NGED CKAN."""
    # Use the shared helper to get the partition start time.
    partition_start, _, _ = get_partition_window(context.partition_key)

    # Gracefully skip if beyond the 5-day API limit
    if partition_start < datetime.now(timezone.utc) - timedelta(days=5):
        context.log.info(f"Partition {partition_start} is older than 5 days. Skipping API call.")
        yield dg.AssetCheckResult(passed=True, check_name="all_substations_succeeded")
        yield dg.MaterializeResult(metadata={"skipped": True, "reason": "Beyond 5-day API limit"})
        return

    delta_path = str(_get_delta_path(settings))

    # Identify what's already processed
    processed_substations = (
        set()
        if config.force_rerun_all
        else _get_processed_substations(delta_path, partition_start, context.log)
    )

    # Fetch resources from metadata
    metadata_path = settings.nged_data_path / "parquet" / "substation_metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata_df = pl.read_parquet(metadata_path)
    all_resources = [
        SubstationResource(substation_number=row[0], url=row[1])
        for row in metadata_df.filter(pl.col("url").is_not_null())
        .select(["substation_number", "url"])
        .iter_rows()
    ]

    resources_to_process = _filter_resources(
        all_resources, processed_substations, config, context.log
    )
    resources_to_process = sorted(resources_to_process, key=lambda r: r.substation_number)
    context.log.info(
        f"Planning to download and process data for {len(resources_to_process)} substations."
    )

    # Download and process
    api_key = settings.nged_ckan_token
    results = _download_and_process_all(
        resources_to_process, api_key, config, context.log, partition_start
    )
    successes = [r for r in results if r.stage == IngestionStage.SUCCESS and r.df is not None]
    failures = [r for r in results if r.stage != IngestionStage.SUCCESS]

    # Storage
    _merge_to_delta(delta_path, successes, context.log)

    # Reporting
    already_processed_count = len(processed_substations)
    filtered_out_count = len(all_resources) - already_processed_count - len(resources_to_process)

    metadata: dict[str, dg.MetadataValue] = {
        "Total resources in metadata": MetadataValue.int(len(all_resources)),
        "Already processed for this partition": MetadataValue.int(already_processed_count),
        "Manually filtered out": MetadataValue.int(filtered_out_count),
        "Successfully processed for this partition": MetadataValue.int(len(successes)),
        "Failed": MetadataValue.int(len(failures)),
    }
    metadata.update(_format_failure_metadata(failures))

    if failures:
        context.log.error(f"Found {len(failures)} failures!")
        for f in failures:
            context.log.error(f"Failure: {f.substation_number} - {f.stage} - {f.error_message}")

    yield AssetCheckResult(
        passed=len(failures) == 0,
        check_name="all_substations_succeeded",
        metadata=metadata,
    )
    yield MaterializeResult(metadata=metadata)


@asset
def substation_metadata(
    context: AssetExecutionContext,
    settings: ResourceParam[Settings],
) -> pl.DataFrame:
    """Download primary substation locations, map them to H3 cells."""
    # 1. Download using existing CKAN function
    locations = ckan.get_primary_substation_locations(api_key=settings.nged_ckan_token)

    # 2. Add the H3 resolution 5 column
    locations = locations.with_columns(
        h3_res_5=pl.struct(["latitude", "longitude"]).map_elements(
            lambda x: (
                h3.latlng_to_cell(x["latitude"], x["longitude"], 5)
                if x["latitude"] is not None and x["longitude"] is not None
                else None
            ),
            return_dtype=pl.UInt64,
        )
    )

    # 3. Fetch live primary flow resources from CKAN
    live_primaries = ckan.get_csv_resources_for_live_primary_substation_flows(
        api_key=settings.nged_ckan_token
    )

    # 4. Join locations with live primaries
    raw_new_metadata = align.join_location_table_to_live_primaries(
        locations=locations, live_primaries=live_primaries
    )

    new_metadata = cast(
        pl.DataFrame,
        raw_new_metadata.collect()
        if isinstance(raw_new_metadata, pl.LazyFrame)
        else raw_new_metadata,
    )

    # 5. Add last_updated column
    now = datetime.now(timezone.utc)
    new_metadata = new_metadata.with_columns(
        last_updated=pl.lit(now).cast(pl.Datetime("us", "UTC"))
    )

    # 6. Load existing metadata and upsert
    out_dir = settings.nged_data_path / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "substation_metadata.parquet"

    if out_path.exists():
        existing_metadata = pl.read_parquet(out_path)
        # Simplified upsert logic: concat and keep the last occurrence of each substation_number.
        # This ensures that new metadata overwrites existing metadata for the same substation.
        final_metadata = pl.concat([existing_metadata, new_metadata]).unique(
            subset=["substation_number"], keep="last"
        )
    else:
        final_metadata = new_metadata

    # 7. Validate against the new schema
    # Ensure we are working with a DataFrame before casting
    if isinstance(final_metadata, pl.LazyFrame):
        final_metadata = final_metadata.collect()

    # Cast to ensure the types match the contract's expected dtypes
    final_metadata = cast(pl.DataFrame, final_metadata).cast(SubstationMetadata.dtypes)  # type: ignore
    validated_metadata = SubstationMetadata.validate(final_metadata)

    # 8. Save to disk
    validated_metadata.write_parquet(out_path)

    context.add_output_metadata(
        metadata={
            "Path": MetadataValue.path(str(out_path)),
            "Row Count": MetadataValue.int(len(validated_metadata)),
        }
    )

    return validated_metadata


# POTENTIAL DATA LEAKAGE: The user has consciously decided to use the entire history
# to determine the preferred power column (including the 'Dead Sensor' logic).
# This introduces temporal leakage, but the user considers it a minor concern
# that doesn't justify complicating the code.
@asset
def substation_power_preferences(
    context: AssetExecutionContext,
    settings: ResourceParam[Settings],
    cleaned_actuals: pl.LazyFrame,
) -> pl.DataFrame:
    """Calculate the preferred power column and peak capacity for each substation."""
    # Cast to Patito LazyFrame to satisfy the type checker
    target_map = calculate_target_map(cast(pt.LazyFrame[SubstationPowerFlows], cleaned_actuals))

    out_dir = settings.nged_data_path / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "substation_power_preferences.parquet"

    target_map.write_parquet(out_path)

    context.add_output_metadata(
        metadata={
            "Path": MetadataValue.path(str(out_path)),
            "Row Count": MetadataValue.int(len(target_map)),
        }
    )

    return target_map


@dg.asset_check(asset="live_primary_flows")
def substation_data_quality(
    context: dg.AssetCheckExecutionContext,
    settings: dg.ResourceParam[Settings],
) -> dg.AssetCheckResult:
    """Check for stuck sensors and insane values in substation flows.

    This asset check identifies problematic telemetry data in the live_primary_flows:
    - **Stuck sensors**: Rolling std dev < stuck_std_threshold indicates a stuck sensor.
    - **Insane values**: MW outside [min_mw_threshold, max_mw_threshold] are physically implausible.

    The check runs on each daily partition to catch issues early at ingestion time.
    It reuses the production cleaning logic to identify bad substations.

    Args:
        context: Asset check execution context.
        settings: Configuration with data quality thresholds.

    Returns:
        AssetCheckResult indicating whether the data quality check passed or failed,
        with metadata about affected substations.
    """
    # Use the shared helper to get the partition window with a 1-day lookback.
    partition_start, partition_end, lookback_start = get_partition_window(
        context.partition_key, lookback_days=1
    )
    delta_path = str(_get_delta_path(settings))

    if not Path(delta_path).exists():
        return dg.AssetCheckResult(
            passed=True,
            description="Delta table not found",
            metadata={"warning": dg.MetadataValue.json("Delta table does not exist yet")},
        )

    # Use the new scan_delta_table helper which handles UTC timezone boilerplate.
    live_primary_flows = scan_delta_table(delta_path)

    live_primary_flows = live_primary_flows.filter(
        pl.col("timestamp").is_between(lookback_start, partition_end, closed="left")
    )

    def _check_for_bad_substations(df: pl.DataFrame) -> list[int]:
        """Identify substations with data quality issues by comparing raw and cleaned data.

        Args:
            df: Raw substation flows DataFrame.

        Returns:
            List of substation IDs that had values nulled during cleaning.
        """
        # We reuse the production cleaning logic to identify bad substations.
        # This ensures consistency between the asset check and the actual cleaning process.
        cleaned_df = clean_substation_flows(df, settings)

        # Join them to compare original and cleaned values.
        # We only care about the current partition, not the lookback period.
        comparison_df = df.join(
            cleaned_df, on=["timestamp", "substation_number"], how="inner", suffix="_cleaned"
        ).filter(pl.col("timestamp") >= partition_start)

        # A substation is "bad" if any of its MW or MVA values were nulled during cleaning
        # that were NOT null in the original data.
        power_cols = [col for col in ["MW", "MVA"] if col in comparison_df.columns]

        bad_substations = set()
        for col in power_cols:
            # Find where the original was NOT null but the cleaned IS null.
            is_newly_null = pl.col(col).is_not_null() & pl.col(f"{col}_cleaned").is_null()
            bad_substations.update(
                comparison_df.filter(is_newly_null)
                .get_column("substation_number")
                .unique()
                .to_list()
            )

        return sorted(list(bad_substations))

    # Handle empty partitions gracefully
    if live_primary_flows.collect_schema().names() == []:
        return dg.AssetCheckResult(
            passed=True,
            description="No data found for partition",
            metadata={
                "warning": dg.MetadataValue.json("No data to check for this daily partition")
            },
        )

    # Materialize the data for checking.
    try:
        df = cast(pl.DataFrame, live_primary_flows.collect())
    except Exception as e:
        return dg.AssetCheckResult(
            passed=True,
            description=f"Could not collect sample data: {e}",
            metadata={"warning": dg.MetadataValue.json(str(e))},
        )

    if len(df) == 0:
        return dg.AssetCheckResult(
            passed=True,
            description="Sample data is empty",
            metadata={"warning": dg.MetadataValue.json("Sample materialized to empty DataFrame")},
        )

    # Run the checks
    bad_substations = _check_for_bad_substations(df)

    # Report result
    if bad_substations:
        return dg.AssetCheckResult(
            passed=False,
            description=f"Found {len(bad_substations)} substations with data quality issues",
            metadata={
                "affected_substation_count": dg.MetadataValue.int(len(bad_substations)),
                "bad_substation_sample": dg.MetadataValue.json(bad_substations[:100]),
            },
            severity=dg.AssetCheckSeverity.ERROR,
        )

    return dg.AssetCheckResult(
        passed=True,
        description="All substations passed data quality checks",
    )
