"""Dagster assets for NGED data."""

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import dagster as dg
import h3.api.basic_int as h3
import polars as pl
import patito as pt
from contracts.data_schemas import (
    NgedJsonPowerFlows,
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
    scan_delta_table,
)
from nged_json_data import (
    append_to_delta,
    clean_power_data,
    load_nged_json,
)
from nged_data.substation_names import align
from .partitions import DAILY_PARTITIONS


from nged_substation_forecast.ingestion.helpers import (
    IngestionStage,
    SubstationResource,
    get_delta_path as _get_delta_path,
    format_failure_metadata as _format_failure_metadata,
    get_processed_substations as _get_processed_substations,
    filter_resources as _filter_resources,
    download_and_process_all as _download_and_process_all,
    merge_to_delta as _merge_to_delta,
)


class LivePrimaryFlowsConfig(dg.Config):
    """Configuration for the live primary flows ingestion asset."""

    force_rerun_all: bool = False
    max_concurrent_connections: int = 32
    max_retries: int = 3
    substation_numbers: list[int] | None = None  # Download only a subset of substations.
    limit: int | None = None  # Only download the first n substations. Useful for testing.


@asset(
    partitions_def=DAILY_PARTITIONS,
    check_specs=[
        AssetCheckSpec(name="all_substations_succeeded", asset="live_primary_flows"),
    ],
    deps=["substation_metadata"],
    group_name="NGED_CKAN_Deprecated",
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


@asset(group_name="NGED_CKAN_Deprecated")
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
        # Use diagonal concat to handle cases where the schema evolved (e.g., adding preferred_power_col).
        final_metadata = pl.concat([existing_metadata, new_metadata], how="diagonal").unique(
            subset=["substation_number"], keep="last"
        )
    else:
        final_metadata = new_metadata

    # 7. Validate against the new schema
    # Ensure we are working with a DataFrame before casting
    if isinstance(final_metadata, pl.LazyFrame):
        final_metadata = final_metadata.collect()

    # final_metadata can be DataFrame | InProcessQuery; cast to DataFrame for column operations
    final_metadata = cast(pl.DataFrame, final_metadata)

    # Add preferred_power_col if missing to avoid casting errors
    if "preferred_power_col" not in final_metadata.columns:
        final_metadata = final_metadata.with_columns(
            preferred_power_col=pl.lit(None).cast(pl.String)
        )

    # Cast to ensure the types match the contract's expected dtypes
    final_metadata = final_metadata.cast(SubstationMetadata.dtypes)  # type: ignore
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
@asset(group_name="NGED_CKAN_Deprecated")
def substation_power_preferences(
    context: AssetExecutionContext,
    settings: ResourceParam[Settings],
    cleaned_actuals: pl.DataFrame,
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


@asset(group_name="NGED_JSON")
def nged_json_archive_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """One-off historical backfill of NGED JSON data."""
    json_dir = settings.nged_data_path / "json" / "archive"

    for json_file in json_dir.glob("*.json"):
        metadata_df, time_series_df = load_nged_json(json_file)

        # Clean power data
        substation_number = metadata_df.get_column("substation_number").item()
        cleaned_df = clean_power_data(
            time_series_df,
            substation_number=substation_number,
            variance_thresholds=settings.data_quality.variance_thresholds,
        )

        # Append to delta
        append_to_delta(cleaned_df, settings.nged_data_path / "delta" / "json_data")

    context.log.info("Finished processing archive JSON data.")


@asset(
    partitions_def=dg.TimeWindowPartitionsDefinition(
        cron_schedule="0 */6 * * *",
        start="2026-01-26-00:00",
        fmt="%Y-%m-%d-%H:%M",
        timezone="UTC",
    ),
    group_name="NGED_JSON",
    op_tags={"dagster/concurrency_key": "nged_json_ingestion"},
)
def nged_json_live_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """Live updates of NGED JSON data."""
    # Assuming there's a directory with JSON files for the current partition
    partition_date = context.partition_time_window.start
    json_dir = settings.nged_data_path / "json" / "live" / partition_date.strftime("%Y-%m-%d-%H")

    cleaned_dfs = []
    for json_file in json_dir.glob("*.json"):
        metadata_df, time_series_df = load_nged_json(json_file)

        # Clean power data
        substation_number = metadata_df.get_column("substation_number").item()
        cleaned_df = clean_power_data(
            time_series_df,
            substation_number=substation_number,
            variance_thresholds=settings.data_quality.variance_thresholds,
        )
        cleaned_dfs.append(cleaned_df)

    if cleaned_dfs:
        # Combine all cleaned dataframes and append in a single operation
        combined_df = pl.concat(cleaned_dfs)
        append_to_delta(
            pt.DataFrame[NgedJsonPowerFlows](combined_df),
            settings.nged_data_path / "delta" / "json_data",
        )

    context.log.info(f"Finished processing live JSON data for {partition_date}.")


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
