"""Dagster assets for NGED data."""

import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple, cast

import dagster as dg
import h3.api.numpy_int as h3
import httpx
import nged_data
import polars as pl
from contracts.data_schemas import (
    MissingCorePowerVariablesError,
    SubstationMetadata,
)
from contracts.settings import Settings
from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    ResourceParam,
    asset,
)
from nged_data import ckan


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
        new_df = nged_data.process_live_primary_substation_flows(csv_data)
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

    # Ensure uniform schema
    for col in ["MW", "MVA", "MVAr"]:
        if col not in new_df.columns:
            new_df = new_df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))
        else:
            new_df = new_df.with_columns(pl.col(col).cast(pl.Float32))

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
            pl.scan_delta(delta_path)
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


def _filter_resources(
    all_resources: list[tuple[int, str]],
    processed_substations: set[int],
    config: LivePrimaryFlowsConfig,
    log: dg.DagsterLogManager,
) -> list[tuple[int, str]]:
    """Filter resources based on processing state and configuration."""
    # Filter out already processed
    unprocessed = [r for r in all_resources if r[0] not in processed_substations]

    # Apply manual filters
    if config.substation_numbers:
        filtered = [r for r in unprocessed if r[0] in config.substation_numbers]
        log.info("Filtered to %d substations by number", len(filtered))
        return filtered

    if config.limit:
        limited = unprocessed[: config.limit]
        log.info("Limited to %d substations", config.limit)
        return limited

    return unprocessed


def _download_and_process_all(
    resources: list[tuple[int, str]],
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
                        r[0], r[1], api_key, config.max_retries, log, client, partition_date
                    ),
                    resources,
                )
            )


def _merge_to_delta(
    delta_path: str, successes: list[SubstationIngestionResult], log: dg.DagsterLogManager
):
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
    partitions_def=DailyPartitionsDefinition(start_date="2026-03-10", end_offset=1),
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
    partition_date = datetime.strptime(context.partition_key, "%Y-%m-%d")
    delta_path = str(_get_delta_path(settings))

    # Identify what's already processed
    processed_substations = (
        set()
        if config.force_rerun_all
        else _get_processed_substations(delta_path, partition_date, context.log)
    )

    # Fetch resources from metadata
    metadata_path = settings.nged_data_path / "parquet" / "substation_metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    metadata_df = pl.read_parquet(metadata_path)
    all_resources = (
        metadata_df.filter(pl.col("url").is_not_null())
        .select(["substation_number", "url"])
        .iter_rows()
    )
    all_resources = list(all_resources)

    resources_to_process = _filter_resources(
        all_resources, processed_substations, config, context.log
    )
    resources_to_process = sorted(resources_to_process, key=lambda r: r[0])
    context.log.info(
        f"Planning to download and process data for {len(resources_to_process)} substations."
    )

    # Download and process
    api_key = settings.nged_ckan_token
    results = _download_and_process_all(
        resources_to_process, api_key, config, context.log, partition_date
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
) -> MaterializeResult:
    """Download primary substation locations and map them to H3 cells."""
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
    new_metadata = nged_data.substation_names.align.join_location_table_to_live_primaries(
        locations=locations, live_primaries=live_primaries
    )

    # 5. Add last_updated column
    now = datetime.now().astimezone(datetime.now().astimezone().tzinfo)
    new_metadata = new_metadata.with_columns(
        last_updated=pl.lit(now).cast(pl.Datetime("us", "UTC"))
    )

    # 6. Load existing metadata and upsert
    out_dir = settings.nged_data_path / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "substation_metadata.parquet"

    if out_path.exists():
        existing_metadata = pl.read_parquet(out_path)
        # Upsert: update existing rows, keep old rows that are not in new_metadata
        # We use an outer join and then coalesce
        combined = existing_metadata.join(
            new_metadata, on="substation_number", how="outer", suffix="_new"
        )

        # Coalesce columns
        cols = [c for c in existing_metadata.columns if c != "substation_number"]
        coalesce_exprs = [pl.coalesce(f"{c}_new", c).alias(c) for c in cols]
        final_metadata = combined.select(["substation_number"] + coalesce_exprs)
    else:
        final_metadata = new_metadata

    # 7. Validate against the new schema
    final_metadata = final_metadata.cast(SubstationMetadata.dtypes)
    validated_metadata = SubstationMetadata.validate(final_metadata)

    # 8. Save to disk
    validated_metadata.write_parquet(out_path)

    return MaterializeResult(
        metadata={
            "Path": MetadataValue.path(str(out_path)),
            "Row Count": MetadataValue.int(len(validated_metadata)),
        }
    )
