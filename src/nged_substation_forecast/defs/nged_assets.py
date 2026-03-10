"""Dagster assets for NGED data."""

import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import dagster as dg
import httpx
import nged_data
import polars as pl
from dagster import (
    AssetCheckResult,
    AssetCheckSpec,
    AssetExecutionContext,
    DailyPartitionsDefinition,
    MaterializeResult,
    MetadataValue,
    asset,
)
from deltalake import DeltaTable, write_deltalake
from nged_data import ckan
from nged_data.process_flows import MissingCorePowerVariablesError

from nged_substation_forecast.config_resource import NgedConfig


class IngestionStage(str, Enum):
    """Stages of the ingestion process."""

    DOWNLOAD = "Download"
    PROCESSING = "Processing"
    STORAGE = "Storage"
    SUCCESS = "Success"


class SubstationIngestionResult(NamedTuple):
    """The result of ingesting a single substation's data."""

    substation_name: str
    stage: IngestionStage  # This tells us at which stage in the process the failure occurred.
    error_message: str | None = None
    csv_snippet: str | None = None
    df: pl.DataFrame | None = None


class LivePrimaryFlowsConfig(dg.Config):
    """Configuration for the live primary flows ingestion asset."""

    force_rerun_all: bool = False
    max_concurrent_connections: int = 10
    max_retries: int = 3
    substation_names: list[str] | None = None  # Download only a subset of substations.
    limit: int | None = None  # Only download the first n substations. Useful for testing.


def _get_delta_path(nged_config: NgedConfig) -> Path:
    settings = nged_config.to_settings()
    return settings.NGED_DATA_PATH / "delta" / "live_primary_flows"


def _download_and_process_substation(
    resource: nged_data.CkanResource,
    api_key: str,
    max_retries: int,
    log: dg.DagsterLogManager,
    client: httpx.Client,
    partition_date: datetime,
) -> SubstationIngestionResult:
    substation_name = resource.name
    log.info(f"Processing substation {substation_name}...")

    # Download
    url = str(resource.url)
    try:
        response = ckan.httpx_get_with_auth(
            url, api_key=api_key, max_retries=max_retries, client=client
        )
        csv_data = response.content
    except Exception as e:
        return SubstationIngestionResult(
            substation_name=substation_name, stage=IngestionStage.DOWNLOAD, error_message=str(e)
        )

    # Process
    try:
        new_df = nged_data.process_live_primary_substation_flows(csv_data)
    except MissingCorePowerVariablesError:
        log.info(f"Skipping substation {substation_name} because it lacks MW/MVA data.")
        # Return success but with no dataframe, so it gets recorded as processed
        # but nothing is merged into Delta Lake.
        return SubstationIngestionResult(
            substation_name=substation_name, stage=IngestionStage.SUCCESS, df=None
        )
    except Exception as e:
        # Handle other errors normally
        try:
            csv_snippet = "\n".join(csv_data.decode("utf-8", errors="replace").splitlines()[:3])
        except Exception:
            csv_snippet = "Could not decode CSV snippet"
        return SubstationIngestionResult(
            substation_name=substation_name,
            stage=IngestionStage.PROCESSING,
            error_message=str(e),
            csv_snippet=csv_snippet,
        )

    # Add required columns for Delta
    parquet_filename = PurePosixPath(url).stem
    new_df = new_df.with_columns(
        [
            pl.lit(parquet_filename).alias("substation_name"),
            pl.lit(partition_date).cast(pl.Datetime("us", "UTC")).alias("ingested_at"),
        ]
    )

    # Ensure uniform schema
    for col in ["MW", "MVA", "MVAr"]:
        if col not in new_df.columns:
            new_df = new_df.with_columns(pl.lit(None).cast(pl.Float32).alias(col))
        else:
            new_df = new_df.with_columns(pl.col(col).cast(pl.Float32))

    new_df = new_df.select(["timestamp", "substation_name", "MW", "MVA", "MVAr", "ingested_at"])

    return SubstationIngestionResult(
        substation_name=substation_name, stage=IngestionStage.SUCCESS, df=new_df
    )


def _format_failure_metadata(
    failures: list[SubstationIngestionResult],
) -> dict[str, dg.MetadataValue]:
    if not failures:
        return {}

    # Build Markdown Table
    md_table = "| Substation | Stage | Error |\n| :--- | :--- | :--- |\n"
    for failure in failures:
        md_table += (
            f"| {failure.substation_name} | {failure.stage.value} | {failure.error_message} |\n"
        )

    # Build Snippets
    md_snippets = "### Failed CSV Snippets\n"
    for failure in failures:
        if failure.csv_snippet:
            md_snippets += f"**{failure.substation_name}**:\n```text\n{failure.csv_snippet}\n```\n"

    return {"Failure Details": MetadataValue.md(md_table + "\n" + md_snippets)}


@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2026-03-10", end_offset=1),
    check_specs=[
        AssetCheckSpec(name="all_substations_succeeded", asset="live_primary_flows"),
    ],
)
def live_primary_flows(
    context: AssetExecutionContext, config: LivePrimaryFlowsConfig, nged_config: NgedConfig
) -> Iterable[dg.AssetCheckResult | dg.MaterializeResult]:
    """Download and process live primary substation flows from NGED CKAN."""
    partition_date_str = context.partition_key
    partition_date = datetime.strptime(partition_date_str, "%Y-%m-%d")
    delta_path = str(_get_delta_path(nged_config))

    # Check what we've already processed today
    processed_substations = set()
    if not config.force_rerun_all and Path(delta_path).exists():
        try:
            # Query the delta table for today's ingestions
            # We use the parquet_filename (which is what we store in substation_name column)
            processed_df = pl.read_delta(
                delta_path, pyarrow_options={"filters": [("ingested_at", "=", partition_date)]}
            )
            if not processed_df.is_empty():
                processed_substations = set(processed_df["substation_name"].unique().to_list())
        except Exception as e:
            context.log.warning(f"Failed to read Delta table state: {e}")

    api_key = nged_config.to_settings().NGED_CKAN_TOKEN.get_secret_value()
    all_resources = ckan.get_csv_resources_for_live_primary_substation_flows(api_key=api_key)

    # Filter resources
    unprocessed_resources = []
    for r in all_resources:
        parquet_filename = PurePosixPath(str(r.url)).stem
        if config.force_rerun_all or parquet_filename not in processed_substations:
            unprocessed_resources.append(r)

    already_processed_count = len(all_resources) - len(unprocessed_resources)

    # Apply manual filters
    resources_to_process = unprocessed_resources
    if config.substation_names:
        resources_to_process = [
            r for r in resources_to_process if r.name in config.substation_names
        ]
        context.log.info("Filtered to %d substations by name", len(resources_to_process))

    if config.limit:
        resources_to_process = resources_to_process[: config.limit]
        context.log.info("Limited to %d substations", config.limit)

    filtered_out_count = len(unprocessed_resources) - len(resources_to_process)

    # Download concurrently
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=config.max_concurrent_connections) as executor:
            results = list(
                executor.map(
                    lambda r: _download_and_process_substation(
                        r, api_key, config.max_retries, context.log, client, partition_date
                    ),
                    resources_to_process,
                )
            )

    successes = [r for r in results if r.stage == IngestionStage.SUCCESS and r.df is not None]
    failures = [r for r in results if r.stage != IngestionStage.SUCCESS]

    # Merge into Delta Lake
    if successes:
        context.log.info(f"Merging {len(successes)} dataframes into Delta Lake...")
        combined_df = pl.concat([r.df for r in successes if r.df is not None])

        if not Path(delta_path).exists():
            write_deltalake(delta_path, combined_df.to_arrow(), partition_by=["substation_name"])
        else:
            # Retry loop for concurrent writes (just in case, though we are doing one big write here)
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    dt = DeltaTable(delta_path)
                    (
                        dt.merge(
                            source=combined_df.to_arrow(),
                            predicate="s.timestamp = t.timestamp AND s.substation_name = t.substation_name",
                            source_alias="s",
                            target_alias="t",
                        )
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

    metadata: dict[str, dg.MetadataValue] = {
        "Total Resources on CKAN": MetadataValue.int(len(all_resources)),
        "Already Processed": MetadataValue.int(already_processed_count),
        "Filtered Out": MetadataValue.int(filtered_out_count),
        "Successfully Processed Today": MetadataValue.int(len(successes)),
        "Failed": MetadataValue.int(len(failures)),
    }
    metadata.update(_format_failure_metadata(failures))

    if failures:
        context.log.error(f"Found {len(failures)} failures!")
        for f in failures:
            context.log.error(f"Failure: {f.substation_name} - {f.stage} - {f.error_message}")

    yield AssetCheckResult(
        passed=len(failures) == 0,
        check_name="all_substations_succeeded",
        metadata=metadata,
    )

    yield MaterializeResult(metadata=metadata)
