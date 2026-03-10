"""Dagster assets for NGED data."""

import httpx
import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import dagster as dg
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
    define_asset_job,
)
from nged_data import ckan

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
    stage: IngestionStage
    error_message: str | None = None
    csv_snippet: str | None = None


class LivePrimaryFlowsConfig(dg.Config):
    """Configuration for the live primary flows ingestion asset."""

    force_rerun_all: bool = False
    max_concurrent_connections: int = 10
    max_retries: int = 3


def _get_parquet_path(nged_config: NgedConfig, substation_name: str) -> Path:
    settings = nged_config.to_settings()
    return settings.NGED_DATA_PATH / "parquet" / "live_primary_flows" / f"{substation_name}.parquet"


def _get_status_file_path(nged_config: NgedConfig, partition_key: str) -> Path:
    settings = nged_config.to_settings()
    return (
        settings.NGED_DATA_PATH
        / "status"
        / "live_primary_flows"
        / f"processed_{partition_key}.json"
    )


def _load_list_of_processed_substations(nged_config: NgedConfig, partition_key: str) -> set[str]:
    status_file = _get_status_file_path(nged_config, partition_key)
    if status_file.exists():
        with open(status_file) as f:
            return set(json.load(f))
    return set()


def _save_processed_substations(
    nged_config: NgedConfig, partition_key: str, processed_substations: set[str]
) -> None:
    status_file = _get_status_file_path(nged_config, partition_key)
    status_file.parent.mkdir(exist_ok=True, parents=True)
    with open(status_file, "w") as f:
        json.dump(list(processed_substations), f)


def _download_and_process_substation(
    resource: nged_data.CkanResource,
    api_key: str,
    nged_config: NgedConfig,
    max_retries: int,
    log: dg.DagsterLogManager,
    client: httpx.Client,
) -> SubstationIngestionResult:
    substation_name = resource.name
    log.info(f"Processing substation {substation_name}...")
    # 1. Download
    try:
        response = ckan.httpx_get_with_auth(
            str(resource.url), api_key=api_key, max_retries=max_retries, client=client
        )
        csv_data = response.content
    except Exception as e:
        return SubstationIngestionResult(
            substation_name=substation_name, stage=IngestionStage.DOWNLOAD, error_message=str(e)
        )

    # Process
    try:
        new_df = nged_data.process_live_primary_substation_flows(csv_data)
    except Exception as e:
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

    # Merge & Save
    try:
        parquet_filename = PurePosixPath(str(resource.url)).stem
        parquet_path = _get_parquet_path(nged_config, parquet_filename)
        if parquet_path.exists():
            old_df = pl.read_parquet(parquet_path)
            last_timestamp = old_df.select("timestamp").max().item()
            new_df_filtered = new_df.filter(pl.col("timestamp") > last_timestamp)

            if new_df_filtered.is_empty():
                return SubstationIngestionResult(
                    substation_name=substation_name, stage=IngestionStage.SUCCESS
                )

            merged_df = (
                pl.concat([old_df, new_df_filtered]).unique(subset="timestamp").sort(by="timestamp")
            )
        else:
            parquet_path.parent.mkdir(exist_ok=True, parents=True)
            merged_df = new_df

        merged_df.write_parquet(parquet_path, compression="zstd")
        return SubstationIngestionResult(
            substation_name=substation_name, stage=IngestionStage.SUCCESS
        )
    except Exception as e:
        return SubstationIngestionResult(
            substation_name=substation_name, stage=IngestionStage.STORAGE, error_message=str(e)
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
    partition_key = context.partition_key
    if config.force_rerun_all:
        processed_substations = set([])
    else:
        processed_substations = _load_list_of_processed_substations(nged_config, partition_key)

    api_key = nged_config.to_settings().NGED_CKAN_TOKEN.get_secret_value()
    all_resources = ckan.get_csv_resources_for_live_primary_substation_flows(api_key=api_key)

    resources_to_process = [r for r in all_resources if r.name not in processed_substations]
    skipped_count = len(all_resources) - len(resources_to_process)

    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=config.max_concurrent_connections) as executor:
            results = list(
                executor.map(
                    lambda r: _download_and_process_substation(
                        r, api_key, nged_config, config.max_retries, context.log, client
                    ),
                    resources_to_process,
                )
            )

    successes = [r.substation_name for r in results if r.stage == IngestionStage.SUCCESS]
    failures = [r for r in results if r.stage != IngestionStage.SUCCESS]

    processed_substations.update(successes)
    _save_processed_substations(nged_config, partition_key, processed_substations)

    metadata: dict[str, dg.MetadataValue] = {
        "Total Resources on CKAN": MetadataValue.int(len(all_resources)),
        "Skipped (Already Processed)": MetadataValue.int(skipped_count),
        "Successfully Processed Today": MetadataValue.int(len(successes)),
        "Failed": MetadataValue.int(len(failures)),
    }
    metadata.update(_format_failure_metadata(failures))

    yield AssetCheckResult(
        passed=len(failures) == 0,
        check_name="all_substations_succeeded",
        metadata=metadata,
    )

    yield MaterializeResult(metadata=metadata)


update_live_primary_flows = define_asset_job(
    name="update_live_primary_flows",
    selection=[live_primary_flows],
    executor_def=dg.in_process_executor,
)
