"""Dagster assets for NGED data."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


class LivePrimaryFlowsConfig(dg.Config):
    """Configuration for the live primary flows ingestion asset."""

    force_rerun_all: bool = False
    max_concurrent_connections: int = 10
    max_retries: int = 3


def _get_parquet_path(config: NgedConfig, substation_name: str) -> Path:
    settings = config.to_settings()
    return settings.NGED_DATA_PATH / "parquet" / "live_primary_flows" / f"{substation_name}.parquet"


def _get_status_file_path(config: NgedConfig, partition_key: str) -> Path:
    settings = config.to_settings()
    return (
        settings.NGED_DATA_PATH
        / "status"
        / "live_primary_flows"
        / f"processed_{partition_key}.json"
    )


@asset(
    partitions_def=DailyPartitionsDefinition(start_date="2026-03-10", end_offset=1),
    check_specs=[
        AssetCheckSpec(name="all_substations_succeeded", asset="live_primary_flows"),
    ],
)
def live_primary_flows(
    context: AssetExecutionContext, config: LivePrimaryFlowsConfig, nged_config: NgedConfig
):
    """Download and process live primary substation flows from NGED CKAN."""
    settings = nged_config.to_settings()
    partition_key = context.partition_key
    status_file = _get_status_file_path(nged_config, partition_key)

    # Load state
    processed_substations = set()
    if not config.force_rerun_all and status_file.exists():
        try:
            with open(status_file) as f:
                processed_substations = set(json.load(f))
        except Exception as e:
            context.log.warning(f"Failed to load status file {status_file}: {e}")

    # Fetch all resources
    api_key = settings.NGED_CKAN_TOKEN.get_secret_value()
    all_resources = ckan.get_csv_resources_for_live_primary_substation_flows(api_key=api_key)

    resources_to_process = [r for r in all_resources if r.name not in processed_substations]
    skipped_count = len(all_resources) - len(resources_to_process)

    def process_substation(resource: nged_data.CkanResource):
        name = resource.name
        # Download with retries
        csv_data = None
        error_msg = None
        for attempt in range(config.max_retries):
            try:
                response = ckan.httpx_get_with_auth(str(resource.url), api_key=api_key)
                response.raise_for_status()
                csv_data = response.content
                break
            except Exception as e:
                error_msg = str(e)
                if attempt < config.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))

        if csv_data is None:
            return name, "Download", error_msg, None

        # Process
        try:
            new_df = nged_data.process_live_primary_substation_flows(csv_data)
        except Exception as e:
            # Extract first ~3 lines for debugging
            try:
                snippet = "\n".join(csv_data.decode("utf-8", errors="replace").splitlines()[:3])
            except:
                snippet = "Could not decode CSV snippet"
            return name, "Processing", str(e), snippet

        # Merge & Save
        try:
            parquet_path = _get_parquet_path(nged_config, name)
            if parquet_path.exists():
                old_df = pl.read_parquet(parquet_path)
                last_timestamp = old_df.select("timestamp").max().item()

                # Filter new data
                new_df_filtered = new_df.filter(pl.col("timestamp") > last_timestamp)

                if new_df_filtered.is_empty():
                    return name, "Success", None, None

                merged_df = (
                    pl.concat([old_df, new_df_filtered])
                    .unique(subset="timestamp")
                    .sort(by="timestamp")
                )
            else:
                parquet_path.parent.mkdir(exist_ok=True, parents=True)
                merged_df = new_df

            merged_df.write_parquet(parquet_path, compression="zstd")
            return name, "Success", None, None
        except Exception as e:
            return name, "Storage", str(e), None

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=config.max_concurrent_connections) as executor:
        results = list(executor.map(process_substation, resources_to_process))

    # Analyze results
    successes = [r[0] for r in results if r[1] == "Success"]
    failures = [r for r in results if r[1] != "Success"]

    # Update state
    processed_substations.update(successes)
    status_file.parent.mkdir(exist_ok=True, parents=True)
    with open(status_file, "w") as f:
        json.dump(list(processed_substations), f)

    # Prepare reporting
    metadata: dict[str, dg.MetadataValue] = {
        "Total Resources on CKAN": MetadataValue.int(len(all_resources)),
        "Skipped (Already Processed)": MetadataValue.int(skipped_count),
        "Successfully Processed Today": MetadataValue.int(len(successes)),
        "Failed": MetadataValue.int(len(failures)),
    }

    if failures:
        # Build Markdown Table
        md_table = "| Substation | Stage | Error |\n| :--- | :--- | :--- |\n"
        for name, stage, err, _ in failures:
            md_table += f"| {name} | {stage} | {err} |\n"

        # Build Snippets
        md_snippets = "### Failed CSV Snippets\n"
        for name, stage, _, snippet in failures:
            if snippet:
                md_snippets += f"**{name}**:\n```text\n{snippet}\n```\n"

        metadata["Failure Details"] = MetadataValue.md(md_table + "\n" + md_snippets)

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
