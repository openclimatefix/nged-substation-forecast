import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple, cast, TYPE_CHECKING

import dagster as dg
import httpx
import polars as pl
from contracts.settings import Settings
from nged_data import (
    ckan,
    process_live_primary_substation_power_flows,
    scan_delta_table,
)

if TYPE_CHECKING:
    from ..defs.nged_assets import LivePrimaryFlowsConfig


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


class SubstationResource(NamedTuple):
    """A resource for a substation's live telemetry."""

    substation_number: int
    url: str


def get_delta_path(settings: Settings) -> Path:
    return settings.nged_data_path / "delta" / "live_primary_flows"


def download_and_process_substation(
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


def format_failure_metadata(
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

    return {"Failure Details": dg.MetadataValue.md(md_table + "\n" + md_snippets)}


def get_processed_substations(
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


def filter_resources(
    all_resources: list[SubstationResource],
    processed_substations: set[int],
    config: "LivePrimaryFlowsConfig",
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


def download_and_process_all(
    resources: list[SubstationResource],
    api_key: str,
    config: "LivePrimaryFlowsConfig",
    log: dg.DagsterLogManager,
    partition_date: datetime,
) -> list[SubstationIngestionResult]:
    """Download and process multiple substations concurrently."""
    with httpx.Client() as client:
        with ThreadPoolExecutor(max_workers=config.max_concurrent_connections) as executor:
            return list(
                executor.map(
                    lambda r: download_and_process_substation(
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


def merge_to_delta(
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
