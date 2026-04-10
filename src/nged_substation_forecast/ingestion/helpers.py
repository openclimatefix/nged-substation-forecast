import random
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple, cast

import dagster as dg
import polars as pl
from contracts.settings import Settings
from ..utils import scan_delta_table


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
    return settings.nged_data_path / "delta" / "raw_power_time_series"


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
