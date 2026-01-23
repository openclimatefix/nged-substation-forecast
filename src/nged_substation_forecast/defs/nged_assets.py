"""Dagster assets for NGED data."""

from pathlib import Path

import requests
from dagster import (
    AssetExecutionContext,
    AssetObservation,
    Config,
    RetryPolicy,
    StaticPartitionsDefinition,
    asset,
)
from data_nged.ckan_client import NGEDCKANClient
from data_nged.live_primary_data import (
    get_substation_resource_urls,
    read_primary_substation_csv,
)


class RawCSVConfig(Config):
    """Configuration for downloading raw CSV data."""

    raw_data_path: str = "~/data/NGED/CSV"


class ParquetConfig(Config):
    """Configuration for processing data to Parquet."""

    raw_data_path: str = "~/data/NGED/CSV"
    output_path: str = "~/data/NGED/parquet"


def _get_all_substation_names() -> list[str]:
    """Helper to fetch all substation names for partition definition."""
    regions = [
        "live-primary-data---south-wales",
        "live-primary-data---south-west",
        "live-primary-data---west-midlands",
        "live-primary-data---east-midlands",
    ]
    client = NGEDCKANClient()
    names = set()
    for region in regions:
        try:
            resources = get_substation_resource_urls(client, region)
            for r in resources:
                names.add(r.substation_name)
        except Exception:
            pass
    return sorted(list(names)) if names else ["Placeholder"]


substation_partitions = StaticPartitionsDefinition(_get_all_substation_names())


@asset(
    partitions_def=substation_partitions,
    group_name="nged",
    retry_policy=RetryPolicy(max_retries=3, delay=10),
)
def live_primary_flows_csv(
    context: AssetExecutionContext,
    config: RawCSVConfig,
) -> None:
    """Download CSV from CKAN. Save CSV to disk."""
    substation_name = context.partition_key
    client = NGEDCKANClient()
    regions = [
        "live-primary-data---south-wales",
        "live-primary-data---south-west",
        "live-primary-data---west-midlands",
        "live-primary-data---east-midlands",
    ]

    # Find the specific URL for this partition
    url = None
    for region in regions:
        resources = get_substation_resource_urls(client, region)
        for r in resources:
            if r.substation_name == substation_name:
                url = r.url
                break
        if url:
            break

    if not url:
        raise ValueError(f"URL not found for substation {substation_name}")

    dest_path = Path(config.raw_data_path).expanduser() / f"{substation_name}.csv"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    context.log.info(f"Downloading {substation_name} from {url}")
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    dest_path.write_bytes(res.content)


@asset(
    partitions_def=substation_partitions,
    group_name="nged",
    deps=[live_primary_flows_csv],
)
def live_primary_flows_parquet(
    context: AssetExecutionContext,
    config: ParquetConfig,
) -> None:
    """Read CSV from disk. Convert to Parquet. Validate."""
    substation_name = context.partition_key
    csv_path = Path(config.raw_data_path).expanduser() / f"{substation_name}.csv"

    context.log.info(f"Processing {substation_name}")
    df = read_primary_substation_csv(csv_path, substation_name=substation_name)

    if df.is_empty():
        return

    output_path = (
        Path(config.output_path).expanduser()
        / f"substation_name={substation_name}"
        / "data.parquet"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort("timestamp").write_parquet(output_path)

    context.log_event(
        AssetObservation(
            asset_key=context.asset_key,
            partition=substation_name,
            metadata={"row_count": df.height},
        )
    )
