"""Dagster assets for NGED data."""

import functools
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


NGED_REGIONS = [
    "live-primary-data---south-wales",
    "live-primary-data---south-west",
    "live-primary-data---west-midlands",
    "live-primary-data---east-midlands",
]


def _fetch_substation_resource_dict() -> dict[str, str]:
    """Fetch all substation resources from CKAN and return as a name->url mapping."""
    client = NGEDCKANClient()
    substation_name_to_url: dict[str, str] = {}
    for region in NGED_REGIONS:
        resources = get_substation_resource_urls(client, region)
        substation_name_to_url.update(resources)
    return substation_name_to_url


@functools.cache
def _get_all_substation_names() -> list[str]:
    """Helper to fetch all substation names for partition definition."""
    mapping = _fetch_substation_resource_dict()
    return sorted(list(mapping.keys()))


substation_partitions = StaticPartitionsDefinition(_get_all_substation_names())


@asset(group_name="nged")
def fetch_substation_resource_urls() -> dict[str, str]:
    """Mapping of substation names to their CKAN CSV URLs.

    This asset is non-partitioned and fetches all URLs from CKAN in a single pass.
    Downstream partitioned assets use this mapping to find their specific URLs
    without re-querying CKAN for every partition.
    """
    return _fetch_substation_resource_dict()


@asset(
    partitions_def=substation_partitions,
    group_name="nged",
    retry_policy=RetryPolicy(max_retries=3, delay=10),
)
def download_csv(
    context: AssetExecutionContext,
    config: RawCSVConfig,
    fetch_substation_resource_urls: dict[str, str],
) -> None:
    """Download CSV from CKAN. Save CSV to disk."""
    substation_name = context.partition_key
    url = fetch_substation_resource_urls[substation_name]

    dest_path = Path(config.raw_data_path).expanduser() / f"{substation_name}.csv"
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    context.log.info(f"Downloading {substation_name} from {url}")
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    dest_path.write_bytes(res.content)


@asset(
    partitions_def=substation_partitions,
    group_name="nged",
    deps=[download_csv],
)
def convert_csv_to_parquet(
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
