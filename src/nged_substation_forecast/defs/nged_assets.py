"""Dagster assets for NGED data."""

from datetime import datetime
from pathlib import Path, PurePosixPath

import dagster as dg
import nged_data
import polars as pl
from nged_substation_forecast.config_resource import NgedConfig
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MultiPartitionsDefinition,
    asset,
    define_asset_job,
)
from nged_data import ckan

last_modified_dates_def = DynamicPartitionsDefinition(name="last_modified_dates")
substation_names_def = DynamicPartitionsDefinition(name="substation_names")
composite_def = MultiPartitionsDefinition(
    {"last_modified_date": last_modified_dates_def, "substation_name": substation_names_def}
)


def _json_filename_for_ckan_resource(
    config: NgedConfig, last_modified_date: str, substation_name: str
) -> Path:
    settings = config.to_settings()
    return (
        settings.NGED_DATA_PATH
        / "raw"
        / "live_primary_flows"
        / last_modified_date
        / f"{substation_name} CKAN Resource.json"
    )


def _local_csv_filename(config: NgedConfig, last_modified_date: str, substation_name: str) -> Path:
    settings = config.to_settings()
    return (
        settings.NGED_DATA_PATH
        / "raw"
        / "live_primary_flows"
        / last_modified_date
        / f"{substation_name}.csv"
    )


@asset
def list_resources_for_live_primaries(context: AssetExecutionContext, config: NgedConfig):
    settings = config.to_settings()
    substation_names: set[str] = set()
    last_modified_date_strings: set[str] = set()
    for resource in ckan.get_csv_resources_for_live_primary_substation_flows(
        api_key=settings.NGED_CKAN_TOKEN.get_secret_value()
    ):
        last_modified_date_str = resource.last_modified.strftime("%Y-%m-%d")
        last_modified_date_strings.add(last_modified_date_str)
        substation_names.add(resource.name)

        json_string = resource.model_dump_json()
        json_path = _json_filename_for_ckan_resource(config, last_modified_date_str, resource.name)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json_string)

    context.instance.add_dynamic_partitions("substation_names", list(substation_names))
    context.instance.add_dynamic_partitions("last_modified_dates", list(last_modified_date_strings))


@asset(partitions_def=composite_def, deps=[list_resources_for_live_primaries])
def live_primary_csv(context: AssetExecutionContext, config: NgedConfig) -> Path:
    settings = config.to_settings()
    partition = composite_def.get_partition_key_from_str(context.partition_key).keys_by_dimension
    last_modified_date_str = partition["last_modified_date"]
    substation_name = partition["substation_name"]

    json_filename = _json_filename_for_ckan_resource(
        config, last_modified_date_str, substation_name
    )
    context.log.info(f"Loading {json_filename}...")
    json_text = json_filename.read_text()
    resource = nged_data.CkanResource.model_validate_json(json_text)

    csv_url = str(resource.url)
    context.log.info(f"Downloading CSV for {substation_name} from {resource.url}")
    response = ckan.httpx_get_with_auth(
        csv_url, api_key=settings.NGED_CKAN_TOKEN.get_secret_value()
    )
    response.raise_for_status()

    csv_filename = PurePosixPath(csv_url).name
    dst_full_path = (
        settings.NGED_DATA_PATH
        / "raw"
        / "live_primary_flows"
        / last_modified_date_str
        / csv_filename
    )
    dst_full_path.parent.mkdir(exist_ok=True, parents=True)
    dst_full_path.write_bytes(response.content)

    return dst_full_path


@asset(partitions_def=composite_def)
def live_primary_parquet(
    context: AssetExecutionContext, live_primary_csv: Path, config: NgedConfig
) -> None:
    settings = config.to_settings()
    new_df = nged_data.process_live_primary_substation_flows(live_primary_csv)
    parquet_path = (
        settings.NGED_DATA_PATH
        / "parquet"
        / "live_primary_flows"
        / live_primary_csv.with_suffix(".parquet").name
    )
    if parquet_path.exists():
        old_df = pl.read_parquet(parquet_path)
        last_timestamp: datetime = old_df.select("timestamp").max().item()
        new_df = (
            new_df.filter(pl.col("timestamp") > last_timestamp)
            .unique(subset="timestamp")
            .sort(by="timestamp")
        )
        merged_df = old_df.merge_sorted(new_df, key="timestamp")
    else:
        parquet_path.parent.mkdir(exist_ok=True, parents=True)
        merged_df = new_df
    merged_df.write_parquet(parquet_path, compression="zstd")


update_live_primary_flows = define_asset_job(
    name="update_live_primary_flows",
    selection=[list_resources_for_live_primaries, live_primary_csv, live_primary_parquet],
    executor_def=dg.in_process_executor,
)
