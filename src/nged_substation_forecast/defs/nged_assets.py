"""Dagster assets for NGED data."""

from pathlib import Path, PurePosixPath

import nged_data
import polars as pl
from dagster import (
    AssetExecutionContext,
    DynamicPartitionsDefinition,
    MultiPartitionsDefinition,
    asset,
    define_asset_job,
)
from nged_data import ckan

# Define Partitions
# We use Multi-Partitions so every day's download is saved uniquely by (Date, Name)
last_modified_dates_def = DynamicPartitionsDefinition(name="last_modified_dates")
substation_names_def = DynamicPartitionsDefinition(name="substation_names")
composite_def = MultiPartitionsDefinition(
    {"last_modified_date": last_modified_dates_def, "substation_name": substation_names_def}
)


def _json_filename_for_ckan_resource(last_modified_date: str, substation_name: str) -> Path:
    # TODO(Jack): Data path should be configurable.
    return (
        Path("data")
        / "NGED"
        / "raw"
        / "live_primary_flows"
        / last_modified_date
        / f"{substation_name} CKAN Resource.json"
    )


def _local_csv_filename(last_modified_date: str, substation_name: str) -> Path:
    # TODO(Jack): Data path should be configurable.
    return (
        Path("data")
        / "NGED"
        / "raw"
        / "live_primary_flows"
        / last_modified_date
        / f"{substation_name}.csv"
    )


@asset
def list_resources_for_live_primaries(context: AssetExecutionContext):
    substation_names: set[str] = set()
    last_modified_date_strings: set[str] = set()
    for resource in ckan.get_csv_resources_for_live_primary_substation_flows():
        last_modified_date_str = resource.last_modified.strftime("%Y-%m-%d")
        last_modified_date_strings.add(last_modified_date_str)
        substation_names.add(resource.name)

        # Save JSON representation of CKAN Resource:
        json_string = resource.model_dump_json()
        json_path = _json_filename_for_ckan_resource(last_modified_date_str, resource.name)
        json_path.write_text(json_string)

    # Update dynamic partitions
    context.instance.add_dynamic_partitions("substation_names", substation_names)
    context.instance.add_dynamic_partitions("last_modified_dates", last_modified_date_strings)


@asset(partitions_def=composite_def, deps=[list_resources_for_live_primaries])
def live_primary_csv(context: AssetExecutionContext) -> Path:
    # Retrieve the keys
    partition = composite_def.get_partition_key_from_str(context.partition_key).keys_by_dimension
    last_modified_date_str = partition["last_modified_date"]
    substation_name = partition["substation_name"]

    # Load JSON representation of CKAN resource to get the CSV filename
    json_filename = _json_filename_for_ckan_resource(last_modified_date_str, substation_name)
    context.log.info(f"Loading {json_filename}...")
    json_text = json_filename.read_text()
    resource = nged_data.CkanResource.model_validate_json(json_text)

    # Get CSV from CKAN
    csv_url = str(resource.url)
    context.log.info(f"Downloading CSV for {substation_name} from {resource.url}")
    response = ckan.httpx_get_with_auth(csv_url)
    response.raise_for_status()

    # Save CSV file to disk
    # TODO(Jack): Don't save here? Instead, return the CSV file and let an IO manager save it??
    # TODO(Jack): Data path should be configurable.
    csv_filename = PurePosixPath(csv_url).name
    dst_full_path = (
        Path("data") / "NGED" / "raw" / "live_primary_flows" / last_modified_date_str / csv_filename
    )
    dst_full_path.parent.mkdir(exist_ok=True, parents=True)
    dst_full_path.write_bytes(response.content)

    return dst_full_path


@asset(partitions_def=composite_def)
def live_primary_parquet(context: AssetExecutionContext, live_primary_csv: Path) -> None:
    df_of_new_data = nged_data.process_live_primary_substation_flows(live_primary_csv)
    # TODO(Jack): Data path should be configurable.
    parquet_path = (
        Path("data")
        / "NGED"
        / "parquet"
        / "live_primary_flows"
        / live_primary_csv.with_suffix(".parquet").name
    )
    if parquet_path.exists():
        df_of_old_data = pl.read_parquet(parquet_path)
        merged_df = pl.concat((df_of_old_data, df_of_new_data), how="vertical")
    else:
        parquet_path.parent.mkdir(exist_ok=True, parents=True)
        merged_df = df_of_new_data
    # TODO(Jack): More efficient strategy: If the parquet exists, then copy the new df to only
    # contain rows not in the old df, de-dupe the cropped new df, and append.
    merged_df = merged_df.unique(subset="timestamp")
    merged_df = merged_df.sort(by="timestamp")
    merged_df.write_parquet(parquet_path, compression="zstd")


update_live_primary_flows = define_asset_job(
    name="update_live_primary_flows",
    selection=[list_resources_for_live_primaries, live_primary_csv, live_primary_parquet],
)
