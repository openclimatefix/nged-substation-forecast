"""Dagster assets for NGED data."""

from pathlib import Path, PurePosixPath
from typing import Final

import polars as pl
from dagster import (
    AddDynamicPartitionsRequest,
    AssetExecutionContext,
    Config,
    DefaultSensorStatus,
    DynamicPartitionsDefinition,
    MultiPartitionKey,
    MultiPartitionsDefinition,
    RunConfig,
    RunRequest,
    SensorEvaluationContext,
    SensorResult,
    asset,
    define_asset_job,
    sensor,
)
from nged_data import ckan
from nged_data.process_flows import process_live_primary_substation_flows

# Define Partitions
# We use Multi-Partitions so every day's download is saved uniquely by (Date, Name)
last_modified_dates_def = DynamicPartitionsDefinition(name="last_modified_dates")
substation_names_def = DynamicPartitionsDefinition(name="substation_names")
composite_def = MultiPartitionsDefinition(
    {"last_modified_date": last_modified_dates_def, "substation_name": substation_names_def}
)


class CkanCsvConfig(Config):
    url: str


@asset(partitions_def=composite_def)
def live_primary_csv(context: AssetExecutionContext, config: CkanCsvConfig) -> Path:
    # Retrieve the keys
    partition = composite_def.get_partition_key_from_str(context.partition_key).keys_by_dimension
    last_modified_date_str = partition["last_modified_date"]
    substation_name = partition["substation_name"]
    csv_filename = PurePosixPath(config.url).name

    context.log.info(f"Downloading {substation_name} from {config.url}")

    # Get CSV from CKAN
    response = ckan.httpx_get_with_auth(config.url)
    response.raise_for_status()

    # Save CSV file to disk
    # TODO(Jack): Don't save here? Instead, return the CSV file and let an IO manager save it??
    dst_full_path = Path(
        Path("data") / "NGED" / "raw" / "live_primary_flows" / last_modified_date_str / csv_filename
    )
    dst_full_path.parent.mkdir(exist_ok=True, parents=True)
    dst_full_path.write_bytes(response.content)

    return dst_full_path


@asset(partitions_def=composite_def)
def live_primary_parquet(context: AssetExecutionContext, live_primary_csv: Path) -> None:
    df_of_new_data = process_live_primary_substation_flows(live_primary_csv)
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
    merged_df = merged_df.unique(subset="timestamp")
    merged_df = merged_df.sort(by="timestamp")
    merged_df.write_parquet(parquet_path, compression="zstd")


update_live_primary_flows = define_asset_job(
    name="update_live_primary_flows", selection=[live_primary_csv, live_primary_parquet]
)


_SECONDS_IN_AN_HOUR: Final[int] = 60 * 60


@sensor(
    job=update_live_primary_flows,
    default_status=DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=_SECONDS_IN_AN_HOUR * 6,
)
def live_primaries_sensor(context: SensorEvaluationContext) -> SensorResult:
    # Retrieve the full list of primary substation_names and URLs of CSVs
    ckan_resources = ckan.get_csv_resources_for_live_primary_substation_flows()

    selected_for_testing = [  # TODO(Jack): Remove these after testing!
        "Albrighton 11Kv Primary Transformer Flows",
        "Alderton 11Kv Primary Transformer Flows",
        "Alveston 11Kv Primary Transformer Flows",
        "Bayston Hill 11Kv Primary Transformer Flows",
        "Bearstone 11Kv Primary Transformer Flows",
    ]
    ckan_resources = [r for r in ckan_resources if r.name in selected_for_testing]

    run_requests: list[RunRequest] = []
    substation_names: set[str] = set()
    last_modified_date_strings: set[str] = set()
    for resource in ckan_resources:
        substation_name = resource.name
        csv_url = str(resource.url)
        last_modified_date_str = resource.last_modified.strftime("%Y-%m-%d")

        partition_key = MultiPartitionKey(
            {"last_modified_date": last_modified_date_str, "substation_name": substation_name}
        )

        run_requests.append(
            RunRequest(
                # This unique run_key prevents the sensor from triggering duplicate
                # runs for the same data on every tick.
                run_key=f"{last_modified_date_str}|{substation_name}",
                partition_key=partition_key,
                run_config=RunConfig(ops={"live_primary_csv": CkanCsvConfig(url=csv_url)}),
                tags={
                    "nged/substation_name": substation_name,
                    "nged/substation_type": "primary",
                },
            )
        )

        substation_names.add(substation_name)
        last_modified_date_strings.add(last_modified_date_str)

    return SensorResult(
        dynamic_partitions_requests=[
            AddDynamicPartitionsRequest(
                partitions_def_name="substation_names",
                partition_keys=sorted(substation_names),
            ),
            AddDynamicPartitionsRequest(
                partitions_def_name="last_modified_dates",
                partition_keys=sorted(last_modified_date_strings),
            ),
        ],
        run_requests=run_requests,
    )
