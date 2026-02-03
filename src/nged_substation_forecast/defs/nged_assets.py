"""Dagster assets for NGED data."""

from datetime import timedelta
from pathlib import Path, PurePosixPath

import polars as pl
from dagster import (
    AddDynamicPartitionsRequest,
    AssetExecutionContext,
    Config,
    DailyPartitionsDefinition,
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
from nged_data.ckan_client import NgedCkanClient, httpx_get_with_auth
from nged_data.process_flows import process_live_primary_substation_flows

# Define Partitions
# We use Multi-Partitions so every day's download is saved uniquely by (Date, Name)
daily_def = DailyPartitionsDefinition(start_date="2026-02-03", timezone="UTC", end_offset=1)
substations_def = DynamicPartitionsDefinition(name="substations")
composite_def = MultiPartitionsDefinition(
    {"last_modified_date": daily_def, "substation": substations_def}
)


class CkanCsvConfig(Config):
    url: str


@asset(partitions_def=composite_def)
def live_primary_csv(context: AssetExecutionContext, config: CkanCsvConfig) -> Path:
    # Retrieve the keys
    partition = composite_def.get_partition_key_from_str(context.partition_key).keys_by_dimension
    last_modified_date_str = partition["last_modified_date"]
    substation_name = partition["substation"]
    csv_filename = PurePosixPath(config.url).name

    context.log.info(f"Downloading {substation_name} from {config.url}")

    # Get CSV from CKAN
    response = httpx_get_with_auth(config.url)
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


download_live_primary_csvs = define_asset_job(
    name="download_live_primary_csvs", selection=[live_primary_csv]
)


@sensor(
    job=download_live_primary_csvs,
    default_status=DefaultSensorStatus.RUNNING,
    minimum_interval_seconds=round(timedelta(days=1).total_seconds()),
)
def live_primaries_sensor(context: SensorEvaluationContext) -> SensorResult:
    # Retrieve the full list of primary substations and URLs of CSVs
    ckan = NgedCkanClient()
    ckan_resources = ckan.get_csv_resources_for_live_primary_substation_flows()

    selected_for_testing = [  # TODO(Jack): Remove these after testing!
        "Albrighton 11Kv Primary Transformer Flows",
        "Alderton 11Kv Primary Transformer Flows",
        "Alveston 11Kv Primary Transformer Flows",
        "Bayston Hill 11Kv Primary Transformer Flows",
        "Bearstone 11Kv Primary Transformer Flows",
    ]
    ckan_resources = [r for r in ckan_resources if r.name in selected_for_testing]

    run_requests = []
    for resource in ckan_resources:
        substation_name = resource.name
        csv_url = str(resource.url)
        last_modified_date_str = resource.last_modified.strftime("%Y-%m-%d")

        partition_key = MultiPartitionKey(
            {"last_modified_date": last_modified_date_str, "substation": substation_name}
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

    substation_names = [resource.name for resource in ckan_resources]
    return SensorResult(
        dynamic_partitions_requests=[
            AddDynamicPartitionsRequest(
                partitions_def_name="substations",
                partition_keys=substation_names,
            )
        ],
        run_requests=run_requests,
    )
