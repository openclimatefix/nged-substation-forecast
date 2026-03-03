from datetime import datetime, timezone

from contracts.config import NWP_DATA_PATH
from dagster import AssetExecutionContext, DailyPartitionsDefinition, asset
from dynamical_data import download_and_scale_ecmwf

# Daily partitions starting from 2024-04-01
weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01", end_offset=1)


@asset(partitions_def=weather_partitions, op_tags={"dagster/max_runtime_concurrency": 1})
def ecmwf_ens_forecast(context: AssetExecutionContext) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    context.log.info(f"Downloading ECMWF ENS for {partition_key}")
    scaled_df = download_and_scale_ecmwf(nwp_init_time)

    # Save as partitioned Parquet files: data/nwp/ecmwf/ens/YYYY-MM-DDTHHZ.parquet
    # NWP_DATA_PATH is the base NWP directory
    output_dir = NWP_DATA_PATH / "ecmwf" / "ens"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{nwp_init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    output_path = output_dir / filename

    scaled_df.write_parquet(output_path)

    context.log.info(f"Saved {len(scaled_df)} rows to {output_path}")
