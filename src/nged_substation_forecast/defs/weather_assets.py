from datetime import datetime, timezone

from contracts.config import NWP_DATA_PATH
from dagster import AssetExecutionContext, DailyPartitionsDefinition, asset
from dynamical_data import download_and_scale_ecmwf

# Daily partitions starting from 2024-04-01
weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01", end_offset=1)


@asset(partitions_def=weather_partitions)
def ecmwf_ens_forecast(context: AssetExecutionContext) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    context.log.info(f"Downloading ECMWF ENS for {partition_key}")
    scaled_df = download_and_scale_ecmwf(nwp_init_time)
    scaled_df.write_delta(NWP_DATA_PATH, mode="append")

    context.log.info(f"Saved {len(scaled_df)} rows to {NWP_DATA_PATH}")
