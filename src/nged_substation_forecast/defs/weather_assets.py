from datetime import datetime, timezone

import dagster as dg
from dagster import AssetExecutionContext, DailyPartitionsDefinition, asset, define_asset_job
from dynamical_data.processing import download_and_scale_ecmwf

from nged_substation_forecast.config_resource import NgedConfig

weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01", end_offset=1)


# The `pool="ECMWF"` works in conjunction with `concurrent.pools.default_limit` in
# $DAGSTER_HOME/dagster.yaml to limit the number of times this asset can be run concurrently.
# The ECMWF download script uses a lot of RAM, so it's best to run it one-by-one.
# See: https://docs.dagster.io/guides/operate/managing-concurrency/concurrency-pools
@asset(partitions_def=weather_partitions, pool="ECMWF")
def ecmwf_ens_forecast(context: AssetExecutionContext, config: NgedConfig) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    settings = config.to_settings()
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    context.log.info(f"Downloading ECMWF ENS for {partition_key}")
    scaled_df = download_and_scale_ecmwf(nwp_init_time)

    output_dir = settings.NWP_DATA_PATH / "ECMWF" / "ENS"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{nwp_init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    output_path = output_dir / filename

    scaled_df.write_parquet(output_path, compression="zstd", compression_level=14)

    context.log.info(f"Saved {len(scaled_df)} rows to {output_path}")


update_ecmwf_ens_forecast = define_asset_job(
    name="update_ecmwf_ens_forecast",
    selection=[ecmwf_ens_forecast],
    executor_def=dg.in_process_executor,
)
