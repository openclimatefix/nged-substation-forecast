from datetime import datetime
from pathlib import Path

import icechunk
import xarray as xr
from contracts.config import NWP_DATA_PATH
from contracts.data_schemas import Nwp
from dagster import AssetExecutionContext, DailyPartitionsDefinition, asset
from dynamical_data.processing import download_and_process_ecmwf, get_gb_h3_grid
from dynamical_data.scaling import load_scaling_params, scale_to_uint8

# Daily partitions starting from 2024-04-01
weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01")


@asset(partitions_def=weather_partitions)
def ecmwf_ens_forecast(context: AssetExecutionContext) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    partition_key = context.partition_key
    nwp_init_time = datetime.strptime(partition_key, "%Y-%m-%d")

    # Path to GeoJSON
    # TODO: This geojson should be moved into a new `geo` package.
    geojson_path = Path("packages/dynamical_data/england_scotland_wales.geojson")
    scaling_csv_path = Path("packages/dynamical_data/scaling/ecmwf_scaling_params.csv")

    context.log.info(f"Processing ECMWF for {partition_key}")

    h3_grid = get_gb_h3_grid(geojson_path)

    storage = icechunk.s3_storage(
        bucket="dynamical-ecmwf-ifs-ens",
        prefix="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, chunks=None)

    if nwp_init_time not in ds.init_time.values:
        raise ValueError(f"{nwp_init_time} is not in ds.init_time.values")

    context.log.info(f"Downloading init_time: {nwp_init_time}")
    processed_df = download_and_process_ecmwf(nwp_init_time, ds, h3_grid)

    scaling_params = load_scaling_params(scaling_csv_path)
    scaled_df = scale_to_uint8(processed_df, scaling_params)

    Nwp.validate(scaled_df)

    scaled_df.write_delta(NWP_DATA_PATH, mode="append", overwrite_schema=True)

    context.log.info(f"Saved {len(scaled_df)} rows to {NWP_DATA_PATH}")
