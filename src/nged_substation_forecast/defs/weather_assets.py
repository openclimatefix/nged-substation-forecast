import polars as pl
from dagster import AssetExecutionContext, asset, DailyPartitionsDefinition
from datetime import datetime, timedelta
import xarray as xr
import icechunk
import numpy as np
from pathlib import Path

from contracts.config import WEATHER_DATA_PATH
from dynamical_data.processing import download_and_process_ecmwf, get_gb_h3_grid
from dynamical_data.scaling import load_scaling_params, scale_to_uint8

# Daily partitions starting from 2024-04-01
weather_partitions = DailyPartitionsDefinition(start_date="2024-04-01")


@asset(partitions_def=weather_partitions)
def ecmwf_ens_forecast(context: AssetExecutionContext) -> None:
    """Download and process ECMWF ENS forecast for Great Britain."""
    partition_key = context.partition_key
    target_date = datetime.strptime(partition_key, "%Y-%m-%d")

    # Path to GeoJSON (should be handled better in production, but okay for now)
    geojson_path = Path("packages/dynamical_data/england_scotland_wales.geojson")
    scaling_csv_path = Path("packages/dynamical_data/scaling/ecmwf_scaling_params.csv")

    context.log.info(f"Processing ECMWF for {partition_key}")

    # 1. Get H3 grid
    h3_grid = get_gb_h3_grid(geojson_path)

    # 2. Open ECMWF from Dynamical
    storage = icechunk.s3_storage(
        bucket="dynamical-ecmwf-ifs-ens",
        prefix="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
        region="us-west-2",
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, chunks=None)

    # 3. Download and process
    # Note: init_time in the dataset might not exactly match target_date midnight
    # We should find the closest init_time for that date
    available_init_times = ds.init_time.values
    # Filter for times on the target date
    target_init_times = [
        t
        for t in available_init_times
        if np.datetime64(target_date) <= t < np.datetime64(target_date + timedelta(days=1))
    ]

    if not target_init_times:
        context.log.warning(f"No init times found for {partition_key}")
        return

    all_processed = []
    for init_time in target_init_times:
        context.log.info(f"Downloading init_time: {init_time}")
        processed_df = download_and_process_ecmwf(init_time, ds, h3_grid)

        # 4. Scale to uint8
        scaling_params = load_scaling_params(scaling_csv_path)
        scaled_df = scale_to_uint8(processed_df, scaling_params)

        all_processed.append(scaled_df)

    if not all_processed:
        return

    final_df = pl.concat(all_processed)

    # 5. Save to Delta Lake
    # The WEATHER_DATA_PATH from config.py is a Path object
    # deltalake write_deltalake supports local paths
    final_df.write_delta(str(WEATHER_DATA_PATH), mode="append", overwrite_schema=True)

    context.log.info(f"Saved {len(final_df)} rows to {WEATHER_DATA_PATH}")
