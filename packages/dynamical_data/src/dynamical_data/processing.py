import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import h3.api.numpy_int as h3
import icechunk
import numpy as np
import patito as pt
import polars as pl
import polars_h3 as plh3
import shapely
import xarray as xr
from contracts.data_schemas import Nwp
from shapely.geometry.base import BaseGeometry

from .scaling import load_scaling_params, scale_to_uint8

H3_RES: Final[int] = 5
GRID_SIZE: Final[float] = 0.25


def get_gb_h3_grid(geojson_path: Path) -> pl.DataFrame:
    """Generate H3 grid for Great Britain from a GeoJSON file."""
    with open(geojson_path) as f:
        file_contents = f.read()

    shape: BaseGeometry = shapely.from_geojson(file_contents)

    # Buffer by 0.25 degrees to catch islands/coasts
    # This is expensive! It takes about 30 seconds.
    # TODO: Let's not do this for every NWP init. Instead let's compute the h3 grid for GB once, and
    # save that to disk as a Parquet? Maybe in Dagster, to make it easy to run for different
    # geographies.
    shape = shape.buffer(0.25)

    cells = h3.geo_to_cells(shape, res=H3_RES)
    df = pl.DataFrame({"h3_index": list(cells)}, schema={"h3_index": pl.UInt64}).sort("h3_index")

    # Pre-compute the proportion mapping
    df_with_counts = compute_h3_grid_weights(df)

    return df_with_counts


def compute_h3_grid_weights(df: pl.DataFrame) -> pl.DataFrame:
    """Computes the proportion mapping for H3 grid cells."""
    return (
        df.with_columns(h3_res7=plh3.cell_to_children("h3_index", 7))
        .explode("h3_res7")
        .with_columns(
            nwp_lat=((plh3.cell_to_lat("h3_res7") + (GRID_SIZE / 2)) / GRID_SIZE).floor()
            * GRID_SIZE,
            nwp_lng=((plh3.cell_to_lng("h3_res7") + (GRID_SIZE / 2)) / GRID_SIZE).floor()
            * GRID_SIZE,
        )
        .group_by(["h3_index", "nwp_lat", "nwp_lng"])
        .len()
        .with_columns(total=pl.col("len").sum().over("h3_index"))
        .with_columns(proportion=pl.col("len") / pl.col("total"))
    )


def calculate_wind_speed_and_direction(
    df: pl.DataFrame, u_col: str, v_col: str, speed_name: str, dir_name: str
) -> pl.DataFrame:
    """Convert u and v wind components to speed and direction."""
    u = df[u_col]
    v = df[v_col]

    speed = (u**2 + v**2) ** 0.5
    # atan2(u, v) gives angle from north clockwise.
    # To get "from" direction: (angle + 180) % 360
    direction = (np.arctan2(u, v) * 180 / np.pi + 180) % 360

    return df.with_columns(
        [
            pl.Series(speed_name, speed).cast(pl.Float32),
            pl.Series(dir_name, direction).cast(pl.Float32),
        ]
    )


def download_and_scale_ecmwf(nwp_init_time: datetime) -> pt.DataFrame[Nwp]:
    # TODO: This geojson should be moved into a new `geo` package.
    geojson_path = Path("packages/dynamical_data/england_scotland_wales.geojson")

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

    utc_nwp_init_time = np.datetime64(nwp_init_time.astimezone(timezone.utc).replace(tzinfo=None))

    if utc_nwp_init_time not in ds.init_time.values:
        raise ValueError(f"{utc_nwp_init_time} is not in ds.init_time.values")

    loaded_ds = download_ecmwf(utc_nwp_init_time, ds, h3_grid)
    return process_ecmwf_dataset(nwp_init_time=nwp_init_time, loaded_ds=loaded_ds, h3_grid=h3_grid)


def download_ecmwf(
    nwp_init_time: np.datetime64,
    ds: xr.Dataset,
    h3_grid: pl.DataFrame,
) -> xr.Dataset:
    """Download and process ECMWF data for a specific initialization time."""

    # Find spatial bounds from grid
    min_lat, max_lat, min_lng, max_lng = h3_grid.select(
        min_lat=pl.col("nwp_lat").min(),
        max_lat=pl.col("nwp_lat").max(),
        min_lng=pl.col("nwp_lng").min(),
        max_lng=pl.col("nwp_lng").max(),
    ).row(0)

    # Robust slicing: xarray slice(a, b) is sensitive to coordinate direction.
    # We check the first two elements to determine if latitude is ascending or descending.
    lat_is_descending = ds.latitude.values[0] > ds.latitude.values[1]
    lat_slice = slice(max_lat, min_lat) if lat_is_descending else slice(min_lat, max_lat)

    # Xarray datasets from Dynamical are usually tz-naive UTC.
    # If nwp_init_time is aware, we need to make it naive for the selection.
    ds_cropped = ds.sel(
        latitude=lat_slice, longitude=slice(min_lng, max_lng), init_time=nwp_init_time
    )

    def download_array(var_name: str) -> dict[str, xr.DataArray]:
        return {var_name: ds_cropped[var_name].compute()}

    data_arrays: dict[str, xr.DataArray] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_array, str(name)) for name in ds_cropped.data_vars.keys()
        ]
        for future in concurrent.futures.as_completed(futures):
            data_arrays.update(future.result())

    return xr.Dataset(data_arrays)


def process_ecmwf_dataset(
    nwp_init_time: datetime,
    loaded_ds: xr.Dataset,
    h3_grid: pl.DataFrame,
) -> pt.DataFrame[Nwp]:
    lat_grid, lon_grid = np.meshgrid(
        loaded_ds.latitude.values, loaded_ds.longitude.values, indexing="ij"
    )

    dfs = []
    for lead_time in loaded_ds.lead_time.values:
        for ensemble_member in loaded_ds.ensemble_member.values:
            member_ds = loaded_ds.sel(lead_time=lead_time, ensemble_member=ensemble_member)

            # TODO: Don't do all this computation on every loop!
            nwp_data = {name: arr.values.ravel() for name, arr in member_ds.items()}
            nwp_data.update({"longitude": lon_grid.ravel(), "latitude": lat_grid.ravel()})

            nwp_df = pl.DataFrame(nwp_data)

            joined = h3_grid.join(
                nwp_df, left_on=["nwp_lng", "nwp_lat"], right_on=["longitude", "latitude"]
            )

            all_nwp_vars = list(member_ds.data_vars.keys())
            categorical_vars = ["categorical_precipitation_type_surface"]
            numeric_vars = [v for v in all_nwp_vars if v not in categorical_vars]

            # Aggregate to H3 resolution 5

            processed = (
                joined.with_columns(
                    (pl.col([str(x) for x in numeric_vars]).fill_nan(None) * pl.col("proportion"))
                )
                .group_by("h3_index")
                .agg(
                    [
                        pl.when(pl.col(str(x)).null_count() < pl.count())
                        # The .then().otherwise() pattern ensures that any aggregations consisting
                        # entirely of nulls (common at step 0) are null after aggregation.
                        .then(pl.col(str(x)).sum())
                        .otherwise(None)
                        .alias(str(x))
                        for x in numeric_vars
                    ]
                    + [pl.col(categorical_vars).mode().first()],
                )
                .with_columns(
                    valid_time=pl.lit(
                        nwp_init_time + lead_time.astype("timedelta64[s]").item()
                    ).cast(pl.Datetime("us", "UTC")),
                    init_time=pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")),
                    ensemble_member=pl.lit(ensemble_member, dtype=pl.UInt16),
                    nwp_source=pl.lit("ecmwf_ens"),
                )
            )

            # Convert wind
            processed = calculate_wind_speed_and_direction(
                processed,
                "wind_u_10m",
                "wind_v_10m",
                "wind_speed_10m",
                "wind_direction_10m",
            )
            processed = calculate_wind_speed_and_direction(
                processed,
                "wind_u_100m",
                "wind_v_100m",
                "wind_speed_100m",
                "wind_direction_100m",
            )

            # Drop raw wind components
            processed = processed.drop(["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"])

            dfs.append(processed)

    processed_df = pl.concat(dfs)

    scaling_params = load_scaling_params(
        Path("packages/dynamical_data/scaling/ecmwf_scaling_params.csv")
    )
    scaled_df = scale_to_uint8(processed_df, scaling_params)

    return Nwp.validate(scaled_df)
