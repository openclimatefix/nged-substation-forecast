import concurrent.futures
import importlib.resources
from datetime import datetime, timezone
from typing import Final

import icechunk
import numpy as np
import patito as pt
import polars as pl
import polars_h3 as plh3
import xarray as xr
from contracts.data_schemas import Nwp

from .scaling import load_scaling_params, scale_to_uint8

H3_RES: Final[int] = 5
GRID_SIZE: Final[float] = 0.25


class MalformedZarrError(ValueError):
    """Raised when an ECMWF Zarr store does not match the expected schema."""


def validate_dataset_schema(ds: xr.Dataset) -> None:
    """Enforces the data contract for incoming ECMWF Zarr stores.

    Ensures all required spatial and temporal dimensions and coordinates exist
    before processing begins.

    Args:
        ds: The xarray Dataset to validate.

    Raises:
        MalformedZarrError: If any required coordinates or variables are missing.
    """
    required_coords = {"latitude", "longitude", "init_time", "lead_time", "ensemble_member"}
    missing_coords = required_coords - set(ds.coords)
    if missing_coords:
        raise MalformedZarrError(f"Dataset is missing required coordinates: {missing_coords}")

    # Check for a minimal set of required data variables
    required_vars = {
        "temperature_2m",
        "wind_u_10m",
        "wind_v_10m",
        "wind_u_100m",
        "wind_v_100m",
        "categorical_precipitation_type_surface",
    }
    missing_vars = required_vars - set(ds.data_vars)
    if missing_vars:
        raise MalformedZarrError(f"Dataset is missing required data variables: {missing_vars}")

    # Check that all data variables have the expected dimensions
    for var_name, da in ds.data_vars.items():
        missing_dims = required_coords - set(da.dims)
        if missing_dims:
            # Note: init_time might be a scalar coordinate after selection,
            # so we allow it to be missing from dimensions of the DataArray
            # if it's present in the Dataset coordinates.
            actual_missing = missing_dims - {"init_time"}
            if actual_missing:
                raise MalformedZarrError(
                    f"Variable '{var_name}' is missing required dimensions: {actual_missing}"
                )


def get_gb_h3_grid() -> pl.DataFrame:
    """Load the pre-computed H3 grid for Great Britain.

    The grid is pre-computed to avoid a 30-second penalty on every ingestion.
    """
    grid_path = importlib.resources.files("dynamical_data.assets").joinpath("gb_h3_grid.parquet")
    with importlib.resources.as_file(grid_path) as path:
        return pl.read_parquet(path)


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
    h3_grid = get_gb_h3_grid()

    # nwp_init_time is aware, we need to make it naive for the selection if it's not already.
    utc_nwp_init_time = np.datetime64(nwp_init_time.astimezone(timezone.utc).replace(tzinfo=None))

    loaded_ds = download_ecmwf(utc_nwp_init_time, h3_grid=h3_grid)
    return process_ecmwf_dataset(nwp_init_time=nwp_init_time, loaded_ds=loaded_ds, h3_grid=h3_grid)


def download_ecmwf(
    nwp_init_time: np.datetime64,
    h3_grid: pl.DataFrame,
    ds: xr.Dataset | None = None,
) -> xr.Dataset:
    """Download and process ECMWF data for a specific initialization time.

    Args:
        nwp_init_time: The initialization time to download.
        h3_grid: The H3 grid to use for spatial bounds.
        ds: Optional xarray Dataset. If None, connects to the production icechunk store.
            This allows dependency injection during testing.
    """
    if ds is None:
        # Connect to the production icechunk store
        storage = icechunk.s3_storage(
            bucket="dynamical-ecmwf-ifs-ens",
            prefix="ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk/",
            region="us-west-2",
            anonymous=True,
        )
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        ds = xr.open_zarr(session.store, chunks=None)

    validate_dataset_schema(ds)

    if nwp_init_time not in ds.init_time.values:
        raise ValueError(f"{nwp_init_time} is not in ds.init_time.values")

    # Find spatial bounds from grid
    min_lat, max_lat, min_lng, max_lng = h3_grid.select(
        min_lat=pl.col("nwp_lat").min(),
        max_lat=pl.col("nwp_lat").max(),
        min_lng=pl.col("nwp_lng").min(),
        max_lng=pl.col("nwp_lng").max(),
    ).row(0)

    # Robust slicing: xarray slice(a, b) is sensitive to coordinate direction.
    # We check the first two elements to determine if latitude is ascending or descending.
    # Single-point forecasts (length 1) do not have a direction, so we bypass the
    # ascending/descending check to avoid an IndexError.
    if len(ds.latitude.values) > 1:
        lat_is_descending = ds.latitude.values[0] > ds.latitude.values[1]
        lat_slice = slice(max_lat, min_lat) if lat_is_descending else slice(min_lat, max_lat)
    else:
        lat_slice = slice(min_lat, max_lat)

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
    """Vectorized processing of ECMWF dataset to H3 grid."""
    validate_dataset_schema(loaded_ds)

    # Convert the entire Xarray Dataset to a Polars DataFrame in a single operation.
    # This avoids thousands of slow loop iterations over lead_time and ensemble_member.
    # We reset the index to make coordinates (lead_time, ensemble_member, latitude, longitude)
    # available as columns.
    nwp_df = pl.from_pandas(loaded_ds.to_dataframe().reset_index())

    # Perform a single spatial join with the pre-computed h3_grid.
    # The join is on latitude and longitude.
    joined = h3_grid.join(
        nwp_df, left_on=["nwp_lng", "nwp_lat"], right_on=["longitude", "latitude"]
    )

    all_nwp_vars = list(loaded_ds.data_vars.keys())
    categorical_vars = ["categorical_precipitation_type_surface"]
    numeric_vars = [v for v in all_nwp_vars if v not in categorical_vars]

    # Aggregate to H3 resolution 5.
    # We group by h3_index, lead_time, and ensemble_member.
    # To avoid biasing the aggregated weather variables towards zero when an H3 cell
    # partially overlaps with missing NWP data (e.g., at the edges of the domain),
    # we compute a true weighted average by dividing the weighted sum by the sum
    # of the proportions of the valid (non-NaN) cells.
    processed = (
        joined.with_columns(
            [
                (pl.col(str(x)).fill_nan(None) * pl.col("proportion")).alias(f"{x}_weighted")
                for x in numeric_vars
            ]
            + [
                pl.when(pl.col(str(x)).fill_nan(None).is_not_null())
                .then(pl.col("proportion"))
                .otherwise(0.0)
                .alias(f"{x}_weight_sum")
                for x in numeric_vars
            ]
        )
        .group_by(["h3_index", "lead_time", "ensemble_member"])
        .agg(
            [
                pl.when(pl.col(f"{x}_weighted").null_count() < pl.len())
                .then(pl.col(f"{x}_weighted").sum() / pl.col(f"{x}_weight_sum").sum())
                .otherwise(None)
                .alias(str(x))
                for x in numeric_vars
            ]
            + [pl.col(categorical_vars).mode().first()],
        )
    )

    # Compute valid_time and init_time.
    # lead_time is a timedelta64[ns] in the DataFrame.
    processed = processed.with_columns(
        init_time=pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")),
        valid_time=(
            pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")) + pl.col("lead_time")
        ).cast(pl.Datetime("us", "UTC")),
        ensemble_member=pl.col("ensemble_member").cast(pl.UInt8),
    ).drop("lead_time")

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

    # Load scaling parameters using importlib.resources
    scaling_params_path = importlib.resources.files("dynamical_data.assets").joinpath(
        "ecmwf_scaling_params.csv"
    )
    with importlib.resources.as_file(scaling_params_path) as path:
        scaling_params = load_scaling_params(path)

    scaled_df = scale_to_uint8(processed, scaling_params)
    scaled_df = scaled_df.sort(by=["init_time", "valid_time", "ensemble_member", "h3_index"])

    return Nwp.validate(scaled_df, drop_superfluous_columns=True)
