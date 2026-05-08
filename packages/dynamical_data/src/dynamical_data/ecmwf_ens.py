import concurrent.futures
from datetime import datetime, timezone
from typing import Any, Final, Literal, cast

import dynamical_catalog
import numpy as np
import patito as pt
import polars as pl
import xarray as xr
from contracts.common import UTC_DATETIME_DTYPE
from contracts.geo_schemas import H3GridWeights
from contracts.settings import Settings
from contracts.weather_schemas import NwpInMemory

_SETTINGS = Settings()


_ECMWF_ENS_VARS_TO_DOWNLOAD: Final[tuple[str, ...]] = (
    "temperature_2m",
    "dew_point_temperature_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_u_100m",
    "wind_v_100m",
    "pressure_surface",
    "pressure_reduced_to_mean_sea_level",
    "geopotential_height_500hpa",
    "downward_long_wave_radiation_flux_surface",
    "downward_short_wave_radiation_flux_surface",
    "precipitation_surface",
    "categorical_precipitation_type_surface",
)


def download_and_save_ecmwf_ens_run(
    nwp_init_time: datetime,
    h3_grid: pt.DataFrame[H3GridWeights],
) -> None:
    """Download and save one ECMWF run, for a specific initialization time.

    Args:
        nwp_init_time: The initialization time to download. Must be timezone aware.
        h3_grid: The H3 grid weights to use for spatial aggregation.
            This is allows dynamic resolution scaling without hardcoding static file paths.
    """
    loaded_ds = download_ecmwf_ens_run(nwp_init_time=nwp_init_time, h3_grid=h3_grid)
    processed = convert_nwp_xarray_dataset_to_polars_dataframe(
        nwp_init_time=nwp_init_time, ds=loaded_ds, h3_grid=h3_grid
    )


def download_ecmwf_ens_run(
    nwp_init_time: datetime,
    h3_grid: pt.DataFrame[H3GridWeights],
) -> xr.Dataset:
    """Download and process ECMWF data for a specific initialization time.

    Args:
        nwp_init_time: The initialization time to download. Must be timezone aware.
        h3_grid: The H3 grid to use for spatial bounds.
    """
    if h3_grid.is_empty():
        raise ValueError("h3_grid is empty. Cannot download ECMWF data for an empty grid.")

    # nwp_init_time is aware, we need to make it naive for the selection if it's not already.
    utc_nwp_init_time = np.datetime64(nwp_init_time.astimezone(timezone.utc).replace(tzinfo=None))

    ds = dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None)

    # Cast to xr.Dataset to satisfy the type checker, as indexing with a list
    # can sometimes be misidentified as returning a DataArray.
    ds = cast(xr.Dataset, ds[_ECMWF_ENS_VARS_TO_DOWNLOAD])

    if utc_nwp_init_time not in ds.init_time.values:
        raise ValueError(f"{utc_nwp_init_time} is not in ds.init_time.values")

    # Check for empty coordinates before computing bounds to fail gracefully.
    if ds.longitude.size == 0 or ds.latitude.size == 0:
        raise ValueError("Dataset has empty longitude or latitude coordinates.")

    # Validate longitude range.
    # NOTE: Dynamical.org converts the longitude range to [-180, 180].
    if ds.longitude.min() < -180 or ds.longitude.max() > 180:
        raise ValueError("Dataset longitude must be in the range [-180, 180]")

    min_lat, max_lat, min_lon, max_lon = h3_grid.select(
        min_lat=pl.col("nwp_lat").min(),
        max_lat=pl.col("nwp_lat").max(),
        min_lon=pl.col("nwp_lon").min(),
        max_lon=pl.col("nwp_lon").max(),
    ).row(0)

    lat_slice = _calc_slice_for_lat_or_lng("latitude", ds, min_lat, max_lat)
    lon_slice = _calc_slice_for_lat_or_lng("longitude", ds, min_lon, max_lon)

    # NOTE: This will fail if the region crosses the anti-meridian. But we do not anticipate
    # forecasting near the anti-meridian.
    ds_sliced = ds.sel(latitude=lat_slice, longitude=lon_slice, init_time=utc_nwp_init_time)

    # Explicitly check for an empty spatial intersection after slicing.
    # This prevents downstream KeyErrors during DataFrame conversion.
    if ds_sliced.longitude.size == 0 or ds_sliced.latitude.size == 0:
        raise ValueError("No spatial overlap found between H3 grid and NWP dataset.")

    def download_array(var_name: str) -> dict[str, xr.DataArray]:
        return {var_name: ds_sliced[var_name].compute()}

    # The download is I/O bound (S3 network requests). We use a ThreadPoolExecutor to parallelize
    # network latency across multiple variables. A ProcessPoolExecutor would be less efficient here
    # due to the high serialization overhead of Xarray objects between processes.
    # TODO: This block uses an insane amount of RAM. Use raw Zarr, not xarray. See issue 93.
    data_arrays: dict[str, xr.DataArray] = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(download_array, str(name)) for name in ds_sliced.data_vars.keys()
        ]
        for future in concurrent.futures.as_completed(futures):
            data_arrays.update(future.result())

    return xr.Dataset(data_arrays)


def _calc_slice_for_lat_or_lng(
    coord_name: Literal["latitude", "longitude"],
    ds: xr.Dataset,
    min_coord: float,
    max_coord: float,
) -> slice:
    """Robust slicing: xarray slice(a, b) is sensitive to coordinate direction.

    We determine if latitude/longitude are ascending or descending. In ECMWF ENS on Dynamical,
    latitude goes from +90 to -90, and longitude goes from -180 to +179.8.
    """
    if min_coord == max_coord:
        raise ValueError(f"{min_coord=} cannot be equal to {max_coord=} for {coord_name}")

    coord_array = ds[coord_name].values
    if len(coord_array) <= 1:
        raise ValueError(
            f"ds.{coord_name}.values must have multiple values. Found {len(coord_array)} values"
        )

    is_ascending = coord_array[0] < coord_array[-1]
    return slice(min_coord, max_coord) if is_ascending else slice(max_coord, min_coord)


def convert_nwp_xarray_dataset_to_polars_dataframe(
    nwp_init_time: datetime,
    ds: xr.Dataset,
    h3_grid: pt.DataFrame[H3GridWeights],
) -> pt.DataFrame[NwpInMemory]:
    """Vectorized processing of ECMWF dataset to H3 grid."""
    # Precompute latitude and longitude grids
    lat_grid, lon_grid = np.meshgrid(
        ds.latitude.values.astype(np.float32),
        ds.longitude.values.astype(np.float32),
        indexing="ij",
    )
    lat_grid_raveled = lat_grid.ravel()
    lon_grid_raveled = lon_grid.ravel()

    # Iterate over lead_time and ensemble_member to process in chunks.
    dfs: list[pl.DataFrame] = []
    for lead_time in ds.lead_time.values:
        for ensemble_member in ds.ensemble_member.values:
            ds_chunk = ds.sel(lead_time=lead_time, ensemble_member=ensemble_member)
            dfs.append(
                _process_chunk(
                    ds_chunk,
                    h3_grid,
                    lat_grid=lat_grid_raveled,
                    lon_grid=lon_grid_raveled,
                )
            )

    df = pl.concat(dfs)

    # Compute valid_time and init_time.
    df = df.with_columns(
        init_time=pl.lit(nwp_init_time).cast(UTC_DATETIME_DTYPE),
        valid_time=(
            pl.lit(nwp_init_time).cast(UTC_DATETIME_DTYPE)
            + pl.col("lead_time").cast(pl.Duration("us"))
        ).cast(UTC_DATETIME_DTYPE),
        ensemble_member=pl.col("ensemble_member").cast(pl.UInt8),
    ).drop("lead_time")

    # TODO: Calculate wind speed and direction, and drop u and v

    # Sort before validation to ensure consistent output order, and to optimise compression.
    df = df.sort(by=["init_time", "valid_time", "ensemble_member", "h3_index"])

    # Validate to ensure the interpolated data matches the expected schema
    return NwpInMemory.validate(df, drop_superfluous_columns=True)


def _process_chunk(
    ds: xr.Dataset,
    h3_grid: pt.DataFrame[H3GridWeights],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> pl.DataFrame:
    """Processes a single chunk of the ECMWF dataset."""
    # Prepare data dictionary
    data_dict: dict[str, Any] = {"latitude": lat_grid, "longitude": lon_grid}

    # Add data variables
    for var_name in ds.data_vars:
        data_dict[str(var_name)] = ds[var_name].values.ravel()

    # Add scalar coordinates
    for coord_name in ["init_time", "lead_time", "ensemble_member"]:
        val = ds[coord_name].values
        # Broadcast to the same length as the flattened arrays
        data_dict[coord_name] = np.full(lat_grid.size, val)

    nwp_df = pl.DataFrame(data_dict)

    joined = h3_grid.join(
        nwp_df, left_on=["nwp_lon", "nwp_lat"], right_on=["longitude", "latitude"], how="left"
    )

    # Aggregate to H3 resolution 5.
    cols_to_aggregate_by = ("h3_index", "lead_time", "ensemble_member")

    # TODO: I think a lot of this aggregation logic is over-complex, and possibly is filling things
    # with zeros when they shouldn't be.
    def weight_sum_expr(x: str) -> pl.Expr:
        return (
            pl.when(pl.col(x).fill_nan(None).is_not_null())
            .then(pl.col("proportion"))
            .otherwise(0.0)  # TODO: This zero makes me nervous!
            .sum()
        )

    # Define variables for aggregation
    all_nwp_vars = [str(v) for v in ds.data_vars]
    numeric_vars = [v for v in all_nwp_vars if v not in NwpInMemory.categorical_var_names]

    # Aggregate numeric variables
    agg_exprs = [
        pl.when(weight_sum_expr(x) > 0)
        .then((pl.col(x).fill_nan(None) * pl.col("proportion")).sum() / weight_sum_expr(x))
        .otherwise(None)
        .cast(pl.Float32)
        .alias(x)
        for x in numeric_vars
    ]
    processed = joined.group_by(cols_to_aggregate_by).agg(agg_exprs)

    # WEIGHTED CATEGORICAL AGGREGATION:
    for c in NwpInMemory.categorical_var_names:
        df_cat = (
            joined.group_by(cols_to_aggregate_by + (c,))
            .agg(pl.col("proportion").sum().alias("weight"))
            .sort(cols_to_aggregate_by + ("weight",))
            .group_by(cols_to_aggregate_by)
            .agg(pl.col(c).last().fill_nan(0).cast(pl.Float32).cast(pl.UInt8))
        )
        processed = processed.join(df_cat, on=cols_to_aggregate_by, how="left")

    return processed
