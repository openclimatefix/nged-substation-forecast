import os
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import icechunk
import numpy as np
import patito as pt
import polars as pl
import xarray as xr
from contracts.data_schemas import H3GridWeights, Nwp
from contracts.settings import Settings

from .scaling import load_scaling_params, scale_to_uint8

_SETTINGS = Settings()

# Centralized list of required NWP variables to ensure consistency
# between validation and download steps, preventing logic drift.
REQUIRED_NWP_VARS = {
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
}

DEFAULT_AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

ASSETS_PATH = Path(__file__).resolve().parent.parent.parent / "assets"


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
    # These are the variables required by the Nwp schema in contracts.data_schemas
    missing_vars = REQUIRED_NWP_VARS - set(ds.data_vars)
    if missing_vars:
        raise MalformedZarrError(f"Dataset is missing required data variables: {missing_vars}")

    # Check that all data variables have the expected dimensions and order.
    # The canonical order is (latitude, longitude, init_time, lead_time, ensemble_member).
    # Note: init_time might be a scalar coordinate after selection, so we allow it to be missing
    # from dimensions of the DataArray if it's present in the Dataset coordinates.
    expected_dims = ("latitude", "longitude", "init_time", "lead_time", "ensemble_member")
    for var_name, da in ds.data_vars.items():
        # Check for missing dimensions (excluding init_time)
        missing_dims = required_coords - set(da.dims) - {"init_time"}
        if missing_dims:
            raise MalformedZarrError(
                f"Variable '{var_name}' is missing required dimensions: {missing_dims}"
            )

        # Check dimension order (ignoring init_time if it's a scalar coordinate)
        actual_dims = [d for d in da.dims if d in expected_dims]
        expected_dims_filtered = [d for d in expected_dims if d in da.dims]
        if actual_dims != expected_dims_filtered:
            # Transpose the data to the expected order
            ds[var_name] = da.transpose(*expected_dims_filtered)

    # Check coordinate dtypes
    if not np.issubdtype(ds.latitude.dtype, np.floating):
        raise MalformedZarrError(
            f"latitude has wrong dtype: {ds.latitude.dtype}, expected floating"
        )
    if not np.issubdtype(ds.longitude.dtype, np.floating):
        raise MalformedZarrError(
            f"longitude has wrong dtype: {ds.longitude.dtype}, expected floating"
        )
    if not np.issubdtype(ds.init_time.dtype, np.datetime64):
        raise MalformedZarrError(
            f"init_time has wrong dtype: {ds.init_time.dtype}, expected datetime64"
        )


def download_and_scale_ecmwf(
    nwp_init_time: datetime, h3_grid: pt.DataFrame[H3GridWeights]
) -> pl.DataFrame:
    """Download and scale ECMWF data for a specific initialization time.

    Args:
        nwp_init_time: The initialization time to download.
        h3_grid: The H3 grid weights to use for spatial aggregation.
            This is injected via Dagster, allowing dynamic resolution scaling
            without hardcoding static file paths.
    """
    # nwp_init_time is aware, we need to make it naive for the selection if it's not already.
    utc_nwp_init_time = np.datetime64(nwp_init_time.astimezone(timezone.utc).replace(tzinfo=None))

    loaded_ds = download_ecmwf(utc_nwp_init_time, h3_grid=h3_grid)
    processed = process_ecmwf_dataset(
        nwp_init_time=nwp_init_time, loaded_ds=loaded_ds, h3_grid=h3_grid
    )

    # DATA TYPE TRANSITION RATIONALE:
    # 1. Disk (UInt8): Weather variables are stored as scaled 8-bit unsigned integers to save
    #    space and bandwidth.
    # 2. Interpolation (Float64): During spatial weighting and H3 grid joins, we use Float64
    #    to maintain precision and avoid rounding errors during aggregation.
    # 3. Memory/Model (Float32): After processing, we cast to Float32 for memory efficiency
    #    in the ML model.
    #
    # We do NOT cast back to UInt8 in memory because:
    # - It would cause a "staircase" effect by quantizing interpolated values, defeating
    #   the purpose of 30-minute upsampling.
    # - It risks silent underflow during differencing operations (e.g., calculating trends
    #   like temp_trend_6h = current - lagged).

    # Load scaling parameters
    scaling_params_path = ASSETS_PATH / "ecmwf_scaling_params.csv"
    scaling_params = load_scaling_params(scaling_params_path)

    # CIRCULAR VARIABLE SCALING:
    # Exclude categorical variables and wind components from empirical min-max scaling.
    # Storing wind components as Float32 avoids destroying circular topology via
    # min-max scaling and simplifies downstream processing.
    scaling_params = scaling_params.filter(
        ~pl.col("col_name").is_in(
            [
                "categorical_precipitation_type_surface",
                "wind_u_10m",
                "wind_v_10m",
                "wind_u_100m",
                "wind_v_100m",
            ]
        )
    )

    return scale_to_uint8(processed, scaling_params)


def download_ecmwf(
    nwp_init_time: np.datetime64,
    h3_grid: pt.DataFrame[H3GridWeights],
    ds: xr.Dataset | None = None,
) -> xr.Dataset:
    """Download and process ECMWF data for a specific initialization time.

    Args:
        nwp_init_time: The initialization time to download.
        h3_grid: The H3 grid to use for spatial bounds.
        ds: Optional xarray Dataset. If None, connects to the production icechunk store.
            This allows dependency injection during testing.
    """
    if h3_grid.is_empty():
        raise ValueError("h3_grid is empty. Cannot download ECMWF data for an empty grid.")

    if ds is None:
        # Connect to the production icechunk store
        storage = icechunk.s3_storage(
            bucket=_SETTINGS.ecmwf_s3_bucket,
            prefix=_SETTINGS.ecmwf_s3_prefix,
            region=DEFAULT_AWS_REGION,
            anonymous=True,
        )
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session("main")
        # chunks=None is a deliberate design choice to disable Dask and avoid memory overhead.
        #
        # Explicitly setting decode_timedelta=True avoids reliance on Xarray's
        # deprecated automatic decoding of time units, ensuring lead_time is
        # correctly parsed as timedelta64[ns].
        ds = xr.open_zarr(session.store, chunks=None, decode_timedelta=True)

    if ds is None:
        raise MalformedZarrError("Dataset could not be loaded")

    validate_dataset_schema(ds)

    if nwp_init_time not in ds.init_time.values:
        raise ValueError(f"{nwp_init_time} is not in ds.init_time.values")

    # We subset the dataset to only the required variables defined in the Nwp schema
    # to save network bandwidth and memory during the download process.
    # We also include the raw wind components (10u, 10v, 100u, 100v) which are
    # needed for calculating wind speed and direction later.
    # Cast to xr.Dataset to satisfy the type checker, as indexing with a list
    # can sometimes be misidentified as returning a DataArray.
    ds = cast(xr.Dataset, ds[list(REQUIRED_NWP_VARS)])

    # Check for empty coordinates before computing bounds to fail gracefully.
    if ds.longitude.size == 0 or ds.latitude.size == 0:
        raise ValueError("Dataset has empty longitude or latitude coordinates.")

    # Validate longitude range.
    # NOTE: The ECMWF ENS dataset from Dynamical.org already uses the [-180, 180]
    # range for longitude. We only validate this here to ensure the upstream
    # data format hasn't changed unexpectedly, rather than rolling the
    # coordinates ourselves.
    if ds.longitude.min() < -180 or ds.longitude.max() > 180:
        raise ValueError("Dataset longitude must be in the range [-180, 180]")

    # Find spatial bounds from grid
    min_lat, max_lat = h3_grid.select(
        pl.col("nwp_lat").min().alias("min_lat"),
        pl.col("nwp_lat").max().alias("max_lat"),
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

    # NOTE: Anti-meridian logic has been removed as we do not anticipate
    # forecasting near the anti-meridian.
    ds_cropped = ds.sel(
        latitude=lat_slice,
        longitude=slice(h3_grid.get_column("nwp_lng").min(), h3_grid.get_column("nwp_lng").max()),
        init_time=nwp_init_time,
    )

    # Explicitly check for an empty spatial intersection after slicing.
    # This prevents downstream KeyErrors during DataFrame conversion.
    if ds_cropped.longitude.size == 0 or ds_cropped.latitude.size == 0:
        raise ValueError("No spatial overlap found between H3 grid and NWP dataset.")

    def download_array(var_name: str) -> dict[str, xr.DataArray]:
        return {var_name: ds_cropped[var_name].compute()}

    data_arrays: dict[str, xr.DataArray] = {}
    # The download is I/O bound (S3 network requests). We use a
    # ThreadPoolExecutor to parallelize network latency across multiple
    # variables. A ProcessPoolExecutor would be less efficient here due to the
    # high serialization overhead of Xarray objects between processes.
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
    h3_grid: pt.DataFrame[H3GridWeights],
) -> pt.DataFrame[Nwp]:
    """Vectorized processing of ECMWF dataset to H3 grid."""
    # Convert the entire Xarray Dataset to a Polars DataFrame directly.
    # We use to_dataframe() which returns a pandas DataFrame, then convert to Polars.
    # While this uses pandas, it is the most direct way to convert Xarray to Polars
    # without manual iteration.
    nwp_df = pl.from_pandas(loaded_ds.to_dataframe().reset_index())

    # Check for missing spatial data.
    # NWPs should never have "random" missing data. If data is missing then there's a bug
    # and I want the code to fail loudly!
    if (
        nwp_df.drop(["init_time", "lead_time", "ensemble_member", "latitude", "longitude"])
        .null_count()
        .sum()
        .item(0, 0)
        > 0
    ):
        raise MalformedZarrError(
            "Missing spatial data detected in NWP dataset. This indicates a bug."
        )

    # Define variables for aggregation
    all_nwp_vars = [str(v) for v in loaded_ds.data_vars.keys()]
    categorical_vars = ["categorical_precipitation_type_surface"]
    numeric_vars = [v for v in all_nwp_vars if v not in categorical_vars]
    # The join is on latitude and longitude.
    # We cast coordinates to Float32 before the join to ensure exact bit-level matching.
    # Polars performs exact equality checks for joins, and floating-point precision
    # differences between Float32 (often used in Zarr) and Float64 (used in the H3 grid)
    # can cause the join to fail silently.
    nwp_df = nwp_df.with_columns(
        [pl.col("longitude").cast(pl.Float32), pl.col("latitude").cast(pl.Float32)]
    )
    h3_grid = h3_grid.with_columns(
        [pl.col("nwp_lng").cast(pl.Float32), pl.col("nwp_lat").cast(pl.Float32)]
    )

    joined = h3_grid.join(
        nwp_df, left_on=["nwp_lng", "nwp_lat"], right_on=["longitude", "latitude"], how="left"
    ).with_columns(
        # Explicitly cast h3_index to UInt64 after the join to prevent silent type
        # coercion (e.g., to Float64 or Int64) during Polars joins, which is
        # critical for downstream H3 operations and memory efficiency.
        pl.col("h3_index").cast(pl.UInt64)
    )

    # Aggregate to H3 resolution 5.
    # We group by h3_index, lead_time, and ensemble_member.
    # To avoid biasing the aggregated weather variables towards zero when an H3 cell
    # partially overlaps with missing NWP data (e.g., at the edges of the domain),
    # we compute a true weighted average by dividing the weighted sum by the sum
    # of the proportions of the valid (non-NaN) cells.
    group_cols = ["h3_index", "lead_time", "ensemble_member"]

    def weight_sum_expr(x: str) -> pl.Expr:
        return (
            pl.when(pl.col(x).fill_nan(None).is_not_null())
            .then(pl.col("proportion"))
            .otherwise(0.0)
            .sum()
        )

    # Aggregate numeric variables
    agg_exprs = [
        pl.when(weight_sum_expr(x) > 0)
        .then((pl.col(x).fill_nan(None) * pl.col("proportion")).sum() / weight_sum_expr(x))
        .otherwise(None)
        .cast(pl.Float32)
        .alias(x)
        for x in numeric_vars
    ]
    processed = joined.group_by(group_cols).agg(agg_exprs)

    # WEIGHTED CATEGORICAL AGGREGATION:
    # We use a weighted mode calculation that respects the H3 cell overlap proportions.
    # This ensures categorical weather features (like precipitation type) accurately
    # reflect the area-weighted majority of the H3 cell, rather than treating a 1%
    # overlap the same as a 99% overlap.
    for c in categorical_vars:
        df_cat = (
            joined.group_by(group_cols + [c])
            .agg(pl.col("proportion").sum().alias("weight"))
            .sort(group_cols + ["weight"])
            .group_by(group_cols)
            .agg(pl.col(c).last().cast(pl.UInt8))
        )
        processed = processed.join(df_cat, on=group_cols, how="left")

    # Compute valid_time and init_time.
    # lead_time is a timedelta64[ns] in the DataFrame.
    processed = processed.with_columns(
        init_time=pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC")),
        valid_time=(
            pl.lit(nwp_init_time).cast(pl.Datetime("us", "UTC"))
            + pl.col("lead_time").cast(pl.Duration("us"))
        ).cast(pl.Datetime("us", "UTC")),
        ensemble_member=pl.col("ensemble_member").cast(pl.UInt8),
    ).drop("lead_time")

    # Sort before validation to ensure consistent output order
    processed = processed.sort(by=["init_time", "valid_time", "ensemble_member", "h3_index"])

    # Validate to ensure the interpolated data matches the expected schema
    # (The Nwp schema expects Float32 for weather variables)
    return Nwp.validate(processed, drop_superfluous_columns=True)
