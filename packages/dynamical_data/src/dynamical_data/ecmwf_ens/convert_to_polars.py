from typing import Literal

import numpy as np
import patito as pt
import polars as pl
import polars.selectors as cs
import xarray as xr
from contracts.common import UTC_DATETIME_DTYPE
from contracts.geo_schemas import H3GridWeights
from contracts.settings import Settings
from contracts.weather_schemas import NWP_MODEL_ID_DTYPE, Nwp, NwpModelId

_SETTINGS = Settings()


def convert_nwp_xarray_dataset_to_polars_dataframe(
    ds: xr.Dataset,
    h3_grid: pt.DataFrame[H3GridWeights],
) -> pt.DataFrame[Nwp]:
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
            df = _process_chunk_for_1_lead_time_and_1_ens_member(
                ds_chunk,
                h3_grid,
                lat_grid=lat_grid_raveled,
                lon_grid=lon_grid_raveled,
            ).with_columns(
                ensemble_member=pl.lit(ensemble_member).cast(pl.UInt8),
                valid_time=pl.lit(ds_chunk["valid_time"].values).cast(UTC_DATETIME_DTYPE),
            )
            dfs.append(df)

    df = (
        pl.concat(dfs)
        .with_columns(
            nwp_model_id=pl.lit(NwpModelId.ECMWF_ENS_0_25_degree.name).cast(NWP_MODEL_ID_DTYPE),
            init_time=pl.lit(ds["init_time"].values).cast(UTC_DATETIME_DTYPE),
            wind_speed_10m=_calc_wind_speed(height="10m"),
            wind_speed_100m=_calc_wind_speed(height="100m"),
            wind_direction_10m=_calc_wind_direction(height="10m"),
            wind_direction_100m=_calc_wind_direction(height="100m"),
        )
        .drop(cs.matches("^wind_u_.*") | cs.matches("^wind_v_.*"))
    )

    # No sort here: physical row order is delta_store.nwp's job (the single source of truth for
    # on-disk layout), and validation is order-independent.
    return Nwp.validate(df)


def _calc_wind_speed(height: Literal["10m", "100m"]) -> pl.Expr:
    return (pl.col(f"wind_u_{height}") ** 2 + pl.col(f"wind_v_{height}") ** 2).sqrt()


def _calc_wind_direction(height: Literal["10m", "100m"]) -> pl.Expr:
    """Compute Wind Direction (Meteorological Convention)."""
    RAD_TO_DEG = 180 / np.pi
    # The arctan2 order: In standard math, it's often atan2(y, x). In Polars, pl.arctan2("y", "x")
    # follows this convention. By passing u as y and v as x, we align the 0° angle with the North
    # (v) axis. The +180 offset: Since u and v describe where the wind is going, arctan2 gives you
    # the direction of travel. Adding 180° flips the vector to show where the wind is coming *from*.
    return (pl.arctan2(f"wind_u_{height}", f"wind_v_{height}") * RAD_TO_DEG + 180) % 360


def _process_chunk_for_1_lead_time_and_1_ens_member(
    ds: xr.Dataset,
    h3_grid: pt.DataFrame[H3GridWeights],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> pl.DataFrame:
    """Processes a single chunk of the ECMWF dataset."""
    # Prepare data dictionary
    data_dict: dict[str, np.ndarray] = {"latitude": lat_grid, "longitude": lon_grid}

    all_nwp_vars = [str(v) for v in ds.data_vars]
    categorical_vars = list(Nwp.categorical_var_names)

    # Add data variables
    for var_name in all_nwp_vars:
        data_dict[var_name] = ds[var_name].values.ravel()

    # Create Polars DataFrame.
    nwp_df = pl.DataFrame(data_dict).with_columns(
        pl.col(categorical_vars).fill_nan(None).cast(pl.UInt8)
    )

    joined = h3_grid.join(
        nwp_df, left_on=["nwp_lon", "nwp_lat"], right_on=["longitude", "latitude"], how="left"
    )

    # Aggregate NWP variables to H3 index.
    numeric_vars = [v for v in all_nwp_vars if v not in Nwp.categorical_var_names]
    return (
        joined.with_columns(pl.col(numeric_vars) * pl.col("proportion"))
        .group_by("h3_index")
        .agg(pl.col(numeric_vars).sum(), pl.col(categorical_vars).mode().first(ignore_nulls=True))
        # The ordering matters! It's essential to to `fill_nan(None)` *after* the aggregation,
        # otherwise the None values get silently filled with zeros!
        .with_columns(pl.col(numeric_vars).fill_nan(None))
    )
