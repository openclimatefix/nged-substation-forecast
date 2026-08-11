"""Converting a downloaded ECMWF ENS xarray dataset into the `Nwp` Polars contract.

The regular lat/lon grid is aggregated onto H3 cells on the way.
"""

from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import patito as pt
import polars as pl
import polars.selectors as cs
import xarray as xr
from contracts.common import UTC_DATETIME_DTYPE
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import NWP_MODEL_ID_DTYPE, Nwp, NwpModelId

_CONTRIBUTING_WEIGHT_SUFFIX: Final[str] = "__contributing_weight"
"""Suffix of the per-variable renormalisation denominator built by the H3 aggregation.

These columns are intermediate: they are dropped before the aggregated frame is returned.
"""

_CATEGORY_WEIGHT_SUFFIX: Final[str] = "__category_weight"
"""Suffix of the per-category area weight the H3 aggregation ranks a categorical variable by.

These columns are intermediate: they exist only on the pre-`group_by` frame, so they never reach
the aggregated output.
"""


def convert_nwp_xarray_dataset_to_polars_dataframe(
    ds: xr.Dataset,
    h3_grid: pt.DataFrame[H3GridWeights],
) -> pt.DataFrame[Nwp]:
    """Vectorized processing of ECMWF dataset to H3 grid."""
    # Convention-sensitive to real ECMWF ENS data: the dim/coord order feeds the ravel + value-join,
    # and the physical units feed Nwp.validate. The offline tests share those assumptions, so after
    # changing this function run the network-gated test manually:
    #     uv run pytest --run-network -m network
    # See
    # <https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#network-gated-tests>.
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
    numeric_vars = [v for v in all_nwp_vars if v not in Nwp.categorical_var_names]

    # Add data variables
    for var_name in all_nwp_vars:
        data_dict[var_name] = ds[var_name].values.ravel()

    # Normalise NaN to null here, at the boundary, for *every* weather variable. This is what gives
    # "missing" a single representation downstream: xarray delivers upstream corruption as NaN,
    # while a grid point the H3 weights name but the dataset does not carry arrives as a null from
    # the left join below. The aggregation must treat the two identically, and it can only do that
    # if they look the same.
    nwp_df = pl.DataFrame(data_dict).with_columns(
        pl.col(numeric_vars).fill_nan(None),
        pl.col(categorical_vars).fill_nan(None).cast(pl.UInt8),
    )

    joined = h3_grid.join(
        nwp_df, left_on=["nwp_lon", "nwp_lat"], right_on=["longitude", "latitude"], how="left"
    )

    return _aggregate_grid_points_to_h3_cells(joined, numeric_vars, categorical_vars)


def _aggregate_grid_points_to_h3_cells(
    joined: pl.DataFrame,
    numeric_vars: Sequence[str],
    categorical_vars: Sequence[str],
) -> pl.DataFrame:
    """Reduce one H3 cell's overlapping NWP grid points to a single row per cell.

    Guarantees, per cell:

    - A **numeric** variable is the area-weighted mean of the points that supplied a value:
      `sum(v * p)`, where the `proportion` weights `p` sum to 1 over the whole cell by construction
      (see `geo.h3.compute_h3_grid_weights`), divided by the *contributing* weight rather than by
      that 1.0, so a missing point costs only its own share instead of biasing the cell low.
    - A **categorical** variable is the category covering most of the cell's area, with an exact
      tie resolved to the lowest category code. Points that supplied no category are excluded from
      the ranking rather than competing in it.
    - Either kind yields **null**, never `0.0` or a spurious category, when *no* point contributed.

    Each variable is renormalised over its *own* denominator, which is what keeps one variable's
    corruption from nulling the others. The cost a caller must know about: two variables in one
    cell can then be averaged over different sub-areas of the hexagon, so if `wind_u_*` and
    `wind_v_*` ever have different null footprints, the vector `_calc_wind_speed` and
    `_calc_wind_direction` derive from them mixes two sub-areas. Upstream corruption has always
    been co-located across variables, so that is theoretical today.

    Why it is done this way, with the measurements and the tie-break's dry bias:
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/ecmwf-ens-known-issues/#spatial-aggregation-is-where-a-grid-points-null-is-resolved>.

    Args:
        joined: The H3 grid weights left-joined onto one chunk's NWP grid values. NaN must already
            have been normalised to null, since the contributing weight is computed from nullness.
        numeric_vars: Weather variables aggregated as a renormalised weighted mean.
        categorical_vars: Weather variables aggregated as an area-weighted mode.
    """
    contributing_weights = [
        pl.col("proportion")
        .filter(pl.col(var).is_not_null())
        .sum()
        .alias(var + _CONTRIBUTING_WEIGHT_SUFFIX)
        for var in numeric_vars
    ]
    # Total area each category covers within the cell. Null where the category itself is null, so
    # that `nulls_last` below drops missing points out of the running rather than letting them win.
    category_weights = [
        pl.when(pl.col(var).is_not_null())
        .then(pl.col("proportion").sum().over(["h3_index", var]))
        .otherwise(None)
        .alias(var + _CATEGORY_WEIGHT_SUFFIX)
        for var in categorical_vars
    ]
    # The category covering the most of the cell's area wins. An exact tie goes to the lowest
    # category code, which leans dry: code 0 is "no precipitation", so a cell split exactly between
    # "no precipitation" and "snow" is recorded as "no precipitation".
    weighted_modes = [
        pl.col(var)
        .sort_by(
            pl.col(var + _CATEGORY_WEIGHT_SUFFIX),
            pl.col(var),
            descending=[True, False],
            nulls_last=True,
        )
        .first()
        for var in categorical_vars
    ]
    return (
        joined.with_columns(pl.col(numeric_vars) * pl.col("proportion"), *category_weights)
        .group_by("h3_index")
        .agg(
            pl.col(numeric_vars).sum(),
            *contributing_weights,
            *weighted_modes,
        )
        # The `> 0` guard is load-bearing, not defensive: Polars sums an all-null group to `0.0`,
        # so without it a cell that lost every point would divide 0.0 by 0.0.
        .with_columns(
            **{
                var: pl.when(pl.col(var + _CONTRIBUTING_WEIGHT_SUFFIX) > 0)
                .then(pl.col(var) / pl.col(var + _CONTRIBUTING_WEIGHT_SUFFIX))
                .otherwise(None)
                for var in numeric_vars
            }
        )
        .drop(cs.ends_with(_CONTRIBUTING_WEIGHT_SUFFIX))
    )
