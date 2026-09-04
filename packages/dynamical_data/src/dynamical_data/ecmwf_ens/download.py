"""Opening and downloading one ECMWF ENS run from the Dynamical.org catalog."""

import concurrent.futures
from datetime import UTC, datetime
from typing import Final, Literal

import dynamical_catalog
import numpy as np
import patito as pt
import polars as pl
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import Nwp


class NwpRunNotYetAvailable(Exception):
    """Raised when ``nwp_init_time`` is not yet in the catalog (Dynamical has not published it)."""


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

ECMWF_ENS_INSTANTANEOUS_VARS: Final[frozenset[str]] = (
    frozenset(_ECMWF_ENS_VARS_TO_DOWNLOAD) - Nwp.deaccumulated_var_names - Nwp.categorical_var_names
)
"""The downloaded variables describing conditions at one instant, under their *download* names.

None of these is ever legitimately null, anywhere in a run, which is why
:func:`dynamical_data.ecmwf_ens.upstream_nulls.assess_upstream_grid_point_nulls` counts them
separately from the de-accumulated ones and the ``ecmwf_ens`` asset gates a check on that count
being zero.

Derived from the download list rather than from ``Nwp``'s fields, because the two namespaces differ:
we download ``wind_u_10m``/``wind_v_10m`` (and the 100 m pair), and
:func:`dynamical_data.ecmwf_ens.convert_to_polars.convert_nwp_xarray_dataset_to_polars_dataframe`
derives ``wind_speed_*``/``wind_direction_*`` from them. A set taken from the contract would name
four variables the downloaded dataset does not carry, and indexing it would raise ``KeyError``.
"""


def open_ecmwf_ens_run(
    nwp_init_time: datetime,
    h3_grid: pt.DataFrame[H3GridWeights],
) -> xr.Dataset:
    """Lazily open the ECMWF ENS Icechunk store and slice it to the requested run and H3 grid.

    No data is downloaded: the returned dataset is still backed by lazy Dask/Zarr arrays.
    Call :func:`download_ecmwf_ens_data` to actually fetch the data.

    Args:
        nwp_init_time: The initialisation time to open. Must be timezone aware.
        h3_grid: The H3 grid to use for spatial bounds.

    Returns:
        The catalog's dataset, sliced to the one `init_time` and to the latitude/longitude
        bounding box of `h3_grid`'s `nwp_lat`/`nwp_lon` columns, still lazy and holding the
        13 downloaded ECMWF ENS variables.
    """
    # Convention-sensitive to the *real* Dynamical.org catalog: this function bakes in assumptions
    # about its shape (longitude in [-180, 180], descending latitude, coordinate/dimension names).
    # The offline tests share those assumptions and cannot catch a mismatch with the live
    # catalog, so after changing this function run the network-gated test manually:
    #     uv run pytest --run-network -m network
    # See
    # <https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#network-gated-tests>.

    # Reusable-package input validation, not a reachable production state: the `ecmwf_ens` asset
    # always sources `h3_grid` from `h3_grid_weights`, which raises on an empty cell list before
    # writing anything, so the file it reads can never hold zero rows.
    if h3_grid.is_empty():
        raise ValueError("h3_grid is empty. Cannot download ECMWF data for an empty grid.")

    if nwp_init_time.utcoffset() is None:
        raise ValueError(f"nwp_init_time must be timezone aware. {nwp_init_time.tzinfo=}")

    # The xarray selection needs nwp_init_time timezone-naive, so it is converted to UTC and then
    # stripped of tzinfo here.
    utc_nwp_init_time = np.datetime64(nwp_init_time.astimezone(UTC).replace(tzinfo=None))

    ds = dynamical_catalog.open("ecmwf-ifs-ens-forecast-15-day-0-25-degree", chunks=None)

    ds = ds[list(_ECMWF_ENS_VARS_TO_DOWNLOAD)]

    if utc_nwp_init_time not in ds.init_time.values:
        raise NwpRunNotYetAvailable(f"{utc_nwp_init_time} is not in ds.init_time.values")

    # This check guards the Dynamical.org catalog itself, an external substrate we neither control
    # nor version-pin, so its shape can change under us between runs.
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

    # NOTE: The slice below fails if the requested region crosses the anti-meridian. The GB
    # service area never does, so that case is not handled.
    ds_sliced = ds.sel(latitude=lat_slice, longitude=lon_slice, init_time=utc_nwp_init_time)

    # An empty spatial intersection here would otherwise surface much later as a confusing
    # KeyError during DataFrame conversion, so it is checked and named explicitly now.
    if ds_sliced.longitude.size == 0 or ds_sliced.latitude.size == 0:
        raise ValueError("No spatial overlap found between H3 grid and NWP dataset.")

    return ds_sliced


def download_ecmwf_ens_data(ds_sliced: xr.Dataset) -> xr.Dataset:
    """Download (compute) a lazily-opened, already-sliced ECMWF ENS dataset.

    Args:
        ds_sliced: A lazy dataset as returned by :func:`open_ecmwf_ens_run`.

    Returns:
        The same variables and coordinates as `ds_sliced`, each variable now backed by an
        in-memory `xr.DataArray` rather than a lazy Dask/Zarr array, fetched with up to 4
        variables downloaded concurrently.
    """

    def download_array(var_name: str) -> dict[str, xr.DataArray]:
        return {var_name: ds_sliced[var_name].compute()}

    # The download is I/O bound (S3 network requests). We use a ThreadPoolExecutor to parallelise
    # network latency across multiple variables. A ProcessPoolExecutor would be less efficient here
    # due to the high serialisation overhead of Xarray objects between processes.
    #
    # max_workers is capped rather than left at the default (one thread per variable, i.e. 13).
    # Investigation of issue #276 found that 13 concurrent chunked-zarr fetches self-contend badly
    # (S3 rate limiting or connection-pool starvation): most variables finish in 5-20s, but a few
    # straggle for minutes, making the whole download 600s+. Capping at 4 removed the stragglers
    # entirely and cut a real download from 645s to 22.5s.
    #
    # 4 is the cap *per partition*, not per machine: `ecmwf_ens` runs up to its `ECMWF` pool limit
    # of partitions at once (4, set in `dagster.yaml`), so the fetches in flight against
    # Dynamical.org are the product of the two. Lower this cap before raising that limit.
    #
    # The slowdown is a recent regression, not a pre-existing property of the download: Dagster's
    # run history shows per-partition downloads holding a steady ~48-54s right up to 2026-06-30
    # 12:26 UTC, then every run afterwards (2026-07-01 onwards) taking 3-12 min. That boundary
    # lines up exactly with an `icechunk` 2.0.6 -> 2.1.0 bump in the same `uv.lock` update
    # (commit b46d145, 2026-06-30 12:26:50 UTC) — the leading theory is a change in icechunk's
    # underlying S3 client (connection pooling/concurrency handling) between those two versions,
    # though that has not been confirmed by pinning back to 2.0.6 and re-testing.
    data_arrays: dict[str, xr.DataArray] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(download_array, str(name)) for name in ds_sliced.data_vars]
        for future in concurrent.futures.as_completed(futures):
            data_arrays.update(future.result())

    return xr.Dataset(data_arrays)


def _calc_slice_for_lat_or_lng(
    coord_name: Literal["latitude", "longitude"],
    ds: xr.Dataset,
    min_coord: float,
    max_coord: float,
) -> slice:
    """Build a `slice` in whichever direction the coordinate is stored, ascending or descending.

    `xarray`'s `.sel(dim=slice(a, b))` is sensitive to coordinate direction: passing `(min, max)`
    against a descending coordinate returns an empty selection rather than raising, so the
    direction has to be checked rather than assumed. In the ECMWF ENS catalog on Dynamical.org,
    latitude runs from +90 to -90 (descending) and longitude runs from -180 to 179.75 (ascending,
    at the catalog's native 0.25-degree grid spacing).

    Args:
        coord_name: Which coordinate this slice is for — used only to name the two error cases
            below.
        ds: The dataset `coord_name` is read from.
        min_coord: The lower bound of the region to slice to.
        max_coord: The upper bound of the region to slice to.

    Returns:
        `slice(min_coord, max_coord)` if the coordinate is ascending, or `slice(max_coord,
        min_coord)` if it is descending — whichever order `xarray` needs for that direction.
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
