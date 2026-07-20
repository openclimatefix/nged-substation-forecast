"""Shared fixtures for the ``dynamical_data`` tests.

These build in-memory ``xarray`` datasets that mimic the ECMWF ENS structure the code consumes,
plus small :class:`H3GridWeights` frames, so every test runs fully offline (no Dynamical.org
network access).
"""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Final

import numpy as np
import patito as pt
import pytest
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from dynamical_data.ecmwf_ens.download import _ECMWF_ENS_VARS_TO_DOWNLOAD

# Physically-plausible constant defaults for each downloaded variable, chosen so the final
# ``Nwp.validate()`` (min/max bounds per column) passes. Individual tests override specific
# variables via ``var_values``.
_VAR_DEFAULTS: Final[dict[str, float]] = {
    "temperature_2m": 15.0,
    "dew_point_temperature_2m": 10.0,
    "wind_u_10m": 3.0,
    "wind_v_10m": 4.0,
    "wind_u_100m": 6.0,
    "wind_v_100m": 8.0,
    "pressure_surface": 101_000.0,
    "pressure_reduced_to_mean_sea_level": 101_500.0,
    "geopotential_height_500hpa": 5_500.0,
    "downward_long_wave_radiation_flux_surface": 300.0,
    "downward_short_wave_radiation_flux_surface": 200.0,
    "precipitation_surface": 0.0001,
    "categorical_precipitation_type_surface": 0.0,
}

# Guard against drift between this fixture and the real download list.
assert set(_VAR_DEFAULTS) == set(_ECMWF_ENS_VARS_TO_DOWNLOAD), (
    "conftest _VAR_DEFAULTS is out of sync with _ECMWF_ENS_VARS_TO_DOWNLOAD"
)

# After 2024-11-12, so a populated categorical_precipitation_type_surface passes Nwp validation
# (that column must be all-null on/before 2024-11-12 and never-null afterwards).
_DEFAULT_INIT_TIME: Final[datetime] = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _build_ens_dataset(
    *,
    init_time: datetime = _DEFAULT_INIT_TIME,
    latitudes: Sequence[float] = (52.0, 51.75),
    longitudes: Sequence[float] = (-1.0, -0.75),
    lead_time_hours: Sequence[int] = (0, 6, 12),
    ensemble_members: Sequence[int] = (0, 1),
    var_values: dict[str, np.ndarray | float] | None = None,
    init_time_as_dim: bool = False,
) -> xr.Dataset:
    """Build a synthetic ECMWF ENS dataset matching the structure the pipeline consumes.

    The dataset always carries all 13 downloaded variables, a ``valid_time`` coordinate equal to
    ``init_time + lead_time``, and (in real Dynamical orientation) descending latitudes.

    Args:
        init_time: The run initialisation time (must be timezone aware); stored tz-naive in UTC.
        latitudes: Latitude coordinate values (descending, as in real ECMWF ENS on Dynamical).
        longitudes: Longitude coordinate values, in the [-180, 180] range.
        lead_time_hours: Forecast lead times in hours (become ``timedelta64`` coordinates).
        ensemble_members: Ensemble member indices.
        var_values: Optional per-variable overrides, each broadcastable to
            ``(lead_time, ensemble_member, latitude, longitude)``. Missing variables use their
            constant default.
        init_time_as_dim: If True, keep ``init_time`` as a size-1 dimension (the shape
            ``open_ecmwf_ens_run`` expects from the catalog). If False, reduce it to a scalar
            coordinate (the post-``open`` shape ``convert`` consumes).
    """
    lats = np.asarray(latitudes, dtype=np.float32)
    lons = np.asarray(longitudes, dtype=np.float32)
    lead = np.asarray(lead_time_hours, dtype="timedelta64[h]").astype("timedelta64[ns]")
    members = np.asarray(ensemble_members, dtype=np.int64)
    init = np.datetime64(init_time.astimezone(timezone.utc).replace(tzinfo=None), "ns")
    valid = init + lead  # datetime64[ns], one per lead time

    shape = (len(lead), len(members), len(lats), len(lons))
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name, default in _VAR_DEFAULTS.items():
        if var_values is not None and name in var_values:
            base = np.broadcast_to(np.asarray(var_values[name], dtype=np.float32), shape)
            base = base.astype(np.float32)
        else:
            base = np.full(shape, default, dtype=np.float32)
        # Always build with a leading init_time axis; optionally collapse it below.
        dims = ("init_time", "lead_time", "ensemble_member", "latitude", "longitude")
        data_vars[name] = (dims, base[np.newaxis, ...])

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "init_time": ("init_time", np.asarray([init])),
            "lead_time": ("lead_time", lead),
            "ensemble_member": ("ensemble_member", members),
            "latitude": ("latitude", lats),
            "longitude": ("longitude", lons),
            "valid_time": (("init_time", "lead_time"), valid[np.newaxis, :]),
        },
    )

    if not init_time_as_dim:
        # Reduce init_time to a scalar coordinate exactly as ds.sel(init_time=...) does in
        # open_ecmwf_ens_run, so valid_time collapses to dims (lead_time,).
        ds = ds.isel(init_time=0)

    return ds


def _build_h3_grid(
    *,
    h3_index: Sequence[int],
    nwp_lat: Sequence[float],
    nwp_lon: Sequence[float],
    proportion: Sequence[float],
) -> pt.DataFrame[H3GridWeights]:
    """Build a valid :class:`H3GridWeights` frame from column values."""
    return (
        pt.DataFrame(
            {
                "h3_index": list(h3_index),
                "nwp_lat": list(nwp_lat),
                "nwp_lon": list(nwp_lon),
                "proportion": list(proportion),
            }
        )
        .set_model(H3GridWeights)
        .cast()
    )


@pytest.fixture
def make_ens_dataset() -> Callable[..., xr.Dataset]:
    """Return the synthetic-ECMWF-ENS-dataset factory."""
    return _build_ens_dataset


@pytest.fixture
def make_h3_grid() -> Callable[..., pt.DataFrame[H3GridWeights]]:
    """Return the H3GridWeights factory."""
    return _build_h3_grid


@pytest.fixture
def default_h3_grid() -> pt.DataFrame[H3GridWeights]:
    """A two-cell grid whose lat/lon exactly match the default dataset's grid points."""
    return _build_h3_grid(
        h3_index=[10, 20],
        nwp_lat=[52.0, 51.75],
        nwp_lon=[-1.0, -0.75],
        proportion=[1.0, 1.0],
    )
