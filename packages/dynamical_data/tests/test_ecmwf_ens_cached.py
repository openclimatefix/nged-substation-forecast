"""Hermetic end-to-end test over a *cached real* ECMWF ENS slice (no network).

This is the offline twin of ``test_ecmwf_ens_network.py``. It reads a tiny real slice committed at
``data/ecmwf_ens_real_slice.nc`` (captured once by ``data/capture_ecmwf_ens_slice.py``) and runs it
through ``convert``. Because the bytes are genuine Dynamical.org output, the slice carries the real
conventions — descending latitude, longitude in [-180, 180], dimension order, dtypes, physical
units — that the synthetic ``conftest.py`` fixtures merely assume. Running every PR, it is a
regression guard on those conventions and on the value↔(lat, lon) mapping, without the flakiness or
credentials a live catalog call needs. The ``network`` test remains the guard against *future*
upstream drift.
"""

from pathlib import Path

import patito as pt
import pytest
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import Nwp
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe as convert,
)

_SLICE = Path(__file__).parent / "data" / "ecmwf_ens_real_slice.nc"


def _one_cell_per_grid_point(
    ds: xr.Dataset,
) -> tuple[pt.DataFrame[H3GridWeights], dict[int, float]]:
    """Give every grid point its own single-point H3 cell, and the temperature it should carry.

    Mirrors the synthetic orientation test, but on real coordinates and values: the expected
    ``{h3_index: temperature_2m}`` mapping is read straight from the dataset with xarray's own
    ``.sel``, independently of ``convert``, so a lat/lon swap or flip inside ``convert`` shows up as
    a mismatch.
    """
    h3_index: list[int] = []
    nwp_lat: list[float] = []
    nwp_lon: list[float] = []
    expected: dict[int, float] = {}
    next_id = 0
    for lat in ds.latitude.values:
        for lon in ds.longitude.values:
            next_id += 1
            h3_index.append(next_id)
            nwp_lat.append(float(lat))
            nwp_lon.append(float(lon))
            expected[next_id] = float(ds["temperature_2m"].sel(latitude=lat, longitude=lon).item())
    grid = (
        pt.DataFrame(
            {
                "h3_index": h3_index,
                "nwp_lat": nwp_lat,
                "nwp_lon": nwp_lon,
                "proportion": [1.0] * len(h3_index),
            }
        )
        .set_model(H3GridWeights)
        .cast()
    )
    return grid, expected


def test_cached_real_slice_conventions_and_orientation() -> None:
    ds = xr.open_dataset(_SLICE)

    # --- Conventions the offline synthetic fixtures assume, checked on real bytes ---
    lats = ds.latitude.values
    lons = ds.longitude.values
    assert lats[0] > lats[-1], "latitude must be descending, as the offline fixture assumes"
    assert lons.min() >= -180.0 and lons.max() <= 180.0, "longitude must be in [-180, 180]"

    grid, expected = _one_cell_per_grid_point(ds)
    df = convert(ds=ds, h3_grid=grid)

    Nwp.validate(df)  # dtype/bounds/unit contract on genuine data
    assert df.height == len(expected)  # single lead time, single member -> one row per cell

    # Each grid point's real temperature reached the H3 cell at its own (lat, lon): no swap/flip.
    produced = dict(zip(df["h3_index"].to_list(), df["temperature_2m"].to_list(), strict=True))
    assert produced == pytest.approx(expected)

    # Physical sanity: GB near-surface air temperature is in this band; Kelvin would blow it.
    temperatures = df["temperature_2m"].to_list()
    assert min(temperatures) >= -30.0
    assert max(temperatures) <= 45.0
