"""Tests for ``dynamical_data.ecmwf_ens.convert_to_polars``.

These run entirely on in-memory synthetic datasets (see ``conftest.py``); the final integration
test drives the whole open -> download -> convert pipeline with the network call patched out.
"""

from collections.abc import Callable
from datetime import datetime, timezone

import numpy as np
import patito as pt
import polars as pl
import pytest
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import Nwp
from dynamical_data.ecmwf_ens import download
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe as convert,
)

# After 2024-11-12 so the populated categorical_precipitation_type_surface column is valid; see
# Nwp._check_variables_that_were_introduced_after_start_of_dataset.
_INIT_TIME = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _single_cell_grid(
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> pt.DataFrame[H3GridWeights]:
    """One H3 cell fed by the single grid point (52.0, -1.0) with full weight."""
    return make_h3_grid(h3_index=[10], nwp_lat=[52.0], nwp_lon=[-1.0], proportion=[1.0])


def test_convert_happy_path_shape_and_schema(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    ds = make_ens_dataset(
        init_time=_INIT_TIME,
        latitudes=(52.0, 51.75),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0, 6, 12),
        ensemble_members=(0, 1),
    )
    # Four grid points map onto two H3 cells (two points each), weights summing to 1 per cell.
    h3 = make_h3_grid(
        h3_index=[10, 10, 20, 20],
        nwp_lat=[52.0, 52.0, 51.75, 51.75],
        nwp_lon=[-1.0, -0.75, -1.0, -0.75],
        proportion=[0.5, 0.5, 0.5, 0.5],
    )

    df = convert(ds=ds, h3_grid=h3)

    Nwp.validate(df)  # convert already validates; assert it explicitly too.
    # One row per (h3_index, valid_time, ensemble_member): 2 cells x 3 lead times x 2 members.
    assert df.height == 2 * 3 * 2
    assert df["nwp_model_id"].unique().to_list() == ["ECMWF_ENS_0_25_degree"]
    # Each member's rows carry its own id (pins the per-member value routing, not just the count).
    assert sorted(df["ensemble_member"].unique().to_list()) == [0, 1]
    # Wind speed/direction are derived; the raw u/v components are dropped.
    for col in ("wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m"):
        assert col in df.columns
    for col in ("wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"):
        assert col not in df.columns
    assert df["init_time"].unique().to_list() == [_INIT_TIME]
    assert set(df["valid_time"].unique().to_list()) == {
        datetime(2025, 1, 1, 0, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 6, tzinfo=timezone.utc),
        datetime(2025, 1, 1, 12, tzinfo=timezone.utc),
    }


def test_convert_derives_wind_speed(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0,),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "wind_u_10m": 3.0,
            "wind_v_10m": 4.0,
            "wind_u_100m": 6.0,
            "wind_v_100m": 8.0,
        },
    )
    df = convert(ds=ds, h3_grid=_single_cell_grid(make_h3_grid))

    assert df.height == 1
    assert df["wind_speed_10m"].item() == pytest.approx(5.0)  # sqrt(3^2 + 4^2)
    assert df["wind_speed_100m"].item() == pytest.approx(10.0)  # sqrt(6^2 + 8^2)


@pytest.mark.parametrize(
    "wind_u, wind_v, expected_direction",
    # Meteorological "direction from"; formula is (arctan2(u, v) * 180/pi + 180) % 360. All four
    # quadrants plus a non-axis-aligned bearing; values kept clear of the 0/360 wrap boundary so
    # float rounding can't flip 359.99 <-> 0.
    [
        (0.0, 1.0, 180.0),  # from the south
        (1.0, 0.0, 270.0),  # from the west
        (-1.0, 0.0, 90.0),  # from the east
        (1.0, 1.0, 225.0),  # from the south-west
        (-1.0, -1.0, 45.0),  # from the north-east
    ],
)
def test_convert_derives_wind_direction(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
    wind_u: float,
    wind_v: float,
    expected_direction: float,
) -> None:
    # Inject identical components at both heights so the 100 m derivation is checked too, not just
    # its column presence.
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0,),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "wind_u_10m": wind_u,
            "wind_v_10m": wind_v,
            "wind_u_100m": wind_u,
            "wind_v_100m": wind_v,
        },
    )
    df = convert(ds=ds, h3_grid=_single_cell_grid(make_h3_grid))

    assert df["wind_direction_10m"].item() == pytest.approx(expected_direction)
    assert df["wind_direction_100m"].item() == pytest.approx(expected_direction)


def test_convert_proportion_weighted_aggregation(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    # One latitude, two longitudes: temperature is 10 at lon -1.0 and 20 at lon -0.75.
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={"temperature_2m": np.array([[10.0, 20.0]], dtype=np.float32)},
    )
    # Both points feed one cell with weights 0.75 / 0.25.
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.75, 0.25],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["temperature_2m"].item() == pytest.approx(0.75 * 10.0 + 0.25 * 20.0)


def test_convert_maps_each_grid_point_to_its_own_lat_lon(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """Each grid point's value reaches the H3 cell at its own (lat, lon): no flip or transpose.

    This is the one test that guards the classic NWP orientation bugs: flipping the grid
    vertically (latitude reversed), flipping it horizontally (longitude reversed), transposing
    latitude and longitude, or rotating it. It uses a 2x2 grid carrying a distinct temperature at
    every corner (north-west 10, north-east 20, south-west 30, south-east 40) over two distinct
    latitudes and two distinct longitudes, and isolates each corner into its own single-point H3
    cell. Because all four values and both axis coordinates are distinct, any permutation of the
    values against the coordinates — vertical flip, horizontal flip, transpose, 180-degree
    rotation — yields a different {h3_index: temperature} mapping, so the single equality
    assertion catches every one of them. Verified by mutation: reversing the latitude coordinate,
    reversing the longitude coordinate, and switching np.meshgrid from indexing="ij" to "xy" each
    make this test fail. Every other value test in this module uses a single latitude, so this is
    the only one that can see an orientation error.

    Scope — what this proves and what it delegates. ``convert`` assigns grid cells to H3 hexagons
    by a value-join on (lat, lon), never by position, so it structurally cannot place a grid cell
    into the wrong hexagon on its own: the geographic truth of which hexagon owns a given (lat,
    lon) lives entirely in the ``H3GridWeights`` table built upstream by the ``h3_grid_weights``
    asset. The only orientation bug ``convert`` itself can introduce is a ravel misalignment —
    pairing a data value with the wrong (lat, lon) before the join — and that is exactly what this
    test pins down. End-to-end geographic correctness (a real ECMWF cell landing in the correct
    real-world hexagon) depends on that upstream asset and is out of scope here.

    See the orientation-coverage table in
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/#nwp-grid-h3-orientation-coverage>
    for how this test, the cached-real-slice test, and the geo landmark test divide the work.
    """
    ds = make_ens_dataset(
        latitudes=(52.0, 51.75),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "temperature_2m": np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        },
    )
    # Four single-point cells: (lat, lon) -> expected temperature.
    h3 = make_h3_grid(
        h3_index=[1, 2, 3, 4],
        nwp_lat=[52.0, 52.0, 51.75, 51.75],
        nwp_lon=[-1.0, -0.75, -1.0, -0.75],
        proportion=[1.0, 1.0, 1.0, 1.0],
    )

    df = convert(ds=ds, h3_grid=h3)

    temps = dict(zip(df["h3_index"].to_list(), df["temperature_2m"].to_list(), strict=True))
    assert temps == {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}


def test_convert_preserves_nulls_after_aggregation(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    # A NaN input for a nullable variable must survive aggregation as null, NOT be filled to 0
    # (guards the fill_nan-after-aggregation ordering in convert_to_polars).
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0,),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={"downward_short_wave_radiation_flux_surface": float("nan")},
    )
    df = convert(ds=ds, h3_grid=_single_cell_grid(make_h3_grid))

    assert df["downward_short_wave_radiation_flux_surface"].item() is None


def test_convert_categorical_precipitation_type(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    # One H3 cell fed by five grid points whose categories are [null, 1, 2, 2, 3]. The aggregation
    # (mode().first(ignore_nulls=True)) must return the dominant category 2 — which is neither the
    # min (1), the max (3), nor a value the null could corrupt. A NaN input becomes a null category
    # that must be ignored, not counted or forbidden by validation.
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75, -0.5, -0.25, 0.0),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "categorical_precipitation_type_surface": np.array(
                [[float("nan"), 1.0, 2.0, 2.0, 3.0]], dtype=np.float32
            )
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10, 10, 10, 10],
        nwp_lat=[52.0, 52.0, 52.0, 52.0, 52.0],
        nwp_lon=[-1.0, -0.75, -0.5, -0.25, 0.0],
        proportion=[0.2, 0.2, 0.2, 0.2, 0.2],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["categorical_precipitation_type_surface"].dtype == pl.UInt8
    assert df["categorical_precipitation_type_surface"].item() == 2


def test_full_pipeline_open_download_convert(
    monkeypatch: pytest.MonkeyPatch,
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """End-to-end: open (network patched) -> download -> convert -> validated Nwp frame."""
    ds = make_ens_dataset(init_time=_INIT_TIME, init_time_as_dim=True)
    monkeypatch.setattr(download.dynamical_catalog, "open", lambda *args, **kwargs: ds)

    h3 = make_h3_grid(
        h3_index=[10, 10, 20, 20],
        nwp_lat=[52.0, 52.0, 51.75, 51.75],
        nwp_lon=[-1.0, -0.75, -1.0, -0.75],
        proportion=[0.5, 0.5, 0.5, 0.5],
    )

    sliced = download.open_ecmwf_ens_run(_INIT_TIME, h3)
    downloaded = download.download_ecmwf_ens_data(sliced)
    df = convert(ds=downloaded, h3_grid=h3)

    Nwp.validate(df)
    assert df.height == 2 * 3 * 2
