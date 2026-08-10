"""Tests for ``dynamical_data.ecmwf_ens.convert_to_polars``.

These run entirely on in-memory synthetic datasets (see ``conftest.py``); the final integration
test drives the whole open -> download -> convert pipeline with the network call patched out.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import patito as pt
import polars as pl
import pytest
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import Nwp
from dynamical_data.ecmwf_ens import download
from dynamical_data.ecmwf_ens.convert_to_polars import (
    _process_chunk_for_1_lead_time_and_1_ens_member,
)
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe as convert,
)
from patito.exceptions import DataFrameValidationError

# After 2024-11-12 so the populated categorical_precipitation_type_surface column is valid; see
# Nwp._check_variables_that_were_introduced_after_start_of_dataset.
_INIT_TIME = datetime(2025, 1, 1, tzinfo=UTC)

# On or before 2024-11-12, where categorical_precipitation_type_surface is legitimately all-null —
# the only era in which a wholly-uncovered H3 cell reaches the numeric columns without that
# column's own historical invariant rejecting the frame first.
_INIT_TIME_BEFORE_PTYPE = datetime(2024, 1, 1, tzinfo=UTC)


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
        datetime(2025, 1, 1, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 6, tzinfo=UTC),
        datetime(2025, 1, 1, 12, tzinfo=UTC),
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
    ("wind_u", "wind_v", "expected_direction"),
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
    <https://openclimatefix.github.io/nged-substation-forecast/architecture/testing/#nwp-grid-h3-orientation-coverage>
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


def test_convert_renormalises_weights_over_a_corrupt_grid_point(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """One NaN grid point costs its own contribution to the cell, not the whole cell.

    The cell's surviving point carries full information about it, so the renormalised value is that
    point's own value rather than 0.75 of it. Without the renormalisation the cell would be NaN
    (NaN propagates through the weighted ``sum``), which is fatal for a non-nullable instantaneous
    variable.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={"temperature_2m": np.array([[10.0, float("nan")]], dtype=np.float32)},
    )
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.75, 0.25],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["temperature_2m"].item() == pytest.approx(10.0)


def test_convert_renormalises_a_deaccumulated_variable(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """The renormalised value is the mean of the surviving points, not their weighted sum.

    Five equally-weighted points, one corrupt: the answer must be the mean of the four that
    arrived (0.0025), not 0.8 of it (0.002). Pins the arithmetic rather than merely asserting the
    cell is non-null. Values are inside ``precipitation_surface``'s ``le=0.01`` bound.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75, -0.5, -0.25, 0.0),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "precipitation_surface": np.array(
                [[0.001, 0.002, 0.003, 0.004, float("nan")]], dtype=np.float32
            )
        },
    )
    h3 = make_h3_grid(
        h3_index=[10] * 5,
        nwp_lat=[52.0] * 5,
        nwp_lon=[-1.0, -0.75, -0.5, -0.25, 0.0],
        proportion=[0.2] * 5,
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["precipitation_surface"].item() == pytest.approx(0.0025)


def test_convert_renormalises_over_a_grid_point_missing_from_the_dataset(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """A grid point the weights name but the dataset lacks is excluded, not counted as zero.

    The H3 weights expect (52.0, -0.75); the dataset carries only (52.0, -1.0), so the join misses.
    Without renormalisation the absent point is silently worth 0, and the cell reports 0.75 * 10.0
    = 7.5 degC forever — a plausible-looking value that nothing downstream can detect.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0,),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={"temperature_2m": 10.0},
    )
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.75, 0.25],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["temperature_2m"].item() == pytest.approx(10.0)


def test_convert_renormalises_each_variable_independently(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """A null in one variable must not disturb another variable in the same cell.

    The contributing weight is per (cell, variable), never per cell. A single shared denominator
    would be wrong in both directions here: it would either drag ``temperature_2m`` off its correct
    15.0, or leave ``precipitation_surface`` un-renormalised at half its true value. That
    distinction matters far beyond this test, because the three de-accumulated variables are
    legitimately null at lead-0 in every single run.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "temperature_2m": np.array([[10.0, 20.0]], dtype=np.float32),
            "precipitation_surface": np.array([[0.004, float("nan")]], dtype=np.float32),
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.5, 0.5],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["temperature_2m"].item() == pytest.approx(0.5 * 10.0 + 0.5 * 20.0)
    assert df["precipitation_surface"].item() == pytest.approx(0.004)


def test_aggregation_of_a_wholly_uncovered_cell_is_null_not_zero(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """A cell whose every grid point is absent yields null, never a physically-plausible zero.

    Asserted on the aggregation itself rather than through ``convert``, because this is the defect:
    Polars sums an all-null group to 0.0, so the cell would otherwise land as 0 degC / 0 Pa —
    inside every bound the ``Nwp`` contract declares. ``convert``'s own rejection of that frame is
    the downstream symptom, covered by the test below.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,), longitudes=(-1.0,), lead_time_hours=(0,), ensemble_members=(0,)
    )
    # Cell 10 sits on the dataset's only grid point; cell 20's grid point is nowhere in it.
    h3 = make_h3_grid(
        h3_index=[10, 20],
        nwp_lat=[52.0, 49.0],
        nwp_lon=[-1.0, -5.0],
        proportion=[1.0, 1.0],
    )
    lat_grid, lon_grid = np.meshgrid(
        ds.latitude.values.astype(np.float32),
        ds.longitude.values.astype(np.float32),
        indexing="ij",
    )

    aggregated = _process_chunk_for_1_lead_time_and_1_ens_member(
        ds=ds.isel(lead_time=0, ensemble_member=0),
        h3_grid=h3,
        lat_grid=lat_grid.ravel(),
        lon_grid=lon_grid.ravel(),
    )

    uncovered = aggregated.filter(pl.col("h3_index") == 20)
    assert uncovered["temperature_2m"].item() is None
    assert uncovered["pressure_surface"].item() is None
    # The covered cell is untouched, so this is not merely "everything became null".
    assert aggregated.filter(pl.col("h3_index") == 10)["temperature_2m"].item() == pytest.approx(
        15.0
    )


def test_convert_rejects_a_run_with_a_wholly_uncovered_cell(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """A wholly-uncovered cell fails ingest, rather than landing as a plausible all-zero cell.

    The instantaneous variables are non-nullable, so the null from the aggregation above becomes a
    validation failure and the whole partition is missed. That is deliberately blunt — a missed run
    is rung 1 of the degradation ladder, whereas a silently-zeroed cell would train and serve
    forever — and it is temporary; see
    <https://github.com/openclimatefix/nged-substation-forecast/issues/478>.

    The init time is deliberately pre-2024-11-13. Later runs must carry a populated
    ``categorical_precipitation_type_surface``, and an uncovered cell nulls that column too, so
    they are rejected by that column's historical invariant before the numeric columns are ever
    judged. This era is therefore the only one where the numeric nulls are what fails the frame.
    """
    ds = make_ens_dataset(
        init_time=_INIT_TIME_BEFORE_PTYPE,
        latitudes=(52.0,),
        longitudes=(-1.0,),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={"categorical_precipitation_type_surface": float("nan")},
    )
    h3 = make_h3_grid(
        h3_index=[10, 20],
        nwp_lat=[52.0, 49.0],
        nwp_lon=[-1.0, -5.0],
        proportion=[1.0, 1.0],
    )

    # Match on "missing value" specifically: an implementation that let the empty cell through as
    # NaN rather than null would also raise here, but as an out-of-bounds error, which is not the
    # behaviour this test is for.
    with pytest.raises(DataFrameValidationError, match="missing value"):
        convert(ds=ds, h3_grid=h3)


def test_convert_nulls_a_multi_point_cell_whose_every_point_is_corrupt(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """Zero contributing weight yields null, not the 0.0 that a naive refactor would produce.

    ``test_convert_preserves_nulls_after_aggregation`` covers the single-point case; this is its
    multi-point sibling, and it is the one that would break if NaN were normalised to null before
    the aggregation without a contributing-weight denominator to catch the empty group. It is also
    the shape every lead-0 row of the three de-accumulated variables takes in every real run.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={"downward_short_wave_radiation_flux_surface": float("nan")},
    )
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.5, 0.5],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df.height == 1
    assert df["downward_short_wave_radiation_flux_surface"].item() is None


def test_convert_categorical_precipitation_type(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    # One H3 cell fed by five equally-weighted grid points whose categories are [null, 1, 2, 2, 3].
    # The area-weighted mode must return the dominant category 2 — which is neither the min (1),
    # the max (3), nor a value the null could corrupt. A NaN input becomes a null category that is
    # excluded from the ranking, not counted and not forbidden by validation.
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


def test_convert_weights_the_categorical_variable_by_area(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """The category covering most of the cell wins, not the one held by the most grid points.

    One dominant point at 0.9 says category 1; two slivers at 0.05 each agree on category 5. An
    unweighted mode reports 5, so the cell claims snow because a tenth of its area is snowy.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75, -0.5),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "categorical_precipitation_type_surface": np.array([[1.0, 5.0, 5.0]], dtype=np.float32)
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10, 10],
        nwp_lat=[52.0, 52.0, 52.0],
        nwp_lon=[-1.0, -0.75, -0.5],
        proportion=[0.9, 0.05, 0.05],
    )

    df = convert(ds=ds, h3_grid=h3)

    assert df["categorical_precipitation_type_surface"].item() == 1


def test_convert_weights_the_categorical_variable_within_each_cell_separately(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """Each cell ranks categories by *its own* area, not by area across the whole grid.

    The per-category weight is a window over ``(h3_index, variable)``. Dropping ``h3_index`` from
    those keys turns it into a grid-wide total, so every cell reports whichever category covers
    most of GB — and no single-cell test can see the difference. Here cell 10 is 0.6 category 1 to
    0.4 category 5, so it must report 1, while cell 20 is wholly category 5. A grid-wide weight
    would give category 5 a total of 1.4 against category 1's 0.6 and flip cell 10.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75, -0.5),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "categorical_precipitation_type_surface": np.array([[1.0, 5.0, 5.0]], dtype=np.float32)
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10, 20],
        nwp_lat=[52.0, 52.0, 52.0],
        nwp_lon=[-1.0, -0.75, -0.5],
        proportion=[0.6, 0.4, 1.0],
    )

    df = convert(ds=ds, h3_grid=h3).sort("h3_index")

    assert df["categorical_precipitation_type_surface"].to_list() == [1, 5]


def test_convert_does_not_let_missing_points_win_the_categorical_vote(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """A cell keeps its category when most of its points are missing but one supplied a value.

    Polars' ``mode()`` counts null as a candidate, so nulls that strictly outnumber every real
    value make the cell null even though data arrived — and that null is fatal for any
    ``init_time`` after 2024-11-12. Missing points must be excluded from the ranking instead.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75, -0.5),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "categorical_precipitation_type_surface": np.array(
                [[float("nan"), float("nan"), 7.0]], dtype=np.float32
            )
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10, 10],
        nwp_lat=[52.0, 52.0, 52.0],
        nwp_lon=[-1.0, -0.75, -0.5],
        proportion=[0.4, 0.35, 0.25],
    )

    df = convert(ds=ds, h3_grid=h3)

    # The single surviving point carries the least area of the three, so this cannot pass by the
    # weighting alone — the two nulls have to be out of the running entirely.
    assert df["categorical_precipitation_type_surface"].item() == 7


def test_convert_breaks_a_categorical_tie_deterministically(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """Two categories covering exactly half the cell each resolve to the lower category code.

    An evenly-split cell is ordinary geometry, and an unweighted ``mode().first()`` returns
    whichever value Polars happens to emit first — unspecified, and not necessarily stable across
    versions. This is a **regression guard, not a proof**: the old implementation cannot be made to
    fail it reliably, precisely because its answer was unspecified rather than wrong. What this
    pins is that the documented tie-break is the one actually implemented.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "categorical_precipitation_type_surface": np.array([[5.0, 1.0]], dtype=np.float32)
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.5, 0.5],
    )

    # The higher code is listed first, which an order-preserving implementation would return. An
    # unweighted mode is not even that: its answer for this input is unspecified, so the assertion
    # pins the documented tie-break rather than discriminating against any particular wrong one.
    assert convert(ds=ds, h3_grid=h3)["categorical_precipitation_type_surface"].item() == 1


def test_aggregation_nulls_the_categorical_variable_only_when_no_point_supplied_one(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """A cell whose every point lacks a category is null, matching the numeric rule.

    Asserted on the aggregation, because ``convert`` would reject this frame at the ``Nwp``
    boundary for an ``init_time`` after 2024-11-12 rather than returning it. This is a **regression
    guard**: the old implementation nulled this cell too, for the wrong reason. It is here because
    excluding missing points from the ranking must not go so far that a cell with nothing behind it
    acquires a category.
    """
    ds = make_ens_dataset(
        latitudes=(52.0,),
        longitudes=(-1.0, -0.75),
        lead_time_hours=(0,),
        ensemble_members=(0,),
        var_values={
            "categorical_precipitation_type_surface": np.array(
                [[float("nan"), float("nan")]], dtype=np.float32
            )
        },
    )
    h3 = make_h3_grid(
        h3_index=[10, 10],
        nwp_lat=[52.0, 52.0],
        nwp_lon=[-1.0, -0.75],
        proportion=[0.5, 0.5],
    )
    lat_grid, lon_grid = np.meshgrid(
        ds.latitude.values.astype(np.float32),
        ds.longitude.values.astype(np.float32),
        indexing="ij",
    )

    aggregated = _process_chunk_for_1_lead_time_and_1_ens_member(
        ds=ds.isel(lead_time=0, ensemble_member=0),
        h3_grid=h3,
        lat_grid=lat_grid.ravel(),
        lon_grid=lon_grid.ravel(),
    )

    assert aggregated["categorical_precipitation_type_surface"].item() is None
    # The numeric variables arrived, so this is not merely "the whole cell is empty".
    assert aggregated["temperature_2m"].item() == pytest.approx(15.0)


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
