"""Counting upstream nulls on the raw NWP grid, before the H3 aggregation absorbs them."""

from collections.abc import Callable
from typing import Final

import numpy as np
import patito as pt
import pytest
import xarray as xr
from contracts.geo_schemas import H3GridWeights
from contracts.weather_schemas import Nwp
from dynamical_data.ecmwf_ens.convert_to_polars import (
    convert_nwp_xarray_dataset_to_polars_dataframe as convert,
)
from dynamical_data.ecmwf_ens.upstream_nulls import assess_upstream_grid_point_nulls

_DEACCUMULATED: Final[frozenset[str]] = Nwp.deaccumulated_var_names

# The default fixture grid: 3 lead times (one of them lead-0), 2 members, 2x2 grid points.
_N_STEPS_BEYOND_LEAD_0: Final[int] = 2
_N_MEMBERS: Final[int] = 2
_N_GRID_POINTS: Final[int] = 4
_DEFAULT_SHAPE: Final[tuple[int, int, int, int]] = (3, _N_MEMBERS, 2, 2)
_EXPECTED_N_TOTAL: Final[int] = (
    len(_DEACCUMULATED) * _N_STEPS_BEYOND_LEAD_0 * _N_MEMBERS * _N_GRID_POINTS
)


def _two_points_per_cell() -> dict[str, list[float]]:
    """Grid-weight columns mapping two grid points into each of two H3 cells.

    ``default_h3_grid`` cannot express any test about the aggregation absorbing a null: it gives
    each cell a single point at ``proportion=1.0``, so one null point always nulls its cell.
    """
    return {
        "h3_index": [10, 10, 20, 20],
        "nwp_lat": [52.0, 52.0, 51.75, 51.75],
        "nwp_lon": [-1.0, -0.75, -1.0, -0.75],
        "proportion": [0.5, 0.5, 0.5, 0.5],
    }


def test_clean_run_counts_every_grid_point_beyond_lead_0(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    """A run with no upstream nulls is healthy, and the denominator spans every counted variable."""
    rate = assess_upstream_grid_point_nulls(ds=make_ens_dataset(), variables=_DEACCUMULATED)

    assert rate.is_healthy
    assert rate.n_null_nwp_grid_points == 0
    assert rate.null_nwp_grid_point_fraction == 0.0
    assert rate.n_affected_nwp_slices == 0
    assert rate.affected_nwp_variables == ()
    # Over all three de-accumulated variables, not one of them.
    assert rate.n_total_nwp_grid_points == _EXPECTED_N_TOTAL


def test_lead_0_nulls_are_excluded(make_ens_dataset: Callable[..., xr.Dataset]) -> None:
    """The de-accumulated variables are null at lead-0 by design, so lead-0 never counts."""
    all_null_at_lead_0 = np.full(_DEFAULT_SHAPE, 0.0001, dtype=np.float32)
    all_null_at_lead_0[0, :, :, :] = np.nan
    ds = make_ens_dataset(var_values=dict.fromkeys(_DEACCUMULATED, all_null_at_lead_0))

    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    assert rate.is_healthy
    assert rate.n_null_nwp_grid_points == 0
    assert rate.n_total_nwp_grid_points == _EXPECTED_N_TOTAL


def test_scattered_nulls_are_counted_exactly(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    """Two nulls in two different slices of one variable, counted and attributed."""
    precipitation = np.full(_DEFAULT_SHAPE, 0.0001, dtype=np.float32)
    precipitation[1, 0, 0, 0] = np.nan  # lead 6 h, member 0
    precipitation[2, 1, 1, 1] = np.nan  # lead 12 h, member 1
    ds = make_ens_dataset(var_values={"precipitation_surface": precipitation})

    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    assert not rate.is_healthy
    assert rate.n_null_nwp_grid_points == 2
    assert rate.n_affected_nwp_slices == 2
    assert rate.affected_nwp_variables == ("precipitation_surface",)
    assert rate.null_nwp_grid_point_fraction == pytest.approx(2 / _EXPECTED_N_TOTAL)


def test_scatter_the_aggregation_absorbs_is_still_counted(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """The blindness this measure exists for: no null cell survives, yet the feed was corrupt.

    One of a cell's two grid points arrives null, so renormalisation leaves the cell non-null and
    the stored run looks perfect. The grid-point count still reports the corruption.
    """
    precipitation = np.full(_DEFAULT_SHAPE, 0.0001, dtype=np.float32)
    precipitation[1, 0, 0, 0] = np.nan
    ds = make_ens_dataset(var_values={"precipitation_surface": precipitation})
    h3_grid = make_h3_grid(**_two_points_per_cell())

    nwp = convert(ds=ds, h3_grid=h3_grid)
    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    assert nwp["precipitation_surface"].null_count() == 0
    assert rate.n_null_nwp_grid_points == 1


def test_instantaneous_variables_are_not_counted(
    make_ens_dataset: Callable[..., xr.Dataset],
    make_h3_grid: Callable[..., pt.DataFrame[H3GridWeights]],
) -> None:
    """The mirror of the absorbed-scatter case, for a variable this measure deliberately ignores.

    A null in an instantaneous variable is anomalous rather than expected upstream corruption, so
    pooling it into one rate would average over opposite null semantics. It is out of scope here
    *and* absorbed by the aggregation, so nothing downstream sees it either.
    """
    temperature = np.full(_DEFAULT_SHAPE, 15.0, dtype=np.float32)
    temperature[1, 0, 0, 0] = np.nan
    ds = make_ens_dataset(var_values={"temperature_2m": temperature})
    h3_grid = make_h3_grid(**_two_points_per_cell())

    nwp = convert(ds=ds, h3_grid=h3_grid)
    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    assert nwp["temperature_2m"].null_count() == 0
    assert rate.n_null_nwp_grid_points == 0
    assert rate.is_healthy


def test_a_run_with_no_step_beyond_lead_0_reports_zero_rather_than_dividing(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    """A run carrying only lead-0 has nothing to measure, and must not raise.

    Lead-0-only runs are real: the committed real ECMWF slice is one. Raising here would let a
    warning path fail the run it is warning about.
    """
    rate = assess_upstream_grid_point_nulls(
        ds=make_ens_dataset(lead_time_hours=(0,)), variables=_DEACCUMULATED
    )

    assert rate.n_total_nwp_grid_points == 0
    assert rate.null_nwp_grid_point_fraction == 0.0
    assert rate.is_healthy
