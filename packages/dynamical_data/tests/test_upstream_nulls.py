"""Counting upstream nulls on the raw NWP grid, before the H3 aggregation absorbs them."""

from collections.abc import Callable
from typing import Final

import numpy as np
import pytest
import xarray as xr
from contracts.weather_schemas import Nwp
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


@pytest.mark.parametrize(
    ("first", "second", "expected_n_slices"),
    [
        # Same step, different members: a slice is per member.
        ((1, 0, 0, 0), (1, 1, 0, 0), 2),
        # Same member, different steps: a slice is per step.
        ((1, 0, 0, 0), (2, 0, 0, 0), 2),
        # Same (member, step): one slice however many of its grid points are null.
        ((1, 0, 0, 0), (1, 0, 0, 1), 1),
    ],
)
def test_a_slice_is_one_variable_member_and_step(
    make_ens_dataset: Callable[..., xr.Dataset],
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
    expected_n_slices: int,
) -> None:
    """Both slice keys are pinned independently, and neither is the grid.

    Each case fixes one index and varies the other, so a count grouped by the wrong dimension gives
    a different answer to at least one of them. ``n_affected_nwp_slices`` exists to separate one bad
    slice from a hundred, which a grouping that silently drops ``ensemble_member`` or ``lead_time``
    would not do.
    """
    precipitation = np.full(_DEFAULT_SHAPE, 0.0001, dtype=np.float32)
    precipitation[first] = np.nan
    precipitation[second] = np.nan
    ds = make_ens_dataset(var_values={"precipitation_surface": precipitation})

    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    assert rate.n_null_nwp_grid_points == 2
    assert rate.n_affected_nwp_slices == expected_n_slices


def test_per_variable_breakdown_counts_each_variable_separately(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    """Which variable is corrupt, and by how much — neither of which the pooled totals can say.

    The two corrupt variables differ in both counts, and in the relationship between them: two
    nulls in one slice against one null in another. A breakdown that split the totals evenly, or
    attributed either variable's nulls to the other, disagrees with this.
    """
    precipitation = np.full(_DEFAULT_SHAPE, 0.0001, dtype=np.float32)
    precipitation[1, 0, 0, 0] = np.nan
    precipitation[1, 0, 1, 1] = np.nan  # the same (member, step) slice, so two nulls but one slice
    radiation = np.full(_DEFAULT_SHAPE, 200.0, dtype=np.float32)
    radiation[2, 1, 0, 0] = np.nan
    ds = make_ens_dataset(
        var_values={
            "precipitation_surface": precipitation,
            "downward_short_wave_radiation_flux_surface": radiation,
        }
    )

    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    n_per_variable = _N_STEPS_BEYOND_LEAD_0 * _N_MEMBERS * _N_GRID_POINTS
    # Rows are sorted by variable name, so the clean variable comes first here.
    assert rate.per_variable.to_dicts() == [
        {
            "variable": "downward_long_wave_radiation_flux_surface",
            "n_null": 0,
            "n_affected_slices": 0,
            "n_total": n_per_variable,
        },
        {
            "variable": "downward_short_wave_radiation_flux_surface",
            "n_null": 1,
            "n_affected_slices": 1,
            "n_total": n_per_variable,
        },
        {
            "variable": "precipitation_surface",
            "n_null": 2,
            "n_affected_slices": 1,
            "n_total": n_per_variable,
        },
    ]
    assert rate.n_null_nwp_grid_points == 3
    assert rate.n_affected_nwp_slices == 2


def test_affected_variables_are_sorted(make_ens_dataset: Callable[..., xr.Dataset]) -> None:
    """Two corrupt variables come back in a stable, ascending order."""
    corrupt = np.full(_DEFAULT_SHAPE, 0.0001, dtype=np.float32)
    corrupt[1, 0, 0, 0] = np.nan
    radiation = np.full(_DEFAULT_SHAPE, 200.0, dtype=np.float32)
    radiation[1, 0, 0, 0] = np.nan
    ds = make_ens_dataset(
        var_values={
            "precipitation_surface": corrupt,
            "downward_short_wave_radiation_flux_surface": radiation,
        }
    )

    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

    assert rate.affected_nwp_variables == (
        "downward_short_wave_radiation_flux_surface",
        "precipitation_surface",
    )


def test_instantaneous_variables_are_not_counted(
    make_ens_dataset: Callable[..., xr.Dataset],
) -> None:
    """A null in an instantaneous variable is anomalous, not the tolerated corruption this counts.

    Pooling the two into one rate would average over opposite null semantics.
    """
    temperature = np.full(_DEFAULT_SHAPE, 15.0, dtype=np.float32)
    temperature[1, 0, 0, 0] = np.nan
    ds = make_ens_dataset(var_values={"temperature_2m": temperature})

    rate = assess_upstream_grid_point_nulls(ds=ds, variables=_DEACCUMULATED)

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
