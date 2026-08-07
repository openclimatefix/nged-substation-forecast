"""Unit tests for ``assess_nwp_run_completeness`` — the run-level shape check behind the
``nwp_run_is_complete`` asset check.

The synthetic "complete" run is built against the *real* ``ECMWF_ENS_ENSEMBLE_MEMBERS`` and
``ECMWF_ENS_LEAD_TIME_HOURS`` defaults (51 x 85 x 2 cells = 8,670 rows), so these tests would fail
if either constant drifted away from the shape the asset actually expects.
"""

from datetime import datetime, timedelta, timezone

import patito as pt
import polars as pl
import pytest
from contracts.weather_schemas import (
    ECMWF_ENS_ENSEMBLE_MEMBERS,
    ECMWF_ENS_LEAD_TIME_HOURS,
    Nwp,
    NwpRunCompletenessReport,
    assess_nwp_run_completeness,
)

_INIT_TIME = datetime(2024, 12, 1, tzinfo=timezone.utc)
"""A run initialised after 2024-11-12, when ``categorical_precipitation_type_surface`` became
non-null (``Nwp.validate`` enforces that split)."""

_H3_INDICES: tuple[int, ...] = (100, 101)
"""Two cells is enough to exercise the cell-count arithmetic while keeping the frame small."""

_CONTINUOUS_VALUES: dict[str, float] = {
    "temperature_2m": 10.0,
    "dew_point_temperature_2m": 5.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 180.0,
    "wind_speed_100m": 5.0,
    "wind_direction_100m": 180.0,
    "pressure_surface": 100_000.0,
    "pressure_reduced_to_mean_sea_level": 100_000.0,
    "geopotential_height_500hpa": 5000.0,
}

_DEACCUMULATED_VALUES: dict[str, float] = {
    "downward_long_wave_radiation_flux_surface": 100.0,
    "downward_short_wave_radiation_flux_surface": 100.0,
    "precipitation_surface": 0.001,
}


def _complete_run() -> pt.DataFrame[Nwp]:
    """A fully-populated ECMWF ENS run: every member x every native step x every H3 cell.

    The de-accumulated variables are null at lead-0, matching the real data (there is no previous
    step to difference against), which also keeps the frame realistic for ``Nwp.validate``.
    """
    grid = (
        pl.DataFrame({"ensemble_member": sorted(ECMWF_ENS_ENSEMBLE_MEMBERS)})
        .join(pl.DataFrame({"lead_time_hours": list(ECMWF_ENS_LEAD_TIME_HOURS)}), how="cross")
        .join(pl.DataFrame({"h3_index": list(_H3_INDICES)}), how="cross")
    )
    is_lead0 = pl.col("lead_time_hours") == 0
    df = (
        grid.with_columns(
            nwp_model_id=pl.lit("ECMWF_ENS_0_25_degree"),
            init_time=pl.lit(_INIT_TIME),
            valid_time=pl.lit(_INIT_TIME) + pl.duration(hours=pl.col("lead_time_hours")),
            categorical_precipitation_type_surface=pl.lit(1),
            **{name: pl.lit(value) for name, value in _CONTINUOUS_VALUES.items()},
            **{
                name: pl.when(is_lead0).then(None).otherwise(pl.lit(value))
                for name, value in _DEACCUMULATED_VALUES.items()
            },
        )
        .drop("lead_time_hours")
        .pipe(pt.DataFrame)
        .set_model(Nwp)
        .cast()
    )
    return Nwp.validate(df)


def _assess(df: pl.DataFrame) -> NwpRunCompletenessReport:
    """Assess a (possibly mutilated) run against the two-cell expectation used throughout."""
    return assess_nwp_run_completeness(
        pt.DataFrame(df).set_model(Nwp).cast(), expected_n_h3_cells=len(_H3_INDICES)
    )


def test_complete_run_passes() -> None:
    """The synthetic full grid is reported complete, with the expected marginal counts."""
    report = _assess(_complete_run())

    assert report.is_complete
    assert report.describe().startswith("Complete run:")
    assert report.n_ensemble_members == 51
    assert report.n_valid_times == 85
    assert report.n_h3_cells == 2
    assert report.n_rows == report.expected_n_rows == 51 * 85 * 2
    assert report.h3_cell_shortfall == 0
    assert report.valid_time_min == _INIT_TIME
    assert report.valid_time_max == _INIT_TIME + timedelta(days=15)


def test_native_step_structure_is_3_hourly_then_6_hourly() -> None:
    """Pin the step structure itself: 85 steps, 3-hourly to 144 h then 6-hourly to 360 h."""
    assert len(ECMWF_ENS_LEAD_TIME_HOURS) == 85
    assert ECMWF_ENS_LEAD_TIME_HOURS[0] == 0
    assert ECMWF_ENS_LEAD_TIME_HOURS[-1] == 24 * 15
    three_hourly = [h for h in ECMWF_ENS_LEAD_TIME_HOURS if h <= 144]
    six_hourly = [h for h in ECMWF_ENS_LEAD_TIME_HOURS if h >= 144]
    assert {b - a for a, b in zip(three_hourly, three_hourly[1:], strict=False)} == {3}
    assert {b - a for a, b in zip(six_hourly, six_hourly[1:], strict=False)} == {6}


def test_dropping_one_ensemble_member_is_caught_and_named() -> None:
    """A missing member is reported by index, and the row-count shortfall is reported too."""
    report = _assess(_complete_run().filter(pl.col("ensemble_member") != 17))

    assert not report.is_complete
    assert report.missing_ensemble_members == (17,)
    assert report.n_ensemble_members == 50
    assert report.n_valid_times == 85  # the other marginals are untouched
    assert report.n_h3_cells == 2
    assert report.n_rows == report.expected_n_rows - 85 * 2
    assert "missing ensemble member(s) [17]" in report.describe()


def test_dropping_one_h3_cell_is_caught_and_named() -> None:
    """A dropped grid cell shows up as a cell-count shortfall, not as a member or step gap."""
    report = _assess(_complete_run().filter(pl.col("h3_index") != 101))

    assert not report.is_complete
    assert report.n_h3_cells == 1
    assert report.h3_cell_shortfall == 1
    assert report.missing_ensemble_members == ()
    assert report.missing_lead_time_hours == ()
    assert "1 H3 cells, expected 2" in report.describe()


def test_truncated_forecast_horizon_is_caught_and_named() -> None:
    """Truncating the run at 10 days names every absent lead time, abbreviated past ten of them."""
    cutoff = _INIT_TIME + timedelta(days=10)
    report = _assess(_complete_run().filter(pl.col("valid_time") <= cutoff))

    assert not report.is_complete
    assert report.valid_time_max == cutoff
    assert report.missing_lead_time_hours == tuple(h for h in ECMWF_ENS_LEAD_TIME_HOURS if h > 240)
    assert report.missing_ensemble_members == ()
    description = report.describe()
    assert "missing lead time(s) in hours [246, 252" in description
    assert "+10 more]" in description  # 20 missing steps, capped at 10 in the sentence


def test_dropping_one_interior_lead_time_is_caught() -> None:
    """A single missing step in the middle of the run (the 2026-07-14 upstream failure mode)."""
    report = _assess(
        _complete_run().filter(pl.col("valid_time") != _INIT_TIME + timedelta(hours=6))
    )

    assert not report.is_complete
    assert report.missing_lead_time_hours == (6,)
    assert report.n_valid_times == 84


def test_off_grid_valid_time_is_reported_as_unexpected() -> None:
    """A ``valid_time`` that is not on any native step is surfaced rather than rounded onto one."""
    off_grid = _complete_run().with_columns(
        valid_time=pl.when(pl.col("valid_time") == _INIT_TIME + timedelta(hours=6))
        .then(pl.lit(_INIT_TIME + timedelta(hours=7)))
        .otherwise(pl.col("valid_time"))
    )
    report = _assess(off_grid)

    assert not report.is_complete
    assert report.missing_lead_time_hours == (6,)
    assert report.unexpected_valid_times == (_INIT_TIME + timedelta(hours=7),)


def test_ragged_run_with_complete_marginals_is_caught_by_the_row_count() -> None:
    """Every member, step and cell is present, but one (member, step, cell) row is missing.

    The marginal counts alone cannot see this, which is why ``is_complete`` also compares the row
    count against the full grid.
    """
    complete = _complete_run()
    hole = (
        (pl.col("ensemble_member") == 3)
        & (pl.col("valid_time") == _INIT_TIME + timedelta(hours=9))
        & (pl.col("h3_index") == 101)
    )
    report = _assess(complete.filter(~hole))

    assert report.n_ensemble_members == 51
    assert report.n_valid_times == 85
    assert report.n_h3_cells == 2
    assert not report.is_complete
    assert f"{report.expected_n_rows - 1} rows, expected {report.expected_n_rows}" in (
        report.describe()
    )


def test_two_init_times_in_one_frame_is_reported() -> None:
    """Completeness is a property of *one* run; a multi-run frame says so rather than raising."""
    complete = _complete_run()
    two_runs = pl.concat(
        [
            complete,
            complete.with_columns(
                init_time=pl.col("init_time") + pl.duration(days=1),
                valid_time=pl.col("valid_time") + pl.duration(days=1),
            ),
        ]
    )
    report = _assess(two_runs)

    assert not report.is_complete
    assert len(report.init_times) == 2
    assert "expected exactly one init_time, found 2" in report.describe()
    # With no single origin to measure from, no lead-time gaps are invented.
    assert report.missing_lead_time_hours == ()
    assert report.unexpected_valid_times == ()


def test_empty_frame_is_incomplete_and_does_not_raise() -> None:
    """The reporter never raises — an empty frame reports zeros, it does not blow up."""
    report = _assess(_complete_run().clear())

    assert not report.is_complete
    assert report.n_rows == 0
    assert report.valid_time_min is None
    assert report.valid_time_max is None
    assert report.missing_ensemble_members == tuple(sorted(ECMWF_ENS_ENSEMBLE_MEMBERS))


def test_validate_does_not_run_the_completeness_check() -> None:
    """``Nwp.validate`` must stay usable on arbitrary filtered subsets.

    Completeness is deliberately *not* wired into ``validate``: a single-member, single-step slice
    is a perfectly legal ``Nwp`` frame (it is what a pruned training scan returns), and validating
    it must not fail.
    """
    one_slice = _complete_run().filter(
        (pl.col("ensemble_member") == 0) & (pl.col("valid_time") == _INIT_TIME)
    )

    Nwp.validate(pt.DataFrame(one_slice).set_model(Nwp).cast())  # no raise

    # ...while the completeness reporter, asked directly, does flag it.
    assert not _assess(one_slice).is_complete


@pytest.mark.parametrize("expected_n_h3_cells", [1, 3])
def test_expectation_comes_from_the_caller_not_a_hard_coded_grid_size(
    expected_n_h3_cells: int,
) -> None:
    """The cell expectation is a caller argument (the asset passes the H3 grid weights' cell
    count), so a grid of a different size is judged against that grid, not against V1's 1671."""
    report = assess_nwp_run_completeness(_complete_run(), expected_n_h3_cells=expected_n_h3_cells)

    assert not report.is_complete
    assert report.expected_n_h3_cells == expected_n_h3_cells
