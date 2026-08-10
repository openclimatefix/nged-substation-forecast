from datetime import datetime, timedelta, timezone

import patito as pt
import polars as pl
import pytest
from contracts.weather_schemas import Nwp, NwpVariableWhollyMissing, assess_nwp_quality

# A run initialised after 2024-11-12, when categorical_precipitation_type_surface became non-null.
_INIT_TIME = datetime(2024, 12, 1, tzinfo=timezone.utc)
_BEYOND_LEAD0 = _INIT_TIME + timedelta(hours=3)
_LATER_BEYOND_LEAD0 = _INIT_TIME + timedelta(hours=6)

_DEACCUMULATED_VARS = (
    "precipitation_surface",
    "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface",
)


def _valid_value(deaccumulated_var: str) -> float:
    """A physically-in-bounds value for one de-accumulated variable (precipitation's `le` is 0.01)."""
    return 0.001 if deaccumulated_var == "precipitation_surface" else 100.0


def _nwp_slice(
    *,
    valid_time: datetime = _BEYOND_LEAD0,
    h3_indices: tuple[int, ...] = (100, 101, 102),
    member: int = 0,
    overrides: dict[str, list[object]] | None = None,
) -> pl.DataFrame:
    """A valid single-(member, valid_time) NWP slice spanning several h3 cells.

    Every column is physically valid by default; pass ``overrides={"column": [...]}`` to replace one
    with a per-cell list (e.g. inject nulls) so a test can build scattered vs whole-slice gaps.
    """
    n = len(h3_indices)
    columns: dict[str, object] = {
        "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * n,
        "init_time": [_INIT_TIME] * n,
        "valid_time": [valid_time] * n,
        "ensemble_member": [member] * n,
        "h3_index": list(h3_indices),
        "categorical_precipitation_type_surface": [1] * n,
        "temperature_2m": [10.0] * n,
        "dew_point_temperature_2m": [5.0] * n,
        "wind_speed_10m": [5.0] * n,
        "wind_direction_10m": [180.0] * n,
        "wind_speed_100m": [5.0] * n,
        "wind_direction_100m": [180.0] * n,
        "pressure_surface": [1000.0] * n,
        "pressure_reduced_to_mean_sea_level": [1000.0] * n,
        "geopotential_height_500hpa": [5000.0] * n,
        "downward_long_wave_radiation_flux_surface": [100.0] * n,
        "downward_short_wave_radiation_flux_surface": [100.0] * n,
        "precipitation_surface": [0.001] * n,
    }
    if overrides:
        columns.update(overrides)
    return pl.DataFrame(columns)


def _nwp_run(*slices: pl.DataFrame) -> pl.DataFrame:
    """Stack several `_nwp_slice` frames into one multi-slice run.

    ``vertical_relaxed`` because an all-null override column arrives as Polars' `Null` dtype, which
    will not vstack onto the `Float64` of an intact slice; the relaxed supertype is the one
    `_validate`'s model cast would land on anyway.
    """
    return pl.concat(slices, how="vertical_relaxed")


def _validate(df: pl.DataFrame) -> pt.DataFrame[Nwp]:
    return Nwp.validate(pt.DataFrame(df).set_model(Nwp).cast())


def test_valid_slice_passes_and_is_healthy() -> None:
    validated = _validate(_nwp_slice())
    report = assess_nwp_quality(validated)
    assert report.is_healthy
    assert report.n_null_cells == 0
    assert report.affected_variables == ()


@pytest.mark.parametrize(
    "instantaneous_var",
    [
        "temperature_2m",
        "pressure_surface",
        "wind_speed_10m",
        "geopotential_height_500hpa",
    ],
)
def test_instantaneous_null_beyond_lead0_is_fatal(instantaneous_var: str) -> None:
    """A structural gap that nulls an instantaneous variable (the 2026-07-14 class) must fail
    ingest — those fields are non-nullable, so base Patito validation rejects it."""
    df = _nwp_slice(overrides={instantaneous_var: [10.0, None, 10.0]})
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _validate(df)


@pytest.mark.parametrize("deaccumulated_var", _DEACCUMULATED_VARS)
def test_scattered_deaccumulated_null_beyond_lead0_is_tolerated(deaccumulated_var: str) -> None:
    """Scattered per-pixel nulls in a de-accumulated variable (the 2026-07-12 / #722 class) are
    tolerated at ingest and surfaced by the quality assessor, not failed."""
    valid_value = _valid_value(deaccumulated_var)
    df = _nwp_slice(
        overrides={deaccumulated_var: [valid_value, None, valid_value]}
    )  # 1 of 3 cells null

    validated = _validate(df)  # does not raise

    report = assess_nwp_quality(validated)
    assert not report.is_healthy
    assert report.n_null_cells == 1
    assert report.n_affected_slices == 1
    assert report.n_scattered_slices == 1
    assert report.n_whole_null_slices == 0
    assert report.affected_variables == (deaccumulated_var,)


@pytest.mark.parametrize("deaccumulated_var", _DEACCUMULATED_VARS)
def test_whole_slice_deaccumulated_null_beyond_lead0_is_tolerated(deaccumulated_var: str) -> None:
    """One wholly-null (member, valid_time) slice is tolerated while the variable still carries
    weather elsewhere in the run — the 2026-08-09 class, where 2 of 4284 slices arrived empty.

    Failing here would discard an otherwise-good run for a fraction of one already-nullable
    variable, so the slice is landed and reported by the quality assessor instead.
    """
    df = _nwp_run(
        _nwp_slice(overrides={deaccumulated_var: [None, None, None]}),  # wholly null
        _nwp_slice(valid_time=_LATER_BEYOND_LEAD0),  # intact
    )

    validated = _validate(df)  # does not raise

    report = assess_nwp_quality(validated)
    assert not report.is_healthy
    assert report.n_whole_null_slices == 1
    assert report.n_scattered_slices == 0
    assert report.n_null_cells == 3
    assert report.affected_variables == (deaccumulated_var,)


@pytest.mark.parametrize("deaccumulated_var", _DEACCUMULATED_VARS)
def test_wholly_missing_deaccumulated_variable_is_fatal(deaccumulated_var: str) -> None:
    """A de-accumulated variable null in *every* slice beyond lead-0 carries no weather at all —
    an absent column rather than a degraded one, which must fail ingest so the live forecast falls
    back on the previous (complete) run."""
    df = _nwp_run(
        _nwp_slice(overrides={deaccumulated_var: [None, None, None]}),
        _nwp_slice(valid_time=_LATER_BEYOND_LEAD0, overrides={deaccumulated_var: [None] * 3}),
    )
    with pytest.raises(NwpVariableWhollyMissing, match="wholesale absent"):
        _validate(df)


@pytest.mark.parametrize("deaccumulated_var", _DEACCUMULATED_VARS)
def test_wholly_missing_variable_is_judged_beyond_lead0_only(deaccumulated_var: str) -> None:
    """Lead-0 nulls never count towards "wholly missing": every de-accumulated variable is
    legitimately null there, so a run whose only null slice is lead-0 must pass."""
    df = _nwp_run(
        _nwp_slice(
            valid_time=_INIT_TIME,  # lead-0, legitimately null
            overrides={deaccumulated_var: [None, None, None]},
        ),
        _nwp_slice(),  # intact, beyond lead-0
    )
    validated = _validate(df)  # does not raise
    assert assess_nwp_quality(validated).is_healthy


@pytest.mark.parametrize("deaccumulated_var", _DEACCUMULATED_VARS)
def test_lead0_only_frame_is_not_wholly_missing(deaccumulated_var: str) -> None:
    """A frame with no rows beyond lead-0 has no slice that *could* carry weather, so the fatal
    check must abstain rather than read "every slice is null" off an empty set."""
    df = _nwp_slice(valid_time=_INIT_TIME, overrides={deaccumulated_var: [None, None, None]})
    validated = _validate(df)  # does not raise
    assert assess_nwp_quality(validated).is_healthy


def test_lead0_deaccumulated_nulls_are_allowed() -> None:
    """De-accumulated variables are legitimately null at lead-0 (valid_time == init_time); that is
    not flagged as fatal nor as a quality issue."""
    df = _nwp_slice(
        valid_time=_INIT_TIME,  # lead-0
        overrides={
            "precipitation_surface": [None, None, None],
            "downward_short_wave_radiation_flux_surface": [None, None, None],
            "downward_long_wave_radiation_flux_surface": [None, None, None],
        },
    )
    validated = _validate(df)
    assert assess_nwp_quality(validated).is_healthy


def test_categorical_precipitation_type_surface_validation() -> None:
    # Test case 1: Valid data (all null before 2024-11-12, not null after)
    df_valid = pl.DataFrame(
        {
            "nwp_model_id": ["ECMWF_ENS_0_25_degree", "ECMWF_ENS_0_25_degree"],
            "init_time": [
                datetime(2024, 11, 12, tzinfo=timezone.utc),
                datetime(2024, 11, 13, tzinfo=timezone.utc),
            ],
            "valid_time": [
                datetime(2024, 11, 12, 0, tzinfo=timezone.utc),
                datetime(2024, 11, 13, 1, tzinfo=timezone.utc),
            ],
            "ensemble_member": [1, 1],
            "h3_index": [1, 1],
            "categorical_precipitation_type_surface": [None, 1],
            "temperature_2m": [10.0, 10.0],
            "dew_point_temperature_2m": [5.0, 5.0],
            "wind_speed_10m": [5.0, 5.0],
            "wind_direction_10m": [180.0, 180.0],
            "wind_speed_100m": [5.0, 5.0],
            "wind_direction_100m": [180.0, 180.0],
            "pressure_surface": [1000.0, 1000.0],
            "pressure_reduced_to_mean_sea_level": [1000.0, 1000.0],
            "geopotential_height_500hpa": [5000.0, 5000.0],
            "downward_long_wave_radiation_flux_surface": [None, 100.0],
            "downward_short_wave_radiation_flux_surface": [None, 100.0],
            "precipitation_surface": [None, 0.01],
        }
    )

    # This should pass
    Nwp.validate(pt.DataFrame(df_valid).set_model(Nwp).cast())

    # Test case 2: Invalid data (not null before 2024-11-12)
    df_invalid_before = df_valid.with_columns(
        pl.Series("categorical_precipitation_type_surface", [1, 1])
    )
    with pytest.raises(ValueError, match="must be all null for init_time <= 2024-11-12"):
        Nwp.validate(pt.DataFrame(df_invalid_before).set_model(Nwp).cast())

    # Test case 3: Invalid data (null after 2024-11-12)
    df_invalid_after = df_valid.with_columns(
        pl.Series("categorical_precipitation_type_surface", [None, None])
    )
    with pytest.raises(ValueError, match="must not be null for init_time > 2024-11-12"):
        Nwp.validate(pt.DataFrame(df_invalid_after).set_model(Nwp).cast())


@pytest.mark.parametrize(
    "column, time, expected_error",
    [
        ("init_time", datetime(1840, 6, 1, tzinfo=timezone.utc), "before MIN_PLAUSIBLE_DATETIME"),
        ("init_time", datetime(3000, 6, 1, tzinfo=timezone.utc), "after MAX_PLAUSIBLE_DATETIME"),
        ("valid_time", datetime(1840, 6, 1, tzinfo=timezone.utc), "before MIN_PLAUSIBLE_DATETIME"),
        ("valid_time", datetime(3000, 6, 1, tzinfo=timezone.utc), "after MAX_PLAUSIBLE_DATETIME"),
    ],
)
def test_out_of_range_nwp_time_is_fatal(column: str, time: datetime, expected_error: str) -> None:
    """`init_time` and `valid_time` are bounded to the plausible range, same as PowerTimeSeries."""
    df = _nwp_slice(overrides={column: [time] * 3})
    with pytest.raises(ValueError, match=expected_error) as exc_info:
        _validate(df)

    assert f"`{column}` is outside the plausible datetime range" in str(exc_info.value)
