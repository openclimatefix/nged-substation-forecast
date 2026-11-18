from datetime import UTC, datetime, timedelta

import patito as pt
import polars as pl
import pytest
from contracts.weather_schemas import (
    Nwp,
    NwpVariableWhollyMissing,
    assess_nwp_quality,
    variables_known_empty_at,
)

# A run initialised after every window in NWP_VARIABLE_EMPTY_WINDOWS, so that
# categorical_precipitation_type_surface and total_cloud_cover_atmosphere are both populated.
_INIT_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_BEYOND_LEAD0 = _INIT_TIME + timedelta(hours=3)
_LATER_BEYOND_LEAD0 = _INIT_TIME + timedelta(hours=6)

_DEACCUMULATED_VARS = (
    "precipitation_surface",
    "downward_short_wave_radiation_flux_surface",
    "downward_long_wave_radiation_flux_surface",
)


def _valid_value(deaccumulated_var: str) -> float:
    """A physically-in-bounds value for one de-accumulated variable.

    Precipitation is the exception: its `le` is 0.01.
    """
    return 0.001 if deaccumulated_var == "precipitation_surface" else 100.0


def _nwp_slice(
    *,
    init_time: datetime = _INIT_TIME,
    valid_time: datetime = _BEYOND_LEAD0,
    h3_indices: tuple[int, ...] = (100, 101, 102),
    member: int = 0,
    overrides: dict[str, list[object]] | None = None,
) -> pl.DataFrame:
    """A valid single-(member, valid_time) NWP slice spanning several h3 cells.

    Every column is physically valid by default; pass ``overrides={"column": [...]}`` to replace
    one with a per-cell list (e.g. inject nulls) so a test can build scattered vs whole-slice
    gaps.
    """
    n = len(h3_indices)
    columns: dict[str, object] = {
        "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * n,
        "init_time": [init_time] * n,
        "valid_time": [valid_time] * n,
        "ensemble_member": [member] * n,
        "h3_index": list(h3_indices),
        "categorical_precipitation_type_surface": [1] * n,
        "total_cloud_cover_atmosphere": [50.0] * n,
        "temperature_2m": [10.0] * n,
        "dew_point_temperature_2m": [5.0] * n,
        "wind_v_10m": [5.0] * n,
        "wind_u_10m": [0.0] * n,
        "wind_v_100m": [5.0] * n,
        "wind_u_100m": [0.0] * n,
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

    ``vertical_relaxed`` because an all-null override column arrives as Polars' `Null` dtype,
    which will not vstack onto the `Float64` of an intact slice; the relaxed supertype is the one
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
        "wind_v_10m",
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
    """Scattered null *cells* in a de-accumulated variable (the 2026-07-12 / #722 class, as it
    survives the H3 aggregation) are tolerated at ingest and surfaced by the quality assessor,
    not failed."""
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
    weather elsewhere in the run — the 2026-08-09 class, where 2 of that variable's 4284 (member,
    step) slices arrived empty.

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
    an absent column rather than a degraded one, which must fail ingest so the live forecast
    falls back on the previous (complete) run."""
    df = _nwp_run(
        _nwp_slice(overrides={deaccumulated_var: [None, None, None]}),
        _nwp_slice(valid_time=_LATER_BEYOND_LEAD0, overrides={deaccumulated_var: [None] * 3}),
    )
    with pytest.raises(NwpVariableWhollyMissing, match="wholesale absent"):
        _validate(df)


@pytest.mark.parametrize("deaccumulated_var", _DEACCUMULATED_VARS)
def test_wholly_missing_variable_is_judged_per_init_time(deaccumulated_var: str) -> None:
    """A run with an empty column is fatal even inside a frame that also holds a healthy run.

    Counting a variable's wholly-null slices against the *frame's* total slice count instead of
    its run's would let the healthy run's slices outvote the broken one, and the empty column
    would land.
    """
    other_init_time = _INIT_TIME + timedelta(days=1)
    df = _nwp_run(
        # The broken run: its only slice beyond lead-0 carries no weather for this variable.
        _nwp_slice(overrides={deaccumulated_var: [None] * 3}),
        # A healthy run alongside it.
        _nwp_slice(init_time=other_init_time, valid_time=other_init_time + timedelta(hours=3)),
    )
    with pytest.raises(NwpVariableWhollyMissing, match="wholesale absent"):
        _validate(df)


def test_wholly_missing_variables_are_all_named() -> None:
    """Two empty columns at once are both named, so the message is actionable rather than
    reporting whichever happened to sort first."""
    df = _nwp_run(
        _nwp_slice(
            overrides={
                "precipitation_surface": [None] * 3,
                "downward_short_wave_radiation_flux_surface": [None] * 3,
            }
        ),
        _nwp_slice(
            valid_time=_LATER_BEYOND_LEAD0,
            overrides={
                "precipitation_surface": [None] * 3,
                "downward_short_wave_radiation_flux_surface": [None] * 3,
            },
        ),
    )
    with pytest.raises(NwpVariableWhollyMissing) as exc_info:
        _validate(df)

    message = str(exc_info.value)
    assert "precipitation_surface" in message
    assert "downward_short_wave_radiation_flux_surface" in message


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
    check must abstain rather than treat "no surviving slices" as "the column is empty"."""
    df = _nwp_slice(valid_time=_INIT_TIME, overrides={deaccumulated_var: [None, None, None]})
    validated = _validate(df)  # does not raise
    assert assess_nwp_quality(validated).is_healthy


def test_lead0_deaccumulated_nulls_are_allowed() -> None:
    """De-accumulated variables are legitimately null at lead-0 (valid_time == init_time); that
    is not flagged as fatal nor as a quality issue."""
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


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("temperature_2m", 150.0),  # exceeds le=100
        ("temperature_2m", -150.0),  # exceeds ge=-100
        *[
            (f"wind_{axis}_{height}", value)
            for height in ("10m", "100m")
            for axis in ("u", "v")
            for value in (-201.0, 201.0)
        ],
    ],
)
def test_out_of_range_continuous_weather_var_is_fatal(column: str, bad_value: float) -> None:
    """A continuous variable outside its physical range must fail ingest.

    Guards `Nwp`'s declared `ge`/`le` bounds from inside the package that owns them: a widened
    range (e.g. `temperature_2m`'s `le=100` relaxed to `le=10000`, or `wind_u_10m`'s
    `le=200` relaxed to `le=3600`) would otherwise pass every test in this file undetected.
    """
    df = _nwp_slice(overrides={column: [bad_value, bad_value, bad_value]})
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _validate(df)


@pytest.mark.parametrize("bad_value", [300, -1])
def test_out_of_range_categorical_precipitation_type_is_fatal(bad_value: int) -> None:
    """`categorical_precipitation_type_surface` widened from `UInt8` to `Int16` (needed so the
    documented `255` "Missing" sentinel survives the write path), so its `0-255` range is no
    longer implied by the dtype alone and must be enforced by explicit `ge`/`le` bounds
    instead. New coverage: a `UInt8` column made an out-of-range value physically
    inexpressible, so this test never previously existed."""
    df = _nwp_slice(overrides={"categorical_precipitation_type_surface": [bad_value] * 3})
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _validate(df)


def test_unrecognised_nwp_model_id_is_fatal() -> None:
    """`nwp_model_id` widened from `Enum` to `String` (Delta cannot store `Enum`/`Categorical`),
    so its vocabulary is no longer enforced by the dtype and must be enforced by an explicit
    `constraints=` field instead. New coverage: the `Enum` dtype previously made an
    unrecognised model id un-constructible with a cast error, rather than a validation error."""
    df = _nwp_slice(overrides={"nwp_model_id": ["NOT_A_REAL_MODEL"] * 3})
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _validate(df)


def test_check_unique_allows_same_key_different_model_rows() -> None:
    """`_check_unique` must check `nwp_model_id` alongside the other four primary-key columns —
    `Nwp`'s own class docstring already documents `nwp_model_id` as part of the key. Checking
    fewer columns is *stricter*, not more lenient: it would wrongly reject two rows that agree on
    `(init_time, valid_time, ensemble_member, h3_index)` but differ in `nwp_model_id` — exactly
    the row shape a second ingested NWP model produces.

    Calls `_check_unique` directly rather than through `Nwp.validate()`: once the vocabulary
    constraint above is in place, only one `nwp_model_id` value is legal, so a two-model frame
    cannot pass the full validation pipeline at all.
    """
    df = _nwp_slice(h3_indices=(100,), overrides={"nwp_model_id": ["ECMWF_ENS_0_25_degree"]})
    other_model = _nwp_slice(h3_indices=(100,), overrides={"nwp_model_id": ["A_SECOND_MODEL"]})
    combined = pt.DataFrame(_nwp_run(df, other_model)).set_model(Nwp).cast()

    Nwp._check_unique(combined)  # does not raise


def test_continuous_weather_vars_are_stored_as_float32() -> None:
    """`Nwp`'s continuous variables are declared `Float32`, the physical-unit dtype the on-disk
    Delta table stores (see the `Nwp` docstring). Pins the literal dtype, not one derived from
    the model, so a field's declared dtype widening to `Float64` fails here rather than only
    downstream — as measured, such a change causes 26 failures elsewhere and none inside
    `contracts`.
    """
    validated = _validate(_nwp_slice())
    for var in sorted(Nwp.continuous_var_names()):
        assert validated[var].dtype == pl.Float32, var


def _run_at(
    init_time: datetime, *, ptype: int | None, cloud: float | None, cells: int = 2
) -> pl.DataFrame:
    """One run, beyond lead-0, with the two windowed variables set to the given value."""
    return _nwp_slice(
        init_time=init_time,
        valid_time=init_time + timedelta(hours=3),
        h3_indices=tuple(range(cells)),
        overrides={
            "categorical_precipitation_type_surface": [ptype] * cells,
            "total_cloud_cover_atmosphere": [cloud] * cells,
        },
    )


_BEFORE_WINDOWS = datetime(2024, 3, 31, tzinfo=UTC)
_FIRST_DAY_IN_WINDOWS = datetime(2024, 4, 1, tzinfo=UTC)
_LAST_PTYPE_EMPTY_DAY = datetime(2024, 11, 12, tzinfo=UTC)
_FIRST_PTYPE_POPULATED_DAY = datetime(2024, 11, 13, tzinfo=UTC)
_LAST_CLOUD_EMPTY_DAY = datetime(2025, 11, 20, tzinfo=UTC)
_FIRST_CLOUD_POPULATED_DAY = datetime(2025, 11, 21, tzinfo=UTC)


def test_windowed_variables_are_valid_in_every_era_of_the_archive() -> None:
    """Each run is valid when its two variables are populated or empty as upstream has them."""
    runs = _nwp_run(
        _run_at(datetime(2020, 10, 15, tzinfo=UTC), ptype=1, cloud=50.0),
        _run_at(_BEFORE_WINDOWS, ptype=1, cloud=50.0),
        _run_at(_FIRST_DAY_IN_WINDOWS, ptype=None, cloud=None),
        _run_at(_LAST_PTYPE_EMPTY_DAY, ptype=None, cloud=None),
        _run_at(_FIRST_PTYPE_POPULATED_DAY, ptype=1, cloud=None),
        _run_at(_LAST_CLOUD_EMPTY_DAY, ptype=1, cloud=None),
        _run_at(_FIRST_CLOUD_POPULATED_DAY, ptype=1, cloud=50.0),
    )

    _validate(runs)  # does not raise


@pytest.mark.parametrize(
    "init_time", [_FIRST_DAY_IN_WINDOWS, _LAST_PTYPE_EMPTY_DAY], ids=["first day", "last day"]
)
def test_ptype_must_be_all_null_in_its_empty_window(init_time: datetime) -> None:
    with pytest.raises(
        ValueError,
        match="categorical_precipitation_type_surface must be all null for init_time from "
        "2024-04-01 to 2024-11-12",
    ):
        _validate(_run_at(init_time, ptype=1, cloud=None))


@pytest.mark.parametrize(
    "init_time", [_FIRST_DAY_IN_WINDOWS, _LAST_CLOUD_EMPTY_DAY], ids=["first day", "last day"]
)
def test_total_cloud_must_be_all_null_in_its_empty_window(init_time: datetime) -> None:
    with pytest.raises(
        ValueError,
        match="total_cloud_cover_atmosphere must be all null for init_time from "
        "2024-04-01 to 2025-11-20",
    ):
        _validate(
            _run_at(
                init_time, ptype=None if init_time < _FIRST_PTYPE_POPULATED_DAY else 1, cloud=50.0
            )
        )


def test_ptype_wholly_null_before_its_window_is_fatal() -> None:
    with pytest.raises(ValueError, match="must not be wholly null for init_time outside"):
        _validate(_run_at(_BEFORE_WINDOWS, ptype=None, cloud=50.0))


def test_ptype_scattered_null_before_its_window_is_tolerated() -> None:
    """Before the window, only a wholly-null run is rejected: MARS-era cells are unchecked."""
    run = _nwp_slice(
        init_time=_BEFORE_WINDOWS,
        valid_time=_BEFORE_WINDOWS + timedelta(hours=3),
        overrides={"categorical_precipitation_type_surface": [1, None, 1]},
    )

    _validate(run)  # does not raise


def test_ptype_with_any_null_after_its_window_is_fatal() -> None:
    run = _nwp_slice(
        init_time=_FIRST_PTYPE_POPULATED_DAY,
        valid_time=_FIRST_PTYPE_POPULATED_DAY + timedelta(hours=3),
        overrides={
            "categorical_precipitation_type_surface": [1, None, 1],
            "total_cloud_cover_atmosphere": [None] * 3,
        },
    )
    with pytest.raises(ValueError, match="must not be null for init_time after 2024-11-12"):
        _validate(run)


@pytest.mark.parametrize("init_time", [_BEFORE_WINDOWS, _FIRST_CLOUD_POPULATED_DAY])
def test_total_cloud_wholly_null_outside_its_window_is_fatal(init_time: datetime) -> None:
    with pytest.raises(
        ValueError,
        match="total_cloud_cover_atmosphere must not be wholly null for init_time outside",
    ):
        _validate(_run_at(init_time, ptype=1, cloud=None))


def test_total_cloud_scattered_null_outside_its_window_is_tolerated() -> None:
    run = _nwp_slice(
        init_time=_FIRST_CLOUD_POPULATED_DAY,
        valid_time=_FIRST_CLOUD_POPULATED_DAY + timedelta(hours=3),
        overrides={"total_cloud_cover_atmosphere": [50.0, None, 50.0]},
    )

    _validate(run)  # does not raise


def test_a_wholly_null_run_is_found_even_beside_a_populated_run() -> None:
    """The wholly-null judgement is per `init_time`, so a healthy run cannot mask a missing one."""
    runs = _nwp_run(
        _run_at(_BEFORE_WINDOWS, ptype=1, cloud=50.0),
        _run_at(_BEFORE_WINDOWS - timedelta(days=1), ptype=1, cloud=None),
    )
    with pytest.raises(ValueError, match="2024-03-30"):
        _validate(runs)


@pytest.mark.parametrize("bad_value", [-1.0, 100.5])
def test_out_of_range_total_cloud_is_fatal(bad_value: float) -> None:
    df = _nwp_slice(overrides={"total_cloud_cover_atmosphere": [bad_value] * 3})
    with pytest.raises(pt.exceptions.DataFrameValidationError):
        _validate(df)


@pytest.mark.parametrize("value", [0.0, 100.0])
def test_total_cloud_bounds_are_inclusive(value: float) -> None:
    _validate(_nwp_slice(overrides={"total_cloud_cover_atmosphere": [value] * 3}))


def test_variables_known_empty_at_reads_the_windows() -> None:
    assert variables_known_empty_at(_BEFORE_WINDOWS) == frozenset()
    assert variables_known_empty_at(_FIRST_DAY_IN_WINDOWS) == {
        "categorical_precipitation_type_surface",
        "total_cloud_cover_atmosphere",
    }
    assert variables_known_empty_at(_FIRST_PTYPE_POPULATED_DAY) == {"total_cloud_cover_atmosphere"}
    assert variables_known_empty_at(_FIRST_CLOUD_POPULATED_DAY) == frozenset()


@pytest.mark.parametrize(
    ("column", "time", "expected_error"),
    [
        ("init_time", datetime(1840, 6, 1, tzinfo=UTC), "before MIN_PLAUSIBLE_DATETIME"),
        ("init_time", datetime(3000, 6, 1, tzinfo=UTC), "after MAX_PLAUSIBLE_DATETIME"),
        ("valid_time", datetime(1840, 6, 1, tzinfo=UTC), "before MIN_PLAUSIBLE_DATETIME"),
        ("valid_time", datetime(3000, 6, 1, tzinfo=UTC), "after MAX_PLAUSIBLE_DATETIME"),
    ],
)
def test_out_of_range_nwp_time_is_fatal(column: str, time: datetime, expected_error: str) -> None:
    """`init_time` and `valid_time` are bounded to the plausible range, same as PowerTimeSeries."""
    df = _nwp_slice(overrides={column: [time] * 3})
    with pytest.raises(ValueError, match=expected_error) as exc_info:
        _validate(df)

    assert f"`{column}` is outside the plausible datetime range" in str(exc_info.value)
