"""Tests of the linear-program capacity estimators CX1 and CX2."""

import itertools
from types import SimpleNamespace
from typing import Final

import numpy as np
import pytest
from studies.capacity_convex import (
    CONVEX_DEFAULT_PARAMS,
    MIN_POINTS,
    WINDOW_DAYS,
    WindowPoints,
    classify_clipped,
    cx1_objective,
    cx1_trace,
    cx2_objective,
    cx2_traces,
    estimate_convex_plant,
    run_convex_estimator,
    window_points,
)
from studies.capacity_estimators import (
    CapacityRecord,
    reference_levels,
    scramble_from_day,
    synthetic_record,
    valid_days,
)

from studies import capacity_convex

DAYS: Final[int] = 680
LOSS: Final[slice] = slice(400, 460)
LOSS_PLANT: Final[int] = 1
EVALUATION_DAYS: Final[np.ndarray] = np.array(
    [380, 398, 402, 410, 420, 440, 459, 462, 470, 480, 500]
)


@pytest.fixture(scope="module")
def lossy_record() -> CapacityRecord:
    """A record whose plant 1 loses 30% of its capacity on days 400 to 459."""
    factor = np.ones((3, DAYS))
    factor[LOSS_PLANT, LOSS] = 0.7
    return synthetic_record(capacity_factor=factor)


@pytest.mark.parametrize("name", ["CX1", "CX2"])
def test_a_convex_estimator_recovers_a_planted_loss_and_returns_when_it_ends(
    name: str, lossy_record: CapacityRecord
) -> None:
    days = np.arange(200, 340, 7)
    reference = np.nanmedian(
        run_convex_estimator(
            name=name, record=lossy_record, params=None, days_by_plant=[days] * 3
        ).level[:, days],
        axis=1,
    )
    raw = run_convex_estimator(
        name=name, record=lossy_record, params=None, days_by_plant=[EVALUATION_DAYS] * 3
    )
    ratio = raw.level[LOSS_PLANT, EVALUATION_DAYS] / reference[LOSS_PLANT]
    by_day = dict(zip(EVALUATION_DAYS.tolist(), ratio, strict=True))
    assert abs(by_day[380] - 1.0) < 0.1
    assert abs(by_day[440] - 0.7) < 0.1
    assert abs(by_day[459] - 0.7) < 0.1
    assert abs(by_day[500] - 1.0) < 0.1
    # The plant without a loss stays flat.
    quiet = raw.level[0, EVALUATION_DAYS] / reference[0]
    assert np.abs(quiet - 1.0).max() < 0.15


@pytest.mark.parametrize("name", ["CX1", "CX2"])
def test_a_convex_estimate_for_day_d_ignores_every_input_after_day_d(name: str) -> None:
    record = synthetic_record(seed=3)
    day = 600
    params = CONVEX_DEFAULT_PARAMS[name]
    original = estimate_convex_plant(name=name, record=record, plant=0, params=params, days=[day])
    after = estimate_convex_plant(
        name=name,
        record=scramble_from_day(record=record, first_day=day + 1, seed=1),
        plant=0,
        params=params,
        days=[day],
    )
    from_day = estimate_convex_plant(
        name=name,
        record=scramble_from_day(record=record, first_day=day, seed=1),
        plant=0,
        params=params,
        days=[day],
    )
    assert original.level[day] == after.level[day]
    assert original.level[day] != from_day.level[day]


def _small_points(*, n_days: int, per_day: int, seed: int) -> WindowPoints:
    """Return a few hours per day with a day-to-day capacity change, for brute-force checks."""
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), per_day)
    unit_output = rng.uniform(0.3, 1.0, len(day))
    capacity = np.where(np.arange(n_days) < n_days // 2, 1.0, 0.7)[day]
    power = capacity * unit_output * rng.uniform(0.8, 1.0, len(day))
    return WindowPoints(day=day, power=power, unit_output=unit_output)


def test_cx1_matches_a_brute_force_search_over_four_days() -> None:
    """An optimal vertex of the program has every level at a point's ratio of power to output.

    So the exact minimum is the least objective over all assignments of those ratios to the days.
    """
    points = _small_points(n_days=4, per_day=4, seed=1)
    quantile, weight = 0.9, 0.3
    trace = cx1_trace(
        points=points, quantile=quantile, fused_weight=weight, plant="A", end_day=3, n_days=4
    )
    ratios = points.power / points.unit_output
    best = min(
        cx1_objective(
            trace=np.array(candidate), points=points, quantile=quantile, fused_weight=weight
        )
        for candidate in itertools.product(ratios, repeat=4)
    )
    found = cx1_objective(trace=trace, points=points, quantile=quantile, fused_weight=weight)
    assert found == pytest.approx(best, rel=1e-7)


def test_cx2_matches_a_brute_force_search_over_two_days_without_the_hinge() -> None:
    points = _small_points(n_days=2, per_day=4, seed=2)
    quantile, weight = 0.7, 0.2
    dc_trace, ac_trace = cx2_traces(
        points=points,
        quantile=quantile,
        fused_weight=weight,
        hinge_weight=0.0,
        plant="A",
        end_day=1,
        n_days=2,
    )
    clipped = classify_clipped(points=points)
    dc_levels = points.power[~clipped] / points.unit_output[~clipped]
    ac_levels = points.power[clipped]
    best = min(
        cx2_objective(
            dc_trace=np.array(dc),
            ac_trace=np.array(ac),
            points=points,
            quantile=quantile,
            fused_weight=weight,
            hinge_weight=0.0,
        )
        for dc in itertools.product(dc_levels, repeat=2)
        for ac in itertools.product(ac_levels, repeat=2)
    )
    found = cx2_objective(
        dc_trace=dc_trace,
        ac_trace=ac_trace,
        points=points,
        quantile=quantile,
        fused_weight=weight,
        hinge_weight=0.0,
    )
    assert found == pytest.approx(best, rel=1e-7)


def test_the_cx2_hinge_requires_the_direct_current_output_to_reach_the_alternating_limit() -> None:
    points = _small_points(n_days=6, per_day=6, seed=4)
    kwargs = {"quantile": 0.7, "fused_weight": 0.2, "plant": "A", "end_day": 5, "n_days": 6}
    free_dc, free_ac = cx2_traces(points=points, hinge_weight=0.0, **kwargs)
    held_dc, held_ac = cx2_traces(points=points, hinge_weight=50.0, **kwargs)
    clipped = classify_clipped(points=points)
    shortfall = (
        held_ac[points.day[clipped]] - points.unit_output[clipped] * held_dc[points.day[clipped]]
    )
    free_shortfall = (
        free_ac[points.day[clipped]] - points.unit_output[clipped] * free_dc[points.day[clipped]]
    )
    assert free_shortfall.max() > 1e-3
    assert shortfall.max() < 1e-6


def test_the_clipping_classifier_marks_only_the_hours_near_the_windows_top() -> None:
    power = np.concatenate([np.full(90, 0.5), np.full(9, 0.99), [1.0]])
    points = WindowPoints(day=np.zeros(100, dtype=int), power=power, unit_output=np.ones(100))
    assert classify_clipped(points=points).tolist() == [False] * 90 + [True] * 10


def test_a_window_leaves_out_invalid_days_and_low_sun_hours_and_needs_enough_points() -> None:
    record = synthetic_record(seed=5)
    valid = valid_days(record=record)[0].copy()
    valid[595:600] = False
    points = window_points(
        record=record, plant=0, end_day=600, unit_output=record.cams_shape[0], valid=valid
    )
    assert points is not None
    assert not np.isin(np.arange(WINDOW_DAYS - 6, WINDOW_DAYS - 1), points.day).any()
    assert (points.unit_output >= capacity_convex.MIN_UNIT_OUTPUT).all()
    assert len(points.day) >= MIN_POINTS
    # Before the record has 15 days there is no window.
    assert (
        window_points(
            record=record, plant=0, end_day=5, unit_output=record.cams_shape[0], valid=valid
        )
        is None
    )


def test_a_program_that_is_not_optimal_raises_naming_the_plant_and_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = _small_points(n_days=5, per_day=6, seed=3)
    monkeypatch.setattr(
        capacity_convex,
        "linprog",
        lambda *args, **kwargs: SimpleNamespace(status=2, message="infeasible", x=None),
    )
    with pytest.raises(RuntimeError, match=r"plant Q .* day index 77"):
        cx1_trace(points=points, quantile=0.9, fused_weight=1.0, plant="Q", end_day=77, n_days=5)


def test_an_estimator_that_is_not_convex_is_refused() -> None:
    record = synthetic_record(seed=6, days=30)
    with pytest.raises(ValueError, match="not a convex estimator"):
        estimate_convex_plant(name="PR", record=record, plant=0, params={}, days=[10])  # type: ignore[arg-type]


def test_calibrating_a_convex_level_by_its_reference_years_median_gives_the_static_power() -> None:
    record = synthetic_record(seed=7)
    days = np.arange(190, 330, 7)
    raw = run_convex_estimator(name="CX2", record=record, params=None, days_by_plant=[days] * 3)
    reference = reference_levels(raw=raw, record=record)
    assert np.isfinite(reference).all()
    assert (reference > 0).all()
