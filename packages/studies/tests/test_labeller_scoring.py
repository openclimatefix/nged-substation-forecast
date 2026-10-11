"""Tests of the planted-event scoring and the event bootstrap."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from studies.capacity_estimators import synthetic_record
from studies.capacity_labeller import Answer, Knot
from studies.capacity_recovery import PlantedEvent
from studies.labeller_scoring import (
    answers_to_scale,
    effective_size,
    event_bootstrap_interval,
    false_alarms,
    score_draw,
    score_event,
    segments_of,
    synthetic_null_record,
)

DATES = np.datetime64("2023-06-01") + np.arange(400)
FIRST = date(2023, 6, 1)


def _knots(*points: tuple[int, float]) -> list[Knot]:
    return [
        Knot(day=FIRST + timedelta(days=day), capacity_mw=level, cause="degradation", evidence=())  # ty: ignore[invalid-argument-type]
        for day, level in points
    ]


def _event(*, shape: str = "step", size: float = 0.3) -> PlantedEvent:
    return PlantedEvent(plant=0, shape=shape, size=size, first_day=100, month_days=30)  # ty: ignore[invalid-argument-type]


def _step_down_and_up(
    *, drop_day: int = 100, recovery_day: int = 130, level: float = 7.0
) -> list[Knot]:
    return _knots(
        (0, 10.0),
        (drop_day - 1, 10.0),
        (drop_day, level),
        (recovery_day - 1, level),
        (recovery_day, 10.0),
    )


def test_a_drop_and_a_recovery_of_the_right_size_and_place_is_a_hit() -> None:
    score = score_event(
        event=_event(), segments=segments_of(knots=_step_down_and_up(), dates=DATES)
    )
    assert score.hit
    assert score.date_error_days == 0
    assert score.size_error == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("knots", "name"),
    [
        (_step_down_and_up(drop_day=140), "drop too late"),
        (_step_down_and_up(level=9.0), "drop too small"),
        (_step_down_and_up(level=3.0), "drop too large"),
        (_knots((0, 10.0), (99, 10.0), (100, 7.0)), "no recovery"),
        (_knots((0, 10.0), (99, 10.0), (100, 12.0), (129, 12.0), (130, 10.0)), "wrong sign"),
    ],
)
def test_a_wrong_date_size_sign_or_a_missing_recovery_is_not_a_hit(
    knots: list[Knot], name: str
) -> None:
    score = score_event(event=_event(), segments=segments_of(knots=knots, dates=DATES))
    assert not score.hit, name


def test_an_intermittent_loss_is_scored_against_its_window_mean() -> None:
    event = _event(shape="intermittent", size=0.4)
    assert effective_size(event=event) == pytest.approx(0.2)
    knots = _step_down_and_up(level=8.0)
    assert score_event(event=event, segments=segments_of(knots=knots, dates=DATES)).hit


def test_a_step_no_event_and_no_original_run_explains_is_a_false_alarm() -> None:
    stray = segments_of(knots=_knots((0, 10.0), (299, 10.0), (300, 7.0)), dates=DATES)
    assert false_alarms(events=[_event()], planted=stray, original=[]) == 1
    assert false_alarms(events=[_event()], planted=stray, original=stray) == 0
    on_event = segments_of(knots=_step_down_and_up(), dates=DATES)
    assert false_alarms(events=[_event()], planted=on_event, original=[]) == 0


def test_score_draw_scores_every_event_and_counts_false_alarms_once_per_plant() -> None:
    record = synthetic_record(plants=1, days=400)
    planted = {
        0: Answer(
            plant="A",
            first_day=FIRST,
            last_day=FIRST,
            no_change=False,
            knots=tuple(_step_down_and_up()),
        )
    }
    original = {0: Answer(plant="A", first_day=FIRST, last_day=FIRST, no_change=True, knots=())}
    table = score_draw(events=[_event()], planted=planted, original=original, record=record)
    assert table["hit"].to_list() == [True]
    assert table["false_alarms"].to_list() == [0]


def test_answers_without_knots_scale_to_the_static_power() -> None:
    record = synthetic_record(plants=2, days=400)
    scale = answers_to_scale(answers={}, record=record)
    assert np.allclose(scale.scale_mw, record.static_p99_mw[:, None])


def test_the_bootstrap_interval_excludes_zero_for_a_clear_difference_not_for_noise() -> None:
    rng = np.random.default_rng(0)
    clear = -0.03 + 0.005 * rng.standard_normal((33, 2))
    mean, lower, upper = event_bootstrap_interval(differences=clear, confidence=0.975, seed=1)
    assert lower < mean < upper < 0
    noise = 0.02 * rng.standard_normal((33, 2))
    _, lower, upper = event_bootstrap_interval(differences=noise, confidence=0.975, seed=1)
    assert lower < 0 < upper


def test_the_synthetic_null_record_keeps_the_gaps_and_resamples_the_quiet_days() -> None:
    record = synthetic_record(plants=3, days=400, seed=2)
    power = record.power_mw.copy()
    power[1, 50] = np.nan
    record = record.with_power(power_mw=power)
    windows = [(plant, 100, 60) for plant in range(3)]
    null = synthetic_null_record(record=record, windows=windows, seed=3)
    assert np.array_equal(np.isnan(null.power_mw), np.isnan(record.power_mw))
    assert not np.array_equal(null.power_mw, record.power_mw, equal_nan=True)
    assert (null.power_mw[np.isfinite(null.power_mw)] >= 0).all()
    again = synthetic_null_record(record=record, windows=windows, seed=3)
    assert np.array_equal(null.power_mw, again.power_mw, equal_nan=True)
    assert isinstance(pl.DataFrame(), pl.DataFrame)


def test_movement_counts_days_away_from_the_trailing_median_and_resamples_windows() -> None:
    from studies.labeller_scoring import movement_by_window, movement_interval

    scale = np.full((1, 300), 10.0)
    scale[0, 200:210] = 8.0
    counts = movement_by_window(windows=[(0, 195, 20), (0, 100, 20)], scale_mw=scale)
    assert counts.tolist() == [[10.0, 20.0], [0.0, 20.0]]
    share, lower, upper = movement_interval(per_window=counts, confidence=0.975, seed=1)
    assert share == 0.25
    assert lower <= share <= upper
