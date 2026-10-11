"""Tests of the planted losses, the recovery metrics, the false-movement gate, and the selection."""

import json
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.capacity_estimators import (
    FALLBACK_NONE,
    REFERENCE_END,
    REFERENCE_START,
    CapacityRecord,
    IssueScale,
    synthetic_record,
    valid_days,
)
from studies.capacity_recovery import (
    FALSE_MOVEMENT_GATE,
    FOOTPRINT_AFTER_DAYS,
    FOOTPRINT_BEFORE_DAYS,
    EventShapeType,
    PlantedEvent,
    SelectionInputs,
    calendar_month_spans,
    choose_estimators,
    detection_delay,
    draw_control_steps,
    draw_events,
    draw_primary_metric,
    event_factor,
    expected_factor,
    factor_matrix,
    false_movement_days,
    false_movement_share,
    false_movement_windows,
    interval_coverage,
    plant_losses,
    relative_capacity_error,
    score_events,
    t_interval,
    verdict_of,
    write_selection,
)

from studies import capacity_recovery

DAYS: Final[int] = 2500


@pytest.fixture(scope="module")
def long_record() -> CapacityRecord:
    """Three plants from 2019-09-01 for 2,500 days, each with every day valid."""
    return synthetic_record(plants=3, first_day=date(2019, 9, 1), days=DAYS, seed=21)


def _event(
    *,
    shape: EventShapeType = "step",
    size: float = 0.2,
    first: int = 100,
    length: int = 30,
    hit: tuple[int, ...] = (),
) -> PlantedEvent:
    return PlantedEvent(0, shape, size, first, length, hit)


def test_a_step_a_ramp_and_an_intermittent_loss_have_their_own_daily_factor() -> None:
    step = event_factor(event=_event(), days=200)
    assert step[99] == 1.0
    assert step[100:130] == pytest.approx(0.8)
    assert step[130] == 1.0
    ramp = event_factor(event=_event(shape="ramp", size=0.3), days=200)
    assert ramp[100] == pytest.approx(1.0)
    assert ramp[129] == pytest.approx(0.7)
    assert np.all(np.diff(ramp[100:130]) < 0)
    assert ramp[130] == 1.0
    hits = (0, 2, 4, 6)
    intermittent = event_factor(event=_event(shape="intermittent", hit=hits), days=200)
    assert np.flatnonzero(intermittent < 1.0).tolist() == [100, 102, 104, 106]
    # An intermittent loss is scored against the month's mean, 1 - size / 2.
    scored = expected_factor(event=_event(shape="intermittent", size=0.4, hit=hits), days=200)
    assert scored[100:130] == pytest.approx(0.8)
    assert scored[130] == 1.0


def test_planting_changes_exactly_the_drawn_months_and_scales_every_hour_of_a_day(
    long_record: CapacityRecord,
) -> None:
    events = [PlantedEvent(1, "step", 0.25, 800, 31), PlantedEvent(2, "ramp", 0.3, 1200, 30)]
    power = long_record.power_mw.copy()
    power[1, 805, :4] = np.nan
    planted = plant_losses(record=long_record.with_power(power_mw=power), events=events)
    factor = factor_matrix(events=events, plants=3, days=DAYS)
    changed = ~np.isclose(planted.power_mw, power, equal_nan=True).all(axis=2)
    assert np.array_equal(changed, factor != 1.0)
    assert np.allclose(planted.power_mw[1, 810], 0.75 * power[1, 810])
    assert np.isnan(planted.power_mw[1, 805, :4]).all()
    # Nothing but power moves.
    assert np.array_equal(planted.cams_ghi_w_m2, long_record.cams_ghi_w_m2)


def test_events_are_drawn_one_of_each_shape_per_plant_and_obey_the_footprint_rules(
    long_record: CapacityRecord,
) -> None:
    reference = (long_record.dates >= np.datetime64(REFERENCE_START)) & (
        long_record.dates < np.datetime64(REFERENCE_END)
    )
    valid = valid_days(record=long_record)
    for seed in range(5):
        events = draw_events(record=long_record, seed=seed)
        assert len(events) == 9
        for plant in range(3):
            mine = [e for e in events if e.plant == plant]
            assert sorted(e.shape for e in mine) == ["intermittent", "ramp", "step"]
        for event in events:
            start, stop = event.footprint
            assert stop - start == FOOTPRINT_BEFORE_DAYS + event.month_days + FOOTPRINT_AFTER_DAYS
            assert not reference[start:stop].any()
            assert 0.05 <= event.size <= 0.40
            assert event.first_day >= int(np.argmax(valid[event.plant])) + 60
        for index, first in enumerate(events):
            for second in events[index + 1 :]:
                assert not capacity_recovery._footprints_clash(first=first, second=second)
        assert events == draw_events(record=long_record, seed=seed)
    assert events != draw_events(record=long_record, seed=99)


def test_an_intermittent_event_hits_half_of_its_months_days(long_record: CapacityRecord) -> None:
    events = [e for e in draw_events(record=long_record, seed=3) if e.shape == "intermittent"]
    assert all(len(e.hit_days) == e.month_days // 2 for e in events)
    assert all(len(set(e.hit_days)) == len(e.hit_days) for e in events)


def test_a_reserved_event_and_an_earliest_day_keep_the_draw_away(
    long_record: CapacityRecord,
) -> None:
    first_days = [first for first, _ in calendar_month_spans(record=long_record).values()]
    reserved = [PlantedEvent(0, "step", 0.3, first_days[30], 31)]
    late = {1: first_days[45]}
    for seed in range(8):
        events = draw_events(
            record=long_record, seed=seed, reserved=reserved, earliest_first_day=late
        )
        for event in events:
            assert not capacity_recovery._footprints_clash(first=event, second=reserved[0])
            if event.plant == 1:
                assert event.first_day >= first_days[45]


def test_a_draw_that_cannot_place_an_event_raises(long_record: CapacityRecord) -> None:
    with pytest.raises(ValueError, match="cannot place the events"):
        draw_events(record=long_record, seed=0, earliest_first_day={0: DAYS})


def test_control_steps_use_distinct_months_and_avoid_each_plants_barred_months(
    long_record: CapacityRecord,
) -> None:
    spans = calendar_month_spans(record=long_record)
    scored = list(spans)[60:78]
    barred = {0: scored[:3]}
    for seed in range(6):
        events = draw_control_steps(
            record=long_record,
            months_per_plant=2,
            size=0.3,
            scored_months=scored,
            barred=barred,
            seed=seed,
        )
        firsts = [e.first_day for e in events]
        assert len(events) == 6
        assert len(set(firsts)) == 6
        assert all(e.size == 0.3 and e.shape == "step" for e in events)
        own = {first for first, _ in (spans[m] for m in barred[0])}
        assert not own & {e.first_day for e in events if e.plant == 0}
    with pytest.raises(ValueError, match="cannot hold"):
        draw_control_steps(
            record=long_record,
            months_per_plant=7,
            size=0.3,
            scored_months=scored,
            barred={},
            seed=0,
        )


def _scale(
    *, ratio: np.ndarray, plant: int = 0, sd: float | None = None
) -> tuple[IssueScale, IssueScale]:
    """Return an original scale of ones and a planted scale equal to `ratio` at one plant."""
    original = np.ones((1, DAYS))
    planted = original.copy()
    planted[plant, : len(ratio)] = ratio
    rsd = np.full((1, DAYS), np.nan if sd is None else sd)
    reason = np.zeros((1, DAYS), dtype=np.int8)
    return IssueScale(original, rsd, reason), IssueScale(planted, rsd, reason)


def test_the_primary_metric_is_the_mean_gap_to_the_expected_factor_over_the_scored_days() -> None:
    event = _event(size=0.2, first=100, length=30)
    _, perfect = _scale(ratio=event_factor(event=event, days=DAYS)[:200])
    original, _ = _scale(ratio=np.ones(1))
    scores = score_events(events=[event], planted=perfect, original=original, days=DAYS)
    assert scores["error"][0] == pytest.approx(0.0)
    _, blind = _scale(ratio=np.ones(200))
    blind_scores = score_events(events=[event], planted=blind, original=original, days=DAYS)
    assert blind_scores["error"][0] == pytest.approx(0.2 * 30 / 90)
    assert draw_primary_metric(scores=blind_scores) == pytest.approx(0.2 * 30 / 90)
    # A ratio that is wrong by 0.1 on every scored day gives 0.1, whatever the factor.
    off = event_factor(event=event, days=DAYS)[:200] + 0.1
    _, off_scale = _scale(ratio=off)
    assert score_events(events=[event], planted=off_scale, original=original, days=DAYS)["error"][
        0
    ] == pytest.approx(0.1)


def test_an_intermittent_event_is_scored_against_its_months_mean() -> None:
    event = _event(shape="intermittent", size=0.4, hit=(0, 1, 2, 3))
    ratio = np.ones(200)
    ratio[100:130] = 0.8
    original, planted = _scale(ratio=ratio)
    assert score_events(events=[event], planted=planted, original=original, days=DAYS)["error"][
        0
    ] == pytest.approx(0.0)


def test_the_detection_delay_is_the_days_until_the_ratio_first_reaches_halfway() -> None:
    event = _event(size=0.3, first=100, length=30)
    ratio = np.ones(90)
    ratio[5:] = 0.7
    assert detection_delay(event=event, ratio=ratio) == 5.0
    ratio_at_halfway = np.ones(90)
    ratio_at_halfway[8:] = 0.85
    assert detection_delay(event=event, ratio=ratio_at_halfway) == 8.0
    assert np.isnan(detection_delay(event=event, ratio=np.full(90, 0.9)))
    original, planted = _scale(ratio=np.concatenate([np.ones(100), ratio, np.ones(110)]))
    scored = score_events(
        events=[event, _event(shape="ramp")], planted=planted, original=original, days=DAYS
    )
    assert scored["delay"][0] == 5.0
    assert scored["delay"][1] is None


def test_interval_coverage_counts_days_whose_interval_holds_the_expected_factor() -> None:
    event = _event(size=0.2, first=100, length=30)
    expected = event_factor(event=event, days=DAYS)[100:190]
    ratio = expected.copy()
    ratio[:45] += 0.05
    covered_all = interval_coverage(
        event=event, ratio=ratio, relative_sd=np.full(90, 0.05), days=DAYS
    )
    assert covered_all == pytest.approx(1.0)
    half = interval_coverage(event=event, ratio=ratio, relative_sd=np.full(90, 0.02), days=DAYS)
    assert half == pytest.approx(0.5)
    assert np.isnan(
        interval_coverage(event=event, ratio=ratio, relative_sd=np.full(90, np.nan), days=DAYS)
    )


def test_the_false_movement_share_counts_days_more_than_five_percent_from_the_trailing_median() -> (
    None
):
    series = np.ones((1, 300))
    series[0, 205:] = 1.2
    original = IssueScale(series, np.full((1, 300), np.nan), np.zeros((1, 300), dtype=np.int8))
    # Days 205 to 209 sit 20% above a median of 1.0, which fewer than half the trailing days exceed.
    assert false_movement_share(windows=[(0, 200, 10)], original=original) == pytest.approx(0.5)
    assert false_movement_share(windows=[(0, 100, 10)], original=original) == 0.0
    nudged = series.copy()
    nudged[0, 200:210] = 1.04
    nudged_scale = IssueScale(nudged, np.full((1, 300), np.nan), np.zeros((1, 300), dtype=np.int8))
    assert false_movement_share(windows=[(0, 200, 10)], original=nudged_scale) == 0.0


def test_false_movement_windows_are_quiet_whole_months_chosen_by_seed(
    long_record: CapacityRecord,
) -> None:
    spans = calendar_month_spans(record=long_record)
    months = list(spans)[60:70]
    ratios = pl.DataFrame(
        {
            "site": ["A"] * 10 + ["B"] * 10,
            "month": months + months,
            "ratio": [0.7, 1.0, 0.99, 1.04, 1.2, 0.96, 0.5, 1.01, 1.0, 0.95] + [0.7] * 10,
        }
    )
    windows = false_movement_windows(ratios=ratios, record=long_record, per_plant=3, seed=1)
    assert len(windows) == 3
    assert all(plant == 0 for plant, _, _ in windows)
    quiet = {
        spans[m] for m, r in zip(months, ratios["ratio"][:10], strict=True) if 0.95 <= r <= 1.05
    }
    assert {(first, length) for _, first, length in windows} <= quiet
    assert windows == false_movement_windows(ratios=ratios, record=long_record, per_plant=3, seed=1)
    needed = false_movement_days(windows=windows)
    assert all(
        days[0] == first - 92 and days[-1] == first + length - 3
        for days, (_, first, length) in zip(needed, windows, strict=True)
    )


def test_the_t_interval_uses_the_students_quantile_for_the_number_of_draws() -> None:
    differences = np.arange(1.0, 11.0)
    mean, lower, upper = t_interval(differences=differences, confidence=0.975)
    half_width = 2.6850108468164575 * np.std(differences, ddof=1) / np.sqrt(10)
    assert mean == pytest.approx(5.5)
    assert (lower, upper) == (pytest.approx(5.5 - half_width), pytest.approx(5.5 + half_width))


def test_a_verdict_names_which_side_of_zero_the_interval_lies_on() -> None:
    assert verdict_of(lower=-0.3, upper=-0.1) == "beats"
    assert verdict_of(lower=0.1, upper=0.3) == "loses"
    assert verdict_of(lower=-0.1, upper=0.3) == "no detectable difference"
    assert verdict_of(lower=-0.3, upper=0.0) == "no detectable difference"


def _inputs(
    *, errors: dict[str, float], movement: dict[str, float], tuning: dict[str, float] | None = None
) -> SelectionInputs:
    rng = np.random.default_rng(0)
    draws = {name: value + 0.001 * rng.standard_normal(10) for name, value in errors.items()}
    return SelectionInputs(
        tuning_error=tuning or dict(errors), selection_errors=draws, false_movement=movement
    )


ALL_ESTIMATORS: Final[tuple[str, ...]] = ("R30", "CSQ", "NR", "PR", "KF", "HM", "CX1", "CX2", "DP1")


def test_the_gate_comes_first_and_the_selected_estimator_is_the_lowest_surviving_error() -> None:
    errors = dict(
        zip(ALL_ESTIMATORS, [0.12, 0.11, 0.05, 0.06, 0.07, 0.09, 0.04, 0.03, 0.02], strict=True)
    )
    movement = dict.fromkeys(ALL_ESTIMATORS, 0.02)
    movement["DP1"] = 0.5
    movement["CX2"] = FALSE_MOVEMENT_GATE + 1e-6
    movement["NR"] = FALSE_MOVEMENT_GATE
    selection = choose_estimators(inputs=_inputs(errors=errors, movement=movement))
    assert "DP1" not in selection.survivors
    assert "CX2" not in selection.survivors
    assert "NR" in selection.survivors
    assert selection.selected == "CX1"
    assert not selection.exploratory
    assert set(selection.shortlist) == {"NR", "HM", "CX1"}
    assert selection.representatives == {"cheap_and_wacky": "NR", "convex": "CX1"}
    assert [c.name for c in selection.contrasts] == ["E1"]
    assert selection.missing
    assert selection.missing[0].startswith("E2")


def test_representatives_are_chosen_on_the_tuning_draw_and_contrasts_use_the_selection_draws() -> (
    None
):
    errors = dict(
        zip(ALL_ESTIMATORS, [0.12, 0.11, 0.05, 0.06, 0.07, 0.09, 0.04, 0.03, 0.02], strict=True)
    )
    tuning = {**errors, "PR": 0.001, "CX1": 0.2}
    selection = choose_estimators(
        inputs=_inputs(errors=errors, movement=dict.fromkeys(ALL_ESTIMATORS, 0.0), tuning=tuning)
    )
    assert selection.representatives == {"cheap_and_wacky": "PR", "convex": "CX2", "pytorch": "DP1"}
    assert selection.selected == "DP1"
    inputs = _inputs(errors=errors, movement=dict.fromkeys(ALL_ESTIMATORS, 0.0), tuning=tuning)
    e1, e2 = selection.contrasts
    mean, lower, upper = t_interval(
        differences=inputs.selection_errors["CX2"] - inputs.selection_errors["PR"], confidence=0.975
    )
    assert (e1.treatment, e1.reference) == ("CX2", "PR")
    assert (e1.difference, e1.lower, e1.upper) == (
        pytest.approx(mean),
        pytest.approx(lower),
        pytest.approx(upper),
    )
    assert e1.verdict == "beats"
    assert (e2.treatment, e2.reference) == ("DP1", "CX2")
    assert e2.verdict == "beats"


def test_when_nothing_passes_the_gate_the_lowest_error_estimator_is_selected_as_exploratory() -> (
    None
):
    errors = dict(
        zip(ALL_ESTIMATORS, [0.12, 0.11, 0.05, 0.06, 0.07, 0.09, 0.04, 0.03, 0.02], strict=True)
    )
    selection = choose_estimators(
        inputs=_inputs(errors=errors, movement=dict.fromkeys(ALL_ESTIMATORS, 0.3))
    )
    assert selection.exploratory
    assert selection.selected == "DP1"
    assert selection.survivors == ()
    assert selection.shortlist == ("DP1",)
    assert len(selection.missing) == 2


def test_selection_json_records_the_choice_and_refuses_to_overwrite(tmp_path: Path) -> None:
    errors = dict(
        zip(ALL_ESTIMATORS, [0.12, 0.11, 0.05, 0.06, 0.07, 0.09, 0.04, 0.03, 0.02], strict=True)
    )
    inputs = _inputs(errors=errors, movement=dict.fromkeys(ALL_ESTIMATORS, 0.01))
    selection = choose_estimators(inputs=inputs)
    path = tmp_path / "selection.json"
    write_selection(path=path, selection=selection, inputs=inputs)
    saved = json.loads(path.read_text())
    assert saved["selected"] == selection.selected == "DP1"
    assert saved["shortlist"] == list(selection.shortlist)
    assert saved["contrasts"][0]["name"] == "E1"
    assert saved["mean_selection_error"]["R30"] == pytest.approx(
        float(np.mean(inputs.selection_errors["R30"]))
    )
    assert not (tmp_path / "selection.json.tmp").exists()
    with pytest.raises(FileExistsError):
        write_selection(path=path, selection=selection, inputs=inputs)


def test_choosing_among_no_estimators_is_refused() -> None:
    with pytest.raises(ValueError, match="no estimator"):
        choose_estimators(inputs=SelectionInputs({}, {}, {}))


def test_relative_capacity_error_is_zero_for_a_perfect_ratio() -> None:
    event = _event(shape="ramp", size=0.3)
    ratio = event_factor(event=event, days=DAYS)[100:190]
    assert relative_capacity_error(event=event, ratio=ratio, days=DAYS) == pytest.approx(0.0)
    assert FALLBACK_NONE == 0


def test_a_plant_with_little_valid_data_may_get_fewer_events_but_never_none(
    long_record: CapacityRecord,
) -> None:
    power = long_record.power_mw.copy()
    power[2, :2200] = np.nan
    sparse = long_record.with_power(power_mw=power)
    with pytest.raises(ValueError, match="cannot place the events"):
        draw_events(record=sparse, seed=1)
    events = draw_events(record=sparse, seed=1, may_have_fewer=[2])
    assert 1 <= sum(e.plant == 2 for e in events) < 3
    assert sum(e.plant == 0 for e in events) == 3
