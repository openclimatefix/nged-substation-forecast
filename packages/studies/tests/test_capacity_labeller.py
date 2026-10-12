"""Tests of knots, case files, prompts, and the cache key."""

import re
from datetime import date

import numpy as np
import pytest
from studies.capacity_estimators import CapacityRecord, scramble_from_day, synthetic_record
from studies.capacity_evidence import Evidence, PlantFit, build_evidence
from studies.capacity_labeller import (
    SYSTEM_PROMPT,
    Answer,
    AnswerError,
    Knot,
    Prompt,
    allowed_window,
    build_prompt,
    cache_key,
    capacity_series,
    case_file_text,
    chart_pngs,
    parse_answer,
    ratio_chart_spec,
    tools_only_answer,
)

PLANTS = 6
DAYS = 700
FIRST = date(2024, 1, 1)
LAST = date(2024, 12, 31)
FIT = PlantFit(ac_capacity_mw=1.0, dc_capacity_mw=1.2, tilt_deg=25.0, azimuth_deg=180.0)


def _knot(*, day: date, capacity: float, cause: str = "degradation") -> dict[str, object]:
    return {"date": day.isoformat(), "capacity_mw": capacity, "cause": cause, "evidence": ["x"]}


def _reply(
    *, knots: list[dict[str, object]], no_change: bool | None = None, plant: str = "A"
) -> str:
    import json

    return json.dumps(
        {
            "plant": plant,
            "period": [FIRST.isoformat(), LAST.isoformat()],
            "no_change": (not knots) if no_change is None else no_change,
            "knots": knots,
        }
    )


def _parse(*, text: str) -> Answer:
    return parse_answer(text=text, plant="A", first_day=FIRST, last_day=LAST)


def test_a_valid_reply_inside_a_code_fence_parses() -> None:
    knots = [_knot(day=date(2024, 3, 1), capacity=5.0)]
    answer = _parse(text="```json\n" + _reply(knots=knots) + "\n```")
    assert answer.knots[0].capacity_mw == 5.0
    assert not answer.no_change


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        _reply(
            knots=[
                _knot(day=date(2024, 5, 1), capacity=5.0),
                _knot(day=date(2024, 3, 1), capacity=4.0),
            ]
        ),
        _reply(knots=[_knot(day=date(2023, 5, 1), capacity=5.0)]),
        _reply(knots=[_knot(day=date(2024, 5, 1), capacity=-1.0)]),
        _reply(knots=[_knot(day=date(2024, 5, 1), capacity=5.0, cause="aliens")]),
        _reply(knots=[], no_change=False),
        _reply(knots=[_knot(day=date(2024, 5, 1), capacity=5.0)], no_change=True),
        _reply(knots=[], plant="B"),
        _reply(
            knots=[
                _knot(day=date(2024, 5, 1), capacity=5.0),
                _knot(day=date(2024, 5, 1), capacity=4.0),
            ]
        ),
    ],
)
def test_an_invalid_reply_is_rejected(text: str) -> None:
    with pytest.raises(AnswerError):
        _parse(text=text)


def test_capacity_is_linear_between_knots_and_constant_outside_them() -> None:
    dates = np.datetime64("2024-01-01") + np.arange(11)
    knots = (
        Knot(day=date(2024, 1, 3), capacity_mw=10.0, cause="degradation", evidence=()),
        Knot(day=date(2024, 1, 7), capacity_mw=6.0, cause="degradation", evidence=()),
    )
    series = capacity_series(knots=knots, dates=dates, static_p99_mw=99.0)
    assert series.tolist() == [10, 10, 10, 9, 8, 7, 6, 6, 6, 6, 6]
    assert capacity_series(knots=(), dates=dates, static_p99_mw=99.0).tolist() == [99.0] * 11


def _prompt(*, text: str = "case", image: bytes = b"png") -> Prompt:
    return build_prompt(text=text, images=[image])


def test_one_changed_word_byte_sample_or_model_changes_the_cache_key() -> None:
    base = cache_key(model="m", prompt=_prompt(), sample_index=0)
    assert cache_key(model="m", prompt=_prompt(), sample_index=0) == base
    assert cache_key(model="m", prompt=_prompt(text="case2"), sample_index=0) != base
    assert cache_key(model="m", prompt=_prompt(image=b"pnh"), sample_index=0) != base
    assert cache_key(model="m", prompt=_prompt(), sample_index=1) != base
    assert cache_key(model="n", prompt=_prompt(), sample_index=0) != base
    changed_system = Prompt(system=SYSTEM_PROMPT + ".", text="case", images=(b"png",))
    assert cache_key(model="m", prompt=changed_system, sample_index=0) != base


@pytest.mark.parametrize("path", ["data/studies/x", "see /home/jack/data/studies/x", "'data/x'"])
def test_a_prompt_naming_a_path_under_data_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="data/"):
        build_prompt(text=f"open {path}", images=[])


def _record(*, factor: np.ndarray | None = None) -> CapacityRecord:
    return synthetic_record(plants=PLANTS, days=DAYS, capacity_factor=factor, seed=1)


def _evidence(*, record: CapacityRecord, last_day: int = DAYS - 1) -> Evidence:
    return build_evidence(record=record, plant=0, last_day=last_day, fit=FIT, cap=None)


def test_a_case_file_names_no_path_under_data_and_is_the_same_every_time() -> None:
    evidence = _evidence(record=_record())
    text = case_file_text(evidence=evidence, kind="tools", shown_months=None)
    assert text == case_file_text(evidence=evidence, kind="tools", shown_months=None)
    build_prompt(text=text, images=[])


def test_the_prompt_text_and_chart_bytes_ignore_every_input_after_the_cut_off_day() -> None:
    record = _record()
    last = 640
    scrambled = scramble_from_day(record=record, first_day=last + 1, seed=5, scramble_static=True)
    kept = _evidence(record=record, last_day=last)
    other = build_evidence(
        record=scrambled, plant=0, last_day=last, cap=None,
        fit=PlantFit(ac_capacity_mw=9.0, dc_capacity_mw=9.0, tilt_deg=1.0, azimuth_deg=2.0),
    )  # fmt: skip
    assert case_file_text(evidence=kept, kind="tools", shown_months=12) == case_file_text(
        evidence=other, kind="tools", shown_months=12
    )
    assert chart_pngs(evidence=kept, shown_months=12) == chart_pngs(evidence=other, shown_months=12)
    spec = ratio_chart_spec(evidence=kept)
    assert str(kept.dates[-1]) in str(spec)
    assert str(record.dates[-1]) not in str(spec)


def test_the_tools_only_rule_writes_a_step_at_a_strong_changepoint_and_nothing_otherwise() -> None:
    quiet = _evidence(record=_record())
    first, last = quiet.dates[0].astype(object), quiet.dates[-1].astype(object)
    assert tools_only_answer(evidence=quiet, first_day=first, last_day=last).no_change
    factor = np.ones((PLANTS, DAYS))
    factor[0, 450:] = 0.6
    dropped = _evidence(record=_record(factor=factor))
    answer = tools_only_answer(evidence=dropped, first_day=first, last_day=last)
    levels = [knot.capacity_mw for knot in answer.knots]
    assert answer.knots[0].day == first
    assert min(levels) < 0.7 * levels[0]
    assert all(a.day < b.day for a, b in zip(answer.knots, answer.knots[1:], strict=False))


def test_a_cut_record_prints_a_static_power_and_no_fit_computed_from_its_own_readings() -> None:
    record = _record()
    cut = _evidence(record=record, last_day=300)
    assert cut.fit is None
    assert cut.static_p99_mw != float(record.static_p99_mw[0])
    readings = record.power_mw[0, :301]
    assert cut.static_p99_mw == pytest.approx(float(np.quantile(readings[readings > 0], 0.99)))
    text = case_file_text(evidence=cut, kind="tools", shown_months=12)
    assert "Fitted AC" not in text
    whole = _evidence(record=record)
    assert whole.fit is not None
    assert whole.static_p99_mw == float(record.static_p99_mw[0])
    assert "Fitted AC" in case_file_text(evidence=whole, kind="tools", shown_months=None)


def test_a_reply_written_from_the_causal_text_alone_parses_and_the_stated_window_binds() -> None:
    evidence = _evidence(record=_record(), last_day=400)
    period_first = date(2024, 1, 1)
    text = case_file_text(
        evidence=evidence, kind="tools", shown_months=12, current_knots=(),
        issue_day=date(2024, 8, 1), period_first=period_first,
    )  # fmt: skip
    stated = re.search(r"Knot dates may run from (\S+) to (\S+)\.", text)
    assert stated is not None
    first_day, last_day = date.fromisoformat(stated.group(1)), date.fromisoformat(stated.group(2))
    assert (
        (first_day, last_day)
        == allowed_window(text=text)
        == (period_first, evidence.dates[-1].astype(date))
    )
    assert "first date of the period" not in SYSTEM_PROMPT
    opening = _reply(knots=[_knot(day=first_day, capacity=5.0)])
    answer = parse_answer(text=opening, plant="A", first_day=first_day, last_day=last_day)
    assert answer.knots[0].day == period_first
    before_window = _reply(knots=[_knot(day=date(2023, 6, 1), capacity=5.0)])
    with pytest.raises(AnswerError):
        parse_answer(text=before_window, plant="A", first_day=first_day, last_day=last_day)
    hindsight = case_file_text(
        evidence=_evidence(record=_record()), kind="tools", shown_months=None
    )
    assert allowed_window(text=hindsight)[0] == date(2023, 6, 1)
    with pytest.raises(ValueError, match="no window"):
        allowed_window(text="nothing")


def test_round_two_adds_the_local_scan_table_only_when_asked_and_keeps_round_one_text() -> None:
    from studies.capacity_labeller import SYSTEM_PROMPT_V2, local_scan_text

    factor = np.ones((PLANTS, DAYS))
    factor[0, 450:480] = 0.65
    evidence = _evidence(record=_record(factor=factor))
    plain = case_file_text(evidence=evidence, kind="tools", shown_months=None)
    scanned = case_file_text(evidence=evidence, kind="tools", shown_months=None, local_scan=True)
    assert "## Local scan" not in plain
    assert scanned.startswith(plain)
    assert "## Local scan" in scanned
    assert "down" in local_scan_text(evidence=evidence)
    assert SYSTEM_PROMPT_V2 != SYSTEM_PROMPT
    assert build_prompt(text=plain, images=[], system=SYSTEM_PROMPT_V2).system == SYSTEM_PROMPT_V2
    assert build_prompt(text=plain, images=[]).system == SYSTEM_PROMPT


def test_the_round_two_tools_only_rule_writes_a_knot_pair_bracketing_each_episode() -> None:
    from studies.capacity_labeller import tools_only_v2_answer

    quiet = _evidence(record=_record())
    first, last = quiet.dates[0].astype(object), quiet.dates[-1].astype(object)
    assert tools_only_v2_answer(evidence=quiet, first_day=first, last_day=last).no_change
    factor = np.ones((PLANTS, DAYS))
    factor[0, 450:480] = 0.65
    dropped = _evidence(record=_record(factor=factor))
    answer = tools_only_v2_answer(evidence=dropped, first_day=first, last_day=last)
    days = [knot.day for knot in answer.knots]
    assert days == sorted(set(days))
    levels = [knot.capacity_mw for knot in answer.knots]
    assert min(levels) < 0.8 * levels[0]
    low = [knot for knot in answer.knots if knot.capacity_mw < 0.8 * levels[0]]
    first_low = (low[0].day - first).days
    assert 435 <= first_low <= 465


def test_the_round_two_tools_only_rule_never_exceeds_the_knot_limit() -> None:
    from dataclasses import replace

    from studies.capacity_evidence import LocalEpisode
    from studies.capacity_labeller import MAX_KNOTS, tools_only_v2_answer

    evidence = _evidence(record=_record())
    episodes = tuple(
        LocalEpisode(
            first_day=40 * i + 10, last_day=40 * i + 39, deepest_start=40 * i + 10,
            direction="down", effect=-0.2 - 0.01 * i, all_sky_effect=float("nan"),
        )
        for i in range(15)
    )  # fmt: skip
    answer = tools_only_v2_answer(
        evidence=replace(evidence, local_episodes=episodes),
        first_day=evidence.dates[0].astype(object),
        last_day=evidence.dates[-1].astype(object),
    )
    assert 0 < len(answer.knots) <= MAX_KNOTS
