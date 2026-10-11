"""Tests of knots, case files, prompts, and the cache key."""

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
    scrambled = scramble_from_day(record=record, first_day=last + 1, seed=5)
    kept = _evidence(record=record, last_day=last)
    other = _evidence(record=scrambled, last_day=last)
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
