"""Tests of the four hard checks and the two fallback policies."""

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
from studies.capacity_evidence import Changepoint, Evidence, PlantFit
from studies.capacity_labeller import Answer, Knot
from studies.capacity_sanity import (
    apply_causal,
    apply_hindsight,
    check_anm,
    check_bounds,
    check_reproduces,
    check_steps,
)

DAYS = 200
START = np.datetime64("2024-01-01")
STATIC = 10.0
FIRST = date(2024, 1, 1)


def _evidence(
    *,
    ratio: np.ndarray | None = None,
    changes: tuple[Changepoint, ...] = (),
    binding: np.ndarray | None = None,
) -> Evidence:
    ratio = np.ones(DAYS) if ratio is None else ratio
    nans = np.full(DAYS, np.nan)
    return Evidence(
        plant="A",
        dates=START + np.arange(DAYS),
        static_p99_mw=STATIC,
        fit=PlantFit(ac_capacity_mw=9.0, dc_capacity_mw=11.0, tilt_deg=20.0, azimuth_deg=180.0),
        gap=np.zeros(DAYS, dtype=bool),
        stuck=np.zeros(DAYS, dtype=bool),
        clear_raw=ratio,
        clear_corrected=ratio,
        shared=nans,
        own=nans,
        neighbour=nans,
        energy_mwh=nans,
        modelled_mwh=nans,
        cap_binding_share=None if binding is None else binding.astype(float),
        cap_mean_mw=None if binding is None else np.full(DAYS, 5.0),
        months=("2024-01",),
        hour_table=np.full((1, 24), np.nan),
        envelope=np.full((1, 8), np.nan),
        plateau=np.full(1, np.nan),
        clip_share=np.full(1, np.nan),
        changepoints=changes,
    )


def _change(*, index: int, size: float) -> Changepoint:
    return Changepoint(
        day_index=index,
        day=(START + np.timedelta64(index, "D")).item(),
        direction="down" if size < 0 else "up",
        size=size,
        p_value=0.001,
        unconfirmed=False,
    )


def _knot(*, index: int, capacity: float, cause: str = "degradation") -> Knot:
    return Knot(day=FIRST + timedelta(days=index), capacity_mw=capacity, cause=cause, evidence=())  # ty: ignore[invalid-argument-type]


def _answer(*knots: Knot) -> Answer:
    return Answer(
        plant="A", first_day=FIRST, last_day=FIRST + timedelta(days=DAYS - 1),
        no_change=not knots, knots=knots,
    )  # fmt: skip


def test_a_knot_at_one_point_two_times_the_static_power_fails_the_bounds() -> None:
    assert [
        f.check for f in check_bounds(knots=(_knot(index=5, capacity=12.0),), static_p99_mw=STATIC)
    ] == ["bounds"]
    assert check_bounds(knots=(_knot(index=5, capacity=11.0),), static_p99_mw=STATIC) == []
    assert check_bounds(knots=(_knot(index=5, capacity=1.9),), static_p99_mw=STATIC)


def test_a_twenty_percent_step_fails_against_a_five_percent_candidate_not_a_twenty_one() -> None:
    knots = (_knot(index=99, capacity=10.0), _knot(index=100, capacity=8.0))
    small = _evidence(changes=(_change(index=100, size=-0.05),))
    matching = _evidence(changes=(_change(index=102, size=-0.19),))
    assert [f.day for f in check_steps(knots=knots, evidence=small)] == [knots[1].day]
    assert check_steps(knots=knots, evidence=matching) == []


def test_a_step_with_no_candidate_fails_and_a_small_change_is_not_a_step() -> None:
    knots = (_knot(index=99, capacity=10.0), _knot(index=100, capacity=8.0))
    assert check_steps(knots=knots, evidence=_evidence())
    flat = (_knot(index=99, capacity=10.0), _knot(index=100, capacity=9.9))
    assert check_steps(knots=flat, evidence=_evidence()) == []


def test_an_estimate_that_adds_a_spurious_step_fails_the_variance_check() -> None:
    ratio = np.ones(DAYS)
    ratio[100:] = 0.7
    evidence = _evidence(ratio=ratio)
    right = (
        _knot(index=0, capacity=10.0),
        _knot(index=99, capacity=10.0),
        _knot(index=100, capacity=7.0),
    )
    wrong = (
        _knot(index=0, capacity=10.0),
        _knot(index=149, capacity=10.0),
        _knot(index=150, capacity=5.0),
    )
    assert check_reproduces(knots=right, evidence=evidence, window_days=None) == []
    assert check_reproduces(knots=wrong, evidence=evidence, window_days=None)
    assert check_reproduces(knots=(), evidence=evidence, window_days=None) == []


def test_a_degradation_knot_inside_a_cap_run_fails_and_a_curtailment_knot_needs_a_cap_day() -> None:
    binding = np.zeros(DAYS, dtype=bool)
    binding[50:56] = True
    evidence = _evidence(binding=binding)
    assert check_anm(knots=(_knot(index=52, capacity=8.0),), evidence=evidence)
    assert (
        check_anm(knots=(_knot(index=52, capacity=8.0, cause="curtailment"),), evidence=evidence)
        == []
    )
    assert check_anm(
        knots=(_knot(index=120, capacity=8.0, cause="curtailment"),), evidence=evidence
    )
    assert check_anm(knots=(_knot(index=120, capacity=8.0),), evidence=evidence) == []
    short_run = np.zeros(DAYS, dtype=bool)
    short_run[50:52] = True
    assert (
        check_anm(knots=(_knot(index=50, capacity=8.0),), evidence=_evidence(binding=short_run))
        == []
    )


def test_the_causal_policy_falls_back_to_the_static_power_and_records_the_failing_check() -> None:
    answer = _answer(_knot(index=0, capacity=10.0), _knot(index=150, capacity=12.5))
    outcome = apply_causal(answer=answer, evidence=_evidence())
    assert outcome.fell_back
    assert outcome.answer.knots == ()
    assert "bounds" in {f.check for f in outcome.failures}
    clean = apply_causal(answer=_answer(), evidence=_evidence())
    assert not clean.fell_back
    assert clean.failures == ()


def test_the_hindsight_policy_drops_only_the_failing_knot() -> None:
    changes = (_change(index=100, size=-0.2),)
    ratio = np.ones(DAYS)
    ratio[100:] = 0.8
    evidence = _evidence(ratio=ratio, changes=changes)
    good = [
        _knot(index=0, capacity=10.0),
        _knot(index=99, capacity=10.0),
        _knot(index=100, capacity=8.0),
    ]
    bad = _knot(index=150, capacity=13.0)
    outcome = apply_hindsight(answer=_answer(*good, bad), evidence=evidence)
    assert [k.day for k in outcome.answer.knots] == [k.day for k in good]
    assert [f.day for f in outcome.failures if f.check == "bounds"] == [bad.day]
    assert not outcome.fell_back


def _noisy_ratio(*, level: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(5)
    return level * (1.0 + 0.03 * rng.standard_normal(DAYS))


def test_a_correct_constant_offset_passes_the_variance_check_and_keeps_its_knots() -> None:
    evidence = _evidence(ratio=_noisy_ratio(level=0.9))
    knots = (_knot(index=0, capacity=9.0), _knot(index=150, capacity=9.0))
    assert check_reproduces(knots=knots, evidence=evidence, window_days=None) == []
    kept = apply_hindsight(answer=_answer(*knots), evidence=evidence)
    assert len(kept.answer.knots) == 2
    supported = replace(evidence, changepoints=(_change(index=100, size=1.0),))
    assert not apply_causal(answer=_answer(*knots), evidence=supported).fell_back


def test_the_step_tolerance_is_five_points_or_forty_percent_of_the_candidate() -> None:
    knots = (_knot(index=99, capacity=10.0), _knot(index=100, capacity=8.0))
    for size in (-0.24, -0.30, -0.33):
        assert (
            check_steps(knots=knots, evidence=_evidence(changes=(_change(index=100, size=size),)))
            == []
        )
    assert check_steps(knots=knots, evidence=_evidence(changes=(_change(index=100, size=-0.38),)))
    assert check_steps(knots=knots, evidence=_evidence(changes=(_change(index=100, size=-0.14),)))


def test_a_candidate_counts_within_fourteen_days_of_the_knot_and_not_beyond() -> None:
    knots = (_knot(index=99, capacity=10.0), _knot(index=100, capacity=8.0))
    assert (
        check_steps(knots=knots, evidence=_evidence(changes=(_change(index=114, size=-0.2),))) == []
    )
    assert check_steps(knots=knots, evidence=_evidence(changes=(_change(index=116, size=-0.2),)))
    assert (
        check_steps(knots=knots, evidence=_evidence(changes=(_change(index=86, size=-0.2),))) == []
    )
    assert check_steps(knots=knots, evidence=_evidence(changes=(_change(index=84, size=-0.2),)))


def test_the_causal_policy_falls_back_when_only_one_check_other_than_bounds_fails() -> None:
    drop = np.ones(DAYS)
    drop[100:] = 0.8
    unsupported_step = (
        _knot(index=0, capacity=10.0),
        _knot(index=99, capacity=10.0),
        _knot(index=100, capacity=8.0),
    )
    outcome = apply_causal(answer=_answer(*unsupported_step), evidence=_evidence(ratio=drop))
    assert {f.check for f in outcome.failures} == {"step_within_evidence"}
    assert outcome.fell_back

    curtailment_without_a_cap = (
        _knot(index=0, capacity=10.0, cause="curtailment"),
        _knot(index=50, capacity=10.0),
    )
    outcome = apply_causal(answer=_answer(*curtailment_without_a_cap), evidence=_evidence())
    assert {f.check for f in outcome.failures} == {"anm_consistency"}
    assert outcome.fell_back

    supported_but_wrong = (
        _knot(index=0, capacity=10.0),
        _knot(index=149, capacity=10.0),
        _knot(index=150, capacity=7.0),
    )
    evidence = _evidence(ratio=_noisy_ratio(), changes=(_change(index=150, size=-0.3),))
    outcome = apply_causal(answer=_answer(*supported_but_wrong), evidence=evidence)
    assert {f.check for f in outcome.failures} == {"reproduces_the_power"}
    assert outcome.fell_back


def test_the_causal_variance_check_reads_only_the_latest_180_days() -> None:
    evidence = _evidence_of_length(ratio=_noisy_ratio_of(length=400))
    knots = (
        _knot(index=0, capacity=5.0),
        _knot(index=99, capacity=5.0),
        _knot(index=100, capacity=10.0),
    )
    assert check_reproduces(knots=knots, evidence=evidence, window_days=None)
    assert check_reproduces(knots=knots, evidence=evidence, window_days=180) == []
    supported = replace(evidence, changepoints=(_change(index=100, size=1.0),))
    assert not apply_causal(answer=_answer(*knots), evidence=supported).fell_back


def _noisy_ratio_of(*, length: int) -> np.ndarray:
    return 1.0 + 0.03 * np.random.default_rng(6).standard_normal(length)


def _evidence_of_length(*, ratio: np.ndarray) -> Evidence:
    base = _evidence()
    length = len(ratio)
    nans = np.full(length, np.nan)
    from dataclasses import replace

    return replace(
        base, dates=START + np.arange(length), gap=np.zeros(length, dtype=bool),
        stuck=np.zeros(length, dtype=bool), clear_raw=ratio, clear_corrected=ratio,
        shared=nans, own=nans, neighbour=nans, energy_mwh=nans, modelled_mwh=nans,
    )  # fmt: skip
