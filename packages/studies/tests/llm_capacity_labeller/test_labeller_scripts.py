"""Tests for the helpers under `studies/llm_capacity_labeller/`: paths, stamps, the call list."""

from pathlib import Path

import labeller_jobs
import labeller_paths
import pytest
from labeller_paths import (
    FirstStudyEvents,
    check_stamp,
    claim_root,
    log_path,
    positive_control_events,
)
from studies.capacity_llm import assert_under_data


def _events(*, size: float, shape: str = "step") -> FirstStudyEvents:
    event = {
        "plant": 0,
        "shape": shape,
        "size": size,
        "first_day": 10,
        "month_days": 30,
        "hit_days": [],
    }
    payload = {
        "control": [event],
        "reserved": [],
        "false_movement_windows": [],
        "draws": [[event]],
        "scored_months": [],
        "low_plant": 0,
        "low_months": [],
        "anm_plant": 0,
    }
    return FirstStudyEvents(payload=payload)


def test_the_log_and_the_call_cache_live_under_data_for_real_and_smoke_runs() -> None:
    for root in (
        labeller_paths.STUDY_DIR,
        labeller_paths.STUDY_DIR / labeller_paths.SMOKE_DIR_NAME,
    ):
        assert_under_data(path=log_path(root=root))
        assert_under_data(path=labeller_paths.calls_dir(root=root))


def test_a_root_claimed_by_the_other_kind_of_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(labeller_paths, "STUDY_DIR", tmp_path)
    smoke = claim_root(smoke=True)
    assert smoke == tmp_path / "_smoke"
    (smoke / "RUN_KIND.txt").write_text("real\n")
    with pytest.raises(RuntimeError, match="claimed by a real run"):
        claim_root(smoke=True)
    real = claim_root(smoke=False)
    (real / "RUN_KIND.txt").write_text("smoke\n")
    with pytest.raises(RuntimeError, match="claimed by a smoke run"):
        claim_root(smoke=False)


def test_a_stamp_refuses_results_made_from_other_inputs_or_code(tmp_path: Path) -> None:
    check_stamp(directory=tmp_path / "a", stamp="one")
    check_stamp(directory=tmp_path / "a", stamp="one")
    with pytest.raises(RuntimeError, match="other inputs or code"):
        check_stamp(directory=tmp_path / "a", stamp="two")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "result.parquet").write_text("x")
    with pytest.raises(RuntimeError, match="no stamp"):
        check_stamp(directory=tmp_path / "b", stamp="one")


def test_the_positive_control_must_be_steps_of_the_copied_size() -> None:
    assert len(positive_control_events(first=_events(size=labeller_paths.CONTROL_SIZE))) == 1
    with pytest.raises(ValueError, match="control"):
        positive_control_events(first=_events(size=0.25))
    with pytest.raises(ValueError, match="control"):
        positive_control_events(first=_events(size=labeller_paths.CONTROL_SIZE, shape="ramp"))


def test_the_planned_contrasts_run_first_and_all_calls_total_138() -> None:
    jobs = labeller_jobs.hindsight_jobs(plants=6)
    assert len(jobs) == 66
    assert [job.arm for job in jobs[:36]] == ["tools_llm"] * 36
    assert {job.record_id for job in jobs[:36]} == set(labeller_jobs.HINDSIGHT_RECORDS)
    assert {job.arm for job in jobs[36:54]} == {"raw_llm"}
    assert {job.record_id for job in jobs[54:]} == set(labeller_jobs.CONTROL_RECORDS)
    causal = len(labeller_jobs.CAUSAL_RECORDS) * 6 * len(labeller_jobs.CAUSAL_ISSUE_DAYS)
    assert len(jobs) + causal == 138
    assert len(labeller_jobs.hindsight_jobs(plants=6, tools_samples=1)) == 48


def test_every_causal_issue_day_is_after_the_cut_off_and_the_period_starts_before_them() -> None:
    assert labeller_jobs.READABLE_LAG_DAYS == 2
    assert all(day > labeller_jobs.CAUSAL_PERIOD_FIRST for day in labeller_jobs.CAUSAL_ISSUE_DAYS)
    assert list(labeller_jobs.CAUSAL_ISSUE_DAYS) == sorted(labeller_jobs.CAUSAL_ISSUE_DAYS)


def test_round_two_lists_84_hindsight_calls_with_v2_tools_first_and_pairs_v1_on_the_new_draws() -> (
    None
):
    jobs = labeller_jobs.round2_jobs(plants=6)
    assert len(jobs) == 84
    assert {(job.version, job.record_id) for job in jobs[:36]} == {
        ("v2", record) for record in labeller_jobs.ROUND2_TOOLS_RECORDS
    }
    assert {(job.version, job.arm, job.record_id) for job in jobs[36:60]} == {
        ("v1", "tools_llm", record) for record in labeller_jobs.ROUND2_V1_RECORDS
    }
    assert {job.arm for job in jobs[60:72]} == {"raw_llm"}
    assert {job.record_id for job in jobs[72:]} == set(labeller_jobs.ROUND2_CONTROL_RECORDS)
    assert len({job.label for job in jobs}) == len(jobs)
    assert all(job.label.startswith(("r2/", "hindsight/")) for job in jobs)


def test_adjacent_control_months_at_one_plant_merge_into_one_run() -> None:
    from score_round2 import merge_adjacent_events, note_of
    from studies.capacity_recovery import PlantedEvent

    events = [
        PlantedEvent(plant=0, shape="step", size=0.3, first_day=100, month_days=30),
        PlantedEvent(plant=0, shape="step", size=0.3, first_day=130, month_days=31),
        PlantedEvent(plant=0, shape="step", size=0.3, first_day=300, month_days=30),
        PlantedEvent(plant=1, shape="step", size=0.3, first_day=161, month_days=30),
    ]
    merged = merge_adjacent_events(events=events)
    assert [(e.plant, e.first_day, e.month_days) for e in merged] == [
        (0, 100, 61),
        (0, 300, 30),
        (1, 161, 30),
    ]
    assert note_of(text='x {"plant": "A", "no_change": true, "note": "flat"} y') == "flat"
    assert note_of(text="no json") == ""


def test_round_three_lists_48_calls_easy_case_first_all_with_the_frozen_v2_arm() -> None:
    jobs = labeller_jobs.round3_jobs(plants=6)
    assert len(jobs) == 48
    assert {(job.version, job.arm) for job in jobs} == {("v2", "tools_llm")}
    assert {job.record_id for job in jobs[:24]} == set(labeller_jobs.ROUND3_EASY_RECORDS)
    assert {job.record_id for job in jobs[24:]} == set(labeller_jobs.ROUND3_DRAW_RECORDS)
    assert len({job.label for job in jobs}) == len(jobs)


def test_the_easy_case_and_the_new_draws_have_their_own_events() -> None:
    assert labeller_paths.EASY_SIZE == 0.6
    assert labeller_paths.EASY_DAYS == 90
    assert set(labeller_paths.ROUND3_DRAW_SEEDS).isdisjoint(labeller_paths.ROUND2_DRAW_SEEDS)
    assert set(labeller_paths.ROUND3_DRAW_SEEDS).isdisjoint({11441, 11442})


def test_round_four_lists_144_calls_grid_first_with_the_frozen_v2_arm() -> None:
    jobs = labeller_jobs.round4_jobs(plants=6)
    assert len(jobs) == 144
    assert {(job.version, job.arm) for job in jobs} == {("v2", "tools_llm")}
    assert {job.record_id for job in jobs[:48]} == set(labeller_jobs.ROUND4_GRID_RECORDS)
    assert {job.record_id for job in jobs[48:]} == set(labeller_jobs.ROUND4_DRAW_RECORDS)
    assert len({job.label for job in jobs}) == len(jobs)


def test_round_four_seeds_are_new_and_the_grid_has_two_designs() -> None:
    used = {11441, 11442, 11443, 11444, 11445, 11446}
    assert set(labeller_paths.ROUND4_DRAW_SEEDS).isdisjoint(used)
    assert len(labeller_paths.ROUND4_DRAW_SEEDS) == 8
    assert labeller_paths.GRID_DESIGNS == ((0.3, 90), (0.6, 30))
    assert len(labeller_paths.GRID_SEEDS) == 4
