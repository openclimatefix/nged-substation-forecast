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
