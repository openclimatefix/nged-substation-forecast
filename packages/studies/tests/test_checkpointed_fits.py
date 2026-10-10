from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from studies.arm_runner import Job
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS

from studies import checkpointed_fits

JOBS: list[Job] = [
    ("a", "primary", "y", ("x",), PRIMARY_HYPER_PARAMETERS, False),
    ("b", "primary", "y", ("x",), PRIMARY_HYPER_PARAMETERS, False),
    ("c", "primary", "y", ("x",), PRIMARY_HYPER_PARAMETERS, False),
]


def _fake_run_all(calls: list[list[str]]) -> Callable[..., pl.DataFrame]:
    def fake(*, dataset: pl.DataFrame, jobs: list, max_workers: int, device: str) -> pl.DataFrame:
        calls.append([name for name, *_ in jobs])
        return pl.concat(
            [
                pl.DataFrame({"arm": [name], "setting": [setting], "loss": [1.0]})
                for name, setting, *_ in jobs
            ]
        )

    return fake


def test_each_group_is_fitted_once_and_a_rerun_reads_the_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls: list[list[str]] = []
    monkeypatch.setattr(checkpointed_fits, "run_all", _fake_run_all(calls))
    kwargs = {
        "dataset": pl.DataFrame(),
        "jobs": JOBS,
        "checkpoint_dir": tmp_path,
        "max_workers": 1,
        "device": "cpu",
        "jobs_per_checkpoint": 2,
    }

    first = checkpointed_fits.fit_in_groups(**kwargs)
    second = checkpointed_fits.fit_in_groups(**kwargs)

    assert calls == [["a", "b"], ["c"]]
    assert sorted(first["arm"].to_list()) == ["a", "b", "c"]
    assert second.sort("arm").equals(first.sort("arm"))
    assert not list(tmp_path.glob("*.tmp"))


def test_a_checkpoint_holding_other_arms_than_its_name_says_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setattr(checkpointed_fits, "run_all", _fake_run_all([]))
    labels = checkpointed_fits.job_labels(group=JOBS[:2])
    wrong = pl.DataFrame({"arm": ["a", "z"], "setting": ["primary", "primary"], "loss": [1.0, 1.0]})
    wrong.write_parquet(tmp_path / checkpointed_fits.checkpoint_name(labels=labels))

    with pytest.raises(RuntimeError, match="holds"):
        checkpointed_fits.fit_in_groups(
            dataset=pl.DataFrame(),
            jobs=JOBS[:2],
            checkpoint_dir=tmp_path,
            max_workers=1,
            device="cpu",
            jobs_per_checkpoint=2,
        )


def test_the_checkpoint_name_depends_on_the_arms_and_not_on_their_order():
    forward = checkpointed_fits.job_labels(group=JOBS[:2])
    backward = checkpointed_fits.job_labels(group=[JOBS[1], JOBS[0]])
    other = checkpointed_fits.job_labels(group=JOBS[1:])

    assert checkpointed_fits.checkpoint_name(labels=forward) == checkpointed_fits.checkpoint_name(
        labels=backward
    )
    assert checkpointed_fits.checkpoint_name(labels=forward) != checkpointed_fits.checkpoint_name(
        labels=other
    )


def test_a_busy_machine_is_refused_unless_the_caller_ignores_the_load(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(checkpointed_fits.os, "getloadavg", lambda: (64.0, 0.0, 0.0))
    monkeypatch.setattr(checkpointed_fits.os, "cpu_count", lambda: 8)

    with pytest.raises(RuntimeError, match="load per core"):
        checkpointed_fits.refuse_if_machine_is_busy(ignore=False)
    checkpointed_fits.refuse_if_machine_is_busy(ignore=True)


def test_the_device_is_the_callers_choice_when_the_caller_makes_one():
    assert checkpointed_fits.choose_device(requested="cpu") == "cpu"
    assert checkpointed_fits.choose_device(requested="cuda") == "cuda"
