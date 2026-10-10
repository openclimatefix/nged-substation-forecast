"""Idempotency tests for the MLflow run-resolution helpers, against file-based MLflow."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pytest
from ml_core.mlflow_runs import (
    get_or_create_experiment,
    get_or_create_fold_run,
    get_or_create_parent_run,
    list_promotable_runs,
)
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point MLflow at a throwaway file-based store for the duration of a test."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")


def test_get_or_create_experiment_is_idempotent(mlflow_tracking: None) -> None:
    first = get_or_create_experiment("my_experiment")
    second = get_or_create_experiment("my_experiment")
    assert first == second


def test_get_or_create_parent_run_is_idempotent_and_tagged(mlflow_tracking: None) -> None:
    experiment_id = get_or_create_experiment("my_experiment")

    first = get_or_create_parent_run(experiment_id)
    second = get_or_create_parent_run(experiment_id)

    assert first == second
    run = MlflowClient().get_run(first)
    assert run.data.tags["cv_role"] == "parent"


def test_get_or_create_fold_run_is_idempotent_nested_and_tagged(mlflow_tracking: None) -> None:
    experiment_id = get_or_create_experiment("my_experiment")
    parent_run_id = get_or_create_parent_run(experiment_id)

    first = get_or_create_fold_run(experiment_id, parent_run_id, "2022")
    second = get_or_create_fold_run(experiment_id, parent_run_id, "2022")

    assert first == second
    run = MlflowClient().get_run(first)
    assert run.data.tags["cv_role"] == "fold"
    assert run.data.tags["fold_id"] == "2022"
    # Nested beneath the parent so the MLflow UI groups folds under cv_summary.
    assert run.data.tags["mlflow.parentRunId"] == parent_run_id


def test_fold_runs_resolve_by_tag(mlflow_tracking: None) -> None:
    experiment_id = get_or_create_experiment("my_experiment")
    parent_run_id = get_or_create_parent_run(experiment_id)

    fold_2022 = get_or_create_fold_run(experiment_id, parent_run_id, "2022")
    fold_2023 = get_or_create_fold_run(experiment_id, parent_run_id, "2023")

    assert fold_2022 != fold_2023
    # Re-resolving each fold returns its own run, distinct from the other.
    assert get_or_create_fold_run(experiment_id, parent_run_id, "2022") == fold_2022
    assert get_or_create_fold_run(experiment_id, parent_run_id, "2023") == fold_2023


def test_list_promotable_runs_lists_fold_runs_across_experiments_newest_first(
    mlflow_tracking: None,
) -> None:
    exp_a = get_or_create_experiment("experiment_a")
    parent_a = get_or_create_parent_run(exp_a)
    fold_a = get_or_create_fold_run(exp_a, parent_a, "2022")

    exp_b = get_or_create_experiment("experiment_b")
    parent_b = get_or_create_parent_run(exp_b)
    fold_b = get_or_create_fold_run(exp_b, parent_b, "2023")

    client = MlflowClient()
    client.set_terminated(run_id=fold_b, end_time=2_000_000_000_000)
    with mlflow.start_run(run_id=fold_a):
        mlflow.log_metric(key="retrained", value=1)
    client.set_terminated(run_id=fold_a, end_time=2_000_000_060_000)

    runs = list_promotable_runs()

    assert {run.run_id for run in runs} == {fold_a, fold_b}
    # The older run was resumed and finished last.
    assert [run.run_id for run in runs] == [fold_a, fold_b]
    by_id = {run.run_id: run for run in runs}
    assert by_id[fold_a].experiment_name == "experiment_a"
    assert by_id[fold_a].fold_id == "2022"
    assert by_id[fold_a].last_finished_at == datetime.fromtimestamp(2_000_000_060, tz=UTC)
    assert by_id[fold_b].experiment_name == "experiment_b"
    assert by_id[fold_b].fold_id == "2023"


def test_list_promotable_runs_excludes_parent_runs(mlflow_tracking: None) -> None:
    experiment_id = get_or_create_experiment("my_experiment")
    get_or_create_parent_run(experiment_id)  # cv_role=parent, not a fold — must be excluded.

    assert list_promotable_runs() == []


def test_list_promotable_runs_keeps_missing_end_times_last(mlflow_tracking: None) -> None:
    client = MlflowClient()
    experiment_id = get_or_create_experiment("unfinished")
    finished = client.create_run(
        experiment_id=experiment_id, start_time=1000, tags={"cv_role": "fold"}
    )
    client.set_terminated(run_id=finished.info.run_id, end_time=2000)
    unfinished = client.create_run(
        experiment_id=experiment_id, start_time=3000, tags={"cv_role": "fold"}
    )

    runs = list_promotable_runs()

    assert [run.run_id for run in runs] == [finished.info.run_id, unfinished.info.run_id]
    assert runs[1].last_finished_at is None


def test_list_promotable_runs_orders_before_limiting(
    mlflow_tracking: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MlflowClient()
    experiment_id = get_or_create_experiment("limited")
    older = client.create_run(
        experiment_id=experiment_id, start_time=1000, tags={"cv_role": "fold"}
    )
    newer = client.create_run(
        experiment_id=experiment_id, start_time=2000, tags={"cv_role": "fold"}
    )
    client.set_terminated(run_id=newer.info.run_id, end_time=3000)
    client.set_terminated(run_id=older.info.run_id, end_time=4000)
    search_runs = MlflowClient.search_runs

    def search_one_run(self: MlflowClient, **kwargs: Any) -> list[Run]:
        kwargs["max_results"] = 1
        return search_runs(self, **kwargs)

    monkeypatch.setattr(MlflowClient, "search_runs", search_one_run)

    assert [run.run_id for run in list_promotable_runs()] == [older.info.run_id]


def test_list_promotable_runs_omits_study_experiments(mlflow_tracking: None) -> None:
    reviewed = get_or_create_experiment("reviewed_experiment")
    reviewed_fold = get_or_create_fold_run(reviewed, get_or_create_parent_run(reviewed), "2022")
    study = get_or_create_experiment("study/some_study")
    get_or_create_fold_run(study, get_or_create_parent_run(study), "2022")

    runs = list_promotable_runs()

    assert [run.run_id for run in runs] == [reviewed_fold]
