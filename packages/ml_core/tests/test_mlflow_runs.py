"""Idempotency tests for the MLflow run-resolution helpers, against file-based MLflow."""

from pathlib import Path

import mlflow
import pytest
from ml_core._mlflow_runs import (
    get_or_create_experiment,
    get_or_create_fold_run,
    get_or_create_parent_run,
)
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
