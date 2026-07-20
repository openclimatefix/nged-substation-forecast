"""Integration test for ``register_experiment_job`` (file-based MLflow + ephemeral instance).

Exercises the real job wiring: it must create the MLflow experiment + parent run (with grouping
tags and flattened config params) and add the right ``cv_experiment_folds`` partition keys, all
idempotently. The pure helpers it delegates to are unit-tested in ``packages/ml_core/tests``.
"""

from pathlib import Path

import mlflow
import pytest
from dagster import DagsterInstance, RunConfig
from mlflow.tracking import MlflowClient

from nged_substation_forecast.defs.cv_assets import CV_EXPERIMENT_FOLDS_NAME
from nged_substation_forecast.defs.jobs import RegisterExperimentConfig, register_experiment_job

pytestmark = pytest.mark.integration


@pytest.fixture
def mlflow_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point Settings + MLflow at a throwaway file-based store and supply dummy secrets."""
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)


def _run(instance: DagsterInstance, experiment_name: str, run_mode: str) -> None:
    result = register_experiment_job.execute_in_process(
        run_config=RunConfig(
            ops={
                "register_experiment": RegisterExperimentConfig(
                    experiment_name=experiment_name,
                    base_model_config="conf/model/xgboost.yaml",
                    run_mode=run_mode,
                    description="a test experiment",
                )
            }
        ),
        instance=instance,
    )
    assert result.success


def _partition_keys(instance: DagsterInstance) -> list[str]:
    return sorted(instance.get_dynamic_partitions(CV_EXPERIMENT_FOLDS_NAME))


def _parent_run(experiment_id: str):
    runs = MlflowClient().search_runs(
        experiment_ids=[experiment_id], filter_string="tags.cv_role = 'parent'"
    )
    assert len(runs) == 1
    return runs[0]


def test_smoke_test_registers_experiment_and_dev_fold(mlflow_env: None) -> None:
    instance = DagsterInstance.ephemeral()
    _run(instance, "exp_smoke", run_mode="smoke_test")

    experiment = mlflow.get_experiment_by_name("exp_smoke")
    assert experiment is not None
    assert experiment.tags["description"] == "a test experiment"
    assert "config" in experiment.tags
    # Class identity is stamped so assets can reconstruct the exact forecaster + config subclass.
    assert experiment.tags["forecaster_target"].endswith("XGBoostForecaster")
    assert experiment.tags["config_target"].endswith("XGBoostConfig")

    parent = _parent_run(experiment.experiment_id)
    assert parent.data.tags["model_family"] == "xgboost"
    assert parent.data.tags["weather_source"] == "ecmwf_control"
    # Resolved config is logged as flattened params (e.g. an XGBoost hyperparameter).
    assert "n_estimators" in parent.data.params

    # smoke_test selects the non-leaderboard dev folds; conf/cv/default.yaml holds one (smoke_test).
    assert _partition_keys(instance) == ["exp_smoke__smoke_test"]


def test_full_cv_registers_the_leaderboard_folds(mlflow_env: None) -> None:
    instance = DagsterInstance.ephemeral()
    _run(instance, "exp_full", run_mode="full_cv")

    # full_cv selects the leaderboard folds; conf/cv/default.yaml currently holds a single one. The
    # flag-based smoke-test-vs-full-CV distinction is unit-tested in tests/test_jobs.py.
    assert _partition_keys(instance) == ["exp_full__mid_2025_to_mid_2026"]


def test_re_registration_is_idempotent(mlflow_env: None) -> None:
    instance = DagsterInstance.ephemeral()
    _run(instance, "exp_idem", run_mode="full_cv")
    _run(instance, "exp_idem", run_mode="full_cv")

    # One experiment, one parent run, no duplicate partition keys.
    experiment = mlflow.get_experiment_by_name("exp_idem")
    assert experiment is not None
    _parent_run(experiment.experiment_id)  # asserts exactly one parent run
    assert _partition_keys(instance) == ["exp_idem__mid_2025_to_mid_2026"]
