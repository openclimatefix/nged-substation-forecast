"""Integration test for the ``production_model`` asset (real file-based MLflow + Dagster).

Trains a tiny real ``XGBoostForecaster`` directly (skipping the full CV pipeline — that's
exercised elsewhere) and saves it to a genuine MLflow run, then materialises
``production_model`` to promote it to local disk: the directory should be populated with
``meta.json`` + ``promotion.json``, output metadata should be correct, and re-promoting with a
second run id should replace the directory rather than merging into it.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import AllFeatures
from dagster import DagsterInstance, RunConfig, TableMetadataValue, materialize
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.production_assets import (
    ProductionModelConfig,
    production_model,
    promotable_model_runs,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
    monkeypatch.setenv("PRODUCTION_MODEL_PATH", str(tmp_path / "production_model"))
    mlflow.set_tracking_uri(tracking_uri)
    return {"production_model_path": str(tmp_path / "production_model")}


def _save_trained_model_to_mlflow(experiment_name: str, n_estimators: int) -> str:
    """Train a tiny real ``XGBoostForecaster`` on synthetic data and save it to a new MLflow run.

    Bypasses the full CV/feature-engineering pipeline (tested elsewhere) — the point here is a
    genuine trained booster + a real ``meta.json`` with ``model_class``, exercised through the
    same ``save_to_mlflow`` mechanism ``trained_cv_model`` uses.
    """
    times = [datetime(2025, 1, 1, hour, tzinfo=timezone.utc) for hour in (0, 1, 2)]
    train_df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "valid_time": times,
            "time_series_type": ["Primary"] * 3,
            "power_fcst_init_time": times,
            "power": [10.0, 12.0, 11.0],
            "temperature_2m": [5.0, 6.0, 7.0],
        }
    )
    train_data = pt.LazyFrame.from_existing(train_df.lazy()).set_model(AllFeatures)
    config = XGBoostConfig(
        selected_features={"temperature_2m"},
        experiment_name=experiment_name,
        n_estimators=n_estimators,
    )
    forecaster = XGBoostForecaster(config)
    forecaster.train(train_data, time_series_ids=[1])

    experiment_id = mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        # Tag as a fold run, matching what trained_cv_model does via get_or_create_fold_run — so
        # list_promotable_runs (and the promotable_model_runs asset) picks this run up.
        mlflow.set_tags({"cv_role": "fold", "fold_id": "test_fold"})
    forecaster.save_to_mlflow(run_id)
    return run_id


def test_production_model_promotes_and_populates_directory(env: dict[str, str]) -> None:
    run_id = _save_trained_model_to_mlflow("promo_test", n_estimators=5)

    instance = DagsterInstance.ephemeral()
    result = materialize(
        [production_model],
        run_config=RunConfig(ops={"production_model": ProductionModelConfig(mlflow_run_id=run_id)}),
        instance=instance,
    )
    assert result.success

    model_dir = Path(env["production_model_path"])
    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta["model_class"] == "xgboost_forecaster.forecaster.XGBoostForecaster"
    assert meta["trained_time_series_ids"] == [1]

    promotion = json.loads((model_dir / "promotion.json").read_text())
    assert promotion["mlflow_run_id"] == run_id
    assert "promoted_at" in promotion

    [materialization] = result.asset_materializations_for_node("production_model")
    metadata = materialization.metadata
    assert metadata["mlflow_run_id"].value == run_id
    assert metadata["model_class"].value == "xgboost_forecaster.forecaster.XGBoostForecaster"
    assert metadata["experiment_name"].value == "promo_test"
    assert metadata["n_trained_time_series"].value == 1


def test_re_promotion_replaces_the_model(env: dict[str, str]) -> None:
    first_run_id = _save_trained_model_to_mlflow("promo_test_v1", n_estimators=5)
    second_run_id = _save_trained_model_to_mlflow("promo_test_v2", n_estimators=7)

    model_dir = Path(env["production_model_path"])

    instance = DagsterInstance.ephemeral()
    assert materialize(
        [production_model],
        run_config=RunConfig(
            ops={"production_model": ProductionModelConfig(mlflow_run_id=first_run_id)}
        ),
        instance=instance,
    ).success
    first_meta = json.loads((model_dir / "meta.json").read_text())
    assert first_meta["model_params"]["experiment_name"] == "promo_test_v1"

    assert materialize(
        [production_model],
        run_config=RunConfig(
            ops={"production_model": ProductionModelConfig(mlflow_run_id=second_run_id)}
        ),
        instance=instance,
    ).success
    second_meta = json.loads((model_dir / "meta.json").read_text())
    assert second_meta["model_params"]["experiment_name"] == "promo_test_v2"

    promotion = json.loads((model_dir / "promotion.json").read_text())
    assert promotion["mlflow_run_id"] == second_run_id


def test_promotable_model_runs_lists_fold_run_candidates(env: dict[str, str]) -> None:
    run_id = _save_trained_model_to_mlflow("candidates_test", n_estimators=5)

    result = materialize([promotable_model_runs], instance=DagsterInstance.ephemeral())
    assert result.success

    [materialization] = result.asset_materializations_for_node("promotable_model_runs")
    metadata = materialization.metadata
    assert metadata["n_candidates"].value == 1
    candidates = metadata["candidates"]
    assert isinstance(candidates, TableMetadataValue)
    [row] = candidates.records
    assert row.data["run_id"] == run_id
    assert row.data["experiment_name"] == "candidates_test"
