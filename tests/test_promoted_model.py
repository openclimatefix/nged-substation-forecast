"""Integration test for the ``promoted_model`` asset (real file-based MLflow + Dagster).

Trains a tiny real ``XGBoostForecaster`` directly (skipping the full CV pipeline — that's
exercised elsewhere) and saves it to a genuine MLflow run, then materialises
``promoted_model`` to promote it to local disk: the directory should be populated with
``meta.json`` + ``promotion.json``, output metadata should be correct, and re-promoting with a
second run id should replace the directory rather than merging into it.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import AllFeatures
from dagster import DagsterInstance, RunConfig, TableMetadataValue, materialize
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.production_assets import (
    PromotedModelConfig,
    promotable_model_runs,
    promoted_model,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("PRODUCTION_MODEL_PATH", str(tmp_path / "production_model"))
    mlflow.set_tracking_uri(tracking_uri)
    return {"production_model_path": str(tmp_path / "production_model")}


def _save_trained_model_to_mlflow(
    experiment_name: str,
    n_estimators: int,
    selected_features: set[str] | None = None,
) -> str:
    """Train a tiny real ``XGBoostForecaster`` on synthetic data and save it to a new MLflow run.

    Bypasses the full CV/feature-engineering pipeline (tested elsewhere) — the point here is a
    genuine trained booster + a real ``meta.json`` with ``model_class``, exercised through the
    same ``save_to_mlflow`` mechanism ``trained_cv_model`` uses.

    Every requested feature gets a synthetic column to train against, so a caller can save a model
    whose feature vocabulary the current code does not recognise.
    """
    features = selected_features or {"temperature_2m"}
    times = [datetime(2025, 1, 1, hour, tzinfo=UTC) for hour in (0, 1, 2)]
    spine = {
        "time_series_id": [1, 1, 1],
        "valid_time": times,
        "time_series_type": ["Primary"] * 3,
        "power_fcst_init_time": times,
        "power": [10.0, 12.0, 11.0],
    }
    # A feature column sharing a spine column's name would win the union and overwrite the spine
    # with floats, quietly changing what every caller of this helper trains on.
    assert not features & spine.keys(), "a requested feature would overwrite a spine column"
    train_df = pl.DataFrame(spine | {name: [5.0, 6.0, 7.0] for name in features})
    train_data = pt.LazyFrame.from_existing(train_df.lazy()).set_model(AllFeatures)
    config = XGBoostConfig(
        selected_features=features,
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


def test_promoted_model_promotes_and_populates_directory(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    run_id = _save_trained_model_to_mlflow("promo_test", n_estimators=5)

    result = materialize(
        [promoted_model],
        run_config=RunConfig(ops={"promoted_model": PromotedModelConfig(mlflow_run_id=run_id)}),
        instance=dagster_instance,
    )
    assert result.success

    model_dir = Path(env["production_model_path"])
    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta["model_class"] == "xgboost_forecaster.forecaster.XGBoostForecaster"
    assert meta["trained_time_series_ids"] == [1]

    promotion = json.loads((model_dir / "promotion.json").read_text())
    assert promotion["mlflow_run_id"] == run_id
    assert "promoted_at" in promotion

    [materialization] = result.asset_materializations_for_node("promoted_model")
    metadata = materialization.metadata
    assert metadata["mlflow_run_id"].value == run_id
    assert metadata["model_class"].value == "xgboost_forecaster.forecaster.XGBoostForecaster"
    assert metadata["experiment_name"].value == "promo_test"
    assert metadata["n_trained_time_series"].value == 1


def test_re_promotion_replaces_the_model(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    first_run_id = _save_trained_model_to_mlflow("promo_test_v1", n_estimators=5)
    second_run_id = _save_trained_model_to_mlflow("promo_test_v2", n_estimators=7)

    model_dir = Path(env["production_model_path"])

    assert materialize(
        [promoted_model],
        run_config=RunConfig(
            ops={"promoted_model": PromotedModelConfig(mlflow_run_id=first_run_id)}
        ),
        instance=dagster_instance,
    ).success
    first_meta = json.loads((model_dir / "meta.json").read_text())
    assert first_meta["model_params"]["experiment_name"] == "promo_test_v1"

    assert materialize(
        [promoted_model],
        run_config=RunConfig(
            ops={"promoted_model": PromotedModelConfig(mlflow_run_id=second_run_id)}
        ),
        instance=dagster_instance,
    ).success
    second_meta = json.loads((model_dir / "meta.json").read_text())
    assert second_meta["model_params"]["experiment_name"] == "promo_test_v2"

    promotion = json.loads((model_dir / "promotion.json").read_text())
    assert promotion["mlflow_run_id"] == second_run_id


def test_promoted_model_refuses_a_model_with_an_unparseable_feature(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    """Promoting a model trained before a feature rename fails, leaving nothing behind on disk.

    ``promotable_model_runs`` lists every fold run ever trained, so an operator picking by eye off
    that table can reach a run whose feature vocabulary this code no longer parses. The refusal
    happens before the directory is replaced, so a first promotion writes nothing at all.
    """
    run_id = _save_trained_model_to_mlflow(
        experiment_name="stale_vocabulary",
        n_estimators=5,
        selected_features={"local_utc_offset"},
    )

    with pytest.raises(ValueError, match="local_utc_offset"):
        materialize(
            [promoted_model],
            run_config=RunConfig(ops={"promoted_model": PromotedModelConfig(mlflow_run_id=run_id)}),
            instance=dagster_instance,
        )

    assert not Path(env["production_model_path"]).exists()


def test_promotable_model_runs_lists_fold_run_candidates(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    run_id = _save_trained_model_to_mlflow("candidates_test", n_estimators=5)

    result = materialize([promotable_model_runs], instance=dagster_instance)
    assert result.success

    [materialization] = result.asset_materializations_for_node("promotable_model_runs")
    metadata = materialization.metadata
    assert metadata["n_candidates"].value == 1
    candidates = metadata["candidates"]
    assert isinstance(candidates, TableMetadataValue)
    [row] = candidates.records
    assert row.data["run_id"] == run_id
    assert row.data["experiment_name"] == "candidates_test"
