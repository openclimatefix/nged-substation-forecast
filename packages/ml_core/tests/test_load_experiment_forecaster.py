"""Unit test for ``load_experiment_forecaster`` (file-based MLflow, no Dagster)."""

from pathlib import Path

import mlflow
import pytest
from ml_core.mlflow_runs import get_or_create_experiment, load_experiment_forecaster
from mlflow.tracking import MlflowClient
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster


@pytest.fixture
def mlflow_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")


def test_round_trips_class_and_config_through_tags(mlflow_tracking: None) -> None:
    """The forecaster class + config rebuild from the ``forecaster_target`` and ``config`` tags.

    The config class is not named by a tag: it is reached through the forecaster's
    ``CONFIG_CLASS``, so no third tag is stamped here.
    """
    config = XGBoostConfig(
        selected_features={"temperature_2m", "power_lag_24h"},
        experiment_name="exp",
        n_estimators=123,
        max_depth=4,
    )
    experiment_id = get_or_create_experiment("exp")
    client = MlflowClient()
    client.set_experiment_tag(experiment_id, "config", config.model_dump_json())
    client.set_experiment_tag(
        experiment_id, "forecaster_target", "xgboost_forecaster.forecaster.XGBoostForecaster"
    )

    forecaster_cls, loaded_config = load_experiment_forecaster("exp")

    assert forecaster_cls is XGBoostForecaster
    assert isinstance(loaded_config, XGBoostConfig)
    assert loaded_config == config


def test_raises_for_unknown_experiment(mlflow_tracking: None) -> None:
    with pytest.raises(ValueError, match="No MLflow experiment named 'missing'"):
        load_experiment_forecaster("missing")
