"""Unit test for ``load_experiment_forecaster`` (file-based MLflow, no Dagster)."""

from pathlib import Path

import mlflow
import pytest
from _dummy_forecaster import DummyForecaster, DummyForecasterConfig
from baseline_forecasters.manual_heuristic import ManualHeuristicForecaster
from contracts.config_schemas import class_target
from ml_core.base_forecaster import BaseForecasterConfig
from ml_core.mlflow_runs import get_or_create_experiment, load_experiment_forecaster
from mlflow.tracking import MlflowClient


@pytest.fixture
def mlflow_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")


def test_round_trips_class_and_config_through_tags(mlflow_tracking: None) -> None:
    """The forecaster class + config rebuild from the ``forecaster_target`` and ``config`` tags.

    The config class is not named by a tag: it is reached through the forecaster's
    ``CONFIG_CLASS``, so no third tag is stamped here.
    """
    config = BaseForecasterConfig(
        selected_features={"power_lag_168h", "power_lag_336h"},
        experiment_name="exp",
        random_seed=123,
    )
    experiment_id = get_or_create_experiment("exp")
    client = MlflowClient()
    client.set_experiment_tag(experiment_id, "config", config.model_dump_json())
    client.set_experiment_tag(
        experiment_id,
        "forecaster_target",
        "baseline_forecasters.manual_heuristic.ManualHeuristicForecaster",
    )

    forecaster_cls, loaded_config = load_experiment_forecaster("exp")

    assert forecaster_cls is ManualHeuristicForecaster
    assert isinstance(loaded_config, BaseForecasterConfig)
    assert loaded_config == config


def test_raises_for_unknown_experiment(mlflow_tracking: None) -> None:
    with pytest.raises(ValueError, match="No MLflow experiment named 'missing'"):
        load_experiment_forecaster("missing")


def test_loads_the_config_subclass_that_the_forecaster_names(mlflow_tracking: None) -> None:
    """``CONFIG_CLASS`` picks the config class, so the subclass's extra field is kept."""
    config = DummyForecasterConfig(selected_features=set(), experiment_name="dummy", dummy_extra=9)
    experiment_id = get_or_create_experiment("dummy")
    client = MlflowClient()
    client.set_experiment_tag(experiment_id, "config", config.model_dump_json())
    client.set_experiment_tag(experiment_id, "forecaster_target", class_target(DummyForecaster))

    forecaster_cls, loaded_config = load_experiment_forecaster("dummy")

    assert forecaster_cls is DummyForecaster
    assert type(loaded_config) is DummyForecasterConfig
    assert loaded_config.dummy_extra == 9
