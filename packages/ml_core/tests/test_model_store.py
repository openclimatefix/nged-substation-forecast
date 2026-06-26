"""Round-trip and cache tests for the model-store helpers, against file-based MLflow.

Uses a tiny fake ``BaseForecaster`` (rather than a concrete model) so the test stays focused on
the store's behaviour — artifact upload/download and the local cache — and free of any
model-library dependency.
"""

from pathlib import Path
from typing import Self

import mlflow
import patito as pt
import pytest
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core._model_store import load_model, save_model
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig

pytestmark = pytest.mark.integration


class _FakeForecaster(BaseForecaster):
    """Minimal forecaster whose entire trained state is a single string payload on disk."""

    MODEL_NAME = "fake"
    MODEL_VERSION = 1

    def __init__(self, model_params: BaseForecasterConfig, payload: str = "") -> None:
        super().__init__(model_params)
        self.payload = payload

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text(self.payload)

    @classmethod
    def load(cls, path: Path) -> Self:
        instance = cls(BaseForecasterConfig(selected_features=set()))
        instance.payload = (path / "payload.txt").read_text()
        return instance

    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        raise NotImplementedError  # pragma: no cover - unused


@pytest.fixture
def saved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A file-based MLflow run id with a saved _FakeForecaster (payload 'hello-model')."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    experiment_id = mlflow.create_experiment("model_store_test")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
    forecaster = _FakeForecaster(
        BaseForecasterConfig(selected_features=set()), payload="hello-model"
    )
    save_model(forecaster, run_id)
    return run_id


def test_save_load_round_trip(saved_run: str, tmp_path: Path) -> None:
    loaded = load_model(saved_run, _FakeForecaster, cache_base_path=tmp_path / "cache")
    assert isinstance(loaded, _FakeForecaster)
    assert loaded.payload == "hello-model"


def test_cache_hit_does_not_contact_mlflow(
    saved_run: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"

    # First load is a cache miss → downloads from MLflow into the cache.
    first = load_model(saved_run, _FakeForecaster, cache_base_path=cache)
    assert isinstance(first, _FakeForecaster)
    assert first.payload == "hello-model"
    assert (cache / saved_run / "model" / "payload.txt").exists()

    # Simulate an MLflow outage: any download attempt now blows up. A cache hit must not call it.
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("MLflow is unreachable")

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", _boom)

    second = load_model(saved_run, _FakeForecaster, cache_base_path=cache)
    assert isinstance(second, _FakeForecaster)
    assert second.payload == "hello-model"
