"""Tests for ``BaseForecaster``'s MLflow persistence, against file-based MLflow.

Uses a tiny fake ``BaseForecaster`` (rather than a concrete model) so the tests stay focused on
the shared behaviour — artifact upload/download — and free of any model-library dependency.
"""

from pathlib import Path
from typing import Self

import mlflow
import patito as pt
import pytest
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig

pytestmark = pytest.mark.integration


class _FakeForecaster(BaseForecaster):
    """Minimal forecaster whose entire trained state is a single string payload on disk."""

    MODEL_NAME = "fake"
    MODEL_VERSION = 1

    def __init__(self, model_params: BaseForecasterConfig, payload: str = "") -> None:
        super().__init__(model_params)
        self.payload = payload

    @property
    def trained_time_series_ids(self) -> list[int]:
        return []

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "payload.txt").write_text(self.payload)

    @classmethod
    def load(cls, path: Path) -> Self:
        instance = cls(BaseForecasterConfig(selected_features=set()))
        instance.payload = (path / "payload.txt").read_text()
        return instance

    def train(
        self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]
    ) -> None:  # pragma: no cover - unused
        raise NotImplementedError

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        raise NotImplementedError  # pragma: no cover - unused


def test_trained_time_series_ids_is_abstract() -> None:
    """A subclass that omits ``trained_time_series_ids`` cannot be instantiated."""

    class _MissingPopulation(BaseForecaster):
        MODEL_NAME = "missing"
        MODEL_VERSION = 1

        def save(self, path: Path) -> None: ...

        @classmethod
        def load(cls, path: Path) -> Self:
            raise NotImplementedError

        def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None: ...

        def predict(
            self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
        ) -> pt.DataFrame[PowerForecast]:
            raise NotImplementedError

    with pytest.raises(TypeError):
        _MissingPopulation(BaseForecasterConfig(selected_features=set()))


def _save(run_id: str, payload: str) -> None:
    """Save a ``_FakeForecaster`` carrying ``payload`` into an existing MLflow run."""
    forecaster = _FakeForecaster(BaseForecasterConfig(selected_features=set()), payload=payload)
    forecaster.save_to_mlflow(run_id)


@pytest.fixture
def saved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A file-based MLflow run id with a saved _FakeForecaster (payload 'hello-model')."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    experiment_id = mlflow.create_experiment("base_forecaster_test")
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
    _save(run_id, "hello-model")
    return run_id


def test_save_load_round_trip(saved_run: str) -> None:
    loaded = _FakeForecaster.load_from_mlflow(saved_run)
    assert loaded.payload == "hello-model"


def test_re_saving_to_the_same_run_is_reflected_on_the_next_load(saved_run: str) -> None:
    """Re-saving into a reused run must be what the next load returns (issue #197).

    CV fold runs are reused across re-materialisations, so the same ``run_id`` can hold a
    different model after re-training. ``load_from_mlflow`` has no local cache (issue #469 removed
    it — see ``load_from_mlflow``'s docstring), so this is really a round-trip test, but it is
    worth keeping explicit: a future re-introduction of caching must not silently reintroduce the
    staleness bug this guards against.
    """
    assert _FakeForecaster.load_from_mlflow(saved_run).payload == "hello-model"

    # Re-train into the *same* run, exactly as a re-materialised fold does.
    _save(saved_run, "retrained-model")

    assert _FakeForecaster.load_from_mlflow(saved_run).payload == "retrained-model"
