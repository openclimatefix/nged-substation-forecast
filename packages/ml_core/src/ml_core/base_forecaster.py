import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Final, Self

import mlflow
import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from pydantic import BaseModel

from ml_core.feature_engineer import FeatureEngineer, TabularNwpFeatureEngineer

_MLFLOW_ARTIFACT_PATH: Final[str] = "model"
"""Sub-path under an MLflow run's artifact root where the model directory is stored."""


class BaseForecasterConfig(BaseModel):
    """Universal configuration for all forecasting models.

    Subclasses add model-specific hyperparameters. Having a shared base ensures that every
    forecaster carries its own feature list and optional MLflow experiment id in one
    serialisable object, simplifying save/load and Hydra config wiring.

    The tag fields (weather_source, training_strategy) are stamped onto MLflow runs for
    leaderboard grouping. They live here so that
    ``hydra.utils.instantiate(model_cfg.model_params)`` validates them at load time — they
    are present in every ``conf/model/*.yaml`` file under ``model_params``.

    ``experiment_name`` is the per-experiment key (set to the MLflow experiment name at
    registration and stored in the saved config); it is stamped onto every ``PowerForecast``
    row, distinct from the model-family ``MODEL_NAME``. ``random_seed`` is threaded into each
    model's training so that re-training a fold reproduces the same model — keeping retries
    and the leaderboard stable.

    Model identity (name and version) lives on the ``BaseForecaster`` class itself as
    ``MODEL_NAME`` and ``MODEL_VERSION`` — those are properties of the implementation, not
    the experiment config.
    """

    selected_features: set[str]
    ml_flow_experiment_id: int | None = None
    experiment_name: str = ""
    weather_source: str = ""
    training_strategy: str = ""
    random_seed: int = 0


class BaseForecaster(ABC):
    """Defines the universal interface for all energy forecasting ML models.

    Every forecasting model subclasses this abstract base to allow shared Dagster assets and
    evaluation code to remain completely agnostic to the underlying model implementation.

    Subclasses must define ``MODEL_NAME`` and ``MODEL_VERSION`` as class-level constants.
    These are stamped onto every ``PowerForecast`` row at predict time and used as the MLflow
    experiment name. Bumping ``MODEL_VERSION`` requires a code change (intentional), not a
    config edit.

    Lazy evaluation contract: `train` and `predict` both accept a `pt.LazyFrame[AllFeatures]`.
    Subclasses should call `.collect()` exactly once, as late as possible — typically right
    before handing data to the underlying model library. Callers must not collect before passing
    data in; doing so wastes memory and prevents Polars from optimising the full query plan.

    Persistence has two layers. Subclasses implement ``save``/``load`` for their own on-disk
    format and need know nothing about MLflow. The concrete ``save_to_mlflow``/``load_from_mlflow``
    methods, shared by all subclasses, wrap that disk format with MLflow's artifact store and a
    local-disk cache, so the same trained model can be shared across machines and served offline.
    """

    MODEL_NAME: ClassVar[str]
    MODEL_VERSION: ClassVar[int]

    feature_engineer: ClassVar[FeatureEngineer] = TabularNwpFeatureEngineer()
    """The feature pipeline this forecaster's data is engineered through.

    Associated by composition (the forecaster *references* a feature engineer rather than
    *implementing* feature engineering), so a forecaster can swap the whole pipeline by overriding
    this with a different ``FeatureEngineer``. The default produces the tabular ``AllFeatures``
    frame that ``train``/``predict`` consume.
    """

    def __init__(self, model_params: BaseForecasterConfig) -> None:
        self.model_params = model_params

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the trained model state to a directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Reconstruct a trained instance from a previously saved directory."""
        pass

    def save_to_mlflow(self, run_id: str) -> None:
        """Upload this trained model's artifacts to the given MLflow run.

        Writes the model to a temporary directory via ``save`` (the subclass's own format), then
        uploads that directory to the run's artifact store under ``model/``. The caller is
        responsible for setting the tracking URI (``mlflow.set_tracking_uri``) beforehand.

        Args:
            run_id: The MLflow run to attach the artifacts to.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save(Path(tmp_dir))
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifacts(tmp_dir, artifact_path=_MLFLOW_ARTIFACT_PATH)

    @classmethod
    def load_from_mlflow(cls, run_id: str, cache_base_path: Path) -> Self:
        """Load a trained model for ``run_id``, serving from the local cache when possible.

        On a cache hit (``{cache_base_path}/{run_id}/model`` already exists) the model is loaded
        straight from disk and MLflow is never contacted — this is what lets the live service keep
        serving during an MLflow outage. On a cache miss the artifacts are downloaded from the run
        into the cache, then loaded. The cache key is the immutable run ID, so a cached model never
        goes stale. The caller sets the tracking URI (``mlflow.set_tracking_uri``) beforehand.

        Args:
            run_id: The MLflow run the model was saved under.
            cache_base_path: Root of the local cache; the model lives at
                ``{cache_base_path}/{run_id}``.

        Returns:
            The reconstructed, trained forecaster.
        """
        run_cache_dir = cache_base_path / run_id
        model_dir = run_cache_dir / _MLFLOW_ARTIFACT_PATH
        if not model_dir.exists():
            run_cache_dir.mkdir(parents=True, exist_ok=True)
            mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=_MLFLOW_ARTIFACT_PATH,
                dst_path=str(run_cache_dir),
            )
        return cls.load(model_dir)

    @abstractmethod
    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        """Fit the model on the given AllFeatures data."""
        pass

    @abstractmethod
    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        """Return power forecasts for all rows in the given AllFeatures data."""
        pass
