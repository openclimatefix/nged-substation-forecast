from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Self

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from pydantic import BaseModel


class BaseForecasterConfig(BaseModel):
    """Universal configuration for all forecasting models.

    Subclasses add model-specific hyperparameters. Having a shared base ensures that every
    forecaster carries its own feature list and optional MLflow experiment id in one
    serialisable object, simplifying save/load and Hydra config wiring.

    The tag fields (weather_source, training_strategy) are stamped onto MLflow runs for
    leaderboard grouping. They live here so that
    ``hydra.utils.instantiate(model_cfg.model_params)`` validates them at load time — they
    are present in every ``conf/model/*.yaml`` file under ``model_params``.

    Model identity (name and version) lives on the ``BaseForecaster`` class itself as
    ``MODEL_NAME`` and ``MODEL_VERSION`` — those are properties of the implementation, not
    the experiment config.
    """

    selected_features: set[str]
    ml_flow_experiment_id: int | None = None
    weather_source: str = ""
    training_strategy: str = ""


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
    """

    MODEL_NAME: ClassVar[str]
    MODEL_VERSION: ClassVar[int]

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

    @abstractmethod
    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        """Fit the model on the given AllFeatures data."""
        pass

    @abstractmethod
    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        """Return power forecasts for all rows in the given AllFeatures data."""
        pass
