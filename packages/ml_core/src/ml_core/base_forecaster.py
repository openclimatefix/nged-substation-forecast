from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from pydantic import BaseModel


class BaseForecasterConfig(BaseModel):
    """Universal configuration for all forecasting models.

    Subclasses add model-specific hyperparameters. Having a shared base ensures that every
    forecaster carries its own identity (name, version, optional MLflow experiment) and its
    feature list in one serialisable object, simplifying save/load and Hydra config wiring.

    The four tag fields (task, model_family, weather_source, training_strategy) are stamped
    onto MLflow runs for leaderboard grouping. They live here so that
    ``hydra.utils.instantiate(model_cfg.model_params)`` validates them at load time — they
    are present in every ``conf/model/*.yaml`` file under ``model_params``.
    """

    selected_features: set[str]
    power_fcst_model_name: str
    power_fcst_model_version: int
    ml_flow_experiment_id: int | None = None
    task: str = ""
    model_family: str = ""
    weather_source: str = ""
    training_strategy: str = ""


class BaseForecaster(ABC):
    """Defines the universal interface for all energy forecasting ML models.

    Every forecasting model subclasses this abstract base to allow shared Dagster assets and
    evaluation code to remain completely agnostic to the underlying model implementation.

    Lazy evaluation contract: `train` and `predict` both accept a `pt.LazyFrame[AllFeatures]`.
    Subclasses should call `.collect()` exactly once, as late as possible — typically right
    before handing data to the underlying model library. Callers must not collect before passing
    data in; doing so wastes memory and prevents Polars from optimising the full query plan.
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

    @abstractmethod
    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        """Fit the model on the given AllFeatures data."""
        pass

    @abstractmethod
    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        """Return power forecasts for all rows in the given AllFeatures data."""
        pass
