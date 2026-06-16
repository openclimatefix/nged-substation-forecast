from abc import ABC, abstractmethod
from pathlib import Path

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from pydantic import BaseModel


class BaseForecaster(ABC):
    """Defines the universal interface for all energy forecasting ML models.

    Every energy forecasting model, ranging from a simple seasonal persistence model, up to a
    sophisticated neural net, will subclass this abstract base class. This will allow us to re-use
    as much code as possible, and to minimise the amount of code that must be written for each new
    ML model.
    """

    def __init__(self, selected_features: set[str], model_params: BaseModel):
        self.selected_features = selected_features
        self.model_params = model_params

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model params. Concrete classes implement their native save logic here."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model params. Concrete classes implement their native load logic here."""
        pass

    @abstractmethod
    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        """Create the AllFeatures LazyFrame using features.engineer_features."""
        pass

    @abstractmethod
    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        pass
