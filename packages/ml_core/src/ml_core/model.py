"""Base classes for ML model inference."""

from abc import ABC, abstractmethod

import patito as pt
from contracts.data_schemas import PowerForecast


class BaseForecaster(ABC):
    """Abstract base class for all ML model forecasters.

    A Forecaster is a lightweight, deployable mathematical artifact that handles
    eager `DataFrames`. It is designed to be serialized by MLflow and used
    for production inference.

    Subclasses should override the `predict` method with explicit, strictly-typed
    keyword arguments for the specific data they require.
    """

    @abstractmethod
    def predict(self, *args, **kwargs) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts.

        Args:
            **kwargs: Model-specific data inputs (e.g., weather, power flows).

        Returns:
            A Patito DataFrame containing the model's predictions.
        """
        pass
