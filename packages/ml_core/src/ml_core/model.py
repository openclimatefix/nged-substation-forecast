"""Base classes for ML model inference."""

from abc import ABC, abstractmethod
from typing import Any

import patito as pt
from contracts.data_schemas import PowerForecast


class BaseForecaster(ABC):
    """Abstract base class for all ML model forecasters.

    A Forecaster handles the full lifecycle of an ML model: training and
    production inference. It handles eager `DataFrames` and is designed to
    be used within Dagster assets or standalone scripts.

    Subclasses should override the `train` and `predict` methods with explicit,
    strictly-typed keyword arguments for the specific data they require.
    """

    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """Train the model.

        Args:
            config: Model-specific configuration.
            **kwargs: Model-specific data inputs (e.g., weather, power flows).

        Returns:
            The trained native model object (e.g., XGBRegressor).
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts.

        Args:
            **kwargs: Model-specific data inputs (e.g., weather, power flows).

        Returns:
            A Patito DataFrame containing the model's predictions.
        """
        pass
