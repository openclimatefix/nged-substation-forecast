"""Base classes for ML model inference."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import patito as pt
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    ProcessedNwp,
    SubstationFlows,
    SubstationMetadata,
)
from contracts.hydra_schemas import ModelConfig, NwpModel

log = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all ML model forecasters.

    A Forecaster handles the full lifecycle of an ML model: training and
    production inference. It handles eager `DataFrames` and is designed to
    be used within Dagster assets or standalone scripts.

    Subclasses should override the `train` and `predict` methods with explicit,
    strictly-typed keyword arguments for the specific data they require.
    """

    @abstractmethod
    def train(
        self,
        config: ModelConfig,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
    ) -> Any:
        """Train the model.

        Args:
            config: Model configuration object.
            substation_power_flows: The historical power flow data.
            substation_metadata: The substation metadata.
            nwps: A dictionary of weather forecast dataframes.

        Returns:
            The trained native model object (e.g., XGBRegressor).
        """
        pass

    @abstractmethod
    def predict(
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        collapse_lead_times: bool = False,
    ) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts.

        Args:
            substation_metadata: The substation metadata.
            inference_params: Parameters for inference.
            substation_power_flows: The historical power flow data (for lags).
            nwps: A dictionary of weather forecast dataframes.
            collapse_lead_times: Whether to collapse lead times (used in backtesting).

        Returns:
            A Patito DataFrame containing the model's predictions.
        """
        pass

    @abstractmethod
    def log_model(self, model_name: str) -> None:
        """Log the model to MLflow.

        Args:
            model_name: The name to register the model under.
        """
        pass
