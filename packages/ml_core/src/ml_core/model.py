"""Base classes for ML model inference."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import polars as pl
import patito as pt
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    TimeSeriesMetadata,
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
        flows_30m: pl.LazyFrame,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
    ) -> Any:
        """Train the model.

        Args:
            config: Model configuration object.
            flows_30m: Historical power flow data downsampled to 30m.
            time_series_metadata: The time series metadata.
            nwps: A dictionary of weather forecast dataframes.

        Returns:
            The trained native model object (e.g., XGBRegressor).
        """
        pass

    @abstractmethod
    def predict(
        self,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        inference_params: InferenceParams,
        flows_30m: pl.LazyFrame,
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
        collapse_lead_times: bool = False,
    ) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts.

        Args:
            time_series_metadata: The time series metadata.
            inference_params: Parameters for inference.
            flows_30m: Historical power flow data downsampled to 30m (for lags).
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
