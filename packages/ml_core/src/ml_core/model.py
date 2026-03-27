"""Base classes for ML model inference."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import patito as pt
import polars as pl
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    SubstationFlows,
    SubstationMetadata,
)
from contracts.hydra_schemas import ModelConfig

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
    def train(self, config: ModelConfig, **kwargs) -> Any:
        """Train the model.

        Args:
            config: Model configuration object.
            **kwargs: Model-specific data inputs (e.g., weather, power flows).

        Returns:
            The trained native model object (e.g., XGBRegressor).
        """
        pass

    @abstractmethod
    def predict(
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts.

        Args:
            substation_metadata: The substation metadata.
            inference_params: Parameters for inference.
            **kwargs: Model-specific data inputs (e.g., nwps, substation_power_flows).
                These should generally be passed as LazyFrames where possible.

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


class LocalForecasters(BaseForecaster):
    """Trains and manages a separate BaseForecaster per substation.

    This class implements the BaseForecaster interface by delegating to
    individual forecaster instances for each substation.
    """

    def __init__(self, forecaster_cls: type[BaseForecaster], **forecaster_kwargs):
        """Initialize the local forecasters.

        Args:
            forecaster_cls: The class of the underlying forecaster to use.
            **forecaster_kwargs: Keyword arguments to pass to the forecaster constructor.
        """
        self.forecaster_cls = forecaster_cls
        self.forecaster_kwargs = forecaster_kwargs
        self.models: dict[int, BaseForecaster] = {}

    def train(  # type: ignore
        self,
        config: ModelConfig,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
        **kwargs,
    ) -> "LocalForecasters":
        """Train a separate model for each substation.

        Args:
            config: Model configuration object.
            substation_power_flows: The historical power flow data.
            substation_metadata: The substation metadata.
            **kwargs: Additional arguments passed to the underlying train methods.

        Returns:
            The trained local forecasters instance.
        """
        substations = substation_metadata["substation_number"].unique().to_list()
        log.info(f"Training local models for {len(substations)} substations...")

        # TODO: Implement parallelization of this sequential loop
        for sub_num in substations:
            log.debug(f"Training model for substation {sub_num}")
            sub_meta = substation_metadata.filter(pl.col("substation_number") == sub_num)
            sub_flows = substation_power_flows.filter(pl.col("substation_number") == sub_num)

            # Instantiate and train
            model = self.forecaster_cls(**self.forecaster_kwargs)
            model.train(
                config=config,
                substation_power_flows=sub_flows,
                substation_metadata=sub_meta,
                **kwargs,
            )
            self.models[sub_num] = model

        return self

    def predict(
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        substation_power_flows: pt.LazyFrame[SubstationFlows] | None = None,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts by routing to local models.

        Args:
            substation_metadata: The substation metadata.
            inference_params: Parameters for inference.
            substation_power_flows: The historical power flow data (optional, for lags).
            **kwargs: Additional arguments passed to the underlying predict methods (e.g., nwps).

        Returns:
            A concatenated Patito DataFrame of PowerForecasts.
        """
        all_preds = []
        substations = substation_metadata["substation_number"].unique().to_list()

        for sub_num in substations:
            if sub_num not in self.models:
                log.warning(f"No model found for substation {sub_num}. Skipping.")
                continue

            sub_meta = substation_metadata.filter(pl.col("substation_number") == sub_num)

            # Filter optional inputs if they exist
            sub_flows = None
            if substation_power_flows is not None:
                sub_flows = substation_power_flows.filter(pl.col("substation_number") == sub_num)

            preds = self.models[sub_num].predict(
                substation_metadata=sub_meta,
                inference_params=inference_params,
                substation_power_flows=sub_flows,
                **kwargs,
            )
            all_preds.append(preds)

        if not all_preds:
            raise ValueError("No predictions were generated by any local model.")

        return pt.DataFrame[PowerForecast](pl.concat(all_preds))

    def log_model(self, model_name: str) -> None:
        """Log all local models to MLflow.

        Args:
            model_name: The base name to register the models under.
        """
        # TODO: Implement logging for local models
        pass
