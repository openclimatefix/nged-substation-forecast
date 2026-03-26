"""Base classes for ML model inference."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import patito as pt
import polars as pl
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    ProcessedNwp,
    SubstationFlows,
    SubstationMetadata,
)

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
            **kwargs: Model-specific data inputs (e.g., nwp, power flows).

        Returns:
            A Patito DataFrame containing the model's predictions.
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

    def train(
        self,
        config: dict,
        nwp: pt.LazyFrame[ProcessedNwp],
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
        **kwargs,
    ) -> "LocalForecasters":
        """Train a separate model for each substation.

        Args:
            config: Model-specific configuration.
            nwp: The weather forecast data.
            substation_power_flows: The historical power flow data.
            substation_metadata: The substation metadata.
            **kwargs: Additional arguments passed to the underlying train methods.

        Returns:
            The trained local forecasters instance.
        """
        substations = substation_metadata["substation_number"].unique().to_list()
        log.info(f"Training local models for {len(substations)} substations...")

        for sub_num in substations:
            log.debug(f"Training model for substation {sub_num}")
            sub_meta = substation_metadata.filter(pl.col("substation_number") == sub_num)
            sub_flows = substation_power_flows.filter(pl.col("substation_number") == sub_num)

            # Instantiate and train
            model = self.forecaster_cls(**self.forecaster_kwargs)
            model.train(
                config=config,
                nwp=nwp,
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
        nwp: pt.DataFrame[ProcessedNwp] | None = None,
        substation_power_flows: pt.DataFrame[SubstationFlows] | None = None,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Generate power forecasts by routing to local models.

        Args:
            substation_metadata: The substation metadata.
            inference_params: Parameters for inference.
            nwp: The weather forecast data.
            substation_power_flows: The historical power flow data (optional, for lags).
            **kwargs: Additional arguments passed to the underlying predict methods.

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
            sub_nwp = None
            if nwp is not None:
                h3_indices = sub_meta["h3_res_5"].unique().to_list()
                sub_nwp = nwp.filter(pl.col("h3_index").is_in(h3_indices))

            sub_flows = None
            if substation_power_flows is not None:
                sub_flows = substation_power_flows.filter(pl.col("substation_number") == sub_num)

            preds = self.models[sub_num].predict(
                substation_metadata=sub_meta,
                inference_params=inference_params,
                nwp=sub_nwp,
                substation_power_flows=sub_flows,
                **kwargs,
            )
            all_preds.append(preds)

        if not all_preds:
            raise ValueError("No predictions were generated by any local model.")

        return pt.DataFrame[PowerForecast](pl.concat(all_preds))
