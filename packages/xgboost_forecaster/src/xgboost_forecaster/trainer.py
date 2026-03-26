"""XGBoost trainer implementation."""

import logging
from typing import Type, cast

import patito as pt
import polars as pl
from contracts.data_schemas import ProcessedNwp, SubstationFlows
from xgboost import XGBRegressor

from ml_core.trainer import BaseDataRequirements, BaseTrainer
from xgboost_forecaster.model import XGBoostInferenceModel

log = logging.getLogger(__name__)


class XGBoostTrainData(BaseDataRequirements):
    """Training data requirements for XGBoost."""

    weather_ecmwf_ens_0_25: pt.LazyFrame[ProcessedNwp]
    substation_power_flows: pt.LazyFrame[SubstationFlows]


class XGBoostTrainer(BaseTrainer[XGBoostTrainData]):
    """Trainer for the XGBoost model."""

    requirements_class: Type[XGBoostTrainData] = XGBoostTrainData

    def train(self, data: XGBoostTrainData, config: dict) -> XGBoostInferenceModel:
        """Train the XGBoost model.

        Args:
            data: The validated training data.
            config: The Hydra configuration.

        Returns:
            The trained XGBoost model wrapper.
        """
        log.info("Starting XGBoost training...")

        # Join SCADA and weather data.
        # We join the historical power flows with the weather forecasts
        # to create the training dataset.
        joined_df = cast(
            pl.DataFrame,
            data.substation_power_flows.rename(
                {"timestamp": "valid_time", "substation_number": "substation_id"}
            )
            .join(
                data.weather_ecmwf_ens_0_25.rename({"h3_index": "substation_id"}),
                on=["valid_time", "substation_id"],
            )
            .collect(),
        )

        # Prepare features and target
        # (This is a simplified version of the original logic)
        X = joined_df.select(
            pl.all().exclude(["MW", "MVA", "MVAr", "ingested_at", "valid_time", "substation_id"])
        ).to_pandas()
        y = joined_df.select("MW").to_pandas()  # Assuming MW is the target

        model = XGBRegressor(**config.get("hyperparameters", {}))
        model.fit(X, y)

        return XGBoostInferenceModel(model)
