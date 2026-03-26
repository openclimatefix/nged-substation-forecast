"""XGBoost implementation of the Forecaster interface."""

import logging
from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast, ProcessedNwp, SubstationFlows
from xgboost import XGBRegressor

from ml_core.features import add_cyclical_temporal_features
from ml_core.model import BaseForecaster

log = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost implementation of the Forecaster interface.

    This class handles the full lifecycle of an XGBoost model: training
    and production inference.
    """

    def __init__(self, model: XGBRegressor | None = None):
        """Initialize the forecaster.

        Args:
            model: An optional pre-trained XGBoost model.
        """
        self.model = model

    def train(  # type: ignore[override]
        self,
        config: dict,
        nwp: pt.LazyFrame[ProcessedNwp],
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pl.DataFrame,
        **kwargs,
    ) -> XGBRegressor:
        """Train the XGBoost model.

        Args:
            config: The model-specific Hydra configuration.
            nwp: The weather forecast data.
            substation_power_flows: The historical power flow data.
            substation_metadata: The substation metadata containing h3 mapping.

        Returns:
            The trained XGBRegressor model.
        """
        log.info("Starting XGBoost training...")

        metadata_lf = substation_metadata.select(["substation_number", "h3_res_5"]).lazy()

        # Join power flows and weather data using substation metadata.
        flows_with_h3 = substation_power_flows.rename({"timestamp": "valid_time"}).join(
            metadata_lf.rename({"h3_res_5": "h3_index"}),
            on="substation_number",
        )

        joined_df = cast(
            pl.DataFrame,
            flows_with_h3.join(
                nwp,
                on=["valid_time", "h3_index"],
            ).collect(),
        )

        # Use shared feature engineering from ml_core
        joined_df = add_cyclical_temporal_features(joined_df, time_col="valid_time")

        # Prepare features and target
        X = joined_df.select(
            pl.all().exclude(
                [
                    "MW",
                    "MVA",
                    "MVAr",
                    "ingested_at",
                    "valid_time",
                    "substation_number",
                    "h3_index",
                    "ensemble_member",
                ]
            )
        ).to_pandas()
        y = joined_df.select("MW").to_pandas()

        self.model = XGBRegressor(**config.get("hyperparameters", {}))
        self.model.fit(X, y)

        return self.model

    def predict(
        self,
        nwp: pt.DataFrame[ProcessedNwp],
        substation_metadata: pl.DataFrame,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            nwp: The validated weather forecast data.
            substation_metadata: The substation metadata containing h3 mapping.
            **kwargs: Additional arguments (unused).

        Returns:
            A Patito DataFrame containing the predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")

        # Expand NWP to all substations
        metadata_df = substation_metadata.select(["substation_number", "h3_res_5"])
        df = nwp.join(
            metadata_df.rename({"h3_res_5": "h3_index"}),
            on="h3_index",
            how="inner",
        )

        # Use shared feature engineering from ml_core
        df = add_cyclical_temporal_features(df, time_col="valid_time")

        # Prepare features (must match training features)
        X = df.select(
            pl.all().exclude(["valid_time", "substation_number", "h3_index", "ensemble_member"])
        ).to_pandas()

        preds = self.model.predict(X)

        # Return predictions with correct schema
        return pt.DataFrame[PowerForecast](
            df.select(["valid_time", "substation_number", "ensemble_member"]).with_columns(
                MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32)
            )
        )
