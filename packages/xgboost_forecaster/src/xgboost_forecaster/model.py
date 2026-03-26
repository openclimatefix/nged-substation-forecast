"""XGBoost implementation of the Forecaster interface."""

import logging
from datetime import datetime, timezone
from typing import cast

import patito as pt
import polars as pl
from contracts.data_schemas import (
    InferenceParams,
    PowerForecast,
    ProcessedNwp,
    SubstationFlows,
    SubstationMetadata,
)
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

    def _prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply feature engineering and extract the feature matrix.

        Args:
            df: The joined input data.

        Returns:
            A Polars DataFrame containing only the feature columns.
        """
        # 1. Apply shared feature engineering
        df = add_cyclical_temporal_features(df, time_col="valid_time")

        # 2. Define columns that are NOT features
        exclude_cols = {
            "MW",
            "MVA",
            "MVAr",
            "ingested_at",
            "valid_time",
            "substation_number",
            "h3_index",
            "ensemble_member",
        }

        # 3. Return the feature matrix as a Polars DataFrame
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return df.select(feature_cols)

    def train(  # type: ignore[override]
        self,
        config: dict,
        nwp: pt.LazyFrame[ProcessedNwp],
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
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

        # Prepare features and target
        X = self._prepare_features(joined_df)
        y = joined_df.select("MW").to_series()

        self.model = XGBRegressor(**config.get("hyperparameters", {}))
        self.model.fit(X, y)

        return self.model

    def predict(  # type: ignore[override]
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        nwp: pt.DataFrame[ProcessedNwp] | None = None,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            substation_metadata: The substation metadata containing h3 mapping.
            inference_params: Parameters for inference.
            nwp: The validated weather forecast data.
            **kwargs: Additional arguments (unused).

        Returns:
            A Patito DataFrame containing the predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")

        if nwp is None:
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        # Expand NWP to all substations
        metadata_df = substation_metadata.select(["substation_number", "h3_res_5"])
        df = nwp.join(
            metadata_df.rename({"h3_res_5": "h3_index"}),
            on="h3_index",
            how="inner",
        )

        # Prepare features (must match training features)
        X = self._prepare_features(df)

        preds = self.model.predict(X)

        # Return predictions with correct schema
        now = datetime.now(timezone.utc)
        model_name = inference_params.power_fcst_model or "xgboost_global"

        res = df.select(["valid_time", "substation_number", "ensemble_member"]).with_columns(
            MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32),
            power_fcst_model_name=pl.lit(model_name).cast(pl.Categorical),
            power_fcst_init_time=pl.lit(now).cast(pl.Datetime("us", "UTC")),
            nwp_init_time=pl.lit(inference_params.nwp_init_time).cast(pl.Datetime("us", "UTC")),
            power_fcst_init_year_month=pl.lit(now.strftime("%Y-%m")).cast(pl.String),
        )

        # Handle potential nulls in ensemble_member (required by schema)
        if "ensemble_member" not in res.columns:
            res = res.with_columns(ensemble_member=pl.lit(0).cast(pl.UInt8))
        else:
            res = res.with_columns(pl.col("ensemble_member").fill_null(0).cast(pl.UInt8))

        return pt.DataFrame[PowerForecast](res)
