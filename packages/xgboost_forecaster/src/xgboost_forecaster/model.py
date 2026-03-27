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
from contracts.hydra_schemas import ModelConfig, NwpModel
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

    def train(  # type: ignore
        self,
        config: ModelConfig,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        **kwargs,
    ) -> XGBRegressor:
        """Train the XGBoost model.

        Args:
            config: The model configuration object.
            substation_power_flows: The historical power flow data.
            substation_metadata: The substation metadata containing h3 mapping.
            nwps: A dictionary of weather forecast dataframes.
            **kwargs: Additional arguments passed to the underlying train methods.

        Returns:
            The trained XGBRegressor model.
        """
        log.info("Starting XGBoost training...")

        if len(config.features.nwps) > 0 and not nwps:
            raise ValueError("Model config requires NWPs, but none were provided.")

        metadata_lf = substation_metadata.select(["substation_number", "h3_res_5"]).lazy()

        # Join power flows and weather data using substation metadata.
        flows_with_h3 = substation_power_flows.rename({"timestamp": "valid_time"}).join(
            metadata_lf.rename({"h3_res_5": "h3_index"}),
            on="substation_number",
        )

        joined_df_lf = flows_with_h3
        if nwps:
            for nwp_name, nwp_lf in nwps.items():
                # Prefix columns to avoid collisions
                prefix = f"{nwp_name.value}_"
                rename_mapping = {
                    col: f"{prefix}{col}"
                    for col in nwp_lf.collect_schema().names()
                    if col not in ["valid_time", "h3_index", "ensemble_member"]
                }
                prefixed_nwp = nwp_lf.rename(rename_mapping)

                joined_df_lf = joined_df_lf.join(
                    prefixed_nwp,
                    on=["valid_time", "h3_index"],
                    how="left",
                )

        joined_df = cast(pl.DataFrame, joined_df_lf.collect())

        # Prepare features and target
        X = self._prepare_features(joined_df)
        y = joined_df.select("MW").to_series()

        self.model = XGBRegressor(**config.hyperparameters.model_dump())
        self.model.fit(X, y)

        return self.model

    def predict(  # type: ignore[override]
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            substation_metadata: The substation metadata containing h3 mapping.
            inference_params: Parameters for inference.
            nwps: A dictionary of weather forecast lazyframes.
            **kwargs: Additional arguments (unused).

        Returns:
            A Patito DataFrame containing the predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")

        # We need to know which NWPs are required from the model config.
        # However, XGBoostForecaster doesn't store the config.
        # For now, we assume if nwps is provided, we use it.
        # In a more robust implementation, we might store the feature names.

        # Expand metadata to all substations
        metadata_df = substation_metadata.select(["substation_number", "h3_res_5"])

        # We'll start with the metadata and join NWPs to it.
        # This is slightly different from train because we don't have power flows.
        # We need a base dataframe with valid_time.
        # We'll take the first NWP as the base for valid_time and ensemble_member.

        if not nwps:
            # If no NWPs, we might be in a pure AR mode, but XGBoostForecaster
            # currently expects weather. Let's fail loudly if nwps is missing
            # but we expect it.
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        first_nwp_name = next(iter(nwps))
        first_nwp = nwps[first_nwp_name]

        df_lf = first_nwp.select(["valid_time", "h3_index", "ensemble_member"]).join(
            metadata_df.rename({"h3_res_5": "h3_index"}).lazy(),
            on="h3_index",
            how="inner",
        )

        for nwp_name, nwp_lf in nwps.items():
            prefix = f"{nwp_name.value}_"
            rename_mapping = {
                col: f"{prefix}{col}"
                for col in nwp_lf.collect_schema().names()
                if col not in ["valid_time", "h3_index", "ensemble_member"]
            }
            prefixed_nwp = nwp_lf.rename(rename_mapping)

            df_lf = df_lf.join(
                prefixed_nwp,
                on=["valid_time", "h3_index", "ensemble_member"],
                how="left",
            )

        df = cast(pl.DataFrame, df_lf.collect())

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
