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

import mlflow

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

    def log_model(self, model_name: str) -> None:
        """Log the model to MLflow."""
        if self.model is not None:
            mlflow.xgboost.log_model(self.model, artifact_path="model")

    def _prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply feature engineering and extract the feature matrix.

        Args:
            df: The joined input data.

        Returns:
            A Polars DataFrame containing only the feature columns.
        """
        # 1. Apply shared feature engineering
        df = add_cyclical_temporal_features(df, time_col="valid_time")

        # Cast substation_number to categorical if it exists
        if "substation_number" in df.columns:
            df = df.with_columns(pl.col("substation_number").cast(pl.String).cast(pl.Categorical))

        # 2. Extract feature matrix using explicit feature names from config
        if not hasattr(self, "config") or not self.config.features.feature_names:
            # Fallback: keep only numeric and categorical columns, excluding the target and h3_index
            exclude_cols = {
                "MW",
                "MVA",
                "MVAr",
                "MW_or_MVA",
                "h3_index",
                "ensemble_member",
                "init_time",
            }
            feature_cols = [
                c
                for c in df.columns
                if c not in exclude_cols
                and (df[c].dtype.is_numeric() or df[c].dtype == pl.Categorical)
            ]
            return df.select(feature_cols)

        return df.select(self.config.features.feature_names)

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
        self.config = config
        log.info("Starting XGBoost training...")

        if len(config.features.nwps) > 0 and not nwps:
            raise ValueError("Model config requires NWPs, but none were provided.")

        metadata_lf = substation_metadata.select(["substation_number", "h3_res_5"]).lazy()

        # 1. Downsample power flows to 30m and calculate target fallback
        flows_30m = (
            substation_power_flows.sort("timestamp")
            .group_by_dynamic(
                "timestamp",
                every="30m",
                group_by="substation_number",
                closed="right",
                label="right",
            )
            .agg(
                [
                    pl.col("MW").mean(),
                    pl.col("MVA").mean(),
                ]
            )
            .with_columns(
                pl.when(pl.col("MW").is_not_null())
                .then(pl.col("MW"))
                .otherwise(pl.col("MVA"))
                .alias("MW_or_MVA")
            )
        )

        # 2. Generate lags by shifting timestamps forward
        lag_7d = flows_30m.select(
            pl.col("substation_number"),
            (pl.col("timestamp") + pl.duration(days=7)).alias("valid_time"),
            pl.col("MW_or_MVA").alias("power_lag_7d"),
        )

        lag_14d = flows_30m.select(
            pl.col("substation_number"),
            (pl.col("timestamp") + pl.duration(days=14)).alias("valid_time"),
            pl.col("MW_or_MVA").alias("power_lag_14d"),
        )

        # Join power flows and weather data using substation metadata.
        flows_with_h3 = flows_30m.rename({"timestamp": "valid_time"}).join(
            metadata_lf.rename({"h3_res_5": "h3_index"}),
            on="substation_number",
        )

        joined_df_lf = flows_with_h3.join(
            lag_7d, on=["substation_number", "valid_time"], how="left"
        ).join(lag_14d, on=["substation_number", "valid_time"], how="left")

        if nwps:
            for nwp_name, nwp_lf in nwps.items():
                # Prefix columns to avoid collisions
                prefix = f"{nwp_name.value}_"
                rename_mapping = {
                    col: f"{prefix}{col}"
                    for col in nwp_lf.collect_schema().names()
                    if col not in ["valid_time", "h3_index", "ensemble_member", "init_time"]
                }
                prefixed_nwp = nwp_lf.rename(rename_mapping)

                # Join on ensemble_member and init_time if they exist in both dataframes
                join_keys = ["valid_time", "h3_index"]
                if "ensemble_member" in joined_df_lf.collect_schema().names():
                    join_keys.append("ensemble_member")
                if "init_time" in joined_df_lf.collect_schema().names():
                    join_keys.append("init_time")

                joined_df_lf = joined_df_lf.join(
                    prefixed_nwp,
                    on=join_keys,
                    how="left",
                )

        joined_df = cast(pl.DataFrame, joined_df_lf.collect())

        # Prepare features and target
        X = self._prepare_features(joined_df)
        y = joined_df.select("MW_or_MVA").to_series()

        # Save feature names as a list of strings
        self.feature_names = X.columns

        self.model = XGBRegressor(**config.hyperparameters.model_dump())
        self.model.fit(X.to_arrow(), y.to_arrow())

        return self.model

    def predict(  # type: ignore[override]
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        substation_power_flows: pt.LazyFrame[SubstationFlows] | None = None,
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
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        first_nwp_name = next(iter(nwps))

        # Filter NWPs to the specific init_time requested for inference
        # If multiple init_times are present, we take the one specified in inference_params
        target_init_time = inference_params.nwp_init_time
        filtered_nwps = {}
        for name, lf in nwps.items():
            filtered_nwps[name] = lf.filter(pl.col("init_time") == target_init_time)

        # FIX: Create a base dataframe with just the keys
        combined_nwps_lf = filtered_nwps[first_nwp_name].select(
            ["valid_time", "h3_index", "ensemble_member", "init_time"]
        )

        # FIX: Loop through ALL nwps to apply prefixes consistently
        for nwp_name, nwp_lf in filtered_nwps.items():
            prefix = f"{nwp_name.value}_"
            rename_mapping = {
                col: f"{prefix}{col}"
                for col in nwp_lf.collect_schema().names()
                if col not in ["valid_time", "h3_index", "ensemble_member", "init_time"]
            }
            prefixed_nwp = nwp_lf.rename(rename_mapping)

            # Join on ensemble_member and init_time if they exist in both dataframes
            join_keys = ["valid_time", "h3_index"]
            if "ensemble_member" in combined_nwps_lf.collect_schema().names():
                join_keys.append("ensemble_member")
            if "init_time" in combined_nwps_lf.collect_schema().names():
                join_keys.append("init_time")

            combined_nwps_lf = combined_nwps_lf.join(
                prefixed_nwp,
                on=join_keys,
                how="left",
            )

        # 2. Join the resulting combined NWP LazyFrame with metadata_df on h3_index
        df_lf = combined_nwps_lf.join(
            metadata_df.rename({"h3_res_5": "h3_index"}).lazy(),
            on="h3_index",
            how="inner",
        )

        # 3. Apply the same lag generation logic to substation_power_flows
        if substation_power_flows is not None:
            flows_30m = (
                substation_power_flows.sort("timestamp")
                .group_by_dynamic(
                    "timestamp",
                    every="30m",
                    group_by="substation_number",
                    closed="right",
                    label="right",
                )
                .agg(
                    [
                        pl.col("MW").mean(),
                        pl.col("MVA").mean(),
                    ]
                )
                .with_columns(
                    pl.when(pl.col("MW").is_not_null())
                    .then(pl.col("MW"))
                    .otherwise(pl.col("MVA"))
                    .alias("MW_or_MVA")
                )
            )

            # FIX: Generate lags by shifting timestamps forward
            lag_7d = flows_30m.select(
                pl.col("substation_number"),
                (pl.col("timestamp") + pl.duration(days=7)).alias("valid_time"),
                pl.col("MW_or_MVA").alias("power_lag_7d"),
            )

            lag_14d = flows_30m.select(
                pl.col("substation_number"),
                (pl.col("timestamp") + pl.duration(days=14)).alias("valid_time"),
                pl.col("MW_or_MVA").alias("power_lag_14d"),
            )

            # 4. Join power flows into the feature matrix
            df_lf = df_lf.join(lag_7d, on=["substation_number", "valid_time"], how="left").join(
                lag_14d, on=["substation_number", "valid_time"], how="left"
            )

        df = cast(pl.DataFrame, df_lf.collect())

        # Prepare features (must match training features)
        X = self._prepare_features(df)

        # Enforce exact column order from training
        if hasattr(self, "feature_names") and self.feature_names:
            X = X.select(self.feature_names)

        preds = self.model.predict(X.to_arrow())

        # Return predictions with correct schema
        now = datetime.now(timezone.utc)
        model_name = inference_params.power_fcst_model_name or "xgboost_global"

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
