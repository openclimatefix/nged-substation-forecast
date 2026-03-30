"""XGBoost implementation of the Forecaster interface."""

import logging
from datetime import datetime, timedelta, timezone
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
from xgboost_forecaster.data import downsample_power_flows

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
        flows_30m = downsample_power_flows(substation_power_flows)

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

        # 3. Create dynamic seasonal lag to prevent data leakage for week-2 forecasts.
        # In production, when predicting > 7 days ahead, the 7-day lag is in the future.
        # We switch to the 14-day lag for these horizons.
        seven_days_h = timedelta(days=7).total_seconds() / 3600
        joined_df_lf = (
            joined_df_lf.with_columns(
                lead_time_hours_temp=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes()
                / 60.0
            )
            .with_columns(
                latest_available_weekly_lag=pl.when(pl.col("lead_time_hours_temp") <= seven_days_h)
                .then(pl.col("power_lag_7d"))
                .otherwise(pl.col("power_lag_14d"))
            )
            .drop(["power_lag_7d", "power_lag_14d", "lead_time_hours_temp"])
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
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        **kwargs,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            substation_metadata: The substation metadata containing h3 mapping.
            inference_params: Parameters for inference.
            substation_power_flows: The historical power flow data (for lags).
            nwps: A dictionary of weather forecast lazyframes.
            **kwargs: Additional arguments (unused).

        Returns:
            A Patito DataFrame containing the predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")

        # Expand metadata to all substations
        metadata_df = substation_metadata.select(["substation_number", "h3_res_5"])

        if not nwps:
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        # Filter NWPs to the specific init_time requested for inference
        target_init_time = inference_params.nwp_init_time
        filtered_nwps = {}
        for name, lf in nwps.items():
            # Robust "as-of" filter: take the latest run available at or before target_init_time
            filtered_nwps[name] = (
                lf.filter(pl.col("init_time") <= target_init_time)
                .sort("init_time")
                .group_by(["valid_time", "h3_index", "ensemble_member"])
                .last()
            )

        # 1. Initialize combined NWP LazyFrame with the first NWP
        nwps_iter = iter(filtered_nwps.items())
        first_nwp_name, first_nwp_lf = next(nwps_iter)

        prefix = f"{first_nwp_name.value}_"
        rename_mapping = {
            col: f"{prefix}{col}"
            for col in first_nwp_lf.collect_schema().names()
            if col not in ["valid_time", "h3_index", "ensemble_member", "init_time"]
        }
        combined_nwps_lf = first_nwp_lf.rename(rename_mapping)

        # Loop through REMAINING nwps to apply prefixes consistently
        for nwp_name, nwp_lf in nwps_iter:
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
        flows_30m = downsample_power_flows(substation_power_flows)

        # Generate lags by shifting timestamps forward
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

        # 5. Create dynamic seasonal lag (same logic as train)
        seven_days_h = timedelta(days=7).total_seconds() / 3600
        df_lf = (
            df_lf.with_columns(
                lead_time_hours_temp=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes()
                / 60.0
            )
            .with_columns(
                latest_available_weekly_lag=pl.when(pl.col("lead_time_hours_temp") <= seven_days_h)
                .then(pl.col("power_lag_7d"))
                .otherwise(pl.col("power_lag_14d"))
            )
            .drop(["power_lag_7d", "power_lag_14d", "lead_time_hours_temp"])
        )

        df = cast(pl.DataFrame, df_lf.collect())

        # Prepare features (must match training features)
        X = self._prepare_features(df)

        # Enforce exact column order from training
        if hasattr(self.model, "feature_names_in_"):
            X = X.select(self.model.feature_names_in_)
        elif hasattr(self, "feature_names") and self.feature_names:
            X = X.select(self.feature_names)

        if X.is_empty():
            log.warning("Feature matrix X is empty. Returning empty predictions.")
            preds = []
        else:
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
