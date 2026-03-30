"""XGBoost implementation of the Forecaster interface."""

import logging
import os
import tempfile
from datetime import datetime
from typing import cast

import mlflow
import patito as pt
import polars as pl

from contracts.data_schemas import (
    InferenceParams,
    NwpColumns,
    PowerForecast,
    ProcessedNwp,
    SubstationFeatures,
    SubstationFlows,
    SubstationMetadata,
    SubstationTargetMap,
)
from contracts.hydra_schemas import ModelConfig, NwpModel
from ml_core.data import downsample_power_flows
from ml_core.features import add_cyclical_temporal_features
from ml_core.model import BaseForecaster
from xgboost import XGBRegressor
from xgboost_forecaster.config import XGBoostHyperparameters
from xgboost_forecaster.features import add_weather_features

log = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost implementation of the Forecaster interface.

    This class handles the full lifecycle of an XGBoost model: training
    and production inference.
    """

    model: XGBRegressor | None
    target_map: pl.DataFrame | None
    config: ModelConfig
    feature_names: list[str]

    def __init__(self, model: XGBRegressor | None = None):
        """Initialize the forecaster.

        Args:
            model: An optional pre-trained XGBoost model.
        """
        self.model = model
        self.target_map = None

    def log_model(self, model_name: str) -> None:
        """Log the model to MLflow."""
        if self.model is not None:
            mlflow.xgboost.log_model(self.model, artifact_path="model")

        if hasattr(self, "target_map") and self.target_map is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "target_map.json")
                self.target_map.write_json(path)
                mlflow.log_artifact(path, artifact_path="metadata")

    def _prepare_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Extract the feature matrix.

        Args:
            df: The joined input data with all features already added.

        Returns:
            A Polars LazyFrame containing only the feature columns.
        """
        # Extract feature matrix using explicit feature names from config
        if not hasattr(self, "config") or not self.config.features.feature_names:
            # Fallback: keep only numeric and categorical columns, excluding the target and metadata
            exclude_cols = {
                "MW",
                "MVA",
                "MVAr",
                "MW_or_MVA",
                "h3_index",
                "ensemble_member",
                "init_time",
                "available_time",
                "lead_time_days",
            }
            schema = df.collect_schema()
            feature_cols = [
                c
                for c, dtype in schema.items()
                if c not in exclude_cols and (dtype.is_numeric() or dtype == pl.Categorical)
            ]
            res = df.select(feature_cols)
        else:
            res = df.select(self.config.features.feature_names)

        res_schema = res.collect_schema()  # Fail loudly if columns are missing

        # Ensure substation_number is treated as a categorical feature by XGBoost
        if "substation_number" in res_schema.names():
            res = res.with_columns(pl.col("substation_number").cast(pl.String).cast(pl.Categorical))

        return res

    def _add_lags(self, df: pl.LazyFrame, flows_30m: pl.LazyFrame) -> pl.LazyFrame:
        """Add autoregressive lags to the feature matrix.

        Args:
            df: The input LazyFrame (must contain valid_time and init_time).
            flows_30m: Historical power flows downsampled to 30m.

        Returns:
            LazyFrame with added lag features.
        """
        # 2. Calculate the required lag dynamically to strictly prevent lookahead bias
        df = (
            df.with_columns(
                lead_time_days=(pl.col("valid_time") - pl.col("init_time")).dt.total_days()
            )
            .with_columns(
                lag_days=pl.max_horizontal(
                    pl.lit(1), (pl.col("lead_time_days") / 7.0).ceil().cast(pl.Int32)
                )
                * 7
            )
            .with_columns(
                target_lag_time=pl.col("valid_time") - pl.duration(days=1) * pl.col("lag_days")
            )
        )

        # 3. Join flows_30m on ["substation_number", "target_lag_time"] to extract the exact
        # latest_available_weekly_lag without needing pre-calculated lag_7d or lag_14d columns.
        lag_df = flows_30m.select(
            pl.col("substation_number"),
            pl.col("timestamp").alias("target_lag_time"),
            pl.col("MW_or_MVA").alias("latest_available_weekly_lag"),
        )

        df = df.join(lag_df, on=["substation_number", "target_lag_time"], how="left")

        return df

    def _prepare_and_join_nwps(
        self,
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]],
        nwp_cutoff: datetime | None = None,
        collapse_lead_times: bool = False,
    ) -> pl.LazyFrame:
        """Prepare and join multiple NWP sources."""
        nwp_list = []
        for i, (name, lf) in enumerate(nwps.items()):
            lf_with_features = add_weather_features(lf)

            if nwp_cutoff is not None:
                if collapse_lead_times:
                    latest_nwp = (
                        lf_with_features.filter(
                            pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=3) <= nwp_cutoff
                        )
                        .sort(NwpColumns.INIT_TIME)
                        .group_by(
                            [
                                NwpColumns.VALID_TIME,
                                NwpColumns.H3_INDEX,
                                NwpColumns.ENSEMBLE_MEMBER,
                            ]
                        )
                        .last()
                    )
                else:
                    latest_nwp = lf_with_features.filter(
                        pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=3) <= nwp_cutoff
                    )
            else:
                latest_nwp = lf_with_features

            if i == 0:
                prefixed_nwp = latest_nwp.with_columns(
                    available_time=pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=3)
                )
            else:
                prefix = f"{name.value}_"
                nwp_schema_names = latest_nwp.collect_schema().names()
                rename_mapping = {
                    col: f"{prefix}{col}"
                    for col in nwp_schema_names
                    if col
                    not in [
                        NwpColumns.VALID_TIME,
                        NwpColumns.H3_INDEX,
                        NwpColumns.ENSEMBLE_MEMBER,
                    ]
                }
                prefixed_nwp = latest_nwp.rename(rename_mapping).with_columns(
                    available_time=pl.col(f"{prefix}{NwpColumns.INIT_TIME}") + pl.duration(hours=3)
                )
            nwp_list.append(prefixed_nwp)

        combined_nwps = nwp_list[0]
        for other_nwp in nwp_list[1:]:
            combined_nwps = combined_nwps.sort("available_time").join_asof(
                other_nwp.sort("available_time"),
                on="available_time",
                by=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX, NwpColumns.ENSEMBLE_MEMBER],
            )
        return combined_nwps

    def train(
        self,
        config: ModelConfig,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        substation_metadata: pt.DataFrame[SubstationMetadata],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
    ) -> "XGBoostForecaster":
        """Train the XGBoost model.

        Args:
            config: The model configuration object.
            substation_power_flows: The historical power flow data.
            substation_metadata: The substation metadata containing h3 mapping.
            nwps: A dictionary of weather forecast dataframes.

        Returns:
            The trained XGBoostForecaster instance.
        """
        self.config = config
        log.info("Starting XGBoost training...")

        if len(config.features.nwps) > 0 and not nwps:
            raise ValueError("Model config requires NWPs, but none were provided.")

        # Calculate target map based on the training data
        target_map_df = cast(
            pl.DataFrame,
            substation_power_flows.group_by("substation_number")
            .agg(
                mw_count=pl.col("MW").is_not_null().sum(),
                mva_count=pl.col("MVA").is_not_null().sum(),
                peak_capacity=pl.max_horizontal(
                    pl.col("MW").abs().max(), pl.col("MVA").abs().max()
                ).fill_null(1.0),
            )
            .with_columns(
                pl.when(pl.col("mw_count") >= pl.col("mva_count"))
                .then(pl.lit("MW"))
                .otherwise(pl.lit("MVA"))
                .alias("target_col"),
                pl.when(pl.col("peak_capacity") == 0.0)
                .then(pl.lit(1.0))
                .otherwise(pl.col("peak_capacity"))
                .alias("peak_capacity"),
            )
            .select(["substation_number", "target_col", "peak_capacity"])
            .collect(),
        )

        self.target_map = SubstationTargetMap.validate(
            target_map_df.with_columns(
                [
                    pl.col("substation_number").cast(pl.Int32),
                    pl.col("peak_capacity").cast(pl.Float32),
                ]
            )
        )

        metadata_lf = substation_metadata.select(["substation_number", "h3_res_5"]).lazy()

        # 1. Downsample power flows to 30m and calculate target fallback
        flows_30m = downsample_power_flows(substation_power_flows, target_map=target_map_df.lazy())

        # Join power flows and weather data using substation metadata.
        joined_df_lf = flows_30m.rename({"timestamp": NwpColumns.VALID_TIME}).join(
            metadata_lf.rename({"h3_res_5": NwpColumns.H3_INDEX}),
            on="substation_number",
        )

        if nwps:
            combined_nwps = self._prepare_and_join_nwps(nwps)
            joined_df_lf = joined_df_lf.join(
                combined_nwps,
                on=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX],
                how="left",
            )

        # 2. Add lags and features
        # Handle missing init_time (e.g. for autoregressive-only models)
        if NwpColumns.INIT_TIME not in joined_df_lf.collect_schema().names():
            joined_df_lf = joined_df_lf.with_columns(
                **{NwpColumns.INIT_TIME: pl.col(NwpColumns.VALID_TIME)}
            )

        joined_df_lf = self._add_lags(joined_df_lf, flows_30m)
        joined_df_lf = (
            joined_df_lf.join(
                target_map_df.lazy().select(["substation_number", "peak_capacity"]),
                on="substation_number",
                how="left",
            )
            .with_columns(
                MW_or_MVA=pl.col("MW_or_MVA") / pl.col("peak_capacity"),
                latest_available_weekly_lag=pl.col("latest_available_weekly_lag")
                / pl.col("peak_capacity"),
            )
            .drop("peak_capacity")
        )
        joined_df_lf = add_cyclical_temporal_features(joined_df_lf, time_col=NwpColumns.VALID_TIME)
        # add_weather_features is now called on each NWP before prefixing

        # 3. Validate schema
        joined_df = cast(pl.DataFrame, joined_df_lf.collect())

        # Ensure correct dtypes before validation
        joined_df = joined_df.with_columns(
            [
                pl.col("substation_number").cast(pl.Int32),
                pl.col("MW_or_MVA").cast(pl.Float32),
            ]
        )
        if NwpColumns.ENSEMBLE_MEMBER in joined_df.columns:
            joined_df = joined_df.with_columns(pl.col(NwpColumns.ENSEMBLE_MEMBER).cast(pl.UInt8))

        # Cast all floats to Float32 for Patito
        joined_df = joined_df.with_columns(pl.col(pl.Float64).cast(pl.Float32))

        # Drop rows with missing critical features before validation
        critical_cols = [
            f"{NwpColumns.TEMPERATURE_2M}_uint8_scaled",
            "MW_or_MVA",
        ]
        joined_df = joined_df.drop_nulls(subset=critical_cols)
        if joined_df.is_empty():
            raise ValueError("No training data remaining after dropping nulls in critical columns.")

        SubstationFeatures.validate(
            joined_df, drop_superfluous_columns=False, allow_superfluous_columns=True
        )

        # Prepare features and target
        X_lf = self._prepare_features(joined_df.lazy())
        X = cast(pl.DataFrame, X_lf.collect())
        y = joined_df.select("MW_or_MVA").to_series()

        # NaN/Inf checks
        if (
            X.select(
                pl.any_horizontal(
                    pl.col(pl.Float32, pl.Float64).is_nan()
                    | pl.col(pl.Float32, pl.Float64).is_infinite()
                )
            )
            .sum()
            .item()
            > 0
        ):
            raise ValueError("Input features X contain NaN or Inf values")

        if y.is_nan().any() or y.is_infinite().any():
            raise ValueError("Target y contains NaN or Inf values")

        # Save feature names
        self.feature_names = X.columns

        hyperparams = XGBoostHyperparameters(**config.hyperparameters)
        model = XGBRegressor(**hyperparams.model_dump())
        model.fit(X.to_arrow(), y.to_arrow())
        self.model = model

        return self

    def predict(
        self,
        substation_metadata: pt.DataFrame[SubstationMetadata],
        inference_params: InferenceParams,
        substation_power_flows: pt.LazyFrame[SubstationFlows],
        nwps: dict[NwpModel, pt.LazyFrame[ProcessedNwp]] | None = None,
        collapse_lead_times: bool = False,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            substation_metadata: The substation metadata containing h3 mapping.
            inference_params: Parameters for inference.
            substation_power_flows: The historical power flow data (for lags).
            nwps: A dictionary of weather forecast lazyframes.
            collapse_lead_times: Whether to collapse lead times to simulate real-time inference by keeping only the latest available NWP forecast for each valid time.

        Returns:
            A Patito DataFrame containing the predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")

        # Expand metadata to all substations
        metadata_df = substation_metadata.select(["substation_number", "h3_res_5"])

        if not nwps:
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        # Filter NWPs to the specific init_time requested for inference.
        # The 3-hour availability delay is already handled during the multi-NWP join_asof in train.
        combined_nwps_lf = self._prepare_and_join_nwps(
            nwps,
            nwp_cutoff=inference_params.nwp_init_time,
            collapse_lead_times=collapse_lead_times,
        )

        # 2. Join with metadata
        df_lf = combined_nwps_lf.join(
            metadata_df.rename({"h3_res_5": NwpColumns.H3_INDEX}).lazy(),
            on=NwpColumns.H3_INDEX,
            how="inner",
        )

        # 3. Add lags and features
        if self.target_map is None:
            raise ValueError("target_map must be set before calling predict.")
        target_map_lf = self.target_map.lazy()
        flows_30m = downsample_power_flows(substation_power_flows, target_map=target_map_lf)
        df_lf = self._add_lags(df_lf, flows_30m)
        df_lf = (
            df_lf.join(
                target_map_lf.select(["substation_number", "peak_capacity"]),
                on="substation_number",
                how="left",
            )
            .with_columns(
                latest_available_weekly_lag=pl.col("latest_available_weekly_lag")
                / pl.col("peak_capacity")
            )
            .drop("peak_capacity")
        )
        df_lf = add_cyclical_temporal_features(df_lf, time_col=NwpColumns.VALID_TIME)

        # add_weather_features is now called on each NWP before prefixing

        # 4. Prepare features
        # We need a dummy MW_or_MVA for validation if we use SubstationFeatures.validate
        df_lf = df_lf.with_columns(MW_or_MVA=pl.lit(0.0, dtype=pl.Float32))

        df = cast(pl.DataFrame, df_lf.collect())

        # Ensure correct dtypes before validation
        df = df.with_columns(
            [
                pl.col("substation_number").cast(pl.Int32),
                pl.col("MW_or_MVA").cast(pl.Float32),
            ]
        )
        if NwpColumns.ENSEMBLE_MEMBER in df.columns:
            df = df.with_columns(pl.col(NwpColumns.ENSEMBLE_MEMBER).cast(pl.UInt8))

        # Cast all floats to Float32 for Patito
        df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))

        # Drop rows with missing critical features before validation
        critical_cols = [f"{NwpColumns.TEMPERATURE_2M}_uint8_scaled"]
        initial_len = len(df)
        df = df.drop_nulls(subset=critical_cols)
        dropped_len = initial_len - len(df)
        if dropped_len > 0:
            log.warning(
                f"Dropped {dropped_len} rows due to missing critical features during inference."
            )

        if df.is_empty():
            raise ValueError(
                "No inference data remaining after dropping nulls in critical columns."
            )

        # Validate schema
        SubstationFeatures.validate(
            df, drop_superfluous_columns=False, allow_superfluous_columns=True
        )

        X_lf = self._prepare_features(df.lazy())
        X = cast(pl.DataFrame, X_lf.collect())

        if (
            X.select(
                pl.any_horizontal(
                    pl.col(pl.Float32, pl.Float64).is_nan()
                    | pl.col(pl.Float32, pl.Float64).is_infinite()
                )
            )
            .sum()
            .item()
            > 0
        ):
            raise ValueError("Input features X contain NaN or Inf values")

        # Enforce exact column order from training
        if hasattr(self.model, "feature_names_in_"):
            X = X.select(self.model.feature_names_in_)
        elif hasattr(self, "feature_names") and self.feature_names:
            X = X.select(self.feature_names)

        preds = self.model.predict(X.to_arrow())

        # Descale predictions
        peak_capacities = (
            df.join(
                cast(
                    pl.DataFrame,
                    target_map_lf.select(["substation_number", "peak_capacity"]).collect(),
                ),
                on="substation_number",
                how="left",
            )
            .select("peak_capacity")
            .to_series()
            .to_numpy()
        )

        preds = preds * peak_capacities

        # Return predictions with correct schema
        fcst_init_time = inference_params.nwp_init_time
        model_name = inference_params.power_fcst_model_name or "xgboost_global"

        res = (
            df.select(
                [
                    NwpColumns.VALID_TIME,
                    "substation_number",
                    NwpColumns.ENSEMBLE_MEMBER,
                    NwpColumns.INIT_TIME,
                ]
            )
            .with_columns(
                MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32),
                power_fcst_model_name=pl.lit(model_name).cast(pl.Categorical),
                power_fcst_init_time=pl.lit(fcst_init_time).cast(pl.Datetime("us", "UTC")),
                nwp_init_time=pl.col(NwpColumns.INIT_TIME).cast(pl.Datetime("us", "UTC")),
                power_fcst_init_year_month=pl.lit(fcst_init_time.strftime("%Y-%m")).cast(pl.String),
            )
            .drop(NwpColumns.INIT_TIME)
        )

        # Handle potential nulls in ensemble_member (required by schema)
        res = res.with_columns(pl.col(NwpColumns.ENSEMBLE_MEMBER).fill_null(0).cast(pl.UInt8))

        return PowerForecast.validate(res)
