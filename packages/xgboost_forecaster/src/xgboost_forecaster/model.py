"""XGBoost implementation of the Forecaster interface."""

import logging
import os
import tempfile
from collections.abc import Mapping
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

    def _prepare_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Extract the feature matrix.

        Args:
            df: The joined input data with all features already added.

        Returns:
            A Polars DataFrame or LazyFrame containing only the feature columns.
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
            }
            if isinstance(df, pl.LazyFrame):
                schema = df.collect_schema()
            else:
                schema = df.schema
            feature_cols = [
                c
                for c, dtype in schema.items()
                if c not in exclude_cols and (dtype.is_numeric() or dtype == pl.Categorical)
            ]
            res = df.select(feature_cols)
        else:
            res = df.select(self.config.features.feature_names)

        if isinstance(res, pl.LazyFrame):
            res_schema = res.collect_schema()
        else:
            res_schema = res.schema  # Fail loudly if columns are missing

        # Ensure substation_number is treated as a categorical feature by XGBoost
        if "substation_number" in res_schema.names():
            res = res.with_columns(pl.col("substation_number").cast(pl.String).cast(pl.Categorical))

        # Ensure categorical precipitation is treated as a categorical feature
        if "categorical_precipitation_type_surface" in res_schema.names():
            res = res.with_columns(
                pl.col("categorical_precipitation_type_surface")
                .cast(pl.String)
                .cast(pl.Categorical)
            )

        return res

    def _add_lags(self, df: pl.LazyFrame, flows_30m: pl.LazyFrame) -> pl.LazyFrame:
        """Add autoregressive lags to the feature matrix.

        Args:
            df: The input LazyFrame (must contain valid_time and init_time).
            flows_30m: Historical power flows downsampled to 30m.

        Returns:
            LazyFrame with added lag features.
        """
        # Get telemetry delay from config, default to 24 hours if not set
        telemetry_delay_hours = (
            getattr(self.config, "telemetry_delay_hours", 24) if hasattr(self, "config") else 24
        )

        # 2. Calculate the required lag dynamically to strictly prevent lookahead bias
        df = (
            df.with_columns(
                lead_time_days=(pl.col("valid_time") - pl.col("init_time")).dt.total_seconds()
                / (24 * 3600)
            )
            .with_columns(
                lag_days=pl.max_horizontal(
                    pl.lit(1),
                    ((pl.col("lead_time_days") + telemetry_delay_hours / 24.0) / 7.0)
                    .ceil()
                    .cast(pl.Int32),
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
        nwps: Mapping[NwpModel, pl.LazyFrame],
        nwp_cutoff: datetime | None = None,
        collapse_lead_times: bool = False,
    ) -> pl.LazyFrame:
        """Prepare and join multiple NWP sources."""
        # Get delay from config, default to 3 hours if not set
        delay_hours = (
            getattr(self.config, "nwp_availability_delay_hours", 3)
            if hasattr(self, "config")
            else 3
        )

        nwp_list = []
        for i, (name, lf) in enumerate(nwps.items()):
            lf_with_features = add_weather_features(lf)

            # Always filter to ensure NWP is available before valid_time
            # This prevents lookahead bias during training
            available_nwp = lf_with_features.filter(
                pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours)
                <= pl.col(NwpColumns.VALID_TIME)
            )

            if nwp_cutoff is not None:
                if collapse_lead_times:
                    latest_nwp = (
                        available_nwp.filter(
                            pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours)
                            <= nwp_cutoff
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
                    latest_nwp = available_nwp.filter(
                        pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours) <= nwp_cutoff
                    )
            else:
                latest_nwp = available_nwp

            if i == 0:
                prefixed_nwp = latest_nwp.with_columns(
                    available_time=pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours)
                )
            else:
                prefix = f"{name.value}_"
                # For secondary NWP models, filter to ensemble_member == 0 (control member)
                # and drop the column to avoid arbitrary pairing with primary NWP members.
                latest_nwp = latest_nwp.filter(pl.col(NwpColumns.ENSEMBLE_MEMBER) == 0).drop(
                    NwpColumns.ENSEMBLE_MEMBER
                )

                nwp_schema_names = latest_nwp.collect_schema().names()
                rename_mapping = {
                    col: f"{prefix}{col}"
                    for col in nwp_schema_names
                    if col
                    not in [
                        NwpColumns.VALID_TIME,
                        NwpColumns.H3_INDEX,
                    ]
                }
                prefixed_nwp = latest_nwp.rename(rename_mapping).with_columns(
                    available_time=pl.col(f"{prefix}{NwpColumns.INIT_TIME}")
                    + pl.duration(hours=delay_hours)
                )
            nwp_list.append(prefixed_nwp)

        combined_nwps = nwp_list[0]
        for other_nwp in nwp_list[1:]:
            combined_nwps = (
                combined_nwps.sort("available_time")
                .join_asof(
                    other_nwp.sort("available_time"),
                    on="available_time",
                    by=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX],
                )
                .with_columns(
                    # Explicitly cast h3_index to UInt64 after the join to prevent silent type
                    # coercion (e.g., to Float64 or Int64) during Polars joins, which is
                    # critical for downstream H3 operations and memory efficiency.
                    pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
                )
            )
        return combined_nwps

    def _prepare_data_for_model(
        self,
        substation_power_flows: pl.LazyFrame,
        substation_metadata: pl.LazyFrame | pl.DataFrame,
        target_map_df: pl.DataFrame,
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
        inference_params: InferenceParams | None = None,
        collapse_lead_times: bool = False,
    ) -> pl.LazyFrame:
        """Prepares data for training or prediction.

        Args:
            substation_power_flows: Historical power flows.
            substation_metadata: Substation metadata.
            target_map_df: Target mapping dataframe.
            nwps: Dictionary of NWP data.
            inference_params: Inference parameters (only for prediction).
            collapse_lead_times: Whether to collapse lead times (only for prediction).

        Returns:
            Prepared LazyFrame ready for feature extraction.
        """
        metadata_lf = substation_metadata.select(["substation_number", "h3_res_5"]).lazy()
        target_map_lf = target_map_df.lazy()

        # 1. Downsample power flows to 30m and calculate target fallback
        flows_30m = downsample_power_flows(substation_power_flows, target_map=target_map_lf)

        # 2. Prepare NWPs and join
        is_training = inference_params is None

        if nwps:
            nwp_cutoff = inference_params.forecast_time if inference_params is not None else None
            combined_nwps_lf = self._prepare_and_join_nwps(
                nwps,
                nwp_cutoff=nwp_cutoff,
                collapse_lead_times=collapse_lead_times,
            )

            if not is_training:
                # For prediction, start with NWPs and join metadata
                df_lf = combined_nwps_lf.join(
                    metadata_lf.rename({"h3_res_5": NwpColumns.H3_INDEX}),
                    on=NwpColumns.H3_INDEX,
                    how="inner",
                ).with_columns(
                    # Explicitly cast h3_index to UInt64 after the join to prevent silent type
                    # coercion (e.g., to Float64 or Int64) during Polars joins, which is
                    # critical for downstream H3 operations and memory efficiency.
                    pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
                )
            else:
                # For training, start with flows and join metadata, then NWPs
                df_lf = (
                    flows_30m.rename({"timestamp": NwpColumns.VALID_TIME})
                    .join(
                        metadata_lf.rename({"h3_res_5": NwpColumns.H3_INDEX}),
                        on="substation_number",
                    )
                    .with_columns(
                        # Explicitly cast h3_index to UInt64 after the join to prevent silent type
                        # coercion (e.g., to Float64 or Int64) during Polars joins, which is
                        # critical for downstream H3 operations and memory efficiency.
                        pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
                    )
                )
                df_lf = df_lf.join(
                    combined_nwps_lf,
                    on=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX],
                    how="left",
                ).with_columns(
                    # Explicitly cast h3_index to UInt64 after the join to prevent silent type
                    # coercion (e.g., to Float64 or Int64) during Polars joins, which is
                    # critical for downstream H3 operations and memory efficiency.
                    pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
                )
        else:
            if not is_training:
                raise ValueError("XGBoostForecaster requires NWP data for prediction.")
            # For training without NWPs
            df_lf = (
                flows_30m.rename({"timestamp": NwpColumns.VALID_TIME})
                .join(
                    metadata_lf.rename({"h3_res_5": NwpColumns.H3_INDEX}),
                    on="substation_number",
                )
                .with_columns(
                    # Explicitly cast h3_index to UInt64 after the join to prevent silent type
                    # coercion (e.g., to Float64 or Int64) during Polars joins, which is
                    # critical for downstream H3 operations and memory efficiency.
                    pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
                )
            )

        # 3. Add lags and features
        # Handle missing init_time (e.g. for autoregressive-only models)
        if NwpColumns.INIT_TIME not in df_lf.collect_schema().names():
            df_lf = df_lf.with_columns(**{NwpColumns.INIT_TIME: pl.col(NwpColumns.VALID_TIME)})

        df_lf = self._add_lags(df_lf, flows_30m)

        # Normalize by peak capacity
        df_lf = df_lf.join(
            target_map_lf.select(["substation_number", "peak_capacity_MW_or_MVA"]),
            on="substation_number",
            how="left",
        ).with_columns(
            latest_available_weekly_lag=pl.col("latest_available_weekly_lag")
            / pl.col("peak_capacity_MW_or_MVA")
        )

        if is_training:
            # For training, also normalize target
            df_lf = df_lf.with_columns(
                MW_or_MVA=pl.col("MW_or_MVA") / pl.col("peak_capacity_MW_or_MVA")
            )
        else:
            # For prediction, add dummy target for validation
            df_lf = df_lf.with_columns(MW_or_MVA=pl.lit(0.0, dtype=pl.Float32))

        df_lf = df_lf.drop("peak_capacity_MW_or_MVA")
        df_lf = add_cyclical_temporal_features(df_lf, time_col=NwpColumns.VALID_TIME)

        # 4. Type casting
        df_lf = df_lf.with_columns(
            [
                pl.col("substation_number").cast(pl.Int32),
                pl.col("MW_or_MVA").cast(pl.Float32),
            ]
        )
        if NwpColumns.ENSEMBLE_MEMBER in df_lf.collect_schema().names():
            df_lf = df_lf.with_columns(pl.col(NwpColumns.ENSEMBLE_MEMBER).cast(pl.UInt8))

        # DATA TYPE RATIONALE:
        # Weather features are kept as Float32 in memory rather than UInt8 to:
        # 1. Preserve precision from 30-minute interpolation (avoiding "staircase" effects).
        # 2. Prevent silent underflow during feature engineering (e.g., calculating trends
        #    via subtraction).
        # 3. Align with XGBoost's native internal data type (Float32).

        # Cast all floats to Float32 for Patito
        df_lf = df_lf.with_columns(pl.col(pl.Float64).cast(pl.Float32))

        return df_lf

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
                peak_capacity_MW_or_MVA=pl.max_horizontal(
                    pl.col("MW").abs().max(), pl.col("MVA").abs().max()
                ).fill_null(1.0),
            )
            .with_columns(
                pl.when(pl.col("mw_count") >= pl.col("mva_count"))
                .then(pl.lit("MW"))
                .otherwise(pl.lit("MVA"))
                .alias("power_col"),
                pl.when(pl.col("peak_capacity_MW_or_MVA") == 0.0)
                .then(pl.lit(1.0))
                .otherwise(pl.col("peak_capacity_MW_or_MVA"))
                .alias("peak_capacity_MW_or_MVA"),
            )
            .select(["substation_number", "power_col", "peak_capacity_MW_or_MVA"])
            .collect(),
        )

        self.target_map = SubstationTargetMap.validate(
            target_map_df.with_columns(
                [
                    pl.col("substation_number").cast(pl.Int32),
                    pl.col("peak_capacity_MW_or_MVA").cast(pl.Float32),
                ]
            )
        )

        joined_lf = self._prepare_data_for_model(
            substation_power_flows=substation_power_flows,
            substation_metadata=substation_metadata,
            target_map_df=target_map_df,
            nwps=nwps,
        )

        # Prepare features and target
        feature_lf = self._prepare_features(joined_lf)
        feature_cols = feature_lf.collect_schema().names()

        # Collect only necessary columns and drop nulls
        critical_cols = ["MW_or_MVA"]
        if nwps:
            critical_cols.extend(
                [
                    f"{NwpColumns.TEMPERATURE_2M}_uint8_scaled",
                    f"{NwpColumns.SW_RADIATION}_uint8_scaled",
                    NwpColumns.WIND_SPEED_10M,
                ]
            )

        raw_df = cast(
            pl.DataFrame,
            joined_lf.select(list(set(feature_cols + ["MW_or_MVA"]))).collect(),
        )
        joined_df = raw_df.drop_nulls(subset=critical_cols)

        dropped_rows = len(raw_df) - len(joined_df)
        if dropped_rows > 0:
            log.warning(
                f"Dropped {dropped_rows} rows during training due to nulls in critical columns: {critical_cols}"
            )

        if joined_df.is_empty():
            raise ValueError("No training data remaining after dropping nulls in critical columns.")

        SubstationFeatures.validate(
            joined_df, allow_missing_columns=True, allow_superfluous_columns=True
        )

        X = cast(pl.DataFrame, self._prepare_features(joined_df))
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

        if not nwps:
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        if self.target_map is None:
            raise ValueError("target_map must be set before calling predict.")
        target_map_lf = self.target_map.lazy()

        df_lf = self._prepare_data_for_model(
            substation_power_flows=substation_power_flows,
            substation_metadata=substation_metadata,
            target_map_df=self.target_map,
            nwps=nwps,
            inference_params=inference_params,
            collapse_lead_times=collapse_lead_times,
        )

        feature_lf = self._prepare_features(df_lf)
        feature_cols = feature_lf.collect_schema().names()

        # Output columns needed for the final result
        output_cols = [
            NwpColumns.VALID_TIME,
            "substation_number",
            NwpColumns.ENSEMBLE_MEMBER,
            NwpColumns.INIT_TIME,
        ]

        critical_cols = []
        if nwps:
            critical_cols.extend(
                [
                    f"{NwpColumns.TEMPERATURE_2M}_uint8_scaled",
                    f"{NwpColumns.SW_RADIATION}_uint8_scaled",
                    NwpColumns.WIND_SPEED_10M,
                ]
            )

        raw_df = cast(
            pl.DataFrame,
            df_lf.select(list(set(feature_cols + output_cols + ["MW_or_MVA"]))).collect(),
        )
        df = raw_df.drop_nulls(subset=critical_cols)

        dropped_rows = len(raw_df) - len(df)
        if dropped_rows > 0:
            log.warning(
                f"Dropped {dropped_rows} rows during prediction due to nulls in critical columns: {critical_cols}"
            )

        if df.is_empty():
            raise ValueError(
                "No inference data remaining after dropping nulls in critical columns."
            )

        SubstationFeatures.validate(df, allow_missing_columns=True, allow_superfluous_columns=True)

        X = cast(pl.DataFrame, self._prepare_features(df))

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

        # Descale predictions and return with correct schema
        fcst_init_time = inference_params.forecast_time
        model_name = inference_params.power_fcst_model_name or "xgboost_global"

        res = (
            df.with_columns(
                MW_or_MVA=pl.Series(values=preds, dtype=pl.Float32),
            )
            .join(
                cast(
                    pl.DataFrame,
                    target_map_lf.select(
                        ["substation_number", "peak_capacity_MW_or_MVA"]
                    ).collect(),
                ),
                on="substation_number",
                how="left",
            )
            .with_columns(
                MW_or_MVA=pl.col("MW_or_MVA") * pl.col("peak_capacity_MW_or_MVA"),
                power_fcst_model_name=pl.lit(model_name).cast(pl.Categorical),
                power_fcst_init_time=pl.lit(fcst_init_time).cast(pl.Datetime("us", "UTC")),
                nwp_init_time=pl.col(NwpColumns.INIT_TIME).cast(pl.Datetime("us", "UTC")),
                power_fcst_init_year_month=pl.lit(fcst_init_time.strftime("%Y-%m")).cast(pl.String),
            )
            .select(
                [
                    NwpColumns.VALID_TIME,
                    "substation_number",
                    NwpColumns.ENSEMBLE_MEMBER,
                    "power_fcst_model_name",
                    "power_fcst_init_time",
                    "nwp_init_time",
                    "power_fcst_init_year_month",
                    "MW_or_MVA",
                ]
            )
        )

        # Handle potential nulls in ensemble_member (required by schema)
        res = res.with_columns(pl.col(NwpColumns.ENSEMBLE_MEMBER).fill_null(0).cast(pl.UInt8))

        return PowerForecast.validate(res)
