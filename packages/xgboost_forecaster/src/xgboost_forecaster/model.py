"""XGBoost implementation of the Forecaster interface."""

import logging
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
    PowerTimeSeries,
    TimeSeriesMetadata,
    XGBoostInputFeatures,
)
from contracts.hydra_schemas import ModelConfig, NwpModel
from ml_core.data import calculate_peak_capacity
from ml_core.features import add_cyclical_temporal_features
from ml_core.model import BaseForecaster
from xgboost import XGBRegressor

from xgboost_forecaster.config import XGBoostHyperparameters
from xgboost_forecaster.features import (
    add_autoregressive_lags,
    add_time_features,
    add_weather_features,
)

log = logging.getLogger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost implementation of the Forecaster interface.

    This class handles the full lifecycle of an XGBoost model: training
    and production inference.
    """

    model: XGBRegressor | None
    config: ModelConfig
    feature_names: list[str]

    def __init__(self, model: XGBRegressor | None = None):
        """Initialize the forecaster.

        Args:
            model: An optional pre-trained XGBoost model.
        """
        self.model = model
        self.feature_names = []

    # TODO: Use Patito data contracts for output of this function.
    def log_model(self, model_name: str) -> None:
        """Log the model to MLflow."""
        if self.model is not None:
            mlflow.xgboost.log_model(self.model, artifact_path="model")

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def _prepare_features(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Extract the feature matrix.

        Args:
            df: The joined input data with all features already added.

        Returns:
            A Polars DataFrame or LazyFrame containing only the feature columns.
        """
        # Extract feature matrix using explicit feature names.
        # We prioritize self.feature_names (set during training) over the config
        # to ensure consistency between training and inference.
        if self.feature_names:
            feature_names = self.feature_names
        elif hasattr(self, "config") and self.config.features.feature_names:
            feature_names = self.config.features.feature_names
        else:
            raise ValueError(
                "Feature names must be explicitly provided either in the config or "
                "during training. Fallback feature selection is disabled to prevent "
                "silent model degradation."
            )

        res = df.select(feature_names)

        if isinstance(res, pl.LazyFrame):
            res_schema = res.collect_schema()
        else:
            res_schema = res.schema  # Fail loudly if columns are missing

            # Ensure time_series_id is treated as a categorical feature by XGBoost
        if "time_series_id" in res_schema.names():
            df_unique_lf_or_df = df.select("time_series_id").unique()
            df_unique_df = (
                df_unique_lf_or_df.collect()
                if isinstance(df_unique_lf_or_df, pl.LazyFrame)
                else df_unique_lf_or_df
            )
            df_unique_df = cast(pl.DataFrame, df_unique_df)
            time_series_ids = [str(s) for s in df_unique_df.get_column("time_series_id").to_list()]

            res = res.with_columns(
                pl.col("time_series_id").cast(pl.String).cast(pl.Enum(time_series_ids))
            )

        # Ensure categorical precipitation is treated as a categorical feature
        if "categorical_precipitation_type_surface" in res_schema.names():
            res = res.with_columns(
                pl.col("categorical_precipitation_type_surface")
                .cast(pl.String)
                .cast(pl.Enum(["0", "1", "2", "3", "4", "5", "6", "7", "8"]))
            )

        return res

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def _collapse_lead_times(
        self,
        df: pl.LazyFrame,
        nwp_cutoff: datetime,
        delay_hours: int,
    ) -> pl.LazyFrame:
        """Collapse lead times to keep only the latest available forecast for each valid time.

        This simulates real-time inference by ensuring that for any given valid_time,
        we only use the most recent NWP forecast that would have been available
        at the nwp_cutoff time.

        Args:
            df: The input LazyFrame containing NWP data.
            nwp_cutoff: The time at which we are making the forecast.
            delay_hours: The availability delay of the NWP data.

        Returns:
            A LazyFrame with collapsed lead times.
        """
        return (
            df.filter(pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours) <= nwp_cutoff)
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

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def _prepare_and_join_nwps(
        self,
        nwp_lf: pl.LazyFrame,
        nwp_cutoff: datetime | None = None,
        collapse_lead_times: bool = False,
        is_training: bool = False,
    ) -> pl.LazyFrame:
        """Prepare NWP data."""
        # Get delay from config, default to 3 hours if not set
        delay_hours = self.config.nwp_availability_delay_hours if hasattr(self, "config") else 3

        # Add weather features (lags, trends, etc.)
        lf = add_weather_features(nwp_lf)
        # Add time features (lead_time_hours, nwp_init_hour)
        lf = add_time_features(lf)

        # CRITICAL: This filter is the sole mechanism preventing future data leakage.
        # It ensures that we only use NWP forecasts that would have been available
        # at the valid time, accounting for the data ingestion delay.
        # init_time + delay_hours <= valid_time
        available_nwp = lf.filter(
            pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours)
            <= pl.col(NwpColumns.VALID_TIME)
        )

        if nwp_cutoff is None:
            # During training, we use all available lead times.
            # The target_horizon_hours filter is removed to maximize training data.

            # For training, we only use the control member (0)
            # to avoid inflating the training set with highly correlated ensemble members.
            if is_training:
                available_nwp = available_nwp.filter(pl.col(NwpColumns.ENSEMBLE_MEMBER) == 0)

            latest_nwp = available_nwp
        else:
            if collapse_lead_times:
                latest_nwp = self._collapse_lead_times(available_nwp, nwp_cutoff, delay_hours)
            else:
                latest_nwp = available_nwp.filter(
                    pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours) <= nwp_cutoff
                )

        return latest_nwp.with_columns(
            available_time=pl.col(NwpColumns.INIT_TIME) + pl.duration(hours=delay_hours)
        )

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def _prepare_training_data(
        self,
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        metadata_lf: pl.LazyFrame,
        combined_nwps_lf: pl.LazyFrame | None = None,
    ) -> pl.LazyFrame:
        """Prepare data for training.

        For training, we start with the historical power flows and join the
        substation metadata and NWP forecasts.

        Args:
            power_time_series: Historical power flows at 30m resolution.
            metadata_lf: Substation metadata.
            combined_nwps_lf: Combined NWP forecasts.

        Returns:
            A LazyFrame containing the joined training data.
        """
        df_lf = (
            power_time_series.rename({"period_end_time": NwpColumns.VALID_TIME})
            .join(
                metadata_lf.rename({"h3_res_5": NwpColumns.H3_INDEX}),
                on="time_series_id",
            )
            .with_columns(
                # Explicitly cast h3_index to UInt64 after the join to prevent silent type
                # coercion (e.g., to Float64 or Int64) during Polars joins, which is
                # critical for downstream H3 operations and memory efficiency.
                pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
            )
        )

        if combined_nwps_lf is not None:
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
        return df_lf

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def _prepare_inference_data(
        self,
        metadata_lf: pl.LazyFrame,
        combined_nwps_lf: pl.LazyFrame,
    ) -> pl.LazyFrame:
        """Prepare data for inference.

        For inference, we start with the NWP forecasts and join the substation
        metadata.

        Args:
            metadata_lf: Substation metadata.
            combined_nwps_lf: Combined NWP forecasts.

        Returns:
            A LazyFrame containing the joined inference data.
        """
        return combined_nwps_lf.join(
            metadata_lf.rename({"h3_res_5": NwpColumns.H3_INDEX}),
            on=NwpColumns.H3_INDEX,
            how="inner",
        ).with_columns(
            # Explicitly cast h3_index to UInt64 after the join to prevent silent type
            # coercion (e.g., to Float64 or Int64) during Polars joins, which is
            # critical for downstream H3 operations and memory efficiency.
            pl.col(NwpColumns.H3_INDEX).cast(pl.UInt64)
        )

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def _prepare_data_for_model(
        self,
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
        inference_params: InferenceParams | None = None,
        collapse_lead_times: bool = False,
    ) -> pl.LazyFrame:
        """Prepares data for training or prediction.

        Args:
            power_time_series: Historical power flows at 30m resolution.
            time_series_metadata: Time series metadata.
            nwps: Dictionary of NWP data.
            inference_params: Inference parameters (only for prediction).
            collapse_lead_times: Whether to collapse lead times (only for prediction).

        Returns:
            Prepared LazyFrame ready for feature extraction.
        """
        metadata_lf = time_series_metadata.select(
            ["time_series_id", "h3_res_5", "substation_number"]
        ).lazy()

        # Calculate peak capacity
        peak_capacity_df = calculate_peak_capacity(power_time_series)
        peak_capacity_lf = peak_capacity_df.lazy()

        # 1. Prepare NWPs and join
        is_training = inference_params is None

        combined_nwps_lf = None
        if nwps:
            # We only support a single NWP source now.
            nwp_lf = list(nwps.values())[0]
            nwp_cutoff = inference_params.forecast_time if inference_params is not None else None
            combined_nwps_lf = self._prepare_and_join_nwps(
                nwp_lf,
                nwp_cutoff=nwp_cutoff,
                collapse_lead_times=collapse_lead_times,
                is_training=is_training,
            )

        if is_training:
            df_lf = self._prepare_training_data(
                power_time_series,
                metadata_lf,
                combined_nwps_lf,
            )
        else:
            if combined_nwps_lf is None:
                raise ValueError("XGBoostForecaster requires NWP data for prediction.")
            df_lf = self._prepare_inference_data(metadata_lf, combined_nwps_lf)

        # 3. Add lags and features
        # Handle missing init_time (e.g. for autoregressive-only models)
        if NwpColumns.INIT_TIME not in df_lf.collect_schema().names():
            df_lf = df_lf.with_columns(pl.col(NwpColumns.VALID_TIME).alias(NwpColumns.INIT_TIME))

        # Get telemetry delay from config, default to 24 hours if not set
        telemetry_delay_hours = self.config.telemetry_delay_hours if hasattr(self, "config") else 24
        df_lf = add_autoregressive_lags(
            df_lf,
            power_time_series,
            telemetry_delay_hours=telemetry_delay_hours,
        )

        # Normalize by peak capacity
        df_lf = df_lf.join(
            peak_capacity_lf.select(["time_series_id", "peak_capacity"]),
            on="time_series_id",
            how="inner",  # FIX: Drop substations missing from peak_capacity
        ).with_columns(
            latest_available_weekly_power_lag=pl.col("latest_available_weekly_power_lag")
            / pl.col("peak_capacity")
        )

        if is_training:
            # For training, also normalize target
            df_lf = df_lf.with_columns(power=pl.col("power") / pl.col("peak_capacity"))
        else:
            # For prediction, add dummy target for validation
            df_lf = df_lf.with_columns(power=pl.lit(0.0, dtype=pl.Float32))

        df_lf = add_cyclical_temporal_features(df_lf, time_col=NwpColumns.VALID_TIME)

        # 4. Type casting
        if NwpColumns.ENSEMBLE_MEMBER in df_lf.collect_schema().names():
            df_lf = df_lf.with_columns(
                pl.col(NwpColumns.ENSEMBLE_MEMBER).fill_null(0).cast(pl.UInt8)
            )

        # DATA TYPE RATIONALE:
        # Weather features are kept as Float32 in memory rather than UInt8 to:
        # 1. Preserve precision from 30-minute interpolation (avoiding "staircase" effects).
        # 2. Prevent silent underflow during feature engineering (e.g., calculating trends
        #    via subtraction).
        # 3. Align with XGBoost's native internal data type (Float32).

        # Cast all floats to Float32 for Patito
        df_lf = df_lf.with_columns(pl.col(pl.Float64).cast(pl.Float32))

        return df_lf

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def train(
        self,
        config: ModelConfig,
        power_time_series: pl.LazyFrame,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
    ) -> "XGBoostForecaster":
        """Train the XGBoost model.

        Args:
            config: The model configuration object.
            power_time_series: Historical power flow data downsampled to 30m.
            time_series_metadata: The time series metadata containing h3 mapping.
            nwps: A dictionary of weather forecast dataframes.

        Returns:
            The trained XGBoostForecaster instance.
        """
        self.config = config
        log.info("Starting XGBoost training...")

        # Log input data info
        # Note: We don't collect the full LazyFrames here to avoid OOM,
        # just logging their presence and schema.
        log.info(f"Input power_time_series columns: {power_time_series.collect_schema().names()}")
        # Log the range of timestamps in power_time_series
        log.info(
            f"power_time_series range: {cast(pl.DataFrame, power_time_series.select(pl.col('period_end_time').min().alias('min'), pl.col('period_end_time').max().alias('max')).collect())}"
        )
        if nwps:
            for name, lf in nwps.items():
                log.info(f"Input NWP {name.value} columns: {lf.collect_schema().names()}")
                # Log the range of valid_time in NWP
                log.info(
                    f"NWP range: {cast(pl.DataFrame, lf.select(pl.col(NwpColumns.VALID_TIME).min().alias('min'), pl.col(NwpColumns.VALID_TIME).max().alias('max')).collect())}"
                )

        if len(config.features.nwps) > 0 and not nwps:
            raise ValueError("Model config requires NWPs, but none were provided.")

        joined_lf = self._prepare_data_for_model(
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], power_time_series),
            time_series_metadata=time_series_metadata,
            nwps=nwps,
        )

        # Prepare features and target
        feature_lf = self._prepare_features(joined_lf)
        feature_cols = feature_lf.collect_schema().names()

        # Collect only necessary columns and drop nulls
        critical_cols = ["power"]
        if nwps:
            critical_cols.extend(
                [
                    NwpColumns.TEMPERATURE_2M,
                    NwpColumns.SW_RADIATION,
                    NwpColumns.WIND_SPEED_10M,
                ]
            )

        # Apply random sampling if max_training_samples is set to prevent OOM errors
        # during collection.
        if config.max_training_samples is not None:
            log.info(f"Sampling training data to {config.max_training_samples} samples.")
            # LazyFrame doesn't have a direct sample(n=...) method, so we collect and then sample.
            # This still helps prevent OOM in XGBoost training itself, even if the collection
            # is the bottleneck.
            raw_df = cast(
                pl.DataFrame,
                joined_lf.select(list(set(feature_cols + ["power"]))).collect(),
            ).sample(n=config.max_training_samples, seed=42)
        else:
            raw_df = cast(
                pl.DataFrame,
                joined_lf.select(list(set(feature_cols + ["power"]))).collect(),
            )
        log.info(f"Collected raw_df shape before dropping nulls: {raw_df.shape}")
        log.info(f"Null counts: {raw_df.null_count()}")
        joined_df = raw_df.drop_nulls(subset=critical_cols)

        dropped_rows = len(raw_df) - len(joined_df)
        if dropped_rows > 0:
            log.warning(
                f"Dropped {dropped_rows} rows during training due to nulls in critical columns: {critical_cols}"
            )

        if joined_df.is_empty():
            raise ValueError("No training data remaining after dropping nulls in critical columns.")

        XGBoostInputFeatures.validate(
            joined_df, allow_missing_columns=True, allow_superfluous_columns=True
        )

        X = cast(pl.DataFrame, self._prepare_features(joined_df))
        y = joined_df.get_column("power")

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

    # TODO: Use Patito data contracts for inputs and outputs of this function.
    def predict(
        self,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        inference_params: InferenceParams,
        power_time_series: pl.LazyFrame,
        nwps: Mapping[NwpModel, pl.LazyFrame] | None = None,
        collapse_lead_times: bool = False,
    ) -> pt.DataFrame[PowerForecast]:
        """Execute the inference logic.

        Args:
            time_series_metadata: The time series metadata containing h3 mapping.
            inference_params: Parameters for inference.
            power_time_series: Historical power flow data downsampled to 30m (for lags).
            nwps: A dictionary of weather forecast lazyframes.
            collapse_lead_times: Whether to collapse lead times to simulate real-time inference by keeping only the latest available NWP forecast for each valid time.

        Returns:
            A Patito DataFrame containing the predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before calling predict.")

        if not nwps:
            raise ValueError("XGBoostForecaster requires NWP data for prediction.")

        df_lf = self._prepare_data_for_model(
            power_time_series=cast(pt.LazyFrame[PowerTimeSeries], power_time_series),
            time_series_metadata=time_series_metadata,
            nwps=nwps,
            inference_params=inference_params,
            collapse_lead_times=collapse_lead_times,
        )

        feature_lf = self._prepare_features(df_lf)
        feature_cols = feature_lf.collect_schema().names()

        # Output columns needed for the final result
        output_cols = [
            NwpColumns.VALID_TIME,
            "time_series_id",
            NwpColumns.ENSEMBLE_MEMBER,
            NwpColumns.INIT_TIME,
            "nwp_init_hour",
            "lead_time_hours",
        ]

        # FIX: Remove drop_nulls logic during prediction
        df = cast(
            pl.DataFrame,
            df_lf.select(
                list(set(feature_cols + output_cols + ["power", "peak_capacity"]))
            ).collect(),
        )

        if df.is_empty():
            raise ValueError("No inference data remaining.")

        XGBoostInputFeatures.validate(
            df, allow_missing_columns=True, allow_superfluous_columns=True
        )

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
                power_fcst=pl.Series(values=preds, dtype=pl.Float32),
            )
            .with_columns(
                power_fcst=pl.col("power_fcst") * pl.col("peak_capacity"),
                power_fcst_model_name=pl.lit(model_name).cast(pl.Categorical),
                power_fcst_init_time=pl.lit(fcst_init_time).cast(pl.Datetime("us", "UTC")),
                nwp_init_time=pl.col(NwpColumns.INIT_TIME).cast(pl.Datetime("us", "UTC")),
                power_fcst_init_year_month=pl.lit(fcst_init_time.strftime("%Y-%m")).cast(pl.String),
            )
            .select(
                [
                    NwpColumns.VALID_TIME,
                    "time_series_id",
                    NwpColumns.ENSEMBLE_MEMBER,
                    "power_fcst_model_name",
                    "power_fcst_init_time",
                    "nwp_init_time",
                    "power_fcst_init_year_month",
                    "nwp_init_hour",
                    "lead_time_hours",
                    "power_fcst",
                ]
            )
        )

        # Handle potential nulls in ensemble_member (required by schema)
        res = res.with_columns(pl.col(NwpColumns.ENSEMBLE_MEMBER).fill_null(0).cast(pl.UInt8))

        return PowerForecast.validate(res)
