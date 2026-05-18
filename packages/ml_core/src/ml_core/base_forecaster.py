from abc import ABC, abstractmethod
from pathlib import Path

import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk

from ml_core.features import (
    STATIC_FEATURE_REGISTRY,
    apply_lag_feature,
    apply_latest_weekly_lag_feature,
    apply_local_time_features,
    apply_rolling_mean_feature,
    calculate_lead_time,
)


class BaseForecaster(ABC):
    """Defines the universal interface for all energy forecasting ML models.

    Every energy forecasting model, ranging from a simple seasonal persistence model, up to a
    sophisticated neural net, will subclass this abstract base class. This will allow us to re-use
    as much code as possible, and to minimise the amount of code that must be written for each new
    ML model.
    """

    # TODO: The type of `model_params` in each concrete class should be the Pydantic Model for
    # that ML model
    def __init__(self, selected_features: set[str], model_params: dict):
        self.selected_features = selected_features
        self.model_params = model_params

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model params. Concrete classes implement their native save logic here."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model params. Concrete classes implement their native load logic here."""
        pass

    def train(
        self,
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpOnDisk] | None,
    ) -> None:
        self._train_algo(
            self._engineer_features(
                power_time_series=power_time_series,
                time_series_metadata=time_series_metadata,
                nwp=nwp,
            )
        )

    def predict(
        self,
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpOnDisk] | None,
    ) -> pt.DataFrame[PowerForecast]:
        return self._predict_algo(
            self._engineer_features(
                power_time_series=power_time_series,
                time_series_metadata=time_series_metadata,
                nwp=nwp,
            )
        )

    @abstractmethod
    def _train_algo(self, data: pt.LazyFrame[AllFeatures]) -> None:
        pass

    @abstractmethod
    def _predict_algo(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        pass

    def _engineer_features(
        self,
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpOnDisk] | None,
    ) -> pt.LazyFrame[AllFeatures]:
        """Engineer features.

        Ensemble filtering (e.g., selecting the control member for training) must happen *before*
        _engineer_features is called (likely in Dagster or the train/predict methods).
        _engineer_features is designed to safely process whatever ensemble members it is given.
        """

        # Rename 'time' to 'valid_time' in power_time_series to match NWP and AllFeatures
        power_lf = power_time_series.rename({"time": "valid_time"})

        power_lf = self._prepare_power_lag_features(power_lf)

        nwp_lf = nwp
        if nwp_lf is not None:
            nwp_lf = self._prepare_weather_features(nwp_lf)

        raw_data = self._join_data(power_lf, time_series_metadata, nwp_lf)

        engineered_lf = self._apply_post_join_features(raw_data, power_lf, nwp_lf)

        # Dynamic Schema Assertion
        # Why two-step validation?
        # Step 1 (Here): We dynamically check the Polars schema to ensure all requested features
        # (especially dynamic ones like 'power_lag_24') were actually created or exist.
        # This catches typos in the config early.
        available_columns = engineered_lf.collect_schema().names()
        missing_cols = set(self.selected_features) - set(available_columns)
        if missing_cols:
            raise ValueError(f"Feature engineering failed to create or find: {missing_cols}")

        # Select & Cast
        # Step 2: Cast to the Patito model. Patito ignores extra columns, so we explicitly select
        # the base columns and the requested features to keep the dataframe clean and memory-efficient.
        base_cols = ["valid_time", "time_series_id", "time_series_type", "power"]
        if "lead_time_hours" in engineered_lf.collect_schema().names():
            base_cols.append("lead_time_hours")
        if "ensemble_member" in engineered_lf.collect_schema().names():
            base_cols.append("ensemble_member")

        # Ensure we don't duplicate columns if a base column is also in selected_features
        cols_to_select = list(set(base_cols + list(self.selected_features)))

        final_lf = engineered_lf.select(cols_to_select)

        return pt.LazyFrame.from_existing(final_lf).set_model(AllFeatures)

    def _prepare_power_lag_features(self, power_lf: pl.LazyFrame) -> pl.LazyFrame:
        for feature_name in self.selected_features:
            if feature_name.startswith("power_lag_") and feature_name.endswith("h"):
                lag_val = int(feature_name.split("_")[-1][:-1])
                power_lf = apply_lag_feature(power_lf, "power", lag_val)
        return power_lf

    def _prepare_weather_features(self, nwp_lf: pl.LazyFrame) -> pl.LazyFrame:
        for feature_name in self.selected_features:
            if feature_name.startswith("temperature_rolling_mean_") and feature_name.endswith("h"):
                window = int(feature_name.split("_")[-1][:-1])
                nwp_lf = apply_rolling_mean_feature(nwp_lf, "temperature_2m", window)

        if (  # "temperature_2m_lag_6h" is required by "temperature_2m_trend_6h"
            "temperature_2m_6h_ago" in self.selected_features
            or "temperature_2m_trend_6h" in self.selected_features
        ):
            nwp_lf = apply_lag_feature(nwp_lf, "temperature_2m", 6)
        return nwp_lf

    def _join_data(
        self,
        power_lf: pl.LazyFrame,
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp_lf: pl.LazyFrame | None,
    ) -> pl.LazyFrame:
        raw_data = power_lf.join(time_series_metadata.lazy(), on="time_series_id", how="left")

        if nwp_lf is not None:
            # Why NWP left joins Power: NWP data often contains multiple ensemble members.
            # If we left join NWP onto Power, we lose all but one ensemble member (or get a cross join explosion if not careful).
            # By left joining Power onto NWP, we preserve all ensemble members and broadcast the power data to each member.
            # We use _from_pyldf to cast to standard pl.LazyFrame to bypass Patito's strict type checking on joins
            raw_data = pl.LazyFrame._from_pyldf(nwp_lf._ldf).join(
                pl.LazyFrame._from_pyldf(raw_data._ldf),
                on=["time_series_id", "valid_time"],
                how="left",
            )
            raw_data = calculate_lead_time(raw_data)
        return raw_data

    def _apply_post_join_features(
        self, raw_data: pl.LazyFrame, power_lf: pl.LazyFrame, nwp_lf: pl.LazyFrame | None
    ) -> pl.LazyFrame:
        engineered_lf = raw_data

        # Dynamic Lags
        if "latest_weekly_lagged_power" in self.selected_features:
            engineered_lf = apply_latest_weekly_lag_feature(
                engineered_lf, power_lf, "power", "latest_weekly_lagged_power", ["time_series_id"]
            )

        if nwp_lf is not None:
            join_cols = ["time_series_id"]
            if "ensemble_member" in nwp_lf.collect_schema().names():
                join_cols.append("ensemble_member")

            if "latest_weekly_lagged_temperature_2m" in self.selected_features:
                engineered_lf = apply_latest_weekly_lag_feature(
                    engineered_lf,
                    nwp_lf,
                    "temperature_2m",
                    "latest_weekly_lagged_temperature_2m",
                    join_cols,
                )
            if "latest_weekly_lagged_wind_speed_10m" in self.selected_features:
                engineered_lf = apply_latest_weekly_lag_feature(
                    engineered_lf,
                    nwp_lf,
                    "wind_speed_10m",
                    "latest_weekly_lagged_wind_speed_10m",
                    join_cols,
                )
            if (
                "latest_weekly_lagged_downward_short_wave_radiation_flux_surface"
                in self.selected_features
            ):
                engineered_lf = apply_latest_weekly_lag_feature(
                    engineered_lf,
                    nwp_lf,
                    "downward_short_wave_radiation_flux_surface",
                    "latest_weekly_lagged_downward_short_wave_radiation_flux_surface",
                    join_cols,
                )

        # Local Time Features
        local_time_features = {
            "local_time_of_day_sin",
            "local_time_of_day_cos",
            "local_time_of_year_sin",
            "local_time_of_year_cos",
            "local_day_of_week_sin",
            "local_day_of_week_cos",
            "local_day_of_week",
            "local_utc_offset",
        }
        if any(f in self.selected_features for f in local_time_features):
            engineered_lf = apply_local_time_features(engineered_lf)

        # Static Features
        exprs_to_evaluate: list[pl.Expr] = []
        for feature_name in self.selected_features:
            if feature_name in STATIC_FEATURE_REGISTRY:
                exprs_to_evaluate.append(STATIC_FEATURE_REGISTRY[feature_name])

        if exprs_to_evaluate:
            engineered_lf = engineered_lf.with_columns(exprs_to_evaluate)

        return engineered_lf
