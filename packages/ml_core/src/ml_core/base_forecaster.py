from abc import ABC, abstractmethod
from pathlib import Path

import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpOnDisk

from ml_core.features import STATIC_FEATURE_REGISTRY, build_lag_expr, build_rolling_mean_expr


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

        # 1. Data Joining
        # Rename 'time' to 'valid_time' in power_time_series to match NWP and AllFeatures
        power_lf = power_time_series.rename({"time": "valid_time"})

        # Join power data with metadata
        raw_data = power_lf.join(time_series_metadata.lazy(), on="time_series_id", how="left")

        if nwp is not None:
            # TODO: Convert nwp to NwpInMemory.
            # TODO: Interpolate nwp to half-hourly.
            raw_data = raw_data.join(nwp, on=["time_series_id", "valid_time"], how="left")
            # Calculate lead_time_hours if nwp is provided
            if "init_time" in raw_data.collect_schema().names():
                raw_data = raw_data.with_columns(
                    lead_time_hours=(
                        (pl.col("valid_time") - pl.col("init_time")).dt.total_seconds() / 3600
                    ).cast(pl.Float32)
                )

        # 2. Feature Generation
        exprs_to_evaluate: list[pl.Expr] = []

        for feature_name in self.selected_features:
            # Check static registry
            if feature_name in STATIC_FEATURE_REGISTRY:
                exprs_to_evaluate.append(STATIC_FEATURE_REGISTRY[feature_name])

            # Parse parameterized features (e.g., "power_lag_24h")
            elif feature_name.startswith("power_lag_") and feature_name.endswith("h"):
                lag_val = int(feature_name.split("_")[-1][:-1])
                exprs_to_evaluate.append(build_lag_expr("power", lag_val))

            elif feature_name.startswith("temperature_rolling_mean_") and feature_name.endswith("h"):
                window = int(feature_name.split("_")[-1][:-1])
                exprs_to_evaluate.append(build_rolling_mean_expr("temperature_2m", window))

            else:
                # If it's not in the registry or a dynamic feature, it might be a base column.
                # We don't need to generate an expression for it, but we'll check its existence later.
                pass

        # Apply all expressions simultaneously
        if exprs_to_evaluate:
            engineered_lf = raw_data.with_columns(exprs_to_evaluate)
        else:
            engineered_lf = raw_data

        # 3. Dynamic Schema Assertion
        # Why two-step validation?
        # Step 1 (Here): We dynamically check the Polars schema to ensure all requested features
        # (especially dynamic ones like 'power_lag_24') were actually created or exist.
        # This catches typos in the config early.
        available_columns = engineered_lf.collect_schema().names()
        missing_cols = set(self.selected_features) - set(available_columns)
        if missing_cols:
            raise ValueError(f"Feature engineering failed to create or find: {missing_cols}")

        # 4. Select & Cast
        # Step 2: Cast to the Patito model. Patito ignores extra columns, so we explicitly select
        # the base columns and the requested features to keep the dataframe clean and memory-efficient.
        base_cols = ["valid_time", "time_series_id", "time_series_type", "power"]
        if "lead_time_hours" in engineered_lf.collect_schema().names():
            base_cols.append("lead_time_hours")

        # Ensure we don't duplicate columns if a base column is also in selected_features
        cols_to_select = list(set(base_cols + list(self.selected_features)))

        final_lf = engineered_lf.select(cols_to_select)

        return pt.LazyFrame.from_existing(final_lf).set_model(AllFeatures)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model params. Concrete classes implement their native save logic here."""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model params. Concrete classes implement their native load logic here."""
        pass
