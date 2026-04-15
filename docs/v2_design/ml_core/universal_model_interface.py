from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from abc import ABC, abstractmethod

import patito as pt
import polars as pl

from .features import (
    STATIC_FEATURE_REGISTRY,
    AllFeatures,
    RawData,
    build_lag_expr,
    build_rolling_mean_expr,
)


class MLModel(ABC):
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
        self._train_algo(self._engineer_features(data))

    def predict(
        self, 
        power_time_series: pt.LazyFrame[PowerTimeSeries], 
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata], 
        nwp: pt.LazyFrame[NwpOnDisk] | None,
    ) -> pt.DataFrame[PowerForecast]:
        return self._predict_algo(self._engineer_features(data))

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

        exprs_to_evaluate: list[pl.Expr] = []

        for feature_name in self.selected_features:
            # 1. Check static registry
            if feature_name in STATIC_FEATURE_REGISTRY:
                exprs_to_evaluate.append(STATIC_FEATURE_REGISTRY[feature_name])

            # 2. Parse parameterized features (e.g., "power_lag_24")
            elif feature_name.startswith("power_lag_"):
                lag_val = int(feature_name.split("_")[-1])
                exprs_to_evaluate.append(build_lag_expr("power", lag_val))

            elif feature_name.startswith("temperature_rolling_mean_"):
                window = int(feature_name.split("_")[-1])
                exprs_to_evaluate.append(build_rolling_mean_expr("temperature", window))

            else:
                raise ValueError(f"Unknown feature requested: {feature_name}")

        # Apply all expressions simultaneously! Polars will optimize this beautifully.
        raw_data_with_engineered_features = raw_data.with_columns(exprs_to_evaluate)

        # TODO: Lazily validate the *schema* of raw_data_with_engineered_features.
        # We can't materialized the data yet, so we can't call AllFeatures.validate() yet.

        # Assert all requested features were actually created (Catch silent failures)
        missing_cols = set(self.selected_features) - set(raw_data_with_engineered_features.columns)
        if missing_cols:
            raise ValueError(f"Feature engineering failed to create: {missing_cols}")

        # We can't validate yet because we need to keep this lazy (so, for example,
        # we can train batch-by-batch.) Validation must be done after the LazyFrame is
        # materialized into RAM.
        return pt.LazyFrame[AllFeatures](raw_data_with_engineered_features)
