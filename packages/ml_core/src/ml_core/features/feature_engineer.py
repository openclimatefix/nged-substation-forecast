"""Abstract base for pluggable feature-engineering strategies.

A ``FeatureEngineer`` turns the raw inputs (observed power, gridded NWP, time-series metadata)
into the model-ready frame a forecaster consumes. It is a strategy object **referenced** by a
forecaster (composition), not a method **implemented** on it — so the forecaster keeps its single
responsibility (train/predict) and a new model can swap the whole feature pipeline by pointing at
a different ``FeatureEngineer``.
"""

from abc import ABC, abstractmethod

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpInMemory


class FeatureEngineer(ABC):
    """Turns raw power + NWP + metadata into a model-ready ``AllFeatures`` frame."""

    @abstractmethod
    def engineer(
        self,
        *,
        selected_features: set[str],
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpInMemory],
    ) -> pt.LazyFrame[AllFeatures]:
        """Engineer features for training/inference.

        Args:
            selected_features: The feature names to produce.
            power_time_series: Observed power, one row per ``(time_series_id, time)``.
            time_series_metadata: Per-time-series metadata (carries ``h3_res_5`` for the
                spatial mapping).
            nwp: Gridded NWP in physical units, keyed by ``h3_index``.

        Returns:
            A lazy ``AllFeatures`` frame, ready to hand to ``BaseForecaster.train``/``predict``.
        """
