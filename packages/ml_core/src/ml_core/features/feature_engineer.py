"""Abstract base for pluggable feature-engineering strategies.

A ``FeatureEngineer`` turns the raw inputs (observed power, gridded NWP, time-series metadata)
into the model-ready frame a forecaster consumes. It is a strategy object **referenced** by a
forecaster (composition), not a method **implemented** on it — so the forecaster keeps its single
responsibility (train/predict) and a new model can swap the whole feature pipeline by pointing at
a different ``FeatureEngineer``.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Final

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp

from ml_core.features._nwp import NWP_PUBLICATION_DELAY_HOURS

DEFAULT_LOCAL_TIMEZONE: Final[str] = "Europe/London"
"""IANA zone the local-time features (time of day, day of week, UTC offset) are computed in.

Defined here, on the interface, rather than inside ``TabularFeatureEngineer``, so a future
``FeatureEngineer`` forecasting another region can override it through the same ``engineer()``
call every production call site already uses — not just through the one implementation.
"""


class FeatureEngineer(ABC):
    """Turns raw power + NWP + metadata into a model-ready ``AllFeatures`` frame."""

    @abstractmethod
    def engineer(
        self,
        *,
        selected_features: set[str],
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[Nwp],
        power_fcst_init_time: datetime | None = None,
        nwp_init_time: datetime | None = None,
        nwp_publication_delay_hours: int = NWP_PUBLICATION_DELAY_HOURS,
        local_timezone: str = DEFAULT_LOCAL_TIMEZONE,
    ) -> pt.LazyFrame[AllFeatures]:
        """Engineer features for training/inference.

        Two operating modes, matching ``tabular_feature_engineer._engineer_features`` (see
        that function for the full detail this docstring summarises):

        - **Bulk mode** (``power_fcst_init_time=None``, the default): NWP-centric, vectorised
          over every NWP run in the input, one forecast per ``(nwp_init_time, valid_time)``
          pair. Used for training and multi-run backtesting.
        - **Single-run mode** (``power_fcst_init_time`` given): power-centric, joins exactly
          one NWP run (``nwp_init_time``, or derived from ``power_fcst_init_time`` minus
          ``nwp_publication_delay_hours`` if omitted) and stamps a constant
          ``power_fcst_init_time`` on every row. Used for production inference and replay
          backfilling.

        Args:
            selected_features: The feature names to produce.
            power_time_series: Observed power, one row per ``(time_series_id, time)``.
            time_series_metadata: Per-time-series metadata (carries ``h3_res_5`` for the
                spatial mapping).
            nwp: Gridded NWP in physical units, keyed by ``h3_index``.
            power_fcst_init_time: ``None`` for bulk mode; a single datetime for single-run
                mode. See ``_engineer_features`` for the full contract.
            nwp_init_time: The NWP run to join in single-run mode. Must be ``None`` in bulk
                mode. See ``_engineer_features``.
            nwp_publication_delay_hours: Delay used to derive whichever of
                ``power_fcst_init_time``/``nwp_init_time`` is not supplied.
            local_timezone: IANA zone the local-time features are computed in.

        Returns:
            A lazy ``AllFeatures`` frame, ready to hand to ``BaseForecaster.train``/``predict``.
        """
