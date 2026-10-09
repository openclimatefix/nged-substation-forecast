"""The feature engineer that gives a baseline the same forecast rows as every other forecaster."""

from datetime import datetime

import patito as pt
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import Nwp
from ml_core.features import NWP_PUBLICATION_DELAY_HOURS, FeatureEngineer, TabularFeatureEngineer
from ml_core.features.feature_engineer import DEFAULT_LOCAL_TIMEZONE


class NwpRunRowsWithoutWeatherFeatureEngineer(FeatureEngineer):
    """Gives one row per series, NWP run, and valid time, reading no weather value.

    A baseline has to forecast the same ``(power_fcst_init_time, valid_time)`` rows as every other
    forecaster, so that a leaderboard compares like with like. This engineer therefore reads from
    the numerical weather prediction (NWP) frame only its four key columns other than the ensemble
    member: ``nwp_model_id``, ``init_time``, ``h3_index``, and ``valid_time``. It deduplicates those
    keys and hands the key-only frame to ``TabularFeatureEngineer``. With no weather column
    present, the tabular pipeline's upsample interpolates nothing, and the half-hourly time axis,
    the hindcast filter, and the cell join all come from the tabular code. The output carries one
    row per series, run, and valid time, with ``power`` and one column per requested power lag
    (possibly none), and has no ``ensemble_member`` column.
    """

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
        """Strip ``nwp`` to its run, cell, and valid-time keys, then run the tabular pipeline.

        Args:
            selected_features: The power-lag feature names to produce.
            power_time_series: Observed power, one row per ``(time_series_id, time)``.
            time_series_metadata: Per-time-series metadata. Carries the H3 cell each series sits in.
            nwp: Gridded NWP. Only its run, cell, and valid-time keys are read.
            power_fcst_init_time: Must be ``None``, which selects bulk mode.
            nwp_init_time: Must be ``None`` in bulk mode.
            nwp_publication_delay_hours: Hours after a run's ``init_time`` before the run is
                usable, which sets each row's ``power_fcst_init_time``.
            local_timezone: IANA zone the local-time features are computed in.

        Returns:
            A lazy ``AllFeatures`` frame with one row per ``(time_series_id, power_fcst_init_time,
            valid_time)``, carrying ``power`` and one column per requested power lag.

        Raises:
            NotImplementedError: ``power_fcst_init_time`` is given. Delegating would not raise in
                this engineer, but ``predict`` would then fail on the ``PowerForecast``
                ``valid_time`` constraint, and ``live_forecasts`` would fail on ``ensemble_member``.
        """
        if power_fcst_init_time is not None:
            raise NotImplementedError(
                "NwpRunRowsWithoutWeatherFeatureEngineer supports bulk mode only. Baselines are "
                "research baselines and are not served live, so a baseline must not be promoted."
            )
        nwp_keys = pt.LazyFrame.from_existing(
            nwp.select("nwp_model_id", "init_time", "h3_index", "valid_time").unique()
        ).set_model(Nwp)
        return TabularFeatureEngineer().engineer(
            selected_features=selected_features,
            power_time_series=power_time_series,
            time_series_metadata=time_series_metadata,
            nwp=nwp_keys,
            power_fcst_init_time=power_fcst_init_time,
            nwp_init_time=nwp_init_time,
            nwp_publication_delay_hours=nwp_publication_delay_hours,
            local_timezone=local_timezone,
        )
