"""Pluggable feature-engineering strategies, associated with a ``BaseForecaster``.

A ``FeatureEngineer`` turns the raw inputs (observed power, gridded NWP, time-series metadata)
into the model-ready frame a forecaster consumes. It is a strategy object **referenced** by a
forecaster (composition), not a method **implemented** on it — so the forecaster keeps its single
responsibility (train/predict) and a new model can swap the whole feature pipeline by pointing at
a different ``FeatureEngineer``.

The default ``TabularNwpFeatureEngineer`` produces the tabular ``AllFeatures`` frame used by
``XGBoostForecaster``: it first maps each gridded-NWP H3 cell to the time series sitting in it
(nearest-cell spatial join), then delegates to the declarative ``engineer_features`` pipeline.
A future non-tabular model (e.g. a CNN wanting a spatial NWP *crop* per time series) would
implement its own ``FeatureEngineer`` and pair it with a matching ``train``/``predict``.
"""

from abc import ABC, abstractmethod

import patito as pt
import polars as pl
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerTimeSeries, TimeSeriesMetadata
from contracts.weather_schemas import NwpInMemory

from ml_core.features import engineer_features


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


def _attach_nearest_nwp_cell(
    nwp: pt.LazyFrame[NwpInMemory],
    time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
) -> pl.LazyFrame:
    """Map each gridded-NWP H3 cell to the time series that sits in it (nearest-cell join).

    NWP is stored per H3 cell at resolution 5; each time series carries its own resolution-5 cell
    in ``h3_res_5``. Joining ``h3_index == h3_res_5`` gives every time series the weather of its
    containing cell. The inner join drops NWP cells with no time series and replicates a cell
    shared by several series across each of them — both correct. The result is keyed by
    ``time_series_id`` (not ``h3_index``), which is what ``engineer_features`` expects.
    """
    # Strip the Patito subclasses so Polars' cross-subclass join type check doesn't reject the
    # join (see CLAUDE.md). Zero-copy: same underlying Rust LazyFrames.
    nwp_plain = pl.LazyFrame._from_pyldf(nwp._ldf)
    cell_to_ts = pl.LazyFrame._from_pyldf(time_series_metadata.lazy()._ldf).select(
        "time_series_id", "h3_res_5"
    )
    return nwp_plain.join(cell_to_ts, left_on="h3_index", right_on="h3_res_5", how="inner").drop(
        "h3_index"
    )


class TabularNwpFeatureEngineer(FeatureEngineer):
    """Nearest res-5 NWP cell per time series, then the declarative tabular pipeline."""

    def engineer(
        self,
        *,
        selected_features: set[str],
        power_time_series: pt.LazyFrame[PowerTimeSeries],
        time_series_metadata: pt.DataFrame[TimeSeriesMetadata],
        nwp: pt.LazyFrame[NwpInMemory],
    ) -> pt.LazyFrame[AllFeatures]:
        nwp_per_time_series = _attach_nearest_nwp_cell(nwp, time_series_metadata)
        return engineer_features(
            selected_features,
            power_time_series,
            time_series_metadata,
            nwp=pt.LazyFrame.from_existing(nwp_per_time_series).set_model(NwpInMemory),
        )
