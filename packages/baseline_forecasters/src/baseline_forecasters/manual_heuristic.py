"""The manual heuristic: an ensemble of observed powers at the same weekday and time of day.

A distribution network operator forecasts a target time by reading the power observed at the same
weekday and time of day on earlier weeks. This module expresses each of those observations as a
power lag, so the feature pipeline that every other forecaster uses builds the observations, and
nulls a lag that would not yet have been observed when the forecast was issued.
"""

from pathlib import Path
from typing import ClassVar, Self

import patito as pt
import polars as pl
from contracts.common import UTC_DATETIME_DTYPE
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig
from ml_core.features import FeatureEngineer
from ml_core.features._parsed_features import ParsedFeatures

from baseline_forecasters._saved_model import clear_directory_and_write_meta, read_meta
from baseline_forecasters.nwp_run_rows import NwpRunRowsWithoutWeatherFeatureEngineer


class ManualHeuristicForecaster(BaseForecaster):
    """Forecasts each target time as an ensemble of earlier observed powers at that time of day.

    Each selected power lag is one ensemble member. The member index is the lag's rank among the
    selected power lags, shortest lag first. Under ``conf/model/manual_heuristic.yaml``, members 0
    to 5 are the weekly analogues from the last 6 weeks and members 6 to 12 are the annual
    analogues from 49 to 55 weeks back. A lag no longer than the forecast lead time is null, so
    the member is absent from that row. A member keeps the same index on every row that carries
    the member. A row with no member at all is dropped. ``nwp_init_time`` is null on every row,
    because the forecaster consumes no weather.

    The forecaster learns nothing. ``train`` records which series have observed power, and
    ``save`` writes ``meta.json`` alone.
    """

    MODEL_NAME = "manual_heuristic"
    MODEL_VERSION = 1
    CONFIG_CLASS: ClassVar[type[BaseForecasterConfig]] = BaseForecasterConfig

    feature_engineer: ClassVar[FeatureEngineer] = NwpRunRowsWithoutWeatherFeatureEngineer()

    def __init__(self, model_params: BaseForecasterConfig) -> None:
        """Parse the power lags that become ensemble members.

        Args:
            model_params: The config, whose ``selected_features`` names the power lags.

        Raises:
            ValueError: ``model_params.selected_features`` holds no power lag. ``predict`` would
                otherwise return zero rows for every input, with no error.
        """
        super().__init__(model_params)
        power_lags = [
            lag
            for lag in ParsedFeatures.from_strings(model_params.selected_features).lags
            if lag.base_col == "power"
        ]
        if not power_lags:
            raise ValueError(
                "ManualHeuristicForecaster needs at least one power lag in selected_features, "
                f"such as power_lag_168h, but got {sorted(model_params.selected_features)}."
            )
        self._lag_hours: list[int] = sorted(lag.hours for lag in power_lags)
        self._lag_column_to_member: dict[str, int] = {
            f"power_lag_{hours}h": member for member, hours in enumerate(self._lag_hours)
        }
        self._trained_ids: list[int] = []

    @property
    def trained_time_series_ids(self) -> list[int]:
        """The sorted ``time_series_id``s that had at least one non-null ``power`` in training."""
        return self._trained_ids

    def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None:
        """Record the requested series that have at least one non-null ``power`` row.

        Nothing is fitted. The rule mirrors ``XGBoostForecaster``: a series with no usable rows is
        not trained, so train and predict cover the same population.
        """
        requested = set(time_series_ids)
        lazy: pl.LazyFrame = data
        observed = (
            lazy.select("time_series_id", "power")
            .drop_nulls(subset=["power"])
            .select("time_series_id")
            .unique()
            .collect(engine="streaming")
        )
        self._trained_ids = sorted(
            time_series_id
            for time_series_id in observed["time_series_id"].to_list()
            if time_series_id in requested
        )

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        """Unpivot the power-lag columns into one ensemble-member row per available lag.

        The loaders already restrict ``data`` to ``trained_time_series_ids``, and the engineer emits
        one row per series, run, and valid time, so nothing is filtered or deduplicated here. A
        null lag value comes from a lag that ``_nullify_leaky_lags`` shed or from history that does
        not reach back far enough. The member row carrying a null lag value is dropped without an
        error. Empty input gives an empty frame.

        Args:
            data: Features engineered by ``NwpRunRowsWithoutWeatherFeatureEngineer``.
            fold_id: The value stamped onto every row's ``fold_id`` column.

        Returns:
            One row per ``(time_series_id, power_fcst_init_time, valid_time, ensemble_member)``.
        """
        config = self.model_params
        features = data.collect(engine="streaming").as_polars()
        members = (
            features.unpivot(
                on=list(self._lag_column_to_member),
                index=["time_series_id", "valid_time", "power_fcst_init_time"],
                variable_name="lag_column",
                value_name="lag_value",
            )
            .drop_nulls(subset=["lag_value"])
            .select(
                "valid_time",
                "time_series_id",
                "power_fcst_init_time",
                ensemble_member=pl.col("lag_column").replace_strict(
                    self._lag_column_to_member, return_dtype=pl.Int8
                ),
                power_fcst=pl.col("lag_value").cast(pl.Float32),
                nwp_init_time=pl.lit(None, dtype=UTC_DATETIME_DTYPE),
                power_fcst_model_name=pl.lit(self.MODEL_NAME),
                power_fcst_model_version=pl.lit(self.MODEL_VERSION, dtype=pl.Int16),
                ml_flow_experiment_id=pl.lit(config.ml_flow_experiment_id, dtype=pl.Int32),
                experiment_name=pl.lit(config.experiment_name),
                fold_id=pl.lit(fold_id),
            )
        )
        return PowerForecast.validate(members)

    def save(self, path: Path) -> None:
        """Replace ``path`` with a ``meta.json`` holding the config and the trained population."""
        clear_directory_and_write_meta(path=path, forecaster=self)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reconstruct a ManualHeuristicForecaster from the ``meta.json`` that ``save`` wrote."""
        meta = read_meta(path)
        instance = cls(cls.CONFIG_CLASS.model_validate(meta["model_params"]))
        instance._trained_ids = meta["trained_time_series_ids"]
        return instance
