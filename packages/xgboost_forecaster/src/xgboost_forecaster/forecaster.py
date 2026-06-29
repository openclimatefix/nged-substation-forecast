"""XGBoost-based power forecasting model."""

import json
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import patito as pt
import polars as pl
import xgboost as xgb
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast
from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig

from xgboost_forecaster._data_iter import LazyFrameBatchIter, _prepare_features


class XGBoostConfig(BaseForecasterConfig):
    """Configuration for XGBoostForecaster.

    Inherits the universal fields (selected_features, MLflow experiment id) from
    BaseForecasterConfig and adds XGBoost-specific hyperparameters.
    """

    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: int = 1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    device: str = "cpu"
    objective: str = "reg:squarederror"
    train_batch_size: int = 100_000
    """Rows collected per batch when streaming training data into XGBoost's QuantileDMatrix.

    Bounds peak training memory: only one batch of Float32 features is resident at a time, while
    QuantileDMatrix keeps the rest as compressed 8-bit quantile bins. Not an XGBoost
    hyperparameter, so it is excluded from to_xgb_params(); it serialises with the rest of the
    config and so round-trips through save/load/MLflow.
    """

    def to_xgb_params(self) -> dict[str, Any]:
        """Return the params dict accepted by xgb.train()."""
        return {
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "device": self.device,
            "objective": self.objective,
            "seed": self.random_seed,
        }


class XGBoostForecaster(BaseForecaster):
    """Trains and serves one XGBoost Booster per time_series_id.

    All lead times for a given time_series_id are handled by a single Booster. Ensemble
    forecasts are produced by passing each NWP ensemble member through the relevant Booster
    independently; the AllFeatures input to predict() should already be filtered to a single
    ensemble member per call, or iterated over ensemble members by the caller.

    Save layout: a directory containing one ``{time_series_id}.ubj`` file per trained
    Booster plus a ``meta.json`` that stores the full XGBoostConfig so that load() is
    self-contained.
    """

    MODEL_NAME = "xgboost"
    MODEL_VERSION = 1

    model_params: XGBoostConfig  # narrows the base class annotation for type checkers

    def __init__(self, model_params: XGBoostConfig) -> None:
        super().__init__(model_params)
        self._models: dict[int, xgb.Booster] = {}

    @property
    def _feature_cols(self) -> list[str]:
        return sorted(self.model_params.selected_features)

    @property
    def trained_time_series_ids(self) -> list[int]:
        """The sorted ``time_series_id``s this forecaster will serve a ``predict`` for.

        For ``XGBoostForecaster`` this is exactly the set of series it holds a trained Booster for
        (one Booster per ``time_series_id``), so ``predict`` raises ``KeyError`` if asked for any
        other series. ``save()`` records this set in ``meta.json`` and ``load()`` reconstructs it
        from the ``.ubj`` files on disk. See ``BaseForecaster.trained_time_series_ids`` for the
        model-agnostic contract this implements (the train==predict population invariant).
        """
        return sorted(self._models.keys())

    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        """Fit one Booster per time_series_id found in data.

        Each series' rows are streamed into an ``xgb.QuantileDMatrix`` one batch at a time via
        ``LazyFrameBatchIter`` (batch size from ``train_batch_size``), so the full Float32 dataset
        is never resident and XGBoost never builds an uncompressed ``DMatrix`` copy. The lazy plan
        is never collected whole — only the id column (to learn the population) and one row-batch
        at a time.
        """
        feature_cols = self._feature_cols
        batch_size = self.model_params.train_batch_size

        # One cheap collect of just the id column yields the non-empty population (a series with no
        # non-null power rows simply does not appear, so it gets no Booster).
        counts = data.drop_nulls(subset=["power"]).group_by("time_series_id").len().collect()

        for ts_id in counts["time_series_id"]:
            per_ts = data.filter(pl.col("time_series_id") == ts_id).drop_nulls(subset=["power"])
            data_iter = LazyFrameBatchIter(per_ts, feature_cols, batch_size=batch_size)
            dtrain = xgb.QuantileDMatrix(data_iter)
            booster = xgb.train(
                self.model_params.to_xgb_params(),
                dtrain,
                num_boost_round=self.model_params.n_estimators,
            )
            self._models[int(ts_id)] = booster

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        """Generate one power_fcst per row, dispatching by time_series_id to the right Booster.

        ``fold_id`` is stamped onto every output row (the model has no inherent fold; the caller
        supplies it). Defaults to the ``"live"`` production sentinel.
        """
        df = data.collect()
        feature_cols = self._feature_cols
        cfg = self.model_params
        parts: list[pl.DataFrame] = []

        for group_key, group_df in df.group_by(["time_series_id"]):
            group_df = cast(pt.DataFrame[AllFeatures], group_df)
            ts_id: int = group_key[0]
            booster = self._models.get(ts_id)
            if booster is None:
                raise KeyError(f"No trained model for time_series_id={ts_id!r}")
            X = _prepare_features(group_df, feature_cols)
            predictions: np.ndarray = booster.predict(xgb.DMatrix(X))

            part = group_df.select(
                [
                    "valid_time",
                    "time_series_id",
                    "ensemble_member",
                    "nwp_init_time",
                    "power_fcst_init_time",
                ]
            ).with_columns(
                pl.col("ensemble_member").cast(pl.Int8),
                pl.Series("power_fcst", predictions, dtype=pl.Float32),
                power_fcst_model_name=pl.lit(self.MODEL_NAME),
                power_fcst_model_version=pl.lit(self.MODEL_VERSION, dtype=pl.Int16),
                ml_flow_experiment_id=pl.lit(cfg.ml_flow_experiment_id, dtype=pl.Int32),
                experiment_name=pl.lit(cfg.experiment_name),
                fold_id=pl.lit(fold_id),
            )
            parts.append(part)

        # ``group_by`` and ``pl.concat`` preserve the input's Patito ``AllFeatures`` subclass, so a
        # Patito ``.cast`` here would ignore the mapping and instead revert every column to its
        # ``AllFeatures`` dtype (e.g. ``ensemble_member`` back to ``UInt8``) and leave the
        # forecast-only columns untouched. Strip the Patito model so the dict-cast uses plain Polars
        # semantics. ``_from_pydf`` reuses the same underlying frame (zero-copy).
        combined = pl.concat(parts)
        result = pl.DataFrame._from_pydf(combined._df).cast(
            {
                "power_fcst_model_name": pl.Categorical,
                "experiment_name": pl.Categorical,
                "fold_id": pl.Categorical,
            }
        )
        return PowerForecast.validate(result)

    def save(self, path: Path) -> None:
        """Save all Boosters as .ubj files plus a meta.json with the full config."""
        path.mkdir(parents=True, exist_ok=True)
        for ts_id, booster in self._models.items():
            booster.save_model(str(path / f"{ts_id}.ubj"))
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "model_params": self.model_params.model_dump(mode="json"),
                    "trained_time_series_ids": self.trained_time_series_ids,
                }
            )
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Reconstruct an XGBoostForecaster from a saved directory."""
        meta = json.loads((path / "meta.json").read_text())
        config = XGBoostConfig.model_validate(meta["model_params"])
        instance = cls(config)
        for ubj_file in sorted(path.glob("*.ubj")):
            booster = xgb.Booster()
            booster.load_model(str(ubj_file))
            instance._models[int(ubj_file.stem)] = booster
        return instance
