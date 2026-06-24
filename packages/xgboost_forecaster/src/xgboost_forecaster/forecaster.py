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
        }


def _prepare_features(df: pt.DataFrame[AllFeatures], feature_cols: list[str]) -> pl.DataFrame:
    """Return a Float32 DataFrame containing only the feature columns.

    String, Categorical, and Enum columns are encoded as integer codes before casting,
    so XGBoost treats them as ordinal numerics. Nulls are preserved as NaN, which XGBoost
    handles natively as missing values.
    """
    exprs = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == pl.String or dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            exprs.append(pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Float32).alias(col))
        else:
            exprs.append(pl.col(col).cast(pl.Float32).alias(col))
    return df.select(exprs)


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

    def train(self, data: pt.LazyFrame[AllFeatures]) -> None:
        """Fit one Booster per time_series_id found in data."""
        df = data.collect()
        feature_cols = self._feature_cols
        for group_key, group_df in df.group_by(["time_series_id"]):
            ts_id: int = group_key[0]
            clean = cast(pt.DataFrame[AllFeatures], group_df.drop_nulls(subset=["power"]))
            if len(clean) == 0:
                continue
            X = _prepare_features(clean, feature_cols)
            y = clean["power"].cast(pl.Float32)
            dtrain = xgb.DMatrix(X, label=y)
            booster = xgb.train(
                self.model_params.to_xgb_params(),
                dtrain,
                num_boost_round=self.model_params.n_estimators,
            )
            self._models[ts_id] = booster

    def predict(self, data: pt.LazyFrame[AllFeatures]) -> pt.DataFrame[PowerForecast]:
        """Generate one power_fcst per row, dispatching by time_series_id to the right Booster."""
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
                fold_id=pl.lit("live"),
            )
            parts.append(part)

        result = pl.concat(parts).cast(
            {"power_fcst_model_name": pl.Categorical, "fold_id": pl.Categorical}
        )
        return PowerForecast.validate(result)

    def save(self, path: Path) -> None:
        """Save all Boosters as .ubj files plus a meta.json with the full config."""
        path.mkdir(parents=True, exist_ok=True)
        for ts_id, booster in self._models.items():
            booster.save_model(str(path / f"{ts_id}.ubj"))
        (path / "meta.json").write_text(
            json.dumps({"model_params": self.model_params.model_dump(mode="json")})
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
