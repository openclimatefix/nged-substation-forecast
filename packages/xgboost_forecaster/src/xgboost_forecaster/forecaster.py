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


def _prepare_features(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Return a Float32 DataFrame containing only the feature columns.

    String, Categorical, and Enum columns are encoded as integer codes before casting,
    so XGBoost treats them as ordinal numerics. Nulls are preserved as NaN, which XGBoost
    handles natively as missing values. The Patito model is stripped from the result (zero-copy)
    so XGBoost sees a plain ``pl.DataFrame``.
    """
    exprs = []
    for col in feature_cols:
        dtype = df[col].dtype
        if dtype == pl.String or dtype == pl.Categorical or isinstance(dtype, pl.Enum):
            exprs.append(pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Float32).alias(col))
        else:
            exprs.append(pl.col(col).cast(pl.Float32).alias(col))
    return pl.DataFrame._from_pydf(df.select(exprs)._df)


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
            "seed": self.random_seed,
        }


class XGBoostForecaster(BaseForecaster):
    """Trains and serves one XGBoost Booster per time_series_id.

    All lead times for a given time_series_id are handled by a single Booster. The model is
    deterministic; ensemble forecasts arise because each NWP ensemble member's weather is a separate
    row through the relevant Booster. ``predict`` scores every ensemble member present in its input
    in one call (it groups by ``time_series_id`` and dispatches each group to its Booster).

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

    def train(self, data: pt.LazyFrame[AllFeatures], time_series_ids: list[int]) -> None:
        """Fit one Booster per ``time_series_id`` in ``time_series_ids``.

        ``data`` is collected once and grouped in memory by ``time_series_id``; each group's rows
        feed an ``xgb.QuantileDMatrix`` (compressed to 8-bit quantile bins, not an uncompressed
        Float32 copy). Keeping this bounded is the *caller's* job — the NWP scan must be pruned at
        the inputs (control member, the relevant H3 cells, the window's ``init_time`` partitions),
        because filtering the engineered output cannot prune the upstream join/upsample. See
        ``_load_engineering_inputs`` and the "NWP scan pruning" notes in
        ``docs/architecture/overview.md``.

        Only the requested ``time_series_ids`` are trained; a requested series with no non-null
        ``power`` rows (e.g. none in the training window) simply does not appear and gets no Booster.
        """
        feature_cols = self._feature_cols
        requested = set(time_series_ids)
        # Stream the collect. init_time prunes whole NWP partitions, and the member-early sort
        # (delta_store.nwp.NWP_SORT_COLS) lets row-group stats skip most of each partition for the
        # control-member read — but h3_index is not a sort-early column, so cell filtering is still
        # decode-then-filter within the surviving row groups. The streaming engine applies those
        # predicates per morsel, holding peak memory to a few GB where the in-memory engine would
        # materialise every surviving row first. See docs/architecture/overview.md.
        df = data.drop_nulls(subset=["power"]).collect(engine="streaming")
        for group_key, group in df.group_by(["time_series_id"]):
            ts_id = int(group_key[0])
            if ts_id not in requested:
                continue
            group = cast(pt.DataFrame[AllFeatures], group)
            features = _prepare_features(group, feature_cols)
            label = group["power"].cast(pl.Float32)
            dtrain = xgb.QuantileDMatrix(features, label=label)
            booster = xgb.train(
                self.model_params.to_xgb_params(),
                dtrain,
                num_boost_round=self.model_params.n_estimators,
            )
            self._models[ts_id] = booster

    def predict(
        self, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        """Generate one power_fcst per row, dispatching by time_series_id to the right Booster.

        ``data`` is collected once and grouped in memory by ``time_series_id``. Rows for a
        ``time_series_id`` this model was not trained on are ignored (the model only scores its own
        trained population — see ``trained_time_series_ids``). Keeping the collect bounded is the
        caller's job: at validation every NWP ensemble member is present, so the caller engineers one
        H3 cell at a time (see ``cv_power_forecasts`` and ``docs/architecture/overview.md``).

        ``fold_id`` is stamped onto every output row (the model has no inherent fold; the caller
        supplies it). Defaults to the ``"live"`` production sentinel.
        """
        feature_cols = self._feature_cols
        cfg = self.model_params

        def _build_part(
            group_df: pt.DataFrame[AllFeatures], predictions: np.ndarray
        ) -> pl.DataFrame:
            return group_df.select(
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

        df = data.collect(engine="streaming")  # stream the NWP scan — see train() / overview.md
        parts: list[pl.DataFrame] = []
        for group_key, group in df.group_by(["time_series_id"]):
            booster = self._models.get(int(group_key[0]))
            if booster is None:
                continue  # ignore series this model was not trained on
            group = cast(pt.DataFrame[AllFeatures], group)
            X = _prepare_features(group, feature_cols)
            predictions: np.ndarray = booster.predict(xgb.DMatrix(X))
            parts.append(_build_part(group, predictions))

        if not parts:
            empty = cast(pt.DataFrame[AllFeatures], df.head(0))
            parts.append(_build_part(empty, np.empty(0, dtype=np.float32)))

        # The identity columns (power_fcst_model_name / experiment_name / fold_id) are built as
        # ``pl.lit(str)`` above, so they are already ``String`` — the dtype PowerForecast declares —
        # and need no cast before validation.
        return PowerForecast.validate(pl.concat(parts))

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
                    "model_class": f"{type(self).__module__}.{type(self).__qualname__}",
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
