from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import pytest

from xgboost_forecaster import XGBoostConfig, XGBoostForecaster

_UTC = pl.Datetime("us", "UTC")
_BASE_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
_FEATURES: set[str] = {"local_time_of_day_sin", "local_time_of_day_cos"}


def _make_df(
    n_per_ts: int = 48, ts_ids: list[int] | None = None, ens_member: int = 0
) -> pl.DataFrame:
    """Minimal AllFeatures-compatible DataFrame for testing train/predict."""
    if ts_ids is None:
        ts_ids = [1, 2]
    init_time = _BASE_TIME - timedelta(hours=6)
    half_hours = [_BASE_TIME + timedelta(minutes=30 * i) for i in range(n_per_ts)]
    rows = []
    for ts_id in ts_ids:
        for i, vt in enumerate(half_hours):
            rows.append(
                {
                    "valid_time": vt,
                    "time_series_id": ts_id,
                    "ensemble_member": ens_member,
                    "power_fcst_init_time": init_time,
                    "nwp_init_time": init_time,
                    "power": float(ts_id * 5 + i % 10),
                    "local_time_of_day_sin": float(np.sin(2 * np.pi * i / n_per_ts)),
                    "local_time_of_day_cos": float(np.cos(2 * np.pi * i / n_per_ts)),
                }
            )
    return pl.DataFrame(rows).with_columns(
        pl.col("valid_time").cast(_UTC),
        pl.col("power_fcst_init_time").cast(_UTC),
        pl.col("nwp_init_time").cast(_UTC),
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("ensemble_member").cast(pl.UInt8),
        pl.col("power").cast(pl.Float32),
        pl.col("local_time_of_day_sin").cast(pl.Float32),
        pl.col("local_time_of_day_cos").cast(pl.Float32),
    )


def _make_config(**overrides) -> XGBoostConfig:
    defaults: dict = {
        "selected_features": _FEATURES,
        "power_fcst_model_name": "test_xgboost",
        "power_fcst_model_version": 1,
        "ml_flow_experiment_id": None,
        "n_estimators": 5,  # tiny for fast tests
    }
    return XGBoostConfig(**{**defaults, **overrides})


def _trained(df: pl.DataFrame, **config_overrides) -> XGBoostForecaster:
    lf = pt.LazyFrame.from_existing(df.lazy())
    forecaster = XGBoostForecaster(_make_config(**config_overrides))
    forecaster.train(lf)
    return forecaster


def test_train_creates_one_booster_per_ts_id() -> None:
    df = _make_df(ts_ids=[1, 2, 3])
    forecaster = _trained(df)
    assert set(forecaster._models.keys()) == {1, 2, 3}


def test_predict_output_schema() -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    result = _trained(df).predict(lf)

    assert len(result) == len(df)
    assert result["power_fcst"].dtype == pl.Float32
    assert result["power_fcst_model_name"].dtype == pl.Categorical
    assert result["power_fcst_model_version"].dtype == pl.Int16
    # ml_flow_experiment_id=None → every row is null
    assert result["ml_flow_experiment_id"].is_null().all()


def test_predict_with_mlflow_experiment_id() -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    result = _trained(df, ml_flow_experiment_id=42).predict(lf)
    assert (result["ml_flow_experiment_id"] == 42).all()


def test_predict_raises_for_unseen_ts_id() -> None:
    df = _make_df(ts_ids=[1, 2])
    forecaster = _trained(df)

    unknown = df.with_columns(pl.lit(999, dtype=pl.Int32).alias("time_series_id"))
    with pytest.raises(KeyError, match="999"):
        forecaster.predict(pt.LazyFrame.from_existing(unknown.lazy()))


def test_save_load_roundtrip(tmp_path: Path) -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    forecaster = _trained(df)

    save_dir = tmp_path / "model"
    forecaster.save(save_dir)

    assert (save_dir / "meta.json").exists()
    assert (save_dir / "1.ubj").exists()
    assert (save_dir / "2.ubj").exists()

    loaded = XGBoostForecaster.load(save_dir)
    assert loaded.model_params == forecaster.model_params

    sort_cols = ["time_series_id", "valid_time", "ensemble_member"]
    orig = forecaster.predict(lf).sort(sort_cols)
    restored = loaded.predict(lf).sort(sort_cols)
    assert orig["power_fcst"].to_list() == pytest.approx(restored["power_fcst"].to_list())


def test_save_creates_one_ubj_per_ts_id(tmp_path: Path) -> None:
    df = _make_df(ts_ids=[10, 20, 30])
    forecaster = _trained(df)
    forecaster.save(tmp_path / "m")
    ubj_stems = {int(f.stem) for f in (tmp_path / "m").glob("*.ubj")}
    assert ubj_stems == {10, 20, 30}


def test_selected_features_round_trips_through_save_load(tmp_path: Path) -> None:
    extra_features = {"local_time_of_day_sin", "local_time_of_day_cos"}
    df = _make_df()
    forecaster = _trained(df, selected_features=extra_features)
    forecaster.save(tmp_path / "m")
    loaded = XGBoostForecaster.load(tmp_path / "m")
    assert loaded.model_params.selected_features == extra_features
