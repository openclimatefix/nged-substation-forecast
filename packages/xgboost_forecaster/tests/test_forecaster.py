import json
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
        "ml_flow_experiment_id": None,
        "n_estimators": 5,  # tiny for fast tests
    }
    return XGBoostConfig(**{**defaults, **overrides})


def _ts_ids(df: pl.DataFrame) -> list[int]:
    return df["time_series_id"].unique().sort().to_list()


def _trained(df: pl.DataFrame, **config_overrides) -> XGBoostForecaster:
    lf = pt.LazyFrame.from_existing(df.lazy())
    forecaster = XGBoostForecaster(_make_config(**config_overrides))
    forecaster.train(lf, _ts_ids(df))
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
    assert result["power_fcst_model_name"].dtype == pl.String
    assert result["power_fcst_model_version"].dtype == pl.Int16
    # ml_flow_experiment_id=None → every row is null
    assert result["ml_flow_experiment_id"].is_null().all()


def test_predict_with_mlflow_experiment_id() -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    result = _trained(df, ml_flow_experiment_id=42).predict(lf)
    assert (result["ml_flow_experiment_id"] == 42).all()


def test_predict_ignores_untrained_ts_ids() -> None:
    """predict scores only its trained population; rows for other series are ignored."""
    df = _make_df(ts_ids=[1, 2])
    forecaster = _trained(df)

    mixed = pl.concat([df, _make_df(ts_ids=[999])])  # 999 was never trained
    result = forecaster.predict(pt.LazyFrame.from_existing(mixed.lazy()))

    assert set(result["time_series_id"].unique().to_list()) == {1, 2}
    assert len(result) == len(df)


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


def test_predict_stamps_experiment_name() -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    result = _trained(df, experiment_name="my_experiment").predict(lf)
    assert result["experiment_name"].dtype == pl.String
    assert (result["experiment_name"] == "my_experiment").all()


def test_predict_defaults_fold_id_to_live() -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    result = _trained(df).predict(lf)
    assert (result["fold_id"] == "live").all()


def test_predict_stamps_supplied_fold_id() -> None:
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    result = _trained(df).predict(lf, fold_id="2024")
    assert result["fold_id"].dtype == pl.String
    assert (result["fold_id"] == "2024").all()


def test_save_records_trained_time_series_ids(tmp_path: Path) -> None:
    df = _make_df(ts_ids=[10, 20, 30])
    forecaster = _trained(df)
    assert forecaster.trained_time_series_ids == [10, 20, 30]

    forecaster.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "meta.json").read_text())
    assert meta["trained_time_series_ids"] == [10, 20, 30]


def test_random_seed_makes_training_deterministic() -> None:
    """Two models trained with the same random_seed produce identical predictions."""
    df = _make_df()
    lf = pt.LazyFrame.from_existing(df.lazy())
    sort_cols = ["time_series_id", "valid_time", "ensemble_member"]

    a = _trained(df, random_seed=7, subsample=0.5, colsample_bytree=0.5).predict(lf).sort(sort_cols)
    b = _trained(df, random_seed=7, subsample=0.5, colsample_bytree=0.5).predict(lf).sort(sort_cols)
    assert a["power_fcst"].to_list() == b["power_fcst"].to_list()


def test_training_skips_series_with_all_null_power() -> None:
    """A series with no non-null power rows gets no Booster."""
    df = _make_df(ts_ids=[1, 2])
    df = df.with_columns(
        power=pl.when(pl.col("time_series_id") == 2)
        .then(None)
        .otherwise(pl.col("power"))
        .cast(pl.Float32)
    )
    forecaster = _trained(df)
    assert forecaster.trained_time_series_ids == [1]


def test_training_skips_requested_id_absent_from_data() -> None:
    """An id in time_series_ids but with no rows in data is silently skipped."""
    df = _make_df(ts_ids=[1, 2])
    lf = pt.LazyFrame.from_existing(df.lazy())
    forecaster = XGBoostForecaster(_make_config())
    forecaster.train(lf, [1, 2, 777])  # 777 has no rows
    assert forecaster.trained_time_series_ids == [1, 2]
