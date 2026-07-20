"""Integration test for the ``cv_power_forecasts`` asset.

Exercises the real wiring end-to-end against a file-based MLflow + temp Delta tables: register an
experiment, train a fold model with ``trained_cv_model``, then materialise ``cv_power_forecasts``
and assert the forecasts land in the ``power_forecasts`` Delta table — stamped with the fold label
(not ``"live"``), restricted to the model's trained population, covering every NWP ensemble member,
and written idempotently so a re-materialisation does not duplicate rows.
"""

from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pytest
from contracts.ml_schemas import EligibleTimeSeries
from dagster import DagsterInstance, RunConfig, materialize
from deltalake import write_deltalake
from mlflow.tracking import MlflowClient

from _nwp_test_data import NWP_CONTINUOUS_COL_VALUES
from nged_substation_forecast.defs.cv_assets import cv_power_forecasts, trained_cv_model
from nged_substation_forecast.defs.jobs import RegisterExperimentConfig, register_experiment_job

pytestmark = pytest.mark.integration

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_predict_smoke"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

# ts1 sits in H3 cell 10. It has training-window data (so the fold model trains a Booster for it)
# and validation-window NWP across three ensemble members (so prediction spans the ensemble).
_TS1_CELL = 10
_TRAIN_DAY = datetime(
    2024, 6, 1, tzinfo=timezone.utc
)  # inside train window [2024-04-01, 2025-06-30]
_VAL_DAY = datetime(2025, 8, 1, tzinfo=timezone.utc)  # inside val window [2025-07-01, 2026-06-30]
_VAL_MEMBERS = (0, 1, 2)  # stands in for the full 51-member ensemble in this synthetic fixture


def _half_hours(day: datetime) -> pl.Series:
    return pl.datetime_range(
        day.replace(hour=6), day.replace(hour=8), interval="30m", time_zone="UTC", eager=True
    )


def _write_power(path: str) -> None:
    """Observed power for ts1 in the training window (the validation forecast needs no actuals)."""
    rows = [
        {"time_series_id": 1, "time": t, "power": 100.0 + i}
        for i, t in enumerate(_half_hours(_TRAIN_DAY))
    ]
    pl.DataFrame(rows).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path)


def _nwp_records(cell: int, day: datetime, members: tuple[int, ...]) -> list[dict]:
    records = []
    init_time = day.replace(hour=0)
    for member in members:
        for valid_time in _half_hours(day):
            record = {
                "nwp_model_id": "ECMWF_ENS_0_25_degree",
                "init_time": init_time,
                "valid_time": valid_time,
                "ensemble_member": member,
                "h3_index": cell,
                "categorical_precipitation_type_surface": None,
            }
            record.update(NWP_CONTINUOUS_COL_VALUES)
            records.append(record)
    return records


def _write_nwp(path: str) -> None:
    """Training NWP (control member) plus validation NWP across the ensemble, both for ts1's cell."""
    records = _nwp_records(_TS1_CELL, _TRAIN_DAY, (0,)) + _nwp_records(
        _TS1_CELL, _VAL_DAY, _VAL_MEMBERS
    )
    df = pl.DataFrame(records).cast(
        {
            "init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "ensemble_member": pl.UInt8,
            "h3_index": pl.UInt64,
            "categorical_precipitation_type_surface": pl.UInt8,
            **{col: pl.Float32 for col in NWP_CONTINUOUS_COL_VALUES},
        }
    )
    write_deltalake(table_or_uri=path, data=df.to_arrow())


def _write_metadata(path: Path) -> None:
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "h3_res_5": pl.Series([_TS1_CELL], dtype=pl.UInt64),
            "time_series_type": ["Primary"],
        }
    ).write_parquet(path)


def _write_eligible(path: str) -> None:
    eligible = EligibleTimeSeries.validate(
        pl.DataFrame(
            {
                "fold_id": pl.Series([FOLD_ID], dtype=pl.String),
                "time_series_id": pl.Series([1], dtype=pl.Int32),
            }
        )
    )
    write_deltalake(table_or_uri=path, data=eligible.to_arrow(), partition_by=["fold_id"])


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    forecasts_path = tmp_path / "power_forecasts"
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(tmp_path / "eligible"))
    monkeypatch.setenv("MODEL_CACHE_BASE_PATH", str(tmp_path / "cache"))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(forecasts_path))
    mlflow.set_tracking_uri(tracking_uri)

    _write_power(str(nged_path / "power_time_series.delta"))
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _write_eligible(str(tmp_path / "eligible"))
    return {"forecasts": str(forecasts_path)}


def _register(instance: DagsterInstance) -> None:
    result = register_experiment_job.execute_in_process(
        run_config=RunConfig(
            ops={
                "register_experiment": RegisterExperimentConfig(
                    experiment_name=EXPERIMENT_NAME,
                    base_model_config="conf/model/xgboost.yaml",
                    config_overrides={"selected_features": ["temperature_2m"], "n_estimators": 5},
                    # full_cv so the leaderboard fold's partition (FOLD_ID) is registered; this test
                    # drives that fold's window and synthetic data, not the smoke_test fold.
                    run_mode="full_cv",
                )
            }
        ),
        instance=instance,
    )
    assert result.success


def _read_forecasts(env: dict[str, str]) -> pl.DataFrame:
    return pl.read_delta(env["forecasts"])


def test_cv_power_forecasts_predicts_validation_fold(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()
    _register(instance)
    assert materialize([trained_cv_model], partition_key=PARTITION_KEY, instance=instance).success
    assert materialize([cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance).success

    forecasts = _read_forecasts(env)
    assert forecasts.height > 0

    # Stamped with the fold label and the experiment, not the "live" production sentinel.
    assert (forecasts["fold_id"] == FOLD_ID).all()
    assert (forecasts["experiment_name"] == EXPERIMENT_NAME).all()

    # Only the model's trained population (ts1) is scored.
    assert set(forecasts["time_series_id"].unique().to_list()) == {1}

    # Every ensemble member flows through — not collapsed to the control member.
    assert set(forecasts["ensemble_member"].unique().to_list()) == set(_VAL_MEMBERS)

    # The fold's child run gained prediction stats.
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    fold_runs = MlflowClient().search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.cv_role = 'fold' and tags.fold_id = '{FOLD_ID}'",
    )
    assert len(fold_runs) == 1
    assert fold_runs[0].data.metrics["n_forecast_rows"] == float(forecasts.height)


def test_cv_power_forecasts_storage_format(env: dict[str, str]) -> None:
    """The written parquet files carry the compression-oriented storage format end-to-end.

    Guards that ``cv_power_forecasts`` writes through ``delta_store.power_forecasts``: ZSTD
    compression with the per-column encodings, rows sorted within each file, and ``power_fcst``
    rounded to ``POWER_FCST_SIGNIFICAND_BITS``. The format itself is unit-tested in
    ``packages/delta_store/tests/``; this asserts the real asset actually uses it.
    """
    instance = DagsterInstance.ephemeral()
    _register(instance)
    assert materialize([trained_cv_model], partition_key=PARTITION_KEY, instance=instance).success
    assert materialize([cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance).success

    parquet_files = sorted(Path(env["forecasts"]).rglob("*.parquet"))
    assert parquet_files

    for parquet_file in parquet_files:
        metadata = pq.ParquetFile(parquet_file).metadata
        row_group = metadata.row_group(0)
        encodings_by_column = {
            row_group.column(i).path_in_schema: (
                row_group.column(i).compression,
                set(row_group.column(i).encodings),
            )
            for i in range(row_group.num_columns)
        }
        for column, (compression, encodings) in encodings_by_column.items():
            assert compression == "ZSTD", f"{column} written as {compression}, expected ZSTD"
        assert "DELTA_BINARY_PACKED" in encodings_by_column["valid_time"][1]
        assert "BYTE_STREAM_SPLIT" in encodings_by_column["power_fcst"][1]

        rows = pl.read_parquet(parquet_file)
        sort_key = rows.select(
            key=pl.struct("time_series_id", "power_fcst_init_time", "valid_time", "ensemble_member")
        )["key"]
        assert sort_key.is_sorted()

    # power_fcst is rounded to a 13-bit significand: the low 11 fraction bits of every finite
    # value are zero, yet the values were not destroyed outright. (The synthetic fixture's
    # constant NWP features make every prediction near-identical, so don't assert on value
    # diversity — the fixture trains on power ≈ 100, so surviving values must be in that
    # ballpark.)
    stored = _read_forecasts(env)["power_fcst"].to_numpy()
    assert np.isfinite(stored).all()
    assert (stored.view(np.uint32) & np.uint32((1 << 11) - 1) == 0).all()
    assert (np.abs(stored) > 50).all()


def test_cv_power_forecasts_is_idempotent(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()
    _register(instance)
    assert materialize([trained_cv_model], partition_key=PARTITION_KEY, instance=instance).success

    assert materialize([cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance).success
    first_height = _read_forecasts(env).height

    # Re-materialising the same fold overwrites its partition rather than appending.
    assert materialize([cv_power_forecasts], partition_key=PARTITION_KEY, instance=instance).success
    assert _read_forecasts(env).height == first_height
