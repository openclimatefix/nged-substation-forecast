"""Integration test: the manual heuristic through the real ``register -> train -> predict`` chain.

Registers ``conf/model/manual_heuristic.yaml``, writes synthetic power and NWP to temp Delta tables,
materialises ``trained_cv_model`` and ``cv_power_forecasts``, and reads back the ``power_forecasts``
rows. The unit tests in ``packages/baseline_forecasters/tests/`` cover the forecaster on its own.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import polars as pl
import pytest
from _cleaned_power_test_data import write_cleaned_copy, write_metadata
from _nwp_test_data import half_hours, nwp_records, write_test_nwp
from contracts.ml_schemas import EligibleTimeSeries
from dagster import DagsterInstance, materialize
from deltalake import write_deltalake

from nged_substation_forecast.defs.cv_assets import cv_power_forecasts, trained_cv_model

pytestmark = pytest.mark.integration

RegisterExperiment = Callable[..., None]
"""Type of the ``register_experiment`` fixture (``tests/conftest.py``)."""

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_manual_heuristic"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

_CELL = 599423199024775167  # the h3_res_5 that `write_metadata` hard-codes
_TRAIN_DAY = datetime(2024, 6, 1, tzinfo=UTC)  # inside train window [2024-04-01, 2025-06-30]
_VAL_DAY = datetime(2025, 8, 1, tzinfo=UTC)  # inside val window [2025-07-01, 2026-06-30]
_VAL_MEMBERS = (0, 1, 2)
_EPOCH = datetime(2020, 1, 1, tzinfo=UTC)
_WEEKLY_AND_ANNUAL_LAG_HOURS = [168 * k for k in (*range(1, 7), *range(49, 56))]


def _power_at(time: datetime) -> float:
    """Integer power equal to the half-hour index since a fixed epoch, mod 1999, shifted by -999.

    1999 is prime and the 13 lags are multiples of 336 half-hours, so the 13 lag values at one
    target time are distinct. A lag off by one half-hour lands on yet another value.
    """
    return float(int((time - _EPOCH).total_seconds() // 1800) % 1999 - 999)


def _write_power(path: str) -> None:
    times = pl.datetime_range(
        _TRAIN_DAY - timedelta(days=30),
        _VAL_DAY + timedelta(hours=12),
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1] * len(times), dtype=pl.Int32),
            "time": times,
            "power": pl.Series([_power_at(t) for t in times.to_list()], dtype=pl.Float32),
        }
    ).write_delta(path)


def _write_nwp(path: str) -> None:
    """A control-member run on a training day, plus a validation-day run with three members."""
    records = nwp_records(_CELL, _TRAIN_DAY, (0,)) + nwp_records(_CELL, _VAL_DAY, _VAL_MEMBERS)
    write_test_nwp(path, records)


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
def forecasts_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("ELIGIBLE_TIME_SERIES_DATA_PATH", str(tmp_path / "eligible"))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(tmp_path / "power_forecasts"))
    mlflow.set_tracking_uri(tracking_uri)

    _write_power(str(nged_path / "power_time_series.delta"))
    write_cleaned_copy(nged_path / "power_time_series.delta")
    _write_nwp(str(tmp_path / "NWP"))
    write_metadata(nged_path / "metadata.parquet", {1: "Primary"})
    _write_eligible(str(tmp_path / "eligible"))
    return str(tmp_path / "power_forecasts")


def test_manual_heuristic_emits_the_thirteen_lagged_powers_as_members(
    forecasts_path: str,
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
) -> None:
    # config_overrides={} keeps the YAML's own selected_features.
    register_experiment(
        dagster_instance,
        EXPERIMENT_NAME,
        base_model_config="conf/model/manual_heuristic.yaml",
        config_overrides={},
    )
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    forecasts = pl.read_delta(forecasts_path)

    assert (forecasts["power_fcst_model_name"] == "manual_heuristic").all()
    assert forecasts["nwp_init_time"].is_null().all()
    # One row per valid time and member: the three NWP members are not repeated.
    valid_times = half_hours(_VAL_DAY).to_list()
    assert forecasts.height == len(valid_times) * len(_WEEKLY_AND_ANNUAL_LAG_HOURS)
    for row in forecasts.iter_rows(named=True):
        lag_hours = _WEEKLY_AND_ANNUAL_LAG_HOURS[row["ensemble_member"]]
        assert row["power_fcst"] == _power_at(row["valid_time"] - timedelta(hours=lag_hours))
    assert set(forecasts["ensemble_member"].to_list()) == set(range(13))
