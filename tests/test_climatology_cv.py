"""Integration test: climatology through the real ``register -> train -> predict`` chain.

Registers ``conf/model/climatology.yaml``, writes synthetic power and NWP to temp Delta tables,
materialises ``trained_cv_model`` and ``cv_power_forecasts``, and reads back the ``power_forecasts``
rows. The unit tests in ``packages/baseline_forecasters/tests/`` cover the forecaster on its own.
"""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import numpy as np
import patito as pt
import polars as pl
import pytest
from _cleaned_power_test_data import write_cleaned_copy, write_metadata
from _nwp_test_data import half_hours, nwp_records, write_test_nwp
from _power_test_data import power_at
from contracts.ml_schemas import EligibleTimeSeries
from contracts.power_schemas import EffectiveCapacity, PowerForecast, TimeSeriesMetadata
from dagster import DagsterInstance, materialize
from delta_store.power_forecasts import POWER_FCST_SIGNIFICAND_BITS
from delta_store.precision import round_to_significand_bits
from deltalake import write_deltalake
from ml_core.metrics import compute_metrics
from nged_data.storage import scan_cleaned_power

from nged_substation_forecast.defs.cv_assets import cv_power_forecasts, trained_cv_model

pytestmark = pytest.mark.integration

RegisterExperiment = Callable[..., None]
"""Type of the ``register_experiment`` fixture (``tests/conftest.py``)."""

FOLD_ID = "mid_2025_to_mid_2026"
EXPERIMENT_NAME = "exp_climatology"
PARTITION_KEY = f"{EXPERIMENT_NAME}__{FOLD_ID}"

_CELL = 599423199024775167  # the h3_res_5 that `write_metadata` hard-codes
_TRAIN_DAYS = [datetime(2024, 8, day, tzinfo=UTC) for day in range(5, 10)]  # Monday to Friday
_VAL_FRIDAY = datetime(2025, 8, 1, tzinfo=UTC)
_VAL_SATURDAY = datetime(2025, 8, 2, tzinfo=UTC)
_VAL_MEMBERS = (0, 1, 2)
_MEMBER_COUNT = 51
_LONDON = ZoneInfo("Europe/London")


def _local_half_hour(time: datetime) -> int:
    local = time.astimezone(_LONDON)
    return local.hour * 2 + local.minute // 30


def _write_power(path: str) -> None:
    times = pl.datetime_range(
        _TRAIN_DAYS[0] - timedelta(days=30),
        _VAL_SATURDAY + timedelta(hours=12),
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1] * len(times), dtype=pl.Int32),
            "time": times,
            "power": pl.Series([power_at(t) for t in times.to_list()], dtype=pl.Float32),
        }
    ).write_delta(path)


def _write_nwp(path: str) -> None:
    """Control-member runs on five training days, plus validation runs on a Friday and Saturday."""
    records = [record for day in _TRAIN_DAYS for record in nwp_records(_CELL, day, (0,))]
    records += nwp_records(_CELL, _VAL_FRIDAY, _VAL_MEMBERS)
    records += nwp_records(_CELL, _VAL_SATURDAY, (0,))
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
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
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
    return {
        "forecasts": tmp_path / "power_forecasts",
        "cleaned_power": nged_path / "cleaned_power_time_series.delta",
        "metadata": nged_path / "metadata.parquet",
    }


def _oracle_members(validation_half_hour: int) -> list[float]:
    """The 51 quantiles, pooled in plain Python from the August 2024 weekday samples."""
    samples = [
        power_at(valid_time)
        for day in _TRAIN_DAYS
        for valid_time in half_hours(day).to_list()
        if min(
            (_local_half_hour(valid_time) - validation_half_hour) % 48,
            (validation_half_hour - _local_half_hour(valid_time)) % 48,
        )
        <= 1
    ]
    levels = [(k + 0.5) / _MEMBER_COUNT for k in range(_MEMBER_COUNT)]
    quantiles = pl.Series(np.quantile(samples, levels, method="linear"), dtype=pl.Float32).to_frame(
        "quantile"
    )
    return quantiles.select(
        round_to_significand_bits(pl.col("quantile"), keep_bits=POWER_FCST_SIGNIFICAND_BITS)
    )["quantile"].to_list()


def test_climatology_emits_pooled_quantiles_as_members_and_crps_flows(
    paths: dict[str, Path],
    dagster_instance: DagsterInstance,
    register_experiment: RegisterExperiment,
) -> None:
    # config_overrides={} keeps the YAML's own empty selected_features.
    register_experiment(
        dagster_instance,
        EXPERIMENT_NAME,
        base_model_config="conf/model/climatology.yaml",
        config_overrides={},
    )
    assert materialize(
        [trained_cv_model], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success
    assert materialize(
        [cv_power_forecasts], partition_key=PARTITION_KEY, instance=dagster_instance
    ).success

    forecasts = pl.read_delta(str(paths["forecasts"]))

    assert forecasts["nwp_init_time"].is_null().all()
    assert forecasts["fold_id"].unique().to_list() == [FOLD_ID]
    assert forecasts["experiment_name"].unique().to_list() == [EXPERIMENT_NAME]
    assert forecasts["power_fcst_model_name"].unique().to_list() == ["climatology"]
    # Five Friday valid times times 51 members: the three NWP members are not repeated, and the
    # Saturday run, whose weekend cells have no training sample nearby, is dropped.
    valid_times = half_hours(_VAL_FRIDAY).to_list()
    assert forecasts.height == len(valid_times) * _MEMBER_COUNT
    assert set(forecasts["valid_time"].to_list()) == set(valid_times)
    for valid_time in valid_times:
        rows = forecasts.filter(pl.col("valid_time") == valid_time).sort("ensemble_member")
        assert rows["ensemble_member"].to_list() == list(range(_MEMBER_COUNT))
        expected = _oracle_members(_local_half_hour(valid_time))
        assert len(set(expected)) == _MEMBER_COUNT
        assert rows["power_fcst"].to_list() == expected

    # CRPS flows over the members: the fair CRPS differs from the MAE of the ensemble mean.
    metadata = pt.DataFrame(pl.read_parquet(paths["metadata"])).set_model(TimeSeriesMetadata)
    capacity = EffectiveCapacity.validate(
        pl.DataFrame(
            {
                "time_series_id": pl.Series([1], dtype=pl.Int32),
                "time": pl.Series([_VAL_SATURDAY], dtype=pl.Datetime("us", "UTC")),
                "effective_capacity_mw": pl.Series([1000.0], dtype=pl.Float32),
            }
        )
    )
    metrics = compute_metrics(
        PowerForecast.validate(forecasts),
        scan_cleaned_power(str(paths["cleaned_power"])),
        metadata,
        capacity,
    ).filter(pl.col("horizon_slice") == "all")
    crps = metrics.filter(pl.col("metric_name") == "crps")["metric_value"]
    mae = metrics.filter(pl.col("metric_name") == "mae")["metric_value"]
    assert crps.len() == 1
    assert crps.is_not_null().all()
    assert crps[0] != pytest.approx(mae[0])
