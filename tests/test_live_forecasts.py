"""Integration test for the ``live_forecasts`` asset.

Exercises the real wiring end-to-end against temp Delta tables and a plain-disk production
model (no MLflow — ``live_forecasts`` never touches it): a tiny trained ``XGBoostForecaster`` is
saved directly to ``PRODUCTION_MODEL_PATH``, then ``live_forecasts`` is materialised for one
6-hourly partition against two NWP runs — a same-day run and a day-earlier run, both covering
valid times just after ``power_fcst_init_time`` — so ``live`` and ``replay`` availability modes
are forced to pick different runs.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from contracts.ml_schemas import AllFeatures
from dagster import DagsterInstance, RunConfig, materialize
from deltalake import write_deltalake
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.production_assets import LiveForecastsConfig, live_forecasts

pytestmark = pytest.mark.integration

# power_fcst_init_time = the partition's forecast init time (window end); the partition *key* is
# the window start, 6 hours earlier (live_forecast_partitions ticks every 6h) — see
# live_forecasts's docstring.
_POWER_FCST_INIT_TIME = datetime(2026, 7, 4, 0, 0, tzinfo=timezone.utc)
_PARTITION_KEY = "2026-07-03-18:00"

# The tick 6h before _POWER_FCST_INIT_TIME: still after _DAY_EARLIER_RUN, still before
# _SAME_DAY_RUN, so "live" mode picks _DAY_EARLIER_RUN here too — used by the accumulation test
# as a second, distinct partition.
_EARLIER_TICK_POWER_FCST_INIT_TIME = _POWER_FCST_INIT_TIME - timedelta(hours=6)
_EARLIER_TICK_PARTITION_KEY = "2026-07-03-12:00"

_SAME_DAY_RUN = _POWER_FCST_INIT_TIME  # 2026-07-04 00Z — only visible in "live" (within the delay).
_DAY_EARLIER_RUN = _POWER_FCST_INIT_TIME - timedelta(
    days=1
)  # 2026-07-03 00Z — visible in "replay".

_TRAINED_CELL = 10
_UNTRAINED_CELL = 20
_MEMBERS = (0, 1, 2)

_NWP_CONTINUOUS_COL_VALUES = {
    "temperature_2m": 15.0,
    "dew_point_temperature_2m": 10.0,
    "wind_speed_10m": 5.0,
    "wind_direction_10m": 180.0,
    "wind_speed_100m": 8.0,
    "wind_direction_100m": 180.0,
    "pressure_surface": 101_000.0,
    "pressure_reduced_to_mean_sea_level": 101_500.0,
    "geopotential_height_500hpa": 5_500.0,
    "downward_long_wave_radiation_flux_surface": 300.0,
    "downward_short_wave_radiation_flux_surface": 200.0,
    "precipitation_surface": 0.001,
}
"""Physically plausible Float32 constants, one per continuous ``Nwp`` variable."""

_VALID_TIMES = [_POWER_FCST_INIT_TIME + timedelta(minutes=30 * i) for i in range(1, 5)]
"""00:30 .. 02:00, after ``_POWER_FCST_INIT_TIME``."""


def _nwp_records(cell: int, init_time: datetime, members: tuple[int, ...]) -> list[dict]:
    records = []
    for member in members:
        for valid_time in _VALID_TIMES:
            record = {
                "nwp_model_id": "ECMWF_ENS_0_25_degree",
                "init_time": init_time,
                "valid_time": valid_time,
                "ensemble_member": member,
                "h3_index": cell,
                "categorical_precipitation_type_surface": None,
            }
            record.update(_NWP_CONTINUOUS_COL_VALUES)
            records.append(record)
    return records


def _write_nwp(path: str) -> None:
    """Two runs for the trained series' cell: the day-earlier run and the same-day run."""
    records = _nwp_records(_TRAINED_CELL, _DAY_EARLIER_RUN, _MEMBERS) + _nwp_records(
        _TRAINED_CELL, _SAME_DAY_RUN, _MEMBERS
    )
    df = pl.DataFrame(records).cast(
        {
            "init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "ensemble_member": pl.UInt8,
            "h3_index": pl.UInt64,
            "categorical_precipitation_type_surface": pl.UInt8,
            **{col: pl.Float32 for col in _NWP_CONTINUOUS_COL_VALUES},
        }
    )
    write_deltalake(
        table_or_uri=path, data=df.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )


def _write_power(path: str) -> None:
    """A little history before ``_POWER_FCST_INIT_TIME`` for both the trained and untrained series."""
    times = [
        _POWER_FCST_INIT_TIME - timedelta(hours=1),
        _POWER_FCST_INIT_TIME - timedelta(minutes=30),
    ]
    rows = [
        {"time_series_id": ts_id, "time": t, "power": 100.0 + i}
        for ts_id in (1, 2)
        for i, t in enumerate(times)
    ]
    pl.DataFrame(rows).sort(["time_series_id", "time"]).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path, delta_write_options={"partition_by": "time_series_id"})


def _write_metadata(path: Path) -> None:
    """ts1 (trained, cell 10) and ts2 (untrained, cell 20 — no NWP data for that cell)."""
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            "h3_res_5": pl.Series([_TRAINED_CELL, _UNTRAINED_CELL], dtype=pl.UInt64),
            "time_series_type": ["Primary", "Primary"],
        }
    ).write_parquet(path)


def _save_production_model(path: Path) -> None:
    """A tiny real ``XGBoostForecaster`` trained on ts1 only, saved straight to disk."""
    times = [datetime(2025, 1, 1, hour, tzinfo=timezone.utc) for hour in (0, 1, 2)]
    train_df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "valid_time": times,
            "time_series_type": ["Primary"] * 3,
            "power_fcst_init_time": times,
            "power": [10.0, 12.0, 11.0],
            "temperature_2m": [5.0, 6.0, 7.0],
        }
    )
    train_data = pt.LazyFrame.from_existing(train_df.lazy()).set_model(AllFeatures)
    config = XGBoostConfig(
        selected_features={"temperature_2m"}, experiment_name="live_test", n_estimators=5
    )
    forecaster = XGBoostForecaster(config)
    forecaster.train(train_data, time_series_ids=[1])
    forecaster.save(path)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    forecasts_path = tmp_path / "power_forecasts"
    production_model_path = tmp_path / "production_model"

    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(forecasts_path))
    monkeypatch.setenv("PRODUCTION_MODEL_PATH", str(production_model_path))

    _write_power(str(nged_path / "power_time_series.delta"))
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _save_production_model(production_model_path)

    return {"forecasts": str(forecasts_path)}


def _materialize(
    instance: DagsterInstance, availability_mode: str, partition_key: str = _PARTITION_KEY
):
    return materialize(
        [live_forecasts],
        partition_key=partition_key,
        run_config=RunConfig(
            ops={"live_forecasts": LiveForecastsConfig(availability_mode=availability_mode)}
        ),
        instance=instance,
    )


def _read_forecasts(env: dict[str, str]) -> pl.DataFrame:
    return pl.read_delta(env["forecasts"])


def test_live_and_replay_select_different_nwp_runs(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()

    assert _materialize(instance, "live").success
    live_forecasts_df = _read_forecasts(env)
    assert (live_forecasts_df["nwp_init_time"] == _SAME_DAY_RUN).all()

    assert _materialize(instance, "replay").success
    replay_forecasts_df = _read_forecasts(env)
    assert (replay_forecasts_df["nwp_init_time"] == _DAY_EARLIER_RUN).all()


def test_only_trained_time_series_are_forecast(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()
    assert _materialize(instance, "live").success

    forecasts = _read_forecasts(env)
    assert set(forecasts["time_series_id"].unique().to_list()) == {1}


def test_all_ensemble_members_present(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()
    assert _materialize(instance, "live").success

    forecasts = _read_forecasts(env)
    assert set(forecasts["ensemble_member"].unique().to_list()) == set(_MEMBERS)


def test_idempotency_same_partition_twice(env: dict[str, str]) -> None:
    instance = DagsterInstance.ephemeral()
    assert _materialize(instance, "live").success
    first_height = _read_forecasts(env).height

    assert _materialize(instance, "live").success
    assert _read_forecasts(env).height == first_height


def test_accumulation_across_partitions(env: dict[str, str]) -> None:
    """A second partition's rows coexist with the first (the write predicate doesn't wipe the
    whole "live" fold — only the one power_fcst_init_time it targets)."""
    instance = DagsterInstance.ephemeral()
    assert _materialize(instance, "live", partition_key=_EARLIER_TICK_PARTITION_KEY).success
    first_rows = _read_forecasts(env)
    assert (first_rows["power_fcst_init_time"] == _EARLIER_TICK_POWER_FCST_INIT_TIME).all()

    assert _materialize(instance, "live", partition_key=_PARTITION_KEY).success

    combined = _read_forecasts(env)
    assert combined.height > first_rows.height
    assert set(combined["power_fcst_init_time"].unique().to_list()) == {
        _EARLIER_TICK_POWER_FCST_INIT_TIME,
        _POWER_FCST_INIT_TIME,
    }
