"""Integration test for the ``live_forecasts`` asset.

Exercises the real wiring end-to-end against temp Delta tables and a plain-disk production
model (no MLflow — ``live_forecasts`` never touches it): a tiny trained ``XGBoostForecaster`` is
saved directly to ``PRODUCTION_MODEL_PATH``, then ``live_forecasts`` is materialised for one
6-hourly partition against two NWP runs — a same-day run and a day-earlier run, both covering
valid times just after ``power_fcst_init_time`` — so ``live`` and ``replay`` availability modes
are forced to pick different runs.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import patito as pt
import polars as pl
import pytest
from _nwp_test_data import NWP_CONTINUOUS_COL_VALUES
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import TimeSeriesMetadata
from dagster import (
    AssetCheckSeverity,
    DagsterInstance,
    ExecuteInProcessResult,
    RunConfig,
    materialize,
)
from deltalake import write_deltalake
from ml_core.base_forecaster import TRAINED_METADATA_FILENAME, write_trained_metadata
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs.checks import live_forecasts_are_healthy
from nged_substation_forecast.defs.production_assets import LiveForecastsConfig, live_forecasts

pytestmark = pytest.mark.integration

# power_fcst_init_time = the partition's forecast init time (window end); the partition *key* is
# the window start, 6 hours earlier (live_forecast_partitions ticks every 6h) — see
# live_forecasts's docstring.
_POWER_FCST_INIT_TIME = datetime(2026, 7, 4, 0, 0, tzinfo=UTC)
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

_VALID_TIMES = [
    _POWER_FCST_INIT_TIME + offset
    for offset in (timedelta(minutes=30), timedelta(days=1), timedelta(days=7), timedelta(days=13))
]
"""Four valid times after ``_POWER_FCST_INIT_TIME``, spread across the forecast horizon.

They span days rather than the first two hours so that the forecast this fixture produces has a
realistic *reach*, which ``live_forecasts_are_healthy`` checks: a slot whose rows stop a couple of
hours ahead is an undeliverable forecast however well-formed each row is."""


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
            record.update(NWP_CONTINUOUS_COL_VALUES)
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
            **dict.fromkeys(NWP_CONTINUOUS_COL_VALUES, pl.Float32),
        }
    )
    write_deltalake(
        table_or_uri=path, data=df.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )


def _write_power(path: str) -> None:
    """A little history before ``_POWER_FCST_INIT_TIME``, for the trained and untrained series."""
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


def _typed_metadata(frame: pl.DataFrame) -> pt.DataFrame[TimeSeriesMetadata]:
    """Label a partial roster frame with its model, as the assets' own readers do.

    These fixtures carry only the columns the pipeline reads, so ``set_model`` (never
    ``validate``) is what the production code does with them too.
    """
    return pt.DataFrame(frame).set_model(TimeSeriesMetadata)


def _metadata_for(time_series_ids: tuple[int, ...], cells: tuple[int, ...]) -> pl.DataFrame:
    """A minimal roster frame: the columns the feature pipeline actually reads."""
    return pl.DataFrame(
        {
            "time_series_id": pl.Series(time_series_ids, dtype=pl.Int32),
            "h3_res_5": pl.Series(cells, dtype=pl.UInt64),
            "time_series_type": ["Primary"] * len(time_series_ids),
        }
    )


def _write_metadata(path: Path) -> None:
    """ts1 (trained, cell 10) and ts2 (untrained, cell 20 — no NWP data for that cell)."""
    _metadata_for((1, 2), (_TRAINED_CELL, _UNTRAINED_CELL)).write_parquet(path)


def _save_promoted_model(path: Path) -> None:
    """A tiny real ``XGBoostForecaster`` trained on ts1 only, saved straight to disk."""
    times = [datetime(2025, 1, 1, hour, tzinfo=UTC) for hour in (0, 1, 2)]
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
    # Stands in for `fetch_model_artifacts`, which is what deposits this in production.
    write_trained_metadata(path, _typed_metadata(_metadata_for((1,), (_TRAINED_CELL,))))


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    forecasts_path = tmp_path / "power_forecasts"
    production_model_path = tmp_path / "production_model"

    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(forecasts_path))
    monkeypatch.setenv("PRODUCTION_MODEL_PATH", str(production_model_path))

    _write_power(str(nged_path / "power_time_series.delta"))
    _write_nwp(str(tmp_path / "NWP"))
    _write_metadata(nged_path / "metadata.parquet")
    _save_promoted_model(production_model_path)

    return {"forecasts": str(forecasts_path)}


def _materialize(
    instance: DagsterInstance,
    availability_mode: str,
    partition_key: str = _PARTITION_KEY,
    raise_on_error: bool = True,
) -> ExecuteInProcessResult:
    return materialize(
        [live_forecasts],
        partition_key=partition_key,
        run_config=RunConfig(
            ops={"live_forecasts": LiveForecastsConfig(availability_mode=availability_mode)}
        ),
        instance=instance,
        raise_on_error=raise_on_error,
    )


def _read_forecasts(env: dict[str, str]) -> pl.DataFrame:
    return pl.read_delta(env["forecasts"])


def test_live_and_replay_select_different_nwp_runs(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    assert _materialize(dagster_instance, "live").success
    live_forecasts_df = _read_forecasts(env)
    assert (live_forecasts_df["nwp_init_time"] == _SAME_DAY_RUN).all()

    assert _materialize(dagster_instance, "replay").success
    replay_forecasts_df = _read_forecasts(env)
    assert (replay_forecasts_df["nwp_init_time"] == _DAY_EARLIER_RUN).all()


def test_only_trained_time_series_are_forecast(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    assert _materialize(dagster_instance, "live").success

    forecasts = _read_forecasts(env)
    assert set(forecasts["time_series_id"].unique().to_list()) == {1}


def test_all_ensemble_members_present(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    assert _materialize(dagster_instance, "live").success

    forecasts = _read_forecasts(env)
    assert set(forecasts["ensemble_member"].unique().to_list()) == set(_MEMBERS)


def test_idempotency_same_partition_twice(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    assert _materialize(dagster_instance, "live").success
    first_height = _read_forecasts(env).height

    assert _materialize(dagster_instance, "live").success
    assert _read_forecasts(env).height == first_height


def test_accumulation_across_partitions(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    """A second partition's rows coexist with the first (the write predicate doesn't wipe the
    whole "live" fold — only the one power_fcst_init_time it targets)."""
    assert _materialize(dagster_instance, "live", partition_key=_EARLIER_TICK_PARTITION_KEY).success
    first_rows = _read_forecasts(env)
    assert (first_rows["power_fcst_init_time"] == _EARLIER_TICK_POWER_FCST_INIT_TIME).all()

    assert _materialize(dagster_instance, "live", partition_key=_PARTITION_KEY).success

    combined = _read_forecasts(env)
    assert combined.height > first_rows.height
    assert set(combined["power_fcst_init_time"].unique().to_list()) == {
        _EARLIER_TICK_POWER_FCST_INIT_TIME,
        _POWER_FCST_INIT_TIME,
    }


def test_check_passes_after_a_real_live_materialisation(
    env: dict[str, str], dagster_instance: DagsterInstance
) -> None:
    """``live_forecasts_are_healthy`` passes against a genuinely produced forecast.

    This is the only test that exercises the check *partitioned*: ``build_asset_check_context``
    (used by ``tests/test_checks.py``) cannot carry a partition key, so only a real materialisation
    proves the check reports on its partition's ``power_fcst_init_time`` — the window's end — and
    not on whatever slot the wall clock happens to be in.
    """
    result = materialize(
        [live_forecasts, live_forecasts_are_healthy],
        partition_key=_PARTITION_KEY,
        run_config=RunConfig(ops={"live_forecasts": LiveForecastsConfig(availability_mode="live")}),
        instance=dagster_instance,
    )
    assert result.success

    (evaluation,) = result.get_asset_check_evaluations()
    assert evaluation.check_name == "live_forecasts_are_healthy"
    assert evaluation.passed, evaluation.description
    assert evaluation.severity == AssetCheckSeverity.WARN
    assert evaluation.metadata["power_fcst_init_time"].value == _POWER_FCST_INIT_TIME.isoformat()
    assert evaluation.metadata["n_missed_nwp_runs"].value == 0
    assert evaluation.metadata["n_invalid_rows"].value == 0
    assert evaluation.metadata["n_time_series"].value == 1


def _capture_checkins(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Enable the monitor flag and capture every Sentry check-in the asset would emit."""
    monkeypatch.setenv("SENTRY_MONITOR_FORECASTS", "true")
    calls: list[dict] = []
    monkeypatch.setattr(
        "nged_substation_forecast._sentry.capture_checkin", lambda **kw: calls.append(kw)
    )
    return calls


def test_live_run_sends_sentry_heartbeat(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A successful live run emits exactly one OK heartbeat to the live-forecasts monitor."""
    calls = _capture_checkins(monkeypatch)
    assert _materialize(dagster_instance, "live").success
    assert len(calls) == 1
    assert calls[0]["monitor_slug"] == "live-forecasts"
    assert calls[0]["status"] == "ok"


def test_replay_run_sends_no_sentry_heartbeat(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """A replay backfill reprocesses the past, so it must not check in to the live monitor."""
    calls = _capture_checkins(monkeypatch)
    assert _materialize(dagster_instance, "replay").success
    assert calls == []


def _write_power_for(path: str, time_series_ids: tuple[int, ...]) -> None:
    """Overwrite the power table with a little history for each of `time_series_ids`."""
    times = [
        _POWER_FCST_INIT_TIME - timedelta(hours=1),
        _POWER_FCST_INIT_TIME - timedelta(minutes=30),
    ]
    rows = [
        {"time_series_id": ts_id, "time": t, "power": 100.0 + i}
        for ts_id in time_series_ids
        for i, t in enumerate(times)
    ]
    pl.DataFrame(rows).sort(["time_series_id", "time"]).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path, mode="overwrite", delta_write_options={"partition_by": "time_series_id"})


def _save_model_trained_on(path: Path, time_series_ids: list[int]) -> None:
    """A tiny real ``XGBoostForecaster`` trained on each of `time_series_ids`."""
    times = [datetime(2025, 1, 1, hour, tzinfo=UTC) for hour in (0, 1, 2)]
    train_df = pl.DataFrame(
        {
            "time_series_id": [ts_id for ts_id in time_series_ids for _ in times],
            "valid_time": times * len(time_series_ids),
            "power_fcst_init_time": times * len(time_series_ids),
            "power": [10.0, 12.0, 11.0] * len(time_series_ids),
            "temperature_2m": [5.0, 6.0, 7.0] * len(time_series_ids),
        }
    )
    train_data = pt.LazyFrame.from_existing(train_df.lazy()).set_model(AllFeatures)
    forecaster = XGBoostForecaster(
        XGBoostConfig(
            selected_features={"temperature_2m"}, experiment_name="live_test", n_estimators=5
        )
    )
    forecaster.train(train_data, time_series_ids=time_series_ids)
    forecaster.save(path)
    write_trained_metadata(
        path,
        _typed_metadata(
            _metadata_for(tuple(time_series_ids), (_TRAINED_CELL,) * len(time_series_ids))
        ),
    )


def test_the_roster_cannot_thin_or_fail_a_live_slot(
    env: dict[str, str], dagster_instance: DagsterInstance, tmp_path: Path
) -> None:
    """A roster fault costs the live service nothing, because it does not read the roster.

    Each series' H3 cell comes from the model's own frozen copy, so a roster that has lost rows,
    or cannot be read at all, leaves the forecast identical. Losing a row used to drop that series
    silently and an unreadable file used to fail the slot outright — both off the degradation
    ladder entirely (issue #528).
    """
    roster = tmp_path / "NGED" / "metadata.parquet"
    # ts3 shares ts1's NWP cell, so both are genuinely forecastable to begin with.
    _metadata_for((1, 3), (_TRAINED_CELL, _TRAINED_CELL)).write_parquet(roster)
    _write_power_for(str(tmp_path / "NGED" / "power_time_series.delta"), (1, 3))
    _save_model_trained_on(tmp_path / "production_model", [1, 3])

    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}

    # Thin (ts3's row is gone), then unreadable, then absent — the three states an incident walks
    # through, since deleting the file is how an operator recovers from an unreadable one.
    _metadata_for((1,), (_TRAINED_CELL,)).write_parquet(roster)
    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}

    roster.write_bytes(b"not a parquet file")
    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}

    roster.unlink()
    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}


def test_a_model_directory_with_no_frozen_metadata_is_refused(
    env: dict[str, str], dagster_instance: DagsterInstance, tmp_path: Path
) -> None:
    """A model saved before the frozen copy existed cannot be served, and says so.

    This is the one metadata state that still fails the slot, deliberately: it is a promotion
    fault — the model on disk is not one this code can serve — not an input outage.
    """
    (tmp_path / "production_model" / TRAINED_METADATA_FILENAME).unlink()

    result = _materialize(dagster_instance, "live", raise_on_error=False)

    assert not result.success
    failure = result.failure_data_for_node("live_forecasts")
    assert failure is not None
    assert "promote that run" in str(failure.error)
