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
from _nwp_test_data import nwp_records, write_test_nwp
from contracts.ml_schemas import AllFeatures
from contracts.power_schemas import PowerForecast, PowerTimeSeries, TimeSeriesMetadata
from contracts.settings import Settings
from contracts.weather_schemas import Nwp
from dagster import (
    AssetCheckSeverity,
    DagsterInstance,
    ExecuteInProcessResult,
    RunConfig,
    materialize,
)
from ml_core.base_forecaster import write_trained_metadata
from xgboost_forecaster.forecaster import XGBoostConfig, XGBoostForecaster

from nged_substation_forecast.defs import production_assets
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
_MEMBERS = (0, 1, 2)

_VALID_TIMES = [
    _POWER_FCST_INIT_TIME + offset
    for offset in (timedelta(minutes=30), timedelta(days=1), timedelta(days=7), timedelta(days=13))
]
"""Four valid times after ``_POWER_FCST_INIT_TIME``, spread across the forecast horizon.

They span days rather than the first two hours so that the forecast this fixture produces has a
realistic *reach*, which ``live_forecasts_are_healthy`` checks: a slot whose rows stop a couple of
hours ahead is an undeliverable forecast however well-formed each row is."""


def _write_nwp(path: str) -> None:
    """Two runs for the trained series' cell: the day-earlier run and the same-day run."""
    records = nwp_records(
        _TRAINED_CELL, _DAY_EARLIER_RUN, _MEMBERS, valid_times=_VALID_TIMES
    ) + nwp_records(_TRAINED_CELL, _SAME_DAY_RUN, _MEMBERS, valid_times=_VALID_TIMES)
    write_test_nwp(path, records)


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


def _metadata_for(
    time_series_ids: tuple[int, ...], cells: tuple[int, ...]
) -> pt.DataFrame[TimeSeriesMetadata]:
    """A roster frame holding the two columns the feature pipeline reads.

    ``set_model`` rather than ``validate``, as the assets' own readers do, so a partial frame is
    enough.
    """
    return pt.DataFrame(
        pl.DataFrame(
            {
                "time_series_id": pl.Series(time_series_ids, dtype=pl.Int32),
                "h3_res_5": pl.Series(cells, dtype=pl.UInt64),
            }
        )
    ).set_model(TimeSeriesMetadata)


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
    write_trained_metadata(
        model_dir=path, time_series_metadata=_metadata_for((1,), (_TRAINED_CELL,))
    )


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
    _save_promoted_model(production_model_path)

    return {"forecasts": str(forecasts_path)}


def _materialize(
    instance: DagsterInstance, availability_mode: str, partition_key: str = _PARTITION_KEY
) -> ExecuteInProcessResult:
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
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, dagster_instance: DagsterInstance
) -> None:
    """The live population is the model's trained population, narrowed before feature engineering.

    ``_write_power`` writes history for both series 1 and 2, but only series 1 has a trained
    booster. Spies on ``load_engineering_inputs`` to pin that ``live_forecasts`` passes it
    ``forecaster.trained_time_series_ids`` rather than every series with power data — checking the
    final output alone would pass even without that filter, because ``XGBoostForecaster.predict``
    also skips any series its own booster dict has nothing for.
    """
    captured: list[list[int]] = []
    real_load_engineering_inputs = production_assets.load_engineering_inputs

    def _spy_load_engineering_inputs(
        settings: Settings,
        *,
        time_series_ids: list[int],
        metadata: pt.DataFrame[TimeSeriesMetadata],
        window_start: datetime,
        window_end: datetime,
        ensemble_members: list[int] | None = None,
        init_time_start: datetime | None = None,
        init_time_end: datetime | None = None,
    ) -> tuple[pt.LazyFrame[PowerTimeSeries], pt.LazyFrame[Nwp]]:
        captured.append(time_series_ids)
        return real_load_engineering_inputs(
            settings,
            time_series_ids=time_series_ids,
            metadata=metadata,
            window_start=window_start,
            window_end=window_end,
            ensemble_members=ensemble_members,
            init_time_start=init_time_start,
            init_time_end=init_time_end,
        )

    monkeypatch.setattr(production_assets, "load_engineering_inputs", _spy_load_engineering_inputs)

    assert _materialize(dagster_instance, "live").success

    assert captured == [[1]]
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
        model_dir=path,
        time_series_metadata=_metadata_for(
            tuple(time_series_ids), (_TRAINED_CELL,) * len(time_series_ids)
        ),
    )


def test_the_roster_cannot_thin_or_fail_a_live_slot(
    env: dict[str, str], dagster_instance: DagsterInstance, tmp_path: Path
) -> None:
    """A roster fault costs the live service nothing, because it does not read the roster.

    Each series' H3 cell comes from the model's own frozen copy, so a roster that has lost rows, or
    cannot be read at all, leaves the forecast identical. Losing a row used to drop that series
    silently and an unreadable file used to fail the slot outright — both off the degradation
    ladder entirely (issue #528). The *absent* roster needs no step here: the ``env`` fixture
    writes none, so every other test in this file is that case.
    """
    roster = tmp_path / "NGED" / "metadata.parquet"
    # ts3 shares ts1's NWP cell, so both are genuinely forecastable to begin with.
    _metadata_for((1, 3), (_TRAINED_CELL, _TRAINED_CELL)).write_parquet(roster)
    _write_power_for(str(tmp_path / "NGED" / "power_time_series.delta"), (1, 3))
    _save_model_trained_on(tmp_path / "production_model", [1, 3])

    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}

    # ts3's row goes.
    _metadata_for((1,), (_TRAINED_CELL,)).write_parquet(roster)
    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}

    # Then the file itself becomes unreadable. Kept separate from the absent case because the
    # repo's other roster reader guards with `object_exists` (`defs/checks.py`), a shape that
    # tolerates absence and still raises on corruption.
    roster.write_bytes(b"not a parquet file")
    assert _materialize(dagster_instance, "live").success
    assert set(_read_forecasts(env)["time_series_id"].unique().to_list()) == {1, 3}


def _write_power_with_long_history(path: str) -> None:
    """Half-hourly power for ts1 reaching back past the champion config's longest lag (336h).

    ``LIVE_POWER_HISTORY`` (15 days) must cover this so ``power_lag_336h`` is non-null for every
    genuine forecast row; a window too short leaves the lag null and the champion silently
    degraded. 340h clears the 336h lag with a margin.
    """
    times = pl.datetime_range(
        _POWER_FCST_INIT_TIME - timedelta(hours=340),
        _POWER_FCST_INIT_TIME,
        interval="30m",
        time_zone="UTC",
        eager=True,
    )
    rows = [{"time_series_id": 1, "time": t, "power": 100.0 + i} for i, t in enumerate(times)]
    pl.DataFrame(rows).sort("time").cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(path, mode="overwrite", delta_write_options={"partition_by": "time_series_id"})


def _save_model_trained_on_power_lag(path: Path) -> None:
    """A tiny real ``XGBoostForecaster`` selecting ``power_lag_336h``, trained on ts1 only."""
    times = [datetime(2025, 1, 1, hour, tzinfo=UTC) for hour in (0, 1, 2)]
    train_df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "valid_time": times,
            "time_series_type": ["Primary"] * 3,
            "power_fcst_init_time": times,
            "power": [10.0, 12.0, 11.0],
            "power_lag_336h": [9.0, 11.0, 10.5],
        }
    )
    train_data = pt.LazyFrame.from_existing(train_df.lazy()).set_model(AllFeatures)
    config = XGBoostConfig(
        selected_features={"power_lag_336h"}, experiment_name="live_test", n_estimators=5
    )
    forecaster = XGBoostForecaster(config)
    forecaster.train(train_data, time_series_ids=[1])
    forecaster.save(path)
    write_trained_metadata(
        model_dir=path, time_series_metadata=_metadata_for((1,), (_TRAINED_CELL,))
    )


def test_live_power_history_covers_the_longest_selected_power_lag(
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
    tmp_path: Path,
) -> None:
    """``LIVE_POWER_HISTORY`` must reach back far enough for the longest power lag a real model
    selects.

    Every other test in this file trains its fixture model on ``{"temperature_2m"}``, so a
    ``LIVE_POWER_HISTORY`` too short to cover a real lag feature is invisible to them — the real
    model config, ``conf/model/xgboost.yaml``, selects ``power_lag_336h`` as of writing, and
    shrinking the window from 15 days to 1 hour leaves every other test in this file green. This
    test hard-codes that same lag rather than reading the YAML, trains a fixture model on it
    directly, and spies on ``XGBoostForecaster.predict`` to capture the exact frame
    ``live_forecasts`` hands it — the frame *after* production's own
    ``valid_time > power_fcst_init_time`` / ``ensemble_member is not null`` filter runs, so this
    test observes that filter's real output rather than re-deriving it — and pins that the
    captured frame carries no nulls in that column.
    """
    captured: list[pt.LazyFrame[AllFeatures]] = []
    real_predict = XGBoostForecaster.predict

    def _spy_predict(
        self: XGBoostForecaster, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        captured.append(data)
        return real_predict(self, data, fold_id=fold_id)

    monkeypatch.setattr(XGBoostForecaster, "predict", _spy_predict)

    _write_power_with_long_history(str(tmp_path / "NGED" / "power_time_series.delta"))
    _save_model_trained_on_power_lag(tmp_path / "production_model")

    assert _materialize(dagster_instance, "live").success

    (predicted_from,) = captured
    genuine = predicted_from.collect()
    assert genuine.height > 0
    assert genuine["power_lag_336h"].null_count() == 0


def _save_model_trained_on_weather_lag(path: Path, lag_hours: int) -> None:
    """A tiny real ``XGBoostForecaster`` selecting a weather lag feature, trained on ts1 only."""
    times = [datetime(2025, 1, 1, hour, tzinfo=UTC) for hour in (0, 1, 2)]
    feature_name = f"temperature_2m_lag_{lag_hours}h"
    train_df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "valid_time": times,
            "time_series_type": ["Primary"] * 3,
            "power_fcst_init_time": times,
            "power": [10.0, 12.0, 11.0],
            feature_name: [9.0, 11.0, 10.5],
        }
    )
    train_data = pt.LazyFrame.from_existing(train_df.lazy()).set_model(AllFeatures)
    config = XGBoostConfig(
        selected_features={feature_name}, experiment_name="live_test", n_estimators=5
    )
    forecaster = XGBoostForecaster(config)
    forecaster.train(train_data, time_series_ids=[1])
    forecaster.save(path)
    write_trained_metadata(
        model_dir=path, time_series_metadata=_metadata_for((1,), (_TRAINED_CELL,))
    )


@pytest.mark.parametrize(
    ("gap_hours", "expect_null"),
    [
        pytest.param(6, True, id="6h_gap_below_the_publication_delay"),
        pytest.param(9, False, id="9h_gap_at_the_publication_delay"),
    ],
)
def test_live_weather_lag_nulls_only_when_the_selected_run_is_too_fresh(
    monkeypatch: pytest.MonkeyPatch,
    dagster_instance: DagsterInstance,
    tmp_path: Path,
    gap_hours: int,
    expect_null: bool,
) -> None:
    """Pins today's asymmetry between "live" availability and the single-run publication-delay cut.

    ``live_forecasts`` selects its one NWP run in ``"live"`` availability mode, which applies no
    modelled delay — any run genuinely present in the Delta table qualifies. But the single-run
    analysis-proxy inside ``_engineer_features`` still gates its freshest-run join with the
    ``NWP_PUBLICATION_DELAY_HOURS`` (9h) ``available_at`` cut. When the selected run is closer
    than 9 hours to ``power_fcst_init_time`` — as at the 06:00 slot, when only that morning's run
    has landed — the cut excludes the run entirely from the freshest-run join and a weather lag
    targeting an earlier time goes null; a run 9 hours or further back passes the cut and the lag
    is populated. This pins the behaviour as it stands, not as a design endorsement: whether
    "live" mode should apply the same delay is a serving-path semantic decision, tracked
    separately from this test.
    """
    gap = timedelta(hours=gap_hours)
    power_fcst_init_time = datetime(2026, 7, 4, 6, 0, tzinfo=UTC)
    partition_key = "2026-07-04-00:00"  # window start; window end (= power_fcst_init_time) is 6h on
    nwp_init = power_fcst_init_time - gap
    forecast_valid_time = power_fcst_init_time + timedelta(minutes=30)
    target_time = forecast_valid_time - gap  # == nwp_init + 30 min: inside the run's own coverage

    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    production_model_path = tmp_path / "production_model"

    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(tmp_path / "NWP"))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(tmp_path / "power_forecasts"))
    monkeypatch.setenv("PRODUCTION_MODEL_PATH", str(production_model_path))

    _write_power_for(str(nged_path / "power_time_series.delta"), (1,))
    records = nwp_records(
        _TRAINED_CELL,
        nwp_init,
        (0,),
        init_time=nwp_init,
        valid_times=[target_time, forecast_valid_time],
    )
    write_test_nwp(str(tmp_path / "NWP"), records)
    _save_model_trained_on_weather_lag(production_model_path, gap_hours)

    captured: list[pt.LazyFrame[AllFeatures]] = []
    real_predict = XGBoostForecaster.predict

    def _spy_predict(
        self: XGBoostForecaster, data: pt.LazyFrame[AllFeatures], *, fold_id: str = "live"
    ) -> pt.DataFrame[PowerForecast]:
        captured.append(data)
        return real_predict(self, data, fold_id=fold_id)

    monkeypatch.setattr(XGBoostForecaster, "predict", _spy_predict)

    result = materialize(
        [live_forecasts],
        partition_key=partition_key,
        run_config=RunConfig(ops={"live_forecasts": LiveForecastsConfig(availability_mode="live")}),
        instance=dagster_instance,
    )
    assert result.success

    (predicted_from,) = captured
    genuine = predicted_from.collect()
    assert genuine.height > 0
    lag_col = f"temperature_2m_lag_{gap_hours}h"
    lag_is_entirely_null = genuine[lag_col].null_count() == genuine.height
    assert lag_is_entirely_null == expect_null
