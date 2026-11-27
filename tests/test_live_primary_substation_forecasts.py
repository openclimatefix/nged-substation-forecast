"""Tests of the ``live_primary_substation_forecasts`` asset, on a synthetic bundle and tiny tables.

The bundle has two Primaries with random weights (``_step5_bundle_test_data``): Primary 101 is in
MVA and Primary 102 in MW. The ``nwp`` table holds two members of each daily 00Z run on the two
Primaries' cells, at the ENS lead schedule out to 360 hours.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest
from _cleaned_power_test_data import write_cleaned_copy
from _nwp_test_data import nwp_records, write_test_nwp
from _step5_bundle_test_data import PRIMARIES, write_bundle
from contracts.power_schemas import POWER_FCST_QUANTILE_COLUMNS
from contracts.settings import Settings
from dagster import AssetExecutionContext, DagsterInstance, RetryRequested, materialize
from substation_demand_forecaster.bundle import load_bundle
from substation_demand_forecaster.features import (
    INPUT_STEPS,
    STEPS_PER_DAY,
    grid_index,
    lag_indices,
    native_origin,
)

from nged_substation_forecast.defs import primary_substation_forecast_assets as module
from nged_substation_forecast.defs.primary_substation_forecast_assets import (
    MAX_RETRIES,
    contract_rows,
    describe_day_run_faults,
    live_primary_substation_forecasts,
)

pytestmark = pytest.mark.integration

_DAY = datetime(2026, 4, 17, tzinfo=UTC)
_PARTITION = "2026-04-17"
_INIT = _DAY + timedelta(hours=12)
_CELLS = tuple(primary["h3_res_5"] for primary in PRIMARIES)
_MEMBERS = (0, 1)


def _run_records(init_time: datetime, cells: tuple[int, ...] = _CELLS) -> list[dict]:
    return [
        record
        for cell in cells
        for record in nwp_records(
            cell,
            init_time,
            _MEMBERS,
            init_time=init_time,
            valid_times=[init_time, init_time + timedelta(hours=360)],
        )
    ]


def _write_nwp(path: Path, *, days_back: list[int]) -> None:
    records = [record for back in days_back for record in _run_records(_DAY - timedelta(days=back))]
    write_test_nwp(str(path), records)


def _write_power(nged_path: Path) -> None:
    origin = native_origin(_DAY)
    times = pl.datetime_range(
        origin - timedelta(days=9), origin, interval="30m", time_zone="UTC", eager=True
    )
    raw = nged_path / "power_time_series.delta"
    pl.DataFrame(
        {
            "time_series_id": [p["time_series_id"] for p in PRIMARIES for _ in times],
            "time": [t for _ in PRIMARIES for t in times],
            "power": [0.5 * p["capacity_mw"] for p in PRIMARIES for _ in times],
        }
    ).cast(
        {"time_series_id": pl.Int32, "time": pl.Datetime("us", "UTC"), "power": pl.Float32}
    ).write_delta(str(raw), delta_write_options={"partition_by": "time_series_id"})
    write_cleaned_copy(raw)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point every setting the asset reads at temporary tables and a synthetic bundle."""
    nged_path = tmp_path / "NGED"
    nged_path.mkdir()
    paths = {
        "nwp": tmp_path / "NWP",
        "forecasts": tmp_path / "power_forecasts",
        "bundle": write_bundle(tmp_path / "bundle"),
    }
    monkeypatch.setenv("NGED_DATA_PATH", str(nged_path))
    monkeypatch.setenv("NWP_DATA_PATH", str(paths["nwp"]))
    monkeypatch.setenv("POWER_FORECASTS_DATA_PATH", str(paths["forecasts"]))
    monkeypatch.setenv("PRIMARY_SUBSTATION_BUNDLE_PATH", str(paths["bundle"]))
    _write_power(nged_path)
    return paths


@pytest.fixture
def degradations(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record every `report_asset_degradation` call the asset makes."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(module, "report_asset_degradation", lambda **kw: calls.append(kw))
    return calls


@pytest.fixture
def today() -> datetime:
    """Today's 00Z run, recent enough for the asset to retry."""
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def _us(stamp: datetime) -> np.ndarray:
    return np.array([int(stamp.timestamp() * 1_000_000)], dtype=np.int64)


def _materialize(instance: DagsterInstance, partition_key: str = _PARTITION) -> None:
    result = materialize(
        [live_primary_substation_forecasts], partition_key=partition_key, instance=instance
    )
    assert result.success


def test_writes_671_rows_per_primary_from_12_30_to_11_30_14_days_later(
    env: dict[str, Path],
    dagster_instance: DagsterInstance,
    degradations: list[dict[str, Any]],
) -> None:
    _write_nwp(env["nwp"], days_back=list(range(14)))
    _materialize(dagster_instance)

    rows = pl.read_delta(str(env["forecasts"]))
    assert rows.height == 2 * 671
    assert rows.group_by("time_series_id").len()["len"].to_list() == [671, 671]
    assert rows["valid_time"].min() == _INIT + timedelta(minutes=30)
    assert rows["valid_time"].max() == _INIT + timedelta(days=14, minutes=-30)
    assert (rows["valid_time"] > rows["power_fcst_init_time"]).all()
    assert (rows["power_fcst_init_time"] == _INIT).all()
    assert (rows["nwp_init_time"] == _DAY).all()
    assert rows["experiment_name"].unique().to_list() == ["primary_substation_demand_forecaster"]
    assert rows["fold_id"].unique().to_list() == ["live"]
    # Two members per cell is an incomplete run, and nothing else is missing.
    assert [call["fingerprint"][1] for call in degradations] == ["day_run_incomplete"]


def test_a_second_run_of_the_partition_replaces_its_rows(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    _write_nwp(env["nwp"], days_back=list(range(14)))
    _materialize(dagster_instance)
    _materialize(dagster_instance)
    assert pl.read_delta(str(env["forecasts"])).height == 2 * 671


def test_a_rerun_replaces_only_its_own_init_time_and_experiment(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    _write_nwp(env["nwp"], days_back=list(range(-1, 14)))
    _materialize(dagster_instance)
    path = str(env["forecasts"])
    # Another live forecaster's rows at the init time the re-run replaces.
    other = pl.read_delta(path).with_columns(
        pl.col("power_fcst_init_time", "valid_time", "nwp_init_time") + timedelta(days=1),
        experiment_name=pl.lit("another_live_forecaster"),
    )
    other.write_delta(path, mode="append")
    next_day = (_DAY + timedelta(days=1)).date().isoformat()
    _materialize(dagster_instance, partition_key=next_day)
    _materialize(dagster_instance, partition_key=next_day)

    rows = pl.read_delta(path)
    ours = rows.filter(pl.col("experiment_name") == module.EXPERIMENT_NAME)
    assert ours.group_by("power_fcst_init_time").len().sort("power_fcst_init_time")[
        "len"
    ].to_list() == [2 * 671, 2 * 671]
    assert rows.filter(pl.col("experiment_name") == "another_live_forecaster").height == 2 * 671


class _Noon(datetime):
    """`datetime` whose `now` is 13:00 UTC on `_DAY`, so `_PARTITION` is today's partition."""

    @classmethod
    def now(cls, tz: Any = None) -> _Noon:
        return cls(2026, 4, 17, 13, tzinfo=UTC)


@pytest.fixture
def checkins(monkeypatch: pytest.MonkeyPatch, env: dict[str, Path]) -> list[bool]:
    """Record each check-in, as whether the forecasts table existed when it was sent."""
    calls: list[bool] = []
    monkeypatch.setattr(
        module,
        "send_primary_substation_forecast_checkin",
        lambda settings: calls.append(env["forecasts"].exists()),
    )
    return calls


def test_todays_partition_checks_in_after_the_write(
    env: dict[str, Path],
    dagster_instance: DagsterInstance,
    degradations: list[dict[str, Any]],
    checkins: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "datetime", _Noon)
    _write_nwp(env["nwp"], days_back=list(range(14)))
    _materialize(dagster_instance)
    assert checkins == [True]


def test_a_past_partition_sends_no_checkin(
    env: dict[str, Path],
    dagster_instance: DagsterInstance,
    degradations: list[dict[str, Any]],
    checkins: list[bool],
) -> None:
    _write_nwp(env["nwp"], days_back=list(range(14)))
    _materialize(dagster_instance)
    assert env["forecasts"].exists()
    assert checkins == []


def test_a_missing_day_run_sends_no_checkin(
    env: dict[str, Path],
    dagster_instance: DagsterInstance,
    degradations: list[dict[str, Any]],
    checkins: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "datetime", _Noon)
    monkeypatch.setattr(module, "MAX_RETRIES", 0)  # today's run, so the first attempt is the last
    _write_nwp(env["nwp"], days_back=list(range(1, 14)))
    _materialize(dagster_instance)
    assert [call["fingerprint"][1] for call in degradations] == ["day_run_missing"]
    assert checkins == []


def test_a_bundle_past_its_refit_month_is_reported_once(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    write_bundle(env["bundle"], manifest_overrides={"refit_month": "2026-02"})
    _write_nwp(env["nwp"], days_back=list(range(14)))
    _materialize(dagster_instance)
    stale = [call for call in degradations if call["fingerprint"][1] == "bundle_stale"]
    assert len(stale) == 1
    assert "2 month(s) after its refit month" in str(stale[0]["exc"])


def test_expected_weather_rows_no_run_covers_are_reported(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    # Only the day's run and the run 1 day back: no run covers the lags' earlier half-hours.
    _write_nwp(env["nwp"], days_back=[0, 1])
    _materialize(dagster_instance)
    kinds = [call["fingerprint"][1] for call in degradations]
    assert "expected_weather_uncovered" in kinds
    uncovered = degradations[kinds.index("expected_weather_uncovered")]
    assert "expected-weather row(s) that no ENS run covers" in str(uncovered["exc"])


def test_a_missing_history_run_is_reported_and_the_forecast_still_written(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    _write_nwp(env["nwp"], days_back=[back for back in range(14) if back != 5])
    _materialize(dagster_instance)

    assert pl.read_delta(str(env["forecasts"])).height == 2 * 671
    history = [call for call in degradations if call["fingerprint"][1] == "history_runs_missing"]
    assert len(history) == 1
    assert "2026-04-12" in str(history[0]["exc"])


def test_an_absent_day_run_writes_nothing_and_reports_the_degradation(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    # Only the history runs: the partition's own run never arrived. The partition is more than
    # 36 hours old, so the asset does not retry.
    _write_nwp(env["nwp"], days_back=list(range(1, 14)))
    _materialize(dagster_instance)

    assert not env["forecasts"].exists()
    assert [call["fingerprint"][1] for call in degradations] == ["day_run_missing"]


def _fake_context(retry_number: int) -> AssetExecutionContext:
    """The two attributes `_wait_for_day_run` reads from a Dagster context."""
    namespace = SimpleNamespace(retry_number=retry_number, log=logging.getLogger(__name__))
    return cast(AssetExecutionContext, namespace)


def test_an_absent_recent_run_requests_a_retry_and_warns_once(
    monkeypatch: pytest.MonkeyPatch, today: datetime
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(module, "_available_nwp_init_times", lambda settings: [])
    monkeypatch.setattr(
        module, "report_asset_retry", lambda **kw: warnings.append(kw["asset_name"])
    )
    for retry_number in (0, 1):
        with pytest.raises(RetryRequested):
            module._wait_for_day_run(
                context=_fake_context(retry_number), settings=Settings(), nwp_init_time=today
            )
        # The first attempt warns, and the second attempt adds nothing.
        assert warnings == ["live_primary_substation_forecasts"]


def test_the_last_retry_reports_the_degradation_and_returns(
    monkeypatch: pytest.MonkeyPatch, degradations: list[dict[str, Any]], today: datetime
) -> None:
    monkeypatch.setattr(module, "_available_nwp_init_times", lambda settings: [])
    present = module._wait_for_day_run(
        context=_fake_context(MAX_RETRIES), settings=Settings(), nwp_init_time=today
    )
    assert present is False
    assert [call["fingerprint"][1] for call in degradations] == ["day_run_missing"]


def test_a_backfill_reads_no_run_newer_than_its_own(env: dict[str, Path]) -> None:
    _write_nwp(env["nwp"], days_back=[-2, -1, 0, 1])
    native = module._read_native_nwp(
        settings=Settings(), bundle=load_bundle(env["bundle"]), nwp_init_time=_DAY
    )
    assert native["init_time"].max() == _DAY
    assert native["init_time"].n_unique() == 2


def _fractions(values: Callable[[int], float], rows: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            column: [values(k) + 0.01 * j for j in range(rows)]
            for k, column in enumerate(POWER_FCST_QUANTILE_COLUMNS)
        }
    )


def test_contract_rows_scale_by_capacity_and_clamp_only_the_mva_primary(
    tmp_path: Path,
) -> None:
    bundle = load_bundle(write_bundle(tmp_path / "bundle"))
    valid = [_INIT + timedelta(minutes=30 * k) for k in (-1, 0, 1, 2)]
    keys = pl.DataFrame(
        {
            "primary_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "time_series_id": [101, 101, 101, 101, 102, 102, 102, 102],
            "valid_time": valid + valid,
        }
    ).cast({"time_series_id": pl.Int32, "valid_time": pl.Datetime("us", "UTC")})
    # Quantile k holds the fraction -0.3 + 0.05 k, so the low quantiles are negative.
    quantiles = _fractions(lambda k: -0.3 + 0.05 * k, rows=8)
    rows = contract_rows(keys=keys, quantiles=quantiles, bundle=bundle, nwp_init_time=_DAY)

    # The rows at and before 12:00 UTC are dropped.
    assert rows.height == 4
    assert (rows["valid_time"] > _INIT).all()
    mw = rows.filter(pl.col("time_series_id") == 102)
    expected_p1 = [(-0.3 + 0.01 * j) * 6.0 for j in (6, 7)]
    assert mw["power_fcst_p1"].to_list() == pytest.approx(expected_p1, rel=1e-6)
    mva = rows.filter(pl.col("time_series_id") == 101)
    assert mva["power_fcst_p1"].to_list() == [0.0, 0.0]
    assert mva["power_fcst_p99"].to_list() == pytest.approx(
        [(-0.3 + 0.05 * 12 + 0.01 * j) * 18.0 for j in (2, 3)], rel=1e-6
    )
    assert (rows["power_fcst_model_version"] == bundle.model_release).all()


def _day_run(cells: tuple[int, ...] = _CELLS) -> pl.DataFrame:
    frame = pl.DataFrame(_run_records(_DAY, cells=cells))
    return frame.with_columns(pl.col("init_time", "valid_time").cast(pl.Datetime("us", "UTC")))


def test_describe_day_run_faults_names_each_shortfall(tmp_path: Path) -> None:
    bundle = load_bundle(write_bundle(tmp_path / "bundle"))
    whole = pl.concat(
        [_day_run().with_columns(ensemble_member=pl.lit(member)) for member in range(51)]
    )
    assert describe_day_run_faults(native=whole, nwp_init_time=_DAY, bundle=bundle) == []

    lead_6 = _DAY + timedelta(hours=6)
    faulty = whole.filter(
        pl.col("h3_index") == _CELLS[0],
        pl.col("valid_time") != lead_6,
        pl.col("valid_time") <= _DAY + timedelta(hours=300),
    ).with_columns(
        downward_short_wave_radiation_flux_surface=pl.when(pl.col("ensemble_member") == 3)
        .then(None)
        .otherwise(pl.col("downward_short_wave_radiation_flux_surface"))
    )
    faulty = faulty.filter(pl.col("ensemble_member") != 50)
    faults = describe_day_run_faults(native=faulty, nwp_init_time=_DAY, bundle=bundle)
    assert len(faults) == 5
    assert str(_CELLS[1]) in faults[0]
    assert "fewer than 51 members" in faults[1]
    assert faults[2] == f"missing lead time(s) in hours, by H3 cell: {{{_CELLS[0]}: [6]}}"
    assert faults[3] == f"the run stops short of 360 h, by H3 cell: {{{_CELLS[0]}: 300}}"
    assert "null downward_short_wave_radiation_flux_surface" in faults[4]


def test_describe_day_run_faults_finds_a_lead_missing_from_one_cell_only(tmp_path: Path) -> None:
    bundle = load_bundle(write_bundle(tmp_path / "bundle"))
    whole = pl.concat(
        [_day_run().with_columns(ensemble_member=pl.lit(member)) for member in range(51)]
    )
    lead_6 = _DAY + timedelta(hours=6)
    faulty = whole.filter((pl.col("h3_index") != _CELLS[1]) | (pl.col("valid_time") != lead_6))
    faults = describe_day_run_faults(native=faulty, nwp_init_time=_DAY, bundle=bundle)
    assert faults == [f"missing lead time(s) in hours, by H3 cell: {{{_CELLS[1]}: [6]}}"]


def test_the_cleaned_power_read_reaches_every_week_lag(env: dict[str, Path]) -> None:
    bundle = load_bundle(env["bundle"])
    power = module._read_cleaned_power(settings=Settings(), bundle=bundle, nwp_init_time=_DAY)
    origin = grid_index(_us(native_origin(_DAY)))[0]
    target = origin + 1 + np.arange(INPUT_STEPS)
    _, week = lag_indices(
        target_index=target, origin_index=np.full(INPUT_STEPS, origin), lags=bundle.lags
    )
    read = set(grid_index(power["time"].dt.epoch("us").to_numpy()).tolist())
    # The earliest week lag is about 8 days before the origin.
    assert origin - week.min() > 7 * STEPS_PER_DAY
    assert set(week.tolist()) <= read


def test_the_written_quantiles_are_the_seed_mean(
    env: dict[str, Path],
    dagster_instance: DagsterInstance,
    degradations: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecasts: list[Any] = []
    written: list[pl.DataFrame] = []
    original_forecast, original_rows = module.forecast, module.contract_rows

    def record_forecast(**kwargs: Any) -> Any:
        forecasts.append(original_forecast(**kwargs))
        return forecasts[-1]

    def record_rows(**kwargs: Any) -> Any:
        written.append(kwargs["quantiles"])
        return original_rows(**kwargs)

    monkeypatch.setattr(module, "forecast", record_forecast)
    monkeypatch.setattr(module, "contract_rows", record_rows)
    _write_nwp(env["nwp"], days_back=list(range(14)))
    _materialize(dagster_instance)

    (out,) = forecasts
    assert out.per_seed.shape[0] == 2
    assert not np.allclose(out.per_seed[0], out.mean)
    np.testing.assert_array_equal(written[0].to_numpy(), out.mean)


def test_a_missing_run_one_day_back_is_named(
    env: dict[str, Path], dagster_instance: DagsterInstance, degradations: list[dict[str, Any]]
) -> None:
    _write_nwp(env["nwp"], days_back=[back for back in range(14) if back != 1])
    _materialize(dagster_instance)
    history = [call for call in degradations if call["fingerprint"][1] == "history_runs_missing"]
    assert len(history) == 1
    assert "missing 1 of the 13 history run(s) (2026-04-16)" in str(history[0]["exc"])


@pytest.mark.parametrize(
    ("refit_month", "day", "months"),
    [
        ("2026-12", datetime(2027, 1, 5, tzinfo=UTC), 1),
        ("2024-03", datetime(2026, 3, 31, tzinfo=UTC), 24),
        ("2026-04", datetime(2026, 4, 30, tzinfo=UTC), 0),
    ],
)
def test_months_past_refit_counts_calendar_months(
    refit_month: str, day: datetime, months: int
) -> None:
    bundle = cast("Any", SimpleNamespace(refit_month=refit_month))
    assert module.months_past_refit(bundle=bundle, nwp_init_time=day) == months
