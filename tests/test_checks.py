"""Tests for the two data-health asset checks and their pure evaluation cores.

The pure ``evaluate_power_freshness`` / ``count_missed_nwp_runs`` /
``evaluate_live_forecast_health`` cases need neither Dagster nor Delta. Each check also has
end-to-end tests that drive the real ``@asset_check`` — writing temp Delta tables, a metadata
roster and a promoted-model ``meta.json`` — so the Settings plumbing, the Delta scans and the
``AssetCheckResult`` mapping are all exercised together.

The one path not reachable from here is ``live_forecasts_are_healthy`` running *partitioned*:
``build_asset_check_context`` cannot carry a partition key, so that path is covered by
``tests/test_live_forecasts.py::test_check_passes_after_a_real_live_materialisation``, which
materialises the asset and its check together for a real partition.
"""

import json
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import TimeSeriesMetadata
from contracts.settings import Settings
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    TableMetadataValue,
    build_asset_check_context,
)
from deltalake import write_deltalake

from nged_substation_forecast.defs import checks
from nged_substation_forecast.defs.checks import (
    LiveForecastHealthResult,
    LiveForecastRows,
    PowerFreshnessResult,
    count_missed_nwp_runs,
    evaluate_live_forecast_health,
    evaluate_power_freshness,
)

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
_THRESHOLD = timedelta(hours=24)


def _coverage(rows: dict[int, datetime]) -> pl.DataFrame:
    """Build a coverage frame (``time_series_id``, ``first_time``, ``last_time``) like the Delta
    scan returns. ``rows`` maps id -> last_time; first_time is irrelevant to freshness, so it is
    set a day earlier just to populate the column realistically."""
    last = pl.Series("last_time", list(rows.values())).cast(UTC_DATETIME_DTYPE)
    return pl.DataFrame(
        {
            "time_series_id": pl.Series(list(rows.keys()), dtype=pl.Int32),
            "first_time": last.dt.offset_by("-1d"),
            "last_time": last,
        }
    )


def _roster(ids: list[int]) -> pl.Series:
    return pl.Series("time_series_id", ids, dtype=pl.Int32)


def test_all_fresh_is_healthy() -> None:
    coverage = _coverage({1: _NOW - timedelta(hours=1), 2: _NOW - timedelta(hours=23)})
    result = evaluate_power_freshness(coverage, _roster([1, 2]), _NOW, _THRESHOLD)
    assert result.is_healthy
    assert result.n_late == 0
    assert result.n_series_total == 2
    assert result.late.is_empty()


def test_stale_series_flagged_most_stale_first() -> None:
    coverage = _coverage(
        {
            1: _NOW - timedelta(hours=1),  # fresh
            2: _NOW - timedelta(hours=30),  # stale
            3: _NOW - timedelta(hours=72),  # more stale
        }
    )
    result = evaluate_power_freshness(coverage, _roster([1, 2, 3]), _NOW, _THRESHOLD)
    assert not result.is_healthy
    assert result.n_stale == 2
    assert result.n_never == 0
    assert result.n_late == 2
    # Most-stale first: id 3 (72h) before id 2 (30h).
    assert result.late["time_series_id"].to_list() == [3, 2]
    assert result.late["status"].to_list() == ["stale", "stale"]
    assert result.late["hours_late"].to_list() == pytest.approx([72.0, 30.0])


def test_never_reported_ids_flagged_from_roster() -> None:
    """A roster id with no rows in the Delta table counts as late with a null ``last_seen``."""
    coverage = _coverage({1: _NOW - timedelta(hours=1)})
    result = evaluate_power_freshness(coverage, _roster([1, 2, 3]), _NOW, _THRESHOLD)
    assert not result.is_healthy
    assert result.n_stale == 0
    assert result.n_never == 2
    assert result.n_series_total == 3
    never_rows = result.late.filter(pl.col("status") == "never")
    assert set(never_rows["time_series_id"].to_list()) == {2, 3}
    assert never_rows["last_seen"].null_count() == 2
    assert never_rows["hours_late"].null_count() == 2


def test_never_reported_sorts_before_stale() -> None:
    coverage = _coverage({1: _NOW - timedelta(hours=48)})  # stale
    result = evaluate_power_freshness(coverage, _roster([1, 2]), _NOW, _THRESHOLD)
    assert result.late["status"].to_list() == ["never", "stale"]
    assert result.late["time_series_id"].to_list() == [2, 1]


def test_no_roster_cannot_detect_never_reported() -> None:
    """With no roster, only stale series are detectable; total is the on-disk id count."""
    coverage = _coverage({1: _NOW - timedelta(hours=48)})
    result = evaluate_power_freshness(coverage, None, _NOW, _THRESHOLD)
    assert result.n_never == 0
    assert result.n_stale == 1
    assert result.n_series_total == 1


def test_empty_table_with_roster_is_all_never() -> None:
    coverage = _coverage({})
    result = evaluate_power_freshness(coverage, _roster([1, 2]), _NOW, _THRESHOLD)
    assert result.n_never == 2
    assert result.n_series_total == 2
    assert not result.is_healthy


def test_result_threshold_hours_reflects_threshold() -> None:
    result = evaluate_power_freshness(_coverage({}), None, _NOW, timedelta(hours=8))
    assert isinstance(result, PowerFreshnessResult)
    assert result.threshold_hours == 8.0


# ---------------------------------------------------------------------------
# End-to-end: the real asset check against a temp Delta table + metadata roster.
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``Settings`` at a temp data root (mirrors ``tests/test_assets.py``)."""
    monkeypatch.setenv("DATA_PATH_INTERNAL", str(tmp_path))
    monkeypatch.setenv("DATA_PATH_DELIVERY", str(tmp_path))
    monkeypatch.setenv("LOCAL_ARTIFACTS_PATH", str(tmp_path))
    return tmp_path


def _write_metadata_roster(path: str, ids: list[int]) -> None:
    """Write a minimal valid ``TimeSeriesMetadata`` parquet covering ``ids``."""
    rows = [
        {
            "time_series_id": i,
            "time_series_name": f"Substation {i}",
            "time_series_type": "Disaggregated Demand",
            "units": "MW",
            "licence_area": "EMids",
            "substation_number": i,
            "substation_type": "Primary",
            "latitude": 52.0,
            "longitude": -1.0,
            "h3_res_5": 599423199024775167,
        }
        for i in ids
    ]
    TimeSeriesMetadata.DataFrame(rows).cast().validate().write_parquet(path)


def test_power_data_is_fresh_end_to_end(env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One fresh series, one stale series, one never-reported roster id → a WARN naming all late."""
    # Freeze "now" so the fresh/stale split is deterministic regardless of wall-clock.
    now = datetime.now(UTC)
    fresh_time = now - timedelta(hours=1)
    stale_time = now - timedelta(hours=48)

    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            "time": pl.Series([fresh_time, stale_time]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    _write_metadata_roster(settings.metadata_path, ids=[1, 2, 99])

    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)  # narrows the directly-invoked check's return

    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_stale"].value == 1  # id 2
    assert result.metadata["n_never_reported"].value == 1  # id 99
    assert result.metadata["n_late"].value == 2
    assert result.metadata["n_series_total"].value == 3
    late_table = result.metadata["late_time_series"]
    assert isinstance(late_table, TableMetadataValue)
    late_ids = {r.data["time_series_id"] for r in late_table.records}
    assert late_ids == {2, 99}


def test_power_data_is_fresh_all_current_passes(env: Path) -> None:
    now = datetime.now(UTC)
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1, 2], dtype=pl.Int32),
            "time": pl.Series([now - timedelta(hours=1), now - timedelta(hours=2)]).cast(
                UTC_DATETIME_DTYPE
            ),
            "power": pl.Series([1.0, 2.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    _write_metadata_roster(settings.metadata_path, ids=[1, 2])

    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)
    assert result.passed is True
    assert result.metadata["n_late"].value == 0


def test_power_data_is_fresh_no_data_yet_warns(env: Path) -> None:
    """No Delta table and no roster → not healthy, WARN, 'no data yet'."""
    result = checks.power_data_is_fresh()
    assert isinstance(result, AssetCheckResult)
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_series_total"].value == 0


def test_power_data_is_fresh_hands_evaluated_result_to_sentry(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check forwards the *exact* ``PowerFreshnessResult`` it evaluated to
    ``report_power_freshness`` — reused, not recomputed. We patch ``evaluate_power_freshness`` to
    return a sentinel and assert *object identity* on the reporter's argument: asserting counts
    alone would not prove reuse, since the pure function is deterministic and a recompute would
    yield equal counts and pass a weaker test. The reporter self-gates on health (tested in
    ``test_sentry.py``), so the check calls it unconditionally."""
    sentinel = PowerFreshnessResult(
        n_series_total=8,
        n_stale=7,
        n_never=0,
        threshold_hours=24.0,
        late=pl.DataFrame(
            {
                "time_series_id": pl.Series(range(7), dtype=pl.Int32),
                "last_seen": pl.Series([datetime(2020, 1, 1, tzinfo=UTC)] * 7).cast(
                    UTC_DATETIME_DTYPE
                ),
                "hours_late": pl.Series([9999.0] * 7, dtype=pl.Float64),
                "status": pl.Series(["stale"] * 7),
            }
        ),
    )
    monkeypatch.setattr(checks, "evaluate_power_freshness", lambda **_: sentinel)
    captured: list[PowerFreshnessResult] = []
    monkeypatch.setattr(
        checks, "report_power_freshness", lambda settings, result: captured.append(result)
    )

    # The check still reads coverage + roster before calling the (patched) evaluator, so a minimal
    # Delta table and roster must exist for those reads to succeed.
    now = datetime.now(UTC)
    settings = Settings()
    pl.DataFrame(
        {
            "time_series_id": pl.Series([1], dtype=pl.Int32),
            "time": pl.Series([now - timedelta(hours=1)]).cast(UTC_DATETIME_DTYPE),
            "power": pl.Series([1.0], dtype=pl.Float32),
        }
    ).write_delta(settings.power_time_series_data_path)
    _write_metadata_roster(settings.metadata_path, ids=[1])

    check_result = checks.power_data_is_fresh()
    assert isinstance(check_result, AssetCheckResult)
    assert len(captured) == 1
    assert captured[0] is sentinel  # the exact evaluated object, not a recomputation
    # ...and that same object drove the returned check result (n_late == n_stale + n_never == 7).
    assert check_result.metadata["n_late"].value == 7


# ---------------------------------------------------------------------------
# count_missed_nwp_runs: the pure missed-daily-run arithmetic.
# ---------------------------------------------------------------------------

_TODAY = datetime(2026, 7, 20, tzinfo=UTC)
"""Midnight of the arbitrary "today" the slot arithmetic tests are anchored on."""

_NWP_DOWNLOAD_HOUR = 9
"""The hour by which a healthy ``ecmwf_ens_schedule`` run (08:30 UTC) has landed the day's 00Z
run, used only to build the *on-disk* state each slot test starts from."""


def _daily_runs(latest: datetime, n: int = 6) -> list[datetime]:
    """``n`` consecutive daily 00Z runs ending at (and including) ``latest``."""
    return [latest - timedelta(days=i) for i in reversed(range(n))]


def _healthy_runs_at(slot: datetime) -> list[datetime]:
    """The NWP runs a *healthy* ingest has on disk at ``slot``.

    Today's 00Z run lands at 08:30 UTC, so the 00:00 and 06:00 slots still see only yesterday's
    run while the 12:00 and 18:00 slots see today's. That is the whole reason healthy NWP age
    spans 12–30 hours.
    """
    midnight = slot.replace(hour=0)
    latest = midnight if slot.hour >= _NWP_DOWNLOAD_HOUR else midnight - timedelta(days=1)
    return _daily_runs(latest)


@pytest.mark.parametrize(
    ("slot_hour", "healthy_age_hours"), [(0, 24.0), (6, 30.0), (12, 12.0), (18, 18.0)]
)
def test_no_missed_runs_at_any_healthy_slot(slot_hour: int, healthy_age_hours: float) -> None:
    """Every one of the four daily slots reports zero missed runs when the feed is healthy.

    This is the test that proves the check counts *runs* rather than hours: the same healthy
    feed presents as 12, 18, 24 or 30-hour-old NWP depending on the slot, so no single absolute
    age threshold could pass all four (one low enough to catch a genuine outage would fire on
    two slots in four, every day).
    """
    slot = _TODAY.replace(hour=slot_hour)
    on_disk = _healthy_runs_at(slot)
    assert (slot - max(on_disk)).total_seconds() / 3600.0 == healthy_age_hours

    result = count_missed_nwp_runs(on_disk, as_of=slot)
    assert result.n_missed == 0
    assert result.is_healthy
    assert result.latest_init_time == max(on_disk)


@pytest.mark.parametrize(
    ("slot_hour", "n_runs_dropped", "expected_missed"),
    [
        (0, 1, 1),
        (6, 1, 1),
        # The 12:00 slot is deliberately lenient by one run. The day's run is due on disk at
        # 08:30, but the deadline has to survive `ecmwf_ens`'s 4-hour retry window, so it only
        # demands the run by 14:00 — a download that failed today is therefore first reported at
        # the 18:00 slot rather than raising a false alarm on every slow-but-healthy morning.
        (12, 1, 0),
        (18, 1, 1),
        (0, 3, 3),
        (6, 3, 3),
        (12, 3, 2),
        (18, 3, 3),
    ],
)
def test_missed_runs_are_counted(slot_hour: int, n_runs_dropped: int, expected_missed: int) -> None:
    """Dropping the most recent ``n`` runs from a healthy feed is reported as missed runs."""
    slot = _TODAY.replace(hour=slot_hour)
    on_disk = _healthy_runs_at(slot)[:-n_runs_dropped]
    result = count_missed_nwp_runs(on_disk, as_of=slot)
    assert result.n_missed == expected_missed
    assert result.is_healthy is (expected_missed == 0)


def test_one_missed_run_persists_across_the_day_boundary() -> None:
    """A download that failed on day D is reported from D's 18:00 slot through D+1's 06:00 slot.

    The 06:00 slot the next morning — just before that day's download lands, when healthy NWP is
    at its stalest 30 hours — is the one most at risk of an off-by-one, so it is asserted
    explicitly alongside the slots either side of it.
    """
    failed_day = _TODAY
    latest_on_disk = failed_day - timedelta(days=1)  # D's run never landed
    for slot in (
        failed_day.replace(hour=18),
        failed_day + timedelta(days=1),
        failed_day + timedelta(days=1, hours=6),
    ):
        result = count_missed_nwp_runs(_daily_runs(latest_on_disk), as_of=slot)
        assert result.n_missed == 1, slot
        assert result.expected_latest_init_time == failed_day

    # ...and it clears as soon as a fresher run lands: the count measures the gap to the freshest
    # run that ought to exist, not holes left behind in history, because a fresher run is what
    # the next forecast will actually use.
    recovered = count_missed_nwp_runs(
        [*_daily_runs(latest_on_disk), failed_day + timedelta(days=1)],
        as_of=failed_day + timedelta(days=1, hours=12),
    )
    assert recovered.n_missed == 0


def test_runs_initialised_after_the_slot_are_ignored() -> None:
    """A run that only landed after the forecast was made could not have been used by it."""
    slot = _TODAY.replace(hour=12)
    usable = _TODAY - timedelta(days=1)
    result = count_missed_nwp_runs([usable, slot + timedelta(days=3)], as_of=slot)
    assert result.latest_init_time == usable
    assert result.n_missed == 0


def test_no_nwp_runs_at_all_is_unknown_not_zero() -> None:
    """With nothing usable on disk, "how many are missing" has no finite answer — and unknown is
    never healthy."""
    result = count_missed_nwp_runs([], as_of=_TODAY.replace(hour=12))
    assert result.n_missed is None
    assert result.latest_init_time is None
    assert not result.is_healthy


# ---------------------------------------------------------------------------
# evaluate_live_forecast_health: the pure "did this slot write usable rows?" core.
# ---------------------------------------------------------------------------

_SLOT = _TODAY.replace(hour=12)
"""The ``power_fcst_init_time`` the live-forecast health cases are written against."""

_HEALTHY_NWP = _healthy_runs_at(_SLOT)


_HEALTHY_ROWS = LiveForecastRows(
    n_rows=4800,
    n_invalid=0,
    n_nonfinite_power=0,
    n_hindcast=0,
    n_ensemble_members=51,
    time_series_ids=(1, 2),
    latest_valid_time=_SLOT + timedelta(days=14),
    nwp_init_time=_SLOT - timedelta(hours=12),
)
"""A healthy slot's row summary; each case below varies one field with ``dataclasses.replace``."""


def _evaluate(
    rows: LiveForecastRows, expected_ids: Sequence[int] | None = (1, 2)
) -> LiveForecastHealthResult:
    return evaluate_live_forecast_health(
        rows, _HEALTHY_NWP, power_fcst_init_time=_SLOT, expected_time_series_ids=expected_ids
    )


def test_healthy_slot_is_healthy() -> None:
    result = _evaluate(_HEALTHY_ROWS)
    assert result.is_healthy
    assert result.horizon_hours == 14 * 24
    assert result.missing_time_series_ids == ()
    assert result.n_expected_time_series == 2


def test_zero_rows_is_not_healthy() -> None:
    """The failure the check exists for: the asset succeeded but nothing landed on disk."""
    result = _evaluate(
        replace(
            _HEALTHY_ROWS,
            n_rows=0,
            n_ensemble_members=0,
            time_series_ids=(),
            latest_valid_time=None,
        ),
        expected_ids=None,
    )
    assert not result.is_healthy
    assert result.horizon_hours is None


@pytest.mark.parametrize("n_bad", [1, 500])
def test_non_finite_power_forecasts_are_invalid(n_bad: int) -> None:
    """Whatever made the rows unusable, the evaluator sees only this one count and reddens.

    Which values the count covers is the Polars expression's job, pinned end-to-end by
    ``test_live_forecasts_check_warns_on_non_finite_forecasts`` and
    ``test_live_forecasts_check_warns_on_null_forecasts``.
    """
    result = _evaluate(replace(_HEALTHY_ROWS, n_invalid=n_bad, n_nonfinite_power=n_bad))
    assert not result.is_healthy
    assert result.rows.n_invalid == n_bad


def test_hindcast_rows_are_invalid() -> None:
    """``valid_time <= power_fcst_init_time`` is forbidden by ``PowerForecast`` itself."""
    result = _evaluate(replace(_HEALTHY_ROWS, n_invalid=7, n_hindcast=7))
    assert not result.is_healthy
    assert result.rows.n_invalid == 7


def test_a_truncated_forecast_horizon_is_not_healthy() -> None:
    """Well-formed rows that stop a few hours ahead are still an undeliverable forecast.

    Structural, not skill: ``live_forecasts`` drops rows outside the NWP run's coverage, so a
    partly-ingested run shortens the delivered forecast without malforming a single row.
    """
    result = _evaluate(replace(_HEALTHY_ROWS, latest_valid_time=_SLOT + timedelta(hours=6)))
    assert result.horizon_is_truncated
    assert not result.is_healthy


def test_a_nearly_full_horizon_is_healthy() -> None:
    """The floor is loose enough that a normal slot — which stops just short of the 14 days asked
    for, because the NWP run is already 12–30 hours old — never trips it."""
    result = _evaluate(replace(_HEALTHY_ROWS, latest_valid_time=_SLOT + timedelta(days=13.5)))
    assert not result.horizon_is_truncated
    assert result.is_healthy


def test_missing_trained_time_series_is_not_healthy() -> None:
    result = _evaluate(replace(_HEALTHY_ROWS, time_series_ids=(1,)), expected_ids=[1, 2, 3])
    assert not result.is_healthy
    assert result.missing_time_series_ids == (2, 3)
    assert result.n_expected_time_series == 3


def test_unknown_trained_population_does_not_gate_the_check() -> None:
    """An unreadable promoted-model ``meta.json`` weakens the check rather than reddening it."""
    result = _evaluate(replace(_HEALTHY_ROWS, time_series_ids=(1,)), expected_ids=None)
    assert result.is_healthy
    assert result.n_expected_time_series is None


def test_missed_nwp_runs_make_an_otherwise_good_slot_unhealthy() -> None:
    """Valid rows built from stale NWP still degrade silently today — which is what this reports."""
    result = evaluate_live_forecast_health(
        _HEALTHY_ROWS,
        _daily_runs(_TODAY - timedelta(days=4)),  # 00Z runs, the freshest four days old
        power_fcst_init_time=_SLOT,
        expected_time_series_ids=[1, 2],
    )
    assert not result.is_healthy
    assert result.rows.n_invalid == 0  # the rows themselves are fine
    assert result.nwp.n_missed == 3


def test_check_result_is_a_non_blocking_warning() -> None:
    """Severity is WARN whether the slot is healthy or not; there is no ERROR path."""
    for rows in (_HEALTHY_ROWS, replace(_HEALTHY_ROWS, n_rows=0)):
        result = checks._to_live_forecast_check_result(_evaluate(rows))
        assert result.severity == AssetCheckSeverity.WARN
    (spec,) = checks.live_forecasts_are_healthy.check_specs
    assert spec.blocking is False
    assert spec.asset_key.to_user_string() == "live_forecasts"


# ---------------------------------------------------------------------------
# End-to-end: the real live-forecast check against temp Delta tables.
# ---------------------------------------------------------------------------


def _current_slot() -> datetime:
    """The slot an unpartitioned invocation of the check reports on: ``now`` floored to 6 hours."""
    now = datetime.now(UTC)
    return now.replace(hour=now.hour // 6 * 6, minute=0, second=0, microsecond=0)


_EXPERIMENT = "test_experiment"
"""The promoted model's experiment; the check scopes its read to this Delta partition."""

_FULL_HORIZON = timedelta(days=13, hours=18)
"""How far ahead a healthy slot's furthest row reaches — just short of the 14 days
``live_forecasts`` asks for, because the NWP run it used is already 12–30 hours old."""


def _write_live_forecasts(
    path: str,
    power_fcst_init_time: datetime,
    *,
    time_series_ids: tuple[int, ...] = (1, 2),
    power_fcst: float | None = 12.5,
    valid_time_offset: timedelta = _FULL_HORIZON,
    fold_id: str = "live",
    experiment_name: str = _EXPERIMENT,
    with_nwp_init_time: bool = True,
    mode: Literal["error", "append"] = "error",
) -> None:
    """Write a minimal ``power_forecasts`` Delta table partitioned the way production writes it.

    Only the columns the check reads are written: the genuine ``write_power_forecasts`` path is
    exercised by ``tests/test_live_forecasts.py`` instead, and a full ``PowerForecast`` frame here
    would obscure what each case is actually varying. ``mode="append"`` adds a second experiment's
    rows to an existing table. ``with_nwp_init_time=False`` omits the column entirely, which is
    what a model using no NWP writes — ``PowerForecast`` marks it ``allow_missing=True``.
    """
    rows = [
        {
            "experiment_name": experiment_name,
            "fold_id": fold_id,
            "power_fcst_init_time": power_fcst_init_time,
            "valid_time": power_fcst_init_time + valid_time_offset,
            "time_series_id": ts_id,
            "ensemble_member": member,
            "power_fcst": power_fcst,
        }
        | (
            {"nwp_init_time": power_fcst_init_time - timedelta(hours=12)}
            if with_nwp_init_time
            else {}
        )
        for ts_id in time_series_ids
        for member in (0, 1)
    ]
    schema = {
        "experiment_name": pl.String,
        "fold_id": pl.String,
        "power_fcst_init_time": UTC_DATETIME_DTYPE,
        "valid_time": UTC_DATETIME_DTYPE,
        "time_series_id": pl.Int32,
        "ensemble_member": pl.Int8,
        "power_fcst": pl.Float32,
    }
    if with_nwp_init_time:
        schema["nwp_init_time"] = UTC_DATETIME_DTYPE
    frame = pl.DataFrame(rows, schema=schema)
    write_deltalake(
        table_or_uri=path,
        data=frame.to_arrow(),
        partition_by=["experiment_name", "fold_id"],
        mode=mode,
    )


def _write_nwp_runs(path: str, init_times: list[datetime]) -> None:
    """Write an NWP Delta table partitioned by ``init_time``, which is all the check reads."""
    frame = pl.DataFrame(
        {
            "nwp_model_id": ["ECMWF_ENS_0_25_degree"] * len(init_times),
            "init_time": pl.Series(init_times).cast(UTC_DATETIME_DTYPE),
            "temperature_2m": pl.Series([1.0] * len(init_times), dtype=pl.Float32),
        }
    )
    write_deltalake(
        table_or_uri=path, data=frame.to_arrow(), partition_by=["nwp_model_id", "init_time"]
    )


def _write_promoted_model_meta(
    production_model_path: str, ids: list[int], experiment_name: str = _EXPERIMENT
) -> None:
    """Write just enough of a promoted model's ``meta.json``, in ``BaseForecaster.save``'s shape."""
    directory = Path(production_model_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "trained_time_series_ids": ids,
                "model_params": {"experiment_name": experiment_name},
            }
        )
    )


def _run_live_check() -> AssetCheckResult:
    result = checks.live_forecasts_are_healthy(build_asset_check_context())
    assert isinstance(result, AssetCheckResult)  # narrows the directly-invoked check's return
    return result


def test_live_forecasts_are_healthy_end_to_end(env: Path) -> None:
    """A slot with valid rows for the whole trained population and a complete NWP feed passes."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot)
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is True
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_rows"].value == 4
    assert result.metadata["n_time_series"].value == 2
    assert result.metadata["n_missed_nwp_runs"].value == 0
    assert result.metadata["n_invalid_rows"].value == 0


def test_live_forecasts_check_warns_when_the_slot_wrote_nothing(env: Path) -> None:
    """Rows exist for an *earlier* slot but not this one — the asset "succeeded" writing nothing."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot - timedelta(hours=6))
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_rows"].value == 0
    assert "no forecast rows" in str(result.description)


def test_live_forecasts_check_ignores_cv_fold_rows(env: Path) -> None:
    """Rows written for a CV fold at the same init time must not be mistaken for the live slot."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, fold_id="mid_2025_to_mid_2026")
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_rows"].value == 0


def test_live_forecasts_check_ignores_a_previous_experiments_rows(env: Path) -> None:
    """Rows left behind by the outgoing champion's experiment must not be read as this slot's.

    ``write_power_forecasts`` replaces one ``(experiment_name, fold_id)`` partition at a time, so
    promoting a model from a different experiment leaves the old experiment's rows for the same
    slot on disk indefinitely. Here those stale rows are all NaN: without the experiment filter
    they would be reported as this slot's fault.
    """
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(
        settings.power_forecasts_data_path,
        slot,
        experiment_name="outgoing_experiment",
        power_fcst=float("nan"),
    )
    _write_live_forecasts(settings.power_forecasts_data_path, slot, mode="append")
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is True
    assert result.metadata["n_rows"].value == 4  # the promoted experiment's rows only
    assert result.metadata["n_nonfinite_power"].value == 0


def test_live_forecasts_check_warns_on_a_truncated_horizon(env: Path) -> None:
    """Well-formed rows that stop hours rather than days ahead are an undeliverable forecast."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(
        settings.power_forecasts_data_path, slot, valid_time_offset=timedelta(hours=6)
    )
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_invalid_rows"].value == 0  # every individual row is well-formed
    assert result.metadata["forecast_horizon_hours"].value == 6.0
    assert "reaches only" in str(result.description)


@pytest.mark.parametrize("bad_power", [float("nan"), float("inf")])
def test_live_forecasts_check_warns_on_non_finite_forecasts(env: Path, bad_power: float) -> None:
    """Rows exist, but every ``power_fcst`` is unusable."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, power_fcst=bad_power)
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_nonfinite_power"].value == 4
    assert result.metadata["n_invalid_rows"].value == 4


def test_live_forecasts_check_warns_on_null_forecasts(env: Path) -> None:
    """A null ``power_fcst`` counts as non-finite, which Kleene logic makes easy to get wrong.

    ``~is_finite()`` is *null* — not true — for a null input, and ``sum()`` skips nulls, so the
    explicit ``is_null()`` arm is the only thing that catches the single likeliest way for the
    asset to succeed while writing garbage. Without it this slot would read as perfectly healthy.
    """
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, power_fcst=None)
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_nonfinite_power"].value == 4
    assert result.metadata["n_invalid_rows"].value == 4


def test_a_row_that_is_both_non_finite_and_a_hindcast_is_counted_once(env: Path) -> None:
    """``n_invalid`` is the union of the two faults, not their sum.

    Every row here is null *and* valid in the past, so both counts are 4 while ``n_invalid`` must
    stay 4: summing would report 8 invalid rows out of 4, breaking ``n_invalid <= n_rows``.
    """
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(
        settings.power_forecasts_data_path,
        slot,
        power_fcst=None,
        valid_time_offset=-timedelta(hours=1),
    )
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_rows"].value == 4
    assert result.metadata["n_nonfinite_power"].value == 4
    assert result.metadata["n_hindcast_rows"].value == 4
    assert result.metadata["n_invalid_rows"].value == 4


def test_live_forecasts_check_reports_on_a_model_that_writes_no_nwp_init_time(env: Path) -> None:
    """An absent optional column costs that one field, not the whole report.

    ``PowerForecast.nwp_init_time`` is ``allow_missing=True`` — a persistence baseline writes no
    such column — so the check must still judge the rows it *can* see rather than failing to
    evaluate anything.
    """
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, with_nwp_init_time=False)
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is True
    assert result.metadata["n_rows"].value == 4
    assert result.metadata["n_time_series"].value == 2
    assert result.metadata["n_invalid_rows"].value == 0
    assert result.metadata["nwp_init_time_on_rows"].value == "None"


def test_live_forecasts_check_degrades_when_meta_json_names_no_experiment(env: Path) -> None:
    """Both promoted-model facts degrade together, never one without the other.

    Keeping ``trained_time_series_ids`` while losing ``experiment_name`` would gate the population
    check on a count drawn from an *unfiltered* read of every live experiment's rows.
    """
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, time_series_ids=(1,))
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    directory = Path(settings.production_model_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps({"trained_time_series_ids": [1, 2]}))

    result = _run_live_check()
    assert result.passed is True
    assert "n_time_series_expected" not in result.metadata
    assert "n_time_series_missing" not in result.metadata


def test_live_forecasts_check_warns_on_hindcast_rows(env: Path) -> None:
    """A row valid at or before its own init time is undeliverable, whatever its value."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(
        settings.power_forecasts_data_path, slot, valid_time_offset=-timedelta(hours=1)
    )
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_hindcast_rows"].value == 4


def test_live_forecasts_check_warns_on_a_missing_trained_series(env: Path) -> None:
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, time_series_ids=(1,))
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    _write_promoted_model_meta(settings.production_model_path, [1, 2, 3])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_time_series_missing"].value == 2
    assert "2, 3" in str(result.metadata["missing_time_series_ids"].value)


def test_live_forecasts_check_reports_missed_nwp_runs_end_to_end(env: Path) -> None:
    """Valid rows, but the NWP feed has been down for days."""
    settings = Settings()
    slot = _current_slot()
    stale_runs = _daily_runs(slot.replace(hour=0) - timedelta(days=5))
    _write_live_forecasts(settings.power_forecasts_data_path, slot)
    _write_nwp_runs(settings.nwp_data_path, stale_runs)
    _write_promoted_model_meta(settings.production_model_path, [1, 2])

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_rows"].value == 4
    assert result.metadata["n_invalid_rows"].value == 0
    # The arithmetic itself is pinned by the pure tests above; what this asserts is that the same
    # answer survives the Delta partition read (the exact figure is 4 or 5 depending on which
    # 6-hourly slot the test happens to run in).
    expected_missed = count_missed_nwp_runs(stale_runs, as_of=slot).n_missed
    assert expected_missed is not None
    assert expected_missed >= 4
    assert result.metadata["n_missed_nwp_runs"].value == expected_missed
    assert "daily run(s) behind" in str(result.description)


def test_live_forecasts_check_does_not_raise_with_no_tables_at_all(env: Path) -> None:
    """A brand-new deployment: no forecasts table, no NWP table, no promoted model.

    The check must warn rather than raise — it is non-blocking, but it runs inside the hooked
    ``live_forecasts_job``, so a raise here would fail the production run it is only meant to
    comment on.
    """
    result = _run_live_check()
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert result.metadata["n_rows"].value == 0
    # Unknown counts are omitted rather than replaced by text, so each numeric metadata key keeps
    # one type across runs and stays plottable; the description carries the explanation instead.
    assert "n_missed_nwp_runs" not in result.metadata
    assert "n_time_series_expected" not in result.metadata
    # An unreadable meta.json makes "how many are missing" as unknown as "how many are expected" —
    # 0 here would be indistinguishable from a verified-complete population.
    assert "n_time_series_missing" not in result.metadata
    assert "missing_time_series_ids" not in result.metadata
    assert "forecast_horizon_hours" not in result.metadata
    assert "no NWP run is available" in str(result.description)


def test_live_forecasts_check_degrades_on_a_malformed_meta_json(env: Path) -> None:
    """A ``trained_time_series_ids`` that isn't a list is malformed, not merely missing.

    ``int(i) for i in ids`` would silently mis-parse a corrupted string (e.g. ``"12"`` iterates
    character-by-character into ``(1, 2)``) instead of degrading — this pins that the check
    rejects that shape and falls back to an unknown population, per CLAUDE.md's "strict about
    malformed inputs" rule.
    """
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, time_series_ids=(1,))
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    directory = Path(settings.production_model_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps({"trained_time_series_ids": "12", "model_params": {"experiment_name": None}})
    )

    result = _run_live_check()
    assert result.passed is True  # the single time series written is not flagged as missing
    assert "n_time_series_expected" not in result.metadata
    assert "n_time_series_missing" not in result.metadata


def test_live_forecasts_check_degrades_on_invalid_json(env: Path) -> None:
    """A ``meta.json`` that isn't valid JSON at all degrades the same way as one that's absent."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot, time_series_ids=(1,))
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))
    directory = Path(settings.production_model_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text("{not valid json")

    result = _run_live_check()
    assert result.passed is True
    assert "n_time_series_expected" not in result.metadata


def test_live_forecasts_check_does_not_raise_on_an_empty_table(env: Path) -> None:
    """An existing but row-less ``power_forecasts`` table is as harmless as no table at all."""
    settings = Settings()
    slot = _current_slot()
    _write_live_forecasts(settings.power_forecasts_data_path, slot)
    pl.read_delta(settings.power_forecasts_data_path).clear().write_delta(
        settings.power_forecasts_data_path, mode="overwrite"
    )
    _write_nwp_runs(settings.nwp_data_path, _healthy_runs_at(slot))

    result = _run_live_check()
    assert result.passed is False
    assert result.metadata["n_rows"].value == 0


def test_live_forecasts_check_contains_an_internal_error(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a bug inside the check itself must surface as a warning, never as a raise."""

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated bug inside the check")

    monkeypatch.setattr(checks, "_read_live_forecast_rows", _boom)
    result = _run_live_check()
    assert result.passed is False
    assert result.severity == AssetCheckSeverity.WARN
    assert "simulated bug inside the check" in str(result.description)
