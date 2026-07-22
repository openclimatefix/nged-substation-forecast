"""Tests for the ``power_data_is_fresh`` asset check and its pure evaluation core.

The pure ``evaluate_power_freshness`` cases need neither Dagster nor Delta. The final test drives
the real check end-to-end — writing a temp Delta table + metadata roster and invoking the
directly-callable ``@asset_check`` — so the Settings plumbing, the Delta scan, the roster read,
and the ``AssetCheckResult`` mapping are all exercised together.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.power_schemas import TimeSeriesMetadata
from dagster import AssetCheckResult, AssetCheckSeverity, TableMetadataValue

from nged_substation_forecast.defs import checks
from nged_substation_forecast.defs.checks import (
    PowerFreshnessResult,
    evaluate_power_freshness,
)

_NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)
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
    monkeypatch.setenv("NGED_S3_BUCKET_URL", "https://example.com")
    monkeypatch.setenv("NGED_S3_BUCKET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("NGED_S3_BUCKET_SECRET", "dummy")
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
    from contracts.settings import Settings

    # Freeze "now" so the fresh/stale split is deterministic regardless of wall-clock.
    now = datetime.now(timezone.utc)
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
    from contracts.settings import Settings

    now = datetime.now(timezone.utc)
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
    from contracts.settings import Settings

    sentinel = PowerFreshnessResult(
        n_series_total=8,
        n_stale=7,
        n_never=0,
        threshold_hours=24.0,
        late=pl.DataFrame(
            {
                "time_series_id": pl.Series(range(7), dtype=pl.Int32),
                "last_seen": pl.Series([datetime(2020, 1, 1, tzinfo=timezone.utc)] * 7).cast(
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
    now = datetime.now(timezone.utc)
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
