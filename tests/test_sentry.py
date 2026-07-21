"""Unit tests for the Sentry telemetry helpers (``nged_substation_forecast._sentry``).

Every Sentry side effect is monkeypatched, so these tests never touch the network and assert the
two invariants that matter: everything is a no-op unless explicitly enabled, and when enabled the
right Sentry call is made with the right arguments.
"""

import logging
from datetime import datetime, timezone
from typing import Any

import polars as pl
import pytest
from contracts.common import UTC_DATETIME_DTYPE
from contracts.settings import Settings
from dagster import build_hook_context

from nged_substation_forecast import _sentry
from nged_substation_forecast.defs.checks import PowerFreshnessResult

_DSN = "https://k@o1.ingest.sentry.io/1"


def _settings(**overrides: Any) -> Settings:
    """Build a ``Settings`` with Sentry fields overridden (nged creds come from the conftest)."""
    return Settings(**overrides)


def _freshness_result(
    *, n_stale: int, n_never: int = 0, n_total: int = 100
) -> PowerFreshnessResult:
    """Build a ``PowerFreshnessResult`` with ``n_stale`` stale + ``n_never`` never-reported rows.

    The ``late`` frame mirrors what ``evaluate_power_freshness`` produces (stale rows carry a
    ``last_seen``/``hours_late``; never rows carry nulls), so the reporter's row-iteration is
    exercised realistically."""
    n_late = n_stale + n_never
    late = pl.DataFrame(
        {
            "time_series_id": pl.Series(range(n_late), dtype=pl.Int32),
            "last_seen": pl.Series(
                [datetime(2026, 7, 1, tzinfo=timezone.utc)] * n_stale + [None] * n_never
            ).cast(UTC_DATETIME_DTYPE),
            "hours_late": pl.Series([30.0] * n_stale + [None] * n_never, dtype=pl.Float64),
            "status": pl.Series(["stale"] * n_stale + ["never"] * n_never),
        }
    )
    return PowerFreshnessResult(
        n_series_total=n_total, n_stale=n_stale, n_never=n_never, threshold_hours=24.0, late=late
    )


def _capture_message_recorder(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ``capture_message`` to snapshot the *current scope* at call time.

    ``report_power_freshness`` sends inside ``with sentry_sdk.new_scope()``, so the current scope
    when ``capture_message`` fires is the forked scope carrying the fingerprint/tags/context we want
    to assert on (verified: ``get_current_scope()`` is that same object inside the block)."""
    calls: list[dict[str, Any]] = []

    def fake(message: str, level: str | None = None, **_: Any) -> None:
        scope = _sentry.sentry_sdk.get_current_scope()
        calls.append(
            {
                "message": message,
                "level": level,
                "fingerprint": scope._fingerprint,
                "tags": dict(scope._tags),
                "contexts": dict(scope._contexts),
            }
        )

    monkeypatch.setattr(_sentry.sentry_sdk, "capture_message", fake)
    return calls


def test_init_sentry_is_noop_without_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DSN (the default) ⇒ ``sentry_sdk.init`` is never called."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "init", lambda **kw: calls.append(kw))
    _sentry.init_sentry(_settings(sentry_dsn=""))
    assert calls == []


def test_init_sentry_passes_environment_when_dsn_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DSN ⇒ ``sentry_sdk.init`` is called with the configured environment and PII disabled."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "init", lambda **kw: calls.append(kw))
    _sentry.init_sentry(
        _settings(sentry_dsn="https://k@o1.ingest.sentry.io/1", sentry_environment="jacks-laptop")
    )
    assert len(calls) == 1
    assert calls[0]["dsn"] == "https://k@o1.ingest.sentry.io/1"
    assert calls[0]["environment"] == "jacks-laptop"
    assert calls[0]["send_default_pii"] is False


def test_send_forecast_checkin_is_noop_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """``sentry_monitor_forecasts`` False (the default) ⇒ no check-in is sent."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_sentry, "capture_checkin", lambda **kw: calls.append(kw))
    _sentry.send_forecast_checkin(_settings(sentry_monitor_forecasts=False))
    assert calls == []


def test_send_forecast_checkin_sends_ok_when_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flag on ⇒ a single ``OK`` check-in to the live-forecasts monitor, carrying the config."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_sentry, "capture_checkin", lambda **kw: calls.append(kw))
    _sentry.send_forecast_checkin(_settings(sentry_monitor_forecasts=True))
    assert len(calls) == 1
    assert calls[0]["monitor_slug"] == _sentry.LIVE_FORECAST_MONITOR_SLUG
    assert calls[0]["status"] == "ok"
    assert calls[0]["monitor_config"] == _sentry.LIVE_FORECAST_MONITOR_CONFIG


def test_send_forecast_checkin_uses_given_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    """A throwaway slug (laptop testing) is honoured instead of the production monitor."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_sentry, "capture_checkin", lambda **kw: calls.append(kw))
    _sentry.send_forecast_checkin(
        _settings(sentry_monitor_forecasts=True), monitor_slug="live-forecasts-test"
    )
    assert calls[0]["monitor_slug"] == "live-forecasts-test"


def test_failure_hook_captures_the_real_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure hook forwards ``context.op_exception`` to ``capture_exception``."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", lambda exc: captured.append(exc))
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    boom = ValueError("boom")
    hook_fn(build_hook_context(op_exception=boom))
    assert captured == [boom]


def test_failure_hook_noop_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """No exception on the context ⇒ nothing is captured."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", lambda exc: captured.append(exc))
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    hook_fn(build_hook_context(op_exception=None))
    assert captured == []


def test_failure_hook_is_attached_to_the_scheduled_jobs() -> None:
    """Regression guard: the failure hook stays wired onto every scheduled asset job, so dropping
    ``hooks={sentry_capture_failure}`` from a ``define_asset_job`` call is caught here."""
    from nged_substation_forecast.defs import schedules

    scheduled_jobs = (
        schedules.power_time_series_and_metadata_job,
        schedules.ecmwf_ens_job,
        schedules.live_forecasts_job,
    )
    for job in scheduled_jobs:
        hooks = job.hooks
        assert hooks is not None
        assert _sentry.sentry_capture_failure in hooks, job.name


def test_monitor_config_schedule_matches_live_partitions() -> None:
    """Drift guard: the Sentry monitor's crontab is a hand-kept copy of the live_forecasts
    partition schedule (the two can't share an import — it would be circular; see the comment on
    ``LIVE_FORECAST_MONITOR_CONFIG``). If someone changes one crontab and not the other, the alarm
    would expect heartbeats on a different cadence than the asset runs; this catches that."""
    from nged_substation_forecast.defs.production_assets import live_forecast_partitions

    assert (
        _sentry.LIVE_FORECAST_MONITOR_CONFIG["schedule"]["value"]
        == live_forecast_partitions.cron_schedule
    )


def test_report_power_freshness_noop_without_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DSN (the default) ⇒ no warning is sent even when series are late."""
    calls = _capture_message_recorder(monkeypatch)
    _sentry.report_power_freshness(_settings(sentry_dsn=""), _freshness_result(n_stale=3))
    assert calls == []


def test_report_power_freshness_noop_when_healthy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DSN but no late series ⇒ nothing sent (the reporter self-gates on health)."""
    calls = _capture_message_recorder(monkeypatch)
    _sentry.report_power_freshness(_settings(sentry_dsn=_DSN), _freshness_result(n_stale=0))
    assert calls == []


def test_report_power_freshness_sends_warning_with_fingerprint_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DSN + late series ⇒ one warning, fingerprinted per environment, with counts as tags/context."""
    calls = _capture_message_recorder(monkeypatch)
    _sentry.report_power_freshness(
        _settings(sentry_dsn=_DSN, sentry_environment="jacks-laptop"),
        _freshness_result(n_stale=2, n_never=1, n_total=32),
    )
    assert len(calls) == 1
    (call,) = calls
    assert call["level"] == "warning"
    # Environment IS in the fingerprint: Sentry's environment is a filter facet, not a grouping
    # dimension, so this is what gives each deployment its own issue.
    assert call["fingerprint"] == [_sentry.POWER_DATA_STALE_FINGERPRINT, "jacks-laptop"]
    # These are the python ints on the pre-serialization scope; Sentry str()s tag values on the
    # wire, so in the UI they filter as e.g. `n_late:3`.
    assert call["tags"] == {"n_late": 3, "n_stale": 2, "n_never_reported": 1}
    ctx = call["contexts"]["power_freshness"]
    assert ctx["n_late"] == 3
    assert ctx["n_series_total"] == 32
    assert ctx["late_series_shown"] == 3
    assert len(ctx["late_series"]) == 3
    assert "3/32" in call["message"]


def test_report_power_freshness_caps_context_but_keeps_true_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A whole-feed stall (more late series than the cap) ⇒ the listed rows are capped, but the
    true late count still surfaces via the tag and count field, so a big stall never looks small."""
    calls = _capture_message_recorder(monkeypatch)
    n_stale = _sentry.MAX_LATE_SERIES_IN_CONTEXT + 10
    _sentry.report_power_freshness(
        _settings(sentry_dsn=_DSN), _freshness_result(n_stale=n_stale, n_total=n_stale)
    )
    (call,) = calls
    ctx = call["contexts"]["power_freshness"]
    assert len(ctx["late_series"]) == _sentry.MAX_LATE_SERIES_IN_CONTEXT  # list capped
    assert ctx["late_series_shown"] == _sentry.MAX_LATE_SERIES_IN_CONTEXT
    assert ctx["n_late"] == n_stale  # true total preserved
    assert call["tags"]["n_late"] == n_stale


def test_report_power_freshness_swallows_and_logs_on_send_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A Sentry send error must not propagate (it would fail the data-health check and trip the
    failure hook), but it must be logged at ERROR with a traceback rather than silently swallowed."""

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("sentry down")

    monkeypatch.setattr(_sentry.sentry_sdk, "capture_message", boom)
    with caplog.at_level(logging.ERROR, logger="nged_substation_forecast._sentry"):
        _sentry.report_power_freshness(_settings(sentry_dsn=_DSN), _freshness_result(n_stale=1))
    assert any(
        "freshness" in r.message.lower() and r.levelno == logging.ERROR for r in caplog.records
    )
    assert any(r.exc_info is not None for r in caplog.records)  # traceback attached


def test_report_power_freshness_does_not_leak_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fingerprint/context live on an isolated ``new_scope()`` — after the call the current
    scope carries neither, so a later unrelated ``capture_exception`` can't inherit them."""
    _capture_message_recorder(monkeypatch)
    _sentry.report_power_freshness(_settings(sentry_dsn=_DSN), _freshness_result(n_stale=1))
    scope = _sentry.sentry_sdk.get_current_scope()
    assert scope._fingerprint is None
    assert scope._contexts.get("power_freshness") is None
