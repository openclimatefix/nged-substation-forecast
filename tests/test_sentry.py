"""Unit tests for the Sentry telemetry helpers (``nged_substation_forecast._sentry``).

Every Sentry side effect is monkeypatched, so these tests never touch the network and assert the
two invariants that matter: everything is a no-op unless explicitly enabled, and when enabled the
right Sentry call is made with the right arguments.
"""

from typing import Any

import pytest
from contracts.settings import Settings
from dagster import build_hook_context

from nged_substation_forecast import _sentry


def _settings(**overrides: Any) -> Settings:
    """Build a ``Settings`` with Sentry fields overridden (nged creds come from the conftest)."""
    return Settings(**overrides)


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
