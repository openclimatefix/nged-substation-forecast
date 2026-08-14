"""Unit tests for the Sentry telemetry helpers (``nged_substation_forecast._sentry``).

These tests never touch the network — every Sentry side effect is monkeypatched, bar the one test
that needs a real client to build a real event, which drops it in ``before_send`` — and they assert
the two invariants that matter: everything is a no-op unless explicitly enabled, and when enabled
the right Sentry call is made with the right arguments.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import sentry_sdk
from contracts.common import UTC_DATETIME_DTYPE
from contracts.settings import Settings
from dagster import DagsterExecutionInterruptedError, RetryRequested, build_hook_context
from sentry_sdk.types import Event, Hint

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
                [datetime(2026, 7, 1, tzinfo=UTC)] * n_stale + [None] * n_never
            ).cast(UTC_DATETIME_DTYPE),
            "hours_late": pl.Series([30.0] * n_stale + [None] * n_never, dtype=pl.Float64),
            "status": pl.Series(["stale"] * n_stale + ["never"] * n_never),
        }
    )
    return PowerFreshnessResult(
        n_series_total=n_total, n_stale=n_stale, n_never=n_never, threshold_hours=24.0, late=late
    )


def _wrap_in_retry_request(cause: BaseException) -> RetryRequested:
    """Build what a guard's ``raise RetryRequested(...) from cause`` gives the failure hook."""
    retry = RetryRequested(max_retries=2)
    retry.__cause__ = cause
    return retry


def _hook_context_with(exception: BaseException | None) -> Any:
    """A duck-typed stand-in for ``HookContext``, carrying only what the failure hook reads.

    ``build_hook_context`` runs ``check.opt_inst_param(op_exception, ..., Exception)``, so it
    rejects every ``BaseException`` that is not an ``Exception`` — including the three deliberate
    exits the hook has to ignore."""
    return SimpleNamespace(op_exception=exception)


def _build_one_event(send: Callable[[], None]) -> Event:
    """Run ``send`` against a real client and return the single event it built.

    The assertion this enables is on the *built event*, not on the arguments to
    ``capture_exception``: a tag set on the wrong scope, or not set at all, still reaches
    ``capture_exception`` intact and would slip past an argument-level check. Building an event
    needs a real client, which is confined to a temporary isolation scope, and ``before_send``
    returns ``None`` so the event is dropped rather than transmitted. Both integration sets are off
    because ``setup_once`` is *irreversible* and process-global — it monkeypatches
    ``sys.excepthook``, ``threading.Thread.run``, ``logging.Logger.callHandlers`` and more, none of
    which leaving the scope would undo, and this suite uses threads (moto), logging (``caplog``)
    and sqlalchemy (the Dagster instance).
    """
    events: list[Event] = []

    def collect(event: Event, _hint: Hint) -> Event | None:
        """Record the built event and return ``None``, which tells the SDK to drop it."""
        events.append(event)
        return None

    with sentry_sdk.isolation_scope() as scope:
        scope.set_client(
            sentry_sdk.Client(
                dsn=_DSN,
                before_send=collect,
                default_integrations=False,
                auto_enabling_integrations=False,
            )
        )
        send()

    (event,) = events
    return event


def _assert_no_tags_leaked() -> None:
    """The tag lived on a scope forked for the one event, so it cannot leak into a later unrelated
    one — including via the isolation scope this whole Dagster process shares."""
    assert sentry_sdk.get_current_scope()._tags == {}
    assert sentry_sdk.get_isolation_scope()._tags == {}


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


def test_init_sentry_disables_log_to_event_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """A DSN ⇒ init installs a ``LoggingIntegration`` with event capture off (``event_level=None``).

    This is the guard that keeps ``ERROR`` logs from anywhere in the process — Dagster's
    startup/step logs, ad-hoc materialisations, the swallowed telemetry error in
    ``report_power_freshness`` — from becoming Sentry events. Only the four explicit senders (the
    failure hook, the freshness ``capture_message``, ``report_check_degradation`` and
    ``report_asset_degradation``) should ever send. If someone drops the ``integrations`` argument,
    the SDK's default ``LoggingIntegration`` (``event_level=ERROR``) comes back and this fails.
    """
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "init", lambda **kw: calls.append(kw))
    _sentry.init_sentry(_settings(sentry_dsn=_DSN))
    (kwargs,) = calls
    logging_integrations = [
        integration
        for integration in kwargs["integrations"]
        if isinstance(integration, _sentry.LoggingIntegration)
    ]
    assert len(logging_integrations) == 1
    # event_level=None ⇒ the integration builds no event handler, so ERROR logs aren't captured as
    # events (only breadcrumbs, via the still-default level=INFO breadcrumb handler).
    assert logging_integrations[0]._handler is None


def test_init_sentry_survives_a_malformed_dsn(caplog: pytest.LogCaptureFixture) -> None:
    """A malformed DSN must not stop the Dagster code location from loading — a typo in one
    environment variable would otherwise take down the whole service (no schedule, no forecasts),
    with Sentry itself unreachable so the only signal is the missed-check-in alarm hours later.

    Uses a real DSN string that genuinely makes ``sentry_sdk.init`` raise ``BadDsn``, rather than
    monkeypatching ``init`` to raise, so this pins the actual failure mode instead of passing even
    if real DSN parsing were fine.
    """
    with caplog.at_level(logging.ERROR, logger="nged_substation_forecast._sentry"):
        _sentry.init_sentry(_settings(sentry_dsn="not-a-dsn"))
    # One record must carry all three: a message naming what failed, ERROR level, and the
    # traceback. Asserting them separately would pass on three different records. The message
    # matters here because it is the only signal a malformed DSN ever produces.
    assert any(
        "Sentry SDK" in r.message and r.levelno == logging.ERROR and r.exc_info is not None
        for r in caplog.records
    )


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


def test_send_forecast_checkin_swallows_and_logs_on_send_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """This runs after ``live_forecasts`` has already committed its Delta write, so a raise here
    would leave the run reported as failed on a run that in fact produced everything. There is no
    input that reliably makes the real ``capture_checkin`` call fail offline, so the sender is
    monkeypatched to raise instead (unlike the DSN test above, which needs a real failure mode)."""

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("sentry down")

    monkeypatch.setattr(_sentry, "capture_checkin", boom)
    with caplog.at_level(logging.ERROR, logger="nged_substation_forecast._sentry"):
        _sentry.send_forecast_checkin(_settings(sentry_monitor_forecasts=True))
    # One record must carry all three: a message naming what failed, ERROR level, and the
    # traceback. Asserting them separately would pass on three different records.
    assert any(
        "heartbeat" in r.message and r.levelno == logging.ERROR and r.exc_info is not None
        for r in caplog.records
    )


def test_failure_hook_captures_the_real_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure hook forwards ``context.op_exception`` to ``capture_exception``."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", captured.append)
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    boom = ValueError("boom")
    hook_fn(build_hook_context(op_exception=boom))
    assert captured == [boom]


def test_failure_hook_noop_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """No exception on the context ⇒ nothing is captured."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", captured.append)
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    hook_fn(build_hook_context(op_exception=None))
    assert captured == []


def test_failure_hook_reports_the_cause_of_a_retry_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exhausted in-band retry must reach Sentry titled and grouped by the fault that actually
    happened, not by the ``RetryRequested`` wrapper Dagster hands the hook. Dagster unwraps only its
    own ``RetryRequestedFromPolicy``, so without this an ``ecmwf_ens`` run that never publishes
    would group as ``RetryRequested`` rather than ``NwpRunNotYetAvailable``."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", captured.append)
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    cause = OSError("upstream down")
    hook_fn(build_hook_context(op_exception=_wrap_in_retry_request(cause)))
    assert captured == [cause]


def test_failure_hook_captures_a_retry_requested_with_no_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``RetryRequested`` raised without ``from exc`` has no cause to unwrap to, so it is reported
    as itself rather than dereferencing ``None``."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", captured.append)
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    retry = RetryRequested(max_retries=2)
    hook_fn(build_hook_context(op_exception=retry))
    assert captured == [retry]


@pytest.mark.parametrize(
    "exception",
    [
        SystemExit(),
        DagsterExecutionInterruptedError(),
        KeyboardInterrupt(),
        _wrap_in_retry_request(DagsterExecutionInterruptedError()),
    ],
    ids=["system_exit", "interrupted", "keyboard_interrupt", "interrupt_wrapped_in_retry"],
)
def test_failure_hook_ignores_a_deliberate_exit(
    exception: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling a run is an operator's decision, not a fault, so none of these reports.

    ``SystemExit`` is the only shape that reaches the hook as things stand: it is neither a
    ``DagsterError`` nor an ``Exception``, so Dagster's interrupt handling does not re-raise it
    ahead of the hook the way it does for the other two. The remaining cases pin the guard's stated
    contract rather than a reachable state — and the wrapped one also pins the *ordering*, since
    reversing the unwrap and this check is the one mutation that only it catches."""
    captured: list[BaseException] = []
    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", captured.append)
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    hook_fn(_hook_context_with(exception))
    assert captured == []


def test_failure_hook_tags_the_fault_category() -> None:
    """A failed production run is the one class an operator should be alerted on, so it carries a
    positive marker rather than being identified by the *absence* of the degradation tags.

    Asserted on the built event, so a tag set on the wrong scope is caught — see
    ``_build_one_event``."""
    hook_fn = _sentry.sentry_capture_failure.decorated_fn
    assert hook_fn is not None
    event = _build_one_event(lambda: hook_fn(build_hook_context(op_exception=ValueError("boom"))))
    # Literals, not the module's own constants: the production alert rule is configured against
    # these exact strings, so a rename has to fail here rather than compare equal to itself.
    assert event["tags"] == {"fault_category": "run_failed"}
    assert event["exception"]["values"][0]["type"] == "ValueError"
    _assert_no_tags_leaked()


@pytest.mark.parametrize(
    ("report", "tag", "name"),
    [
        (
            lambda name, exc: _sentry.report_check_degradation(check_name=name, exc=exc),
            "asset_check",
            "power_data_is_fresh",
        ),
        (
            lambda name, exc: _sentry.report_asset_degradation(asset_name=name, exc=exc),
            "degraded_asset",
            "power_time_series_and_metadata",
        ),
    ],
    ids=["check", "asset"],
)
def test_degradation_reporters_capture_the_exception_and_tag_the_name(
    report: Callable[[str, BaseException], None], tag: str, name: str
) -> None:
    """A degraded check or asset sends the same exception the failure hook would have, tagged so the
    event is filterable per check or asset (``operations.md`` documents
    ``asset_check:power_data_is_fresh`` as the operator's Sentry filter). Without the capture, one
    that caught its own exception would reach nobody, log-to-event capture being disabled.

    The assertion is on the built event rather than on the arguments to ``capture_exception`` — see
    ``_build_one_event`` for why.
    """
    event = _build_one_event(lambda: report(name, ValueError("boom")))
    assert event["tags"] == {tag: name}
    assert event["exception"]["values"][0]["type"] == "ValueError"
    _assert_no_tags_leaked()


def test_report_check_degradation_swallows_and_logs_on_send_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """It is called from inside a check's ``except`` handler, so a raise here would escape the
    guard and fail the very run the handler exists to keep alive."""

    def boom(*_: Any, **__: Any) -> None:
        raise RuntimeError("sentry down")

    monkeypatch.setattr(_sentry.sentry_sdk, "capture_exception", boom)
    with caplog.at_level(logging.ERROR, logger="nged_substation_forecast._sentry"):
        _sentry.report_check_degradation("power_data_is_fresh", ValueError("boom"))
    assert any(
        "power_data_is_fresh" in r.message and r.levelno == logging.ERROR for r in caplog.records
    )
    assert any(r.exc_info is not None for r in caplog.records)  # traceback attached


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
    """DSN + late series ⇒ one warning, fingerprinted per environment, with counts as
    tags/context.
    """
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
    # The message spells out the summary AND each late series with how late it is.
    message = call["message"]
    assert "3/32" in message
    assert "series 0: 30.0h late" in message
    assert "last seen 2026-07-01" in message
    assert "series 2: never reported" in message
    assert "…and" not in message  # all 3 fit under the message cap, so no overflow line


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


def test_report_power_freshness_caps_series_listed_in_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More late series than the message cap ⇒ the message spells out only the leading
    ``MAX_LATE_SERIES_IN_MESSAGE`` and reports the rest as an ``…and N more`` line."""
    calls = _capture_message_recorder(monkeypatch)
    n_stale = _sentry.MAX_LATE_SERIES_IN_MESSAGE + 5
    _sentry.report_power_freshness(
        _settings(sentry_dsn=_DSN), _freshness_result(n_stale=n_stale, n_total=n_stale)
    )
    (call,) = calls
    message = call["message"]
    series_lines = [line for line in message.splitlines() if "• series" in line]
    assert len(series_lines) == _sentry.MAX_LATE_SERIES_IN_MESSAGE
    assert "…and 5 more" in message


def test_report_power_freshness_swallows_and_logs_on_send_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A Sentry send error must not propagate, but must be logged at ERROR with a traceback.

    Propagating would fail the data-health check and trip the failure hook; silently swallowing
    would hide the problem entirely.
    """

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
