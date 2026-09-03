"""Sentry.io telemetry: error reporting, the missed-check-in alarm, and freshness warnings.

Three independent mechanisms, all no-ops when Sentry is unconfigured, so laptops and CI need no
Sentry configuration: error telemetry (:func:`init_sentry`, :data:`sentry_capture_failure`,
:func:`report_asset_degradation`, :func:`report_check_degradation`), the missed-check-in alarm
(:func:`send_forecast_checkin`), and freshness warnings (:func:`report_power_freshness`). See [Send
telemetry to Sentry, and alarm on
absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
for what each mechanism carries, means, and is routed on.
"""

import logging
from typing import TYPE_CHECKING, Final, TypedDict

import sentry_sdk
from contracts.settings import Settings
from dagster import (
    DagsterExecutionInterruptedError,
    HookContext,
    RetryRequested,
    failure_hook,
)
from sentry_sdk.crons import capture_checkin
from sentry_sdk.crons.consts import MonitorStatus
from sentry_sdk.integrations.logging import LoggingIntegration

if TYPE_CHECKING:
    from sentry_sdk._types import MonitorConfig

    # Type-only import: importing this at runtime would be circular (defs/checks.py imports
    # report_power_freshness from this module). Annotate as a string below. Deferring the pure
    # freshness core into its own module — so this could be a real import — is left until a second
    # consumer (the forecast-warnings delivery table) actually lands.
    from nged_substation_forecast.defs.checks import PowerFreshnessResult

logger = logging.getLogger(__name__)

LIVE_FORECAST_MONITOR_SLUG: Final[str] = "live-forecasts"
"""Slug of the Sentry cron monitor fed by ``live_forecasts``' success heartbeat.

Laptop testing must use a different, throwaway slug (e.g. ``"live-forecasts-test"``). See
[Separating laptop telemetry from
production](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#separating-laptop-telemetry-from-production)."""

LIVE_FORECAST_MONITOR_CONFIG: "Final[MonitorConfig]" = {  # noqa: UP037
    # DUPLICATED SCHEDULE: this crontab must match live_forecast_partitions.cron_schedule in
    # defs/production_assets.py — it is the cadence Sentry expects a heartbeat on, so it has to
    # track the cadence the live_forecasts asset actually runs on. The value is copied rather than
    # imported because defs/production_assets.py imports this module (for send_forecast_checkin), so
    # importing back would be a circular import. If you change the live schedule there, change it
    # here.
    "schedule": {"type": "crontab", "value": "0 0,6,12,18 * * *"},
    "timezone": "UTC",
    "checkin_margin": 120,
}
"""Declarative config upserted with each heartbeat: the 6-hourly live schedule plus 120 minutes of
grace after each expected check-in before that slot counts as missed. See [Send telemetry to
Sentry, and alarm on
absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
for the derived alarm timing and why ``max_runtime`` is not set.

The ``schedule`` crontab is a hand-kept copy of ``live_forecast_partitions.cron_schedule`` in
``defs/production_assets.py`` — see the inline comment above for why it is copied, not imported."""


def init_sentry(settings: Settings) -> None:
    """Initialise the Sentry SDK for this process, or do nothing if Sentry is disabled.

    A no-op when ``settings.sentry_dsn`` is empty. Called once per process at import of the Dagster
    definitions module. See [Send telemetry to Sentry, and alarm on
    absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
    for where that runs and what each mechanism sends, including why log-to-event capture is
    disabled here.

    Never raises: a malformed DSN must not stop the Dagster code location from loading, or the
    daemon runs no schedule and no forecast is produced at all — the worst possible outcome for a
    typo in one environment variable, and one Sentry itself cannot alert on. ``sentry_sdk.init``
    raises ``BadDsn`` early in ``Client.__init__``, during DSN parsing, before any global state is
    touched, so catching that failure leaves the SDK with a ``NonRecordingClient``
    (``is_active() == False``) — identical to never having called ``init`` at all. Every other
    sender in this module already treats an inactive/uninitialised client as a silent no-op, so
    none of them needs extra guarding as a consequence of this one being guarded.

    Args:
        settings: The project settings carrying the Sentry DSN, environment, and sample rate.
    """
    if not settings.sentry_dsn:
        return
    try:
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.sentry_environment,
            traces_sample_rate=settings.sentry_traces_sample_rate,
            send_default_pii=False,
            integrations=[LoggingIntegration(event_level=None)],
        )
    except Exception:
        # Telemetry is best-effort, but a genuine bug in here must still be visible, so log at ERROR
        # with the traceback rather than swallowing.
        logger.exception("Failed to initialise the Sentry SDK")


FAULT_CATEGORY_TAG: Final[str] = "fault_category"
"""Tag naming what kind of fault an event reports, so an alert rule can route by urgency.

Only :data:`sentry_capture_failure` sets it — the other three senders already carry a mark of their
own. The reasoning is on the design page linked from this module's docstring; the production alert
rules key off this value, so treat it as a contract rather than a label."""

RUN_FAILED_FAULT_CATEGORY: Final[str] = "run_failed"
"""The one :data:`FAULT_CATEGORY_TAG` value: a scheduled production job failed, so that cycle did
not run."""


@failure_hook
def sentry_capture_failure(context: HookContext) -> None:
    """Report a failed op/asset step to Sentry with its real exception and traceback.

    Runs in the run worker after a step raises, so ``context.op_exception`` is the live exception
    (traceback intact) rather than Dagster's serialized error info. Tagged
    ``fault_category=run_failed`` (see :data:`FAULT_CATEGORY_TAG`). A no-op when Sentry is
    uninitialised (empty DSN), because ``capture_exception`` needs an active Sentry client.

    A ``RetryRequested`` is unwrapped to its cause before reporting, and a cancelled or terminated
    run (``KeyboardInterrupt``, ``SystemExit``, ``DagsterExecutionInterruptedError``) is not
    reported at all. See [Send telemetry to Sentry, and alarm on
    absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
    for why those two cases are special-cased, and why the retry that lands here is raised in-band
    rather than via a Dagster ``RetryPolicy``.

    Args:
        context: The Dagster hook context for the failed step, carrying ``op_exception``.
    """
    exception = context.op_exception
    if exception is None:
        return
    if isinstance(exception, RetryRequested) and exception.__cause__ is not None:
        exception = exception.__cause__
    # Checked *after* the unwrap, so that a wrapped interrupt is still recognised. Nothing here
    # wraps one today — every guard in `defs/` re-raises the three types — but Dagster does it
    # itself for any op carrying a `RetryPolicy`, which is exactly the trap the design page warns
    # against, so the ordering costs nothing and removes one way of falling into it.
    if isinstance(exception, KeyboardInterrupt | SystemExit | DagsterExecutionInterruptedError):
        return
    _capture_tagged(
        tag=FAULT_CATEGORY_TAG,
        value=RUN_FAILED_FAULT_CATEGORY,
        exc=exception,
        failure_note="Failed to report a failed step to Sentry",
    )


def _capture_tagged(tag: str, value: str, exc: BaseException, failure_note: str) -> None:
    """Send ``exc`` to Sentry, tagged ``tag=value`` on a scope forked so the tag cannot leak.

    Args:
        tag: Tag name to set on the forked scope.
        value: Tag value — the check or asset name.
        exc: The exception to capture.
        failure_note: Logged with the traceback if Sentry itself fails.
    """
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag(key=tag, value=value)
            sentry_sdk.capture_exception(exc)
    except Exception:
        # Telemetry is best-effort, but a genuine bug in here must still be visible, so log at ERROR
        # with the traceback rather than swallowing.
        logger.exception(failure_note)


def report_asset_degradation(asset_name: str, exc: BaseException) -> None:
    """Report an asset that degraded rather than failing, as a Sentry error event.

    Sends ``exc`` as an ``error``-level event tagged ``degraded_asset=<asset_name>``. A no-op when
    Sentry is uninitialised (empty DSN), and never raises. See [Send telemetry to Sentry, and alarm
    on
    absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
    for which assets call this and why.

    Args:
        asset_name: The Dagster asset name, attached as a ``degraded_asset`` tag so events can be
            filtered per asset. Set on an isolated scope so it cannot leak into later events.
        exc: The exception the asset degraded on.
    """
    _capture_tagged(
        tag="degraded_asset",
        value=asset_name,
        exc=exc,
        failure_note=f"Failed to report the degraded {asset_name} asset to Sentry",
    )


def report_check_degradation(check_name: str, exc: BaseException) -> None:
    """Report an asset check that could not evaluate its own inputs, as a Sentry error event.

    Sends ``exc`` as an ``error``-level event tagged ``asset_check=<check_name>``, restoring the
    signal that :data:`sentry_capture_failure` no longer sends once a check's catch-all keeps the
    run from failing (rule 7 of
    [The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).
    A no-op when Sentry is uninitialised (empty DSN), and never raises: it is called from inside a
    check's own ``except`` handler, where a raise would escape the guard and fail the run outright —
    the very thing the handler exists to prevent. See [Send telemetry to Sentry, and alarm on
    absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
    for which checks call this and why.

    Args:
        check_name: The Dagster asset-check name, attached as a tag so events can be filtered per
            check. Set on an isolated scope so it cannot leak into later unrelated events.
        exc: The exception the check degraded on.
    """
    _capture_tagged(
        tag="asset_check",
        value=check_name,
        exc=exc,
        failure_note=f"Failed to report the degraded {check_name} check to Sentry",
    )


def send_forecast_checkin(
    settings: Settings, monitor_slug: str = LIVE_FORECAST_MONITOR_SLUG
) -> None:
    """Send a success heartbeat to the Sentry cron monitor, or do nothing if disabled.

    A no-op unless ``settings.sentry_monitor_forecasts`` is set (True only on the always-on
    production box). Sends a single ``OK`` check-in — never ``in_progress`` or ``error``. Never
    raises. See [Send telemetry to Sentry, and alarm on
    absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
    for the alarm this feeds and why this never raises.

    Args:
        settings: The project settings carrying ``sentry_monitor_forecasts`` and the environment.
        monitor_slug: The Sentry monitor to check in to. Defaults to the production
            ``live-forecasts`` monitor; laptop tests should pass a throwaway slug.
    """
    if not settings.sentry_monitor_forecasts:
        return
    try:
        capture_checkin(
            monitor_slug=monitor_slug,
            status=MonitorStatus.OK,
            monitor_config=LIVE_FORECAST_MONITOR_CONFIG,
        )
    except Exception:
        # Telemetry is best-effort, but a genuine bug in here must still be visible, so log at ERROR
        # with the traceback rather than swallowing.
        logger.exception("Failed to send the live-forecasts heartbeat check-in to Sentry")


POWER_DATA_STALE_FINGERPRINT: Final[str] = "nged-power-data-stale"
"""Stable fingerprint root for the power-data staleness warning.

Combined with ``Settings.sentry_environment`` (see :func:`report_power_freshness`) so every
deployment gets its *own* ongoing Sentry issue — Sentry's ``environment`` is a filter facet, not a
grouping dimension, so without the environment in the fingerprint every deployment (production and
each ``<name>-laptop``) would share one issue. A stable fingerprint also collapses the hourly
re-reports of an ongoing stall into that single issue rather than a fresh issue each hour."""

MAX_LATE_SERIES_IN_CONTEXT: Final[int] = 50
"""Cap on the number of late series listed in the Sentry event *context* (the structured payload).

Keeps the payload small at V2 scale (~2,500 series) where a whole-feed stall would otherwise attach
thousands of rows. The *true* late count is always carried by the ``n_late`` tag and context field,
so a capped list never makes a large stall look like a small one."""

MAX_LATE_SERIES_IN_MESSAGE: Final[int] = 20
"""Cap on the number of late series spelled out in the human-readable *message* body.

Smaller than the context cap: the message is the at-a-glance view (Sentry uses its first line as the
issue title), so it lists only the leading late series and how late each is, with an ``…and N more``
line pointing at the fuller context when it overflows."""


class _LateSeriesEntry(TypedDict):
    """One late series in the freshness event.

    Carries its id, when it was last seen (``"never"`` if it never reported), how many hours late
    it is (``None`` if it never reported), and its status.
    """

    time_series_id: int
    last_seen: str
    hours_late: float | None
    status: str


def report_power_freshness(settings: Settings, result: "PowerFreshnessResult") -> None:  # noqa: UP037
    """Forward per-series power-data staleness to Sentry as a warning, or do nothing if healthy.

    A no-op when Sentry is unconfigured (empty ``settings.sentry_dsn``) or when no series is late
    (``result.is_healthy``). Sends a single ``warning``-level event, fingerprinted per environment.
    Never raises: ``power_data_is_fresh`` calls this from inside its own catch-all, so a raise here
    would cost the *whole* freshness report — every late series, in the very hour they went late —
    rather than just this one event. See [Send telemetry to Sentry, and alarm on
    absence](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#send-telemetry-to-sentry-and-alarm-on-absence)
    for the fingerprinting, the message and context caps, and how recovery is signalled.

    Args:
        settings: The project settings carrying the Sentry DSN and environment.
        result: The freshness evaluation to report, reused verbatim from the asset check (never
            recomputed).
    """
    if not settings.sentry_dsn or result.is_healthy:
        return
    try:
        _capture_power_freshness_warning(settings, result)
    except Exception:
        # Catch-all is deliberate: telemetry is best-effort and must never fail the check. Log at
        # ERROR with the traceback (not a silent swallow) so a genuine bug — e.g. in the payload
        # build over result.late — is still visible.
        logger.exception("Failed to report power-data freshness to Sentry")


def _capture_power_freshness_warning(
    settings: Settings,
    result: "PowerFreshnessResult",  # noqa: UP037
) -> None:
    """Build and send the freshness warning event on an isolated Sentry scope.

    Split from :func:`report_power_freshness` so the latter's ``try``/``except`` wraps the whole
    payload build (iterating ``result.late``), not only the network send — a bug in the payload is
    the likelier raiser than ``capture_message`` itself.
    """
    late_preview = result.late.head(MAX_LATE_SERIES_IN_CONTEXT)
    late_series: list[_LateSeriesEntry] = [
        {
            "time_series_id": row["time_series_id"],
            "last_seen": "never" if row["last_seen"] is None else str(row["last_seen"]),
            "hours_late": None if row["hours_late"] is None else round(row["hours_late"], 1),
            "status": row["status"],
        }
        for row in late_preview.iter_rows(named=True)
    ]
    with sentry_sdk.new_scope() as scope:
        scope.fingerprint = [POWER_DATA_STALE_FINGERPRINT, settings.sentry_environment]
        scope.set_tag(key="n_late", value=result.n_late)
        scope.set_tag(key="n_stale", value=result.n_stale)
        scope.set_tag(key="n_never_reported", value=result.n_never)
        scope.set_context(
            "power_freshness",
            {
                "n_late": result.n_late,
                "n_stale": result.n_stale,
                "n_never_reported": result.n_never,
                "n_series_total": result.n_series_total,
                "threshold_hours": result.threshold_hours,
                "late_series_shown": len(late_series),
                "late_series": late_series,
            },
        )
        sentry_sdk.capture_message(_freshness_message(result, late_series), level="warning")


def _freshness_message(
    result: "PowerFreshnessResult",  # noqa: UP037
    late_series: list[_LateSeriesEntry],
) -> str:
    """Compose the warning message.

    The message is a one-line summary (Sentry's issue title) followed by the leading late series
    and how late each one is.

    The per-series lines are capped at :data:`MAX_LATE_SERIES_IN_MESSAGE`; if more series are late,
    a trailing ``…and N more`` line reports the remainder (with the fuller list in the event's
    ``power_freshness`` context). ``late_series`` is already ordered never-reported first, then
    most-stale first, so the message leads with the worst offenders.
    """
    summary = (
        f"NGED power data stale: {result.n_late}/{result.n_series_total} time series late "
        f"({result.n_stale} stale >{result.threshold_hours:.0f}h, {result.n_never} never reported)"
    )
    shown = late_series[:MAX_LATE_SERIES_IN_MESSAGE]
    lines = [summary, *(_late_series_line(entry) for entry in shown)]
    remaining = result.n_late - len(shown)
    if remaining > 0:
        lines.append(f"  …and {remaining} more (context lists up to {MAX_LATE_SERIES_IN_CONTEXT})")
    return "\n".join(lines)


def _late_series_line(entry: _LateSeriesEntry) -> str:
    """One human-readable line for a late series.

    States how many hours late the series is, or that it never reported (a null ``hours_late``
    marks a never-reported series).
    """
    hours_late = entry["hours_late"]
    if hours_late is None:
        return f"  • series {entry['time_series_id']}: never reported"
    return (
        f"  • series {entry['time_series_id']}: {hours_late:.1f}h late "
        f"(last seen {entry['last_seen']})"
    )
