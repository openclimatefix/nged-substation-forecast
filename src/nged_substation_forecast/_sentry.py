"""Sentry.io telemetry: error reporting, the missed-check-in alarm, and freshness warnings.

Three independent mechanisms, all no-ops when Sentry is unconfigured, so laptops and CI need no
Sentry configuration:

- **Error telemetry** — :func:`init_sentry` initialises the SDK once per process (a no-op unless
  ``Settings.sentry_dsn`` is set), and the :data:`sentry_capture_failure` Dagster failure hook
  reports the real exception (with traceback) from inside the run worker.
  :func:`report_check_degradation` covers a production fault the hook cannot see, because it never
  fails a run: a standalone ``@asset_check`` that caught its own exception. The hook is used rather
  than Sentry's ``LoggingIntegration`` log-to-event capture — which :func:`init_sentry` explicitly
  disables — because Dagster logs a step failure without ``exc_info``, so the log-based path would
  yield a message-only event with no stack trace, *and* would fire for every ``ERROR`` log anywhere
  in the process (Dagster's own startup/step logs, ad-hoc materialisations, even a swallowed
  telemetry error), swamping Sentry with events the design never intended to send. The hook is
  attached to the *scheduled* asset jobs only, so it covers the unattended production workload;
  manual/backfill/experiment runs are watched by the operator at the Dagster UI, not Sentry.
- **The missed-check-in alarm** — :func:`send_forecast_checkin` sends a *success-only* heartbeat to
  a Sentry cron monitor after each live ``live_forecasts`` run. It is gated on
  ``Settings.sentry_monitor_forecasts`` (not the DSN), so a laptop with a DSN set for error testing
  never heartbeats the production monitor. Sentry fires the alarm on the *absence* of a heartbeat
  past the margin (a dead daemon cannot report itself); in-band run errors are handled by the
  failure hook above, never by this heartbeat.
- **Freshness warnings** — :func:`report_power_freshness` forwards the ``power_data_is_fresh`` asset
  check's per-series staleness to Sentry as a *warning*-level event when any series is late. Gated
  on the DSN like error telemetry (not the heartbeat flag), and fingerprinted per environment so
  each deployment gets its own ongoing issue. Because a warning event models only one direction of
  a two-way state (stale vs recovered), recovery is signalled by the events *stopping* and is
  resolved by the operator — see the design page's freshness section.

The ``environment`` tag (``Settings.sentry_environment``) separates deployments: ``production`` on
the always-on box, ``<name>-laptop`` on each developer's machine. The alarm's alert rule is scoped
to ``environment:production`` so intermittently-run laptops never page. See
[Production Deployment](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/).
"""

import logging
from typing import TYPE_CHECKING, Final, TypedDict

import sentry_sdk
from contracts.settings import Settings
from dagster import HookContext, failure_hook
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

Laptop testing must use a *different*, throwaway slug (e.g. ``"live-forecasts-test"``) so an
intermittently-run laptop never registers a stale environment on the production monitor."""

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
grace after each expected check-in before that slot counts as missed — so the alarm effectively
fires ~8 hours after the last success (one 6-hourly slot plus the 2-hour margin). Only *success*
check-ins are ever sent, so the alarm keys off missed check-ins alone; ``max_runtime`` is
deliberately omitted because it needs an ``in_progress`` check-in to time against, which the
success-only heartbeat never sends.

The ``schedule`` crontab is a hand-kept copy of ``live_forecast_partitions.cron_schedule`` in
``defs/production_assets.py`` — see the inline comment above for why it is copied, not imported."""


def init_sentry(settings: Settings) -> None:
    """Initialise the Sentry SDK for this process, or do nothing if Sentry is disabled.

    A no-op when ``settings.sentry_dsn`` is empty (the default), so nothing is sent from laptops
    or CI unless a DSN is explicitly configured. Called once per process at import of the Dagster
    definitions module, so it runs in the daemon, the webserver, and every run worker.

    Log-to-event capture is deliberately switched off: passing a ``LoggingIntegration`` with
    ``event_level=None`` overrides the SDK's default integration (whose default ``event_level`` is
    ``ERROR``), so ``ERROR``-level log records no longer become Sentry events. Without this, every
    ``ERROR`` log anywhere in the process — Dagster's startup/step logs, ad-hoc materialisations,
    even the swallowed telemetry error in :func:`report_power_freshness` — would be shipped as an
    event, defeating the design where *only* the four explicit senders reach Sentry: the failure
    hook, the freshness ``capture_message``, :func:`report_check_degradation` and
    :func:`report_asset_degradation`. Breadcrumbs (the integration's default ``level=INFO``) are
    kept, so log context still rides along with the events those senders do send.

    Args:
        settings: The project settings carrying the Sentry DSN, environment, and sample rate.
    """
    if not settings.sentry_dsn:
        return
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
        integrations=[LoggingIntegration(event_level=None)],
    )


@failure_hook
def sentry_capture_failure(context: HookContext) -> None:
    """Report a failed op/asset step to Sentry with its real exception and traceback.

    Runs in the run worker after a step raises, so ``context.op_exception`` is the live exception
    (traceback intact) rather than Dagster's serialized error info. A no-op when Sentry is
    uninitialised (empty DSN), because ``capture_exception`` needs an active Sentry client.

    Args:
        context: The Dagster hook context for the failed step, carrying ``op_exception``.
    """
    exception = context.op_exception
    if exception is not None:
        sentry_sdk.capture_exception(exception)


def _capture_tagged(
    tag: str, value: str, exc_or_reason: BaseException | str, failure_note: str
) -> None:
    """Send ``exc_or_reason`` to Sentry on a scope forked for this one event, tagged ``tag=value``.

    Shared by :func:`report_check_degradation` and :func:`report_asset_degradation`, which differ
    only in their tag and in whether they always hold a live exception. Forking the scope is what
    keeps the tag from leaking into later unrelated events.

    Never raises: both callers run inside an ``except`` handler whose whole purpose is to keep the
    run alive, so an exception escaping here would undo that.

    Args:
        tag: Tag name to set on the forked scope.
        value: Tag value — the check or asset name.
        exc_or_reason: The exception to capture, or a message to send at ``error`` level when
            the caller has only a reason string.
        failure_note: Logged with the traceback if Sentry itself fails.
    """
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_tag(tag, value)
            if isinstance(exc_or_reason, BaseException):
                sentry_sdk.capture_exception(exc_or_reason)
            else:
                sentry_sdk.capture_message(exc_or_reason, level="error")
    except Exception:
        # Telemetry is best-effort, but a genuine bug in here must still be visible, so log at ERROR
        # with the traceback rather than swallowing.
        logger.exception(failure_note)


def report_asset_degradation(asset_name: str, exc_or_reason: BaseException | str) -> None:
    """Report an asset that degraded rather than failing, as a Sentry error event.

    ``power_time_series_and_metadata`` catches a failure of the ``TimeSeriesMetadata`` roster upsert
    so that the power write still happens (rule 1 of
    [The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)).
    Not failing the step means :data:`sentry_capture_failure` no longer fires, and log-to-event
    capture is disabled, so without this the degradation would reach nobody.

    Accepts a reason string as well as an exception because the two call sites differ: the asset's
    own guard holds a live exception, while a roster rebuilt inside ``upsert_metadata`` is reported
    by reason, the exception having been handled in ``nged_data`` — which must not depend on Sentry
    or Dagster.

    A no-op when Sentry is uninitialised (empty DSN), and never raises.

    Args:
        asset_name: The Dagster asset name, attached as a ``degraded_asset`` tag so events can be
            filtered per asset. Set on an isolated scope so it cannot leak into later events.
        exc_or_reason: The exception the asset degraded on, or a one-line reason.
    """
    _capture_tagged(
        tag="degraded_asset",
        value=asset_name,
        exc_or_reason=exc_or_reason,
        failure_note=f"Failed to report the degraded {asset_name} asset to Sentry",
    )


def report_check_degradation(check_name: str, exc: BaseException) -> None:
    """Report an asset check that could not evaluate its own inputs, as a Sentry error event.

    Both of this function's callers — the standalone ``@asset_check``s ``power_data_is_fresh`` and
    ``live_forecasts_are_healthy`` — catch everything rather than raising (rule 7 of
    [The rules](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/#the-rules)),
    so the run no longer fails and :data:`sentry_capture_failure` no longer fires. Without this,
    a check that cannot read its own inputs would show up only as a yellow tick in Dagster's Checks
    view — and ``power_time_series_and_metadata_job`` has no cron monitor to notice the silence, so
    nobody would be told. Sending the same exception the failure hook used to send restores the
    signal without restoring the failure.

    A no-op when Sentry is uninitialised (empty DSN), like every other path here, and never raises:
    it is called from inside a check's own ``except`` handler, where a raise would escape the guard
    and fail the run outright — the very thing the handler exists to prevent.

    Args:
        check_name: The Dagster asset-check name, attached as a tag so events can be filtered per
            check. Set on an isolated scope so it cannot leak into later unrelated events.
        exc: The exception the check degraded on.
    """
    _capture_tagged(
        tag="asset_check",
        value=check_name,
        exc_or_reason=exc,
        failure_note=f"Failed to report the degraded {check_name} check to Sentry",
    )


def send_forecast_checkin(
    settings: Settings, monitor_slug: str = LIVE_FORECAST_MONITOR_SLUG
) -> None:
    """Send a success heartbeat to the Sentry cron monitor, or do nothing if disabled.

    A no-op unless ``settings.sentry_monitor_forecasts`` is set (True only on the always-on
    production box), so a laptop never registers a monitor environment that Sentry would then mark
    as missed. Sends a single ``OK`` check-in — never ``in_progress`` or ``error`` — so the alarm
    fires purely on the *absence* of a heartbeat.

    Args:
        settings: The project settings carrying ``sentry_monitor_forecasts`` and the environment.
        monitor_slug: The Sentry monitor to check in to. Defaults to the production
            ``live-forecasts`` monitor; laptop tests should pass a throwaway slug.
    """
    if not settings.sentry_monitor_forecasts:
        return
    capture_checkin(
        monitor_slug=monitor_slug,
        status=MonitorStatus.OK,
        monitor_config=LIVE_FORECAST_MONITOR_CONFIG,
    )


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
    (``result.is_healthy``), so a healthy feed and an un-configured environment both stay silent.
    Sends a single ``warning``-level event fingerprinted per environment; the hourly re-reports of
    an ongoing stall collapse into one issue, and recovery is signalled by the events stopping (a
    warning event has no "resolved" counterpart — the operator resolves the issue). Never raises:
    ``power_data_is_fresh`` calls this from inside its own catch-all, so a raise here would not
    fail the run but would cost the *whole* freshness report — every late series, in the very hour
    they went late — by degrading the check to "could not evaluate". Guarding here keeps a
    telemetry hiccup from taking the evaluation down with it.

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
        scope.set_tag("n_late", result.n_late)
        scope.set_tag("n_stale", result.n_stale)
        scope.set_tag("n_never_reported", result.n_never)
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
