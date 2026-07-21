"""Sentry.io telemetry: error reporting and the missed-check-in alarm.

Two independent mechanisms, both no-ops unless ``Settings.sentry_dsn`` is set, so laptops and CI
need no Sentry configuration:

- **Error telemetry** — :func:`init_sentry` initialises the SDK once per process, and the
  :data:`sentry_capture_failure` Dagster failure hook reports the real exception (with traceback)
  from inside the run worker. The hook is used rather than relying on Sentry's ``LoggingIntegration``
  alone because Dagster logs a step failure without ``exc_info``, so the log-based path would yield a
  message-only event with no stack trace.
- **The missed-check-in alarm** — :func:`send_forecast_checkin` sends a *success-only* heartbeat to
  a Sentry cron monitor after each live ``live_forecasts`` run. Sentry fires the alarm on the
  *absence* of a heartbeat past the margin (a dead daemon cannot report itself); in-band run errors
  are handled by the failure hook above, never by this heartbeat.

The ``environment`` tag (``Settings.sentry_environment``) separates deployments: ``production`` on
the always-on box, ``<name>-laptop`` on each developer's machine. The alarm's alert rule is scoped
to ``environment:production`` so intermittently-run laptops never page. See
[Production Deployment](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/).
"""

from typing import TYPE_CHECKING, Final

import sentry_sdk
from contracts.settings import Settings
from dagster import HookContext, failure_hook
from sentry_sdk.crons import capture_checkin
from sentry_sdk.crons.consts import MonitorStatus

if TYPE_CHECKING:
    from sentry_sdk._types import MonitorConfig

LIVE_FORECAST_MONITOR_SLUG: Final[str] = "live-forecasts"
"""Slug of the Sentry cron monitor fed by ``live_forecasts``' success heartbeat.

Laptop testing must use a *different*, throwaway slug (e.g. ``"live-forecasts-test"``) so an
intermittently-run laptop never registers a stale environment on the production monitor."""

LIVE_FORECAST_MONITOR_CONFIG: "Final[MonitorConfig]" = {
    "schedule": {"type": "crontab", "value": "0 0,6,12,18 * * *"},
    "timezone": "UTC",
    "checkin_margin": 120,
    "max_runtime": 60,
}
"""Declarative config upserted with each heartbeat: the 6-hourly live schedule, 120-minute grace
before a slot counts as missed (≈ one missed 6-hourly slot plus margin), and a 60-minute max
runtime. Only *success* heartbeats are ever sent, so the alarm keys off missed check-ins alone."""


def init_sentry(settings: Settings) -> None:
    """Initialise the Sentry SDK for this process, or do nothing if Sentry is disabled.

    A no-op when ``settings.sentry_dsn`` is empty (the default), so nothing is sent from laptops
    or CI unless a DSN is explicitly configured. Called once per process at import of the Dagster
    definitions module, so it runs in the daemon, the webserver, and every run worker.

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
    )


@failure_hook
def sentry_capture_failure(context: HookContext) -> None:
    """Report a failed op/asset step to Sentry with its real exception and traceback.

    Runs in the run worker after a step raises, so ``context.op_exception`` is the live exception
    (traceback intact) rather than Dagster's serialized error info. A no-op when Sentry is
    uninitialised (empty DSN), because ``capture_exception`` needs an active Sentry client.
    """
    exception = context.op_exception
    if exception is not None:
        sentry_sdk.capture_exception(exception)


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
