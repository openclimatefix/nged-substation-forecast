# Setting up Sentry telemetry

How to point this project's error telemetry and missed-check-in alarm at a Sentry.io project —
first testing it from your laptop, then turning it on in production. This is the operational
recipe; the design rationale (why the alarm lives outside the deployment, why an explicit failure
hook rather than log capture) is on the
[Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence)
design page.

Everything here is **opt-in**: with no `SENTRY_DSN` configured, every Sentry code path is a no-op,
so a laptop or CI run sends nothing until you deliberately set it up.

## Get a project DSN from Sentry

A **DSN** (Data Source Name) is the only credential you need. It is a project *ingest* key —
write-only, so it lets code send events but cannot read your Sentry data — which makes it far less
sensitive than a password.

1. Sign in to OCF's organisation at [sentry.io](https://sentry.io).
2. You send events to a **project**. Reuse the existing project for this service if there is one;
   otherwise create one with **Projects → Create Project**, platform **Python**, named e.g.
   `nged-substation-forecast`.
3. Open the DSN at **Settings → Projects → _your project_ → Client Keys (DSN)** (a newly created
   project also shows it on its setup screen).
4. Copy the **DSN** string. It looks like
   `https://<hash>@o<org-id>.ingest.<region>.sentry.io/<project-id>`.

## Configure your laptop

Add two lines to the repo-root `.env` (the same git-ignored file that holds the NGED source
credentials — see the [Configuration reference](setup.md#the-env-file-and-nged-source-credentials)):

```dotenv
SENTRY_DSN=<paste your DSN here>
SENTRY_ENVIRONMENT=jacks-laptop
```

`SENTRY_ENVIRONMENT` is the tag that separates your telemetry from everyone else's in the Sentry
UI. Use `<your-name>-laptop` (e.g. `jacks-laptop`, `alexs-laptop`) so error events filter cleanly
by origin, and so the production missed-check-in alert — scoped to `environment:production` — never
fires for your machine.

Do **not** set `SENTRY_MONITOR_FORECASTS` on a laptop. It gates the live heartbeat, and an
intermittently-run laptop must never register a check-in on the production monitor: the monitor
would then expect a 6-hourly heartbeat your laptop won't keep sending, and would flag it as missed.
The verification below sends its heartbeat to a throwaway monitor slug instead.

## Verify it works from your laptop

Save this script and run it with `uv run python <file>`. It exercises the two real code paths — the
failure hook and the heartbeat — and flushes so the events reach Sentry before the process exits:

```python
import sentry_sdk
from contracts.settings import Settings
from dagster import job, op

from nged_substation_forecast._sentry import (
    init_sentry,
    send_forecast_checkin,
    sentry_capture_failure,
)

TEST_MONITOR_SLUG = "live-forecasts-test"  # a throwaway slug — never the production monitor


@op
def _intended_failure() -> None:
    raise ValueError("laptop Sentry acceptance test — this failure is intentional")


@job(hooks={sentry_capture_failure})  # the same hook the production jobs attach
def _sentry_smoke_job() -> None:
    _intended_failure()


settings = Settings()
if not settings.sentry_dsn:
    raise SystemExit("No SENTRY_DSN in .env — nothing to test.")

init_sentry(settings)
_sentry_smoke_job.execute_in_process(raise_on_error=False)  # fires the failure hook
send_forecast_checkin(Settings(sentry_monitor_forecasts=True), monitor_slug=TEST_MONITOR_SLUG)
sentry_sdk.flush(timeout=10)
```

Then check the Sentry UI:

- **Issues**, filtered to `environment:<your-name>-laptop` — a `ValueError: laptop Sentry
  acceptance test…` with a full stack trace. A traceback (not a message-only event) is the point:
  it confirms the failure hook forwards the live exception rather than relying on log capture.
- **Crons → `live-forecasts-test`** — one OK check-in.

Once you are satisfied, delete the throwaway `live-forecasts-test` monitor in Sentry (and
optionally resolve the test issue). Neither affects the production wiring.

## Turn it on in production

On the always-on control-plane box, the three `SENTRY_*` variables go in the box's `.env` — see
[Setting up the live service on AWS, Step 14](aws.md#step-14-configure-dagster-on-the-box):

```dotenv
SENTRY_DSN=<the OCF Sentry project DSN>
SENTRY_ENVIRONMENT=production
SENTRY_MONITOR_FORECASTS=true
```

`SENTRY_MONITOR_FORECASTS=true` makes each successful live `live_forecasts` run check in to the
production `live-forecasts` monitor. Two one-time console steps complete the setup:

1. The monitor is created automatically on the first check-in (the code sends its schedule and
   margin with every heartbeat), so no manual monitor creation is needed.
2. **Scope the missed-check-in alert rule to `environment:production`** in Sentry, so a developer
   testing from a laptop can never trip the production alarm.

One handover note: the Sentry account is OCF's today, so at handover the alert routing (and
possibly the account itself) moves to NGED — see
[Handover to NGED](../roadmap/handover.md#2-alert-on-absence-not-just-failure).

## See also

- [Send telemetry to Sentry, and alarm on absence](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence)
  — the design rationale.
- [Setting up the live service on AWS](aws.md) — where the production `SENTRY_*` variables sit in
  the full bring-up.
- [Configuration reference](setup.md) — how `.env` and environment variables feed `Settings`.
