# Setting up Sentry telemetry

How to point this project's error telemetry, power-freshness warnings, and daily forecast check-in
at a Sentry.io project — first testing it from your laptop, then turning it on in production.
This is the operational recipe;
the design rationale (why an explicit failure hook rather than log capture, how the freshness
warning models recovery) is on the [Send telemetry to
Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry) design page.

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

`SENTRY_ENVIRONMENT` is the tag that separates your telemetry from everyone else's in the Sentry UI.
Use `<your-name>-laptop` (e.g. `jacks-laptop`, `alexs-laptop`) so error events filter cleanly by
origin, and so the production alert rules — scoped to `environment:production` — never fire for
your machine.

## Verify it works from your laptop

Save this script and run it with `uv run python <file>`. It exercises the two real code paths — the
failure hook and the freshness warning — and flushes so the events reach Sentry before the process
exits:

```python
from datetime import datetime, timezone

import polars as pl
import sentry_sdk
from contracts.settings import Settings
from dagster import job, op

from nged_substation_forecast._sentry import (
    init_sentry,
    report_power_freshness,
    sentry_capture_failure,
)
from nged_substation_forecast.defs.checks import PowerFreshnessResult


@op
def _intended_failure() -> None:
    raise ValueError("laptop Sentry acceptance test — this failure is intentional")


@job(hooks={sentry_capture_failure})  # the same hook the production jobs attach
def _sentry_smoke_job() -> None:
    _intended_failure()


# A synthetic "one series is late" result, so report_power_freshness has something to send.
_stale_result = PowerFreshnessResult(
    n_series_total=1,
    n_stale=1,
    n_never=0,
    threshold_hours=24.0,
    late=pl.DataFrame(
        {
            "time_series_id": pl.Series([999], dtype=pl.Int32),
            "last_seen": pl.Series([datetime(2000, 1, 1, tzinfo=timezone.utc)]),
            "hours_late": pl.Series([9999.0], dtype=pl.Float64),
            "status": pl.Series(["stale"]),
        }
    ),
)


settings = Settings()
if not settings.sentry_dsn:
    raise SystemExit("No SENTRY_DSN in .env — nothing to test.")

init_sentry(settings)
_sentry_smoke_job.execute_in_process(raise_on_error=False)  # fires the failure hook
report_power_freshness(settings, _stale_result)  # fires the freshness warning
sentry_sdk.flush(timeout=10)
```

Then check the Sentry UI:

- **Issues**, filtered to `environment:<your-name>-laptop` — two events: a `ValueError: laptop
  Sentry acceptance test…` with a full stack trace (a traceback, not a message-only event, is the
  point — it confirms the failure hook forwards the live exception rather than relying on log
  capture), and a `NGED power data stale: 1/1 time series late…` warning whose message lists the
  late series and how late each is (here, `series 999: 9999.0h late …`) with the full per-series
  detail also in the event context. Both events are fingerprinted with your `<your-name>-laptop`
  environment, so they form their own issues, separate from production's.

Once you are satisfied, resolve the two test issues if you like. Neither affects the production
wiring.

## What gets sent when you run Dagster locally

With a `SENTRY_DSN` configured, `init_sentry` runs at import of the Dagster definitions module. So
every local Dagster process (the webserver and each run worker) has Sentry active and tags
everything with your `SENTRY_ENVIRONMENT`. But the telemetry is deliberately narrow — running
Dagster on your laptop does **not** forward everything to Sentry:

- **Failures of the three scheduled jobs are sent.** `power_time_series_and_metadata_job`,
  `ecmwf_ens_job`, and `live_primary_substation_forecasts_job` each attach the
  `sentry_capture_failure` hook, so a failed step in any of them forwards the live exception —
  traceback intact — to Sentry Issues, tagged with your environment. This fires whether the run was
  launched by its schedule or started by hand as that named job.
- **Ad-hoc asset materialisations are *not* sent.** Clicking "Materialize" on an asset in the
  Dagster UI runs it outside those jobs, so no failure hook is attached and a failure there never
  reaches Sentry — it is yours to watch at the UI. To exercise the real hook end to end, launch
  `ecmwf_ens_job` by name rather than materialising the `ecmwf_ens` asset directly.
- **Freshness warnings *are* sent — if your local power data is stale.** Whenever the
  `power_data_is_fresh` asset check runs (alongside every `power_time_series_and_metadata`
  materialisation) and finds any series past the staleness threshold, it forwards a `warning`-level
  event to Sentry. The freshness warning is gated on the DSN alone, so a laptop with a stale local
  Delta table *will* emit one. The warning can't pollute production's issue, though: the event is
  fingerprinted per environment, so your `<name>-laptop` staleness forms its own Sentry issue,
  separate from production's. Resolve or ignore that Sentry issue as you like.
- **No forecast check-in is sent.** `SENTRY_MONITOR_FORECASTS` defaults to `false`, so a laptop
  never checks in to the `live-primary-substation-forecasts` cron monitor (see [The daily forecast
  check-in](#the-daily-forecast-check-in)).

The failure hook's scope is the same in production. The [design
page](../architecture/production-deployment.md#send-telemetry-to-sentry) explains why the failure
hook covers only the scheduled jobs, and how the freshness warning models recovery.

## Turn it on in production

**The Sentry variables go in two places, because production runs Dagster in two places.** The
always-on control-plane box runs the daemon, the webserver, and the code-location server, and reads
its variables from the box's `.env` — see [Setting up the live service on AWS, Step
14](aws.md#step-14-configure-dagster-on-the-box):

```dotenv
SENTRY_DSN=<the OCF Sentry project DSN>
SENTRY_ENVIRONMENT=production
```

Every scheduled run executes on a Fargate task, which reads its variables from the task
definition instead — see [Step 9](aws.md#step-9-create-the-ecs-cluster-and-fargate-task-definition).
The failure hook, the freshness warning, and the forecast check-in all fire inside those runs, so
the task definition carries `SENTRY_DSN` and `SENTRY_ENVIRONMENT=production` too, plus
`SENTRY_MONITOR_FORECASTS=true`, which only production sets.

Three one-time console steps complete the setup:

1. **Add an alert rule for the freshness warning, scoped to `environment:production`, and set
   it to fire on *regressions*.** The freshness warning is a one-way event: when NGED's feed
   recovers, the code simply stops sending, so the Sentry issue's "last seen" timestamp stops
   advancing but the issue stays open until someone resolves it. Resolve it once you've confirmed
   recovery. Match the rule on the freshness event's message, not on `level:warning` alone, because
   `retrying_asset:*` events are warnings too. Firing on regressions (plus a short project-level
   auto-resolve window) means a *new* stall after a recovery re-pages instead of silently appending
   to the old, unresolved issue.
2. **Route events by their tags.** `fault_category:run_failed` means a scheduled job failed, so that
   cycle did not run — notify whoever is on for the next business day. Everything else belongs in a
   digest rather than a notification: `degraded_asset:*` (an asset carried on with reduced
   function, including the [forecast's six degradation
   kinds](operations.md#the-daily-primary-substation-forecast)), `asset_check:*` (a check could not
   evaluate its own inputs, so one signal is unknown rather than bad), and `retrying_asset:*` (a
   warning that an asset failed and will retry, sent on the first failed attempt and followed by a
   `run_failed` event if every retry fails). Nothing here warrants waking anyone — see [Inherent
   stability](../design-philosophy/inherent-stability.md) for the uptime posture that makes that the
   right call.
3. **Set who is told about a missed forecast check-in**, once the monitor exists (see [The daily
   forecast check-in](#the-daily-forecast-check-in) below). A missed check-in means the service
   wrote no forecast today, so notify whoever is on for the next business day, as for
   `fault_category:run_failed`.

One handover note: the Sentry account is OCF's today, so at handover the alert routing (and possibly
the account itself) moves to NGED — see [Handover to
NGED](../roadmap/handover.md#2-alert-on-absence-not-just-failure).

## The daily forecast check-in

**A Sentry cron monitor raises an issue when no Primary substation forecast has landed by 17:00
UTC.** Each successful run of today's `live_primary_substation_forecasts` partition sends one `OK`
check-in to the monitor with the slug `live-primary-substation-forecasts`. Sentry evaluates the
monitor itself, so a dead daemon, a stopped box, and a failed run all show up as the same missed
check-in, which no process inside the deployment could report.

**The monitor's settings come from the code, and the first check-in creates the monitor.** Each
check-in carries `PRIMARY_SUBSTATION_FORECAST_MONITOR_CONFIG` from
`nged_substation_forecast._sentry`, and Sentry creates the monitor from that configuration, or
updates it to match:

- **Schedule**: crontab `0 12 * * *`, timezone UTC, the cadence of
  `live_primary_substation_forecasts_schedule`. A test fails if the two drift apart.
- **Check-in margin**: 300 minutes. The asset retries a missing ENS run until about 16:00 UTC, and a
  run then takes a few minutes, so a healthy day can check in as late as about 16:10 UTC. The margin
  puts the alarm at 17:00 UTC, after the last retry.
- **Maximum runtime**: not set. Only success check-ins are sent, never `in_progress`, so Sentry has
  no start time to time a run against.

A schedule or margin edited in the Sentry console is overwritten by the next check-in, so change
the configuration in the code. The monitor appears under **Crons** in Sentry after the first
production run that succeeds with `SENTRY_MONITOR_FORECASTS=true` in the task definition. Set who is
told about a missed check-in on that monitor's page, as step 3 above describes.

**Only production sends check-ins.** A backfill of a past day sends none, because the check-in says
that the service is alive today. `SENTRY_MONITOR_FORECASTS` defaults to `false`, so a laptop sends
none either. A laptop that set the variable would register the production slug under its own
environment and then stop checking in, so Sentry would report missed check-ins for that laptop.
To test a check-in from a laptop, pass a throwaway slug instead:

```python
import sentry_sdk
from contracts.settings import Settings

from nged_substation_forecast._sentry import init_sentry, send_primary_substation_forecast_checkin

settings = Settings(sentry_monitor_forecasts=True)
init_sentry(settings)
send_primary_substation_forecast_checkin(
    settings, monitor_slug="live-primary-substation-forecasts-test"
)
sentry_sdk.flush(timeout=10)
```

The test monitor then appears under **Crons**. Delete it once you have seen it, or Sentry reports
its missed check-ins every day after.

## See also

- [Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry) —
  the design rationale.
- [Setting up the live service on AWS](aws.md) — where the production `SENTRY_*` variables sit in
  the full bring-up.
- [Configuration reference](setup.md) — how `.env` and environment variables feed `Settings`.
