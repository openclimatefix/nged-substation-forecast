# Production Deployment — Design

How the live service is orchestrated, and why. The live service ingests the power telemetry of
National Grid Electricity Distribution (NGED) and the ensemble numerical weather prediction (NWP)
from the European Centre for Medium-Range Weather Forecasts (ECMWF), and checks both. For the
step-by-step recipe — build the image, verify it, push it, and stand up the AWS deployment around it
— see [Setting up the live service on AWS](../live_service/aws.md).

## Run the Dagster control plane continuously on one small VM

The Dagster control plane — daemon + webserver + code-location server — runs continuously on one
small VM, managed via Docker Compose. The VM runs neither MLflow nor Marimo. Its schedules are
responsible for the entire recurring pipeline: pulling fresh telemetry from NGED's bucket (hourly)
and downloading the ECMWF ensemble NWP from Dynamical.org (daily). The box does none of that
pipeline compute of its own, though: it dispatches every run — those schedules and UI-launched
backtests alike — to an ephemeral Fargate task via `EcsRunLauncher`. Even the light data-ingest
runs go to Fargate; the reasoning is in [Running the data-ingest runs on the control-plane
VM](#running-the-data-ingest-runs-on-the-control-plane-vm).
This VM-plus-`EcsRunLauncher` split is the [accepted infrastructure
tier](#accepted-option-small-ec2-control-plane-box-ecsrunlauncher); see [Live service: AWS
architecture](../roadmap/live-service.md#aws-architecture) for the roadmap's summary, and [AWS
Running Costs](aws-costs.md) for what the deployed service costs to run — as built at v1, and
projected at v2 scale.

Because scheduling lives inside Dagster rather than in any cloud-specific service, the entire stack
stays portable: the laptop deployment and the cloud deployment are the same artifact, started with
`docker compose up`. What switches a deployment between local and Fargate execution is not code but
**Dagster instance configuration**: the run launcher is declared in `dagster.yaml`, which lives
outside the image. The laptop's `dagster.yaml` declares no run launcher, so runs execute as local
subprocesses (Dagster's `DefaultRunLauncher`); the box's declares `EcsRunLauncher` ([Step 14 of the
runbook](../live_service/aws.md#step-14-configure-dagster-on-the-box)), so runs launch as Fargate
tasks. The code and image are byte-identical either way. The always-on daemon also provides the
cross-run coordination that only a daemon can: schedules, sensors, run monitoring, and declarative
automation.

The design accepts two trade-offs:

- **Single point of failure.** The daemon on one VM has no managed scheduler watching it. A quiet
  box failure could silently miss ingest runs. The blast radius is small, because the project's
  [uptime requirements are lenient by
  design](../background/requirements.md#uptime-lenient-by-design). Nothing outside the box
  currently watches the daemon, so a dead daemon raises no alert: Sentry receives events only from a
  process that is still running.

- **No run-level auto-retry after a hard crash.** We accept this trade-off. The next hourly power
  pull back-fills the missed telemetry. A missed `ecmwf_ens` partition is re-materialised by hand
  from the Dagster UI.

"Always-on" is also less demanding than it sounds, because the box comes with **built-in maintenance
windows**. The ingest runs are short and periodic, and the delivery design has NGED read published
forecasts directly from S3 (see [Forecast Delivery](forecast-delivery.md)). The gap between one
scheduled run and the next is therefore a regular window in which the VM can be stopped, patched, or
rebuilt without NGED noticing. See [Requirements → Uptime: lenient by
design](../background/requirements.md#uptime-lenient-by-design).

Several other orchestration shapes were considered and rejected — see [Considered but rejected
designs](#considered-but-rejected-designs).

## Warn on stale power data with a Dagster asset check

`power_time_series_and_metadata_job` runs hourly and succeeds even when NGED has published nothing
new. A job that merely *ran* therefore tells the operator nothing about whether fresh data actually
arrived. If the telemetry feed stalls, the Delta table silently goes stale. The
`power_data_is_fresh` asset check closes that gap: it reads the `power_time_series` Delta table's
*actual* data recency — the maximum `time` per `time_series_id` — rather than the asset's
materialisation timestamp, and warns when any series has no data within a 24-hour staleness
threshold. A native materialisation-freshness policy would miss this exact failure, because the
materialisation keeps succeeding.

**The 24-hour threshold clears NGED's normal publishing jitter while still catching a genuine stall
the same day.** NGED publishes at irregular intervals several hours apart, and the pipeline
back-fills the gaps by itself once the feed recovers. A shorter threshold would fire on that
ordinary jitter, while 24 hours still catches a multi-slot stall within the day.

The check is attached to `power_time_series_and_metadata`, so the existing hourly schedule runs it
every hour with no extra wiring. The check reports two kinds of lateness: a series that once
reported but has now gone **stale**, and a series in the `TimeSeriesMetadata`
parquet that has **never** sent data. The count of each, plus a table of the offending
`time_series_id`s, lands in the check's Dagster metadata. That table is **capped** at 50 rows,
because an uncapped listing writes thousands of rows into Dagster's event log every hour a
whole-feed stall lasts. That log is durable storage — Postgres in the AWS deployment, `pg_dump`ed to
S3 nightly. At V2 scale (~2,500 series) an uncapped table serialises to about 355 KB per hourly
evaluation, against about 8 KB at 50 rows. The cap matches the Sentry event *context* below (both
50) rather than the tighter 20 on the Sentry message body, so the same leading series appear in both
    and there is one less threshold to keep in sync. The counts beside it are uncapped, and an
    `n_late_listed` field records how many rows the table actually holds, so a truncated table can
    never make a large stall look small.

**The table lists never-reported series before stale ones, so the 50 rows are the head of that order
rather than the 50 series in most trouble.** When never-reported series alone fill the cap, no stale
series is listed at all, however stale it is. The counts `n_stale` and `n_never_reported` stay exact
for the whole watched population, so read those two before reading the table.

**Dagster's Checks view becomes the operator's at-a-glance status for whether the power data is
healthy.** The view shows a green tick when every series is current and a yellow warning when the
feed has stalled — or when a series we had written off starts reporting again. The severity is a
warning rather than a failure: a stalled feed is expected to self-heal once the feed recovers (the
pipeline back-fills the gap automatically), so the check must not block downstream assets.

Nothing the check's **body** does can fail its own step. It runs as a step of the hooked
`power_time_series_and_metadata_job`, and Dagster fails a run whose check step *errors* however
non-blocking that check is — so an object-store error, a half-written `metadata.parquet` or a bug in
the check itself would otherwise fail the hourly production run and page, turning fail-open into
fail-closed at exactly the wrong moment. The whole body therefore sits under a catch-all that logs
the traceback and returns an unhealthy result naming the fault. It catches `BaseException` and
re-raises only cancellation, because a Rust panic in any of the pyo3 extensions the check reads
through (Polars, delta-rs, obstore) does not derive from `Exception`, and each of those extensions
compiles a `PanicException` class of its own — so listing the classes to catch could never stay
complete, while the list of classes that must propagate can. (Dagster's own machinery — resource
init, and serialising the returned result into an event — is still outside the guard.) The
degradation still reaches Sentry, via
[`report_check_degradation`](#send-telemetry-to-sentry).

The check reports per-series staleness from *inside* Dagster while the daemon is alive, and
forwards that same staleness to Sentry as a warning (see the freshness-warning bullet under [Send
telemetry to Sentry](#send-telemetry-to-sentry)). A dead daemon cannot report itself, so the check
says nothing when the whole deployment has stopped. The freshness evaluation is a pure
function, so the Sentry warning reuses the same `PowerFreshnessResult` the check already computed
rather than recomputing it; the same result will later also feed the forecast-warnings delivery
table.

## Silence a series that has stopped reporting

A broken monitor keeps the check yellow for months, and an always-yellow channel trains the human
operator to ignore it. That habit ruins the [provider
channel](../design-philosophy/inherent-stability.md#three-audiences-three-channels) entirely.
`_SILENCED_TIME_SERIES_IDS` in `defs/checks.py` names the `time_series_id`s the check ignores.

**The silenced ids are removed from the check's inputs, not from its output.**
`evaluate_power_freshness` drops them from the coverage frame and from the expected ids before it
classifies anything, so `n_stale`, `n_never_reported`, `n_series_total`, and the late table all
describe the series we are still watching, with no arithmetic anywhere to get wrong. Dropping the
silenced ids before classification also means the Sentry warning inherits the silencing without
knowing silencing exists: `report_power_freshness` is handed the same result and returns early on a
healthy result, so a feed whose only late series are silenced sends no event.

**A returning series turns the check yellow rather than being removed automatically.** The list is
source code shipped read-only in the container image, so the check could not edit it. A check that
writes anything is a warning path that can fail, which [rule
7](../design-philosophy/inherent-stability.md#the-rules) forbids. The yellow is the prompt, and it
clears when a human deletes the line. The check also names every silenced id in its own output every
hour it runs, so the silencing stays visible and cannot quietly be forgotten.

**The list lives with the code because a series that has stopped reporting is a fact about the
world, not about a deployment** — it has equally stopped reporting whether we run on a laptop or on
AWS. The same fact is the argument against the obvious alternative of Dagster's own database, which
is per-deployment. The drawback is that silencing a series takes a commit and a redeploy, which is
the wrong interface for a list expected to change several times a month at V2 scale. Making the list
operator-editable from the UI is designed alongside the `asset_health_history` table, so that
operator-facing health state has one home rather than two.

## Send telemetry to Sentry

Sentry carries both "we broke" and "the weather feed is late", and they need different responses, so
the difference has to be legible in the event itself.

| | Level | Carries | Means | Sent by |
|---|---|---|---|---|
| **Something broke** | `error` | an exception | our code, or something it depends on, could not do its job | `sentry_capture_failure`, `report_asset_degradation`, `report_check_degradation` |
| **Something degraded** | `warning` | a message and its context | nothing threw; an input is late, stale or thin, and the forecast carried on | `report_power_freshness` |
| **Something will be retried** | `warning` | an exception | an asset failed in a way the asset retries, so the run has not failed yet | `report_asset_retry` |

An error event always carries an exception. A warning event carries an exception only when the
warning comes from `report_asset_retry`, and every such warning carries the `retrying_asset` tag.
**But an error event does not mean the run died.** Only `sentry_capture_failure` reports a failed
run; `report_asset_degradation` and `report_check_degradation` fire precisely because their caller
caught the exception and carried on, as the mechanisms below describe. Nor are those two cases
exclusive: where a degradation sender is called from inside an asset that still has a Delta write
ahead of it, a later failure puts a degradation event and a run-failure event on the same run. The
`fault_category:run_failed` tag is what separates them, which is why the routing below — and
[Operating the live service](../live_service/operations.md) — triages on that tag rather than on the
level. **Read the tag, never the level**, whenever the question is whether to tell a human now.

**Both kinds must be delivered.** A late NWP run is not our fault and does not stop the forecast,
and we still have to hear about it, because only a human can ask Dynamical.org what happened;
silence is reserved for a healthy service, not for a degraded service. That even-handedness is [rule
4](../design-philosophy/inherent-stability.md#the-rules), and the two sides of the rule are unevenly
built: three of the five senders cover the broke side, and the degraded side has only
`report_power_freshness`, which is why most of the degradation our checks detect is invisible from
outside Dagster ([#501](https://github.com/openclimatefix/nged-substation-forecast/issues/501)).

Two independent Sentry mechanisms live in `nged_substation_forecast._sentry`, both active in every
Dagster process (daemon, webserver, and each Fargate run worker) only when a `SENTRY_DSN` is
configured — so laptops and CI stay silent by default.

- **Error telemetry.** `init_sentry` initialises the SDK once per process at import of the Dagster
  definitions module, and a Dagster `@failure_hook` (`sentry_capture_failure`, attached to the
  scheduled asset jobs) reports a failed step's exception — traceback intact — from inside the run
  worker. The explicit hook is used rather than Sentry's log-to-event capture, which `init_sentry`
  deliberately disables (it passes a `LoggingIntegration` with `event_level=None`): Dagster logs a
  step failure without `exc_info`, so a purely log-based capture would yield a message-only event
  with no stack trace, and — left at its default `event_level=ERROR` — it would turn *every* `ERROR`
  log in the process into a Sentry event, including Dagster's own startup and ad-hoc-run logs.
  Breadcrumbs — the integration's default `level=INFO` — stay on, so log context still rides along
  with the events the senders below do send. The hook is attached to the two *scheduled* asset
  jobs, `power_time_series_and_metadata_job` and `ecmwf_ens_job`, so it covers the whole unattended
  production workload. Because log capture is off, the hook does not report failures in a manual UI
  materialisation, a backfill, or a `register_experiment_job` run, which the operator watches at
  the Dagster UI. The senders called from inside an asset, such as `report_asset_retry`, still fire
  in those runs.

    One production fault the hook cannot see is an asset check that caught its own exception instead
    of failing the run — which, by design, is every one of them: the two standalone `@asset_check`s,
    and the three per-run checks computed inside the `ecmwf_ens` asset. `report_check_degradation`
    covers exactly that gap: each check's catch-all sends the same exception the hook would have
    sent, tagged `asset_check` with the check's name, and `report_asset_degradation` does the same
    tagged `degraded_asset` for an *asset* that degrades rather than failing — today, when
    `power_time_series_and_metadata` cannot upsert the metadata table or write the downloaded-files
    list, and when `clean_nged_power_data` cannot vacuum the cleaned table. Since log capture is
    off, either handler's `ERROR` log alone would reach nobody. No job has a Sentry cron monitor
    either.
    Absent `report_check_degradation`, a check that cannot read its own inputs would show up only as
    a yellow tick in Dagster's Checks view, and nobody would be told.

    `report_asset_retry` is the early warning for an asset that is about to retry. `ecmwf_ens` sends
    the warning on the first failed attempt, tagged `retrying_asset`, so an ECMWF run that upstream
    never repairs does not stay silent until the last retry fails about 4 hours later.

    Those tags are what an alert rule routes on, and the failure hook is the one sender that would
    otherwise arrive with nothing to route on — so it tags `fault_category:run_failed`. That tag is
    a *positive* marker on the one class worth telling a human about, rather than a rule phrased as
    "error level, and neither degradation tag is set", which is correct today and misclassifies
    silently the day a sixth sender is added.

    **What deliberately never becomes an event.** A transient failure reading NGED's bucket is
    retried in-band by `power_time_series_and_metadata` — twice, seconds apart — because nothing is
    lost by waiting: the asset re-lists the bucket from scratch on each attempt and the next hourly
    run back-fills anyway, so failing would report a fault that has already fixed itself. A
    persistent outage still exhausts the budget, fails, and reports. Cancelling a run is likewise
    not a fault: Dagster re-raises `KeyboardInterrupt` and `DagsterExecutionInterruptedError` ahead
    of the hook, and the hook drops the `SystemExit` that does reach it.

    **Why that retry is in-band rather than a Dagster `RetryPolicy`.** `RetryPolicy` is the shorter
    spelling and it is unsafe here (measured on Dagster 1.13.17). It converts an interrupt into a
    retry request, and Dagster delivers a termination as exactly one interrupt — so the interrupt is
    consumed, the step restarts, and the cancelled run finishes **green** having written its data.
    It also retries the *whole* body, so a bug after the Delta append would dedupe to a no-op on the
    second attempt and turn a real failure into a green run every hour. Raising `RetryRequested`
    from inside a guard that re-raises the three interrupt types avoids both.

    Either way the hook receives a `RetryRequested` wrapper once the budget is exhausted, and
    Dagster unwraps only its own `RetryRequestedFromPolicy` — so the hook unwraps the plain class
    too, or every exhausted retry would group under `RetryRequested` rather than under its cause.
    Unwrapping discards nothing: the whole exception chain is serialised either way, and what the
    unwrap fixes is the event's title and how Sentry groups it.

- **Freshness warnings.** When the `power_data_is_fresh` asset check finds late series,
  `report_power_freshness` forwards that per-series staleness to Sentry as a `warning`-level event,
  reusing the `PowerFreshnessResult` the check already computed. It is gated on the DSN (Sentry's
  Data Source Name), like error telemetry, so it fires from wherever a
  configured environment runs the hourly check, separated by the `environment` tag. The event is
  fingerprinted **per environment** (`["nged-power-data-stale", <environment>]`) — Sentry's
  `environment` is a filter facet, not a grouping key, so without the environment in the fingerprint
  every deployment would share one issue; with it, each deployment gets its own ongoing issue, and
  the hourly re-reports of a continuing stall collapse into that one issue rather than a fresh issue
  each hour. The message body lists the late series and how late each is (`series 12: 48.5h late
  (last seen …)`, or `never reported`), and the full per-series detail is attached as structured
  event context. Both are capped — the message to a short leading slice with an `…and N more` line,
  the context to a larger slice — so a whole-feed stall at V2 scale can't attach thousands of rows;
  the true late count is always carried by the `n_late` tag. Sending is best-effort:
  `report_power_freshness` never raises, so a Sentry hiccup affects no more than its own event. Were
  it to raise, the check's catch-all would swallow it and discard the whole freshness evaluation
  with it — every late series, in the very hour they went late.

Freshness is a *two-way* state (stale ↔ recovered) modelled with a *one-way* primitive: a warning
event has no "resolved" counterpart. So **recovery is signalled by the events stopping** — the
issue's "last seen" timestamp stops advancing — and the operator resolves the issue by hand. The
production alert rule should fire on *regressions* (and the project's Sentry auto-resolve window
kept short) so that a fresh stall after a recovery re-pages rather than silently appending to a
stale-but-unresolved issue. Choosing recovery-by-silence over an explicit resolved event is a
deliberate trade: the check's own green tick in Dagster's Checks view is the signal that the feed
has recovered, and the Sentry warning is the per-series breadcrumb that reaches a human outside
Dagster.

### Separating laptop telemetry from production

Sentry's `environment` tag (from `Settings.sentry_environment`) keeps the two apart. The always-on
box sets `environment=production`. Each developer overrides the `local` default with `<name>-laptop`
(e.g. `jacks-laptop`, `alexs-laptop`), so error events filter cleanly by origin. Scope every alert
rule that pages a human to `environment:production`, so an intermittently-run laptop never pages.
The [AWS runbook](../live_service/aws.md) lists the exact environment variables set on the box.

## Separate "is this run usable?" from "is it perfect?"

Data ingest poses two different questions, and we answer them with two different mechanisms so that
neither compromises the other. **"Is this run usable?"** is a fatal gate: the contract `validate`
inside the ingest asset rejects data that is structurally broken, so a bad run fails loudly and
writes nothing rather than poisoning the table. **"Is it perfect?"** is a separate, non-fatal
Dagster asset check: it records known-but-tolerable imperfections as a warning without blocking the
materialisation or anything downstream. Collapsing the two — failing on every imperfection — throws
away otherwise-good data. Ignoring the distinction the other way lets genuinely broken data land
silently.

The `ecmwf_ens` asset is the worked example. Its `nwp_has_no_unexpected_nulls` check surfaces the
nulls that ECMWF ENS is known to carry (a WARN); its `nwp_instantaneous_variables_have_no_nulls`
check surfaces a null in a variable that should never be null, such as temperature or wind (also a
WARN, but a fault worth raising with Dynamical.org rather than re-running); and its
`nwp_run_is_complete` check surfaces a run that arrived short of its full ensemble-member,
forecast-step, and grid-cell grid (also a WARN — the rows that did arrive are kept), while
`Nwp.validate` still hard-fails a weather column that carries no data at all. The reasoning behind
exactly where that fatal/tolerated line sits is documented in [Known ECMWF ENS data-quality
issues](ecmwf-ens-known-issues.md). The `power_data_is_fresh` check above is the same shape of tool
pointed at a different question — staleness rather than completeness — and is likewise a warning,
never a failure.

## Keep MLflow off the production runtime path

**The deployment runs no MLflow tracking server, so nothing the live service runs loads a model from
MLflow.** Production runs are ephemeral Fargate tasks, dispatched by the always-on Dagster control
plane described [above](#run-the-dagster-control-plane-continuously-on-one-small-vm). An ephemeral
container has no persistent disk and no tracking server to reach. The production image currently
carries no trained forecasting model. A trained forecasting model added to the live service would
reach the container inside the image, as a plain directory that the forecaster's own `load` reads.
The image build never contacts MLflow. (The rejected alternative — fetching the model from MLflow at
container startup — is covered in [Considered but rejected
designs](#fetching-a-model-from-mlflow-at-container-startup).)

This design also serves the operating model for after this Network Innovation Allowance (NIA)
project (see [Requirements → Operating model &
handover](../background/requirements.md#operating-model-handover)). The working assumption is that
NGED runs the service on its own AWS account, operated day to day by NGED staff who did not develop
the code, working from the runbooks. For that operator there is no tracking server on the hot path
to break. A trained forecasting model inside the image would change only when a new image is
deployed, and that stability is a feature, not a limitation.

The cross-validation pipeline does load models from MLflow, through
`BaseForecaster.load_from_mlflow` (see [ML orchestration: model
artifacts](ml-orchestration.md#model-artifacts-one-replaceable-archive-no-local-cache)).
`load_from_mlflow` downloads on every call, for the reasons in [Why there is no local
cache](ml-orchestration.md#why-there-is-no-local-cache), so it offers no way to keep serving through
an MLflow outage. A live service that picked up each new model from MLflow without a redeploy would
need a way to keep serving through an MLflow outage, designed against production's own availability
needs. That design is tracked in [issue
#472](https://github.com/openclimatefix/nged-substation-forecast/issues/472).

## Resolve repo-relative paths via a workspace marker, not directory depth

`contracts.settings.PROJECT_ROOT` anchors every repo-relative default in `Settings` — the CV fold
definitions (`conf/cv/default.yaml`), the local `data/` roots, and the `.env` location.
`PROJECT_ROOT` is resolved by walking up from the installed `settings.py` to the nearest ancestor
directory holding `uv.lock` (the file that exists only at the uv workspace root). The resolution
falls back to the current working directory when no ancestor qualifies.

We resolve via a marker rather than a hard-coded directory depth because `settings.py` sits at a
different depth in each of the two layouts we deploy. An editable dev install has it at
`packages/contracts/src/contracts/settings.py` below the repo root. The production image's `uv sync
--no-editable` installs it at `<venv>/lib/python3.14/site-packages/contracts/settings.py` instead.
Any fixed number of `Path.parents` hops that lands on the repo root in one layout lands somewhere
arbitrary in the other — the venv root, in the image's case. Every repo-relative default then fails
with `FileNotFoundError`
([#287](https://github.com/openclimatefix/nged-substation-forecast/issues/287)). The marker walk
handles both layouts: a dev checkout resolves to the repo root, and the production image resolves to
`/app` because the Dockerfile copies `uv.lock` there alongside `conf/`. As a result, the image needs
no per-path env-var overrides, and `conf/model/` stays available for running training jobs
in-container.

The fallback case — a wheel installed into a venv outside any workspace checkout — is a deployment
shape we don't currently have. If that shape appears, it must either run with its working directory
laid out like the repo root or set each path setting explicitly via its env var. (Packaging `conf/`
and the other repo-relative resource files into a wheel via `importlib.resources` was considered
for #287 and deliberately deferred until that deployment shape actually exists.)

## Considered but rejected designs

Each design below was seriously considered for the live service and rejected. Every subsection
follows the same structure: first it describes the rejected design, then it explains why we rejected
it.

### No control plane: EventBridge Scheduler launching ECS tasks

**The design.** AWS EventBridge Scheduler fires on each schedule's cadence and launches an ECS
(Elastic Container Service) `RunTask` that executes `dagster asset materialize` (or `dagster job
execute`) directly in the production image. Dagster acts purely as the in-process execution
framework, EventBridge acts as the scheduler, and compute is paid per run with nothing always-on.
Run history could still be inspected by pointing an on-demand `dagster-webserver` at shared Postgres
run storage.

**Why we rejected it (July 2026).** Four reasons:

1. **Portability is a hard requirement.** The entire stack must run on a local laptop (or any cloud)
   without AWS-specific services. Once the schedule lives in EventBridge and run-level retry lives
   in Step Functions, the stack no longer runs with `docker compose up`. The Dagster-on-a-VM design
   keeps the laptop and the cloud deployment the same artifact.

2. **Handover to NGED is simpler without AWS coupling.** A Dagster deployment NGED can run anywhere
   is an easier handover than an EventBridge + Step Functions + ECS arrangement they would have to
   recreate inside their own AWS account. ("A cron and a container" is the easier handover only if
   the receiving organisation is committed to AWS.) See [Handover to NGED](../roadmap/handover.md)
   for the full handover plan.

3. **The cost saving is not real for this workload.** EventBridge's pay-per-run economics matter
   when the alternative is an idle fleet. Our alternative, though, is one small VM. Moving the
   scheduler off that VM would save roughly \$10–15/month while adding architectural surface
   area.

4. **Run-level retry is no better under EventBridge.** EventBridge Scheduler retries the
   *invocation* of a task (with backoff, a configurable max age/attempt count, and a Simple Queue
   Service (SQS) dead-letter queue for failed invocations). But once ECS accepts the task,
   EventBridge considers its job done — it does not notice a run that starts and then crashes.
   Relaunching a whole crashed run would have required wrapping the task in a Step Functions state
   machine using the `.sync` integration, while failures *inside* a run are already handled by
   Dagster's `RetryPolicy` on individual assets/ops, which works in-process with or without a
   daemon. So neither design gives run-level retry out of the box. At one telemetry run an hour and
   one NWP run a day, manual backfills plus alerting are a sufficient answer in both worlds.

EventBridge is also graph-blind: it fires a cron and passes a payload, so asset dependency ordering
would have been handled entirely by Dagster within a single self-contained run of the full asset
selection. Cross-run coordination — sensors, declarative automation, freshness-driven
materialisation — would not have been available without the daemon.

**Door left open.** EventBridge remains usable later as a *pure trigger* — an external service that
pokes an otherwise portable stack (e.g. S3 event → EventBridge rule → webhook or task launch) —
provided the stack never depends on it to run. Event-driven behaviour (e.g. "run when NGED drops a
new file") can alternatively be handled natively by Dagster sensors, which the always-on daemon
supports.

### A periodically-woken control plane

**The design.** Run the Dagster daemon only periodically (say, hourly) instead of continuously, to
save cost; on each wake-up, the daemon's schedule catch-up (`max_catchup_runs`, default 5) would
fire any scheduled ticks missed while it slept.

**Why we rejected it.** Sensors, run monitoring, and retries all assume an always-on daemon; the web
UI would be unavailable most of the time; and an external scheduler would still need to reliably
wake the daemon — which reintroduces the scheduling problem one level up.

### An always-on Fargate service for the control plane

**The design.** Run the always-on control plane — daemon, webserver, code-location server, and
Postgres — as a long-running ECS service on Fargate, instead of on an EC2 (Elastic Compute Cloud)
VM. The appeal is real: Fargate has no operating system for us to install or patch (AWS owns and
maintains the hosts), so the one piece of self-maintained OS in the deployment would disappear.

**Why we rejected it.** Five reasons:

1. **Fargate's premium pays off exactly where this workload can't use it.** Fargate's per-second
   billing wins for compute that runs a few minutes per day — which is why every pipeline run goes
   there. For a box that runs 24/7, the same premium just compounds: at `eu-west-2` on-demand rates,
   an always-on 2 vCPU / 4 GB Fargate service costs roughly **2.4× the `t4g.medium` VM** it would
   replace (~\$66/month vs ~\$27/month, from the ARM Fargate and EC2 rates in the [region price
   table](forecast-delivery.md#securing-it)).

2. **A VM can take on further long-running services cheaply.** The VM runs only the control plane
   today; a Marimo dashboard is one option for a later service on the same box. A VM absorbs each
   additional long-running service as one more entry in the same `docker-compose.yml`, using
   headroom already paid for. As Fargate services, each would be its own always-on task paying the
   premium above over again.

3. **Postgres needs a disk that Fargate doesn't have.** A Fargate task has no persistent local
   storage, so the Postgres container (run, event, and schedule history) could not come along. It
   would have to move to managed RDS (Relational Database Service) — more cost, and one more
   AWS-specific service to recreate at handover — or onto EFS (Elastic File System), a network
   filesystem that Postgres tolerates poorly. On the VM, Postgres's data is simply a Docker volume
   on the instance's own disk.

4. **It trades a portable artifact for AWS-specific glue.** The control plane's deployment
   description *is* its `docker-compose.yml`: the laptop and the cloud run the same artifact. As an
   ECS service, that description becomes task definitions, service configuration, and (per the
   previous point) RDS — the same AWS coupling that helped reject the [EventBridge
   design](#no-control-plane-eventbridge-scheduler-launching-ecs-tasks), in both its portability and
   its handover form. The Tailscale-based access design adds a quieter version of the same friction:
   on the VM, joining the tailnet is one `tailscale up`, whereas a Fargate service needs a Tailscale
   sidecar container with its own state management.

5. **The benefit is smaller than it looks.** The OS burden Fargate would remove is one Ubuntu box
   that patches itself (`unattended-upgrades`), can be stopped for maintenance in any of the
   built-in maintenance windows
   ([above](#run-the-dagster-control-plane-continuously-on-one-small-vm)), and is slated for a
   tested rebuild-from-scratch script (see [Handover: make the control-plane box rebuildable from
   scratch](../roadmap/handover.md#3-make-the-control-plane-box-rebuildable-from-scratch)) so that
   the answer to a sick box is *destroy and recreate*, never *diagnose*.

### Running the data-ingest runs on the control-plane VM

**The design.** The recurring data-ingest jobs — the hourly NGED telemetry pull and the daily
Dynamical.org NWP download — are deliberately much lighter than inference, so they could run
directly on the always-on box instead of each being dispatched to its own Fargate task. Running
ingest on the box would save the per-run Fargate cost and the minute or so of task-provisioning
latency each dispatch adds.

One sizing fact frames the whole question: Dagster launches **runs, not assets**, onto Fargate. A
schedule tick launches one job run, which becomes one Fargate task, and every asset in that job's
selection executes inside that single task — so a tiny asset only pays for a task of its own if it
has a schedule of its own (the one-off `h3_grid_weights`, materialised by hand, never does). The
question is therefore about the recurring scheduled runs, and only one of them is genuinely tiny:
the hourly NGED telemetry poll — which, because NGED publishes new files at irregular,
several-hours-apart intervals, spends most of its ticks discovering there is nothing to do.

**Why we rejected it.** Five reasons:

1. **Dagster has one run launcher per instance.** The `EcsRunLauncher` configured in `dagster.yaml`
   applies to *every* run the daemon launches — there is no per-job "run this one locally" switch.
   "Ingest on the box, forecasts on Fargate" would therefore mean writing a custom hybrid launcher
   that reads a run tag and delegates to one of two launchers. Writing that launcher is only a few
   dozen lines. But it is bespoke glue on the critical path of every run — exactly what the
   [handover constraint](../roadmap/handover.md) says to avoid.

2. **Blast-radius isolation.** The box's only job is to always be up, and it is deliberately tiny: a
   `t4g.medium` whose 4 GB is shared by the daemon, webserver, code-location server, and Postgres.
   The ingest jobs are lighter than inference but not
   negligible — the NWP ingest decodes a full daily ENS run (~7 million rows) and is CPU-hungry
   enough that its download once starved *itself* through thread contention
   ([#276](https://github.com/openclimatefix/nged-substation-forecast/issues/276)). Several busy
   worker threads on a 2-vCPU box would starve daemon heartbeats and the UI. A memory spike
   (upstream format drift, an unusually large file) risks the out-of-memory (OOM) killer taking out
   Postgres or the daemon. A bad data file killing an ephemeral task is a shrug; killing the control
   plane is the exact failure this architecture exists to avoid.

3. **The box would have to grow.** Hosting ingest means sizing the box for ingest peaks rather than
   for coordination — realistically a `t4g.large` at roughly double the `t4g.medium`'s ~\$27/month,
   which roughly cancels the Fargate saving.

4. **One execution path.** With everything on Fargate, every run has the same image, the same log
   destination (CloudWatch), the same IAM (Identity and Access Management) story (the task role),
   and the same debugging experience. Two execution environments mean two sets of failure modes for
   the operator. After the NIA project, the operators are NGED staff who did not develop the code,
   working from the runbooks.

5. **V2 scaling.** At ~2,500 time series the ingest workload grows roughly 78×. On Fargate that is a
   task-size change; on the box it is another round of resizing the component that is hardest to
   touch.

**What the accepted design costs.** Every ingest run currently inherits the task definition's 4 vCPU
/ 16 GB, which is sized for a forecast asset and oversized for an hourly telemetry pull. Nobody has
measured the daily NWP ingest's peak memory. At the `eu-west-2` ARM rates in the [region price table](forecast-delivery.md#securing-it)
(~\$0.21/hour for that task size), ~720 hourly runs of a few minutes each — most of them the no-ops
described above — come to on the order of **\$5–10/month**: the same magnitude of saving the
[EventBridge rejection](#no-control-plane-eventbridge-scheduler-launching-ecs-tasks) dismissed as
"not real for this workload".

**A second, quieter cost is idle time billed inside a single run.** `ecmwf_ens`'s not-yet-published
retry loop (`RetryRequested`, 30-minute waits) retries *inside the same run*, so on a
late-publication day its 16 GB task sits idle for up to 4 hours, billed the whole time (still well
under \$1).

**Door left open.** If those figures ever start to matter, two fixes exist inside the accepted
architecture, and neither moves work onto the box:

- **Right-size the ingest tasks.** `EcsRunLauncher` supports per-run `ecs/cpu` and `ecs/memory`
  tags, so the ingest jobs can declare a small task size while forecasts keep 4 vCPU / 16 GB — a
  config change, not an architecture change.
- **Replace polling schedules with Dagster sensors**
  ([#324](https://github.com/openclimatefix/nged-substation-forecast/issues/324)). Sensor evaluation
  functions run **on the daemon** — that is, on the always-on box — while only genuine *runs* go
  through the launcher. A sensor that cheaply lists NGED's bucket every few minutes and fires the
  ingest job only when new files actually exist puts the seconds-of-work detection on the box, where
  Dagster runs it natively, and launches Fargate only for the ~5 real ingests a day. The same
  pattern lets a sensor watch for NWP publication instead of holding a Fargate task idle through the
  retry loop. Sensors are the hybrid that reason 1 said would need a custom launcher — except
  Dagster already ships it, because the tiny recurring work here is *detection*, and detection is
  what sensors are for.

Either way: shrink or skip the ingest tasks; don't move the work onto the box.

### Fetching a model from MLflow at container startup

**The design.** Host the MLflow artifact root on S3 and have each production container fetch its
model from it at startup, instead of [carrying the model inside the
image](#keep-mlflow-off-the-production-runtime-path).

**Why we rejected it.** It adds runtime moving parts, needs tracking-store access from production,
and slows cold starts. A model inside the image has none of those drawbacks. The rejection gets
stronger under the operating model for after the NIA project (see [Requirements → Operating model &
handover](../background/requirements.md#operating-model-handover)). The working assumption is that
NGED runs the service on its own AWS account. With the model inside the image, NGED never has to run
— or depend on — an MLflow tracking server at all. A new model reaches production only as a new
container image.

Rejecting this design says nothing against MLflow itself — MLflow remains the backbone of ML
research and development: training runs log their models, configs, and metrics to it, and the
[cross-validation](../cross-validation/index.md) leaderboard is built from it. The boundary this
rejection draws is between ML R&D and production: research uses MLflow constantly, while
production's hot path never touches it.

Fetching a model from MLflow at container startup may still return as an option if new models start
reaching production often enough that a redeploy for each new model becomes a burden. A mechanism
for serving through an MLflow outage would then need to be designed, because `load_from_mlflow`
supplies none of its own. That mechanism is tracked in
[issue #472](https://github.com/openclimatefix/nged-substation-forecast/issues/472).

## Infrastructure tier: alternatives to the EC2 control-plane box

**This decision settles which AWS infrastructure tier hosts the control plane, not how a run is
dispatched once the control plane exists** — the orchestration designs in [Considered but rejected
designs](#considered-but-rejected-designs) settle that question. The infrastructure-tier decision is
between a small EC2 box, nothing always-on, one big box running all compute, a serverless control
plane backed by RDS, and a managed Dagster+ Solo plan.

### Accepted option: small EC2 control-plane box + `EcsRunLauncher`

A `t4g.medium` (2 vCPU / 4 GB; **£20.55/month on-demand, £12.95/month 1-yr no-upfront reserved**)
runs the Dagster daemon + webserver + code-location server + Postgres-in-Docker, behind
Tailscale (no ALB — Application Load Balancer). Every run — live schedules *and*
UI-launched backtests — is dispatched by `EcsRunLauncher` to an ephemeral Fargate task sized to the
job. Add ~£1.80 EBS (Elastic Block Store, the VM's disk), £9–13/month Fargate — live inference plus
the hourly ingest and polling runs, priced in [AWS Running Costs](aws-costs.md#v1-32-time-series) —
and ~£0.65/backtest.

**One image, four roles.** The daemon, webserver, and code-location server all run from the *same*
production image the ephemeral Fargate runs use. A `docker-compose.yml` on the box launches each as
a separate service with its own command, because the image has no default command. The compose file,
and the entrypoint gotchas it has to navigate, are now written up in the runbook — [Setting up the
live service on AWS: Steps 9 and
14](../live_service/aws.md#step-9-create-the-ecs-cluster-and-fargate-task-definition).

- **Pros:** Dagster properly (history, UI backfills, sensors, concurrency pools enforced centrally);
  backtests get big ephemeral compute; a later dashboard could run on the box at no extra
  cost; EC2 IAM instance roles (no static keys); a standard Dagster-OSS-on-AWS deployment.
- **Cons:** one pet server (patching, disk, daemon liveness — mitigate with systemd restart
  policies + the monitoring plan's "no fresh forecast" alarm); dagster.yaml/run-launcher config
  work; 4 GB is comfortable but not roomy. The risk of a long-lived,
  hand-maintained server grows once the service is operated by staff who did not develop the code —
  the full mitigation list (auto-recovery alarms, disk hygiene, a tested rebuild-from-scratch
  script) is [Handover workstream
  3](../roadmap/handover.md#3-make-the-control-plane-box-rebuildable-from-scratch).
- Cost trims: t4g.small (2 GB) is **free-trial (750 hrs/month) until 31 Dec 2026** and £10.30/£6.50
  after, if everything squeezes into 2 GB.

### Infrastructure-tier options considered and rejected

The four alternatives below — labelled A, C, D, and E, because B is reserved for the accepted
small-box option above — were researched alongside the accepted option and rejected in its favour.
They're kept here as a durable record of the decision, not as live options.

#### Option A — Level 1: nothing always-on ~£12–22/month

Hourly EventBridge Scheduler → one-shot Fargate task → `dagster job execute` against a throwaway
local `DAGSTER_HOME` → exit. Freshness check is the first op (latest Dynamical init vs the NWP init
behind the newest forecast in `power_forecasts` on S3); failure recovery is the cron (stale outputs
→ next tick re-runs); Delta commits provide the atomicity the freshness logic needs; "which
partitions need materialising" derived from Delta contents vs Dynamical availability, never
Dagster's records (they evaporate with the throwaway SQLite).

- **Pros:** cheapest; zero servers; self-healing by construction; everything it builds is needed by
  the accepted option (and C/D) anyway.
- **Cons:** fails both new requirements — no run history, hand-rolled backfills (issue #208 replay),
  backtests stay on the workstation, and the Marimo dashboard needs a separate always-on home
  (+£6/month Fargate service) anyway.

#### Option C — one big box with everything on it ~£56–86/month

Dagster + Postgres + Marimo + *all compute* (inference peaks ~9 GB → 16 GB box): EC2 `t4g.xlarge`
£82.20 on-demand / £51.90 reserved-1yr; Lightsail 16 GB £61.70 (4 vCPU, 40% CPU baseline); Lightsail
memory-optimised 16 GB £54.40 (2 vCPU). Option C is #206's "Level 3" and how OCF runs everything
else (with Airflow).

- **Pros:** simplest architecture possible — no Fargate/ECR-per-run/EventBridge/run-launcher;
  `docker compose` + `git pull`.
- **Cons:** priciest; backtests capped at 2–4 vCPU (slower than the workstation); Lightsail variants
  mean static AWS keys and burst-credit accounting; biggest pet.

#### Option D — serverless control plane (no pets) ~£41–45/month

Daemon + webserver as one always-on 0.5 vCPU / 2 GB ARM Fargate service (£14.70), RDS `db.t4g.micro`
Postgres (£10–12, approximate — the one unverified price), Marimo as its own tiny Fargate service
(approx £6.20), runs on ephemeral Fargate.

- **Pros:** full Dagster with zero servers to patch; IAM-native throughout.
- **Cons:** RDS is the tax (no local disk → no Postgres-in-Docker); webserver access needs an ALB
  (+~£16.50/month) or a Tailscale-sidecar hack; the most Terraform. Cleaner on paper than in
  practice.

#### Option E — Dagster+ Solo, Hybrid ~£37–45/month typical

£7.50/month base + £0.030/credit (1 credit = 1 materialisation or op execution; May 2026 pricing).
Live cadence ≈ 395 credits ≈ £12/month; backtests £0.15–0.52 each (a heavy month of 50 CV runs adds
£7.50–26). Plus a stateless Hybrid agent (~£6.75 tiny Fargate service) and Marimo hosting
(~£3.75–6), and the same per-run Fargate compute. Hybrid incurs no serverless charge; sensor
evaluations are free (an hourly freshness *op* would cost +£22/month — on Dagster+ you use sensors,
the better design anyway).

- **Pros:** managed UI/daemon/alerting/backfills with nothing of ours to keep alive.
- **Cons:** **Solo is 1 user** — collaborators can't log in (Starter is £75/month base); metering
  shapes design decisions; vendor dependency. The 1-user cap is likely disqualifying for an OCF
  collaboration.

### Comparison

| | A: Level 1 | Accepted: small box + Fargate | C: one big box | D: serverless CP | E: Dagster+ Solo |
|---|---|---|---|---|---|
| £/month | 12–22 | **27–40** | 56–86 | ~41–45 (+ALB) | 37–45 |
| Run history + UI backfills | ✗ | ✓ | ✓ | ✓ | ✓ |
| Backtests in AWS | ✗ | ✓ big ephemeral compute | ✓ but 2–4 vCPU | ✓ | ✓ (credits) |
| Adding a Marimo dashboard | +£6 add-on | ✓ free on box | ✓ free on box | +£6 service | +£3.75–6 |
| Servers to patch | 0 | 1 | 1 | 0 | 0 |
| No static AWS keys | ✓ | ✓ | ✓ EC2 / ✗ Lightsail | ✓ | ✓ |
| Multi-user UI | n/a | ✓ (Tailscale) | ✓ | ✓ | ✗ (1 user) |

Only the accepted option's £/month figure is re-derived from the full cost model in [AWS Running
Costs](aws-costs.md#v1-32-time-series). Every option pays the same per-run Fargate compute — the
hourly ingest and polling runs included — so that arithmetic moves all five columns together and
leaves the gaps between them unchanged.

**Recommendation (accepted): the small-box option** — the only shape giving full Dagster *and*
workstation-beating backtest compute *and* a free dashboard home, for ~£13–15/month over Level 1.
Option C is the fallback if operational simplicity trumps backtest speed and ~£30/month. D pays an
RDS+ALB tax for purism; E's 1-user cap rules it out.

## See also

- [Live service roadmap](../roadmap/live-service.md) — the full live-service design and its
  implementation workstreams. The costed AWS infrastructure options behind the accepted architecture
  live on this page, under [Infrastructure tier: alternatives to the EC2 control-plane
  box](#infrastructure-tier-alternatives-to-the-ec2-control-plane-box).
- [AWS Running Costs](aws-costs.md) — what this architecture costs to run: the v1 estimate as
  deployed, and the projected estimate at v2 scale.
- [Setting up the live service on AWS](../live_service/aws.md) — the step-by-step runbook: image
  build/push, and the full AWS bring-up including the control-plane box.
- [Setting up Sentry telemetry](../live_service/sentry.md) — the how-to for the error reporting and
  freshness warnings designed above: get a DSN, test from a laptop, enable in production.
- [Configuration reference](../live_service/setup.md) — where data tables and local artifacts live,
  and how to point `Settings` at S3.
- [ML Orchestration Design](ml-orchestration.md) — how the CV pipeline round-trips trained models
  through MLflow as one replaceable archive.
- [Why Dagster, not Airflow?](why-dagster-not-airflow.md) — why Dagster was chosen at design time,
  whether a migration to Airflow would be possible today, and what a port would take.
