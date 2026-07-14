# Production Deployment — Design

How the live service is orchestrated, and how the champion model gets from an MLflow
leaderboard into a running production container — and why. For the step-by-step recipe —
promote a model, build the image, verify it, push it, and stand up the AWS deployment around
it — see [Setting up the live service on AWS](../live_service/aws.md).

## Orchestration: an always-on Dagster control plane, not EventBridge

**Decision (July 2026): keep the Dagster control plane — daemon + webserver + code-location
server — running continuously on one small VM, alongside MLflow and Marimo, managed via Docker
Compose. Do not adopt AWS EventBridge Scheduler (or the related Step Functions pattern) for
triggering live forecast runs.** This pressure-tested and confirmed the roadmap's
[accepted architecture](../roadmap/live-service.md#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month)
(see [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) for the costed
options); the box dispatches every run — live schedules and UI-launched backtests alike — to an
ephemeral Fargate task via `EcsRunLauncher`.

**The rejected alternative** was a "no control plane" architecture: EventBridge Scheduler fires
on the 6-hourly cadence and launches an ECS `RunTask` that executes `dagster asset materialize`
(or `dagster job execute`) directly in the production image. In that pattern, Dagster acts
purely as the in-process execution framework, EventBridge acts as the scheduler, and compute is
paid per run with nothing always-on; run history could still be inspected by pointing an
on-demand `dagster-webserver` at shared Postgres run storage. We rejected it for four reasons:

1. **Portability is a hard requirement.** The entire stack must run on a local laptop (or any
   cloud) without AWS-specific services. Once the schedule lives in EventBridge and run-level
   retry lives in Step Functions, the stack no longer runs with `docker compose up`. The
   Dagster-on-a-VM design keeps the laptop and the cloud deployment the same artifact.

2. **Handover to NGED is simpler without AWS coupling.** A Dagster deployment NGED can run
   anywhere is an easier handover than an EventBridge + Step Functions + ECS arrangement they
   would have to recreate inside their own AWS account. (This reverses an earlier assumption
   that "a cron and a container" would be the easier handover — that only holds if the
   receiving organisation is committed to AWS.) See
   [Handover to NGED](../roadmap/handover.md) for the full handover plan.

3. **The cost saving is not real for this workload.** EventBridge's pay-per-run economics
   matter when the alternative is an idle fleet, but our alternative is one small VM that must
   exist anyway to host MLflow and Marimo. Splitting the scheduler out of that box would save
   roughly \$10–15/month while adding architectural surface area.

4. **Run-level retry is no better under EventBridge.** EventBridge Scheduler retries the
   *invocation* of a task (with backoff, a configurable max age/attempt count, and an SQS
   dead-letter queue for failed invocations), but once ECS accepts the task, EventBridge
   considers its job done — it does not notice a run that starts and then crashes. Relaunching
   a whole crashed run would have required wrapping the task in a Step Functions state machine
   using the `.sync` integration, while failures *inside* a run are already handled by
   Dagster's `RetryPolicy` on individual assets/ops, which works in-process with or without a
   daemon. So neither design gives run-level retry out of the box, and at four runs per day the
   existing replay/backfill mode plus alerting is a sufficient answer in both worlds.

EventBridge is also graph-blind: it fires a cron and passes a payload, so asset dependency
ordering would have been handled entirely by Dagster within a single self-contained run of the
full asset selection, and cross-run coordination — sensors, declarative automation,
freshness-driven materialisation — would not have been available without the daemon.

**Also rejected — a periodically-woken control plane** (running the daemon only, say, hourly to
save cost): the daemon's schedule catch-up (`max_catchup_runs`, default 5) would technically
fire missed 6-hourly ticks, but sensors, run monitoring, and retries all assume an always-on
daemon, the web UI would be unavailable most of the time, and something else would still need
to reliably wake the daemon — which reintroduces the scheduling problem one level up.

### Accepted trade-offs and mitigations

- **Single point of failure.** The daemon on one VM has no managed scheduler watching it; a
  quiet box failure could silently miss slots. Mitigation: the
  [missed-check-in alarm](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm)
  — each successful 6-hourly run checks in with an external monitoring service (Sentry cron
  monitoring is the planned mechanism), and an alert fires when an expected check-in fails to
  arrive. The alarm is the only component that lives outside the box, and the stack does not
  depend on it to function.

- **No run-level auto-retry after a hard crash.** Accepted; covered by the existing
  replay/backfill mode for missed slots plus the missed-check-in alarm.

### Door left open

EventBridge remains usable later as a *pure trigger* — an external service that pokes an
otherwise portable stack (e.g. S3 event → EventBridge rule → webhook or task launch) — provided
the stack never depends on it to run. Event-driven behaviour (e.g. "run when NGED drops a new
file") can alternatively be handled natively by Dagster sensors, which the always-on daemon
supports.

## Baking the model into the image at build time

Production forecasts run as ephemeral Fargate tasks, dispatched by the always-on Dagster
control plane described [above](#orchestration-an-always-on-dagster-control-plane-not-eventbridge).
An ephemeral container has no persistent disk and, for v0.1, no MLflow tracking server to reach
— so production inference needs some way to get a model without depending on either.

**Decision: bake the champion model into the image at build time, loaded via a plain
`save`/`load` — no MLflow, run ID, or cache involved at runtime.** The model directory is
produced once, out of band (a researcher picks the champion fold from the MLflow leaderboard and
downloads its artifacts to local disk), then `COPY`'d into the image at build time. Promotion
becomes rebuild + redeploy, which is auditable (image tags) and keeps MLflow completely out of
the production runtime.

**Rejected alternative:** MLflow artifact root on S3 fetched at container startup — more runtime
moving parts, needs tracking-store access from prod, and slower cold starts.

This decision also serves the preferred post-NIA operating model, in which a non-expert at
NGED operates the service day to day (see
[Requirements → Operating model & handover](../background/requirements.md#operating-model-handover)):
there is no tracking server on the hot path to break, and the model simply freezes between
OCF's scheduled expert interventions — under a vendor-develops / operator-runs split, that is a
feature, not a limitation.

This is deliberately simpler than reusing `BaseForecaster.load_from_mlflow`'s cache (the
mechanism the CV pipeline already uses — see
[ML orchestration: model artifacts](ml-orchestration.md#model-artifacts-mlflow-artifact-store-immutable-local-cache)):
v0.1 has no MLflow dependency to cache against in the first place.

**Future work:** once production wants to pick up a new champion without a rebuild + redeploy
(e.g. after the [XGBoost quick wins](../roadmap/xgboost-improvements.md) start landing
regularly), switch to fetching the champion model from MLflow dynamically — at that point
`load_from_mlflow`'s local-disk cache becomes the production-resilience mechanism again (serving
from disk on a cache hit so the live service survives an MLflow outage), exactly as it does for
CV today.

## Live inference is single-run, not bulk

The `live_forecasts` asset engineers features in **single-run mode** — an explicit
`power_fcst_init_time` supplied by the partition, joined across all 51 NWP ensemble members —
never bulk mode's one-`power_fcst_init_time`-per-NWP-run derivation (see
[ML orchestration: forecast cadence under-sampling](ml-orchestration.md#known-limitation-forecast-cadence-under-sampling-in-cv)
for where bulk mode's per-NWP-run derivation matters instead: CV/backtesting, not live inference).
A production run issues one forecast for one explicit init time every 6 hours; bulk mode's
derivation has no meaning for a single live materialisation.

## NWP availability: `live` vs `replay` is asymmetric by design

`live_forecasts`' `availability_mode` config resolves which NWP run to join against, and the two
modes are deliberately asymmetric:

- **`"live"`** joins the freshest NWP run actually present in Delta, with **no modelled
  publication delay** — reality already constrains the table to genuinely published runs, so a
  faster provider is used automatically without a config change.
- **`"replay"`** joins the freshest run at least `nwp_publication_delay_hours` old, reconstructing
  what was genuinely available at that historical init time. Without the delay, a replay would
  leak NWP runs that only landed after the fact — a lookahead-bias bug, not just an inaccuracy.

The scheduled path always uses `"live"`; backfills of missed or historical partitions use
`"replay"`. The mode is an explicit, manually-set flag rather than an automatic
live-iff-recent rule — the ambiguity of "recent" isn't worth resolving for a backfill path that's
already manually triggered.

## Serving only the trained population

`live_forecasts` forecasts exactly the production model's `trained_time_series_ids` (recorded in
`meta.json`), never the current day's eligibility set. This is the train==predict population
invariant: a time series the model never saw during training must never receive a live forecast,
even if it would otherwise qualify today.

## Why promotion is a Dagster asset, not a script

The "researcher downloads artifacts" step above is a manually-triggered Dagster asset,
`promoted_model` (config `mlflow_run_id`), rather than a bare script — promotion becomes a
materialisation, giving an audit trail and lineage for free. The download logic itself
(`ml_core._production_helpers.fetch_model_artifacts`) is a pure, asset-independent helper, so
nothing about this decision couples it to Dagster.

**The Docker build reuses this same asset** (headlessly, via `dagster asset materialize`) — no
separate fetch script was built, since a bare script would have duplicated the asset's audit
trail for no benefit. The `docker build` step itself stays outside Dagster: it only ever runs on
a laptop today, and image build/push becomes a CI-shaped concern once an MLflow tracking server
and AWS infra exist — not something worth orchestrating through Dagster in the meantime.

## See also

- [Live service roadmap](../roadmap/live-service.md) — the full v0.1 design, including the
  costed AWS architecture options behind the accepted architecture (small control-plane box +
  `EcsRunLauncher`) and its implementation workstreams.
- [Setting up the live service on AWS](../live_service/aws.md) — the step-by-step runbook:
  promotion, image build/push, and the full AWS bring-up including the control-plane box.
- [Configuration reference](../live_service/setup.md) — where data tables and local
  artifacts live, and how to point `Settings` at S3.
- [ML Orchestration Design](ml-orchestration.md) — why production inference doesn't reuse the
  CV pipeline's MLflow-artifact cache.
