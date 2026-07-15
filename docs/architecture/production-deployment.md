# Production Deployment — Design

How the live service is orchestrated, and how the champion model gets from an MLflow
leaderboard into a running production container — and why. For the step-by-step recipe —
promote a model, build the image, verify it, push it, and stand up the AWS deployment around
it — see [Setting up the live service on AWS](../live_service/aws.md).

## Run the Dagster control plane continuously on one small VM

The Dagster control plane — daemon + webserver + code-location server — runs continuously on
one small VM, alongside MLflow and Marimo, managed via Docker Compose. Its schedules are
responsible for the entire recurring pipeline, not just the forecasts: pulling fresh telemetry
from NGED's bucket (hourly), downloading the ECMWF ensemble NWP from Dynamical.org (daily), and
issuing the forecasts themselves (6-hourly). The box does none of that pipeline compute of its
own, though: it dispatches every run — those schedules and UI-launched backtests alike — to an
ephemeral Fargate task via `EcsRunLauncher`. Even the light data-ingest runs go to Fargate;
the reasoning is in
[Running the data-ingest runs on the control-plane VM](#running-the-data-ingest-runs-on-the-control-plane-vm). This is the roadmap's
[accepted architecture](../roadmap/live-service.md#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month);
see [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) for the
costed options.

Because scheduling lives inside Dagster rather than in any cloud-specific service, the entire
stack stays portable: the laptop deployment and the cloud deployment are the same artifact,
started with `docker compose up`. The always-on daemon also provides the cross-run coordination
that only a daemon can: schedules, sensors, run monitoring, and declarative automation.

The design accepts two trade-offs:

- **Single point of failure.** The daemon on one VM has no managed scheduler watching it; a
  quiet box failure could silently miss slots. The blast radius is small, because the project's
  [uptime requirements are lenient by design](../background/requirements.md#uptime-lenient-by-design):
  previously published forecasts stay readable from S3 and extend 14 days ahead, so a missed
  slot degrades forecast freshness rather than cutting NGED off. Mitigation: the
  [missed-check-in alarm](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm)
  — each successful 6-hourly run checks in with an external monitoring service (Sentry cron
  monitoring is the planned mechanism), and an alert fires when an expected check-in fails to
  arrive. The alarm is the only component that lives outside the box, and the stack does not
  depend on it to function.

- **No run-level auto-retry after a hard crash.** Accepted; covered by the existing
  replay/backfill mode for missed slots plus the missed-check-in alarm.

"Always-on" is also less demanding than it sounds, because the box comes with **built-in
maintenance windows**: forecasts are produced only every 6 hours, and NGED reads published
forecasts directly from S3 (see [Forecast Delivery](forecast-delivery.md)), so the gap between
one forecast run and the next is a regular window in which the VM can be stopped, patched, or
rebuilt without NGED noticing — and even an overrun costs only a single slot, recoverable via
replay-mode backfill. See
[Requirements → Uptime: lenient by design](../background/requirements.md#uptime-lenient-by-design).

Several other orchestration shapes were considered and rejected — see
[Considered but rejected designs](#considered-but-rejected-designs).

## Bake the model into the image at build time

Production forecasts run as ephemeral Fargate tasks, dispatched by the always-on Dagster
control plane described [above](#run-the-dagster-control-plane-continuously-on-one-small-vm).
An ephemeral container has no persistent disk and, for v0.1, no MLflow tracking server to reach
— so production inference needs some way to get a model without depending on either.

The champion model is therefore baked into the image at build time and loaded via a plain
`save`/`load` — no MLflow, run ID, or cache involved at runtime. The model directory is
produced once, out of band (a researcher picks the champion fold from the MLflow leaderboard
and downloads its artifacts to local disk), then `COPY`'d into the image at build time.
Promotion becomes rebuild + redeploy, which is auditable (image tags) and keeps MLflow
completely out of the production runtime. (The rejected alternative — fetching the model from
MLflow at container startup — is covered in
[Considered but rejected designs](#fetching-the-champion-model-from-mlflow-at-container-startup).)

This design also serves the preferred post-NIA operating model, in which a non-expert at
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

## Run live inference in single-run mode, not bulk

The `live_forecasts` asset engineers features in **single-run mode** — an explicit
`power_fcst_init_time` supplied by the partition, joined across all 51 NWP ensemble members —
never bulk mode's one-`power_fcst_init_time`-per-NWP-run derivation (see
[ML orchestration: forecast cadence under-sampling](ml-orchestration.md#known-limitation-forecast-cadence-under-sampling-in-cv)
for where bulk mode's per-NWP-run derivation matters instead: CV/backtesting, not live inference).
A production run issues one forecast for one explicit init time every 6 hours; bulk mode's
derivation has no meaning for a single live materialisation.

## Resolve NWP availability asymmetrically: `live` vs `replay`

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

## Serve only the trained population

`live_forecasts` forecasts exactly the production model's `trained_time_series_ids` (recorded in
`meta.json`), never the current day's eligibility set. This is the train==predict population
invariant: a time series the model never saw during training must never receive a live forecast,
even if it would otherwise qualify today.

## Promote the champion via a Dagster asset, not a script

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

## Considered but rejected designs

Each design below was seriously considered for the live service and rejected. Every subsection
follows the same structure: first it describes the rejected design, then it explains why we
rejected it.

### No control plane: EventBridge Scheduler launching ECS tasks

**The design.** AWS EventBridge Scheduler fires on the 6-hourly cadence and launches an ECS
`RunTask` that executes `dagster asset materialize` (or `dagster job execute`) directly in the
production image. Dagster acts purely as the in-process execution framework, EventBridge acts
as the scheduler, and compute is paid per run with nothing always-on; run history could still
be inspected by pointing an on-demand `dagster-webserver` at shared Postgres run storage.

**Why we rejected it (July 2026).** Four reasons:

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

**Door left open.** EventBridge remains usable later as a *pure trigger* — an external service
that pokes an otherwise portable stack (e.g. S3 event → EventBridge rule → webhook or task
launch) — provided the stack never depends on it to run. Event-driven behaviour (e.g. "run when
NGED drops a new file") can alternatively be handled natively by Dagster sensors, which the
always-on daemon supports.

### A periodically-woken control plane

**The design.** Run the Dagster daemon only periodically (say, hourly) instead of continuously,
to save cost; on each wake-up, the daemon's schedule catch-up (`max_catchup_runs`, default 5)
would fire any 6-hourly ticks missed while it slept.

**Why we rejected it.** Sensors, run monitoring, and retries all assume an always-on daemon;
the web UI would be unavailable most of the time; and something else would still need to
reliably wake the daemon — which reintroduces the scheduling problem one level up.

### An always-on Fargate service for the control plane

**The design.** Run the always-on control plane — daemon, webserver, code-location server, and
Postgres — as a long-running ECS service on Fargate, instead of on an EC2 VM. The appeal is
real: Fargate has no operating system for us to install or patch (AWS owns and maintains the
hosts), so the one piece of self-maintained OS in the deployment would disappear.

**Why we rejected it.** Five reasons:

1. **Fargate's premium pays off exactly where this workload can't use it.** Fargate's
   per-second billing wins for compute that runs a few minutes per day — which is why the
   forecast worker runs there. For a box that runs 24/7, the same premium just compounds: at
   `eu-west-2` on-demand rates, an always-on 2 vCPU / 4 GB Fargate service costs roughly
   **2.4× the `t4g.medium` VM** it would replace (~\$66/month vs ~\$27/month, from the ARM
   Fargate and EC2 rates in the
   [region price table](forecast-delivery.md#securing-it)).

2. **The box is planned to host more than the control plane.** The same VM is the intended
   home of the MLflow tracking server and the Marimo dashboard — which is why the
   [EventBridge rejection](#no-control-plane-eventbridge-scheduler-launching-ecs-tasks) counts
   the VM as a cost that exists anyway. A VM absorbs each additional long-running service as
   one more entry in the same `docker-compose.yml`, using headroom already paid for; as Fargate
   services, each would be its own always-on task paying the premium above over again — and
   MLflow brings its own persistence needs (its backend store and artifact root), running into
   the disk problem below a second time.

3. **Postgres needs a disk that Fargate doesn't have.** A Fargate task has no persistent local
   storage, so the Postgres container (run, event, and schedule history) could not come along.
   It would have to move to managed RDS — more cost, and one more AWS-specific service to
   recreate at handover — or onto EFS, a network filesystem that Postgres tolerates poorly. On
   the VM, Postgres's data is simply a Docker volume on the instance's own disk.

4. **It trades a portable artifact for AWS-specific glue.** The control plane's deployment
   description *is* its `docker-compose.yml`: the laptop and the cloud run the same artifact.
   As an ECS service, that description becomes task definitions, service configuration, and
   (per the previous point) RDS — the same AWS coupling that helped reject the
   [EventBridge design](#no-control-plane-eventbridge-scheduler-launching-ecs-tasks), in both
   its portability and its handover form. The Tailscale-based access design adds a quieter
   version of the same friction: on the VM, joining the tailnet is one `tailscale up`, whereas
   a Fargate service needs a Tailscale sidecar container with its own state management.

5. **The benefit is smaller than it looks.** The OS burden Fargate would remove is one Ubuntu
   box that patches itself (`unattended-upgrades`), can be stopped for maintenance in any of
   the built-in 6-hourly maintenance windows
   ([above](#run-the-dagster-control-plane-continuously-on-one-small-vm)), and is slated for a
   tested rebuild-from-scratch script (see
   [Handover: de-pet the control-plane box](../roadmap/handover.md#3-de-pet-the-control-plane-box))
   so that the answer to a sick box is *destroy and recreate*, never *diagnose*.

### Running the data-ingest runs on the control-plane VM

**The design.** The recurring data-ingest jobs — the hourly NGED telemetry pull and the daily
Dynamical.org NWP download — are deliberately much lighter than inference, so they could run
directly on the always-on box instead of each being dispatched to its own Fargate task. That
would save the per-run Fargate cost and the minute or so of task-provisioning latency each
dispatch pays.

**Why we rejected it.** Five reasons:

1. **Dagster has one run launcher per instance.** The `EcsRunLauncher` configured in
   `dagster.yaml` applies to *every* run the daemon launches — there is no per-job "run this
   one locally" switch. "Ingest on the box, forecasts on Fargate" would therefore mean writing
   a custom hybrid launcher that reads a run tag and delegates to one of two launchers. That is
   only a few dozen lines, but it is bespoke glue on the critical path of every run — exactly
   what the [handover constraint](../roadmap/handover.md) says to avoid.

2. **Blast-radius isolation.** The box's only job is to always be up, and it is deliberately
   tiny: a `t4g.medium` whose 4 GB is shared by the daemon, webserver, code-location server,
   and Postgres, with MLflow and Marimo planned on top. The ingest jobs are lighter than
   inference but not negligible — the NWP ingest decodes a full daily ENS run (~7 million
   rows) and is CPU-hungry enough that its download once starved *itself* through thread
   contention ([#276](https://github.com/openclimatefix/nged-substation-forecast/issues/276)).
   Several busy worker threads on a 2-vCPU box would starve daemon heartbeats and the UI, and
   a memory spike (upstream format drift, an unusually large file) risks the OOM killer taking
   out Postgres or the daemon. A bad data file killing an ephemeral task is a shrug; killing
   the control plane is the exact failure this architecture exists to avoid.

3. **The box would have to grow.** Hosting ingest means sizing the box for ingest peaks rather
   than for coordination — realistically a `t4g.large` at roughly double the `t4g.medium`'s
   ~\$27/month, which roughly cancels the Fargate saving.

4. **One execution path.** With everything on Fargate, every run has the same image, the same
   log destination (CloudWatch), the same IAM story (the task role), and the same debugging
   experience. Two execution environments means two sets of failure modes for an operator who
   is, post-NIA, a non-expert at NGED.

5. **V2 scaling.** At ~2,500 time series the ingest workload grows roughly 78×. On Fargate
   that is a task-size change; on the box it is another round of resizing the component that
   is hardest to touch.

**What the accepted design costs, and the door left open.** Every ingest run currently
inherits the task definition's 4 vCPU / 16 GB — sized for inference's ~9 GB peak, and
oversized for an hourly telemetry pull. At the `eu-west-2` ARM rates in the
[region price table](forecast-delivery.md#securing-it) (~\$0.21/hour for that task size),
~720 hourly runs of a few minutes each come to on the order of **\$5–10/month** — the same
magnitude of saving the [EventBridge rejection](#no-control-plane-eventbridge-scheduler-launching-ecs-tasks)
dismissed as "not real for this workload". If that figure ever starts to matter, the fix is a
config change, not an architecture change: `EcsRunLauncher` supports per-run `ecs/cpu` and
`ecs/memory` tags, so the ingest jobs can declare a small task size while forecasts keep
4 vCPU / 16 GB. Shrink the ingest tasks; don't move the work onto the box.

### Fetching the champion model from MLflow at container startup

**The design.** Host the MLflow artifact root on S3 and have each production container fetch
the champion model from it at startup, instead of
[baking the model into the image](#bake-the-model-into-the-image-at-build-time).

**Why we rejected it.** It adds runtime moving parts, needs tracking-store access from
production, and slows cold starts — baking the model in has none of those costs. The rejection
gets stronger under the preferred post-NIA operating model, in which NGED run this code
themselves (see
[Requirements → Operating model & handover](../background/requirements.md#operating-model-handover)):
with the model baked in, NGED never has to run — or depend on — an MLflow tracking server at
all, and the model simply freezes until a new image arrives. (OCF may well continue developing
the model post-NIA and release new container images for NGED to test, but that arrangement is
TBD — and either way it only changes which *image* NGED runs, never whether their production
runtime needs MLflow.)

Rejecting this design says nothing against MLflow itself — MLflow remains the backbone of ML
experimentation: training runs log their models, configs, and metrics to it, and the champion
is *chosen* from an MLflow leaderboard (see
[ML orchestration: model artifacts](ml-orchestration.md#model-artifacts-mlflow-artifact-store-immutable-local-cache)
for the design, and [ML experimentation](../ml_experimentation/index.md) for the day-to-day
workflow). The boundary this rejection draws is between ML R&D and production: research uses
MLflow constantly, while production's hot path never touches it. MLflow's job ends at the
moment of promotion, when the chosen champion is copied out of the tracking store and baked
into the image.

The idea may still return in a stronger form: the **future work** note at the end of
[Bake the model into the image at build time](#bake-the-model-into-the-image-at-build-time) —
the section describing the accepted design this one lost to — describes fetching the champion
dynamically once redeploys become frequent, with `load_from_mlflow`'s local-disk cache as the
production-resilience mechanism.

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
