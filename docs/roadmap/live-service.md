# Live service (AWS deployment)

> **Status: ✅ v0.1 shipped (July 2026, `v0.1.0`)** on the accepted architecture; the same
> architecture now runs **v0.2** (`v0.2.0`, deployed 13 August 2026) — the naive forecast is
> deployed and running on AWS. Epic
> [#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137)
> is closed. The shipped design now lives in
> [Production Deployment — Design](../architecture/production-deployment.md) (the orchestration
> and champion-model decisions), and the step-by-step operational recipes in the
> [live-service runbooks](../live_service/index.md) (promotion, AWS bring-up, day-to-day
> operation).
>
> This page is **not** retired yet: it remains the home of the post-v0.1 work still tracked as
> open issues — [production monitoring](#production-monitoring)
> ([#224](https://github.com/openclimatefix/nged-substation-forecast/issues/224)), the
> [access-phasing](#access-phasing) Stages 2–3
> ([#328](https://github.com/openclimatefix/nged-substation-forecast/issues/328),
> [#329](https://github.com/openclimatefix/nged-substation-forecast/issues/329)),
> infra-as-code ([#326](https://github.com/openclimatefix/nged-substation-forecast/issues/326)),
> and the MLflow-server / dev-dashboard future work
> ([#235](https://github.com/openclimatefix/nged-substation-forecast/issues/235),
> [#236](https://github.com/openclimatefix/nged-substation-forecast/issues/236)). The
> [AWS-architecture decision](#aws-architecture) itself is built and running; its durable decision
> record now lives in
> [Production Deployment — Design](../architecture/production-deployment.md#infrastructure-tier-alternatives-to-the-ec2-control-plane-box),
> and the running-cost estimate in
> [AWS Running Costs](../architecture/aws-costs.md). It is retired in full once production
> monitoring lands and its design is promoted,
> per the ship-time triage tracked in
> [Engineering health](engineering-health.md#scientific-rigor-tests-and-cleanup).

*History: the inference-asset and monitoring designs were absorbed from the Dagster ML-assets
plan (phases 0–6.7 complete, PRs #182–#214); its final cleanup phase lives in
[Engineering health](engineering-health.md#scientific-rigor-tests-and-cleanup).*

## Requirements

**v0.1 (get a live forecast running on AWS):**

- Deploy the naive forecast so it runs live on AWS, 6-hourly, writing to `power_forecasts`
  (`fold_id="live"`) — the [`live_forecasts` asset](#the-live_forecasts-asset).
- Forecast *quality* does not matter yet — science improvements (baselines, XGBoost) are
  explicitly out of scope for this milestone.
- Production inference must have **zero dependency on MLflow at runtime** — for v0.1 the
  champion model is baked directly into the container image at build time and loaded via a
  plain `save`/`load`, so there is no run ID, cache lookup, or tracking-server call on the hot
  path at all.
- Support both **live** (current partition) and **replay** (historical backfill) NWP
  availability modes, so missed runs can be backfilled
  ([#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208)).
- **Use Dagster "properly"**: persistent run history, one-click UI backfills of missed
  partitions, and the ability to launch backtests on AWS whenever the model improves.
- **Multi-user access** to the Dagster UI (and, later, to the dev dashboard) (rules out single-user tracking
  services).
- **Portability**: the entire stack must also run on a local laptop (or any cloud) via
  `docker compose up` — no AWS-specific service (EventBridge, Step Functions) may be
  load-bearing for scheduling or orchestration. Portability is both a development convenience
  and a handover requirement — see [the orchestration
  decision](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm).
- AWS infrastructure with **no static AWS keys** (IAM roles throughout), basic alerting on task
  failure (SNS → email), and cost-conscious operation (~£25–35/month target).

**Post-v0.1 (explicitly deferred):**

- Forecast quality/science work —
  [baselines](metrics-and-leaderboard.md#baseline-forecasters),
  [XGBoost improvements](xgboost-improvements.md).
- The NGED-delivery schema/contract
  ([#96](https://github.com/openclimatefix/nged-substation-forecast/issues/96)) — v0.1 is
  "forecast running", not "delivery contract live".
- Basic **per-task failure email alerting** (a failed run → SNS → email). Sentry error telemetry
  and the [missed-check-in alarm](#alert-on-absence-the-missed-check-in-alarm) have shipped
  ([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63) — see
  [Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence));
  the remaining piece is the thin SNS→email notification edge for individual run failures.
- An **MLflow tracking server** (issue [#235](https://github.com/openclimatefix/nged-substation-forecast/issues/235)) and a separate **development dashboard** ([#236](https://github.com/openclimatefix/nged-substation-forecast/issues/236)), both hosted on the
  always-on control-plane box once it exists — see the [note below](#aws-architecture).
- [Production monitoring](#production-monitoring): score live forecasts over trailing 24h/7d
  windows, logged to a dedicated MLflow experiment, with a manual, auditable way to retire
  stale experiment partitions.
- For more plans for post-v0.1, please see [the milestones (v0.2 onwards)](index.md#milestones).

## The `live_forecasts` asset

Issue: [#221](https://github.com/openclimatefix/nged-substation-forecast/issues/221)

> **Status: ✅ Implemented**, alongside the `promoted_model` promotion asset and local 6-hourly
> automation (`dg dev` + persistent `DAGSTER_HOME`, part of
> [#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208)). The design
> rationale (single-run vs. bulk mode, the `live`/`replay` asymmetry, the trained-population
> invariant) now lives at
> [Production Deployment — Design: Live inference](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/#run-live-inference-in-single-run-mode-not-bulk);
> the operational runbook — promoting a model, running the schedule, backfilling a missed slot —
> is [Operating the live service](../live_service/operations.md), the permanent home
> this page's shipped material moves to (and, eventually, this whole page, once every section
> below has landed).

## Production model artifacts

Issue: [#222](https://github.com/openclimatefix/nged-substation-forecast/issues/222)

> **Status: ✅ Done.** The design decision (bake the champion model into the image at build
> time; no MLflow at runtime) and its rationale now live at
> [Production Deployment — Design](https://openclimatefix.github.io/nged-substation-forecast/architecture/production-deployment/);
> the promotion/build/verify runbook lives at
> [Setting up the live service on AWS](https://openclimatefix.github.io/nged-substation-forecast/live_service/aws/).

## AWS architecture

Issue: [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206) (done)

> **Status: ✅ Built and running.** The accepted option — a small EC2 control-plane box +
> `EcsRunLauncher` — runs on AWS as v0.2 (`v0.2.0`), the same architecture v0.1 (`v0.1.0`) used,
> unchanged; the bring-up is documented in
> [Setting up the live service on AWS](../live_service/aws.md). The
> [access-phasing](#access-phasing) Stages 2–3 and the future-work items (MLflow server, dev
> dashboard) remain post-v0.1.

The Level 1 ("nothing always-on") design proposed in issue #206 is rejected. Its cost case does
not hold, and there are two requirements it cannot serve. The cost case rests on pricing the
always-on control plane at ~£70–105/month, which is a 16 GB box big enough to run the *compute*.
A small control-plane box costs ~£10–20/month (costed 2026-07-02). Its RDS prerequisite goes
away on a single machine too (Postgres-in-Docker, or SQLite on a real local filesystem). The
two requirements Level 1 does not serve:

1. **Use Dagster "properly"** — persistent run history, one-click UI backfills of missed
   partitions, and the ability to launch backtests on AWS whenever the model improves.
2. **An always-on dev dashboard** (a simple Marimo web app showing the latest forecasts) —
   so a dashboard must be always-on regardless.

**Decision: small EC2 control-plane box + `EcsRunLauncher`, decided 2026-07-11.** The
implementation workstreams are identical under every option except the infrastructure one.

The decision was pressure-tested again in July 2026 against the fully serverless alternative —
EventBridge Scheduler firing an ECS `RunTask` directly, with no always-on control plane — and
stands. The durable rationale (portability, NGED handover, illusory cost saving, retry parity)
and the accepted trade-offs (mitigated by the external
[missed-check-in alarm](#alert-on-absence-the-missed-check-in-alarm)) are recorded at
[Production Deployment — Design: Orchestration](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm).

### Cost summary

The running-cost estimate for the accepted option — the headline **~£25–35/month**, the
workload model behind it, and the storage and data-transfer arithmetic — now lives at its
durable home, [AWS Running Costs](../architecture/aws-costs.md), alongside a projected
estimate for running at v2 scale (~2,500 time series). The per-option figures sit in the
decision record: the rejected alternatives range from ~£12–22/month (Option A, which fails two
requirements) to ~£56–86/month (Option C).

The accepted architecture — a small EC2 control-plane box running Dagster behind Tailscale,
dispatching every run to an ephemeral Fargate task via `EcsRunLauncher` — is built and running
in production. Its rationale, and the infrastructure-tier alternatives rejected in its favour (a
fully serverless "nothing always-on" design, one big box running all compute, a serverless
control plane with RDS, and Dagster+ Solo), are recorded as a decision history in [Production
Deployment — Design: infrastructure tier alternatives to the EC2 control-plane
box](../architecture/production-deployment.md#infrastructure-tier-alternatives-to-the-ec2-control-plane-box).

**Future work (post-v0.1):** once an always-on control-plane box exists (the accepted option or
later), it's also
a natural home for an **MLflow tracking server** (network-reachable, persistent — replacing the
local file-store) and a **"development dashboard"** (a Marimo app for researchers). Neither is
needed to ship v0.1; revisit once the box exists — see [Access phasing](#access-phasing) below for
how each one's exposure rolls out once it does.

### Access phasing

The sections above decide *where* the accepted option's pieces run; this one decides *who can
reach them, when*. Three stages, each additive on top of the last — nothing built in an earlier
stage is reworked in a later one.

The constraint driving all three stages: **none of the three web UIs has any built-in
authentication.** Open-source `dagster-webserver` has no users, roles, or login (auth/RBAC is a
Dagster+ feature) — anyone who can reach the port can launch and terminate runs and wipe
materialisations. MLflow's tracking UI and the Marimo dashboard are equally open. So the network
layer *is* the auth layer: "no public inbound ports" in Stage 1 and the read-only-second-webserver
pattern in Stage 2 are load-bearing security decisions, not tidiness.

#### Stage 1 — solo, Tailscale only

This is exactly what the [accepted
option](../architecture/production-deployment.md#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month)
already describes; it's named explicitly as Stage 1 here only so Stage 2/3 below have something
to say "additive on top of."

- Daemon, full-access Dagster webserver, and the Marimo dashboard all run on the `t4g.medium`
  control-plane box, alongside MLflow once its tracking server lands (the "Future work" item
  just above, issue [#235](https://github.com/openclimatefix/nged-substation-forecast/issues/235))
  — MLflow-on-the-box is not itself a new v0.1 commitment, only its access tier is decided here.
- Security group: no public inbound ports at all. Everything is reached over Tailscale.

#### Stage 2 — team gets read-only Dagster access; MLflow stays private

Issue: [#328](https://github.com/openclimatefix/nged-substation-forecast/issues/328)

- A **second** `dagster-webserver --read-only` process on the same box, against the same
  Postgres + code location — cheap: just another process, no new daemon, no new infra beyond
  this.
- **Caddy** in front of it, handling TLS via Let's Encrypt automatically.
- **oauth2-proxy** in front of the read-only webserver for Google sign-in, using
  `--authenticated-emails-file` (not `--email-domain` — NGED/NIA collaborators aren't on one
  Workspace domain).
- Security group opens port 443 for the first time — this is the real milestone of this stage.
- One DNS subdomain (e.g. `dagster.<domain>`).
- **MLflow is explicitly not exposed in this stage** — it stays Tailscale-only, unchanged from
  Stage 1. There is no native read-only public/gated MLflow mode without either the
  still-experimental auth plugin or allowlisting specific endpoints at the proxy — not worth it
  unless there's a real need.
- The full-access Dagster webserver, Marimo, and MLflow remain exactly as in Stage 1.
- **Lighter alternative if (and only if) the audience is a couple of OCF colleagues:** skip
  Caddy, oauth2-proxy, DNS, and the security-group change entirely by inviting them into the
  tailnet instead. Tailscale already authenticates members with their Google accounts, and a
  Tailscale ACL can expose just the read-only webserver's port to the team while the full-access
  webserver stays restricted. Worth checking whether OCF already runs an organisation tailnet
  before building this — if so, most of the setup (and the per-user pricing question below) is
  already settled. Trade-offs: everyone must run the Tailscale client (fine for colleagues, wrong
  for browser-only NGED/NIA collaborators), and past the free tier's ~3 users Tailscale bills per
  user, whereas oauth2-proxy is free. So treat tailnet-sharing as a cheap Stage-1.5 for
  OCF-internal read access; the Caddy + oauth2-proxy build above remains the destination once
  anyone outside the tailnet needs the UI. The two aren't mutually exclusive: starting with
  tailnet-sharing and adding the proxy later loses nothing, since Stage 1.5 builds none of the
  pieces Stage 2's proxy needs but also breaks none of them.

#### Stage 3 — public Marimo dashboard, curated public data

Issue: [#329](https://github.com/openclimatefix/nged-substation-forecast/issues/329)

- A **separate** Marimo instance, not the private one — reads only a public-safe subset of data
  (ideally its own S3 prefix), runs as its own ECS Fargate task/service, no ALB.
- **No CloudFront/Lambda@Edge** — instead, exploit the fact that the control-plane box is
  already always-on and already running Caddy:
    - Caddy gets one more route, pointed at a small custom **wake-proxy** service (not a direct
      backend).
    - Wake-proxy: checks whether the Fargate task/service is running (`ecs describe-services`);
      if not, sets desired count to 1 and polls until healthy, then reverse-proxies the request
      through.
    - A background loop in the wake-proxy scales desired count back to 0 after an idle period
      (e.g. 15 min).
    - UX: don't hold the connection open for the full ~30–60s cold start — serve an immediate
      "warming up" holding page that polls a `/ready` endpoint every 2–3s, then redirects once
      healthy.
- One more DNS subdomain (e.g. `dashboard.<domain>`), no oauth2-proxy — fully public, no login.
- **Trade-off:** this couples the public dashboard's *availability* to the control-plane box's
  uptime. Compute isolation is preserved — a Marimo bug can't touch the Dagster daemon, since
  it's a separate task — but if the box goes down, the public dashboard becomes unreachable too,
  not just the private Dagster/MLflow. Accepted trade-off for solo-project simplicity over
  running a fourth AWS service (an ALB) just for this.
- Rejected alternative: Marimo's WASM/Pyodide static export (zero server, hosted on
  S3+CloudFront). Ruled out based on direct prior experience — ~30s browser-side cold start on a
  similar personal project, and clunkier than a thin server-side wake mechanism.
- Rejected alternative: the CloudFront + Lambda@Edge wake-proxy pattern (the "textbook" AWS
  scale-to-zero workaround). Ruled out in favour of reusing the existing box's Caddy plus a
  custom script — fewer AWS services (no ALB, no CloudFront distribution, no Lambda@Edge), since
  compute is already sitting there.

**Sequencing:** Stage 1 is built and run solo, by hand — not enough moving parts yet to justify
infrastructure-as-code. Stage 2 (the read-only webserver, Caddy, oauth2-proxy, the
security-group change) is the natural point to start writing infra-as-code, since that's when
IAM roles, security groups, and multiple processes start accumulating enough to be worth
codifying and reproducing — see the open Terraform-vs-CDK question in
[Deployment workstream 3](#deployment-workstream-3-aws-infrastructure).

**Handover caveat (added 2026-07-14):** all three stages are designed for the phase in which
*OCF* runs the service on OCF's AWS account. Once the service moves to NGED's own AWS account
(the preferred post-NIA operating model — see [Handover to NGED](handover.md)), Tailscale
specifically may not survive NGED's security review, and because the network layer is the auth
layer here, that would require an NGED-compatible replacement for the whole access design, not
just a component swap. That risk is a reason to [probe NGED's landing-zone constraints
early](handover.md#5-probe-ngeds-aws-landing-zone-early) — not a reason to change Stages 1–3,
which remain correct for the OCF-run phase.

## Production monitoring

Issue: [#224](https://github.com/openclimatefix/nged-substation-forecast/issues/224)

*Depends on the [`live_forecasts` asset](#the-live_forecasts-asset) — there is nothing to
monitor until live forecasts exist.*

The `metrics` asset implements the `leaderboard` and `ad_hoc` scopes; `production_monitoring`
is declared in `EVALUATION_SCOPES` but unimplemented (`EvalScopeType` in
`contracts/ml_schemas.py:219` deliberately omits it). And with thousands of experiments
planned, the `cv_experiment_folds` dynamic partition set grows without bound — partition keys
need a retirement path that cannot lose results.

### The `production_monitoring` evaluation scope

- Extend `EvalScopeType` to `Literal["leaderboard", "production_monitoring", "ad_hoc"]`,
  bringing it in sync with `EVALUATION_SCOPES` (the docstring at `ml_schemas.py:222` already
  anticipates this).
- Remove the CV-folds-only restriction in `compute_metrics()` (documented in its docstring):
  `fold_id="live"` rows use the same join logic, with window bounds supplied by the caller
  (trailing windows, not fold dates).
- Scope behaviour in the `metrics` asset: score `fold_id="live"` forecasts over two trailing
  `valid_time` windows — **last 24 hours** and **last 7 days**. Each window writes rows to
  `forecast_metrics` Delta with `window_label` (`"24h"`/`"7d"`), the trailing
  `window_start`/`window_end` bounds, and `computed_at = now` (all columns already exist in the
  `Metrics` schema). These rows are **append-only** — successive runs accumulate the
  sliding-window history (unlike the leaderboard scope's idempotent overwrite; recomputations
  are distinguished by `computed_at`).
- MLflow: log the same aggregates to a **dedicated `production_monitoring` MLflow experiment**
  — never to the golden leaderboard — as **time-series points** (MLflow metric
  timestamp/step), one persistent run per window resolved by tag (mirroring the
  `mlflow_runs` get-or-create convention), so MLflow charts live performance over time (e.g.
  trailing-7d NMAE per `time_series_type`). Stamp `mlflow_run_id` on the Delta rows as the
  cross-link.
- Note: evaluating "the last 24h of production" scores forecasts whose `valid_time` has already
  passed and now has observed power — satisfied naturally as observations accumulate.

### The `monitoring_sensor`

A Dagster sensor that fires on each `power_time_series_and_metadata` materialisation (~every
6 h, when new actuals land) and requests a `metrics` run with
`evaluation_scope="production_monitoring"` over `fold_id="live"` for both trailing windows.
Sensor preferred over a schedule so it fires on the actual data update.

Note this sensor needs a running Dagster daemon — the [accepted
option](../architecture/production-deployment.md#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month)
provides one.

### Alert on absence: the missed-check-in alarm

> **Status: ✅ Shipped** ([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63)).
> The heartbeat, the failure hook, and the laptop/production environment split are built and
> documented as-built in
> [Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence).
> This section keeps the *why* — the design rationale for alerting on absence from outside the
> deployment. The remaining monitoring work on this page is the separate per-task failure email
> edge (SNS → email).

Per-task failure alerts ([Deployment workstream 3](#deployment-workstream-3-aws-infrastructure))
only fire when a task runs and fails. Whole classes of failure are silent: a hung daemon, a
full disk, an expired credential, a schedule that stopped firing. The **primary** production
alert is therefore a **missed-check-in alarm** (Sentry's cron-monitoring terminology): it fires
when **no successful forecast has landed in N hours** (e.g. 8 hours — one missed 6-hourly slot
plus margin), regardless of cause. An alert feeding a runbook — rather than paging or automatic
failover — is a proportionate response because the project's [uptime requirements are lenient
by design](../background/requirements.md#uptime-lenient-by-design). The accepted option's
"daemon silently dead" staleness alarm (described in [the architecture
options](../architecture/production-deployment.md#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month))
is this alarm; recording it here makes it a first-class monitoring deliverable rather than a
side note.

The alarm's *evaluation and delivery* must sit **outside** the service being watched, because
a dead daemon cannot report itself — a Dagster sensor alone can never provide this alarm.
The mechanism is **Sentry cron monitoring**
([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63)): each successful
forecast run checks in with Sentry, and Sentry alerts when an expected check-in fails to
arrive. This satisfies both the outside-the-service requirement and the portability preference
in [Deployment workstream 3](#deployment-workstream-3-aws-infrastructure) — Sentry is external
to the whole deployment, and a check-in ping is plain code that works identically from a
laptop, AWS, or any other cloud. One handover consideration: the Sentry account is OCF's
today, so at handover the alert routing (and possibly the account itself) moves to NGED — see
[Handover to NGED](handover.md#2-alert-on-absence-not-just-failure).

Every alert must link to a runbook that ends in either a specific operator action or
"escalate" — a requirement that matters doubly under the post-NIA operating model, where the
day-to-day operator is a non-expert at NGED (see
[Handover to NGED](handover.md#2-alert-on-absence-not-just-failure)).

### The `retire_experiment_job`

A **manually triggered** job (deliberate and auditable — never automatic) with a single
config field `experiment_name: str`:

1. **Verify before deleting**: the MLflow parent run exists and carries aggregate metrics,
   **and** `power_forecasts` Delta contains rows for this `experiment_name`. If either check
   fails, raise and delete nothing.
2. Delete the experiment's dynamic partition keys via
   `context.instance.delete_dynamic_partition("cv_experiment_folds", key)` for each
   `f"{experiment_name}__{fold_id}"`.
3. Log the deleted keys as output metadata.

Retirement does **not** delete MLflow runs or Delta forecasts — those remain the permanent
record; it only prunes Dagster's execution ledger. Lives beside `register_experiment_job` in
`defs/jobs.py`; ops use `OpExecutionContext` (they need `context.instance`).

### Interaction with the probabilistic metrics

Any metric added to `compute_metrics` flows through this scope automatically — once the
[probabilistic metrics](metrics-and-leaderboard.md#delivering-the-probabilistic-metrics)
(PICP/spread-skill) land, production monitoring tracks ensemble calibration over time for
free. No coupling needed; the ordering is flexible.

## Implementation details (deleted when this ships)

### Deployment workstream 1 — the production job (local dress rehearsal)

Issue: [#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208) (done)

> **Status: ✅ Done** (closed 2026-07-10). The native per-asset Dagster schedules that ship with
> [The `live_forecasts` asset](#the-live_forecasts-asset)
> (`power_time_series_and_metadata_schedule`, `ecmwf_ens_schedule`, `live_forecasts_schedule`)
> do the whole job. Closing #208 took a several-day soak under `dg dev` with a persistent
> `DAGSTER_HOME`, which confirmed 6-hourly forecasts landing with no duplicate rows and a missed
> slot backfillable in replay mode. No hand-rolled freshness op is needed, and neither is the
> one-shot `live_pipeline_job` that [Option
> A](../architecture/production-deployment.md#infrastructure-tier-options-considered-and-rejected) would require (Option A has
> no daemon to hold schedules) — Option A is rejected, so that job is not specified here.

### Deployment workstream 3 — AWS infrastructure

> **Status: ✅ Infrastructure built and running.** ECR, the two IAM roles, the Fargate
> task definition, the always-on EC2 control-plane box (`EcsRunLauncher`, Postgres-in-Docker,
> schedules, Tailscale, Marimo), and S3 run in production, serving v0.2 — stood up by hand and
> documented step-by-step in [Setting up the live service on AWS](../live_service/aws.md). The
> items further down are what remains 🚧 after v0.1.

Every piece of the AWS infrastructure above is built and running, documented step by step in
[Setting up the live service on AWS](../live_service/aws.md); the orchestration design behind
the deployment is in [Production Deployment — Design](../architecture/production-deployment.md).

Still 🚧 after v0.1:

- **A bigger backtest task definition** (e.g. 8 vCPU / 32 GB) — the live 4 vCPU / 16 GB
  definition exists; right-size it after a week of CloudWatch metrics.
- **Per-task failure email alerting** (a failed run → SNS → email) — Sentry error telemetry and
  the missed-check-in alarm already shipped in
  [#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63); this is the one
  piece of alerting still open. Decided 2026-07-14: prefer **portable
  alerting logic over AWS-native glue** (no EventBridge rules) — the checks should be plain code
  that runs end-to-end on a laptop or any cloud, with only the thin notification edge (SNS,
  SMTP, …) being platform-specific. Under the accepted option, Dagster's own run-failure sensors
  are the natural portable mechanism (the daemon evaluates them); an EC2 instance-status-check
  alarm and the [missed-check-in alarm](#alert-on-absence-the-missed-check-in-alarm) cover the
  silent-failure classes ("daemon silently dead") that per-task alerts miss. The
  SNS-subscription-confirm step joins the [runbook](../live_service/aws.md) once this lands.
- **Infra-as-code**
  ([#326](https://github.com/openclimatefix/nged-substation-forecast/issues/326)) once there's
  enough to justify it — per the
  [Access phasing sequencing note](#access-phasing), that point is Stage 2, not Stage 1.
  **Open question, not yet decided:** a small Terraform module (one file) versus **AWS CDK
  (Python)**. The case for CDK is specific to this project: it's single-cloud (AWS-only), so
  there's no cross-cloud benefit from HCL, and CDK lets the infra be written in Python rather
  than learning a new language for it. Terraform vs CDK is a call for whoever starts Stage 2
  work; this page does not pick one. The post-NIA operating model (NGED runs the service on
  NGED's AWS — see
  [Handover to NGED](handover.md#4-infrastructure-as-code-portable-to-ngeds-account)) adds two
  inputs to that call: the infra-as-code must be **account-portable** (no OCF-specific names or
  network assumptions baked in), and what NGED's infrastructure teams already know and are
  allowed to run matters as much as what suits OCF — worth asking them before deciding. By
  handover time, infra-as-code is mandatory, not optional.

### Related GitHub issues

The issues below are the ones still open. Everything else under the v0.1 epic
([#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137)) has shipped, and
that epic is the record of what landed when.

| Issue | Where it lands in this plan |
|---|---|
| [#246 Scale `power_fcst` to [−1, +1] using the static P99 effective capacity](https://github.com/openclimatefix/nged-substation-forecast/issues/246) | Not detailed on this page — decided 2026-07-03 (see the issue for the full worklist); an open follow-up, not required for v0.1 |
| [#96 Write power forecasts in schema agreed with NGED](https://github.com/openclimatefix/nged-substation-forecast/issues/96) | Deferred to the v1.0 epic ([#133](https://github.com/openclimatefix/nged-substation-forecast/issues/133)) — v0.1 is "forecast running", not "delivery contract live" |
| [#5 Backup procedure for data & models on Jack's workstation](https://github.com/openclimatefix/nged-substation-forecast/issues/5) | Deferred to the v0.2 epic ([#138](https://github.com/openclimatefix/nged-substation-forecast/issues/138)); largely superseded once S3 is the primary store |

### Monitoring — tests and verification

Tests:

- Sensor fires on a power update and requests the monitoring run.
- Monitoring rows land in Delta (append-only, correct window bounds from an injected clock) and
  in the `production_monitoring` MLflow experiment — and **never** touch a leaderboard run.
- The trailing window-bounds calculation is a pure helper (injected `now`), unit-tested.
- `retire_experiment_job` refuses when results are absent (each check independently); deletes
  keys when both are present; MLflow + Delta untouched either way.

Verification: trigger a power update (or the sensor manually), see trailing-24h/7d metrics
appear in the `production_monitoring` MLflow experiment and `forecast_metrics`; run
`retire_experiment_job` on a throwaway experiment and watch its partitions disappear from the
Dagster UI while its MLflow runs and Delta rows remain.
