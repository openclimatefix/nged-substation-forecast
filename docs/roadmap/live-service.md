# Live service (v0.1 AWS deployment)

> **Status: ✅ v0.1 shipped (July 2026, `v0.1.0`) — the naive forecast is deployed and running
> on AWS.** Epic [#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137)
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
> [#236](https://github.com/openclimatefix/nged-substation-forecast/issues/236)) — plus the
> durable record of the costed [AWS-architecture decision](#aws-architecture) behind the
> deployment. It is retired in full once production monitoring lands and its design is promoted,
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
  load-bearing for scheduling or orchestration. This is both a development convenience and a
  handover requirement — see
  [the orchestration decision](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm).
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

> **Status: ✅ Built and running (`v0.1.0`).** The accepted option below is deployed on AWS; the
> bring-up is documented in
> [Setting up the live service on AWS](../live_service/aws.md). The costed comparison and the
> rejected alternatives are kept below as the durable record of the decision. The
> [access-phasing](#access-phasing) Stages 2–3 and the future-work items (MLflow server, dev
> dashboard) remain post-v0.1.

An earlier version of this plan committed to the Level 1 ("nothing always-on") design from
issue #206. A
2026-07-02 pressure-test of that decision found the #206 cost analysis substantially wrong:
the always-on control plane it priced at ~£70–105/month actually costs ~£10–20/month (it
priced a 16 GB box big enough to run the *compute*, not a small control-plane box), and its
RDS prerequisite dissolves on a single machine (Postgres-in-Docker, or SQLite on a real local
filesystem). Two requirements also firmed up that Level 1 does not serve:

1. **Use Dagster "properly"** — persistent run history, one-click UI backfills of missed
   partitions, and the ability to launch backtests on AWS whenever the model improves.
2. **An always-on dev dashboard** (a simple Marimo web app showing the latest forecasts) —
   so *something* must be always-on regardless.

**Decision: small EC2 control-plane box + `EcsRunLauncher`, decided 2026-07-11 — the
[accepted option](#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month) below.**
The four alternatives considered and rejected are recorded further down so the decision has a
durable record. The implementation workstreams are identical under every option except the
infrastructure one.

The decision was pressure-tested again in July 2026 against the fully serverless alternative —
EventBridge Scheduler firing an ECS `RunTask` directly, with no always-on control plane — and
stands. The durable rationale (portability, NGED handover, illusory cost saving, retry parity)
and the accepted trade-offs (mitigated by the external
[missed-check-in alarm](#alert-on-absence-the-missed-check-in-alarm)) are recorded at
[Production Deployment — Design: Orchestration](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm).

### Cost summary

All prices are for **eu-west-2 (London)** — the nearest AWS region to the UK — taken from
the AWS price-list API (2026-07-03). AWS bills in USD; figures here are converted at
**\$1 = £0.75** (ECB rate, 2026-07-03).

**These estimates do *not* include ML training.** For v1 we expect most — maybe all — ML
training to run on our own laptops; the only training AWS would see is an optional
UI-launched backtest (~£0.65/run, priced below).

Estimated total for the accepted option: **~£25–35/month**, made up of:

| Component | £/month |
|---|---|
| Always-on control plane (EC2 `t4g.medium` + 20 GB EBS) | 15–22 (1-yr reserved vs on-demand) |
| Live Fargate inference (4 runs/day) + hourly polling | 5–9 |
| S3 storage (~130 GB) + requests | 3–4 |
| Data transfer (ingress + egress) | ≈0 |
| Backtests | ~£0.65 per run, as needed |

[Access phasing](#access-phasing) (below) adds negligible spend on top of this accepted-option
estimate: Stage 2's Caddy and oauth2-proxy, and Stage 3's wake-proxy, all run as plain processes on
the existing control-plane box — zero new billable resources. The only new billable resource
across all three stages is the second Fargate task/service for Stage 3's public Marimo instance,
already roughly priced by the same-shaped workload in
[Option D](#option-d-serverless-control-plane-no-pets-4145month) (~£6.20/month for Marimo as its
own tiny Fargate service).

The rejected alternatives range from ~£12–22/month (Option A, which fails two requirements) to
~£56–86/month (Option C). The detailed analysis follows: the
[workload model](#workload-model), [storage & data transfer](#storage-data-transfer-common-to-all-options)
(common to every option), then the accepted option's cost breakdown, then the rejected
alternatives' costs.

### Workload model

Live cadence: `ecmwf_ens` 1/day (daily 00Z partition), `power_time_series_and_metadata` 4/day,
`live_forecasts` 4/day (6-hourly partitions), the monitoring `metrics(production_monitoring)`
step ~4/day → **~13 materialisations/day ≈ 395/month**. This cadence ingests only ECMWF ENS and
the NGED power feed today; a near-real-time ERA5/ERA5T ingest would join it *only if* live capacity
estimation is made to depend on ERA5 — a new external dependency we may prefer to avoid by
[keeping ERA5 offline](capacity-estimation.md#irradiance-inputs). A backtest experiment today is ~4–6
materialisations (`eligible_time_series` + `trained_cv_model` + `cv_power_forecasts` per
fold, plus `metrics`; single leaderboard fold), rising to ~15–20 under the future
multi-yearly-fold epoch.

Fargate compute: x86 £0.0349/vCPU-hr + £0.0038/GB-hr; **ARM 20% cheaper** (£0.0279 +
£0.0031) and polars/XGBoost run fine on arm64. A right-sized 4 vCPU / 16 GB ARM task
(measured inference peak ~9 GB) is £0.16/hr → 4 × 15-min live runs/day ≈ **£5/month**;
hourly empty polling wake-ups add ~£1.90/month; a 2-hour 8 vCPU / 32 GB ARM backtest ≈
**£0.65/run**. (The old £11–19/month anchor assumed 8 vCPU / 32 GB x86.)

### Storage & data transfer (common to all options)

Identical under every option, so counted once and folded into each headline range below —
**~£3–4/month in total**:

- **S3 Standard storage — ~£2.50–3/month.** The working set is ~130 GB (~100 GB NWP +
  ~30 GB forecasts) at £0.018/GB-month → ~£2.40/month, plus headroom for Delta version
  history between vacuums. Grows as daily NWP partitions accumulate.
- **S3 requests — ~£0.50–1.50/month.** Delta Lake is request-heavy (transaction-log JSON
  reads, checkpoints, many small parquet GETs per scan), but ~13 materialisations/day is
  tiny volume: a generous 1–2 M GET (£0.00031/1k) + 100–200 k PUT/COPY/POST/LIST
  (£0.0040/1k) per month lands well under £1.50.
- **Data transfer — ≈£0/month.** Ingress is free (the daily NWP download from Dynamical
  costs nothing on the AWS side); S3 ↔ Fargate/EC2 traffic within eu-west-2 is free;
  internet egress (the Tailscale-tunnelled Dagster UI and Marimo dashboard) is a few
  GB/month, inside AWS's account-wide 100 GB/month free egress allowance (£0.067/GB
  beyond).

### Accepted option: small EC2 control-plane box + `EcsRunLauncher` ~£25–35/month

A `t4g.medium` (2 vCPU / 4 GB; **£20.55/month on-demand, £12.95/month 1-yr no-upfront
reserved**) runs the Dagster daemon + webserver + code-location server + Postgres-in-Docker +
**the Marimo dashboard**, behind Tailscale (no ALB). Every run — live schedules *and*
UI-launched backtests — is dispatched by `EcsRunLauncher` to an ephemeral Fargate task sized
to the job. Add ~£1.80 EBS, £5–7/month live Fargate, ~£0.65/backtest.

**One image, four roles.** The daemon, webserver, and code-location server all run from the
*same* [production image](../architecture/production-deployment.md) the ephemeral Fargate runs
use (harmless — the baked-in champion model is dead weight for the first three, only a run's
actual execution touches it) — a `docker-compose.yml` on the box just launches each as a
separate service with a different command override. The compose file, and the entrypoint
gotchas it has to navigate (`dagster-webserver`/`dagster-daemon` are separate console-script
binaries, not `dagster` subcommands; and `EcsRunLauncher` generates a run command that itself
starts with `dagster`, so the run task definition must neutralise the image's
`ENTRYPOINT ["dagster"]`), are now written up in the runbook —
[Setting up the live service on AWS: Steps 9 and 14](../live_service/aws.md#step-9-create-the-ecs-cluster-and-fargate-task-definition).

- **Pros:** Dagster properly (history, UI backfills, sensors, concurrency pools enforced
  centrally); backtests get big ephemeral compute; dashboard rides free; EC2 IAM instance
  roles (no static keys); the textbook Dagster-OSS-on-AWS deployment.
- **Cons:** one pet server (patching, disk, daemon liveness — mitigate with systemd restart
  policies + the monitoring plan's "no fresh forecast" alarm); dagster.yaml/run-launcher
  config work; 4 GB is comfortable but not roomy (watch Marimo's Delta scans). The pet-server
  risk grows once a non-expert operates the service — the full mitigation list (auto-recovery
  alarms, disk hygiene, a tested rebuild-from-scratch script) is
  [Handover workstream 3](handover.md#3-de-pet-the-control-plane-box).
- Cost trims: t4g.small (2 GB) is **free-trial (750 hrs/month) until 31 Dec 2026** and
  £10.30/£6.50 after, if everything squeezes into 2 GB — likely too tight with Marimo.

### Considered but rejected

The four alternatives below were researched alongside the accepted option above and rejected in
its favour. They're kept here as a durable record of the decision, not as live options.

#### Option A — Level 1: nothing always-on ~£12–22/month

Hourly EventBridge Scheduler → one-shot Fargate task → `dagster job execute` against a
throwaway local `DAGSTER_HOME` → exit. Freshness check is the first op (latest Dynamical init
vs the NWP init behind the newest forecast in `power_forecasts` on S3); failure recovery is
the cron (stale outputs → next tick re-runs); Delta commits provide the atomicity the
freshness logic needs; "which partitions need materialising" derived from Delta contents vs
Dynamical availability, never Dagster's records (they evaporate with the throwaway SQLite).

- **Pros:** cheapest; zero servers; self-healing by construction; everything it builds is
  needed by the accepted option (and C/D) anyway.
- **Cons:** fails both new requirements — no run history, hand-rolled backfills (issue #208
  replay), backtests stay on the workstation, and the Marimo dashboard needs a separate
  always-on home (+£6/month Fargate service) anyway.

#### Option C — one big box with everything on it ~£56–86/month

Dagster + Postgres + Marimo + *all compute* (inference peaks ~9 GB → 16 GB box): EC2
`t4g.xlarge` £82.20 on-demand / £51.90 reserved-1yr; Lightsail 16 GB £61.70 (4 vCPU, 40% CPU
baseline); Lightsail memory-optimised 16 GB £54.40 (2 vCPU). This is #206's "Level 3" and how
OCF runs everything else (with Airflow).

- **Pros:** simplest architecture possible — no Fargate/ECR-per-run/EventBridge/run-launcher;
  `docker compose` + `git pull`.
- **Cons:** priciest; backtests capped at 2–4 vCPU (slower than the workstation); Lightsail
  variants mean static AWS keys and burst-credit accounting; biggest pet.

#### Option D — serverless control plane (no pets) ~£41–45/month

Daemon + webserver as one always-on 0.5 vCPU / 2 GB ARM Fargate service (£14.70), RDS
`db.t4g.micro` Postgres (£10–12, approximate — the one unverified price), Marimo as its own
tiny Fargate service (approx £6.20), runs on ephemeral Fargate.

- **Pros:** full Dagster with zero servers to patch; IAM-native throughout.
- **Cons:** RDS is the tax (no local disk → no Postgres-in-Docker); webserver access needs an
  ALB (+~£16.50/month) or a Tailscale-sidecar hack; the most Terraform. Cleaner on paper than
  in practice.

#### Option E — Dagster+ Solo, Hybrid ~£37–45/month typical

£7.50/month base + £0.030/credit (1 credit = 1 materialisation or op execution; May 2026
pricing). Live cadence ≈ 395 credits ≈ £12/month; backtests £0.15–0.52 each (a heavy
50-experiment month adds £7.50–26). Plus a stateless Hybrid agent (~£6.75 tiny Fargate
service) and Marimo hosting (~£3.75–6), and the same per-run Fargate compute. Hybrid incurs
no serverless charge; sensor evaluations are free (an hourly freshness *op* would cost
+£22/month — on Dagster+ you use sensors, the better design anyway).

- **Pros:** managed UI/daemon/alerting/backfills with nothing of ours to keep alive.
- **Cons:** **Solo is 1 user** — collaborators can't log in (Starter is £75/month base);
  metering shapes design decisions; vendor dependency. The 1-user cap is likely disqualifying
  for an OCF collaboration.

### Comparison

| | A: Level 1 | Accepted: small box + Fargate | C: one big box | D: serverless CP | E: Dagster+ Solo |
|---|---|---|---|---|---|
| £/month | 12–22 | **25–35** | 56–86 | ~41–45 (+ALB) | 37–45 |
| Run history + UI backfills | ✗ | ✓ | ✓ | ✓ | ✓ |
| Backtests in AWS | ✗ | ✓ big ephemeral compute | ✓ but 2–4 vCPU | ✓ | ✓ (credits) |
| Marimo dashboard | +£6 add-on | ✓ free on box | ✓ free on box | +£6 service | +£3.75–6 |
| Servers to patch | 0 | 1 | 1 | 0 | 0 |
| No static AWS keys | ✓ | ✓ | ✓ EC2 / ✗ Lightsail | ✓ | ✓ |
| Multi-user UI | n/a | ✓ (Tailscale) | ✓ | ✓ | ✗ (1 user) |

**Recommendation (accepted): the small-box option** — the only shape giving full Dagster *and*
workstation-beating backtest compute *and* a free dashboard home, for ~£13–15/month over Level 1.
Option C is the fallback if operational simplicity trumps backtest speed and ~£30/month. D pays
an RDS+ALB tax for purism; E's 1-user cap rules it out.

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
materialisations; MLflow's tracking UI and the Marimo dashboard are equally open. So the network
layer *is* the auth layer: "no public inbound ports" in Stage 1 and the read-only-second-webserver
pattern in Stage 2 are load-bearing security decisions, not tidiness.

#### Stage 1 — solo, Tailscale only

This is exactly what the
[accepted option](#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month) section
above already describes; it's named explicitly as Stage 1 here only so Stage 2/3 below have
something to say "additive on top of."

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
  tailnet instead — Tailscale already authenticates members with their Google accounts, and a
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
just a component swap. This is a reason to
[probe NGED's landing-zone constraints early](handover.md#5-probe-ngeds-aws-landing-zone-early)
— not a reason to change Stages 1–3, which remain correct for the OCF-run phase.

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
  `_mlflow_runs` get-or-create convention), so MLflow charts live performance over time (e.g.
  trailing-7d NMAE per `time_series_type`). Stamp `mlflow_run_id` on the Delta rows as the
  cross-link.
- Note: evaluating "the last 24h of production" scores forecasts whose `valid_time` has already
  passed and now has observed power — satisfied naturally as observations accumulate.

### The `monitoring_sensor`

A Dagster sensor that fires on each `power_time_series_and_metadata` materialisation (~every
6 h, when new actuals land) and requests a `metrics` run with
`evaluation_scope="production_monitoring"` over `fold_id="live"` for both trailing windows.
Sensor preferred over a schedule so it fires on the actual data update.

Note this sensor needs a running Dagster daemon — the
[accepted option](#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month) provides
one. If the deployment instead ships Option A (nothing always-on), skip the sensor and run the
monitoring step as the final op of the one-shot production job (the production-job workstream
below already reserves that slot).

### Alert on absence: the missed-check-in alarm

> **Status: ✅ Shipped** ([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63)).
> The heartbeat, the failure hook, and the laptop/production environment split are built and
> documented as-built in
> [Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence).
> This section keeps the *why* — the design rationale for alerting on absence from outside the
> deployment. The remaining monitoring work on this page is the separate per-task failure email
> edge (SNS → email).

Per-task failure alerts ([Deployment workstream 3](#deployment-workstream-3-aws-infrastructure))
only fire when something runs and fails. Whole classes of failure are silent: a hung daemon, a
full disk, an expired credential, a schedule that stopped firing. The **primary** production
alert is therefore a **missed-check-in alarm** (Sentry's cron-monitoring terminology): it fires
when **no successful forecast has landed in N hours** (e.g. 8 hours — one missed 6-hourly slot
plus margin), regardless of cause. An alert feeding a runbook — rather than paging or automatic
failover — is a proportionate response because the project's
[uptime requirements are lenient by design](../background/requirements.md#uptime-lenient-by-design).
The accepted option's "daemon silently dead" staleness alarm (mentioned under
[the architecture options](#accepted-option-small-ec2-control-plane-box-ecsrunlauncher-2535month))
is this alarm; recording it here makes it a first-class monitoring deliverable rather than a
side note.

The alarm's *evaluation and delivery* must sit **outside** the service being watched, because
a dead daemon cannot report itself — a Dagster sensor alone can never provide this alarm.
The planned mechanism is **Sentry cron monitoring**
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

> **Status: ✅ Done** (closed 2026-07-10). It shipped differently than originally sketched: no
> hand-rolled freshness op or one-shot `live_pipeline_job` was built. The native per-asset
> Dagster schedules that ship with [The `live_forecasts` asset](#the-live_forecasts-asset)
> (`power_time_series_and_metadata_schedule`, `ecmwf_ens_schedule`, `live_forecasts_schedule`)
> did the whole job: run under `dg dev` with a persistent `DAGSTER_HOME` for several days,
> confirming 6-hourly forecasts landing with no duplicate rows and a missed slot backfillable in
> replay mode. The one-shot `live_pipeline_job` that [Option A](#considered-but-rejected) would
> have needed (Option A has no daemon to hold schedules) was never required, and is not
> reproduced here now that Option A is rejected.

### Deployment workstream 3 — AWS infrastructure

> **Status: ✅ v0.1 infrastructure built and running.** ECR, the two IAM roles, the Fargate
> task definition, the always-on EC2 control-plane box (`EcsRunLauncher`, Postgres-in-Docker,
> schedules, Tailscale, Marimo), and S3 are all stood up by hand and documented step-by-step in
> [Setting up the live service on AWS](../live_service/aws.md). The items further down are what
> remains 🚧 after v0.1.

Built for v0.1 (see the [runbook](../live_service/aws.md) for the exact console steps):

- **ECR** repository; image pushed by hand (tag = model run-id + git SHA) — a CI container build
  is a later, infra-as-code-era step.
- **S3** data bucket mirroring the local `data/` layout (`nwp_data/`, `power_forecasts/`, …).
  The NGED-delivery bucket/prefix is a later step (v0.1 is "forecast running", not "delivery
  contract live").
- **Two IAM roles**, not one — a task *execution* role
  (`AmazonECSTaskExecutionRolePolicy`: ECR pull + CloudWatch Logs) and a task role carrying the
  S3 read/write policy. No static AWS keys anywhere.
- **Fargate task definition**: 4 vCPU / 16 GB ARM for live runs (measured inference peak ~9 GB).
- **EC2 `t4g.medium` control-plane box** (IAM instance role; ~20 GB gp3 EBS) running the Dagster
  daemon, webserver, code-location server, Postgres (run/event/schedule storage, `pg_dump` to S3
  nightly), and the Marimo dashboard under Docker Compose, dispatching every run to Fargate via
  `EcsRunLauncher` (`dagster.yaml` also carries the `pool="ECMWF"` concurrency limit, now
  enforced centrally). The 6-hourly `live_forecasts` schedule and `ecmwf_ens` daily-partition
  automation run on the daemon (freshness via `SkipReason`); `restart: always` +
  unattended-upgrades keep the box up. Reached over Tailscale — no public ingress.

Still 🚧 after v0.1:

- **A bigger backtest task definition** (e.g. 8 vCPU / 32 GB) — the live 4 vCPU / 16 GB
  definition exists; right-size it after a week of CloudWatch metrics.
- **Alerting** ([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63)):
  basic per-task failure alerting (a failed run → email). Decided 2026-07-14: prefer **portable
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
  **Open question, not yet decided:** this section originally specified a small Terraform
  module (one file), but a later conversation argued for **AWS CDK (Python)** instead —
  specifically for this project, since it's single-cloud (AWS-only), so there's no cross-cloud
  benefit from HCL, and CDK lets the infra be written in Python rather than learning a new
  language for it. Terraform vs CDK is Jack's call to make when Stage 2 work starts; this page
  does not pick one. The post-NIA operating model (NGED runs the service on NGED's AWS — see
  [Handover to NGED](handover.md#4-infrastructure-as-code-portable-to-ngeds-account)) adds two
  inputs to that call: the infra-as-code must be **account-portable** (no OCF-specific names or
  network assumptions baked in), and what NGED's infrastructure teams already know and are
  allowed to run matters as much as what suits OCF — worth asking them before deciding. By
  handover time, infra-as-code is mandatory, not optional.

### Related GitHub issues

Row order matches the sub-issue order on the #137 epic itself (the OCF project board), not the
order issues were opened.

| Issue | Where it lands in this plan |
|---|---|
| [#221 Add the `live_forecasts` Dagster asset](https://github.com/openclimatefix/nged-substation-forecast/issues/221) | [The `live_forecasts` asset](#the-live_forecasts-asset) — done |
| [#208 Run every 6 hours locally and backfill missing runs (as a test)](https://github.com/openclimatefix/nged-substation-forecast/issues/208) | [Deployment workstream 1](#deployment-workstream-1-the-production-job-local-dress-rehearsal) — done, via native per-asset schedules |
| [#121 Use obstore instead of pathlib](https://github.com/openclimatefix/nged-substation-forecast/issues/121) | S3-capable data paths — done ([setup guide](https://openclimatefix.github.io/nged-substation-forecast/live_service/setup/)) |
| [#50 Define all paths in Settings](https://github.com/openclimatefix/nged-substation-forecast/issues/50) | S3-capable data paths — done |
| [#222 Build the production Docker image](https://github.com/openclimatefix/nged-substation-forecast/issues/222) | [Production model artifacts](#production-model-artifacts) — done ([promotion runbook](https://openclimatefix.github.io/nged-substation-forecast/live_service/aws/)) |
| [#206 Deploy to AWS!](https://github.com/openclimatefix/nged-substation-forecast/issues/206) | This page (the options above supersede its cost analysis) — done; deployed and running |
| [#286 Create docs for setting up compute infra on AWS](https://github.com/openclimatefix/nged-substation-forecast/issues/286) | [Deployment workstream 3](#deployment-workstream-3-aws-infrastructure) — done; the option-agnostic pieces and the control-plane box are all in the [AWS runbook](../live_service/aws.md). Infra-as-code ([#326](https://github.com/openclimatefix/nged-substation-forecast/issues/326)) is a separate 🚧 follow-up |
| [#197 Make fold-run param logging re-run-safe](https://github.com/openclimatefix/nged-substation-forecast/issues/197) | Bug fix folded into the v0.1 epic (MLflow param immutability on re-runs) |
| [#63 Send telemetry to OCF's Sentry.io](https://github.com/openclimatefix/nged-substation-forecast/issues/63) | Observability — done; Sentry error telemetry + the [missed-check-in alarm](#alert-on-absence-the-missed-check-in-alarm), as-built in [Send telemetry to Sentry](../architecture/production-deployment.md#send-telemetry-to-sentry-and-alarm-on-absence). Per-task failure email alerting is a separate 🚧 follow-up |
| [#246 Scale `power_fcst` to [−1, +1] using the static P99 effective capacity](https://github.com/openclimatefix/nged-substation-forecast/issues/246) | Not detailed on this page — decided 2026-07-03 (see the issue for the full worklist); an open follow-up, not required for v0.1 |
| [#96 Write power forecasts in schema agreed with NGED](https://github.com/openclimatefix/nged-substation-forecast/issues/96) | Deferred to the v1.0 epic ([#133](https://github.com/openclimatefix/nged-substation-forecast/issues/133)) — v0.1 is "forecast running", not "delivery contract live" |
| [#161 More Dagster-UI metrics + validation for NWP ingestion](https://github.com/openclimatefix/nged-substation-forecast/issues/161) | Deferred to the v0.2 epic ([#138](https://github.com/openclimatefix/nged-substation-forecast/issues/138)) — mostly [NWP ingestion completeness checks](engineering-health.md#nwp-ingestion-completeness-checks-and-dagster-metrics) |
| [#5 Backup procedure for data & models on Jack's workstation](https://github.com/openclimatefix/nged-substation-forecast/issues/5) | Deferred to the v0.2 epic ([#138](https://github.com/openclimatefix/nged-substation-forecast/issues/138)); largely superseded once S3 is the primary store |
| [#209 Bump version number to v0.1](https://github.com/openclimatefix/nged-substation-forecast/issues/209) | The final ship step — done (`v0.1.0` tagged July 2026) |

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
