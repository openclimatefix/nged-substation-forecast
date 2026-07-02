# Deploy v0.1 to AWS (issues #137 / #206)

## Context

**Top priority: get *any* forecast running on AWS.** Forecast quality does not matter yet — the
science plans (06 onwards) wait until v0.1 is live. Recommended sequencing across the plan set
(see also the map in `docs/roadmap/index.md`): **01 (CI) → 02 (`live_forecasts`) → 03
(container + baked model) → this plan (04) → 05 (monitoring) → science.**

This plan originally committed to the Level 1 ("nothing always-on") design from issue
[#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206). A 2026-07-02
pressure-test of that decision found the #206 cost analysis substantially wrong: the
always-on control plane it priced at ~$95–140/month actually costs ~$14–27/month (it priced a
16 GB box big enough to run the *compute*, not a small control-plane box), and its RDS
prerequisite dissolves on a single machine (Postgres-in-Docker, or SQLite on a real local
filesystem). Two requirements also firmed up that Level 1 does not serve:

1. **Use Dagster "properly"** — persistent run history, one-click UI backfills of missed
   partitions, and the ability to launch backtests on AWS whenever the model improves.
2. **An always-on dev dashboard** (a simple Marimo web app showing the latest forecasts) —
   so *something* must be always-on regardless.

**Current decision status: leaning Option B (small EC2 control-plane box + `EcsRunLauncher`),
final call pending.** The five options researched are recorded below so the decision has a
durable record. Workstreams 1–3 are identical under every option; only workstream 4 changes.

## Architecture options (researched 2026-07-02; eu-west-2 prices from the AWS price-list API)

### Workload model (from the code and plans 02/05)

Live cadence: `ecmwf_ens` 1/day (daily 00Z partition), `power_time_series_and_metadata` 4/day,
`live_forecasts` 4/day (6-hourly partitions), plan-05 `metrics(production_monitoring)` ~4/day
→ **~13 materialisations/day ≈ 395/month**. A backtest experiment today is ~4–6
materialisations (`eligible_time_series` + `trained_cv_model` + `cv_power_forecasts` per fold,
+ `metrics`; single leaderboard fold), rising to ~15–20 under the future multi-yearly-fold
epoch.

Fargate compute (eu-west-2): x86 $0.04656/vCPU-hr + $0.00511/GB-hr; **ARM 20% cheaper**
($0.03725 + $0.00409) and polars/XGBoost run fine on arm64. A right-sized 4 vCPU / 16 GB ARM
task (measured inference peak ~9 GB) is $0.21/hr → 4 × 15-min live runs/day ≈ **$7/month**;
hourly empty polling wake-ups add ~$2.50/month; a 2-hour 8 vCPU / 32 GB ARM backtest ≈
**$0.86/run**. (The old $15–25/month anchor assumed 8 vCPU / 32 GB x86.)

### Option A — Level 1: nothing always-on — ~$11–25/month

Hourly EventBridge Scheduler → one-shot Fargate task → `dagster job execute` against a
throwaway local `DAGSTER_HOME` → exit. Freshness check is the first op (latest Dynamical init
vs the NWP init behind the newest forecast in `power_forecasts` on S3); failure recovery is
the cron (stale outputs → next tick re-runs); Delta commits provide the atomicity the
freshness logic needs; "which partitions need materialising" derived from Delta contents vs
Dynamical availability, never Dagster's records (they evaporate with the throwaway SQLite).

- **Pros:** cheapest; zero servers; self-healing by construction; everything it builds is
  needed by B/C/D anyway.
- **Cons:** fails both new requirements — no run history, hand-rolled backfills (issue #208
  replay), backtests stay on the workstation, and the Marimo dashboard needs a separate
  always-on home (+$8/month Fargate service) anyway.

### Option B — small EC2 control-plane box + `EcsRunLauncher` — ~$29–42/month ← recommended

A `t4g.medium` (2 vCPU / 4 GB; **$27.45/month on-demand, $17.30/month 1-yr no-upfront
reserved**) runs the Dagster daemon + webserver + code-location server + Postgres-in-Docker +
**the Marimo dashboard**, behind Tailscale (no ALB). Every run — live schedules *and*
UI-launched backtests — is dispatched by `EcsRunLauncher` to an ephemeral Fargate task sized
to the job. Add ~$2.40 EBS, $7–9/month live Fargate, ~$1/backtest.

- **Pros:** Dagster properly (history, UI backfills, sensors, concurrency pools enforced
  centrally); backtests get big ephemeral compute; dashboard rides free; EC2 IAM instance
  roles (no static keys); the textbook Dagster-OSS-on-AWS deployment.
- **Cons:** one pet server (patching, disk, daemon liveness — mitigate with systemd restart
  policies + plan 05's "no fresh forecast" alarm); dagster.yaml/run-launcher config work;
  4 GB is comfortable but not roomy (watch Marimo's Delta scans).
- Cost trims: t4g.small (2 GB) is **free-trial (750 hrs/month) until 31 Dec 2026** and
  $13.72/$8.69 after, if everything squeezes into 2 GB — likely too tight with Marimo.

### Option C — one big box, everything on it (the OCF pattern) — ~$70–110/month

Dagster + Postgres + Marimo + *all compute* (inference peaks ~9 GB → 16 GB box): EC2
`t4g.xlarge` $109.79 on-demand / $69.28 reserved-1yr; Lightsail 16 GB $82.40 (4 vCPU, 40% CPU
baseline); Lightsail memory-optimised 16 GB $72.61 (2 vCPU). This is #206's "Level 3" and how
OCF runs everything else (with Airflow).

- **Pros:** simplest architecture possible — no Fargate/ECR-per-run/EventBridge/run-launcher;
  `docker compose` + `git pull`.
- **Cons:** priciest; backtests capped at 2–4 vCPU (slower than the workstation); Lightsail
  variants mean static AWS keys and burst-credit accounting; biggest pet.

### Option D — serverless control plane (no pets) — ~$50–55/month

Daemon + webserver as one always-on 0.5 vCPU / 2 GB ARM Fargate service ($19.60), RDS
`db.t4g.micro` Postgres ($13–16, approximate — the one unverified price), Marimo as its own
tiny Fargate service (approx $8.30), runs on ephemeral Fargate.

- **Pros:** full Dagster with zero servers to patch; IAM-native throughout.
- **Cons:** RDS is the tax (no local disk → no Postgres-in-Docker); webserver access needs an
  ALB (+~$22/month) or a Tailscale-sidecar hack; the most Terraform. Cleaner on paper than in
  practice.

### Option E — Dagster+ Solo, Hybrid — ~$45–55/month typical

$10/month base + $0.040/credit (1 credit = 1 materialisation or op execution; May 2026
pricing). Live cadence ≈ 395 credits ≈ $16/month; backtests $0.20–0.70 each (a heavy
50-experiment month adds $10–35). Plus a stateless Hybrid agent (~$9 tiny Fargate service) and
Marimo hosting (~$5–8), and the same per-run Fargate compute. Hybrid incurs no serverless
charge; sensor evaluations are free (an hourly freshness *op* would cost +$29/month — on
Dagster+ you use sensors, the better design anyway).

- **Pros:** managed UI/daemon/alerting/backfills with nothing of ours to keep alive.
- **Cons:** **Solo is 1 user** — collaborators can't log in (Starter is $100/month base);
  metering shapes design decisions; vendor dependency. The 1-user cap is likely disqualifying
  for an OCF collaboration.

### Comparison

| | A: Level 1 | B: small box + Fargate | C: one big box | D: serverless CP | E: Dagster+ Solo |
|---|---|---|---|---|---|
| $/month | 11–25 | **29–42** | 70–110 | ~50–55 (+ALB) | 45–55 |
| Run history + UI backfills | ✗ | ✓ | ✓ | ✓ | ✓ |
| Backtests in AWS | ✗ | ✓ big ephemeral compute | ✓ but 2–4 vCPU | ✓ | ✓ (credits) |
| Marimo dashboard | +$8 add-on | ✓ free on box | ✓ free on box | +$8 service | +$5–8 |
| Servers to patch | 0 | 1 | 1 | 0 | 0 |
| No static AWS keys | ✓ | ✓ | ✓ EC2 / ✗ Lightsail | ✓ | ✓ |
| Multi-user UI | n/a | ✓ (Tailscale) | ✓ | ✓ | ✗ (1 user) |

**Recommendation: Option B** — the only shape giving full Dagster *and* workstation-beating
backtest compute *and* a free dashboard home, for ~$15–25/month over Level 1. Option C is the
fallback if operational simplicity trumps backtest speed and ~$40/month. D pays an RDS+ALB
tax for purism; E's 1-user cap rules it out.

## Workstream 1 — S3-capable data paths (the biggest unknown; start first)

Every data location in `Settings` is a local `Path`
(`packages/contracts/src/contracts/settings.py:104-135`) and all IO assumes a local
filesystem. delta-rs/Polars read and write `s3://` URIs fine, but the plumbing has to allow it:

- Change the data-location fields (`nwp_data_path`, `power_forecasts_data_path`,
  `forecast_metrics_data_path`, `eligible_time_series_data_path`,
  `effective_capacity_data_path`, NGED power/metadata paths) from `Path` to `str` URIs
  (local paths remain valid `str`s; dev defaults unchanged). Keep genuinely-local paths
  (`model_cache_base_path`, `plots_data_path`, `cv_config_path`) as `Path`.
- Audit every consumer (`scan_delta`, `write_delta`, `DeltaTable(...)`, parquet read/write)
  for `Path`-only operations (`.exists()`, `/` joins, `mkdir`) and route them through
  URI-safe helpers; pass `storage_options` where delta-rs needs credentials (prefer IAM
  roles — no static keys — so `storage_options` stays empty on AWS and only dev needs any).
- Verify **predicate pushdown / partition pruning still works over S3** (the whole memory
  design depends on it): run the existing `.explain()`-based pruning test pattern against an
  S3 URI.
- Test tier: unit tests with local paths as today, plus one integration test against MinIO (or
  a dev bucket) exercising Delta round-trip + partition-pruned scan through the changed
  settings.

## Workstream 2 — the production job

A new Dagster job (e.g. `live_pipeline_job`) chaining, in order:

1. **Freshness op** — latest Dynamical init vs latest forecast's NWP init in S3; early-exit.
2. **Ingest** — materialise `power_time_series_and_metadata`, and the missing `ecmwf_ens`
   daily partitions **computed from the Delta table's max `init_time` vs Dynamical
   availability** (an explicit op input).
3. **Forecast** — `live_forecasts` (plan 02) for the current slot, `availability_mode="live"`.
4. *(after plan 05 lands)* — the `metrics(production_monitoring)` step; not a blocker.

Keep each op a thin shell over unit-tested pure helpers (freshness comparison, missing-
partition computation).

How the trigger differs by option: under **A**, the job is the hourly one-shot and the wrapper
computes the current time-window partition key from the clock (injected `now`, per repo
convention). Under **B/C/D**, the daemon runs it — the freshness decision moves up into a
schedule/sensor evaluation (return a `SkipReason` when nothing is new; evaluations are free),
partition keys come from Dagster natively, and missed slots are backfilled from the UI with
plan 02's `availability_mode="replay"`. The op-level freshness check stays regardless: it is
cheap idempotence insurance, and it keeps the job runnable one-shot for the local dress
rehearsal (#208).

## Workstream 3 — container (plan 03, unchanged)

Dockerfile with the champion model baked in (`PRODUCTION_MODEL_RUN_ID` build arg → model dir
copied into the image's cache path). MLflow is **not** contacted at runtime. Non-secret config
via task-definition env vars; the NGED S3 source credentials via Secrets Manager/SSM
references in the task definition. The same image serves as the one-shot entrypoint (option A)
or as the code-location server on the box plus the `EcsRunLauncher` run image (options B–D).
Build for **arm64** (ARM Fargate is 20% cheaper; the control-plane boxes are Graviton).

## Workstream 4 — AWS infrastructure

Common to all options:

- **ECR** repository; image pushed by the plan-01/03 build (tag = model run-id + git SHA).
- **S3**: one data bucket mirroring the local `data/` layout (`nwp_data/`,
  `power_forecasts/`, …). NGED-delivery bucket/prefix is a later step (v0.1 is "forecast
  running", not "delivery contract live").
- **IAM task role**: S3 read/write on the data bucket, ECR pull, CloudWatch Logs — no static
  AWS keys anywhere.
- **Fargate task definitions**: start 4 vCPU / 16 GB ARM for live runs (measured inference
  peak ~9 GB); a bigger (e.g. 8 vCPU / 32 GB) definition for backtests. Right-size after a
  week of CloudWatch metrics.
- **Alerting**: EventBridge rule on task stopped with non-zero exit → SNS → email.
- Codify as a small Terraform module (one file is fine); document the few one-time manual
  steps (SNS subscription confirm, Tailscale join) in a runbook page
  (`docs/architecture/production-deployment.md`, extending plan 03's runbook).

Option B adds (instead of option A's hourly EventBridge Scheduler → `RunTask` cron):

- **EC2 `t4g.medium`** (IAM instance role; EBS gp3 ~20 GB) running Docker Compose:
  `dagster-daemon`, `dagster-webserver`, the code-location server (same image as the runs),
  Postgres (run/event/schedule storage on the local volume, `pg_dump` to S3 nightly), and the
  Marimo dashboard. Tailscale for webserver/dashboard access — no public ingress.
- **`dagster.yaml`**: Postgres storage + `EcsRunLauncher` (cluster, task-definition family,
  network config); the existing `pool="ECMWF"` concurrency limit now enforced centrally.
- **Schedules/sensors**: the 6-hourly `live_forecasts` schedule and `ecmwf_ens`
  daily-partition sensor replace the hourly external cron (freshness via `SkipReason`).
- systemd unit (or Compose `restart: always`) + unattended-upgrades on the box; an EC2
  instance-status-check alarm; plan 05's staleness alarm covers "daemon silently dead".

## Related GitHub issues (sub-issues of the [#137 v0.1 epic](https://github.com/openclimatefix/nged-substation-forecast/issues/137))

| Issue | Where it lands in this plan |
|---|---|
| [#206 Deploy to AWS!](https://github.com/openclimatefix/nged-substation-forecast/issues/206) | This plan (options above supersede its cost analysis — post a correcting comment when decided) |
| [#121 Use obstore instead of pathlib](https://github.com/openclimatefix/nged-substation-forecast/issues/121) | Workstream 1 |
| [#50 Define all paths in Settings](https://github.com/openclimatefix/nged-substation-forecast/issues/50) | Workstream 1 |
| [#208 Run every 6 hours locally and backfill missing runs (as a test)](https://github.com/openclimatefix/nged-substation-forecast/issues/208) | Workstream 2 + the local dress rehearsal (also exercises plan 02's replay mode) |
| [#63 Send telemetry to OCF's Sentry.io](https://github.com/openclimatefix/nged-substation-forecast/issues/63) | Observability — CloudWatch + SNS first; Sentry as the follow-up |
| [#96 Write power forecasts in schema agreed with NGED](https://github.com/openclimatefix/nged-substation-forecast/issues/96) | The NGED-delivery projection — part of the epic, deferred from "forecast running" |
| [#161 More Dagster-UI metrics + validation for NWP ingestion](https://github.com/openclimatefix/nged-substation-forecast/issues/161) | Mostly plan 10 (clip logging); ingestion op metadata here |
| [#5 Backup procedure for data & models on Jack's workstation](https://github.com/openclimatefix/nged-substation-forecast/issues/5) | Largely superseded once S3 is the primary store; close or re-scope when this ships |
| [#209 Bump version number to v0.1](https://github.com/openclimatefix/nged-substation-forecast/issues/209) | The final ship step |

## Verification

1. **Local dress rehearsal**: run the container against the real S3 bucket from a laptop
   (`docker run` with the task role's permissions via assumed credentials); confirm a forecast
   lands in `power_forecasts` on S3 and the next run early-exits on freshness.
2. **On AWS**: (A) manual `RunTask` / (B) "Materialize" from the webserver — watch the run
   launch a Fargate task, follow CloudWatch Logs end-to-end, confirm forecast rows.
3. **Self-healing**: kill a run mid-flight; confirm the next scheduled tick (A) or a one-click
   UI backfill of the missed partition (B) redoes the work with no duplicate rows (Delta
   overwrite semantics).
4. **Option B extras**: reboot the box → daemon, webserver, Postgres, and Marimo all come back
   unattended; launch a backtest from the UI → it runs on the big Fargate task definition;
   Marimo dashboard shows the latest live forecasts over Tailscale.
5. **Alerting**: force a failure (bad env var), confirm the SNS email.
6. Leave it running for several days; check forecasts appear after each daily 00Z NWP and
   costs match the model above (Cost Explorer).
