# Live service (v0.1 AWS deployment)

> **Status: 🚧 Planned — the current focus.** Epic:
> [#137](https://github.com/openclimatefix/nged-substation-forecast/issues/137) (deploy the
> naive forecast on AWS); see also
> [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206) and
> [#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208). Build order:
> the [inference asset](#the-live_forecasts-asset) → the
> [champion-model container](#production-model-artifacts) → the
> [AWS infrastructure](#aws-architecture) → [monitoring](#production-monitoring). Forecast
> *quality* deliberately does not matter yet — the science work
> ([baselines](metrics-and-leaderboard.md#baseline-forecasters),
> [XGBoost improvements](xgboost-improvements.md)) waits until v0.1 is live.

*History: the inference-asset and monitoring designs were absorbed from the Dagster ML-assets
plan (phases 0–6.7 complete, PRs #182–#214); its final cleanup phase lives in
[Engineering health](engineering-health.md#scientific-rigor-tests-and-cleanup).*

## Requirements

**MVP (v0.1 — get a live forecast running on AWS):**

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
- AWS infrastructure with **no static AWS keys** (IAM roles throughout), basic alerting on task
  failure (SNS → email), and cost-conscious operation (~$30–40/month target).

**Post-MVP (explicitly deferred):**

- Forecast quality/science work —
  [baselines](metrics-and-leaderboard.md#baseline-forecasters),
  [XGBoost improvements](xgboost-improvements.md).
- The NGED-delivery schema/contract
  ([#96](https://github.com/openclimatefix/nged-substation-forecast/issues/96)) — v0.1 is
  "forecast running", not "delivery contract live".
- Telemetry to Sentry.io
  ([#63](https://github.com/openclimatefix/nged-substation-forecast/issues/63)) — CloudWatch +
  SNS cover alerting first.
- An **MLflow tracking server** and a separate **development dashboard**, both hosted on the
  always-on control-plane box once it exists — see the [note below](#aws-architecture).
- [Production monitoring](#production-monitoring): score live forecasts over trailing 24h/7d
  windows, logged to a dedicated MLflow experiment, with a manual, auditable way to retire
  stale experiment partitions.

## The `live_forecasts` asset

Everything up to the CV leaderboard loop is built; the production inference path is not. This
asset is what the deployed container runs every 6 hours: load the production model, forecast
the latest NWP, write to `power_forecasts` with `fold_id="live"`.

```text
partitions_def: TimeWindowPartitionsDefinition(cron="0 0,6,12,18 * * *", ...)
deps: [ecmwf_ens, power_time_series_and_metadata]
```

- **Model loading:** `ForecasterClass.load(path)` — a plain disk load of the model directory
  baked into the container image at build time (see
  [Production model artifacts](#production-model-artifacts)); no MLflow client, run ID, or
  cache lookup involved. The concrete class is reconstructed from the `model_class` field in
  the saved model's `meta.json`.
- **Population:** forecast **only** the `trained_time_series_ids` recorded in the production
  model's `meta.json` — never a series the model has not seen (the train==predict invariant).
- **Inference mode:** **single-run** feature engineering
  (`engineer_features(power_fcst_init_time=t0, …)`) across **all 51 NWP ensemble members**,
  where `t0` is the partition's scheduled time. Deliberately *not* bulk mode: production
  forecasts one explicit `power_fcst_init_time`, not one-per-NWP-run.
- **NWP availability semantics** via a `RunConfig` field
  `availability_mode: Literal["live", "replay"]` (default `"live"`):
  - `"live"` — the scheduled path. `t0 = now`; join the **freshest NWP run actually present**
    in Delta with `nwp_init_time <= t0`. **No** modelled publication delay: reality already
    constrains the table to genuinely published runs, so if the provider speeds up we
    automatically use fresher data.
  - `"replay"` — re-running a *past* slot (e.g. yesterday's failed run, today).
    `t0 = the historical partition time`; join the freshest run with
    `nwp_init_time <= t0 − nwp_publication_delay_hours`. The delay reconstructs what was
    actually *available* at the historical `t0` — without it we would leak NWP runs that only
    landed afterwards.
  - The scheduled sensor/schedule always uses `"live"` for the current partition; manual
    backfills of past partitions use `"replay"`. The explicit flag is the source of truth (an
    automatic live-iff-recent rule is a possible later convenience).
- Load the needed NWP via `TimeWindowPartitionMapping` against the daily-partitioned
  `ecmwf_ens`.
- **Write:** `forecaster.predict(..., fold_id="live")`, then an **idempotent overwrite** of
  this run's `power_fcst_init_time` rows in `power_forecasts` (`replaceWhere`), so re-running a
  6-hourly partition (or a replay) never duplicates rows.
- **Logs nothing to MLflow.** A 6-hourly forecast run is not an experiment; live performance
  is tracked by [production monitoring](#production-monitoring), never by this asset.

## Production model artifacts

The deployment below runs forecasts as ephemeral Fargate tasks under every architecture
option. An ephemeral container has no persistent disk and, under some architecture options, no
reachable tracking server — so without a decision here, production inference has no way to get
a model. This must be decided before writing the Dockerfile.

**Decision: bake the champion model into the image at build time, loaded via a plain
`save`/`load` — no MLflow, run ID, or cache involved at runtime.** The model directory is
produced once, out of band (a researcher picks the champion fold from the MLflow leaderboard
and downloads its artifacts to local disk), then `COPY`'d into the image at build time.
Promotion becomes rebuild + redeploy, which is auditable (image tags) and keeps MLflow
completely out of the production runtime under every architecture option. Rejected
alternative: MLflow artifact root on S3 fetched at container startup — more runtime moving
parts, needs tracking-store access from prod, and slower cold starts.

This is deliberately simpler than reusing `BaseForecaster.load_from_mlflow`'s cache (the
mechanism the CV pipeline already uses — see
[ML orchestration](../architecture/ml-orchestration.md#model-artifacts-mlflow-artifact-store-immutable-local-cache)):
v0.1 has no MLflow dependency to cache against in the first place. **Future work:** once
production wants to pick up a new champion without a rebuild + redeploy (e.g. after the
[XGBoost quick wins](xgboost-improvements.md) start landing regularly), switch to fetching the
champion model from MLflow dynamically — at that point `load_from_mlflow`'s local-disk cache
becomes the production-resilience mechanism again (serving from disk on a cache hit so the live
service survives an MLflow outage), exactly as it does for CV today.

## AWS architecture

An earlier version of this plan committed to the Level 1 ("nothing always-on") design from
issue [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206). A
2026-07-02 pressure-test of that decision found the #206 cost analysis substantially wrong:
the always-on control plane it priced at ~$95–140/month actually costs ~$14–27/month (it
priced a 16 GB box big enough to run the *compute*, not a small control-plane box), and its
RDS prerequisite dissolves on a single machine (Postgres-in-Docker, or SQLite on a real local
filesystem). Two requirements also firmed up that Level 1 does not serve:

1. **Use Dagster "properly"** — persistent run history, one-click UI backfills of missed
   partitions, and the ability to launch backtests on AWS whenever the model improves.
2. **An always-on dev dashboard** (a simple Marimo web app showing the latest forecasts) —
   so *something* must be always-on regardless.

**Current decision status: leaning Option B (small EC2 control-plane box + `EcsRunLauncher`),
final call pending.** The five options researched are recorded below so the decision has a
durable record. The implementation workstreams are identical under every option except the
infrastructure one.

### Workload model

Live cadence: `ecmwf_ens` 1/day (daily 00Z partition), `power_time_series_and_metadata` 4/day,
`live_forecasts` 4/day (6-hourly partitions), the monitoring `metrics(production_monitoring)`
step ~4/day → **~13 materialisations/day ≈ 395/month**. A backtest experiment today is ~4–6
materialisations (`eligible_time_series` + `trained_cv_model` + `cv_power_forecasts` per
fold, plus `metrics`; single leaderboard fold), rising to ~15–20 under the future
multi-yearly-fold epoch.

Fargate compute (eu-west-2, prices from the AWS price-list API, 2026-07-02): x86
$0.04656/vCPU-hr + $0.00511/GB-hr; **ARM 20% cheaper** ($0.03725 + $0.00409) and
polars/XGBoost run fine on arm64. A right-sized 4 vCPU / 16 GB ARM task (measured inference
peak ~9 GB) is $0.21/hr → 4 × 15-min live runs/day ≈ **$7/month**; hourly empty polling
wake-ups add ~$2.50/month; a 2-hour 8 vCPU / 32 GB ARM backtest ≈ **$0.86/run**. (The old
$15–25/month anchor assumed 8 vCPU / 32 GB x86.)

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
  policies + the monitoring plan's "no fresh forecast" alarm); dagster.yaml/run-launcher
  config work; 4 GB is comfortable but not roomy (watch Marimo's Delta scans).
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

**Future work (not MVP):** once an always-on control-plane box exists (Option B or later), it's
also a natural home for an **MLflow tracking server** (network-reachable, persistent — replacing
the local file-store) and a **"development dashboard"** (a Marimo app for researchers, distinct
from the production dev dashboard already scoped above). Neither is needed to ship v0.1; revisit
once the box exists.

## Production monitoring

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

Note this sensor needs a running Dagster daemon — [Option B](#option-b-small-ec2-control-plane-box-ecsrunlauncher-2942month-recommended)
(the direction we're leaning) provides one. If the deployment instead ships Option A (nothing
always-on), skip the sensor and run the monitoring step as the final op of the one-shot
production job (the production-job workstream below already reserves that slot).

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

### `live_forecasts` — implementation notes, tests, verification

Implementation notes:

- Keep the asset a thin shell over pure helpers (repo convention): the freshest-NWP-run
  selection given `(t0, availability_mode, delay)` belongs in `ml_core._cv_helpers` (or a
  sibling), unit-tested with an injected clock — `now` is passed in, never read inside the
  asset.
- Reuse the identity-stamping tail shared with `cv_power_forecasts` (predict → stamp →
  validate) rather than duplicating it; extract a shared helper if one doesn't already exist.
- File placement: `defs/cv_assets.py` is already 898 lines — put this in a new
  `defs/production_assets.py` (the split formalised in
  [Engineering health](engineering-health.md#scientific-rigor-tests-and-cleanup)).

Tests:

- `live` vs `replay` select **different NWP runs** for the same `t0` when a fresher run exists
  (replay applies `nwp_publication_delay_hours`; live does not).
- Only `trained_time_series_ids` are forecast.
- All 51 ensemble members present in the output.
- Idempotency: materialise the same partition twice → row count unchanged.
- Injected-clock unit tests for the run-selection helper.

Verification: point `production_model_path` at a smoke-test model's saved directory, materialise
`live_forecasts` for the current partition, and confirm `fold_id="live"` rows in
`power_forecasts` — then plot them via `plot_power_forecast_job` (`fold_id: "live"`). Running
the job 6-hourly on a workstation for a few days, then back-filling any missed slots with
`replay`, is exactly the test described in
[#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208).

### Container build

Dockerfile (repo root):

- Multi-stage `uv` build (standard pattern: `ghcr.io/astral-sh/uv` image, `uv sync --frozen
  --no-dev`, copy `.venv` + source into a slim `python:3.14` runtime stage).
- Build args: `MODEL_RUN_ID` (the champion fold run ID, stamped only as an OCI label for
  traceability — the build never contacts MLflow with it), `GIT_SHA` (stamped as an OCI label
  and env var; complements
  [reproducibility stamping](engineering-health.md#reproducibility-stamp-git-sha-and-delta-table-versions-on-mlflow-runs)).
- Model injection: keep the image build hermetic — the build **copies** a plain model
  directory from the build context (`data/production_model/`, populated beforehand by a small
  script, e.g. `scripts/fetch_model.py`, that downloads the champion run's artifacts straight
  from MLflow — a one-off, researcher-run step, not the `load_from_mlflow` cache wrapper).
  `COPY` it to a fixed in-image path that the entrypoint passes straight to `ForecasterClass.load`.
  Downloading from inside `docker build` is rejected (needs tracking credentials in the build).
- Entrypoint: `dagster job execute` one-shot (the exact job comes with the `live_forecasts`
  asset; until that exists, the entrypoint can materialise the data-ingestion assets so the
  image is testable now). Under Option B the same image also serves as the code-location
  server on the control-plane box and as the `EcsRunLauncher` run image — the launcher
  overrides the command, so the one-shot entrypoint stays the default either way. Build for
  **arm64** (ARM Fargate is 20% cheaper and the candidate control-plane boxes are Graviton).

Settings: add a simple fixed path (e.g. `Settings.production_model_path`) for where the
in-image model directory lives — no MLflow-related setting is needed for v0.1.

Promotion runbook — short page `docs/architecture/production-deployment.md` (permanent docs,
written when this ships):

1. Pick the champion fold run ID from the MLflow leaderboard.
2. `uv run python scripts/fetch_model.py --run-id <id>` → populates `data/production_model/`.
3. `docker build --build-arg MODEL_RUN_ID=<id> --build-arg GIT_SHA=$(git rev-parse HEAD)
   -t nged-forecast:<id-short> .` and push to ECR.
4. Point the ECS task definition at the new tag.

Also record the two issue-#206 subtleties the review surfaced, so they're not lost when the
Fargate work starts: (a) whenever the job executes one-shot without persistent Dagster state
(Option A, or the local dress rehearsal), "which `ecmwf_ens` partitions need materialising"
must be derived from Delta contents vs Dynamical availability, not Dagster's materialisation
records; (b) Delta commits already give the atomic "outputs are the freshness record" property
— just ensure the forecast Delta write is the run's final write.

Container verification:

1. `docker build` with a smoke-test-fold run ID succeeds.
2. `docker run` **with no network access to any MLflow store** loads the model straight from
   disk and executes the entrypoint — this is the critical test.
3. Image labels show the run ID and git SHA (`docker inspect`).

### Deployment workstream 1 — S3-capable data paths (the biggest unknown; start first)

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

### Deployment workstream 2 — the production job

A new Dagster job (e.g. `live_pipeline_job`) chaining, in order:

1. **Freshness op** — latest Dynamical init vs latest forecast's NWP init in S3; early-exit.
2. **Ingest** — materialise `power_time_series_and_metadata`, and the missing `ecmwf_ens`
   daily partitions **computed from the Delta table's max `init_time` vs Dynamical
   availability** (an explicit op input).
3. **Forecast** — `live_forecasts` for the current slot, `availability_mode="live"`.
4. *(after monitoring lands)* — the `metrics(production_monitoring)` step; not a blocker.

Keep each op a thin shell over unit-tested pure helpers (freshness comparison, missing-
partition computation).

How the trigger differs by option: under **A**, the job is the hourly one-shot and the wrapper
computes the current time-window partition key from the clock (injected `now`, per repo
convention). Under **B/C/D**, the daemon runs it — the freshness decision moves up into a
schedule/sensor evaluation (return a `SkipReason` when nothing is new; evaluations are free),
partition keys come from Dagster natively, and missed slots are backfilled from the UI with
`availability_mode="replay"`. The op-level freshness check stays regardless: it is cheap
idempotence insurance, and it keeps the job runnable one-shot for the local dress rehearsal
([#208](https://github.com/openclimatefix/nged-substation-forecast/issues/208)).

### Deployment workstream 3 — AWS infrastructure

Common to all options:

- **ECR** repository; image pushed by the CI/container build (tag = model run-id + git SHA).
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
  (`docs/architecture/production-deployment.md`, extending the promotion runbook above).

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
  instance-status-check alarm; the monitoring staleness alarm covers "daemon silently dead".

### Related GitHub issues (sub-issues of the [#137 v0.1 epic](https://github.com/openclimatefix/nged-substation-forecast/issues/137))

| Issue | Where it lands in this plan |
|---|---|
| [#206 Deploy to AWS!](https://github.com/openclimatefix/nged-substation-forecast/issues/206) | This page (the options above supersede its cost analysis — post a correcting comment when decided) |
| [#121 Use obstore instead of pathlib](https://github.com/openclimatefix/nged-substation-forecast/issues/121) | Workstream 1 |
| [#50 Define all paths in Settings](https://github.com/openclimatefix/nged-substation-forecast/issues/50) | Workstream 1 |
| [#208 Run every 6 hours locally and backfill missing runs (as a test)](https://github.com/openclimatefix/nged-substation-forecast/issues/208) | Workstream 2 + the local dress rehearsal (also exercises the replay mode) |
| [#63 Send telemetry to OCF's Sentry.io](https://github.com/openclimatefix/nged-substation-forecast/issues/63) | Observability — CloudWatch + SNS first; Sentry as the follow-up |
| [#96 Write power forecasts in schema agreed with NGED](https://github.com/openclimatefix/nged-substation-forecast/issues/96) | The NGED-delivery projection — part of the epic, deferred from "forecast running" |
| [#161 More Dagster-UI metrics + validation for NWP ingestion](https://github.com/openclimatefix/nged-substation-forecast/issues/161) | Mostly [NWP clip logging](engineering-health.md#nwp-quantisation-count-and-surface-clipped-values); ingestion op metadata here |
| [#5 Backup procedure for data & models on Jack's workstation](https://github.com/openclimatefix/nged-substation-forecast/issues/5) | Largely superseded once S3 is the primary store; close or re-scope when this ships |
| [#209 Bump version number to v0.1](https://github.com/openclimatefix/nged-substation-forecast/issues/209) | The final ship step |

### Deployment verification

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
