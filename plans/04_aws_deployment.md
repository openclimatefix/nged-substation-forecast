# Deploy v0.1 to AWS (issues #137 / #206, Level 1)

## Context

**Top priority: get *any* forecast running on AWS.** Forecast quality does not matter yet — the
science plans (02, 03, 12, …) wait until v0.1 is live. This plan is the Level 1 ("super
simple") design from issue [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206), which Peter endorsed ("go simple"), made concrete and merged
with the review's caveats. Recommended sequencing across the plan set is in
`00_review_findings.md`: **01 (CI) → 02 (`live_forecasts`) → 03 (container + baked model) →
this plan (04) → 05 (monitoring) → science.**

## Architecture recap (Level 1 — nothing always-on)

- **Hourly EventBridge Scheduler** → launches a one-shot **ECS Fargate task** → the container
  runs a Dagster job in-process (`dagster job execute`) against a throwaway local
  `DAGSTER_HOME`, materialises to S3, and exits.
- **Freshness check is the first op**: compare the latest NWP init available from
  Dynamical.org against the NWP init behind the newest forecast in the `power_forecasts` Delta
  on S3. Nothing new → exit ~0. The forecasts in S3 *are* the record of what's been done.
- **Failure recovery is the cron**: a failed run writes no fresh forecast, so the next hourly
  tick sees stale outputs and re-runs. No markers, no database, no daemon.
- **Observability**: CloudWatch Logs + an EventBridge rule on ECS task-state change (non-zero
  exit) → SNS email. Sentry can come later.
- **Dashboard**: local Marimo reading S3 — out of scope here.
- **Cost anchor**: ~$15–25/month all-in (Fargate only).

Review caveats already folded in: Delta commits provide the atomicity the freshness logic
needs (make the forecast Delta write the run's final write — no temp-key dance);
"which partitions need materialising" must be derived from **Delta contents vs Dynamical
availability**, never Dagster's materialisation records (they evaporate with the throwaway
SQLite); no pre-flight Lambda — one execution path, accept the ~10–20 s Dagster boot on empty
wake-ups.

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
  URI-safe helpers; pass `storage_options` where delta-rs needs credentials (prefer the IAM
  task role — no static keys — so `storage_options` stays empty on AWS and only dev needs any).
- Verify **predicate pushdown / partition pruning still works over S3** (the whole memory
  design depends on it): run the existing `.explain()`-based pruning test pattern against an
  S3 URI.
- Test tier: unit tests with local paths as today, plus one integration test against MinIO (or
  a dev bucket) exercising Delta round-trip + partition-pruned scan through the changed
  settings.

## Workstream 2 — the one-shot production job

A new Dagster job (e.g. `live_pipeline_job`) chaining, in order:

1. **Freshness op** — latest Dynamical init vs latest forecast's NWP init in S3; early-exit.
2. **Ingest** — materialise `power_time_series_and_metadata`, and the missing `ecmwf_ens`
   daily partitions **computed from the Delta table's max `init_time` vs Dynamical
   availability** (an explicit op input, not Dagster partition status).
3. **Forecast** — `live_forecasts` (plan 02) for the current slot, `availability_mode="live"`.
   The job wrapper computes the current time-window partition key from the clock (injected
   `now`, per repo convention) since one-shot execution must pass partitions explicitly.
4. *(after plan 05 lands)* — append a `metrics(production_monitoring)` step; not a blocker.

Keep each op a thin shell over unit-tested pure helpers (freshness comparison, missing-
partition computation).

## Workstream 3 — container (plan 03, unchanged)

Dockerfile with the champion model baked in (`PRODUCTION_MODEL_RUN_ID` build arg → model dir
copied into the image's cache path), entrypoint `dagster job execute` on the job above. MLflow
is **not** contacted at runtime. Non-secret config via task-definition env vars; the NGED S3
source credentials via Secrets Manager/SSM references in the task definition.

## Workstream 4 — AWS infrastructure

- **ECR** repository; image pushed by the plan-05 build (tag = model run-id + git SHA).
- **ECS**: cluster + Fargate task definition. Start 8 vCPU / 32 GB (the issue-#206 sizing);
  measured inference peak is ~9 GB, so right-size down (e.g. 4 vCPU / 16 GB) after a week of
  CloudWatch metrics. Task **IAM role**: S3 read/write on the data bucket, ECR pull,
  CloudWatch Logs — no static AWS keys anywhere.
- **EventBridge Scheduler**: hourly cron with an ECS `RunTask` target (native — no Lambda glue).
- **S3**: one data bucket mirroring the local `data/` layout (`nwp_data/`,
  `power_forecasts/`, …). NGED-delivery bucket/prefix is a later step (v0.1 is "forecast
  running", not "delivery contract live").
- **Alerting**: EventBridge rule on task stopped with non-zero exit → SNS → email.
- Codify as a small Terraform module (one file is fine) so the environment is reproducible;
  document the few one-time manual steps (SNS subscription confirm) in a runbook page
  (`docs/architecture/production-deployment.md`, extending plan 03's runbook).


## Related GitHub issues (sub-issues of the [#137 v0.1 epic](https://github.com/openclimatefix/nged-substation-forecast/issues/137))

| Issue | Where it lands in this plan |
|---|---|
| [#206 Deploy to AWS!](https://github.com/openclimatefix/nged-substation-forecast/issues/206) | This plan (the Level 1 design is in its last comment) |
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
2. **On AWS**: manual `RunTask`, watch CloudWatch Logs end-to-end, confirm forecast rows.
3. **Self-healing**: stop a task mid-run; confirm the next hourly tick redoes the work and no
   duplicate rows exist (Delta overwrite semantics).
4. **Alerting**: force a failure (bad env var), confirm the SNS email.
5. Leave it running for several days; check forecasts appear after each daily 00Z NWP and the
   ~23 empty wake-ups/day cost pennies (Cost Explorer).
