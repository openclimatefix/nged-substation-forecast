# Configuration reference

What the live service's storage and credential settings mean: **where the bytes live** (a local
disk, or S3 — Simple Storage Service, AWS's object store) and **what credentials reach them**.
This page is the reference; the step-by-step journeys that *use* these settings live elsewhere
and deliberately aren't repeated here:

- [Getting started on your laptop](../getting-started.md) owns first-time repo setup — installing
  `uv`, `uv sync`, pre-commit hooks, creating `.env`, and materialising the first data and model.
- [Running the whole stack locally](local.md) owns the laptop bring-up (persistent
  `DAGSTER_HOME`, `dg dev`, the optional MinIO rehearsal).
- [Setting up the live service on AWS](aws.md) owns every AWS console step — buckets, IAM
  (Identity and Access Management — AWS's permissions system), the container image, and the
  control-plane box.
- [Operating the live service](operations.md) owns how to *drive* the assets once an
  environment is up (picking a champion, the 6-hourly schedule, backfilling).

## The configuration model

Everything here is driven by [`Settings`](../api/contracts/index.md) (in
`packages/contracts/src/contracts/settings.py`), populated from a `.env` file in the repo root and
from environment variables. Environment variables win over `.env`, which wins over the defaults;
every field name maps to an upper-case env var (`data_path_internal` → `DATA_PATH_INTERNAL`).

### Three storage roots

The single most important idea is that there are **three** roots, not one:

| Root | Env var | May be `s3://`? | Holds |
|---|---|---|---|
| Internal data tables | `DATA_PATH_INTERNAL` | **Yes** | NWP, power observations, forecast metrics — everything not on the NGED-facing delivery list |
| Delivery data tables | `DATA_PATH_DELIVERY` | **Yes** | The NGED-facing delivery tables (`power_forecast`, `effective_capacity`, …) |
| Local artifacts | `LOCAL_ARTIFACTS_PATH` | No — always local | Trained-model cache, the promoted production model, plot HTML |

> **On AWS, the two data-table roots point at two separate buckets** — `DATA_PATH_DELIVERY`
> hard-codes the five NGED-facing delivery tables (see the "derive from root" convention just
> below), so shipping a new delivery table can't silently leave it in the internal bucket. See
> [Setting up the live service on AWS: Step 1](aws.md#step-1-create-the-s3-buckets) for the
> concrete setup, and
> [Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it) for why.

They are split for an **architectural** reason, not merely because XGBoost and Altair happen to
write to local files (a library that can't write to S3 can always be bridged — write to a tempdir,
upload — as MLflow does internally). The real reason is that **nothing under `LOCAL_ARTIFACTS_PATH`
is part of the S3-backed data plane**. Each item belongs to the laptop/CV workflow or to the
container image, not to shared storage:

- **Trained-model cache** — a local-disk *cache* keyed by MLflow run id, filled by
  `BaseForecaster.load_from_mlflow` to avoid re-downloading artifacts. A cache is node-local by
  definition, and only the CV pipeline (which runs on a laptop) uses it; production inference never
  touches it.
- **Promoted production model** — distributed via the **container image**, not shared storage: the
  v0.1 deployment bakes the champion into the image at build time and loads it with a plain disk
  `load()` (see
  [Production Deployment — Design](../architecture/production-deployment.md),
  [#222](https://github.com/openclimatefix/nged-substation-forecast/issues/222)). This local
  directory is the build-time staging area that gets copied into the image; on the deployed task the
  model is read from the image's own filesystem.
- **Plot HTML** — a **local-dev convenience** only (materialise, open in a browser); see
  [Operating the live service: Inspecting a live forecast](operations.md#inspecting-a-live-forecast)
  for why it is not the way to view forecasts in a deployed service.

So the deployed AWS runtime reads its model from the image, reads data from S3, and writes forecasts
to S3 — it never uses `LOCAL_ARTIFACTS_PATH` as shared storage at all. All three roots default to
`<repo>/data`, so out of the box everything lives under one local directory and the distinction is
invisible; it only matters once `DATA_PATH_INTERNAL` and `DATA_PATH_DELIVERY` become `s3://` URIs.

### The "derive from root" convention

Each individual table has its own setting (`NWP_DATA_PATH`, `POWER_FORECASTS_DATA_PATH`, …), but you
almost never set them. Each defaults to an empty string, a sentinel meaning *derive me from my
root*: after validation, `NWP_DATA_PATH` becomes `<DATA_PATH_INTERNAL>/NWP`, `METADATA_PATH` becomes
`<DATA_PATH_INTERNAL>/NGED/metadata.parquet`, and so on. The NGED-facing delivery tables are the
exception: `POWER_FORECASTS_DATA_PATH` and `EFFECTIVE_CAPACITY_DATA_PATH` derive from
`DATA_PATH_DELIVERY` instead — hard-coded in `Settings._derive_unset_paths`, which is the full
default layout and the only place it lives.

Set `DATA_PATH_INTERNAL` alone and every non-delivery table moves together; set `DATA_PATH_DELIVERY`
alone and both delivery tables move together. Setting an individual table (e.g.
`NWP_DATA_PATH=s3://other-bucket/NWP`) overrides just that one — useful for keeping one big table on
a separate bucket, and nothing else needs to change.

The delivery-side derivation deliberately **fails closed**: a new delivery table added to the
schema but not wired into `Settings._derive_unset_paths` stays on `DATA_PATH_INTERNAL` (invisible
to NGED, a functional bug caught by `test_settings.py`'s guard test), rather than a forgotten
override on a new internal table accidentally exposing it in the delivery bucket. The other three
delivery tables (`power_forecast_warnings`, `asset_health_history`, `substation_switching`) don't
have `Settings` fields yet — none are implemented in code. When they land, they need to be wired
into `Settings._derive_unset_paths` to derive from `DATA_PATH_DELIVERY` alongside the two above
(the per-field env var override — e.g. `POWER_FORECASTS_DATA_PATH` — still works as an escape
valve for a one-off case, but the delivery/internal split is driven by root derivation, not
per-table overrides).

> **Design choice: NGED sees every CV/backtest fold, not just live production forecasts.** The
> `power_forecasts` table holds every CV/backtest experiment alongside live production forecasts,
> distinguished by `fold_id` (`"live"` for production — see
> [Operating the live service](operations.md)). Pointing the whole table at the
> delivery bucket is deliberate, not an accidental side effect of the bucket split: it lets NGED
> see how each of OCF's model versions actually behaves, not just whichever one is currently
> promoted. NGED filters on `fold_id="live"` when it only wants the current production forecast.

### The `.env` file and NGED source credentials

Create a `.env` file in the repo root by copying the committed template — `cp .env.example .env` —
and filling in the values. The three **NGED source-bucket** credentials are always required, in
every environment — they authenticate reads of NGED's telemetry bucket, which is a *different*
account and bucket from our own managed data tables:

```dotenv
NGED_S3_BUCKET_URL=<nged source bucket url>
NGED_S3_BUCKET_ACCESS_KEY=<key>
NGED_S3_BUCKET_SECRET=<secret>
```

`.env` is git-ignored — never commit real credentials. Everything else on this page is optional and
layers on top of these three.

The optional `SENTRY_*` settings (`SENTRY_DSN`, `SENTRY_ENVIRONMENT`, `SENTRY_MONITOR_FORECASTS`,
`SENTRY_TRACES_SAMPLE_RATE`) enable error telemetry and the missed-check-in alarm; an empty
`SENTRY_DSN` (the default) disables Sentry entirely. Their setup — laptop testing and production —
has its own page: [Setting up Sentry telemetry](sentry.md).

### Credentials for our own S3 data (`DATA_STORE_*`)

The four `DATA_STORE_*` fields (`ACCESS_KEY_ID`, `SECRET_ACCESS_KEY`, `REGION`, `ENDPOINT_URL`)
map onto the shared `aws_*` object_store option keys that delta-rs, Polars, and obstore all
understand, so this one dict feeds every read and write — to both buckets alike, since bucket
choice is entirely a matter of which URI each path setting resolves to. Which of them to set
depends on where the code runs:

- **Compute on AWS (an EC2 — Elastic Compute Cloud — virtual machine, or a Fargate container
  task)** — set **none of them**: `object_store`
  auto-discovers the attached IAM role's temporary credentials and region at runtime.
- **A laptop reaching real S3** — set key + secret + region from an IAM user (see
  [Setting up the live service on AWS: Step 2](aws.md#step-2-grant-data-access-with-iam) for
  which user), but **not** `ENDPOINT_URL`.
- **A local MinIO rehearsal** — set all four; `ENDPOINT_URL` is only for non-AWS/S3-compatible
  endpoints (and it deliberately allows plain HTTP, which dev endpoints rarely encrypt). See
  [Running the whole stack locally](local.md#optional-rehearse-s3-locally-with-minio).

### At-a-glance: which settings for which environment

| Environment | `DATA_PATH_INTERNAL` | `DATA_PATH_DELIVERY` | `DATA_STORE_*` |
|---|---|---|---|
| Local (default) | unset → `<repo>/data` | unset → `<repo>/data` | none |
| Local MinIO rehearsal | `s3://…` | same `s3://…` (single bucket) | all four, incl. `ENDPOINT_URL` |
| Laptop → real S3 (pipeline) — **not set up yet; may add later for one-off writes** ([aws.md: Step 2](aws.md#step-2-grant-data-access-with-iam)) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | key + secret + region, read/write IAM user (no endpoint) |
| Laptop → real S3 (dashboard only) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | key + secret + region, read-only IAM user (no endpoint) |
| Compute on AWS (IAM role) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | none (auto-discovered) |

*How* the values are set differs by compute, but `Settings` reads them identically — an
environment variable and a `.env` line are interchangeable, and an environment variable wins if
both are set. A Fargate task has no repo checkout and no `.env` file, so its values are plain
environment variables on the container in the ECS (Elastic Container Service) task definition
(bucket URIs are safe in clear
text; the NGED source credentials are injected from Parameter Store instead — see
[Setting up the live service on AWS: Steps 8–9](aws.md#step-8-store-secrets-in-parameter-store)).
On an EC2 box or a laptop, use a `.env` file.
