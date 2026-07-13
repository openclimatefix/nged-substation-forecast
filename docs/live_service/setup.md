# Environment & storage setup

Where the live service keeps its data and artifacts, and how to configure credentials for each
environment. This is the setup you do **once per environment**, before running anything.

It sits alongside two neighbours and deliberately does not repeat them:

- [The repository README](https://github.com/openclimatefix/nged-substation-forecast#setup) owns
  first-time repo setup — installing `uv`, `uv sync`, pre-commit hooks.
- [Running live forecasts end-to-end](dagster-workflow.md) owns how to *drive* the assets once
  the environment is configured (picking a champion, the 6-hourly schedule, backfilling), and the
  persistent-`DAGSTER_HOME` prerequisite.

This page owns the layer in between: **where the bytes live** (a local disk, or S3) and **what
credentials reach them**.

## The configuration model

Everything here is driven by [`Settings`](../api/contracts/index.md) (in
`packages/contracts/src/contracts/settings.py`), populated from a `.env` file in the repo root and
from environment variables. Environment variables win over `.env`, which wins over the defaults;
every field name maps to an upper-case env var (`data_path` → `DATA_PATH`).

### Two storage roots

The single most important idea is that there are **two** roots, not one:

| Root | Env var | May be `s3://`? | Holds |
|---|---|---|---|
| Data tables | `DATA_PATH` | **Yes** | NWP, power observations, forecasts, metrics, metadata, H3 weights |
| Local artifacts | `LOCAL_ARTIFACTS_PATH` | No — always local | Trained-model cache, the promoted production model, plot HTML |

> **On AWS, "data tables" itself splits across two buckets** — the five NGED-facing delivery
> tables live separately from everything else, via the per-table path overrides described in the
> "derive from root" convention just below. See [Running on AWS: Step
> 3](#step-3-point-settings-at-the-buckets) for the concrete setup, and [Forecast Delivery:
> Securing it](../architecture/forecast-delivery.md#securing-it) for why.

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
  [roadmap: Production model artifacts](https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#production-model-artifacts),
  [#222](https://github.com/openclimatefix/nged-substation-forecast/issues/222)). This local
  directory is the build-time staging area that gets copied into the image; on the deployed task the
  model is read from the image's own filesystem.
- **Plot HTML** — a **local-dev convenience** only (materialise, open in a browser); see the note
  under [Running on AWS](#viewing-forecasts-on-aws) for why it is not the way to view forecasts in a
  deployed service.

So the deployed AWS runtime reads its model from the image, reads data from S3, and writes forecasts
to S3 — it never uses `LOCAL_ARTIFACTS_PATH` as shared storage at all. Both roots default to
`<repo>/data`, so out of the box everything lives under one local directory and the distinction is
invisible; it only matters once `DATA_PATH` becomes an `s3://` URI.

### The "derive from root" convention

Each individual table has its own setting (`NWP_DATA_PATH`, `POWER_FORECASTS_DATA_PATH`, …), but you
almost never set them. Each defaults to an empty string, a sentinel meaning *derive me from my
root*: after validation, `NWP_DATA_PATH` becomes `<DATA_PATH>/NWP`, `METADATA_PATH` becomes
`<DATA_PATH>/NGED/metadata.parquet`, and so on. The full default layout lives in
`Settings._derive_unset_paths` and nowhere else.

Set `DATA_PATH` alone and all nine data tables move together. Setting an individual table (e.g.
`NWP_DATA_PATH=s3://other-bucket/NWP`) overrides just that one — useful for keeping one big table on
a separate bucket, and nothing else needs to change. [Running on AWS: Step
3](#step-3-point-settings-at-the-buckets) is exactly this pattern applied on purpose: `DATA_PATH`
defaults to OCF's internal bucket, and the delivery tables are individually overridden to a
second, NGED-facing bucket.

### The `.env` file and NGED source credentials

Create a `.env` file in the repo root. The three **NGED source-bucket** credentials are always
required, in every environment — they authenticate reads of NGED's telemetry bucket, which is a
*different* account and bucket from our own managed data tables:

```dotenv
NGED_S3_BUCKET_URL=<nged source bucket url>
NGED_S3_BUCKET_ACCESS_KEY=<key>
NGED_S3_BUCKET_SECRET=<secret>
```

`.env` is git-ignored — never commit real credentials. Everything else on this page is optional and
layers on top of these three.

## Running locally

The default. With only the three `NGED_S3_BUCKET_*` values in `.env` and nothing else set, both roots
resolve to `<repo>/data`, `Settings.storage_options` is empty, and every table is a plain file on
disk. Follow the [README setup](https://github.com/openclimatefix/nged-substation-forecast#setup),
then [run the workflow](dagster-workflow.md). Nothing in this section is needed for that path.

### Optional: rehearse S3 locally with MinIO

To exercise the exact S3 read/write code paths without touching AWS, point `DATA_PATH` at a local
[MinIO](https://min.io/) (or any S3-compatible) endpoint and supply all four `DATA_STORE_*`
settings:

```dotenv
DATA_PATH=s3://my-bucket/data
DATA_STORE_ENDPOINT_URL=http://localhost:9000
DATA_STORE_ACCESS_KEY_ID=minioadmin
DATA_STORE_SECRET_ACCESS_KEY=minioadmin
DATA_STORE_REGION=us-east-1
```

`LOCAL_ARTIFACTS_PATH` is left unset, so models and plots stay under `<repo>/data` on your disk while
the data tables round-trip through MinIO. Setting `DATA_STORE_ENDPOINT_URL` also allows plain HTTP,
since dev endpoints rarely have TLS. (This is the same machinery the S3 integration test drives
against an in-process `moto` server.)

## Running on AWS (manual point-and-click)

> **Scope: data tables on S3, not the full unattended deployment.** This section covers pointing the
> **data tables** at S3 — which is all the ephemeral-compute deployment needs from the storage layer,
> and is enough to run the stack from your laptop against an S3 `DATA_PATH` today. Building the
> production container itself is covered by
> [Deploying a new production image](deployment.md);
> running it as an unattended, scheduled **Fargate task** (ECR, EventBridge scheduling) is a
> separate, not-yet-built roadmap item — see the
> [AWS architecture](https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#aws-architecture)
> section for that plan.

### Step 1 — Create the S3 buckets

**Two** buckets, not one — split so the five tables that form NGED's stable delivery contract are
physically separate from OCF's own working data, which may change shape at any time with no
notice. See [Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it)
for why this split exists, and [Delivery tables](https://openclimatefix.github.io/nged-substation-forecast/roadmap/delivery-tables/)
for exactly which five tables count as "delivery."

In the AWS console → **S3** → **Create bucket**, twice:

1. **Region** `eu-west-2` (London) for both — keep every resource in one region so S3 ↔ compute
   transfer stays free.
2. **Names**: `nged-forecast-delivery` (the 5 NGED-facing tables) and `nged-forecast-internal`
   (NWP, raw power telemetry, forecast metrics, and everything else OCF's pipeline needs but
   hasn't promised to keep stable). Bucket names are globally unique; pick your own if these are
   taken.
3. Leave **Block all public access** **on**, and default **SSE-S3** encryption on, for both.
   Nothing here is public.
4. **Versioning** is optional and not required by the app, for either bucket.

No DynamoDB lock table is needed for either bucket. The `deltalake` version we use commits via
S3's native conditional-put, so concurrent-safe Delta writes work on plain S3 with no lock table
and no `AWS_S3_ALLOW_UNSAFE_RENAME` flag.

### Step 2 — Grant access with IAM

OCF's own pipeline needs to read and write **both** buckets (internal tables during ingestion and
training; delivery tables at forecast time), so one IAM policy covers both:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::nged-forecast-delivery",
        "arn:aws:s3:::nged-forecast-delivery/*",
        "arn:aws:s3:::nged-forecast-internal",
        "arn:aws:s3:::nged-forecast-internal/*"
      ]
    }
  ]
}
```

Attach it to **whichever identity runs the code**:

- **Compute running on AWS** (an EC2 box or Fargate task) → attach the policy to that resource's
  **IAM role**. Nothing else is needed: delta-rs' object_store auto-discovers the role's temporary
  credentials and region at runtime, so you leave all four `DATA_STORE_*` settings **empty**.
- **Your laptop, against the real buckets** → attach the policy to an **IAM user**, create an
  access key for it, and set the credentials via `DATA_STORE_*` (Step 3).

### Step 3 — Point `Settings` at the buckets

Unlike a single-bucket setup, this now needs `DATA_PATH` **plus** an explicit override per
delivery table, rather than one setting:

| Setting | Bucket | Why |
|---|---|---|
| `DATA_PATH` (root) | `nged-forecast-internal` | Every table derives from this unless overridden below — deliberately **fails closed**: forgetting to override a new delivery table just leaves it in the internal bucket (invisible to NGED, a functional bug to catch in testing), rather than a forgotten override on a new internal table accidentally exposing it in the delivery bucket. |
| `POWER_FORECASTS_DATA_PATH` | `nged-forecast-delivery` | Table 1, `power_forecast` — see the caveat below. |
| `EFFECTIVE_CAPACITY_DATA_PATH` | `nged-forecast-delivery` | Table 4, `effective_capacity`. |

The other three delivery tables (`power_forecast_warnings`, `asset_health_history`,
`substation_switching`) don't have `Settings` fields yet — none are implemented in code. When
they land, their paths need the same explicit override to `nged-forecast-delivery`.

> **Caveat: `power_forecasts` isn't purely delivery data.** The table holds every CV/backtest
> experiment alongside live production forecasts, distinguished only by `fold_id` (`"live"` for
> production — see [Running live forecasts end-to-end](dagster-workflow.md)). Pointing the whole
> table at the delivery bucket means OCF's own experiment churn is physically stored there too —
> not a new problem this split introduces (it's one table today regardless of bucket), and NGED
> already needs to filter on `fold_id="live"`. Splitting the table itself by `fold_id` across
> buckets would need an actual code change to the write path, not just a `Settings` override, so
> it's out of scope here.

**On AWS compute (IAM role)** — credentials and region are auto-discovered, so just:

```dotenv
DATA_PATH=s3://nged-forecast-internal/data
POWER_FORECASTS_DATA_PATH=s3://nged-forecast-delivery/data/power_forecasts
EFFECTIVE_CAPACITY_DATA_PATH=s3://nged-forecast-delivery/data/effective_capacity
```

**From your laptop (IAM user access key)** — same three paths, plus the key, secret, and region,
but **not** an endpoint URL (that is only for non-AWS/MinIO endpoints, and it would wrongly allow
plain HTTP):

```dotenv
DATA_PATH=s3://nged-forecast-internal/data
POWER_FORECASTS_DATA_PATH=s3://nged-forecast-delivery/data/power_forecasts
EFFECTIVE_CAPACITY_DATA_PATH=s3://nged-forecast-delivery/data/effective_capacity
DATA_STORE_ACCESS_KEY_ID=<access key id>
DATA_STORE_SECRET_ACCESS_KEY=<secret access key>
DATA_STORE_REGION=eu-west-2
```

The four `DATA_STORE_*` fields map onto the shared `aws_*` object_store option keys that delta-rs,
Polars, and obstore all understand, so this one dict feeds every read and write — to both buckets
alike, since bucket choice is entirely a matter of which URI each path setting resolves to.

### Step 4 — Verify

With the above in place, materialise a delivery-table asset (e.g. `live_forecasts`, which writes
`power_forecasts`) from the Dagster UI, then confirm the objects appear under
`s3://nged-forecast-delivery/data/power_forecasts/…` in the S3 console. Then materialise an
internal-table asset (e.g. `power_time_series_and_metadata`) and confirm it lands under
`s3://nged-forecast-internal/data/NGED/…` instead — two different buckets is exactly what you
want to see. Because `LOCAL_ARTIFACTS_PATH` stayed local, a subsequent `promoted_model`
materialisation still writes the model to `<repo>/data/production_model/` on disk.

### Step 5 — Grant NGED read access (recommended; confirm before doing this)

> This step hasn't been agreed as final yet — recorded here as the current recommendation so it
> isn't lost, but check before actually creating anything.

[Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it) already
assumes **a single authenticated AWS user** for NGED, with no per-user entitlement matrix needed
— NGED is the only consumer. That points at a dedicated **IAM user** (not a cross-account role):
Excel and Power BI are named as expected client tools in that same page, and neither supports
AWS role-assumption — they need a plain access key and secret, the same shape of credential an
IAM user provides.

Recommended shape: **one** IAM user (not one per bucket — the stability signal comes from the
bucket split itself, not from access segmentation), with a read-only policy across both bucket
ARNs (`s3:GetObject`, `s3:ListBucket` — no `PutObject`/`DeleteObject`), and an access key handed
to NGED the same way Step 2's laptop credentials are configured. Rotate the key periodically once
this is live.

### Viewing forecasts on AWS

Do **not** rely on the `plot_power_forecast` asset in a deployed service. It writes Altair HTML to
`LOCAL_ARTIFACTS_PATH`, which on an ephemeral Fargate task is a disk that is discarded when the task
exits — the file would never be seen. `plot_power_forecast` is a **local-development convenience**:
materialise it on your laptop, open the HTML in a browser.

The durable, S3-native way to look at forecasts is the **dashboard**
(`packages/dashboard/main.py`), which reads whichever tables it needs directly via their
`Settings` paths (with the same `storage_options`) and renders on demand — no plot files to
persist, and no need to know or care which bucket a given table resolves to. Point it at the same
`Settings` and it works identically whether each path is local or `s3://`.

### At-a-glance: which settings for which environment

| Environment | `DATA_PATH` | Delivery-table overrides | `DATA_STORE_*` |
|---|---|---|---|
| Local (default) | unset → `<repo>/data` | unset | none |
| Local MinIO rehearsal | `s3://…` | unset (single bucket) | all four, incl. `ENDPOINT_URL` |
| Laptop → real S3 | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | key + secret + region (no endpoint) |
| Compute on AWS (IAM role) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | none (auto-discovered) |
