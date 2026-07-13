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
> [Running on AWS: Step 3](#step-3-point-settings-at-the-buckets) for the concrete setup, and
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
  [roadmap: Production model artifacts](https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#production-model-artifacts),
  [#222](https://github.com/openclimatefix/nged-substation-forecast/issues/222)). This local
  directory is the build-time staging area that gets copied into the image; on the deployed task the
  model is read from the image's own filesystem.
- **Plot HTML** — a **local-dev convenience** only (materialise, open in a browser); see the note
  under [Running on AWS](#viewing-forecasts-on-aws) for why it is not the way to view forecasts in a
  deployed service.

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
a separate bucket, and nothing else needs to change. [Running on AWS: Step
3](#step-3-point-settings-at-the-buckets) is exactly this pattern applied on purpose: on AWS,
`DATA_PATH_INTERNAL` points at OCF's internal bucket and `DATA_PATH_DELIVERY` points at the
NGED-facing bucket — no per-table override needed for either of today's two delivery tables.

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

To exercise the exact S3 read/write code paths without touching AWS, point both `DATA_PATH_INTERNAL`
and `DATA_PATH_DELIVERY` at the same local [MinIO](https://min.io/) (or any S3-compatible) endpoint
and supply all four `DATA_STORE_*` settings:

```dotenv
DATA_PATH_INTERNAL=s3://my-bucket/data
DATA_PATH_DELIVERY=s3://my-bucket/data
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
> and is enough to run the stack from your laptop against S3 data-table roots today. Building the
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
   transfer stays free. It isn't the cheapest region available (`eu-west-1` runs meaningfully
   cheaper for Fargate); see [Forecast Delivery: Securing it](../architecture/forecast-delivery.md#securing-it)
   for the price comparison and why `eu-west-2` is picked anyway, provisionally, pending NGED
   confirmation.
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
- **Your laptop, running the pipeline** (ingestion, training, backfilling) → attach the read/write
  policy above to an **IAM user**, create an access key for it, and set the credentials via
  `DATA_STORE_*` (Step 3).
- **Your laptop, running only the dashboard** (`packages/dashboard/main.py` — see
  [#283](https://github.com/openclimatefix/nged-substation-forecast/issues/283)) → it only ever
  reads, so give it a separate, read-only **IAM user** instead of reusing the pipeline's read/write
  credential:

  ```json
  {
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:GetObject", "s3:ListBucket"],
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

  Same `DATA_STORE_*` shape as the pipeline laptop (Step 3), just backed by a narrower key —
  losing this key can't corrupt any table, only leak read access to it.

### Step 3 — Point `Settings` at the buckets

Unlike a single-bucket setup, this now needs **both** data-table roots set, one per bucket:

| Setting | Bucket | Why |
|---|---|---|
| `DATA_PATH_INTERNAL` (root) | `nged-forecast-internal` | Every non-delivery table derives from this. |
| `DATA_PATH_DELIVERY` (root) | `nged-forecast-delivery` | `power_forecasts_data_path` and `effective_capacity_data_path` derive from this, hard-coded in `Settings._derive_unset_paths` — deliberately **fails closed**: a new delivery table added to the schema but not wired into that derivation stays on `DATA_PATH_INTERNAL` (invisible to NGED, a functional bug caught by `test_settings.py`'s guard test), rather than a forgotten override on a new internal table accidentally exposing it in the delivery bucket. |

The other three delivery tables (`power_forecast_warnings`, `asset_health_history`,
`substation_switching`) don't have `Settings` fields yet — none are implemented in code. When
they land, they need to be wired into `Settings._derive_unset_paths` to derive from
`DATA_PATH_DELIVERY` alongside the two above (the per-field env var override — e.g.
`POWER_FORECASTS_DATA_PATH`, still works as an escape valve for a one-off case, but is no longer
the primary mechanism for the delivery/internal split).

> **Design choice: NGED sees every CV/backtest fold, not just live production forecasts.** The
> `power_forecasts` table holds every CV/backtest experiment alongside live production forecasts,
> distinguished by `fold_id` (`"live"` for production — see
> [Running live forecasts end-to-end](dagster-workflow.md)). Pointing the whole table at the
> delivery bucket is deliberate, not an accidental side effect of the bucket split: it lets NGED
> see how each of OCF's model versions actually behaves, not just whichever one is currently
> promoted. NGED filters on `fold_id="live"` when it only wants the current production forecast.

**On AWS compute (IAM role)** — credentials and region are auto-discovered, so just:

```dotenv
DATA_PATH_INTERNAL=s3://nged-forecast-internal/data
DATA_PATH_DELIVERY=s3://nged-forecast-delivery/data
```

**From your laptop (IAM user access key)** — same two roots, plus the key, secret, and region,
but **not** an endpoint URL (that is only for non-AWS/MinIO endpoints, and it would wrongly allow
plain HTTP):

```dotenv
DATA_PATH_INTERNAL=s3://nged-forecast-internal/data
DATA_PATH_DELIVERY=s3://nged-forecast-delivery/data
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

| Environment | `DATA_PATH_INTERNAL` | `DATA_PATH_DELIVERY` | `DATA_STORE_*` |
|---|---|---|---|
| Local (default) | unset → `<repo>/data` | unset → `<repo>/data` | none |
| Local MinIO rehearsal | `s3://…` | same `s3://…` (single bucket) | all four, incl. `ENDPOINT_URL` |
| Laptop → real S3 (pipeline) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | key + secret + region, read/write IAM user (no endpoint) |
| Laptop → real S3 (dashboard only) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | key + secret + region, read-only IAM user (no endpoint) |
| Compute on AWS (IAM role) | `s3://nged-forecast-internal/…` | `s3://nged-forecast-delivery/…` | none (auto-discovered) |
