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
a separate bucket, and nothing else needs to change.

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

### Step 1 — Create the S3 bucket

In the AWS console → **S3** → **Create bucket**:

1. **Region** `eu-west-2` (London) — keep every resource in one region so S3 ↔ compute transfer
   stays free.
2. **Name** e.g. `nged-flexpectation-data` (bucket names are globally unique; pick your own).
3. Leave **Block all public access** **on**, and default **SSE-S3** encryption on. Nothing here is
   public.
4. **Versioning** is optional and not required by the app.

No DynamoDB lock table is needed. The `deltalake` version we use commits via S3's native
conditional-put, so concurrent-safe Delta writes work on plain S3 with no lock table and no
`AWS_S3_ALLOW_UNSAFE_RENAME` flag.

### Step 2 — Grant access with IAM

Create an IAM policy scoped to just this bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::nged-flexpectation-data",
        "arn:aws:s3:::nged-flexpectation-data/*"
      ]
    }
  ]
}
```

Attach it to **whichever identity runs the code**:

- **Compute running on AWS** (an EC2 box or Fargate task) → attach the policy to that resource's
  **IAM role**. Nothing else is needed: delta-rs' object_store auto-discovers the role's temporary
  credentials and region at runtime, so you leave all four `DATA_STORE_*` settings **empty**.
- **Your laptop, against the real bucket** → attach the policy to an **IAM user**, create an access
  key for it, and set the credentials via `DATA_STORE_*` (Step 3).

### Step 3 — Point `Settings` at the bucket

Set `DATA_PATH` to the bucket URI, and leave `LOCAL_ARTIFACTS_PATH` unset so models and plots stay on
local disk. What else you set depends on Step 2:

**On AWS compute (IAM role)** — credentials and region are auto-discovered, so just:

```dotenv
DATA_PATH=s3://nged-flexpectation-data/data
```

**From your laptop (IAM user access key)** — supply the key and region, but **not** an endpoint URL
(that is only for non-AWS/MinIO endpoints, and it would wrongly allow plain HTTP):

```dotenv
DATA_PATH=s3://nged-flexpectation-data/data
DATA_STORE_ACCESS_KEY_ID=<access key id>
DATA_STORE_SECRET_ACCESS_KEY=<secret access key>
DATA_STORE_REGION=eu-west-2
```

The four `DATA_STORE_*` fields map onto the shared `aws_*` object_store option keys that delta-rs,
Polars, and obstore all understand, so this one dict feeds every read and write.

### Step 4 — Verify

With the above in place, materialise any data-writing asset (e.g. `power_time_series_and_metadata`)
from the Dagster UI, then confirm the objects appear under `s3://nged-flexpectation-data/data/…` in
the S3 console. Because `LOCAL_ARTIFACTS_PATH` stayed local, a subsequent `promoted_model`
materialisation still writes the model to `<repo>/data/production_model/` on disk — the two roots
behaving independently is exactly what you want to see.

### Viewing forecasts on AWS

Do **not** rely on the `plot_power_forecast` asset in a deployed service. It writes Altair HTML to
`LOCAL_ARTIFACTS_PATH`, which on an ephemeral Fargate task is a disk that is discarded when the task
exits — the file would never be seen. `plot_power_forecast` is a **local-development convenience**:
materialise it on your laptop, open the HTML in a browser.

The durable, S3-native way to look at forecasts is the **dashboard**
(`packages/dashboard/main.py`), which reads the `power_forecasts` and metadata tables directly from
`DATA_PATH` (with the same `storage_options`) and renders on demand — no plot files to persist.
Point it at the same `Settings` and it works identically whether `DATA_PATH` is local or `s3://`.

### At-a-glance: which settings for which environment

| Environment | `DATA_PATH` | `DATA_STORE_*` |
|---|---|---|
| Local (default) | unset → `<repo>/data` | none |
| Local MinIO rehearsal | `s3://…` | all four, incl. `ENDPOINT_URL` |
| Laptop → real S3 | `s3://…` | key + secret + region (no endpoint) |
| Compute on AWS (IAM role) | `s3://…` | none (auto-discovered) |
