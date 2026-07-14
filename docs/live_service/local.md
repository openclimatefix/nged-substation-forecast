# Running the whole stack locally

The entire live service — telemetry ingestion, NWP download, the 6-hourly forecast schedule,
and the Dagster UI — runs on a laptop with no AWS involved. This is a deliberate portability
requirement, not just a development convenience: no cloud-specific service may be load-bearing
for scheduling or orchestration (see
[the orchestration decision](../architecture/production-deployment.md#orchestration-an-always-on-dagster-control-plane-not-eventbridge)).
This page brings the local stack up; once it's running,
[Operating the live service](operations.md) drives it — promotion, schedules, backfills are
identical to the AWS deployment.

## Step 1 — Repository and credentials

Follow the [repository README's Setup section](https://github.com/openclimatefix/nged-substation-forecast#setup)
for first-time repo setup (`uv`, `uv sync`, pre-commit hooks), and create a `.env` in the repo
root containing the three NGED source-bucket credentials — the only values that are always
required (see [Configuration reference](setup.md#the-env-file-and-nged-source-credentials)):

```dotenv
NGED_S3_BUCKET_URL=<nged source bucket url>
NGED_S3_BUCKET_ACCESS_KEY=<key>
NGED_S3_BUCKET_SECRET=<secret>
```

With nothing else set, every storage root resolves to `<repo>/data` and every table is a plain
file on disk — see the [Configuration reference](setup.md#three-storage-roots) for the model.

## Step 2 — A persistent Dagster instance

The 6-hourly schedule only fires while Dagster's daemon is running continuously, so `uv run dg
dev` needs to keep running (rather than being started and stopped around each manual step) and
needs a persistent `DAGSTER_HOME` so its schedule state survives a restart:

1. `mkdir ~/dagster_home/` and put the `dagster.yaml` shown in the repository README's
   [Setup](https://github.com/openclimatefix/nged-substation-forecast#setup) section into it.
2. `export DAGSTER_HOME=~/dagster_home` (add to `.bashrc` so it persists across terminals).

## Step 3 — Start Dagster

```bash
uv run dg dev
```

Leave it running, and open `http://localhost:3000` for the UI. Schedules fire only while this
process is up — a laptop lid-close stops the service, which is exactly the gap the
[AWS deployment](aws.md) exists to close (and why a missed slot is backfillable — see
[Operating the live service: Backfilling a missed slot](operations.md#backfilling-a-missed-slot)).

From here, everything is driven from the UI: promote a champion model and let the schedules
produce forecasts — [Operating the live service](operations.md).

## Optional — rehearse S3 locally with MinIO

To exercise the exact S3 (Simple Storage Service — AWS's object store) read/write code paths
without touching AWS, point both `DATA_PATH_INTERNAL`
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

## See also

- [Operating the live service](operations.md) — promotion, the 6-hourly schedule, inspecting
  forecasts, and backfilling missed slots, all identical on the local stack.
- [Configuration reference](setup.md) — the storage roots, the derive-from-root convention, and
  which settings each environment uses.
- [Setting up the live service on AWS](aws.md) — the same stack deployed unattended on AWS.
