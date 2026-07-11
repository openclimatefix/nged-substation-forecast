# Deploying a new production image

How to get a promoted champion model into a runnable production image. Design rationale for why
the container is built this way — bake the model in at build time, no MLflow at runtime — lives
in [Production Deployment — Design](../architecture/production-deployment.md); this page is the
mechanical recipe.

> **Scope: building and verifying the image.** Pushing the image to a live AWS deployment (ECR,
> compute) is a separate, not-yet-built step — see the
> [AWS architecture](../roadmap/live-service.md#aws-architecture) roadmap section for that plan.

## Step 1 — Pick and promote a champion model

Follow [Running live forecasts end-to-end: Step 1](dagster-workflow.md#step-1-pick-a-champion-model)
and [Step 2](dagster-workflow.md#step-2-materialise-promoted_model) to materialise
`promoted_model`. This populates `data/production_model/` on disk — the same directory the
image build below `COPY`s from.

## Step 2 — Build the image

The build never contacts MLflow — it only `COPY`s the directory Step 1 just populated, so it
stays hermetic:

```bash
docker build \
  --build-arg MODEL_RUN_ID=<run-id> \
  --build-arg GIT_SHA=$(git rev-parse HEAD) \
  -t nged-forecast:<run-id-short> .
```

`MODEL_RUN_ID` and `GIT_SHA` are stamped as OCI labels (and `GIT_SHA` also as a runtime env
var) purely for traceability — confirm with `docker inspect nged-forecast:<tag>`.

## Step 3 — Verify the build locally

Before trusting a new image, confirm it actually runs with **zero network access** — this is
the test that matters, since the entire point of baking the model in is that production
inference has no MLflow dependency at runtime:

```bash
docker run --network=none nged-forecast:<tag> \
  job execute -m nged_substation_forecast.definitions -j live_forecasts_job \
  --tags '{"dagster/partition": "<key>"}'
```

Partition selection uses `--tags`, not `--select`/`--partition`: `dagster job execute` has no
`--partition` flag at all, and `dagster asset materialize --select <asset>` (which does) hits a
pre-existing, unrelated `antlr4-python3-runtime`/Python 3.14 incompatibility in Dagster's own
asset-selection-string parser — reproduces identically outside Docker, on a plain `dg dev`
checkout, so it's an upstream/environment issue, not something introduced by this image.
Selecting by job name (`-j`) skips that parser entirely.

**Verified 2026-07-10** against a real promoted model: the run reaches
`load_forecaster_from_dir` and loads the model successfully (confirmed via the step ordering in
`production_assets.py` — model load happens before the NWP-availability lookup) with zero
`mlflow` mentions anywhere in the log, then fails only on missing NWP data access (expected —
no `DATA_PATH` was mounted for this isolated test). A full end-to-end run needs a real
`DATA_PATH` (local mount or S3 credentials) supplied via environment variables, per
[Environment & storage setup](setup.md).

## Step 4 — Push to ECR and deploy (not yet built)

Pushing the verified image to ECR and running it as a scheduled Fargate task needs the AWS
compute infrastructure itself, which doesn't exist yet — see
[Live service: AWS architecture](../roadmap/live-service.md#aws-architecture) and
[#286](https://github.com/openclimatefix/nged-substation-forecast/issues/286).

## See also

- [Production Deployment — Design](../architecture/production-deployment.md) — why the image is
  built this way.
- [Running live forecasts end-to-end](dagster-workflow.md) — driving the promoted model day to
  day once it's live, and backfilling missed slots.
- [Environment & storage setup](setup.md) — where data tables and local artifacts live.
