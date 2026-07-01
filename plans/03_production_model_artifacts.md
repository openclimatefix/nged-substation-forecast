# Production model artifacts: bake the champion model into the Docker image

## Finding

The AWS deployment plan (issue [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206), Level 1: hourly EventBridge → one-shot Fargate → exit) is
the right architecture, but it never mentions MLflow — and the inference path loads models via
`BaseForecaster.load_from_mlflow(run_id, cache_base_path)`
(`packages/ml_core/src/ml_core/base_forecaster.py:140`). An ephemeral container has neither a
persistent model cache nor a reachable tracking server, so as written, production inference has
no way to get a model. This must be decided before writing the Dockerfile.

**Decision: bake the champion model into the image at build time.** The cache-hit path in
`load_from_mlflow` (`base_forecaster.py:157-166`) never contacts MLflow when
`{cache_base_path}/{run_id}/model` exists — the existing design already supports fully offline
serving. Promotion becomes rebuild + redeploy, which is auditable (image tags) and preserves
Level 1's "nothing always-on" property. Rejected alternative: MLflow artifact root on S3
fetched at container startup — more runtime moving parts, needs tracking-store access from
prod, and slower cold starts.

## Implementation

### 1. Dockerfile (repo root)

- Multi-stage `uv` build (standard pattern: `ghcr.io/astral-sh/uv` image, `uv sync --frozen
  --no-dev`, copy `.venv` + source into a slim `python:3.14` runtime stage).
- Build args: `PRODUCTION_MODEL_RUN_ID` (required), `GIT_SHA` (stamped as an OCI label and env
  var; complements plan 08).
- Model injection: keep the image build hermetic — the build **copies** the model directory
  from the build context (`data/model_cache/{run_id}/model`, populated beforehand by a small
  script `scripts/fetch_model.py` that calls `load_from_mlflow` against the researcher's
  tracking store). `COPY` it to the image's `model_cache_base_path` location and set
  `ENV PRODUCTION_MODEL_RUN_ID=...`. Downloading from inside `docker build` is rejected
  (needs tracking credentials in the build).
- Entrypoint: `dagster job execute` one-shot (the exact job comes with the future
  `live_forecasts` asset; until that exists, the entrypoint can materialise the data-ingestion
  assets so the image is testable now).

### 2. Settings

`Settings.production_model_run_id` and `model_cache_base_path` already exist
(`packages/contracts/src/contracts/settings.py`) — no schema change; just document that in the
container they're env-var-driven (`.env` is dev-only).

### 3. Promotion runbook

Short page `docs/architecture/production-deployment.md` (permanent docs, not this plan):

1. Pick the champion fold run ID from the MLflow leaderboard.
2. `uv run python scripts/fetch_model.py --run-id <id>` → populates `data/model_cache/`.
3. `docker build --build-arg PRODUCTION_MODEL_RUN_ID=<id> --build-arg GIT_SHA=$(git rev-parse HEAD)
   -t nged-forecast:<id-short> .` and push to ECR.
4. Point the ECS task definition at the new tag.

Also record the two issue-#206 subtleties this review surfaced, so they're not lost when the
Fargate work starts: (a) with throwaway SQLite, "which `ecmwf_ens` partitions need
materialising" must be derived from Delta contents vs Dynamical availability, not Dagster's
materialisation records; (b) Delta commits already give the atomic "outputs are the freshness
record" property — just ensure the forecast Delta write is the run's final write.

## Verification

1. `docker build` with a smoke-test-fold run ID succeeds.
2. `docker run` **with no network access to any MLflow store** loads the model (cache-hit
   path) and executes the entrypoint — this is the critical test.
3. Image labels show the run ID and git SHA (`docker inspect`).
