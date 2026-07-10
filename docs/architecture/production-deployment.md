# Production Deployment — the Container Build

How the champion model gets from an MLflow leaderboard to a running production container.
Design rationale lives in
[Live service → Production model artifacts](../roadmap/live-service.md#production-model-artifacts):
**bake the champion model into the image at build time**, loaded via a plain `save`/`load` — no
MLflow, run ID, or cache lookup at runtime. Promotion is therefore rebuild + redeploy, which is
auditable via image tags and keeps MLflow completely out of the production runtime.

This page assumes you've already read
[Running live forecasts end-to-end](../live_service/dagster-workflow.md), which covers
promoting a model and running `live_forecasts` **locally**. This page covers the one extra step
needed to run the same thing as an unattended container: getting the promoted model *into* an
image.

## The promotion runbook

1. **Pick the champion fold run ID** from the MLflow leaderboard — see
   [Running live forecasts end-to-end → Step 1](../live_service/dagster-workflow.md#step-1--pick-a-champion-model).

2. **Materialise `promoted_model`** to populate `data/production_model/` on disk. This is the
   same asset the local workflow uses — no separate script — either from the Dagster UI
   ("Materialize" with `PromotedModelConfig.mlflow_run_id` filled in), or headlessly:

   ```bash
   uv run dagster asset materialize -m nged_substation_forecast.definitions --select promoted_model \
     --config-json '{"ops": {"promoted_model": {"config": {"mlflow_run_id": "<run-id>"}}}}'
   ```

3. **Build the image.** The build never contacts MLflow — it only `COPY`s the directory step 2
   just populated, so it stays hermetic:

   ```bash
   docker build \
     --build-arg MODEL_RUN_ID=<run-id> \
     --build-arg GIT_SHA=$(git rev-parse HEAD) \
     -t nged-forecast:<run-id-short> .
   ```

   `MODEL_RUN_ID` and `GIT_SHA` are stamped as OCI labels (and `GIT_SHA` also as a runtime env
   var) purely for traceability — confirm with `docker inspect nged-forecast:<tag>`.

4. **Push to ECR and point the ECS task definition at the new tag.** (ECR repository and ECS
   task definitions are provisioned by the AWS infrastructure work,
   [#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206) — not yet
   built at the time this page was written.)

## Verifying a build locally

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
[Environment & storage setup](../live_service/setup.md).

## Two subtleties for the AWS deployment (not yet built)

Recorded here so they aren't lost when the Fargate work
([#206](https://github.com/openclimatefix/nged-substation-forecast/issues/206)) starts:

- **Freshness without persistent Dagster state.** If the eventual AWS deployment runs the
  container one-shot with no daemon behind it (the "nothing always-on" architecture option),
  "which `ecmwf_ens` partitions need materialising" must be derived from **Delta table
  contents vs Dynamical.org availability**, not from Dagster's own materialisation records —
  those evaporate with a throwaway, non-persistent `DAGSTER_HOME`. This doesn't apply to a
  daemon-backed deployment (persistent `DAGSTER_HOME`), where Dagster's records are reliable.
- **Delta commits as the freshness record.** Delta table commits already give an atomic
  "outputs are the freshness record" property for free — just ensure the forecast Delta write
  is a run's *final* write, so a run that fails after writing forecasts but before some later
  step doesn't get treated as stale on the next freshness check.

## See also

- [Live service roadmap](../roadmap/live-service.md) — the full v0.1 design, including AWS
  architecture options still being decided.
- [Environment & storage setup](../live_service/setup.md) — where data tables and local
  artifacts live, and how to point `Settings` at S3.
- [ML Orchestration Design](ml-orchestration.md) — why production inference doesn't reuse the
  CV pipeline's MLflow-artifact cache.
