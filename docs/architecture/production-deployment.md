# Production Deployment — Design

How the champion model gets from an MLflow leaderboard into a running production container, and
why. For the step-by-step recipe — promote a model, build the image, verify it, push it — see
[Deploying a new production image](../live_service/deployment.md).

## Baking the model into the image at build time

Production forecasts run as ephemeral Fargate tasks under every AWS architecture option being
considered (see [Live service: AWS architecture](../roadmap/live-service.md#aws-architecture)).
An ephemeral container has no persistent disk and, under some architecture options, no reachable
tracking server — so production inference needs some way to get a model without depending on
either.

**Decision: bake the champion model into the image at build time, loaded via a plain
`save`/`load` — no MLflow, run ID, or cache involved at runtime.** The model directory is
produced once, out of band (a researcher picks the champion fold from the MLflow leaderboard and
downloads its artifacts to local disk), then `COPY`'d into the image at build time. Promotion
becomes rebuild + redeploy, which is auditable (image tags) and keeps MLflow completely out of
the production runtime under every architecture option.

**Rejected alternative:** MLflow artifact root on S3 fetched at container startup — more runtime
moving parts, needs tracking-store access from prod, and slower cold starts.

This is deliberately simpler than reusing `BaseForecaster.load_from_mlflow`'s cache (the
mechanism the CV pipeline already uses — see
[ML orchestration: model artifacts](ml-orchestration.md#model-artifacts-mlflow-artifact-store-immutable-local-cache)):
v0.1 has no MLflow dependency to cache against in the first place.

**Future work:** once production wants to pick up a new champion without a rebuild + redeploy
(e.g. after the [XGBoost quick wins](../roadmap/xgboost-improvements.md) start landing
regularly), switch to fetching the champion model from MLflow dynamically — at that point
`load_from_mlflow`'s local-disk cache becomes the production-resilience mechanism again (serving
from disk on a cache hit so the live service survives an MLflow outage), exactly as it does for
CV today.

## Why promotion is a Dagster asset, not a script

The "researcher downloads artifacts" step above is a manually-triggered Dagster asset,
`promoted_model` (config `mlflow_run_id`), rather than a bare script — promotion becomes a
materialisation, giving an audit trail and lineage for free. The download logic itself
(`ml_core._production_helpers.fetch_model_artifacts`) is a pure, asset-independent helper, so
nothing about this decision couples it to Dagster.

**The Docker build reuses this same asset** (headlessly, via `dagster asset materialize`) — no
separate fetch script was built, since a bare script would have duplicated the asset's audit
trail for no benefit. The `docker build` step itself stays outside Dagster: it only ever runs on
a laptop today, and image build/push becomes a CI-shaped concern once an MLflow tracking server
and AWS infra exist — not something worth orchestrating through Dagster in the meantime.

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

- [Live service roadmap](../roadmap/live-service.md) — the full v0.1 design, including the
  AWS architecture options still being decided.
- [Deploying a new production image](../live_service/deployment.md) — the step-by-step
  promotion/build/push runbook.
- [Environment & storage setup](../live_service/setup.md) — where data tables and local
  artifacts live, and how to point `Settings` at S3.
- [ML Orchestration Design](ml-orchestration.md) — why production inference doesn't reuse the
  CV pipeline's MLflow-artifact cache.
