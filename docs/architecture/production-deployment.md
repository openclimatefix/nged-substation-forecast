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

## Live inference is single-run, not bulk

The `live_forecasts` asset engineers features in **single-run mode** — an explicit
`power_fcst_init_time` supplied by the partition, joined across all 51 NWP ensemble members —
never bulk mode's one-`power_fcst_init_time`-per-NWP-run derivation (see
[ML orchestration: forecast cadence under-sampling](ml-orchestration.md#known-limitation-forecast-cadence-under-sampling-in-cv)
for where bulk mode's per-NWP-run derivation matters instead: CV/backtesting, not live inference).
A production run issues one forecast for one explicit init time every 6 hours; bulk mode's
derivation has no meaning for a single live materialisation.

## NWP availability: `live` vs `replay` is asymmetric by design

`live_forecasts`' `availability_mode` config resolves which NWP run to join against, and the two
modes are deliberately asymmetric:

- **`"live"`** joins the freshest NWP run actually present in Delta, with **no modelled
  publication delay** — reality already constrains the table to genuinely published runs, so a
  faster provider is used automatically without a config change.
- **`"replay"`** joins the freshest run at least `nwp_publication_delay_hours` old, reconstructing
  what was genuinely available at that historical init time. Without the delay, a replay would
  leak NWP runs that only landed after the fact — a lookahead-bias bug, not just an inaccuracy.

The scheduled path always uses `"live"`; backfills of missed or historical partitions use
`"replay"`. The mode is an explicit, manually-set flag rather than an automatic
live-iff-recent rule — the ambiguity of "recent" isn't worth resolving for a backfill path that's
already manually triggered.

## Serving only the trained population

`live_forecasts` forecasts exactly the production model's `trained_time_series_ids` (recorded in
`meta.json`), never the current day's eligibility set. This is the train==predict population
invariant: a time series the model never saw during training must never receive a live forecast,
even if it would otherwise qualify today.

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
