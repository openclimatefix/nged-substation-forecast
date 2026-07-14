# Live Service

How to operate the production forecasting service day to day. Unlike
[the roadmap](../roadmap/index.md) (forward-looking design for work not yet built), this is the
durable home for live-service **operational** docs — step-by-step recipes for running what's
already built — once each piece ships. Design rationale (the *why*) lives in
[`docs/architecture/`](../architecture/overview.md) instead; see that folder's
[Production Deployment — Design](../architecture/production-deployment.md) page for this area's
counterpart.
[The roadmap's Live Service page](../roadmap/live-service.md) sends readers here as its sections
land (so far: the `live_forecasts` and `promoted_model` assets, local 6-hourly automation, and
the container build/verify runbook; still to come: production monitoring, AWS compute). Once the
whole v0.1 epic ships, the roadmap page is deleted and this section is the sole home for how the
live service works.

This is distinct from [ML Experimentation](../ml_experimentation/index.md): that area covers
training and backtesting candidate models against historical data; this area covers picking one
of those candidates as the running production model and keeping live forecasts flowing from it.

**Audience note:** today these runbooks are written for OCF (Python-literate researchers), but
after the NIA project the day-to-day operator is expected to be a non-expert at NGED — NGED's
stated preference (2026-07-14, pending their internal sign-off) is to run the service
themselves, on their own AWS account. Every routine
operator action must therefore eventually reduce to a dashboard check, a button in the Dagster
UI, or a runbook a non-Python-expert can follow; before handover, these pages get an editing
pass with that operator as the audience, plus a top-level "operator contract" page indexing
them. See [Handover to NGED](../roadmap/handover.md).

## Documents

- [Environment & storage setup](setup.md) — where the data tables and local artifacts live, and
  how to configure credentials for running locally, against a local MinIO, or with the data tables
  on AWS S3.
- [Deploying a new production image](deployment.md) — step-by-step recipe: promote a champion
  model, build the image, verify it runs with zero MLflow dependency.
- [Running live forecasts end-to-end](dagster-workflow.md) — step-by-step recipe: promote a
  champion model to `promoted_model`, let the 6-hourly `live_forecasts` schedule run (or
  materialise a slot by hand), inspect a forecast, and backfill a missed slot in replay mode.
