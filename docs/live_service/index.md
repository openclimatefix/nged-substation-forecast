# Live Service

How to operate the production forecasting service day to day. Unlike
[the roadmap](../roadmap/index.md) (forward-looking design for work not yet built), this is the
durable home for live-service operational docs once each piece ships — where
[the roadmap's Live Service page](../roadmap/live-service.md) sends readers as its sections land
(so far: the `live_forecasts` and `promoted_model` assets and local 6-hourly automation; still
to come: production monitoring, the container build, AWS deployment). Once the whole v0.1 epic
ships, the roadmap page is deleted and this section is the sole home for how the live service
works.

This is distinct from [ML Experimentation](../ml_experimentation/index.md): that area covers
training and backtesting candidate models against historical data; this area covers picking one
of those candidates as the running production model and keeping live forecasts flowing from it.

## Documents

- [Environment & storage setup](setup.md) — where the data tables and local artifacts live, and
  how to configure credentials for running locally, against a local MinIO, or with the data tables
  on AWS S3.
- [Running live forecasts end-to-end](dagster-workflow.md) — step-by-step recipe: promote a
  champion model to `promoted_model`, let the 6-hourly `live_forecasts` schedule run (or
  materialise a slot by hand), inspect a forecast, and backfill a missed slot in replay mode.
