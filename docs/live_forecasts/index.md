# Live Forecasts

How to operate the production forecasting service day to day. Unlike the
[roadmap](../roadmap/index.md) (forward-looking design for work not yet built), this area
documents the local, 6-hourly live service that is **implemented** and running today — the
durable home for live-service operational docs once they leave
[the roadmap](../roadmap/live-service.md#the-live_forecasts-asset).

This is distinct from [ML Experimentation](../ml_experimentation/index.md): that area covers
training and backtesting candidate models against historical data; this area covers picking one
of those candidates as the running production model and keeping live forecasts flowing from it.

## Documents

- [Running live forecasts end-to-end](dagster-workflow.md) — step-by-step recipe: promote a
  champion model to `production_model`, let the 6-hourly `live_forecasts` schedule run (or
  materialise a slot by hand), inspect a forecast, and backfill a missed slot in replay mode.
