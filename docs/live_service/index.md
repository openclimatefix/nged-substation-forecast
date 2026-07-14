# Live Service

How to operate the production forecasting service day to day. Unlike
[the roadmap](../roadmap/index.md) (forward-looking design for work not yet built), this is the
durable home for live-service **operational** docs — step-by-step recipes for running what's
already built — once each piece ships. Design rationale (the *why*) lives in
[`docs/architecture/`](../architecture/overview.md) instead; see that folder's
[Production Deployment — Design](../architecture/production-deployment.md) page for this area's
counterpart.
[The roadmap's Live Service page](../roadmap/live-service.md) sends readers here as its sections
land (so far: the `live_forecasts` and `promoted_model` assets, local 6-hourly automation, the
container build/verify runbook, and the AWS bring-up runbook; still to come: production
monitoring, alerting). Once the whole v0.1 epic ships, the roadmap page is deleted and this
section is the sole home for how the live service works.

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

The pages split by *which environment you're bringing up* and then *how to drive it* — the
driving is identical in both environments, so it lives on one shared page:

- [Running the whole stack locally](local.md) — bring the entire service up on a laptop:
  `.env`, a persistent `DAGSTER_HOME`, `dg dev`, and the optional MinIO rehearsal.
- [Setting up the live service on AWS](aws.md) — every step to stand the service up on AWS, in
  order: S3 buckets and IAM (AWS's storage and permissions services; every AWS acronym is
  spelled out on first use there), promote a champion and build/verify/push its image, the
  Fargate task, the always-on control-plane box, and connecting to the Dagster UI over
  Tailscale.
- [Operating the live service](operations.md) — driving a running stack day to day: promote a
  champion model, let the 6-hourly `live_forecasts` schedule run (or materialise a slot by
  hand), inspect a forecast, and backfill a missed slot in replay mode.
- [Configuration reference](setup.md) — what the storage roots, the derive-from-root
  convention, and the credential settings mean, and which combination each environment uses.
