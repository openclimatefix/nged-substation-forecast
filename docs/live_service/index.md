# Live Service

How to operate the production forecasting service day to day — the durable home for live-service
operational docs once each piece ships, per the
[Documentation Guide](../documentation-guide.md#how-planning-works). Design rationale (the *why*)
lives in [`docs/architecture/`](../architecture/overview.md) instead; see that folder's
[Production Deployment — Design](../architecture/production-deployment.md) page for this area's
counterpart.
[The roadmap's Live Service page](../roadmap/live-service.md) sends readers here as its sections
land (so far: the `live_forecasts` and `promoted_model` assets, local 6-hourly automation, the
container build/verify runbook, the AWS bring-up runbook, and Sentry telemetry with the
missed-check-in alarm; still to come: production monitoring, and per-task failure-email alerting).
Once production monitoring lands and its design is promoted, the roadmap page is deleted and this
section is the sole home for how the live service works.

The live service is distinct from [ML Experimentation](../ml_experimentation/index.md): that
area covers training and backtesting candidate models against historical data; this area covers
picking one of those candidates as the running production model and keeping live forecasts
flowing from it.

**Audience note:** today these runbooks are written for OCF (Python-literate researchers), but
after the NIA project the day-to-day operator is expected to be a non-expert at NGED — NGED's
stated preference (2026-07-14, pending their internal sign-off) is to run the service
themselves, on their own AWS account. Every routine
operator action must therefore eventually reduce to a dashboard check, a button in the Dagster
UI, or a runbook a non-Python-expert can follow; before handover, these pages get an editing
pass with that operator as the audience, plus a top-level "operator contract" page indexing
them. See [Handover to NGED](../roadmap/handover.md).

## Documents

Bringing the whole service up on a laptop is part of [Getting started on your
laptop](../getting-started.md#running-the-live-service-on-your-laptop) rather than a page here,
because the laptop stack is the same one a first-time setup already produces. The pages below
cover the AWS bring-up and then how to drive a running stack; the driving is identical in both
environments, so it lives on one shared page.

- [Setting up the live service on AWS](aws.md) — every step to stand the service up on AWS, in
  order: S3 buckets and IAM (AWS's storage and permissions services; every AWS acronym is
  spelled out on first use there), promote a champion and build/verify/push its image, the
  Fargate task, the always-on control-plane box, and connecting to the Dagster UI over
  Tailscale.
- [Connecting to the AWS control plane](connecting.md) — get a laptop onto the OCF tailnet to
  reach an already-running deployment: install Tailscale, view the Dagster UI, and SSH into the
  always-on box.
- [Operating the live service](operations.md) — driving a running stack day to day: promote a
  champion model, let the 6-hourly `live_forecasts` schedule run (or materialise a slot by
  hand), inspect a forecast, and backfill a missed slot in replay mode.
- [Intervention log](intervention-log.md) — the append-only record of every occasion a human had to
  intervene in the running service, and the artefact the
  [T1.1 operability test](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself)
  is scored from.
- [Setting up Sentry telemetry](sentry.md) — point error reporting and the missed-check-in alarm
  at a Sentry project: get a DSN, test it from your laptop, and turn it on in production.
- [Configuration reference](setup.md) — what the storage roots, the derive-from-root
  convention, and the credential settings mean, and which combination each environment uses.
