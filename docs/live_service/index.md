# Live Service

How to operate the production forecasting service day to day — the durable home for live-service
operational docs once each piece ships, per the [Documentation
Guide](../documentation-guide.md#how-planning-works). Design rationale (the *why*) lives in
[`docs/architecture/`](../architecture/overview.md) instead; see that folder's [Production
Deployment — Design](../architecture/production-deployment.md) page for this area's counterpart.
[The roadmap's Live Service page](../roadmap/live-service.md) sends readers here as its sections
land. So far, the ingest assets and their local automation, the container build/verify runbook, the
AWS bring-up runbook, and Sentry telemetry have landed. A live forecast asset, production
monitoring, and per-task failure-email alerting are still to come. Once production monitoring lands
and its design moves here, the roadmap page is deleted and this section is the sole home for how the
live service works.

The live service is distinct from [Cross-validation](../cross-validation/index.md). The
cross-validation section covers training and backtesting candidate forecasters against historical
data. The live-service section covers the production service that runs unattended, which ingests the
inputs a live forecast reads and checks those inputs. The live service currently writes no forecast.

**Audience note:** today these runbooks are written for OCF (Python-literate researchers). The
working assumption is that, after the Network Innovation Allowance (NIA) project, NGED runs the
service on its own AWS account. The day-to-day operator is therefore one of the NGED staff who did
not develop the code, working from these runbooks. Every routine operator action must therefore
eventually reduce to a dashboard check, a button in the Dagster UI, or a runbook that someone who
does not write Python can follow; before handover, these pages get an editing pass with that
operator as the audience, plus a top-level "operator contract" page indexing them. See [Handover to
NGED](../roadmap/handover.md).

## Documents

Bringing the whole service up on a laptop is part of [Getting started on your
laptop](../getting-started.md#running-the-live-service-on-your-laptop) rather than a page here,
because the laptop stack is the same one a first-time setup already produces. The pages below cover
the AWS bring-up and then how to drive a running stack; the driving is identical in both
environments, so it lives on one shared page.

- [Setting up the live service on AWS](aws.md) — every step to stand the service up on AWS, in
  order: S3 buckets and IAM (AWS's storage and permissions services; every AWS acronym is spelled
  out on first use there), build/verify/push the image, the Fargate task, the always-on
  control-plane box, and connecting to the Dagster UI over Tailscale.
- [Connecting to the AWS control plane](connecting.md) — get a laptop onto the OCF tailnet to reach
  an already-running deployment: install Tailscale, view the Dagster UI, and SSH into the always-on
  box.
- [Operating the live service](operations.md) — driving a running stack day to day: the hourly power
  and daily numerical weather prediction (NWP) ingest schedules, reading the asset checks and Sentry
  events when an input degrades, inspecting a forecast, and backfilling a missed partition.
- [Intervention log](intervention-log.md) — the append-only record of every occasion a human had to
  intervene in the running service, and the artefact the [T1.1 operability
  test](../design-philosophy/engineering-hypotheses.md#h1-a-service-that-mostly-runs-itself) is
  scored from.
- [Setting up Sentry telemetry](sentry.md) — point error reporting and the freshness warnings at a
  Sentry project: get a DSN, test it from your laptop, and turn it on in production.
- [Backing up the workstation](backup.md) — the daily timer that snapshots the data, the MLflow
  store, the literature library, Dagster's history, and the credential files to the 18 TB disk, and
  how to restore from a snapshot.
- [Configuration reference](setup.md) — what the storage roots, the derive-from-root convention, and
  the credential settings mean, and which combination each environment uses.
