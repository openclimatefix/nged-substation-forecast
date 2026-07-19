# Requirements

## Phased Rollout

**Version 1** (current focus): 32 time series in NGED's trial area — 16 primary substations, 6 solar PV farms (5 EHV, 1 HV), 3 wind farms, 2 GSPs, 2 BSPs, 1 biofuel generator, 1 BESS, and 1 reciprocating gas generator. All implemented with a single XGBoost model family.

**Version 2** (future): Scale to approximately 2,500 time series (all of NGED's primary substations, BSPs, GSPs, and most customer meters). See [Roadmap](../roadmap/index.md).

## Core Objectives

This is a research project, and our NGED partners treat it as one: the single hard requirement
is that the project gives NGED **new information about forecasting for their assets**. Even a
negative result carries value — if we try our hardest and cannot, say, detect switching events
from power data alone, that is evidence NGED can take to their senior leadership to argue for
investing in technology to extract switching labels from their operational systems. The
objectives below therefore sit on a **priority continuum**, not a must-have/nice-to-have split.

**Highest priority — probabilistic power forecasts under the normal running arrangement
(NRA):**

* Probabilistic, half-hourly, 14-day horizon forecasts updated every 6 hours. Within that
  horizon, users mostly act on forecasts roughly **1 to 10 days ahead**, so skill in that band
  matters most.
    * For the day-ahead forecast: NGED want to look at the forecast at 11am to see the forecast from midnight to
    23:59 on the next day.
* Cover substations (primary, BSP, GSP), metered generators (solar PV, wind, BESS, etc.), and customer meters.

**Everything else exists primarily to improve those forecasts.** Switching-event handling,
effective-capacity tracking, and faulty-meter detection are pursued first and foremost because
they make the NRA forecast better, and it is acceptable for the models to handle these
phenomena *implicitly*. That said, explicit estimates are genuinely wanted where we can produce
them:

* Track the **effective capacity** of metered generators over time (turbine failures, inverter
  faults, PV panel degradation), ignoring NGED-imposed ANM curtailment — including detecting
  misbehaving generators.
* Detect and compensate for **switching events** — where power is diverted from one substation
  to another due to maintenance, changing the local demand signature. (Whether this ships as a
  discrete event table or as continuous switching-state signals is an open question — see
  [the decision point](../roadmap/switching-events.md#the-decision-point-a-feature-based-mainline-vs-the-staged-detector).)
* Automatically detect and flag **faulty metering** (stuck values, physically impossible
  values, missing data).
* An optional **"prevailing conditions" forecast** assembled from the delivered building
  blocks — explicitly lower priority than the NRA forecast. See
  [forecast building blocks](../roadmap/forecast-building-blocks.md).

The five [delivery tables](../roadmap/delivery-tables.md) were specified in our most recent
formal report to NGED, so a change of shape there (such as replacing the discrete
`substation_switching` table with continuous signals) is something to agree with NGED, not to
decide unilaterally.

## Stretch Goals

Further down the same continuum:

* Model and forecast *unmetered* solar PV and wind power on each primary substation by disaggregating net power flow.
* Disaggregate and forecast other distributed energy resources (DERs): EV chargers, heat pumps, price-sensitive batteries.

## ML experimentation at scale

Nearly every objective above is an open research question — improving NRA forecast skill,
detecting switching events, estimating effective capacity, flagging faulty meters,
disaggregating DERs — and we hold far more ideas than we can try at once. That turns
experimentation throughput into an infrastructure requirement in its own right: we need to run
**on the order of hundreds of ML experiments per month**, and the workflow must make each one
as frictionless as possible.

Three properties matter as much as raw throughput:

* **Re-runnability.** We will inevitably find and fix bugs that invalidate earlier results —
  in feature engineering, in evaluation, in the data itself. When that happens we must be able
  to re-run old experiments cheaply and confidently (same configuration, same folds), so that
  results reflect the fixed world rather than a mixture of before and after.
* **A standardised leaderboard.** Every experiment's metrics land in one comparable place,
  computed the same way, so "is this idea better?" is a lookup, not an analysis project. See
  [Metrics & Leaderboard](../roadmap/metrics-and-leaderboard.md).
* **A short, safe path from R&D to production.** Conducting experiments is only half the
  loop: an experiment that wins on the leaderboard must move into the live service as easily
  and as safely as possible. This is why R&D and production share a single unified codebase —
  the exact feature-engineering and model code that won the experiment is what runs in
  production, and promotion is an
  [audited configuration change](../architecture/production-deployment.md#promote-the-champion-via-a-dagster-asset-not-a-script),
  not a rewrite or a port between systems.

This requirement shapes the architecture: it is why the experiment layer is built around
per-(experiment, fold) partitions that can be run — and re-run — individually (see
[ML Orchestration Design](../architecture/ml-orchestration.md)), why R&D and production live
in one repository and one execution path rather than separate codebases, and it was decisive
in choosing Dagster over Airflow (see
[Why Dagster, not Airflow?](../architecture/why-dagster-not-airflow.md)).

## Operating model & handover

NGED confirmed (2026-07-14) that their preference for running Flexpectation business-as-usual
*after* the NIA project is for **NGED to run our code themselves, on NGED's own AWS
infrastructure**. This is a statement of preference, not yet a commitment: NGED still need to
check with their DSO, Cyber, and IT&D teams before giving a concrete answer. Even so, it sets
a standing design requirement for everything we build:

* **The service must be operable day to day by a non-expert at NGED** — every routine action
  reduced to a dashboard check, a button in the Dagster UI, or a runbook. See
  [Handover to NGED](../roadmap/handover.md) for the engineering consequences and the handover
  workstreams.
* **Uptime requirements are deliberately lenient** — recovery is "next business day, via
  runbook", never a 2am page. See [Uptime: lenient by design](#uptime-lenient-by-design) below
  for exactly why an outage costs so little.

The phasing:

1. For the duration of the NIA project, OCF develops **and runs** Flexpectation on OCF's own
   AWS account (unchanged from the existing plan).
2. We will not know whether the service is truly hand-over-able until OCF has run the full v2
   service (~2,500 time series) for a few months. NGED has accepted this.
3. In the last few months of the NIA project, NGED progressively takes control of the service,
   with OCF support. NGED then decides whether to run it themselves.
4. Post-NIA, OCF is no longer on call; NGED handles day-to-day issues. OCF may continue
   developing the software and models, possibly under a retainer — details TBD.

A **hybrid model** may prove beneficial: NGED runs the production service while OCF continues
to develop the code and ML models, and perhaps runs a second service instance of its own (e.g.
adapted for other DNOs, or feeding OCF's substation forecasts into a commercial demand-forecast
product). This makes account-portable infrastructure doubly valuable — see
[Handover to NGED](../roadmap/handover.md#4-infrastructure-as-code-portable-to-ngeds-account).

## Uptime: lenient by design

Flexpectation carries **no hard availability target**. We aim for a highly robust service, but
the requirement when something breaks is recovery "next business day, via runbook" — never a
2am page, and no on-call rota. This leniency is a property of how NGED consumes the forecasts,
not an aspiration; three things bound the damage of an outage:

1. **Every forecast extends 14 days ahead**, refreshed every 6 hours, and users mostly act on
   the forecast roughly 1 to 10 days ahead (see [Core Objectives](#core-objectives)). If the
   service stops producing new forecasts for a few hours — or even a day — the most recent
   forecast remains useful; it merely ages, degrading skill gradually rather than cutting NGED
   off.
2. **Delivery is decoupled from compute.** Forecasts are delivered as Delta tables on S3 (see
   [Forecast Delivery](#forecast-delivery) below), so every previously published forecast stays
   readable even while all of OCF's compute is down — the read path never touches our
   infrastructure.
3. **A legacy fallback exists.** In the worst case, NGED can temporarily fall back to their
   legacy forecasting approach while Flexpectation is fixed.

Missed forecasts are also not lost for evaluation purposes: once the service is back, missed
slots are backfilled in replay mode, reconstructing what would have been forecast at the time —
see [Operating the live service: Backfilling a missed slot](../live_service/operations.md#backfilling-a-missed-slot).

The same properties give the service **built-in maintenance windows**. New forecasts are
produced only once every 6 hours, and NGED reads published forecasts directly from S3 rather
than from any OCF-run service — so the gap between one forecast run and the next is a regular,
roughly six-hour window in which OCF can stop, patch, upgrade, or even rebuild its compute
(most notably the always-on control-plane VM) without NGED noticing. No downtime needs to be
negotiated or announced, and even a maintenance overrun costs only a missed slot, recovered by
the replay-mode backfill above.

This requirement shapes the architecture: it is why a single always-on control-plane VM is an
acceptable single point of failure (see
[Production Deployment — Design](../architecture/production-deployment.md#run-the-dagster-control-plane-continuously-on-one-small-vm)),
and why the primary production alert is a
[missed-check-in alarm](../roadmap/live-service.md#alert-on-absence-the-missed-check-in-alarm)
feeding a runbook rather than any paging or failover machinery.

## Forecast Delivery

OCF delivers forecasts as **Delta Lake tables on AWS S3**, updated every 6 hours. Delta Lake provides ACID transactions, meaning NGED never reads an incomplete forecast. (Why Delta Lake rather than a REST API? See [Forecast Delivery](../architecture/forecast-delivery.md).) The tables are designed as "building blocks" that NGED can combine to construct:

* A **Normal Operation Forecast** (MW or MVA): the [−1, +1] forecast multiplied by the site's nominal capacity. Assumes the grid is in a normal running arrangement with all generators at full unconstrained capacity.
* A **Prevailing Conditions Forecast** (MW or MVA): the [−1, +1] forecast multiplied by the most recently observed effective capacity, reflecting current switching state and any reduced generator capacity.
