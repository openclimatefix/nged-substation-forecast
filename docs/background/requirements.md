# Requirements

## Phased Rollout

**Version 1** (current focus): 32 time series in NGED's trial area — 16 primary substations, 6 solar PV farms (5 EHV, 1 HV), 3 wind farms, 2 GSPs, 2 BSPs, 1 biofuel generator, 1 BESS, and 1 reciprocating gas generator. All implemented with a single XGBoost model family.

**Version 2** (future): Scale to approximately 2,500 time series (all of NGED's primary substations, BSPs, GSPs, and most customer meters). See [Roadmap](../roadmap/index.md).

## Core Objectives

* Probabilistic, half-hourly, 14-day horizon forecasts updated every 6 hours. Within that
  horizon, users mostly act on forecasts roughly **1 to 10 days ahead**, so skill in that band
  matters most.
    * For the day-ahead forecast: NGED want to look at the forecast at 11am to see the forecast from midnight to
    23:59 on the next day.
* Cover substations (primary, BSP, GSP), metered generators (solar PV, wind, BESS, etc.), and customer meters.
* Automatically detect and compensate for **switching events** — where power is diverted from one substation to another due to maintenance, changing the local demand signature.
* Track the **effective capacity** of metered generators over time (turbine failures, inverter faults, PV panel degradation), ignoring NGED-imposed ANM curtailment.
* Automatically detect and flag **faulty metering** (stuck values, physically impossible values, missing data).

## Stretch Goals

* Model and forecast *unmetered* solar PV and wind power on each primary substation by disaggregating net power flow.
* Disaggregate and forecast other distributed energy resources (DERs): EV chargers, heat pumps, price-sensitive batteries.

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
