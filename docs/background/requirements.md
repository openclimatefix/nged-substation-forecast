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
* **Uptime requirements are deliberately lenient.** Nothing very bad happens if the service
  misses a day: NGED can always read the previous forecasts from S3, and each forecast extends
  14 days ahead. We still aim for a highly robust service, but recovery is "next business day,
  via runbook", never a 2am page.

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

## Forecast Delivery

OCF delivers forecasts as **Delta Lake tables on AWS S3**, updated every 6 hours. Delta Lake provides ACID transactions, meaning NGED never reads an incomplete forecast. (Why Delta Lake rather than a REST API? See [Forecast Delivery](../architecture/forecast-delivery.md).) The tables are designed as "building blocks" that NGED can combine to construct:

* A **Normal Operation Forecast** (MW or MVA): the [−1, +1] forecast multiplied by the site's nominal capacity. Assumes the grid is in a normal running arrangement with all generators at full unconstrained capacity.
* A **Prevailing Conditions Forecast** (MW or MVA): the [−1, +1] forecast multiplied by the most recently observed effective capacity, reflecting current switching state and any reduced generator capacity.
