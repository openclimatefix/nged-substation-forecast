# Requirements

## Phased Rollout

**Version 1** (current focus): 32 time series in NGED's trial area — 16 primary substations, 6 solar PV farms (5 EHV, 1 HV), 3 wind farms, 2 GSPs, 2 BSPs, 1 biofuel generator, 1 BESS, and 1 reciprocating gas generator. All implemented with a single XGBoost model family.

**Version 2** (future): Scale to approximately 2,500 time series (all of NGED's primary substations, BSPs, GSPs, and most customer meters). See [Roadmap](../roadmap/index.md).

## Core Objectives

* Probabilistic, half-hourly, 14-day horizon forecasts updated every 6 hours. Within that
  horizon, users mostly act on forecasts roughly **3 to 10 days ahead**, so skill in that band
  matters most.
* Cover substations (primary, BSP, GSP), metered generators (solar PV, wind, BESS, etc.), and customer meters.
* Automatically detect and compensate for **switching events** — where power is diverted from one substation to another due to maintenance, changing the local demand signature.
* Track the **effective capacity** of metered generators over time (turbine failures, inverter faults, PV panel degradation), ignoring NGED-imposed ANM curtailment.
* Automatically detect and flag **faulty metering** (stuck values, physically impossible values, missing data).

## Stretch Goals

* Model and forecast *unmetered* solar PV and wind power on each primary substation by disaggregating net power flow.
* Disaggregate and forecast other distributed energy resources (DERs): EV chargers, heat pumps, price-sensitive batteries.

## Forecast Delivery

OCF delivers forecasts as **Delta Lake tables on AWS S3**, updated every 6 hours. Delta Lake provides ACID transactions, meaning NGED never reads an incomplete forecast. (Why Delta Lake rather than a REST API? See [Forecast Delivery](../architecture/forecast-delivery.md).) The tables are designed as "building blocks" that NGED can combine to construct:

* A **Normal Operation Forecast** (MW or MVA): the [−1, +1] forecast multiplied by the site's nominal capacity. Assumes the grid is in a normal running arrangement with all generators at full unconstrained capacity.
* A **Prevailing Conditions Forecast** (MW or MVA): the [−1, +1] forecast multiplied by the most recently observed effective capacity, reflecting current switching state and any reduced generator capacity.
