# Philosophy

The code in this git repository is the research component of an innovation project with National Grid Electricity Distribution (NGED), a DNO in Great Britain.

## Ultimate Aims & Scope

**Core Objectives:**

* Forecast demand for ~1,300 primary substations.
* Forecast power for ~1,500 customer meters (mostly single-site PV, but also industrial/mixed sites).
* Resolution: Half-hourly, 14 days ahead, probabilistic, updated every 6 hours.
* Handle "switching events" where power is diverted across substations, changing the grid topology.
* Estimate total installed PV and wind capacity for each primary substation.
* Create aggregated forecasts for BSPs (Bulk Supply Points) and GSPs (Grid Supply Points).

**Stretch Goals:**

* Disaggregate and forecast EV charging, heat pumps, batteries, and other price-driven distributed energy resources (DERs).

## Design Philosophy

The architecture prioritizes developer velocity, idempotent re-runs, and strict **Training-Serving Symmetry**.

The primary aim of the code in this repo is to develop novel, ambitious, state-of-the-art ML approaches to forecasting. However, we are also acutely aware that it's vital for the ML research component of the project to be very cognisant of the realities of running the code in production. As such, this repo also implements a "test-harness" production service that - at the very least - will allow us to test ML algorithms in a "production-like" environment.

The dream is to manage the *entire* data pipeline in Dagster (download data, convert data, check data, train ML models, run ML models, perform back-tests, etc.). MLFlow will be used to track each ML experiment. The dream is that re-running a back-test should be as easy as clicking a button in the Dagster UI. Running a new ML experiment should be as simple as pushing the code and clicking a button in Dagster to run. If a new ML model performs better than the model currently in production then swapping the models should also be as simple as possible. Minimise friction for training new models, comparing the models, and pushing models into production.
