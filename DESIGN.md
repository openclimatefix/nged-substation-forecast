The code in this git repository is the research component of an innovation project with National
Grid Electricity Distribution (NGED), a DNO in Great Britain.

# Ultimate aims

- Forecast net demand, disaggregated into gross demand, solar, and wind for each of NGED's GSPs,
BSPs, and primary substations.
- These forecasts will be at half-hourly resolution, and will extend out to 14 days ahead, and will
  be probabilistic, and will be updated multiple times per day.
- Disaggregate gross demand, solar power, and wind power from NGED's primary substations.
- Handle "switching events" where power is diverted across substations. These switching events can
range in duration from minutes to months. Network switching might be automatic, in response to
faults (in which case we might see a cascade of switching events in quick succession). Or they might
be planned (for maintenance).
- The project is split into two "tracks": An ML research track, and a live service.


## Stretch goals

- Disaggregate and forecast EV charging, heatpumps, batteries, and other price-driven distributed
energy resources.

# Data sources

## Power data from NGED

- NGED has about 1,500 primary substations.
- Their CKAN-powered data portal ["Connected Data"](https://connecteddata.nationalgrid.co.uk/)
provides some historical power data for substations, and some wiring diagrams.
- NGED will provide live power data. Although that won't be ready for a few months. So, to start
with, the first version of our production power forecast won't be able to use recent power data.

## Switching events

- NGED can provide spreadsheets with _some_ labels for switching events (but not all). Each row
represents the change in state of a switch (which changes NGED's grid topology). Each row has a
timestamp (accurate to the second), and whether a switch was opened or closed, and a _rough_
geographical area. But this spreadsheet doesn't tell us exactly which switch was changed.

## Weather forecasts

- Let's try pulling historical NWP datasets from Dynamical.org (which provide a range of NWP models
  as Zarrs, and update these Zarrs about 4 times per day).

## Weather observations & re-analyses

- CERRA (an EU-specific weather reanalysis dataset)
- CM-SAF (a satellite-derived irradiance dataset)


# Software engineering & open source

- We want this code to be as modular as possible.
- Individual components should be `pip` installable (although perhaps from `git` rather than from
`pypi`).
- Use Python 3.14
- All function signatures must use expressive type hints for all arguments and return types.
- Write as little code as possible: Re-use existing tools wherever possible.
- All the code will be open-source. I'd like external researchers and forecasting companies to be
able to use modules from this repo as easily as possible. Developer experience is really important.


# ML Ops

The primary aim of the code in this repo is to develop novel, ambitious, state-of-the-art ML approaches to
forecasting. However, we are also acutely aware that it's vital for the ML research component of the
project to be very cognisant of the realities of running the code in production. As such, this repo
will also implement a "test-harness" production service that - at the very least - will allow me
to test ML algorithms in a "production-like" environment. And, depending on how the work goes, this
"test-harness production service" could also form the basis of the live service.

The dream is to manage the *entire* pipeline in Dagster (download data, convert data, check data,
train ML models, run ML models, perform back-tests, etc.). MLFlow will be used to track each ML
experiment. The dream is that re-running a back-test should be as easy as clicking a button in the
Dagster UI. Running a new ML experiment should be as simple as pushing the code and clicking a
button in Dagster to run. If a new ML model performs better than the model currently in production
then swapping the models should also be as simple as possible. Minimise friction for training new
models, comparing the models, and pushing models into production.


# Software tools

- This git repo is a `uv` workspace.
- `Dagster` to manage the *entire* pipeline (from data ingest to training ML models to running ML
models in production).
- `MLFlow` to track ML experiments.
- `PyTorch` with `PyTorch geometric`
- `Polars` for working with tabular data.
- `Xarray` and `Zarr` for pulling NWP data from Dynamical.org.
- `Patito` for defining data contracts (or another tool if more appropriate).
- `Altair` for visualisation.
- `Marimo` for interactive data exploration and building web apps for data visualisation.
- `Sentry.io` for observability.
- `obstore` for IO.


# Data engineering

We will use rigorous data contracts. The code will download and save the raw data, and then convert
it to a well-documented schema, whilst also rigorously testing for conformance to that schema, and
testing for conformance to some statistical tests (e.g. to test for insanely large values).

Let's make simple things simple. Let's use tabular data wherever possible, stored in partitioned
parquets.

Per data type, let's use the same "converted" dataset for training, back-testing, and inference.
We'd append to this dataset in production. (Using separate datasets for training and inference is a
big source of pain when moving models from research into production.)

To make this research as easy as possible to reproduce, I want to make it super-easy for people to
run parts of the data pipeline on their own machines.

# "Test harness" production service

Output as Parquet files. No need for a full API yet. Assume NGED can consume the Parquet files from
cloud object storage. Trigger from Dagster. Use Sentry to track health of the service.


# Compute infrastructure

- I (Jack) will run ML experiments on my workstation (Ubuntu, with an nvidia A6000 GPU).
- Dagster and MLFlow will run on a small cloud VM.
- If needed, we could consider using Modal (orchestrated by Dagster) to run "compute-heavy" jobs
like ML training or back-tests.


# Plans for the ML research

## Overview

We will build differentiable physics modules that encode the physics in equations written in
pytorch. One aim is to use these differentiable physics models to infer the physically-meaningful
parameters of the model. Don't use any maths above A-Level standard in the physics models.

We will combine these differentiable physics models with "black box" encoders (neural nets).

We like differentiable physics because it makes the models "explainable" and because it allows us to
have different parts of the ML system be responsible for different parts of the physics. e.g. we
have a differentiable physics model which learns the fiddly details of each PV system, which is fed
by a weather encoder that learns how to interpret each NWP, and can be trained across all PV
systems.

## Differentiable physics

### Solar PV
For solar PV, we'd write code to capture the minimal set of equations describing how sunlight is turned into electricity. We'd like to infer
the PV panel tilt and azimuth, and the ratio of DC capacity to AC capacity (because it's
increasingly common for PV developers to install, say, 5 MW of PV panels with a 3 MW inverter).
Further down the line, we'd like to infer shading (represented as a 2D array representing the PV
system's "view" of the sky). We can re-implement equations from pvlib in PyTorch. We'd like to
release a Python package with this minimal set of equations in PyTorch, as a package called
"solar-torch".

To handle fleets of solar PV systems (which might have different parameters), we could use, models
for, say, 3 PV systems (with different tilts and azimuths), and learn the ratio of these, and learn
a scaling parameter.

Use CM-SAF when inferring PV parameters on historical PV power data. Combined with temperature data
from CERRA and/or the nearest weather station.

#### Stretch goals for solar PV forecasting research

- Forecast for individual PV sites.
- Run on all of PVOutput.org and/or Open Climate Fix's UK_PV dataset to infer PV panel tilt,
azimuth, AC:DC ratio, and shading, for each PV system in the dataset and publish as an open dataset.
- Once models are trained for each PV system, infer irradiance from PV power. Do this for thousands
  of PV systems in GB to infer an "irradiance map" (a 2D array), which we advect using wind speed &
  direction from NWP at cloud altitude and/or by running optical flow on 5-minutely satellite data.
  Going even more wild: The irradiance map would combine information from PV systems, geostatonary
satellite data, ground-based weather stations, and polar-orbiting satellite data. And publish this
as a live "irradiance nowcasting" gridded dataset.

### Wind
Follow a similar pattern to the differentiable physics models.

### Gross demand
TODO

## Graph neural network
Use a GNN to model the grid. Try to stick close to the physics.

For example, each primary substation will be represented by a node in the GNN. It'll be connected to
a PV fleet node (to model the unmetered PV), and a wind fleet node (for the unmetered wind), and a gross demand node, and nodes for any metered generation. The edges from the metered generation nodes to their substations will also capture curtailment. The curtailment "gate" will be driven by both ends of the graph edge: the substation (to capture how congested the grid is) and the generation node.

The GNN will also represent the electrical connection between substations: Both the hierarchy GSP ->
BSP -> primary, and the "mesh" horizontal connections.

Sum up the forecasts for primary substations up to BSPs. And sum up BSPs to GSPs. Don't force
a simple sum (because there will be line losses etc.). Instead use the ML loss function to encourage
the BSP power to be the (rough) sum of its primary substations, and the same for GSPs.

## Encoders

### Weather encoder

TODO(Jack)


## Probabilistic

Let's start with purely deterministic models. And, later, we'd like to learn _distributions_ for
each physical parameter (e.g. a distribution of the PV panel tilt).

In terms of the output of the forecast, let's start with a deterministic output. And then move to
outputting a distribution per forwards pass (e.g. a mixture of Gaussians). And finally move to
consuming ensemble NWPs to produce an ensemble of Gaussians.

## Multi-sequence alignment

TODO(Jack)

# Directory layout

This is up for discussion.

nged-substation-forecast/
├── pyproject.toml  ← uv workspace config
│
├── packages/
│   ├── contracts/
│   │   │ # Define the _shape_ and _interfaces_ of the data.
│   │   │ # e.g. "What's a valid forecast?",
│   │   │ #      "What should the forecast model return?"
│   │   │
│   │   ├── pyproject.toml ← Deps: patito, polars, etc.
│   │   ├── tests/ ← tests just for the contracts package
│   │   └── src/contracts/
│   │       ├── __init__.py
│   │       ├── data_schemas.py
│   │       ├── base_sklearn_model.py
│   │       └── base_lightning_model.py
│   │
│   ├── data_nged/
│   │   │  # The *logic* for downloading & prep. No orchestration here.
│   │   │  # All data saved to disk must conform to the data contracts.
│   │   │  # Orchestration lives in the main app. No LightningDataModules
│   │   │  # here (they belong in the encoder packages). Don't write any
│   │   │  # logic here that saves to disk. Dagster wants to control
│   │   │  # *saving*. And Dagster expects to be able to give the data
│   │   │  # loading functions a time *window* to download.
│   │   │
│   │   ├── pyproject.toml ← Deps: contracts, polars, requests, etc.
│   │   ├── tests/
│   │   └── src/data_nged/
│   │       ├── __init__.py
│   │       ├── ckan_client.py
│   │       └── live_primary_data.py
│   │
│   ├── data_cerra/
│   │   │  # CERRA weather re-analysis data.
│   │   ├── pyproject.toml ← Deps: xarray, rioxarray, polars, etc.
│   │   ├── tests/
│   │   └── src/data_cerra/
│   │       ├── __init__.py
│   │       └── download.py
│   │
│   ├── data_cm_saf/
│   │   │  # CM-SAF satellite-derived irradiance data.
│   │   ├── pyproject.toml
│   │   ├── tests/
│   │   └── src/data_cm_saf/
│   │       ├── __init__.py
│   │       └── download.py
│   │
│   ├── data_dynamical/
│   │   │  # NWP data from Dynamical.org.
│   │   ├── pyproject.toml
│   │   ├── tests/
│   │   └── src/data_dynamical/
│   │       ├── __init__.py
│   │       └── client.py
│   │
│   ├── plotting/
│   │   │  # Lightweight, scriptable plotting functions that will primarily
│   │   │  # be used during ML training, and sent to MLFlow. Can also be
│   │   │  # used in manual data analysis.
│   │   │
│   │   ├── pyproject.toml <-- Deps: polars, altair, etc.
│   │   ├── tests/
│   │   └── src/plotting/
│   │       ├── __init__.py
│   │       ├── training_plots.py <-- e.g., plot_loss_curve()
│   │       └── forecast_plots.py <-- e.g., plot_prediction_vs_actual()
│   │
│   ├── notebooks/
│   │   │  # Manual, interactive data analysis using Marimo notebooks.
│   │   │
│   │   ├── pyproject.toml <-- Deps: marimo, altair, etc.
│   │   ├── plot_nged_switching_events_with_weather.py
│   │   └── utils/ <- non-production code that's shared across notebooks.
│   │       ├── __init__.py
│   │       └── …
│   │
│   ├── performance_evaluation/
│   │   │  # Scriptable evaluation. e.g. takes a whole year of
│   │   │  # forecast & ground truth & produces metrics & plots.
│   │   │  # Also evaluate performance on other tasks, e.g. disaggregation.
│   │   │  # Assumes all incoming data adheres to the data contracts.
│   │   │  # Can be run manually as scripts. Or scripted from Dagster app.
│   │   │
│   │   ├── pyproject.toml
│   │   ├── tests/
│   │   ├── scripts/
│   │   └── src/performance_evaluation/
│   │       ├── __init__.py
│   │       ├── eval_forecast_backtest.py
│   │       ├── eval_der_capacity_estimates.py
│   │       ├── eval_disaggregation.py
│   │       └── …
│   │
│   ├── loss_functions/
│   │   │  # Lightweight, high-performance loss funcs. Used in training loop.
│   │   │
│   │   ├── pyproject.toml <- Deps: torch, torchmetrics.
│   │   │    # Crucially, do NOT depend on polars.
│   │   │    # These metrics must be torch-native (and GPU-native).
│   │   │
│   │   ├── tests/
│   │   └── src/loss_functions/
│   │       ├── __init__.py
│   │       ├── gaussian_mixture_model_neg_log_likelihood.py
│   │       └── quantile_loss.py
│   │
│   ├── weather_encoder/
│   │   ├── pyproject.toml ← Deps: torch, lightning, polars.
│   │   │   # Does NOT depend on MLflow.
│   │   │   # Contains the training *logic* specific to this encoder,
│   │   │   # implemented as an instance of LightningModule.
│   │   │   # But the actual training loops are orchestrated by the main app.
│   │   │   # Pre-trained encoders can be installed on their own, and the
│   │   │   # weights & config can be loaded from the GitHub release assets.
│   │   │
│   │   ├── tests/
│   │   └── src/weather_encoder/
│   │       ├── __init__.py
│   │       ├── model.py <- implements our BaseLightningModel
│   │       │    # contract (including the predict_step that returns
│   │       │    # the unified schema)
│   │       ├── data.py <- implements P.LightningDataModule.
│   │       └── load_pretrained_weights_and_config.py
│   │
│   └── time_encoder/
│   │   ├── pyproject.toml ← Deps: torch, lightning, polars.
│   │   ├── tests/
│   │   └── …
│   │
│   ├── tft_forecaster/
│   │   │  # Implements the temporal fusion transformer forecaster.
│   │   │
│   │   ├── pyproject.toml <- Deps: torch, lightning, polars, contracts, etc.
│   │   │    # (tft_forecasts depends on contracts so it can implement BaseLightningModule).
│   │   │
│   │   ├── tests/
│   │   └── src/tft_forecaster/
│   │       ├── __init__.py
│   │       ├── model.py
│   │       ├── data.py
│   │       └── load_pretrained_weights_and_config.py
│   │
│   ├── xgboost_forecaster/
│   │   │  # Implements the XGBoost forecaster.
│   │   │
│   │   ├── pyproject.toml <- Deps: xgboost, contracts
│   │   │    # (deps on contracts so it can implement BaseSKLearnModule).
│   │   │
│   │   ├── tests/
│   │   └── src/xgboost_forecaster/
│   │       ├── __init__.py
│   │       ├── model.py <- implements our BaseSKLearnModel contract
│   │       │    # (including the predict() that returns a pl.DataFrame that
│   │       │    # conforms to our unified schema).
│   │       ├── data.py <- loads data into RAM for XGBoost training
│   │       └── load_pretrained_weights_and_config.py
│   │
│   ├── solar_torch/
│   │   │  # Implements solar PV differentiable physics.
│   │   │
│   │   ├── pyproject.toml <- Deps: pytorch, etc.  Dev deps: pvlib (to create test data)
│   │   │
│   │   ├── tests/
│   │   └── src/solar_torch/
│   │       ├── __init__.py
│   │       └── ...
│   │
│   ├── graph_neural_net/
│   │   │  # Implements the graph neural net forecaster.
│   │   │
│   │   ├── pyproject.toml <- Deps: torch, contracts, torch_geometric, etc.
│   │   │   # Does NOT depend on the encoders. Instead accepts nn.Modules
│   │   │
│   │   │
│   │   ├── tests/
│   │   └── src/st_gnn_forecaster/
│   │       ├── __init__.py
│   │       ├── model.py
│   │       ├── data.py <- or maybe this should live in the app package
│   │       └── load_pretrained_weights_and_config.py
│
├── tests/ ← just tests for the nged_substation_forecast app
│
└── src/
    |    # The main application. Orchestrates all packages above.
    |    # Implements all training & validation loops.
    |    # Orchestrates data download, data prep, and re-training.
    |
    └── nged_substation_forecast/ ← Deps: dagster, mlflow, sub-packages
        ├── dagster_assets.py
        └── …


# Plan

## Stage 1: Download essential data & plot

- [x]  Get Dagster running locally. We'll have a separate python package in our uv workspace for Dagster.
  Perhaps use the "root" python package for Dagster?
- [ ]  Teach Dagster how to:
    - [x] download Live NGED substation power data from NGED's CKAN API
    - [x] locations of substations
    - [ ] Historical substation data from NGED.
- [ ] Visualise NGED power data, perhaps with a map of substations. The dream: A web app
(implemented in Marimo & Altair) with a small map of NGED substations. The user can drag the map
around. Click on a substation in the map to show 7 days of time series data in a line plot on the
right. The map should take up about 1/4 of the width of the screen, and the time series should take
up 3/4 of the width. Show MW (or MVA if MW isn't available) from the substation.
- [ ] Teach Dagster:
    - [ ] CERRA (or maybe save this until later)
    - [ ] ECMWF ens from Dynamical.org
- [ ] Update the Marimo visualisation: In separate subplot below the substation power, show some weather data
(from CERRA: temperature, wind speed, rain; from CM-SAF: irradiance; although maybe start with just
CERRA data). And show weather _predictions_ for the same variables and the same valid period. Have
buttons to select which NWP initialisation to show (only list options that are valid for the time
period). Show ensemble NWP predictions as multiple lines on the same plot. Later in development,
this plot will also show our predictions for demand.
- [ ] Build a simple XGBoost model to predict net demand per primary
    - [ ] Use MLFlow with Dagster.
- [ ] Add our demand predictions to the app.
- [ ] Add a `Sensor` to Dagster to detect new data, and trigger the pipeline.
- [ ] Get the app running live on a cloud VM, to show NGED & OCF.

## Stage 2: Simple PV & wind forecasts
- [ ] Check if NGED already publish total installed capacity of PV and wind per primary.
- [ ] If not: teach Dagster to download the Embedded Capacity Register (ECR) from NGED, and store in
Parquet. Use the ECR to get the installed PV & wind capacity per primary
- [ ] Plot the locations of these sites on the Marimo app.
- [ ] Stretch goal: When you hover over a DER, draw a line to its substation. If you hover over a
substation, draw lines to all its DERs. (Or maybe just draw the region polygons for each
substation?)
- [ ] Create simple PV forecasts using PVLib.
- [ ] Display PV forecasts for each ECR site on the Marimo app. And aggregate by primary.
- [ ] Implement wind power forecasts.

## Stage 3: More data!
- [ ] Teach Dagster to download and process:
    - [ ] NGED generation data?
    - [ ] CM-SAF
- [ ] Visualise in Marimo app

## Stage 4: ML research
- [ ] Implement MVP `solar-torch`. Test on synthetic PV data generated by PVLib driven by CM-SAF.
