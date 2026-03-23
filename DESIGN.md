**System Design: NGED Probabilistic Forecasting Platform**

The code in this git repository is the research component of an innovation project with National Grid Electricity Distribution (NGED), a DNO in Great Britain.

## **1\. Ultimate Aims & Scope**

**Core Objectives:**

* Forecast demand for \~1,300 primary substations.
* Forecast power for \~1,500 customer meters (mostly single-site PV, but also industrial/mixed sites).
* Resolution: Half-hourly, 14 days ahead, probabilistic, updated every 6 hours.
* Handle "switching events" where power is diverted across substations, changing the grid topology.
* Estimate total installed PV and wind capacity for each primary substation.
* Create aggregated forecasts for BSPs (Bulk Supply Points) and GSPs (Grid Supply Points).

**Stretch Goals:**

* Disaggregate and forecast EV charging, heat pumps, batteries, and other price-driven distributed energy resources (DERs).

## **2\. Data Sources & Feature Engineering**

### **Power Data & Topology**

* **Historical & Live SCADA:** Sourced from NGED's CKAN-powered [Connected Data portal](https://connecteddata.nationalgrid.co.uk/) and live S3 buckets. (Around mid-2026, the data transfer system will change back to a public NGED data portal (with auth to get customer data)).
* **Switching Events:** NGED provides spreadsheets with partial labels (timestamps, open/close status, rough areas). We will synthetically augment and verify these using statistical residual detection. NGED will provide full switching event labels for the ~50 substations in the trial area.
* **Geospatial Indexing:** All latitude/longitude coordinates (substations, customer meters, weather pixels) will be converted to **H3 (Uber's Hexagonal Hierarchical Spatial Index)**. This turns expensive nearest-neighbor geospatial joins into blazing-fast string matches (ON h3\_index).

### **Weather Data**

* **Forecasts:** ECMWF ENS 0.25 degree from Dynamical.org (Zarrs updated at 00Z).
* **Observations/Reanalysis:**
  * CERRA (EU-specific weather reanalysis dataset).
  * CM-SAF (satellite-derived irradiance dataset).

## **3\. Software Engineering & MLOps Stack**

The architecture prioritizes developer velocity, idempotent re-runs, and strict **Training-Serving Symmetry**.

The primary aim of the code in this repo is to develop novel, ambitious, state-of-the-art ML approaches to
forecasting. However, we are also acutely aware that it's vital for the ML research component of the
project to be very cognisant of the realities of running the code in production. As such, this repo
will also implement a "test-harness" production service that - at the very least - will allow us
to test ML algorithms in a "production-like" environment. And, depending on how the work goes, this
"test-harness production service" could also form the basis of the live service.

The dream is to manage the *entire* data pipeline in Dagster (download data, convert data, check data,
train ML models, run ML models, perform back-tests, etc.). MLFlow will be used to track each ML
experiment. The dream is that re-running a back-test should be as easy as clicking a button in the
Dagster UI. Running a new ML experiment should be as simple as pushing the code and clicking a
button in Dagster to run. If a new ML model performs better than the model currently in production
then swapping the models should also be as simple as possible. Minimise friction for training new
models, comparing the models, and pushing models into production.

* **Environment & Modularity:** uv workspace (Monorepo). Python 3.14. Individual components must be pip-installable with expressive type hints.
* **Data Processing:** **Polars**. Chosen for extreme speed and its native `join_asof` functionality to guarantee no future-data leakage during feature engineering.
* **Storage:** **Delta Lake** on cloud object storage for the power data and forecasts. Upgraded from raw Parquet to provide ACID transactions, time-travel, and efficient partitioning (by year/month) for K-fold cross-validation. The weather data will use raw Parquet will well-defined naming conventions (because Delta Lake doesn't support `uint8`).
* **Orchestration:** **Dagster**. Manages the pipeline via Software-Defined Assets (SDAs). Partitioned by NWP init time, not substation, allowing models to train globally. Dagster will be responsible for detecting and handling bad data.
* **Configuration Management:** **Hydra** combined with `pydantic-settings`. Dagster passes string overrides (e.g., model=xgboost\_global) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
**System/Infrastructure Config**: Managed via environment variables and validated at runtime using `pydantic-settings`. Injected into Dagster as strongly-typed Resources (ensuring paths, URIs, and credentials fail-fast).
* **ML/Hyperparameter Config**: Managed via Meta's Hydra. Dagster passes simple string overrides (e.g., `model=xgboost`) to trigger Hydra's Config Groups, swapping massive architectural parameter trees. The resolved YAML is logged to MLflow.
* **Experiment Tracking:** **MLflow**.
* **Visualisation:** Altair for plotting, Marimo for interactive data exploration and web apps.
- **Cross-Validation**: Implemented using Dagster Dynamic Mapping. A generator Op calculates temporal expanding/sliding window boundaries, maps parallel training Ops across the cluster, and an aggregator Op calculates the Continuous Ranked Probability Score (CRPS).
- **Production Inference**: A Dagster Asset (`champion_model`) queries MLflow for the production tag, downloads the `pyfunc`, and serves the downstream `primary_substation_forecasts` asset. This asset is completely agnostic to whether it's running XGBoost, differentiable physics, or whatever. Dagster orchestrates these isolated environments via Dagster Pipes (to spin up a subprocess with the `uv` environment for that specific type of ML architecture).
- **Defining the ML models**: The major categories of model (XGBoost, ARIMA, GNN, TFT,
differentiable phyics, etc) will have their own Python package within the `uv workspace`.

### The Universal Model Interface (MLflow PyFunc)
To decouple the Dagster data pipeline from the ML code, all models are saved as custom MLflow `pyfunc` artifacts.

**The Adapter Pattern**: The `pyfunc` wrapper encapsulates the model weights and all translation logic.

- _Input translation_: Transforms the canonical Polars DataFrame into the required model shape (3D tensors, spatial GNN graphs, flat XGBoost matrices).
- _Output translation_: Converts native model outputs into the strict target quantile schema.

When you define your custom `PythonModel` class in MLflow, it primarily requires two methods. Here is what happens inside them:

**Phase A: `load_context` (The Setup)**
This method runs exactly once when Dagster spins up the inference environment.

For ARIMA: It simply loads the statistical model object from a pickle file.

For PyTorch: It loads your Hydra config to dynamically instantiate the exact neural network architecture (TFT, Graph NN, etc.), and then loads the PyTorch state dictionary (the weights) into that architecture.

**Phase B: predict (The Translation)**
This is the workhorse. Your Dagster pipeline passes in a standard Polars DataFrame containing the recent grid and weather data.

- **Native Inference**: The wrapper passes the data to the underlying model in whatever format it expects (e.g., converting the DataFrame to PyTorch Tensors).
- **The Translation Step**: This is the crucial part. The wrapper catches the native output and forces it into the quantile contract.
    - If the model was ARIMA, the wrapper takes the mean and variance, uses the inverse cumulative distribution function, and calculates the required quantiles.
    - If the model was Differentiable Physics outputting 500 samples, the wrapper calculates the empirical quantiles from those samples.
    - If it was a TFT, it just maps the output nodes to the right columns.

**The Return**: Finally, the predict method returns a strictly typed DataFrame with your required schema.

By embedding this translation logic inside the MLflow model artifact itself, you guarantee that the model is entirely self-contained. The production pipeline remains blissfully ignorant of the complex physics or statistics happening under the hood.

### Data Contracts
- **Input Contract** (`canonical_grid_features`): A single Dagster Asset handles all domain logic (joins, NWP alignment, timezones) and outputs a canonical Polars DataFrame.
- **Output Contract**: A single "Long Format" Polars DataFrame representing 9 specific quantiles (P1, P5, P10, P25, P50, P75, P90, P95, P99) to capture hardware and commercial risk.
    - **Schema**: timestamp, substation_id, horizon_step, ensemble_member, q1, q5, ..., q99
    - **Ensembles**: Individual traces are saved under distinct `ensemble_member` IDs (e.g., '1', '2'), while mathematically pooled quantiles are saved under `ensemble_member = 0`. Production systems filter exclusively for `0` (assuming they don't want the ensembles of power forecasts).

### Compute infrastructure

- I (Jack) will run ML experiments on my workstation (Ubuntu, with an nvidia A6000 GPU).
- Dagster and MLFlow will run on AWS App Runner for the "production test harness".
- If needed, we could consider using Modal (orchestrated by Dagster) to run "compute-heavy" jobs
like ML training or back-tests.

## **4\. Phased Implementation Plan**

### **Phase 1: Infrastructure Plumbing & The "Naive" XGBoost Baseline**

*Before attempting to detect complex switching events, we must prove the data engineering, MLflow tracking, and Dagster orchestration work end-to-end in a production environment.*

* **Ingestion:** Download NGED data and ECMWF (done). Convert NGED data to Delta Lake and index via H3 (done). Convert ECMWF to Parquet and index via H3 (done).
* **The "Naive" Model:** Build a very simple XGBoost net demand forecast for the ~50 trial sites (including substations and customer meters). Wrap this model in an MLflow custom `pyfunc` so the inference code can be completely agnostic to the ML model.
* **Purpose:** This model will intentionally ignore switching events. The forecast will be flawed, but it serves as an integration test for our infrastructure: validating live data reading, Polars asof joins, PyFunc model serving, and Marimo visualisations.
* **Additional features**: Implement simple baseline algorithms to compare against (e.g. ARIMA &
7-day lagged persistence). Create leaderboard of forecasts. Define a standard data contract for all
power forecasts.

**Target completion date**: End of May 2026.

### **Phase 2: Topology Switching Detection (Ground Truth Pipeline)**

*Isolating and labeling historical switching events to uncover the true network physics.*

* **Exogenous Baseline:** Train K-Fold Out-of-Fold (OOF) XGBoost models using *only* weather and time features (strictly no lagged power features, which mask step-changes). Use Huber/MAE loss.
* **Residual Detection:** Apply CUSUM or Rolling Difference filters to the baseline residuals ($r\_{i,t} \= y\_{i,t} \- \\hat{y}\_{i,t}$) to flag topological step-changes.
* **Spatial Verification (Super-Node Test):** Validate switches by applying Kirchhoff's laws to neighbors. Calculate the Variance Ratio for a node $i$ and neighbor subset $\\mathcal{S}$:
  $$\\text{VR} \= \\frac{\\text{Var}(r\_i \+ \\sum\_{m \\in \\mathcal{S}} r\_m)}{\\text{Var}(r\_i) \+ \\sum\_{m \\in \\mathcal{S}} \\text{Var}(r\_m)}$$
  If $\\text{VR} \\ll 1$, power was conserved, mathematically confirming the switch.

**Target completion date**: End of June 2026.

### **Phase 3: The MVP Forecasting Pipeline (Advanced XGBoost)**

*With clean topology labels, we build the robust statistical baseline.*

* **Substation NRA Forecast (Global Model):** Train one global XGBoost model. Use the switching labels to estimate the transfer magnitude ($\\Delta P$) and mathematically "correct" the historical SCADA data. Feed these *Virtual Normal Running Arrangement (NRA)* lags into the model.
* **Customer Meter Forecasts (Clustered/Local Models):** Train models clustered by asset type. Discard exact point-lags (which cause "phantom echoes" of random machine outages). Replace with **State-Lags**: 2-hour rolling max/min, 3-hour windowed mean, and weekly rolling medians.

**Target completion date**: End of July 2026.

### **Phase 4: Grey-Box Physics-Informed Neural Network (PyTorch)**

*Transitioning from black-box trees to explainable physics to explicitly model Behind-The-Meter (BTM) assets.*

We will build differentiable physics modules that implement the physics in equations in
pytorch. One aim is to use these differentiable physics models to infer the physically-meaningful
parameters of the model. Don't use any maths above A-Level standard in the physics models. The
parameters will be probability distributions, using sensible priors.

* **Explicit Disaggregation:** The model explicitly outputs unmetered solar and unmetered wind alongside gross demand:
  $$P\_{\\text{net}} \= P\_{\\text{gross\\\_demand}} \- P\_{\\text{solar}} \- P\_{\\text{wind}}$$
* **Asset Discovery:** By feeding CM-SAF irradiance to a differentiable solar-torch module, physical parameters (installed capacity, tilt, azimuth, DC:AC capacity, shading) become learnable via gradient descent.
* **Natively Handling MVA Telemetry:** Older substations report absolute apparent power (MVA), hiding reverse power flows. The PyTorch model handles this natively via its forward pass:
  $$\\hat{S} \= \\sqrt{P\_{\\text{net}}^2 \+ Q\_{\\text{base}}^2}$$
  Where baseline reactive power ($Q\_{\\text{base}}$) is learnable, allowing the real power ($P\_{\\text{net}}$) to smoothly cross zero.
* **Identifiability Constraints:** The gross demand module is heavily regularized to prevent the optimizer from confusing a massive noon-time solar export with a drop in human demand.

**Target completion date**: End of October 2026.

#### Solar PV differentiable physics

For solar PV, we'd write code to capture the minimal set of equations describing how sunlight is turned into electricity. We'd like to infer
the PV panel tilt and azimuth, and the ratio of DC capacity to AC capacity (because it's
increasingly common for PV developers to install, say, 5 MW of PV panels with a 3 MW inverter).
Further down the line, we'd like to infer shading (represented as a 2D array representing the PV
system's "view" of the sky). We can re-implement equations from pvlib in PyTorch.

We may release a Python package with this minimal set of equations in PyTorch, as a package called
"solar-torch".

To handle fleets of solar PV systems (which might have different parameters), we could use, models
for, say, 3 PV systems (with different tilts and azimuths), and learn the ratio of these, and learn
a scaling parameter.

Use CM-SAF when inferring PV parameters on historical PV power data. Combined with temperature data
from CERRA and/or the nearest weather station.

##### Stretch goals for solar PV forecasting research

- Forecast for individual PV sites.
- Run on all of PVOutput.org and/or Open Climate Fix's UK_PV dataset to infer PV panel tilt,
azimuth, AC:DC ratio, and shading, for each PV system in the dataset and publish as an open dataset.
- Once models are trained for each PV system, infer irradiance from PV power. Do this for thousands
  of PV systems in GB to infer an "irradiance map" (a 2D array), which we advect using wind speed &
  direction from NWP at cloud altitude and/or by running optical flow on 5-minutely satellite data.
- Going even more wild: The irradiance map would combine information from PV systems, geostatonary
satellite data, ground-based weather stations, and polar-orbiting satellite data. And publish this
as a live "irradiance nowcasting" gridded dataset.


### **Phase 5: Spatial Graph Neural Network**

*Capturing highly non-linear, cross-network interactions during extreme events.*

For example, each primary substation will be represented by a node in the GNN. It'll be connected to
a PV fleet node (to model the unmetered PV), and a wind fleet node (for the unmetered wind), and a gross demand node, and nodes for any metered generation. The edges from the metered generation nodes to their substations will also capture curtailment. The curtailment "gate" will be driven by both ends of the graph edge: the substation (to capture how congested the grid is) and the generation node.

The GNN will also represent the electrical connection between substations: Both the hierarchy GSP ->
BSP -> primary, and the "mesh" horizontal connections.

Sum up the forecasts for primary substations up to BSPs. And sum up BSPs to GSPs. Don't force
a simple sum (because there will be line losses etc.). Instead use the ML loss function to encourage
the BSP power to be the (rough) sum of its primary substations, and the same for GSPs.

* **Dynamic Adjacency Matrix:** The GNN cannot implicitly learn grid topology. The live Switching Detector (from Phase 2\) acts as a permanent interceptor, continuously updating the adjacency matrix ($A\_t$) before inference.
* This ensures the GNN's message-passing layers route data along the actual physical paths of the switched grid, preventing exploding gradients in the physical loss functions.

**Target completion date**: End of December 2026.

### **Phase 6: Further research**

* Continually improve the current best model
* Experiment with forecasting for BSPs and GSPs. Compare top-down forecasts against bottom-up
forecasts.
* Experiment with pre-trained encoders (e.g. a weather encoder).
* Multi-sequence alignment with axial attention & automatic selection of "similar days".
* Disaggregate other DERs (e.g. EV chargers, or batteries).

## **5\. Evaluation Metrics**

Standard metrics fail on highly stochastic customer meters and counterfactual NRA targets.

* **Customer Meters:** Use **wMAPE** (Weighted MAPE) to handle sites that drop to zero generation, or **nMAE** (Normalized MAE, scaled by installed capacity).
* **Substations (NRA Evaluation):** \* *Masked Evaluation:* Only score the model during verified, normal un-switched periods.
  * *The Super-Node Invariance Test:* If nodes $i$ and $j$ are currently switched, evaluate the error of their sum: $|(\\hat{y}\_i^{\\text{NRA}} \+ \\hat{y}\_j^{\\text{NRA}}) \- (y\_i^{\\text{measured}} \+ y\_j^{\\text{measured}})|$.

## **6\. Directory Layout**

```
nged-substation-forecast/
├── pyproject.toml              ← uv workspace config
├── packages/
│   ├── contracts/              ← Data schemas, Pydantic/Patito models, Base PyFunc classes
│   ├── spatial_utils/          ← H3 conversion and adjacency matrix logic
│   ├── nged_data/              ← Logic for NGED CKAN/S3 download & prep
│   ├── cerra_data/             ← Logic for CERRA reanalysis download (xarray \-\> Polars)
│   ├── cm_saf_data/            ← Logic for satellite irradiance download
│   ├── dynamical_data/         ← NWP client for Dynamical.org
│   ├── plotting/               ← Altair functions for MLFlow and Marimo
│   ├── notebooks/              ← Marimo interactive notebooks for exploration
│   ├── performance_evaluation/ ← Scriptable evaluation (wMAPE, CRPS, Super-Node tests)
│   ├── loss_functions/         ← PyTorch native metrics (Quantile Loss, GMM NLL)
│   ├── weather_encoder/        ← Base encoders for NWP data
│   ├── xgboost_forecaster/     ← Model package (Naive, MVP, State-Lag logic)
│   ├── solar_torch/            ← Differentiable physics for PV (tilt, azimuth inference)
│   └── st_gnn_forecaster/      ← Spatial-Temporal GNN implementation
├── conf/                       ← Hydra Configuration Directory
│   ├── config.yaml
│   ├── dataset/                ← Feature/Target configs
│   └── model/                  ← Architectures (xgboost\_naive, greybox, gnn)
├── tests/                      ← Integration tests for Dagster pipelines
└── src/nged_substation_forecast/ ← The Main Application
    ├── defs/*_assets.py         ← SDAs for data ingestion, feature engineering
    ├── defs/*_io.py             ← Custom PolarsDeltaIOManagers
    └── defs/*_jobs.py           ← Experimentation and K-fold CV orchestrators
```
