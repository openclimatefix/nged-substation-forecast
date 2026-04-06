# Roadmap

This roadmap outlines the long-term architectural vision and planned research phases for the NGED Substation Forecast project.

## Phase 1: Infrastructure Plumbing & The "Universal" XGBoost Baseline
*Before attempting to detect complex switching events, we must prove the data engineering, MLflow tracking, and Dagster orchestration work end-to-end in a production environment.*

* **Ingestion:** Download NGED data and ECMWF ensemble NWP. Convert NGED data to Delta Lake and index via H3. Convert ECMWF to Parquet and index via H3.
* **The "Universal" Model:** Build a single XGBoost forecast for the ~50 trial sites. This model is "universal" in two ways: it is trained globally across all substations and across all forecast horizons (0-14 days) using `lead_time_hours` as a feature. Output an ensemble of power forecasts by passing each NWP ensemble member through the XGBoost model one-by-one. Wrap this model in an MLflow custom `pyfunc`.
* **Purpose:** This model will intentionally ignore switching events. The forecast will be flawed, but it serves as an integration test for our infrastructure.

## Phase 2: Topology Switching Detection
*Isolate and label historical switching events to significantly improve the demand forecasts.*

* **Exogenous Baseline:** Train K-Fold Out-of-Fold (OOF) XGBoost models using *only* weather and time features.
* **Residual Detection:** Apply CUSUM or Rolling Difference filters to the baseline residuals to flag topological step-changes.
* **Spatial Verification (Super-Node Test):** Validate switches by applying Kirchhoff's laws to neighbours.

## Phase 3: Upgrade forecast to handle switching events, and output NRA forecasts
*With clean switching labels, build the robust statistical baseline.*

* **Substation NRA (normal running arrangement) Forecast (Universal Model):** Train one universal demand forecast XGBoost model across all substations and horizons. Use the switching labels to estimate the transfer magnitude and mathematically "correct" the historical SCADA data.
* **Customer Meter Forecasts (Clustered/Local Models):** Train models clustered by asset type.

## Phase 4: Grey-Box Physics-Informed Neural Network (PyTorch)
*Transitioning from black-box trees to explainable physics to explicitly model Behind-The-Meter (BTM) assets.*

* **Explicit Disaggregation:** The model explicitly outputs unmetered solar and unmetered wind alongside gross demand.
* **Asset Discovery:** By feeding CM-SAF irradiance to a differentiable solar-torch module, physical parameters (installed capacity, tilt, azimuth, DC:AC capacity, shading) become learnable via gradient descent.
* **Natively Handling MVA Telemetry:** Older substations report absolute apparent power (MVA), hiding reverse power flows.

## Phase 5: Spatial Graph Neural Network
*Capturing highly non-linear, cross-network interactions during extreme events.*

* **Dynamic Adjacency Matrix:** The live Switching Detector acts as a permanent interceptor, continuously updating the adjacency matrix before inference.
* This ensures the GNN's message-passing layers route data along the actual physical paths of the switched grid.

## Phase 6: Further research
* Continually improve the current best model.
* Implement forecasts for BSPs and GSPs.
* Experiment with pre-trained encoders.
* Multi-sequence alignment with axial attention.
* Disaggregate other DERs (e.g. EV chargers, or batteries).
