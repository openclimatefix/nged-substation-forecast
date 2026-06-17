# Roadmap

This roadmap outlines the planned order of development toward the v1.0 live forecast release (January 2027) and beyond. It reflects the plan as of the Milestone 1 report (28 May 2026). Technical plans change as we learn more — treat this as a best-estimate, not a guarantee.

---

## v0.1 — "Naive" MVP (internal only)

**Goal**: A simple XGBoost forecast that lets us test infrastructure end-to-end and establish a baseline. Intentionally does not detect switching events or estimate effective capacity — hence "naive" (assumes the grid is always in perfect health).

**Data engineering**:
- Download ECMWF ENS NWP from Dynamical.org; convert to Delta Lake with H3 spatial indexes on S3 ✓
- Download NGED data from S3; convert to Delta Lake with H3 spatial indexes ✓
- Automatic data cleaning:
    - Remove isolated zeros in non-zero time series
    - Remove values more than N standard deviations from the mean
    - Remove "stuck" values (std dev near zero over a 24-hour rolling window)
    - Drop first ~2 months of each time series (poor quality during meter ramp-up)

**Software engineering**:
- Universal model interface (`BaseForecaster`) for all ML models ✓
- Orchestrate everything with Dagster: data ingestion, model training, inference, backtesting ✓

**ML model**:
- Separate XGBoost model per `time_series_id` ✓
- Train on the ECMWF control ensemble member; run inference on all 51 members ✓
- Features: ECMWF ENS NWP variables, derived weather features (e.g. wind chill), lagged power (7 and 14 days ago), datetime features (`hour_of_day`, `day_of_week`, `day_of_month`, `day_of_year`, `is_weekend`, `is_bank_holiday`, `is_christmas`, `is_easter`, `utc_offset`, `forecast_horizon`)
- Output: ensemble of 51 half-hourly power forecasts per `time_series_id`, scaled to [−1, +1], on a 14-day horizon
- Static capacity estimate: 99th percentile of observed power as a simple proxy

**Outputs** (Delta tables on S3):
- `power_forecast`
- `power_forecast_warnings`
- `asset_health_history`

---

## v0.2 — Code Quality & Documentation

- More unit tests
- CI on GitHub
- Improve documentation
- Verify daylight savings time handling is correct

---

## v0.3 — Leaderboard / Performance Analysis

- Implement the ML energy forecasting "leaderboard" (cross-fold validation metrics in MLflow), ready for systematic ML experimentation ✓ (CV assets added)
- Metrics: normalised MBE, normalised MAE, RMSE, Pinball loss, PICP, CRPS, Spread-Skill Ratio
- Time-slice filters: nowcasting (0–6 h), day-ahead (6–36 h), medium range (Day 2–7), extended range (Day 8–14), peak events (top 5%)

---

## v0.4 — Improved Automatic Data Cleaning

- More sophisticated automatic cleaning of NGED's power data (building on the basic cleaning in v0.1)

---

## v0.5 — XGBoost Upgrades ("Quick Wins")

Establish a strong XGBoost baseline before investing in capacity estimation and switching event detection.

Ideas to test (informed by literature review):

- Train a "global" XGBoost model across each category of time series (e.g. one model for all MW primary substations, one for all MVA primaries, one per generator type) instead of one model per `time_series_id`
- Train separate XGBoost models per forecast horizon window (e.g. 0–2 days, 2–7 days, 7–14 days)
- Train on all 51 NWP ensemble members (may require XGBoost's iterator API to handle data that doesn't fit in RAM)
- Pre-train XGBoost on CERRA reanalysis (which covers 2020–2024, before our ECMWF dataset starts), then fine-tune on ECMWF ENS

---

## v0.6 & v0.7 — Switching Events & Dynamic Generator Capacity

*Internal only for first month, then shared with NGED.*

**Switching events**:
- Detect "abnormal running arrangement" events from the power time series alone, using statistical methods
- Use the detected switching events to clean training data: train XGBoost only on "normal arrangement" periods
- Populate the `substation_switching` Delta table

**Dynamic effective capacity estimation (differentiable physics)**:
- Implement differentiable physics (DP) models for wind and solar PV generators to estimate effective capacity at each half-hourly timestep
- Two-pass approach: first pass estimates effective capacity; second pass normalises the time series by effective capacity before training the power forecast model
- Ingest additional weather datasets needed for capacity estimation:
    - **CERRA** (Copernicus regional reanalysis) — high-resolution historical weather, useful for pre-training and for estimating historical generator capacity
    - **CM SAF** (Satellite Application Facility on Climate Monitoring) — high-resolution satellite-derived irradiance, used to estimate solar PV capacity
- Populate the `effective_capacity` Delta table

**"Prevailing conditions" building block**:
- Produce example Python code for NGED to construct a "prevailing conditions" forecast from OCF's building blocks

---

## v1.0 — Stable Live Service for NGED's Trial Area

**Target: January 2027**

- All features listed above (v0.1–v0.7), plus fixes discovered during live running
- 32 time series in the NGED trial area: 16 primary substations, 6 solar PV farms, 3 wind farms, 2 GSPs, 2 BSPs, 1 biofuel generator, 1 BESS, 1 reciprocating gas generator
- Five Delta Lake output tables delivered to NGED every 6 hours:
    1. `power_forecast` — [−1, +1] ensemble power forecasts
    2. `power_forecast_warnings` — per-`time_series_id` warnings (HEALTHY, MISSING VALUE, STUCK TIMESERIES, INVALID TIMESERIES VALUE, GENERATOR OR CABLE FAULT, GENERATOR REDUCED CAPACITY, SUBSTATION ABNORMAL RUNNING ARRANGEMENT, STALE NWP, STALE POWER)
    3. `asset_health_history` — complete historical record of each time series's health state
    4. `effective_capacity` — half-hourly probabilistic generator capacity estimates (mean + std)
    5. `substation_switching` — estimated power diverted between substation pairs (mean + std)

---

## v2.0 — Scale-Up (Future Research)

**Required**:
- Scale to approximately 2,500 time series: all of NGED's primary substations (1,161), BSPs (271), GSPs (52), and most customer meters (~1,000)
- Estimate the installed capacity of *unmetered* solar PV and wind on each primary substation (by disaggregating net primary substation power flows)
- Compare top-down forecasts vs. bottom-up forecasts for BSPs and GSPs

**Research (advanced ML)**:
- **Graph Neural Networks (GNNs)**: Model substations, metered generators, and unmetered generator fleets as nodes in an electrical/spatial graph. Graph edges represent physical connections. This is the approach most likely to capture the fact that switching events transfer *behaviour*, not just a constant amount of power.
- **Pre-trained neural network encoders**: "weather encoder" and "time encoder" pre-trained on large datasets, then fine-tuned for substation forecasting
- **Multi-sequence alignment** with axial attention: find "similar" historical days and feed them as additional context to the forecasting model
- **CRPS training objective**: train the ensemble power forecast model to directly optimise CRPS for sharper probabilistic forecasts
- **JEPA** (Joint Embedding Predictive Architecture, à la Yann LeCun): adapt to demand forecasting using JEPA's encoder and predictor as the "load" module in the GNN
- **Differentiable physics for power forecasting** (not just capacity estimation): use DP models to directly forecast power, handling MVA metering natively

**Stretch goals**:
- Forecast *unmetered* solar and wind power at each primary substation
- Disaggregate additional DERs (price-sensitive assets like batteries) from substation power flow
- Estimate cost savings (£) attributable to each forecasting approach in the leaderboard
- Build a REST API on top of the Delta Lake delivery mechanism
