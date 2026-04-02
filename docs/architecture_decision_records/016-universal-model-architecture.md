---
status: "Accepted"
date: "2026-04-02"
author: "Architect"
tags: ["ml-architecture", "xgboost", "forecasting"]
---

# ADR-016: Universal Model Architecture

## 1. Context & Problem Statement
We need to forecast power demand for ~1,300 primary substations across a 14-day horizon at 30-minute resolution. Previously, ADR-002 proposed a `LocalForecasters` approach, which would involve training separate models for each substation and potentially for each forecast horizon. This approach leads to an explosion in the number of models to manage (e.g., 1,300 substations * 672 horizons = 873,600 models), making orchestration, tracking in MLflow, and deployment extremely complex and brittle. We need an architecture that maximizes developer velocity, model robustness, and operational simplicity while supporting ensemble weather forecasts.

## 2. Options Considered

### Option A: Local Forecasters (One model per substation per horizon)
* **Description:** Train a distinct model for every combination of substation and forecast horizon step.
* **Why it was rejected:** Unmanageable operational complexity. It requires tracking hundreds of thousands of artifacts in MLflow. Furthermore, models for longer horizons often suffer from data sparsity and overfit, and the forecasts exhibit discontinuities ("jumps") between adjacent horizons. This approach was originally proposed in ADR-002 but is now superseded.

### Option B: Horizon-Specific Global Models (One model per horizon)
* **Description:** Train one global model across all substations for the 1-hour ahead forecast, another for the 1.5-hour ahead forecast, etc.
* **Why it was rejected:** Still requires managing 672 separate models for a 14-day horizon. It also fails to leverage the shared temporal patterns across horizons, and can still produce discontinuities at the boundaries between models.

### Option C: Universal Model Architecture
* **Description:** Train a **Single Universal Model** for all substations and all horizons. The model is trained on a "long-format" dataset. It explicitly receives `lead_time_hours` as a feature to learn how weather sensitivity decays over time, and `substation_number` as a categorical feature to learn spatial differences. It also natively supports ensemble NWP inputs by passing each ensemble member through the model to generate a distribution of power forecasts.

## 3. Decision
We chose **Option C: Universal Model Architecture**. This decision explicitly **supersedes the `LocalForecasters` approach outlined in ADR-002**.

## 4. Consequences
* **Positive:**
  * **Reduced Operational Complexity:** We only manage a single model artifact in MLflow per architecture type (e.g., one XGBoost model).
  * **Improved Generalization:** The model learns shared temporal patterns and weather sensitivities across all horizons and substations.
  * **Smoother Forecasts:** Produces naturally smooth forecasts across the entire 14-day horizon without discontinuities.
  * **Ensemble Support:** Easily generates probabilistic forecasts by running inference over each NWP ensemble member.
* **Negative/Trade-offs:**
  * Requires larger training datasets in memory (long-format data).
  * Necessitates more complex feature engineering, specifically dynamic seasonal lags, to prevent lookahead bias when training across multiple horizons simultaneously.
