---
status: "Accepted"
date: "2026-03-30"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["mlops", "architecture", "interfaces", "xgboost", "backtesting"]
---

# ADR-009: Unified ML Model Interface and Advanced Forecasting Features

## 1. Context & Problem Statement
As the NGED substation forecast project evolves, we need a robust and standardized way to train and evaluate different machine learning models. The previous iterations lacked a unified interface for models, making it difficult to swap implementations or orchestrate them uniformly in Dagster. Furthermore, we needed to address several critical forecasting requirements:
1. **Multi-NWP Support**: Models must be able to ingest forecasts from multiple Numerical Weather Prediction (NWP) providers simultaneously.
2. **Lookahead Bias Prevention**: We must strictly prevent data leakage, especially concerning the availability delay of NWP forecasts and the calculation of autoregressive lags.
3. **Rigorous Backtesting**: We need a mechanism to simulate real-time inference during backtesting, ensuring that models only use data that would have been available at the time of prediction.

## 2. Decision

We have implemented a unified ML model interface and several advanced forecasting features to address these requirements.

### 2.1 Unified `BaseForecaster` Protocol
We introduced a `BaseForecaster` abstract base class in `ml_core.model`. This protocol defines a standard interface for all forecasting models, requiring them to implement `train()` and `predict()` methods.
- `train()` takes a `ModelConfig`, historical power flows, substation metadata, and a dictionary of NWP dataframes.
- `predict()` takes inference parameters, metadata, power flows, NWP dataframes, and a `collapse_lead_times` flag.

This allows Dagster to orchestrate any model (e.g., `XGBoostForecaster`) uniformly.

### 2.2 Multi-NWP Support with Prefixing
To support multiple NWP sources (e.g., ECMWF and GFS), the `train` and `predict` methods accept a dictionary of NWP dataframes keyed by `NwpModel`.
- The primary NWP features retain their original names.
- Secondary NWP features are prefixed with their model name (e.g., `gfs_temperature_2m`).
- All NWPs are joined using a `join_asof` operation on an `available_time` column, which is calculated as `init_time + 3 hours` to account for the delay in NWP availability.

### 2.3 Dynamic Seasonal Lags
To prevent lookahead bias when using autoregressive features (lags), we implemented dynamic lag calculation. Instead of using fixed 7-day or 14-day lags, the model calculates the required lag dynamically based on the lead time of the forecast:
- `lead_time_days = (valid_time - init_time).dt.total_days()`
- `lag_days = max(1, ceil(lead_time_days / 7)) * 7`
- `target_lag_time = valid_time - lag_days`

This ensures that the model always uses the most recent *available* historical data for a given lead time, strictly preventing data leakage.

### 2.4 Backtesting via `collapse_lead_times`
We introduced a `collapse_lead_times` parameter in the `predict` method.
- When `True` (simulating real-time inference), the model filters the NWP data to keep only the latest available forecast for each `valid_time` prior to the `nwp_cutoff`, enforcing the 3-hour availability delay.
- When `False` (rigorous backtesting), the model evaluates all available lead times up to the cutoff, allowing us to assess performance across different forecast horizons while still respecting the availability delay.

### 2.5 Strict Data Contracts
We updated the `SubstationFeatures` and `PowerForecast` data contracts in `packages/contracts` to enforce these new requirements, including validation for the dynamic `latest_available_weekly_lag` and the `ensemble_member` fields.

## 3. Consequences
* **Positive:** The unified interface makes it trivial to add new model types (e.g., PyTorch GNNs) without changing the orchestration logic.
* **Positive:** Multi-NWP support allows models to leverage diverse weather data, potentially improving accuracy.
* **Positive:** Dynamic lags and strict availability delays completely eliminate lookahead bias, ensuring that backtesting results are reliable and representative of production performance.
* **Negative/Trade-offs:** The dynamic lag calculation and multi-NWP joining logic increase the complexity of the feature engineering pipeline.
