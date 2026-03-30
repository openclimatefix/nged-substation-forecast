---
status: "draft"
version: "v3.1"
after_reviewer: "iteration3"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["packages/xgboost_forecaster", "packages/ml_core", "packages/contracts"]
---

# Implementation Plan: Iteration 3 Code Review Fixes

This document outlines the architectural and implementation changes required to address the 10 flaws (5 critical) identified during the Iteration 3 code reviews.

## 1. Dynamic Seasonal Lag (Critical)
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** The lag logic still uses a hardcoded 7-day threshold (`pl.when(lead_time_days <= 7)`), which leaks future data for lead times > 14 days.
*   **Implementation:**
    *   In `_add_lags`, calculate the required lag dynamically to strictly prevent lookahead bias:
        ```python
        df = df.with_columns(
            lead_time_days=(pl.col("valid_time") - pl.col("init_time")).dt.total_days()
        ).with_columns(
            lag_days=pl.max_horizontal(pl.lit(1), (pl.col("lead_time_days") / 7.0).ceil().cast(pl.Int32)) * 7
        ).with_columns(
            target_lag_time=pl.col("valid_time") - pl.duration(days=1) * pl.col("lag_days")
        )
        ```
    *   Join `flows_30m` on `["substation_number", "target_lag_time"]` to extract the exact `latest_available_weekly_lag` without needing pre-calculated `lag_7d` or `lag_14d` columns.

## 2. Feature Engineering & NWP Prefixing (Critical)
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** `add_weather_features` is called *after* NWP columns are prefixed (e.g., `ECMWF_temperature_2m`), causing it to silently fail because it looks for unprefixed names.
*   **Implementation:**
    *   In `train` and `predict`, call `add_weather_features(nwp_lf)` on each NWP *before* applying the prefixing logic.
    *   **Crucial Fix:** Do not prefix the *primary* NWP (the first one in the list). This ensures that core features like `temperature_2m` retain their expected names, satisfying the `SubstationFeatures` contract. Secondary NWPs will still be prefixed (e.g., `GFS_temperature_2m`).

## 3. Backtesting Metrics (Critical)
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** `predict` hardcodes the scalar `inference_params.nwp_init_time` into the output dataframe, destroying the actual `init_time` of the NWP used for each row. This causes `lead_time_hours` to be calculated as negative values during backtesting.
*   **Implementation:**
    *   In `predict`, retain the `init_time` column from the primary NWP after the `.last()` aggregation.
    *   Output this actual `init_time` as the `nwp_init_time` column in the final `PowerForecast` dataframe, ensuring accurate lead time calculations in `evaluate_and_save_model`.

## 4. NWP Availability Delay (Critical)
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** `predict` subtracts 3 hours from `inference_params.nwp_init_time`. If a user requests a specific NWP run (e.g., 12:00z) and only provides that run, it gets filtered out, resulting in empty predictions.
*   **Implementation:**
    *   Remove the `- timedelta(hours=3)` subtraction in `predict`.
    *   Treat `inference_params.nwp_init_time` as the exact maximum NWP run to use (i.e., `init_time <= target_init_time`). The 3-hour availability delay is already correctly handled during the multi-NWP `join_asof` in `train`.

## 5. NaN/Inf Validation
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** Validation only checks for `Inf`s in `X` during training, missing `NaN`s and ignoring `y` and `predict`.
*   **Implementation:**
    *   In both `train` and `predict`, implement strict validation for `X`:
        ```python
        if X.select(pl.any_horizontal(pl.all().is_nan() | pl.all().is_infinite())).sum().item() > 0:
            raise ValueError("Input features X contain NaN or Inf values")
        ```
    *   In `train`, add identical validation for the target `y`.

## 6. Patito Contracts
**Target:** `packages/contracts/src/contracts/data_schemas.py`
*   **Issue:** `SubstationFeatures` uses `allow_missing=True` for critical features, allowing silent failures if joins break.
*   **Implementation:**
    *   Remove `allow_missing=True` from critical features like `temperature_2m` and `latest_available_weekly_lag` in `SubstationFeatures`. They must be strictly enforced for production models.

## 7. NWP Deaccumulation Comments
**Target:** `packages/contracts/src/contracts/data_schemas.py` & `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
*   **Issue:** Missing documentation regarding accumulated variables.
*   **Implementation:**
    *   Add explicit comments to `ProcessedNwp` and `process_nwp_data` stating that accumulated variables (e.g., precipitation, radiation) are already de-accumulated by Dynamical.org prior to download, and should not be differenced.

## 8. Redundant Temporal Features
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`
*   **Issue:** `add_temporal_features` is duplicated.
*   **Implementation:**
    *   Delete `add_temporal_features` from `xgboost_forecaster/features.py`.
    *   Ensure `ml_core.features.add_cyclical_temporal_features` is used exclusively across the codebase.

## 9. Type Hints
**Target:** `packages/ml_core/src/ml_core/features.py`
*   **Issue:** `add_cyclical_temporal_features` uses `Any` for its dataframe types.
*   **Implementation:**
    *   Replace `Any` with a `TypeVar("T", pl.DataFrame, pl.LazyFrame)` to preserve the exact Polars type through the function.

## 10. Eager Collection Warning
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
*   **Issue:** `process_nwp_data` calls `.collect()` to perform upsampling, which can cause OOM errors on large datasets.
*   **Implementation:**
    *   Add a prominent docstring warning to `process_nwp_data` advising that the input LazyFrame is collected into memory, and users should partition large historical datasets before calling this function.
