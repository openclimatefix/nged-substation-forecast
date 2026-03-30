---
status: "draft"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["packages/ml_core", "packages/xgboost_forecaster", "packages/contracts", "src/nged_substation_forecast"]
---

# Implementation Plan: Iteration 1 Code Review Fixes

This document outlines the architectural and implementation changes required to address the 18 flaws (6 critical) identified during the Iteration 1 code reviews. The plan adheres to the project's core principles: modern tooling (Polars), modularity, scientific rigor, and strict data contracts.

## 1. Fix the Backtesting Bias in `evaluate_and_save_model`
**Target:** `packages/ml_core/src/ml_core/utils.py` & `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** The current evaluation collapses predictions by `valid_time` and selects the most recent forecast (`.last()`), artificially inflating performance metrics by only evaluating the shortest lead times.
*   **Implementation:**
    *   Add a `collapse_lead_times: bool = False` parameter to `XGBoostForecaster.predict`. When `False`, it skips the `.group_by(...).last()` step.
    *   Refactor `evaluate_and_save_model` to call `predict(collapse_lead_times=False)`.
    *   Calculate `lead_time = valid_time - init_time` in the evaluation results.
    *   Group the predictions by `lead_time` (and optionally `h3_index`) and compute metrics (MAE, RMSE, MAPE) for each horizon.
    *   Return a DataFrame of metrics indexed by lead time, providing a realistic view of model degradation over the forecast horizon.
*   **Testing:**
    *   **Unit Test:** Create a synthetic dataset in `packages/ml_core/tests/test_utils.py` with multiple `init_time` and `valid_time` pairs.
    *   **Verification:** Assert that metrics are calculated per `lead_time` and match manually calculated values.
    *   **Edge Case:** Test with a single lead time and with missing `init_time` (should handle gracefully or error clearly).

## 2. Correct the NWP Accumulation Handling (Differencing)
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
*   **Issue:** ECMWF accumulated variables (e.g., precipitation, radiation) are de-accumulated by Dynamical.org. The scientist agent wasn't aware of this and tried to recommend a fix. We must add liberal comments to explain that these variables are de-accumulated by Dynamical.org before we download them.
*   **Implementation:**
    *   Comment in the NWP data contract and the `dynamical_data` package to point out that
    all accumulated values (including precipitation and radiation) are de-accumulated by
    Dynamical.org *before* we download them.

## 3. Fix the Multi-NWP Join Logic
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py` (or `data.py`)
*   **Issue:** Joining multiple NWP models on exact `init_time` and `ensemble_member` fails due to differing initialization schedules and ensemble sizes, resulting in silent data loss.
*   **Implementation:**
    *   Redesign the multi-NWP join to use an "as-of" join strategy.
    *   For a primary NWP (e.g., ECMWF) and a secondary NWP (e.g., GFS), aggregate the secondary NWP across its ensemble members to create deterministic features (e.g., `gfs_temp_mean`, `gfs_temp_std`).
    *   **CRITICAL:** Define an `available_time = init_time + publication_delay` for all NWPs to prevent lookahead bias. A GFS forecast initialized at 12:00z is not available at 12:00z.
    *   Use `pl.DataFrame.join_asof` to find the most recently available secondary NWP forecast. The join must strictly enforce that `secondary_NWP.available_time <= primary_NWP.init_time`.
    *   Do NOT join on `valid_time` without this strict `available_time` condition, as it will leak future weather forecasts.
    *   Broadcast these deterministic secondary features to all ensemble members of the primary NWP.
*   **Testing:**
    *   **Integration Test:** Simulate misaligned `init_time`s between two NWP sources and verify that the `join_asof` correctly picks the latest available secondary forecast.
    *   **Data Loss Check:** Verify that the number of rows in the primary NWP is preserved after the join.

## 4. Call Missing Feature Engineering Functions & Enforce Contracts
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, `packages/xgboost_forecaster/src/xgboost_forecaster/features.py` & `packages/contracts/src/contracts/data_schemas.py`
*   **Issue:** `add_weather_features` and `add_physical_features` are defined but never called. The `SubstationFeatures` Patito contract is unenforced.
*   **Implementation:**
    *   Refactor all functions in `features.py` (`add_temporal_features`, `add_physical_features`, `add_weather_features`) to accept and return `pl.LazyFrame` instead of `pl.DataFrame`.
    *   In the `_prepare_features` method of `XGBoostForecaster`, explicitly call `add_weather_features(df)` (which internally calls `add_physical_features`) before selecting the final feature columns.
    *   Enforce the `SubstationFeatures` contract by calling `SubstationFeatures.validate(df)` on the resulting DataFrame before it is passed to the model for training or inference.
*   **Testing:**
    *   **Adversarial Test:** Pass a DataFrame missing required weather columns to `_prepare_features` and assert that it raises a `ValidationError` or `ColumnNotFoundError`.
    *   **Contract Test:** Verify that `SubstationFeatures.validate` is called by mocking it in a unit test.

## 5. Refactor `process_nwp_data` to Native Polars Operations
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
*   **Issue:** Nested Python loops over `init_time`, `h3_index`, and `ensemble_member` cause severe performance bottlenecks and bypass Polars' lazy optimizations.
*   **Implementation:**
    *   Remove all Python `for` loops in `process_nwp_data`.
    *   Use Polars' native temporal upsampling: `df.upsample(time_column="valid_time", every="30m", group_by=["init_time", "h3_index", "ensemble_member"])`.
    *   Chain `.interpolate()` to fill missing values in a fully vectorized, lazy execution graph.
*   **Testing:**
    *   **Regression Test:** Compare the output of the refactored `process_nwp_data` with a small dataset processed by the old (loop-based) logic to ensure identical results.
    *   **Edge Case:** Test with irregular time intervals and missing ensemble members.

## 6. Fix the `init_time` Dependency in `train`
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Issue:** The `train` method unconditionally attempts to calculate lead time using `init_time`, crashing if NWP data (and thus `init_time`) is not provided.
*   **Implementation:**
    *   Add a conditional check in `train`: `if "init_time" in df.columns:`.
    *   If `init_time` is present, calculate `lead_time` and apply lead-time-specific lag logic.
    *   If `init_time` is absent (e.g., training an autoregressive-only baseline), bypass the lead-time logic or default to a standard lag strategy based purely on `valid_time`.
*   **Testing:**
    *   **Unit Test:** Call `train` with a DataFrame that lacks an `init_time` column and verify it completes successfully.
    *   **Logic Check:** Ensure that when `init_time` is present, the lead-time features are correctly generated.

## 7. Address Logic Duplication and Type Safety Issues
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py` & `packages/ml_core/src/ml_core/model.py`
*   **Issue:** Lag logic is duplicated between `train` and `predict`. `BaseForecaster` uses `**kwargs` leading to `# type: ignore`. Hardcoded constants (e.g., 7-day threshold) are used.
*   **Implementation:**
    *   **Deduplication:** Extract the lag generation and dynamic seasonal lag logic into a private method `_add_lags(self, df: pl.LazyFrame) -> pl.LazyFrame` within `XGBoostForecaster`.
    *   **Robustness:** Ensure `_add_lags` handles the absence of `init_time` by defaulting to a static lag strategy (e.g., always using `power_lag_7d` if available).
    *   **Type Safety:** Refactor `BaseForecaster.train` and `predict` signatures to use explicit, strictly-typed arguments (e.g., `target_df: pl.LazyFrame`, `features_df: pl.LazyFrame | None = None`) instead of `**kwargs`. Remove `# type: ignore` overrides in subclasses.
    *   **Constants & Leakage Fix:** The current `if lead_time <= 7 days use 7d lag else use 14d lag` logic leaks future data for lead times > 14 days (ECMWF goes up to 15 days). **CRITICAL:** Generalize the lag logic dynamically to prevent lookahead bias: `weeks_ahead = ceil(lead_time / 7_days)` and use `lag = weeks_ahead * 7_days`. Remove the hardcoded 168-hour threshold entirely in favor of this dynamic calculation.
    *   **Eager Operations:** Remove `.collect()` calls in `evaluate_and_save_model` that break the lazy pipeline; pass `nwp_init_time` explicitly if needed.
    *   **Magic Strings:** Replace `"power_flows" in key` slicing logic with explicit mapping or arguments.
    *   **Unimplemented Methods:** Raise `NotImplementedError` in `LocalForecasters.log_model` instead of leaving a silent TODO.
*   **Testing:**
    *   **Consistency Test:** Verify that `_add_lags` produces identical results when called with the same input data in different contexts.
    *   **Type Check:** Run `uv run ty check` to ensure all `# type: ignore` removals are valid and types are correctly inferred.

## 8. Strengthen Patito Data Contracts and Validation
**Target:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py` & `packages/contracts/src/contracts/data_schemas.py`
*   **Issue:** Missing column validation, lack of NaN/Inf checks before training, and weak validation in `SubstationFlows`.
*   **Implementation:**
    *   **Contract Update:** Update `SubstationFeatures` to include `latest_available_weekly_lag` and make raw lags optional.
    *   **Column Validation:** In `_prepare_features`, verify that all `self.config.features.feature_names` exist in `df.columns`. Raise a clear `ValueError` if any are missing. Ensure `feature_names` is persisted with the model to prevent brittle alignment in `predict`.
    *   **NaN/Inf Validation:** Add a Polars expression check (`df.select(pl.all().is_nan().any())` and `is_infinite()`) on the final feature matrix `X` and target `y` before calling `XGBRegressor.fit`. Raise an error if invalid values are found.
    *   **SubstationFlows Validation:** Update `SubstationFlows.validate` to ensure that at least one of the `MW` or `MVA` columns contains at least one non-null value (e.g., `df.select(pl.col("MW").is_not_null().any() | pl.col("MVA").is_not_null().any()).item()`).
*   **Testing:**
    *   **Adversarial Test:** Pass DataFrames with NaNs, Infs, or all-null `MW`/`MVA` columns to the respective validation functions and assert they raise `ValueError`.
    *   **Schema Test:** Ensure `SubstationFeatures` correctly validates the output of the feature engineering pipeline.
