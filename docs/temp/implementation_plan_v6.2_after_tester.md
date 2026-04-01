---
status: "draft"
version: "v6.2"
after_reviewer: "tester"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast", "packages/xgboost_forecaster"]
---

# Implementation Plan: Fix Memory Spikes and Add Validation

This plan addresses the flaws identified in the Tester's review (Iteration 6).

## 1. Fix Memory Spike in `forecast_vs_actual_plot` (FLAW-001)

**Issue:** The `forecast_vs_actual_plot` asset collects the entire `combined_actuals` LazyFrame into memory before filtering by substation, causing an Out-Of-Memory (OOM) error.
**Action:**
- In `src/nged_substation_forecast/defs/plotting_assets.py`, modify `forecast_vs_actual_plot` to extract the unique `substation_number`s from the `predictions` DataFrame.
- Filter the `combined_actuals` LazyFrame using `pl.col("substation_number").is_in(pred_substations)` *before* passing it to `downsample_power_flows` and calling `.collect()`.
**Rationale:** Filtering the LazyFrame before collection ensures that only the relevant actuals are loaded into memory, drastically reducing the peak memory footprint.

## 2. Fix OOM Risk in `XGBoostForecaster` (FLAW-002)

**Issue:** `_prepare_data_for_model` eagerly collects the entire joined dataset into a `pl.DataFrame` before feature selection and null-dropping, causing OOM errors during large-scale training.
**Action:**
- In `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, refactor `_prepare_data_for_model` to return a `pl.LazyFrame`. Remove the eager `.collect()` and the row-count logging (which requires eager execution).
- Refactor `_prepare_features` to accept and return either a `pl.LazyFrame` or `pl.DataFrame`. Use `.collect_schema()` for LazyFrames and `.schema` for DataFrames.
- In `train`, call `_prepare_features(joined_lf)` to get the feature columns lazily. Then, select only the required columns (`feature_cols + ["MW_or_MVA"]`) from `joined_lf` and call `.collect()`. Validate this collected DataFrame with `SubstationFeatures.validate(..., allow_missing_columns=True)`. Finally, extract `X` and `y`.
- In `predict`, follow a similar pattern: select `feature_cols` plus the required output columns (`valid_time`, `substation_number`, `ensemble_member`, `init_time`) from `df_lf`, collect, validate, and extract `X`.
**Rationale:** Keeping the data preparation pipeline lazy until the final feature selection and null-dropping minimizes the memory footprint, as only the required columns and rows are collected into memory.

## 3. Add Date Validation to `XGBoostConfig` (FLAW-003)

**Issue:** `XGBoostConfig` uses `date.fromisoformat()` to parse date strings without any pre-validation, which can cause a `ValueError` during asset execution if the input is invalid.
**Action:**
- In `src/nged_substation_forecast/defs/xgb_assets.py`, import `field_validator` from `pydantic`.
- Add a `@field_validator` method to the `XGBoostConfig` class to validate `train_start`, `train_end`, `test_start`, and `test_end`.
- The validator should attempt to parse the string using `date.fromisoformat(v)` and raise a descriptive `ValueError` if it fails.
**Rationale:** Validating the configuration inputs using Pydantic ensures that invalid dates are caught early during configuration validation, rather than failing gracefully during asset execution.

## Review Responses & Rejections

* **FLAW-001 (Tester):** ACCEPTED. Filtering the LazyFrame before collection is a standard Polars optimization that prevents OOM errors.
* **FLAW-002 (Tester):** ACCEPTED. Refactoring the pipeline to stay lazy until the final feature selection significantly reduces peak memory usage. We will collect only the necessary columns for training and inference, and validate the collected subset with `allow_missing_columns=True` to maintain data contracts.
* **FLAW-003 (Tester):** ACCEPTED. Adding a Pydantic `@field_validator` to `XGBoostConfig` ensures robust configuration validation and prevents runtime errors.
