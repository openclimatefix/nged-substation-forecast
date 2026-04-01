---
status: "draft"
version: "v7.3"
after_reviewer: "reviewer"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/xgb_assets.py", "src/nged_substation_forecast/defs/weather_assets.py", "packages/xgboost_forecaster/src/xgboost_forecaster/model.py"]
---

# Implementation Plan: Addressing Reviewer Flaws

This plan addresses the flaws identified by the Reviewer in `docs/temp/reviewer_code_review_7.md`.

## 1. FLAW-001: Missing Field Documentation

**Target Files:**
- `src/nged_substation_forecast/defs/xgb_assets.py`
- `src/nged_substation_forecast/defs/weather_assets.py`

**Changes:**
- Import `Field` from `pydantic` in both files.
- Update `XGBoostConfig` in `xgb_assets.py` to use `Field(default=None, description="...")` for each field:
  - `train_start`: "Start date for training data (YYYY-MM-DD)"
  - `train_end`: "End date for training data (YYYY-MM-DD)"
  - `test_start`: "Start date for testing data (YYYY-MM-DD)"
  - `test_end`: "End date for testing data (YYYY-MM-DD)"
  - `substation_ids`: "List of substation IDs to train on"
- Update `ProcessedNWPConfig` in `weather_assets.py` to use `Field(default=None, description="...")` for `substation_ids`:
  - `substation_ids`: "List of substation IDs to process NWP data for"

## 2. FLAW-002: OOM Risk in NWP Processing

**Target File:**
- `src/nged_substation_forecast/defs/weather_assets.py`

**Changes:**
- Add a warning comment to the docstring of the `processed_nwp_data` asset.
- The comment should explicitly state: "WARNING: The underlying `process_nwp_data` function eagerly collects the LazyFrame into memory to perform upsampling and interpolation. This is a known OOM risk for large datasets. The `substation_ids` filtering is applied here as early as possible to mitigate this bottleneck, but a full refactor to lazy interpolation may be required in the future."

## 3. FLAW-003: Silent Null Dropping

**Target File:**
- `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`

**Changes:**
- In `XGBoostForecaster.train`, update the null-dropping logic to log the number of dropped rows.
  - Calculate the length of the dataframe before and after calling `drop_nulls(subset=critical_cols)`.
  - If the difference is greater than 0, log a warning using `log.warning(f"Dropped {dropped_len} rows due to nulls in critical columns: {critical_cols}")`.
- In `XGBoostForecaster.predict`, apply the same logic to log dropped rows during inference.

## Review Responses & Rejections

*   **FLAW-001 (Reviewer):** [ACCEPTED]. We will add `Field(description="...")` to all fields in `XGBoostConfig` and `ProcessedNWPConfig` to improve documentation and Dagster UI integration.
*   **FLAW-002 (Reviewer):** [ACCEPTED]. We will add a warning comment to `processed_nwp_data` documenting the OOM risk associated with eager collection during interpolation, noting that `substation_ids` filtering is applied early as a mitigation.
*   **FLAW-003 (Reviewer):** [ACCEPTED]. We will update `XGBoostForecaster.train` and `predict` to log the number of rows dropped due to nulls in critical columns, improving observability.
