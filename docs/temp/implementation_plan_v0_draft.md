---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/nged_data", "packages/xgboost_forecaster", "src/nged_substation_forecast", "packages/ml_core", "tests"]
---
# Implementation Plan: Rename `substation_number` to `time_series_id`

## Objective
Rename `substation_number` (and related terms like `time_series_ids`, `power_time_series`) to `time_series_id` (and `time_series_ids`, `power_time_series`) throughout the codebase to align with the `TimeSeriesMetadata` and `PowerTimeSeries` data contracts.

**CRITICAL CONSTRAINT:** Do NOT remove or rename `substation_number` within the `TimeSeriesMetadata` schema or any mock dataframes that are validated against it, as it remains a required field in the contract.

## Step-by-Step Plan

### Step 1: Update Data Cleaning Assets
**File:** `packages/nged_data/src/nged_data/clean.py`
*   **Action:** Update the comment on line 60.
    *   *From:* `# Note: The original cleaning.py used "substation_number" for rolling_std.`
    *   *To:* `# Note: The original cleaning.py used "time_series_id" for rolling_std.`

### Step 2: Update XGBoost Forecaster Model
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Action:** In `_prepare_data_for_model`, remove `"substation_number"` from the `select` list when creating `metadata_lf`.
    *   *From:* `["time_series_id", "h3_res_5", "substation_number"]`
    *   *To:* `["time_series_id", "h3_res_5"]`
*   *Rationale:* `substation_number` is not used for feature engineering or joining in the model pipeline.

### Step 3: Update Dagster Assets and Configs
**File:** `src/nged_substation_forecast/defs/xgb_assets.py`
*   **Action:** Rename `time_series_ids` to `time_series_ids` in `XGBoostConfig`.
*   **Action:** Rename `allow_empty_time_series` to `allow_empty_time_series` in `XGBoostConfig`.
*   **Action:** Rename the helper function `_get_target_time_series` to `_get_target_time_series`.
*   **Action:** Rename internal variables: `healthy_time_series` -> `healthy_time_series`, `sub_ids` -> `ts_ids`.
*   **Action:** Rename `power_time_series` to `power_time_series` in `train_xgboost` and `evaluate_xgboost` signatures and internal logic.
*   **Action:** Rename `power_time_series_filtered` to `power_time_series_filtered`.

**File:** `src/nged_substation_forecast/defs/xgb_jobs.py`
*   **Action:** Rename `power_time_series` to `power_time_series` in function signatures and docstrings.

**File:** `src/nged_substation_forecast/defs/weather_assets.py`
*   **Action:** Rename `time_series_ids` to `time_series_ids` in `WeatherConfig` and update the filtering logic accordingly.

### Step 4: Update ML Core Utilities
**File:** `packages/ml_core/src/ml_core/utils.py`
*   **Action:** In `train_and_log_model` and `evaluate_and_save_model`, update the temporal slicing logic to check for `"power_time_series"` instead of `"power_flows"`.
    *   *From:* `time_col = "period_end_time" if "power_flows" in key else "valid_time"`
    *   *To:* `time_col = "period_end_time" if "power_time_series" in key else "valid_time"`
    *   *From:* `if "power_flows" in key or "nwps" in key:`
    *   *To:* `if "power_time_series" in key or "nwps" in key:`
*   **Action:** Remove the redundant `if "power_time_series" in sliced_data:` block, as the caller will now pass `power_time_series` directly.

### Step 5: Update Tests
**File:** `tests/conftest.py`
*   **Action:** Rename `"substation_number"` to `"time_series_id"` in the `mock_ckan_primary_substation_locations` fixture.

**File:** `tests/test_xgboost_dagster_integration.py`
*   **Action:** Update the config dictionary keys from `"time_series_ids"` to `"time_series_ids"`.
*   **Action:** Remove the `# TODO: Change time_series_ids to time_series_ids` comments.

**File:** `tests/test_xgboost_mlflow.py`
*   **Action:** Rename `power_time_series` to `power_time_series` in the `evaluate_and_save_model` and `train_and_log_model` function calls.

**File:** `packages/xgboost_forecaster/tests/test_universal_model.py`
*   **Action:** Rename `power_time_series` to `power_time_series` in the `evaluate_and_save_model` function call.

*Note: Do NOT remove `substation_number` from `TimeSeriesMetadata` mock dataframes in any tests (e.g., `test_xgboost_dagster_mocked.py`, `test_xgboost_adversarial.py`, `test_plotting_robustness.py`, `test_xgboost_forecaster.py`, `test_xgboost_robustness.py`), as it is still a required field in the schema.*

## Review Responses & Rejections
*(To be filled during the review phase)*
