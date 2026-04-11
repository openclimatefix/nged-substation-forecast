---
status: "draft"
version: "v0"
after_reviewer: "none"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/xgboost_forecaster", "packages/ml_core", "tests"]
---

# Implementation Plan: Remove Downsampling References and Update Mock Data

## Objective
1. Remove all references to 'downsampling' or 'downsampled' from the codebase, as this operation is no longer performed.
2. Update all mock test data to use the new column names (`period_end_time` and `power`) instead of the old ones (`timestamp` and `MW`).
3. Remove any redundant renaming operations (e.g., `.rename({"timestamp": "period_end_time", "MW": "power"})`) in the codebase.
4. Ensure all tests are updated to reflect these changes.

## 1. Remove Downsampling References

The following files contain outdated docstrings or comments referring to downsampling. These should be updated to simply refer to "Historical power flows" or "Historical power flow data".

*   **`packages/xgboost_forecaster/src/xgboost_forecaster/features.py`**:
    *   Update docstring for `flows_30m` in `add_lag_features`: Change "Historical power flows downsampled to 30m." to "Historical power flows."
*   **`packages/xgboost_forecaster/src/xgboost_forecaster/model.py`**:
    *   Update docstrings for `flows_30m` in `_prepare_training_data`, `_prepare_data`, `train`, and `predict`: Change "Historical power flows downsampled to 30m." to "Historical power flows." (or "Historical power flow data (for lags).").
*   **`packages/xgboost_forecaster/tests/test_universal_model.py`**:
    *   Remove the comment `# This will be downsampled in evaluate_and_save_model` in `test_evaluate_and_save_model`.
*   **`packages/ml_core/src/ml_core/utils.py`**:
    *   Remove the comment `# We downsample power flows to 30m to ensure consistency across models.` in `train_and_evaluate_model`.
    *   Remove the comment `# Downsample power flows to 30m for inference (lags)` in `generate_forecasts`.
*   **`packages/ml_core/src/ml_core/model.py`**:
    *   Update docstrings for `flows_30m` in `train` and `predict`: Change "Historical power flow data downsampled to 30m." to "Historical power flow data."
*   **`packages/ml_core/src/ml_core/experimental.py`**:
    *   Update docstrings for `flows_30m` in `train` and `predict`: Change "Historical power flow data downsampled to 30m." to "Historical power flow data."

## 2. Update Mock Test Data Column Names

Update the mock data definitions in the test suite to use `period_end_time` and `power` directly.

*   **`tests/test_nged_assets.py`**:
    *   In `mock_clean.return_value`, change `"timestamp"` to `"period_end_time"` and `"MW"` to `"power"`.
    *   In the raw JSON mock data written to `test.json`, change `{"timestamp": "2026-01-01T00:00:00Z", "MW": 10.0, "MVA": 12.0}` to `{"startTime": "2026-01-01T00:00:00Z", "endTime": "2026-01-01T00:30:00Z", "value": 10.0}` to better reflect the actual raw data format expected by `io.py`.
*   **`tests/test_xgboost_dagster_integration.py`**:
    *   In `test_xgboost_dagster_integration`, change `pl.col("timestamp")` to `pl.col("period_end_time")` and `get_column("timestamp")` to `get_column("period_end_time")`.
*   **`tests/test_xgboost_robustness.py`**:
    *   In all `pl.DataFrame` definitions for `flows`, change `"timestamp"` to `"period_end_time"` and `"MW"` to `"power"`.
    *   Change `pl.col("MW").cast(pl.Float32)` to `pl.col("power").cast(pl.Float32)`.
*   **`tests/test_xgboost_adversarial.py`**:
    *   In the `flows` DataFrame definition, change `"timestamp"` to `"period_end_time"` and `"MW"` to `"power"`.
*   **`tests/test_xgboost_forecaster.py`**:
    *   In all `pl.DataFrame` definitions for `flows`, change `"timestamp"` to `"period_end_time"` and `"MW"` to `"power"`.
    *   Change `pl.col("timestamp").str.to_datetime()` to `pl.col("period_end_time").str.to_datetime()`.

*(Note: `units: ["MW"]` in `tests/test_plotting_robustness.py` is metadata and should be kept as is, as it is not a column name.)*

## 3. Remove Redundant Renaming Operations

Since the mock data will now use the correct column names, the redundant renaming operations can be removed.

*   **`tests/test_xgboost_forecaster.py`**:
    *   Remove `.rename({"timestamp": "period_end_time", "MW": "power"})` from all `flows_30m` assignments.
    *   Example: `flows_30m = flows.rename(...).with_columns(...)` becomes `flows_30m = flows.with_columns(...)`.
*   **`tests/test_xgboost_robustness.py`**:
    *   Remove `.rename({"timestamp": "period_end_time", "MW": "power"})` from all `flows_30m` assignments.

## 4. Code Comments and Architecture

*   Ensure that no new code comments reference FLAW-XXX IDs.
*   Ensure that any new comments focus on the *why* (intent and rationale) rather than the *how*.
*   The changes are purely refactoring and test updates, so no new architectural components are introduced.

## Review Responses & Rejections
*   None yet.
