---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["tests", "packages/xgboost_forecaster"]
---
# Implementation Plan: Fix XGBoost Integration Test

## Investigation Summary

The integration test `test_xgboost_dagster_integration` fails with `ValueError: No training data remaining after dropping nulls in critical columns.` 

Investigation revealed two root causes for the null values in critical NWP columns:

1. **NWP Data Filtering Mismatch:** The `processed_nwp_data` asset was configured in the test with `start_date=nwp_init_time`. However, `nwp_init_time` marks the start of the *evaluation* period. The `train_xgboost` asset trains on data *before* this date. Consequently, `processed_nwp_data` yielded no NWP data for the training period, causing all NWP feature columns to be null after the join.
2. **Premature Training Execution:** The test attempted to backfill the `cleaned_actuals` Delta table for the training period by iterating over `training_partitions` and calling `job.execute_in_process()`. Because `train_xgboost` is an unpartitioned asset, Dagster executed it during *every* partition run. On the very first partition run, `cleaned_actuals` only contained one day of data, and there was no NWP data available for that specific day, leading to the immediate failure.

Additionally, a secondary issue was discovered during evaluation: the `categorical_precipitation_type_surface` column contains the value `12`, but the `pl.Enum` cast in `XGBoostForecaster._prepare_features` only allowed values `"0"` through `"8"`.

## Implementation Steps

### 1. Fix NWP Data Filtering in Test
Modify the `run_config` in `tests/test_xgboost_dagster_integration.py` to ensure `processed_nwp_data` includes the training period and its required lookback.

*   **File:** `tests/test_xgboost_dagster_integration.py`
*   **Action:** Change the `start_date` for `processed_nwp_data` from `str(nwp_init_time)` to `str(train_start - timedelta(days=14))`.

### 2. Fix Backfill Execution in Test
Modify the test to only execute the `cleaned_actuals` asset when backfilling the training partitions, preventing premature execution of the full ML pipeline.

*   **File:** `tests/test_xgboost_dagster_integration.py`
*   **Action:** Add `op_selection=["cleaned_actuals"]` to the `job.execute_in_process()` call inside the `training_partitions` loop.

### 3. Update Categorical Precipitation Enum
Update the `XGBoostForecaster` to accept a wider range of categorical precipitation values to prevent casting errors during inference.

*   **File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
*   **Action:** In `_prepare_features`, update the `pl.Enum` cast for `categorical_precipitation_type_surface` to include values up to `"15"` (e.g., `["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]`).

## Code Comments Mandate
*   In `tests/test_xgboost_dagster_integration.py`, add a comment explaining *why* `op_selection=["cleaned_actuals"]` is used (to backfill the Delta table without triggering the unpartitioned ML assets).
*   In `tests/test_xgboost_dagster_integration.py`, add a comment explaining *why* the `start_date` for `processed_nwp_data` includes a 14-day lookback before `train_start` (to provide sufficient history for autoregressive NWP features).
