---
status: "draft"
version: "v0"
after_reviewer: "architect"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/xgboost_forecaster", "packages/nged_data", "packages/ml_core"]
---
# Implementation Plan: Replace 'value' with 'power'

## Objective
Replace all occurrences of the column name `'value'` with `'power'` across the codebase to ensure consistency with the `PowerTimeSeries` and `XGBoostInputFeatures` data contracts. Explicitly forbid the use of `'value'` as a column name.

## Context
The `PowerTimeSeries` and `XGBoostInputFeatures` contracts in `packages/contracts/src/contracts/data_schemas.py` already define the target variable as `power`. However, several modules (like `xgboost_forecaster` and `ml_core`) were renaming `power` to `value` internally, causing inconsistencies and potential validation failures. The raw JSON data ingestion in `nged_data` was also renaming `value` to `power`, but the mock data in tests still used `value`.

## Steps

### 1. Update `xgboost_forecaster`
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`
- In `_prepare_training_data` (line 220), remove `.rename({"power": "value"})`.
- In `_prepare_data_for_model` (line 339), remove `.rename({"power": "value"})`.
- In `_prepare_data_for_model` (line 355), change `value=pl.col("value") / pl.col("peak_capacity")` to `power=pl.col("power") / pl.col("peak_capacity")`.
- In `_prepare_data_for_model` (line 358), change `value=pl.lit(0.0, dtype=pl.Float32)` to `power=pl.lit(0.0, dtype=pl.Float32)`.
- In `train` (line 432), change `critical_cols = ["value"]` to `critical_cols = ["power"]`.
- In `train` (lines 451, 456), change `["value"]` to `["power"]` in the `select` calls.
- In `train` (line 475), change `y = joined_df.get_column("value")` to `y = joined_df.get_column("power")`.
- In `predict` (line 556), change `["value", "peak_capacity"]` to `["power", "peak_capacity"]` in the `select` call.

**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`
- In `add_autoregressive_lags` (line 55), change `pl.col("value").alias(...)` to `pl.col("power").alias(...)`.

**File:** `packages/xgboost_forecaster/tests/test_universal_model.py`
- In `test_mlflow_metric_thinning` (line 184), remove `.rename({"power_fcst": "value"})`.
- In `test_mlflow_metric_thinning` (line 199), remove `.rename({"power": "value"})`.
- In `test_autoregressive_lag_consistency` (line 280), change `"value": pl.Series([], dtype=pl.Float32)` to `"power": pl.Series([], dtype=pl.Float32)`.

**File:** `packages/xgboost_forecaster/tests/test_xgboost_features.py`
- In `test_add_autoregressive_lags_prevents_lookahead` (line 41), change `"value"` to `"power"`.
- In `test_add_autoregressive_lags_handles_missing_flows` (line 111), change `"value"` to `"power"`.

### 2. Update `nged_data`
**File:** `packages/nged_data/src/nged_data/io.py`
- Remove the renaming of `"value"` to `"power"` (lines 108-109), as we assume the upstream JSON data will now provide `"power"` directly.

**File:** `packages/nged_data/tests/test_io.py`
- In `test_load_nged_json_valid` (lines 20, 21), change `"value"` to `"power"` in the dummy JSON data.

### 3. Update `ml_core`
**File:** `packages/ml_core/src/ml_core/utils.py`
- In `evaluate_and_save_model` (lines 175-176), remove `.rename({"value": "power_fcst"})` and `.rename({"power_fcst": "value"})`. The `predict` method returns a `PowerForecast` dataframe which already has the `power_fcst` column.

**File:** `packages/ml_core/tests/test_utils.py`
- In `test_slice_temporal_data_lazyframe` (lines 36, 48), change `"value"` to `"power"`.
- In `test_slice_temporal_data_no_time_col` (lines 92, 103), change `"value"` to `"power"`.

### 4. Verify Contracts
**File:** `packages/contracts/src/contracts/data_schemas.py`
- No changes are required here. The `PowerTimeSeries` and `XGBoostInputFeatures` contracts already use `power`. This plan aligns the rest of the codebase with these contracts.

## Constraints & Rules
- **Forbidden:** The use of `'value'` as a column name is explicitly forbidden across the codebase. Always use `'power'` for power flow data.
- **No FLAW IDs in Comments:** Do not reference FLAW-XXX IDs in code comments.
- **Code Comments:** Ensure any new or modified code includes comments explaining the *why* (intent and rationale) rather than the *how*.
