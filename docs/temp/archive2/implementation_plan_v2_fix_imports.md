---
status: "draft"
version: "v2.0"
after_reviewer: "architect"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/ml_core", "packages/xgboost_forecaster", "packages/dashboard", "packages/nged_json_data"]
---
# Implementation Plan: Fix Broken Imports and Update Downstream Code

This plan addresses the broken imports and downstream code updates required following the recent data contract changes (specifically the introduction of `PowerTimeSeries` and `TimeSeriesMetadata`).

## 1. Mapping of Removed Classes/Constants to New Equivalents

The recent data contract changes simplified the schemas and unified the naming conventions. The following mappings apply:

| Old Class / Constant | New Equivalent | Notes |
| :--- | :--- | :--- |
| `SubstationMetadata` | `TimeSeriesMetadata` | `substation_number` is still present, but `time_series_id` is the new primary key for time series data. |
| `NgedJsonPowerFlows` | `PowerTimeSeries` | |
| `SubstationPowerFlows` | `PowerTimeSeries` | |
| `SimplifiedSubstationPowerFlows` | `PowerTimeSeries` | The new schema already enforces 30-minute intervals and a single `value` column. |
| `SubstationFeatures` | `XGBoostInputFeatures` | |
| `SubstationTargetMap` | *Removed* | Peak capacity calculation should be done directly on `PowerTimeSeries` (group by `time_series_id`, max absolute `value`). |
| `POWER_MW`, `POWER_MVA` | `value` | The ingestion pipeline now handles the selection between MW and MVA, providing a single `value` column. |
| `substation_number` | `time_series_id` | Use `time_series_id` (string) as the primary key for joining time series data, instead of `substation_number` (int). |
| `timestamp` | `end_time` | `PowerTimeSeries` uses `end_time` for the timestamp of the reading. |

## 2. Strategy for Updating Downstream Code

### `packages/ml_core`
*   **`src/ml_core/model.py` & `src/ml_core/experimental.py`**:
    *   Update imports: Replace `SubstationMetadata` with `TimeSeriesMetadata`.
    *   Update `BaseForecaster.train` signature: Change `substation_metadata: pt.DataFrame[SubstationMetadata]` to `substation_metadata: pt.DataFrame[TimeSeriesMetadata]`.
*   **`src/ml_core/data.py`**:
    *   Remove `calculate_target_map` and `downsample_power_flows`. The new `PowerTimeSeries` schema already enforces 30-minute intervals and a single `value` column.
    *   Add a new function `calculate_peak_capacity(flows: pt.LazyFrame[PowerTimeSeries]) -> pt.DataFrame` that groups by `time_series_id` and calculates the maximum absolute `value` (aliased as `peak_capacity_MW_or_MVA` for backward compatibility with the model, or rename it to `peak_capacity`).
*   **`src/ml_core/utils.py`**:
    *   Update `_slice_temporal_data` and `evaluate_and_save_model` to use `time_series_id` instead of `substation_number`.
    *   Update `evaluate_and_save_model` to use `end_time` instead of `timestamp` for power flows.
    *   Update `evaluate_model` to use the `value` column instead of `POWER_MW` or `POWER_MVA`.

### `packages/xgboost_forecaster`
*   **`src/xgboost_forecaster/model.py`**:
    *   Update imports: Replace `SubstationFeatures` with `XGBoostInputFeatures`.
    *   Replace `substation_number` with `time_series_id` throughout the file (e.g., in joins, categorical feature casting, and missing data checks).
    *   Replace `timestamp` with `end_time` when referencing power flow data.
    *   Remove calls to `calculate_target_map` and `downsample_power_flows`. Use the new `calculate_peak_capacity` from `ml_core.data`.
    *   Rename `_get_target_map_df` to `_get_capacity_map_df`.
*   **`src/xgboost_forecaster/features.py`**:
    *   Update docstrings to reference `XGBoostInputFeatures` instead of `SubstationFeatures`.
    *   Ensure `time_series_id` is used instead of `substation_number` for grouping and joining.
*   **`src/xgboost_forecaster/data.py`**:
    *   Update `join_data` (if applicable) or any data loading functions to join on `time_series_id` instead of `substation_number`.

### `packages/dashboard`
*   **`main.py`**:
    *   Ensure it uses `time_series_id` and `value` instead of `substation_number` and `POWER_MW`/`POWER_MVA`.
    *   Remove any fallback logic that references `POWER_MW` or `POWER_MVA` strings.

### `packages/nged_json_data`
*   Already uses `PowerTimeSeries`, but verify if any internal logic still assumes `timestamp` instead of `end_time`.

## 3. Strategy for Updating Tests

*   **`packages/ml_core/tests/test_target_selection.py`**:
    *   Rename to `test_peak_capacity.py`.
    *   Rewrite tests to verify `calculate_peak_capacity` using `PowerTimeSeries` mock data (with `time_series_id` and `value`).
*   **`packages/ml_core/tests/test_ml_core_model.py`**:
    *   Update mock data and type hints to use `TimeSeriesMetadata` and `PowerTimeSeries`.
*   **`packages/xgboost_forecaster/tests/`**:
    *   Update `test_universal_model.py`, `test_xgboost_model.py`, and `test_xgboost_features.py` to use `time_series_id`, `PowerTimeSeries`, and `XGBoostInputFeatures`.
    *   Ensure mock data uses `end_time` instead of `timestamp`.

## 4. Verification Plan

1.  **Type Checking**: Run `uv run mypy packages/` to ensure all type hints align with the new data contracts.
2.  **Unit Tests**: Run `uv run pytest packages/` to verify that all tests pass and no `ImportError` or `KeyError` (e.g., missing `substation_number` or `timestamp`) occurs.
3.  **Data Contract Validation**: Ensure that the mock data used in tests successfully passes the `.validate()` methods of `PowerTimeSeries`, `TimeSeriesMetadata`, and `XGBoostInputFeatures`.
4.  **Integration Check**: Run a dry-run of the XGBoost training pipeline (if a script is available) to ensure data flows correctly from ingestion to model training without schema mismatches.

## Review Responses & Rejections

*(To be filled after review)*
