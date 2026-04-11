---
status: "draft"
version: "v1.0"
after_reviewer: "architect"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["packages/contracts", "packages/nged_json_data", "packages/ml_core", "packages/xgboost_forecaster", "src/nged_substation_forecast", "tests"]
---

# Implementation Plan: Data Contract Refactoring

## Objective
Refactor the core data contracts to improve type safety and naming consistency:
1. Change the data type of `time_series_id` from `str` to `int32` across all schemas and usages.
2. Rename `value` to `power` in `PowerTimeSeries` and downstream usages.
3. Rename `end_time` to `period_end_time` in `PowerTimeSeries` and downstream usages.

## 1. Data Contracts (`packages/contracts/src/contracts/data_schemas.py`)

*   **`PowerTimeSeries`**:
    *   Change `time_series_id: str = pt.Field(dtype=pl.String)` to `time_series_id: int = pt.Field(dtype=pl.Int32)`.
    *   Rename `end_time` to `period_end_time`.
    *   Rename `value` to `power`.
    *   Update the `validate` method to reference `period_end_time` instead of `end_time`.
*   **`TimeSeriesMetadata`**:
    *   Change `time_series_id: str = pt.Field(dtype=pl.String, unique=True)` to `time_series_id: int = pt.Field(dtype=pl.Int32, unique=True)`.
*   **`PowerForecast`**:
    *   Change `time_series_id: str = pt.Field(dtype=pl.String)` to `time_series_id: int = pt.Field(dtype=pl.Int32)`.
*   **`XGBoostInputFeatures`**:
    *   Change `time_series_id: str = pt.Field(dtype=pl.String)` to `time_series_id: int = pt.Field(dtype=pl.Int32)`.

## 2. Data Ingestion & Cleaning (`packages/nged_json_data/`)

*   **`src/nged_json_data/io.py`**:
    *   In `read_json_to_dataframe`, update the schema definition to use `pl.Int32` for `time_series_id`, `period_end_time` for the datetime, and `power` for the float.
    *   Rename `endTime` to `period_end_time` during the rename operation.
    *   Ensure `time_series_id` is cast to `pl.Int32`.
    *   Select `["time_series_id", "period_end_time", "power"]`.
*   **`src/nged_json_data/clean.py`**:
    *   Replace all occurrences of `"end_time"` with `"period_end_time"`.
    *   Replace all occurrences of `"value"` with `"power"`.
*   **`src/nged_json_data/storage.py`**:
    *   Replace all occurrences of `"end_time"` with `"period_end_time"`.

## 3. ML Core (`packages/ml_core/`)

*   **`src/ml_core/utils.py`**:
    *   Replace `"end_time"` with `"period_end_time"`.
    *   Update `actuals.rename({"end_time": "valid_time", "value": "actual"})` to `actuals.rename({"period_end_time": "valid_time", "power": "actual"})`.
*   **`src/ml_core/experimental.py`**:
    *   Ensure `time_series_id` filtering works with integers instead of strings.

## 4. XGBoost Forecaster (`packages/xgboost_forecaster/`)

*   **`src/xgboost_forecaster/model.py`**:
    *   Replace `"end_time"` with `"period_end_time"`.
    *   In `_prepare_features`, when casting `time_series_id` to `pl.Enum`, ensure the integer is cast to string first: `pl.col("time_series_id").cast(pl.String).cast(pl.Enum(time_series_ids))`. The `time_series_ids` list should contain string representations of the integers.
*   **`src/xgboost_forecaster/features.py`**:
    *   Replace `"end_time"` with `"period_end_time"`.

## 5. Dagster Assets (`src/nged_substation_forecast/defs/`)

*   **`nged_assets.py`**:
    *   Ensure `time_series_id` is treated as an integer.
*   **`data_cleaning_assets.py`**:
    *   Replace `"value"` with `"power"`.
*   **`xgb_assets.py`**:
    *   Replace `"value"` with `"power"`.
    *   Replace `"end_time"` with `"period_end_time"`.
*   **`plotting_assets.py`**:
    *   Replace `"end_time"` with `"period_end_time"`.
    *   Replace `"value"` with `"power"`.

## 6. Tests

*   Update all test files across `packages/contracts`, `packages/nged_json_data`, `packages/ml_core`, `packages/xgboost_forecaster`, and `tests/` to reflect the new schema:
    *   Change string `"1"`, `"123"`, etc. for `time_series_id` to integers `1`, `123`.
    *   Rename `"end_time"` keys in dictionaries/dataframes to `"period_end_time"`.
    *   Rename `"value"` keys in dictionaries/dataframes to `"power"`.

## 7. Code Comments & Rationale

*   **Mandatory Comments**: When updating the `validate` method in `PowerTimeSeries` or the IO parsing logic, ensure comments explicitly state *why* `period_end_time` is used (to clarify that the timestamp represents the end of the 30-minute settlement period) and *why* `time_series_id` is an `int32` (for memory efficiency and consistency with substation numbers).
*   **No FLAW IDs**: Do not include any FLAW-XXX IDs in the source code comments.

## Review Responses & Rejections
*   N/A (Initial Draft)

## 8. Notebooks & Dashboards

*   **`packages/notebooks/NGED_JSON_data_from_sharepoint.py`**:
    *   Update any references to `"value"` to `"power"`.
    *   Update any references to `"endTime"` or `"end_time"` to `"period_end_time"`.
