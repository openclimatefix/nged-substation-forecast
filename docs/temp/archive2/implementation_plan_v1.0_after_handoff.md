---
status: "draft"
version: "v1.0"
after_reviewer: "handoff"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/xgboost_forecaster/src/xgboost_forecaster/model.py", "tests", "packages/xgboost_forecaster/tests"]
---

# Implementation Plan: Remove `target_map` and Fix Type Errors

## 1. Remove `target_map` from `XGBoostForecaster`
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`

*   **Class Attributes & Initialization:**
    *   Remove `target_map` from the `XGBoostForecaster` class attributes.
    *   Remove `self.target_map = None` from `__init__`.
*   **Methods to Remove:**
    *   Delete the `_get_capacity_map_df` method entirely.
*   **Method Updates:**
    *   `log_model`: Remove the logic that saves `target_map.json` as an MLflow artifact.
    *   `_prepare_data_for_model`:
        *   Remove the `target_map_df` parameter.
        *   Dynamically calculate peak capacity from the input `flows_30m` using `calculate_peak_capacity(flows_30m)`.
        *   Join the calculated peak capacity instead of the removed `target_map_df`.
    *   `train` & `predict`:
        *   Remove the `self.target_map is None` validation checks.
        *   Remove the `target_map_df` argument when calling `_prepare_data_for_model`.
    *   `predict`:
        *   Remove the validation block that checks if all substations in the inference set are present in the `target_map`.

## 2. Remove `forecaster.target_map` Assignments in Tests
**Files:**
*   `packages/xgboost_forecaster/tests/test_xgboost_model.py`
*   `tests/test_xgboost_adversarial.py`
*   `tests/test_xgboost_forecaster.py`
*   `tests/test_xgboost_robustness.py`
*   `tests/test_xgboost_mlflow.py`

*   **Updates:**
    *   Search for and remove all instances of `forecaster.target_map = ...`.
    *   Remove any local `target_map` variables or fixtures that were created solely to be assigned to the forecaster.
    *   Clean up unused imports (e.g., `calculate_target_map` or `calculate_peak_capacity`) if they are no longer needed in the test files.

## 3. Fix Type Errors and Deprecated Polars Calls
**File:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`

*   **Fix `to_series` on `LazyFrame` / `DataFrame` and Redundant Casts:**
    *   In `_prepare_features` (around line 100), the code currently does:
        ```python
        df_unique = df.select("time_series_id").unique()
        if isinstance(df_unique, pl.LazyFrame):
            df_unique = df_unique.collect()
        df_unique = cast(pl.DataFrame, df_unique)
        time_series_ids = [str(s) for s in df_unique.to_series().to_list()]
        ```
    *   **Refactor to:**
        ```python
        df_unique_lf_or_df = df.select("time_series_id").unique()
        df_unique_df = df_unique_lf_or_df.collect() if isinstance(df_unique_lf_or_df, pl.LazyFrame) else df_unique_lf_or_df
        time_series_ids = [str(s) for s in df_unique_df.get_column("time_series_id").to_list()]
        ```
        *(This removes the deprecated `to_series()` call and avoids the redundant `cast` by using a new variable name for the collected DataFrame).*

*   **Fix `to_series` in `train`:**
    *   Around line 471, the code currently does:
        ```python
        y = joined_df.select("value").to_series()
        ```
    *   **Refactor to:**
        ```python
        y = joined_df.get_column("value")
        ```

## Review Responses & Rejections
*(No reviews yet - initial draft)*
