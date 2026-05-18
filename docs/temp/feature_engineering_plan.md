---
status: "draft"
version: "v1.1"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/ml_core/src/ml_core", "packages/contracts/src/contracts", "packages/ml_core/tests"]
---
# Feature Engineering Implementation Plan

## Overview
This plan implements a "Hybrid" feature engineering architecture. 
1. **Contracts (`contracts/ml_schemas.py`)**: Acts as the central, human-readable "menu" of all available features and provides strict type validation for static features.
2. **Logic (`ml_core/features.py`)**: Contains the actual Polars expressions (Registry for static features, Factories for dynamic/parameterized features).
3. **Orchestration (`ml_core/base_forecaster.py`)**: Joins the raw data, parses the requested features, applies the Polars expressions, dynamically asserts that all requested features were created, and finally casts the result to the Patito contract.

## Step 1: Update the Data Contract
**File:** `packages/contracts/src/contracts/ml_schemas.py`

*   **Action:** Update the docstring of the `AllFeatures` class.
*   **Details:** The docstring must explicitly document the dynamic feature patterns so that new data scientists know what they can request in their model configs.
*   **Example Docstring Addition:**
    ```python
    """Final joined dataset ready for ML models.
    
    DYNAMIC FEATURES:
    In addition to the explicitly defined columns below, the pipeline supports 
    dynamically generated features. You can request these in your model config:
    
    * `power_lag_{hours}`: The power value shifted by X hours (e.g., `power_lag_24`).
    * `temperature_rolling_mean_{hours}`: Rolling average of temperature over X hours.
    """
    ```
*   **Commenting Mandate:** Add a comment explaining *why* dynamic features are not explicitly typed as Pydantic fields (to allow infinite parameterization without metaprogramming overhead).

## Step 2: Implement the Feature Registry & Factories
**File:** `packages/ml_core/src/ml_core/features.py` (Create this file)

*   **Action:** Implement the Polars expression registry and factory functions.
*   **Details:**
    *   Define `STATIC_FEATURE_REGISTRY: dict[str, pl.Expr]`. Populate it with basic static features (e.g., `local_time_of_day_sin`, `local_day_of_week`).
    *   Define factory functions for dynamic features:
        *   `build_lag_expr(base_col: str, lag: int) -> pl.Expr`
        *   `build_rolling_mean_expr(base_col: str, window: int) -> pl.Expr`
*   **Commenting Mandate:** Every factory function must have a docstring explaining *why* it is a factory rather than a static registry entry (to support dynamic Hydra configuration). Comments must connect these functions back to the `AllFeatures` contract documentation.

## Step 3: Refactor `BaseForecaster`
**File:** `packages/ml_core/src/ml_core/base_forecaster.py`

*   **Action:** Rewrite `_engineer_features` to join data, use the new registry, and perform two-step validation.
*   **Details:**
    1.  **Data Joining:** Join `power_time_series` and `time_series_metadata` on `time_series_id`. If `nwp` is provided, join it as well (on `time_series_id` and `valid_time`). This creates `raw_data`.
    2.  **TODO Management:** 
        *   *Remove* the TODO about Patito `derive`.
        *   *Keep* the TODOs about converting NWP to `NwpInMemory` and interpolating NWP to half-hourly. Explicitly state in comments that NWP interpolation is out of scope for this step.
    3.  **Feature Generation:** Iterate over `self.selected_features`.
        *   If the feature is in `STATIC_FEATURE_REGISTRY`, append the expression.
        *   If the feature matches a dynamic pattern (e.g., `.startswith("power_lag_")`), parse the integer and call the appropriate factory, appending the expression.
    4.  **Apply:** `engineered_lf = raw_data.with_columns(exprs_to_evaluate)`
    5.  **Dynamic Schema Assertion:** Extract `engineered_lf.collect_schema().names()`. Calculate `missing_cols = set(self.selected_features) - set(available_columns)`. If `missing_cols` is not empty, raise a `ValueError`.
    6.  **Select & Cast:** Select only the required base columns (`valid_time`, `time_series_id`, `time_series_type`, `power`, `lead_time_hours`) plus `self.selected_features`. Return `pt.LazyFrame[AllFeatures](final_lf)`.
*   **Commenting Mandate:** Add inline comments explaining the two-step validation process: *why* we check the Polars schema dynamically (to catch typos in dynamic features) before casting to the Patito model (which ignores extra columns).

## Step 4: Testing Strategy
**Files:** `packages/ml_core/tests/test_features.py` and `packages/ml_core/tests/test_base_forecaster.py`

*   **Action:** Implement rigorous unit and integration tests.
*   **Details:**
    *   **`test_features.py`:** 
        *   Test that `build_lag_expr` and `build_rolling_mean_expr` generate valid Polars expressions that produce the expected mathematical output on a dummy `pl.DataFrame`.
    *   **`test_base_forecaster.py`:**
        *   Create a dummy subclass of `BaseForecaster` for testing.
        *   **Success Case:** Pass a list of `selected_features` containing both static and dynamic features. Assert that the resulting `LazyFrame` contains exactly those columns.
        *   **Failure Case (Typo):** Pass a misspelled feature (e.g., `"power_lagg_24"`). Assert that `_engineer_features` raises a `ValueError` detailing the missing column.
        *   **Failure Case (Missing Base Data):** Request a feature that depends on a missing base column. Assert it fails cleanly.

## Review Responses & Rejections
*(To be filled during the review phase)*
