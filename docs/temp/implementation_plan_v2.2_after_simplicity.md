---
status: "draft"
version: "v2.2"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/xgb_assets.py"]
---

# Implementation Plan: Simplicity Refactoring for XGBoost Assets

## Overview

This plan addresses the flaws identified in `docs/temp/simplicity_review_2.md`. The primary goal is to refactor `train_xgboost` and `evaluate_xgboost` in `src/nged_substation_forecast/defs/xgb_assets.py` to eliminate significant code duplication and reduce function complexity. By extracting the common setup logic into a shared helper function, we ensure the main asset functions focus purely on high-level orchestration, improving readability and maintainability.

## Implementation Steps

### 1. Create `_prepare_xgboost_inputs` Helper Function
**Target File:** `src/nged_substation_forecast/defs/xgb_assets.py`

Add a new helper function `_prepare_xgboost_inputs` to centralize the data preparation and configuration loading logic.

*   **Arguments:** `context: dg.AssetExecutionContext`, `config: XGBoostConfig`, `settings: dg.ResourceParam[Settings]`, `model_name: str = "xgboost"`.
*   **Logic to Move:**
    *   Setting the MLflow tracking URI (`mlflow.set_tracking_uri`).
    *   Loading and overriding the Hydra configuration (`load_hydra_config` and `_apply_config_overrides`).
    *   Loading `time_series_metadata`.
    *   Loading `power_time_series` via `get_cleaned_power_time_series_lazy`.
    *   Identifying `healthy_time_series` (filtering for non-null power and collecting unique IDs).
    *   Calling `_get_target_time_series` to get `sub_ids`.
    *   Filtering `power_time_series` and `time_series_metadata` by `sub_ids`.
*   **Return Value:** A tuple containing `(hydra_config, power_time_series_filtered, time_series_metadata_filtered, sub_ids)`.
*   **Mandatory Commenting:** Add a comprehensive docstring explaining *why* this helper exists (to centralize data preparation, ensure consistency between training and evaluation, and reduce boilerplate in the main assets). Do not reference FLAW IDs in the code comments.

### 2. Refactor `train_xgboost` Asset
**Target File:** `src/nged_substation_forecast/defs/xgb_assets.py`

Refactor the `train_xgboost` asset to use the new helper function.

*   **Action:** Replace the duplicated setup logic (lines ~127-185) with a single call to `_prepare_xgboost_inputs`.
*   **Retained Logic:**
    *   The early return if `sub_ids` is empty (returning an untrained `XGBoostForecaster`).
    *   The NWP filtering logic (`nwp_train = nwp.filter(pl.col("ensemble_member") == 0)`).
    *   The call to `train_and_log_model`.
*   **Mandatory Commenting:** Ensure the asset's docstring reflects that it now focuses purely on orchestrating the training process, delegating data preparation to the helper.

### 3. Refactor `evaluate_xgboost` Asset
**Target File:** `src/nged_substation_forecast/defs/xgb_assets.py`

Refactor the `evaluate_xgboost` asset to use the new helper function.

*   **Action:** Replace the duplicated setup logic (lines ~221-275) with a single call to `_prepare_xgboost_inputs`.
*   **Retained Logic:**
    *   The early return if `sub_ids` is empty (returning an empty `PowerForecast` dataframe).
    *   Logging the shapes and ensemble members for debugging.
    *   Setting `forecaster.config = hydra_config.model`.
    *   The call to `evaluate_and_save_model`.
*   **Mandatory Commenting:** Ensure the asset's docstring reflects that it now focuses purely on orchestrating the evaluation process.

## Review Responses & Rejections

*   **FLAW-001 ([simplicity_review]):** ACCEPTED. The duplicated setup logic (loading config, fetching data, identifying healthy substations, filtering) will be extracted into a shared `_prepare_xgboost_inputs` helper function.
*   **FLAW-002 ([simplicity_review]):** ACCEPTED. The main asset functions (`train_xgboost` and `evaluate_xgboost`) will be refactored to delegate data preparation to the new helper, significantly reducing their length and complexity to focus on high-level orchestration.
