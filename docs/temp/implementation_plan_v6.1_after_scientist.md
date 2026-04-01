---
status: "draft"
version: "v6.1"
after_reviewer: "scientist"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/ml_core", "tests"]
---

# Implementation Plan: Fix MLflow, Unit Tests, and LSP Errors

This plan addresses the flaws identified in the Scientist's review (Iteration 6) and additional LSP errors.

## 1. MLflow Experiment Issue (`packages/ml_core/src/ml_core/utils.py`)

**Issue:** `mlflow.start_run()` is called without explicitly setting an experiment. If the default experiment (ID 0) is deleted or corrupted, MLflow throws an exception.
**Action:**
- In `train_and_log_model`, add `mlflow.set_experiment(model_name)` immediately before `with mlflow.start_run(run_name=model_name) as run:`.
- In `evaluate_and_save_model`, add `mlflow.set_experiment(model_name)` immediately before `with mlflow.start_run(run_name=f"{model_name}_eval"):`.
**Rationale:** Explicitly setting the experiment ensures MLflow creates it if it doesn't exist, preventing "Experiment ID 0 not found" errors and keeping runs organized by model name.

## 2. Unit Test Failures (`tests/test_xgboost_forecaster.py`)

**Issue:** The `test_train_xgboost_asset_filters_to_control_member` unit test calls the `train_xgboost` asset with incorrect arguments (4 instead of 6), causing a `TypeError` when Dagster tries to resolve the configuration.
**Action:**
- Update `test_train_xgboost_asset_filters_to_control_member` to instantiate an `XGBoostConfig` object and a dummy `healthy_substations` list.
- Pass the arguments in the correct order: `train_xgboost(context=context, config=config, nwp=nwp, substation_power_flows=flows, substation_metadata=metadata, healthy_substations=healthy_substations)`.
- Ensure any other tests calling `train_xgboost` or `evaluate_xgboost` are also updated if necessary.
**Rationale:** Dagster assets require all defined inputs to be provided in the correct order when called directly in unit tests.

## 3. LSP Errors (`tests/test_xgboost_dagster_integration.py`)

**Issue:** Type checker/LSP errors related to `max_dt` and `min_dt` when extracting dates from Polars DataFrames.
**Action:**
- Ensure `max_dt` and `min_dt` are properly cast to `datetime` before calling `.date()`.
- Example: `test_end = cast(datetime, max_dt).date()` and `train_start = cast(datetime, min_dt).date()`.
- Add a check or default fallback in case `min_dt` is `None` to satisfy strict type checking and prevent runtime errors.
**Rationale:** Polars `.min()` and `.max()` on a column can return `Any` or `None`. Explicit casting and `None` checks satisfy the LSP and ensure runtime safety.

## Review Responses & Rejections

* **FLAW-001 (Scientist):** ACCEPTED. Explicitly setting the MLflow experiment prevents errors related to the default experiment ID 0 and improves run organization.
* **FLAW-002 (Scientist):** ACCEPTED. Updating the unit test to pass the correct arguments resolves the `TypeError` and aligns with the asset's signature.
