---
status: "draft"
version: "v1.2"
after_reviewer: "tester"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/contracts", "src/nged_substation_forecast", "packages/ml_core", "tests"]
---

# Implementation Plan: Fix Tester Errors

## 1. Schema Mismatch
**File:** `packages/contracts/src/contracts/data_schemas.py`
**Action:** Add `preferred_power_col` and `substation_name_in_location_table` to the `TimeSeriesMetadata` class.
**Details:**
```python
    preferred_power_col: str | None = pt.Field(dtype=pl.String, allow_missing=True)
    substation_name_in_location_table: str | None = pt.Field(dtype=pl.String, allow_missing=True)
```
**Why:** The tester identified a schema mismatch. These fields are required by downstream components but are missing from the contract.

## 2. Remove Asset References
**File:** `src/nged_substation_forecast/defs/xgb_jobs.py`
**Action:** Remove `"substation_metadata"` from the `AssetSelection.assets()` list in `xgboost_integration_job`.
**Why:** The `substation_metadata` asset has been removed from the pipeline, so it must be removed from the job definition to prevent Dagster from failing to resolve the asset.

**File:** `src/nged_substation_forecast/defs/xgb_assets.py`
**Action:** Remove the outdated comment in the `train_xgboost` docstring that mentions `substation_metadata` (`- _get_target_time_series now uses substation_metadata as an efficient fallback...`).
**Why:** To keep documentation accurate and avoid confusion.

## 3. Argument Mismatch
**File:** `packages/ml_core/src/ml_core/utils.py`
**Action:** Update `train_and_log_model` to use `time_series_metadata` instead of `substation_metadata`.
**Details:**
Change:
```python
        if key == "substation_metadata":
            sliced_data[key] = val
            continue
```
To:
```python
        if key == "time_series_metadata":
            sliced_data[key] = val
            continue
```
**Why:** The argument name was changed to `time_series_metadata` to reflect the new schema, but `train_and_log_model` was still looking for the old name.

## 4. Update Tests
**File:** `tests/test_xgboost_dagster_mocked.py`
**Action:** Remove the `sub_meta` dummy data definition, as it is no longer used by the assets.
**Why:** The tests need to reflect the updated asset signatures and job definitions.

## Review Responses & Rejections
* **FLAW-XXX (Tester):** ACCEPTED. The schema mismatch, removed asset references, and argument mismatches are all valid errors that need to be fixed.
