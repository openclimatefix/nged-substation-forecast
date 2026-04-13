---
status: "draft"
version: "v1.3"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["tests"]
---

# Implementation Plan: Fix Test Mocks and Delta Table Names

This plan addresses the flaws identified in `docs/temp/tester_code_review_1.md` regarding broken test mocks and mismatched Delta table names in the test suite.

## 1. Fix Broken Test Mocks in `tests/test_plotting_robustness.py`

The tests in `tests/test_plotting_robustness.py` are currently attempting to patch a non-existent function `get_cleaned_actuals_lazy`. This function was renamed to `get_cleaned_power_time_series_lazy`.

**Instructions for Builder:**
- In `tests/test_plotting_robustness.py`, update the `@patch` decorators on lines 11 and 87.
- Change `@patch("src.nged_substation_forecast.defs.plotting_assets.get_cleaned_actuals_lazy")` to `@patch("src.nged_substation_forecast.defs.plotting_assets.get_cleaned_power_time_series_lazy")`.
- Ensure that the mock argument names in the test function signatures (e.g., `mock_get_lazy`) remain the same, or update them if you prefer, but the patch target must be corrected.

## 2. Update Delta Table Names in XGBoost Tests

The tests `tests/test_xgboost_dagster_mocked.py` and `tests/test_xgboost_forecaster.py` are creating a mock Delta table named `cleaned_actuals`. The application code expects this table to be named `cleaned_power_time_series`.

**Instructions for Builder:**

**In `tests/test_xgboost_dagster_mocked.py`:**
- Locate the section where the Delta tables are created (around line 49).
- Change the variable `cleaned_actuals_path` to `cleaned_power_time_series_path`.
- Update the path string from `"cleaned_actuals"` to `"cleaned_power_time_series"`.
- Update the `mkdir` and `write_delta` calls to use the new variable name.

**In `tests/test_xgboost_forecaster.py`:**
- Locate the `test_train_xgboost_asset_filters_to_control_member` function (around line 315).
- Change the variable `cleaned_actuals_path` to `cleaned_power_time_series_path`.
- Update the path string from `"cleaned_actuals"` to `"cleaned_power_time_series"`.
- Update the `mkdir` and `write_delta` calls to use the new variable name.

## General Guidelines
- **No FLAW IDs in Comments:** Do not include `FLAW-XXX` IDs in any code comments.
- **Focus on the Why:** If adding any comments, ensure they explain *why* the code is doing something, not *how*. In this case, the changes are straightforward test fixes, so additional comments may not be necessary.

## Review Responses & Rejections

* **FLAW-001 (tester):** ACCEPTED. The mock target `get_cleaned_actuals_lazy` is outdated and must be updated to `get_cleaned_power_time_series_lazy` to match the current codebase.
* **FLAW-002 (tester):** ACCEPTED. The mock Delta table name `cleaned_actuals` is outdated and must be updated to `cleaned_power_time_series` to match the application's expectations.
