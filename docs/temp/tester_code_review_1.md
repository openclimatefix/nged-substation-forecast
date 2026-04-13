---
reviewer: "tester"
total_flaws: 2
critical_flaws: 0
test_status: "tests_failed"
---

# Testability & QA Review

## FLAW-001: Broken Test Mocks in `tests/test_plotting_robustness.py`
* **File & Line Number:** `tests/test_plotting_robustness.py`, lines 11 and 87
* **The Issue:** The tests are attempting to patch `get_cleaned_actuals_lazy` in `src.nged_substation_forecast.defs.plotting_assets`. This function does not exist in that module. It appears the function was renamed or moved, but the tests were not updated.
* **Concrete Failure Mode:** The test suite fails with an `AttributeError` during setup, preventing verification of the plotting logic.
* **Required Fix:** Update the tests to patch the correct function, which is likely `get_cleaned_power_time_series_lazy` imported from `.data_cleaning_assets`.

## FLAW-002: Mismatched Delta Table Names in Tests
* **File & Line Number:** `tests/test_xgboost_dagster_mocked.py` (lines 49, 54) and `tests/test_xgboost_forecaster.py` (lines 315, 318)
* **The Issue:** The tests are creating a Delta table named `cleaned_actuals`, but the application code (specifically in `src/nged_substation_forecast/defs/xgb_assets.py`) expects a table named `cleaned_power_time_series`.
* **Concrete Failure Mode:** The test suite fails with a `FileNotFoundError` when the application code attempts to read the expected Delta table, preventing verification of the XGBoost training and evaluation pipeline.
* **Required Fix:** Update the tests to create the Delta table with the name `cleaned_power_time_series` to match the application's expectations.
