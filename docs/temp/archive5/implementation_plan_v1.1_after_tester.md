---
status: "draft"
version: "v1.1"
after_reviewer: "tester"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/contracts", "packages/nged_data", "packages/xgboost_forecaster", "tests", "src/nged_substation_forecast/defs", "packages/ml_core"]
---

# Implementation Plan: Fix Test Failures and Add SharePoint Asset Test

## Review Responses & Rejections

* **FLAW-001 (Tester):** ACCEPTED. The test suite is failing due to recent schema changes in `PowerTimeSeries`, `PowerForecast`, and `TimeSeriesMetadata`. The tests and some ML core code need to be updated to match the new schemas.
* **FLAW-002 (Tester):** ACCEPTED. A test for `nged_sharepoint_json_asset` will be added to `tests/test_nged_assets.py`.

## 1. Update `TimeSeriesMetadata` Schema
**File:** `packages/contracts/src/contracts/data_schemas.py`
*   **Action:** Update `area_wkt`, `area_center_lat`, and `area_center_lon` in `TimeSeriesMetadata` to be optional (`| None` and `allow_missing=True`). This is necessary because not all NGED JSON files (e.g., CKAN resources) contain the `Area` struct.
*   **Rationale:** Ensures backward compatibility with existing JSON files that lack area information while supporting the new SharePoint JSON files that include it.

## 2. Update `load_nged_json` to Extract Area Fields
**File:** `packages/nged_data/src/nged_data/io.py`
*   **Action:** Modify `load_nged_json` to extract `WKT`, `CenterLat`, and `CenterLon` from the `Area` struct if it exists in the JSON, and map them to `area_wkt`, `area_center_lat`, and `area_center_lon` respectively.
*   **Rationale:** Correctly parses the nested `Area` struct from the new SharePoint JSON files into the flat `TimeSeriesMetadata` schema.

## 3. Fix `PowerTimeSeries` and `PowerForecast` Tests
**File:** `packages/contracts/tests/test_data_schemas.py`
*   **Action:** Update `test_power_time_series_validation_mw_or_mva`, `test_power_time_series_validation_both`, and `test_power_time_series_property_based` to use the new `PowerTimeSeries` schema:
    *   Change `time_series_id` to `pl.Int32` (e.g., `123` instead of `"123"`).
    *   Rename `end_time` to `period_end_time`.
    *   Rename `value` to `power`.
    *   Remove `start_time`.
*   **Action:** Update `test_power_forecast_validation` to use the new `PowerForecast` schema:
    *   Add `time_series_id` (e.g., `123`).
    *   Rename `MW_or_MVA` to `power_fcst`.
    *   Remove `substation_number`.
*   **Rationale:** Aligns the tests with the updated data contracts.

## 4. Fix ML Core and XGBoost Tests
**Files:**
*   `packages/xgboost_forecaster/tests/test_universal_model.py`
*   `packages/xgboost_forecaster/tests/test_xgboost_model.py`
*   `packages/xgboost_forecaster/tests/test_xgboost_features.py`
*   `tests/test_data_cleaning_robustness.py`
*   `tests/test_plotting_robustness.py`
*   `packages/ml_core/src/ml_core/utils.py`
*   **Action:** In all test files where `flows_data` or similar mock data is created for `PowerTimeSeries`, replace `"value"` with `"power"` and remove `"start_time"`. Ensure `period_end_time` is used instead of `end_time` or `start_time` where appropriate. Ensure `time_series_id` is cast to `pl.Int32`.
*   **Action:** In `tests/test_plotting_robustness.py`, update the mock `TimeSeriesMetadata` to use the new schema fields (`time_series_id`, `time_series_name`, `time_series_type`, `units`, `licence_area`, `substation_number`, `substation_type`, `latitude`, `longitude`) instead of the old ones (`substation_name_in_location_table`, `preferred_power_col`, `last_updated`).
*   **Action:** In `packages/ml_core/src/ml_core/utils.py`, update `train_and_log_model` to check for `time_series_metadata` instead of `substation_metadata` when skipping temporal slicing.
*   **Rationale:** Fixes the `polars.exceptions.ColumnNotFoundError: unable to find column "power"`, `patito.exceptions.DataFrameValidationError`, and `TypeError: XGBoostForecaster.train() missing 1 required positional argument: 'time_series_metadata'` errors during test execution.

## 5. Add Test for `nged_sharepoint_json_asset`
**File:** `tests/test_nged_assets.py`
*   **Action:** Add a new test function `test_nged_sharepoint_json_asset`.
*   **Action:** Mock `Path.exists` and `Path.glob` to simulate the presence of JSON files in the hardcoded SharePoint directory.
*   **Action:** Mock `load_nged_json`, `clean_power_data`, and `append_to_delta` similarly to the existing asset tests.
*   **Action:** Verify that the asset correctly processes the mocked files and calls the mocked functions.
*   **Rationale:** Fulfills FLAW-002 and ensures the new asset is covered by the test suite.

## 6. Add `substation_metadata` Asset
**File:** `src/nged_substation_forecast/defs/nged_assets.py`
*   **Action:** Add a new Dagster asset named `substation_metadata` that reads the Parquet file at `settings.nged_data_path / "metadata" / "json_metadata"` and returns it as a Polars DataFrame.
*   **Action:** Use `polars_h3` to compute and add the `h3_res_5` column to the DataFrame, as downstream assets (`processed_nwp_data`) depend on it.
*   **Rationale:** Fixes the `DagsterInvalidDefinitionError` caused by the removal of the old `substation_metadata` asset. Downstream assets (`processed_nwp_data`, `forecast_vs_actual_plot`) depend on this asset.

## 7. Fix `upsert_metadata` Bug
**File:** `packages/nged_data/src/nged_data/metadata.py`
*   **Action:** Modify `upsert_metadata` to properly merge the new metadata with the existing metadata (e.g., using `pl.concat` and `unique(subset=["time_series_id"], keep="last")`) instead of overwriting the entire file with the metadata of a single substation.
*   **Rationale:** Prevents data loss where the metadata file only contains the metadata of the last processed JSON file.
