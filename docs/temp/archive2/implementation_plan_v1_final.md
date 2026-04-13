---
status: "draft"
version: "v1"
after_reviewer: "merged"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["packages/nged_data", "packages/contracts", "packages/ml_core", "packages/xgboost_forecaster", "src/nged_substation_forecast", "packages/dashboard"]
---

# Implementation Plan: Removing CKAN Code and Updating Data Contracts

This plan details the steps to remove the deprecated NGED CKAN data ingestion code and update the data contracts to reflect the new JSON-based data source.

## 1. Remove `packages/nged_data`

*   **Action:** Delete the entire `packages/nged_data` directory.
*   **Action:** Remove `nged-data` from `pyproject.toml` in both the `dependencies` list and the `[tool.uv.sources]` section.
*   **Rationale:** The CKAN data source is deprecated, and the code is no longer needed. Removing it simplifies the codebase.

## 2. Remove NGED CKAN Dagster Assets

*   **Action:** In `src/nged_substation_forecast/defs/nged_assets.py`, delete the following assets and checks:
    *   `live_primary_flows`
    *   `substation_metadata`
    *   `substation_power_preferences`
    *   `substation_data_quality`
*   **Action:** Remove the `LivePrimaryFlowsConfig` class.
*   **Action:** Remove imports from `nged_data` in `nged_assets.py`.
*   **Rationale:** These assets are tied to the deprecated CKAN data source.

## 3. Update Data Contracts (`packages/contracts/src/contracts/data_schemas.py`)

*   **Action:** Delete `SubstationPowerFlows`.
*   **Action:** Rename `NgedJsonPowerFlows` to `PowerTimeSeries`.
*   **Action:** Delete `SubstationLocations` and `SubstationLocationsWithH3`.
*   **Action:** Update `PowerForecast`:
    *   Rename `MW_or_MVA` to `power_fcst`.
    *   Replace `substation_number` with `time_series_id` (type `pl.String`).
*   **Action:** Rename `SubstationMetadata` to `TimeSeriesMetadata`.
    *   Remove fields: `substation_name_in_location_table`, `substation_name_in_live_primaries`, `url`, `preferred_power_col`.
*   **Action:** Rename `SubstationFeatures` to `XGBoostInputFeatures`.
    *   Replace `substation_number` with `time_series_id` (type `pl.String`).
    *   Rename `MW_or_MVA` to `power`.
    *   Add `time_series_type` (type `pl.Categorical` or `pl.String`).
*   **Action:** Simplify MW vs MVA:
    *   Delete constants: `POWER_MW`, `POWER_MVA`, `POWER_MW_OR_MVA`, `PowerColumn`.
    *   Delete `MissingCorePowerVariablesError`.
    *   Delete `SimplifiedSubstationPowerFlows`.
    *   Delete `SubstationTargetMap`.
*   **Action:** Add Data Validation Logic:
    *   Implement validation logic for the new JSON data to ensure that the time span between `startTime` and `endTime` in the JSON's `data` is always exactly one half hour.
    *   Validate that the `endTime` is always at the top or bottom of the hour (e.g., minute 00 or 30).
*   **Rationale:** The new JSON data provides a single `value` column and uses `time_series_id` instead of `substation_number`. The contracts must reflect this simpler structure. The added validation ensures data alignment and prevents lookahead bias.

## 4. Remove Downsampling Code

*   **Action:** In `packages/ml_core/src/ml_core/data.py`, delete the functions `calculate_target_map` and `downsample_power_flows`.
*   **Action:** In `packages/ml_core/src/ml_core/utils.py`, remove the calls to `downsample_power_flows` in `train_and_log_model` and `evaluate_and_save_model`. The input data (`power_time_series`) will now be expected to be half-hourly and have a `value` column (which should be renamed to `power` or `power_fcst` as needed by the model). Remove any logic related to `target_map`.
*   **Action:** Verification Step: Ensure that the removal of `downsample_power_flows` and `target_map` does not leave any orphaned imports or unused helper functions in `packages/ml_core/src/ml_core/utils.py` or elsewhere. Perform a grep for these names after removal.
*   **Rationale:** The new JSON data is already half-hourly, so downsampling is no longer required. The `target_map` concept is obsolete.

## 5. Update Downstream Code

*   **Action:** Update `packages/xgboost_forecaster` to use `XGBoostInputFeatures` instead of `SubstationFeatures`, and `PowerTimeSeries` instead of `SubstationPowerFlows`. Update feature engineering to use `power` instead of `MW_or_MVA` and `time_series_id` instead of `substation_number`.
*   **Action:** Update `packages/dashboard/main.py` to use the new `TimeSeriesMetadata` and `PowerTimeSeries` contracts, and remove references to `preferred_power_col` and `POWER_MW`.
*   **Action:** Update `packages/nged_json_data` to use `PowerTimeSeries` instead of `NgedJsonPowerFlows`.
*   **Action:** Ensure all code that previously joined on `substation_number` now joins on `time_series_id`.
*   **Rationale:** All components must be aligned with the new data contracts to ensure the pipeline functions correctly.

## 6. Update Tests

*   **Action:** Audit and update *all* test suites across the repository (`packages/contracts`, `packages/xgboost_forecaster`, `packages/dashboard`, `packages/ml_core`, etc.) to reflect the new data contracts.
*   **Action:** Identify and remove any unnecessary or redundant unit tests (e.g., tests for the removed CKAN data ingestion, downsampling logic, or obsolete contracts) to keep the test suite clean and maintainable.
*   **Action:** Implement property-based tests (using `hypothesis`) for the new `PowerTimeSeries`, `TimeSeriesMetadata`, and `XGBoostInputFeatures` contracts to ensure they correctly validate the new JSON structure.
*   **Action:** Create new unit tests for the data transformation logic (e.g., column renaming, joining on `time_series_id`) to ensure data integrity during the migration.
*   **Rationale:** Comprehensive testing is required to prevent silent failures and ensure the new schemas are correctly enforced. Removing obsolete tests prevents confusion and reduces maintenance overhead.

## 7. Code Comments

*   **Mandate:** When updating the downstream code, ensure that comments explain *why* we are using `time_series_id` and `power` (because the new JSON data structure is unified and agnostic to the specific power unit, which is now tracked in the metadata). Connect the dots between the `TimeSeriesMetadata` contract and the `PowerTimeSeries` contract.

## Review Responses & Rejections

* **FLAW-001 (Scientist):** ACCEPTED. Added explicit validation logic in Section 3 to ensure the time span between `startTime` and `endTime` is exactly one half hour, and that `endTime` is at the top or bottom of the hour.
* **FLAW-001 (Tester):** ACCEPTED. Added a dedicated "Update Tests" section (Section 6) to audit all test suites, implement property-based tests using `hypothesis` for the new contracts, and add unit tests for data transformation logic.
* **FLAW-001 (Simplicity Audit):** ACCEPTED. Added `packages/dashboard` to `target_modules` in the YAML frontmatter.
* **Suggestion 1 (Simplicity Audit):** ACCEPTED. Renamed `Metadata` to `TimeSeriesMetadata` in Section 3.
* **Suggestion 2 (Simplicity Audit):** ACCEPTED. Added a verification step in Section 4 to ensure no orphaned imports or unused helper functions remain after removing downsampling code.
