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
*   **Action:** Rename `SubstationMetadata` to `Metadata`.
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
*   **Rationale:** The new JSON data provides a single `value` column and uses `time_series_id` instead of `substation_number`. The contracts must reflect this simpler structure.

## 4. Remove Downsampling Code

*   **Action:** In `packages/ml_core/src/ml_core/data.py`, delete the functions `calculate_target_map` and `downsample_power_flows`.
*   **Action:** In `packages/ml_core/src/ml_core/utils.py`, remove the calls to `downsample_power_flows` in `train_and_log_model` and `evaluate_and_save_model`. The input data (`power_time_series`) will now be expected to be half-hourly and have a `value` column (which should be renamed to `power` or `power_fcst` as needed by the model). Remove any logic related to `target_map`.
*   **Rationale:** The new JSON data is already half-hourly, so downsampling is no longer required. The `target_map` concept is obsolete.

## 5. Update Downstream Code

*   **5a. Update `packages/xgboost_forecaster`:** Use `XGBoostInputFeatures` instead of `SubstationFeatures`, and `PowerTimeSeries` instead of `SubstationPowerFlows`. Update feature engineering to use `power` instead of `MW_or_MVA` and `time_series_id` instead of `substation_number`.
*   **5b. Update `packages/ml_core`:** Update models to reflect the new contracts.
*   **5c. Update `packages/dashboard`:** Update `packages/dashboard/main.py` to use the new `Metadata` and `PowerTimeSeries` contracts, and remove references to `preferred_power_col` and `POWER_MW`.
*   **5d. Update `packages/nged_json_data`:** Update to use `PowerTimeSeries` instead of `NgedJsonPowerFlows`.
*   **5e. Global Join Update:** Ensure all code that previously joined on `substation_number` now joins on `time_series_id`.
*   **5f. Global Search-and-Replace Verification:** Perform a global search (e.g., using `grep` or IDE refactoring tools) to ensure all instances of old contract names (`SubstationPowerFlows`, `SubstationMetadata`, etc.) are updated across the entire codebase, including configuration files.
*   **Rationale:** All components must be aligned with the new data contracts to ensure the pipeline functions correctly. Breaking this down into sub-tasks ensures a more manageable implementation and review process.

## 6. Code Comments

*   **Mandate:** When updating the downstream code, ensure that comments explain *why* we are using `time_series_id` and `power` (because the new JSON data structure is unified and agnostic to the specific power unit, which is now tracked in the metadata). Connect the dots between the `Metadata` contract and the `PowerTimeSeries` contract.
*   **Mandate:** Do not reference FLAW-XXX IDs in code comments.

## 7. Verification and Testing

*   **Action:** Update all unit and integration tests for `packages/contracts`, `packages/xgboost_forecaster`, and `packages/dashboard` to align with the new data structures and contracts.
*   **Action:** Implement new validation tests for the JSON-based ingestion pipeline, specifically verifying the half-hourly frequency and the new `time_series_id` structure.
*   **Action:** Run existing tests and fix any breakages.
*   **Action:** Add new tests for the updated `PowerTimeSeries` and `Metadata` contracts.
*   **Action:** Perform a manual verification run of the pipeline with the new JSON data source to ensure end-to-end functionality.

## 8. Documentation

*   **Action:** Update high-level documentation (e.g., `README.md` files) to reflect the removal of CKAN and the new data structure. Ensure any references to old contract names in documentation are updated.

## Conflicts

*   **Assessment:** No major unresolvable conflicts or fundamental shifts were found among the reviews. The Scientist confirmed the ML rigor of the plan. The Tester, Reviewer, and Simplicity Audit all focused on adding necessary verification, testing, and documentation steps, which complement each other.

## Review Responses & Rejections

* **FLAW-001 (Tester):** ACCEPTED. Added explicit steps in Section 7 to update tests for `packages/contracts`, `packages/xgboost_forecaster`, and `packages/dashboard`.
* **FLAW-002 (Tester):** ACCEPTED. Added a step in Section 7 to implement new validation tests for the JSON-based ingestion pipeline.
* **FLAW-001 (Reviewer):** ACCEPTED. Added Step 5f for global search-and-replace verification and Section 8 for updating documentation.
* **FLAW-001 (Simplicity):** ACCEPTED. Added a dedicated Section 7 for Verification and Testing, including manual verification.
* **Suggestion 1 (Simplicity):** ACCEPTED. Broke down Section 5 into granular sub-tasks (5a-5f).
* **Suggestion 2 (Simplicity):** ACCEPTED. Added Section 8 for updating high-level documentation.
