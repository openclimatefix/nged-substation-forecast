---
plan_version: "v3.0"
based_on: "v2.7"
status: "READY_FOR_IMPLEMENTATION"
review_responses:
  total_flaws: 5
  accepted: 5
  rejected: 0
---

# Implementation Plan: NWP Robustness & Data Quality Refactoring

## Overview
This plan addresses a duplication bug in the ECMWF ingestion pipeline, introduces robust testing for NWP data using a local Zarr sample, and refactors the data quality checks for substation flows. We are centralizing data quality thresholds in the `Settings` object, moving away from a separate `healthy_substations` asset towards a more robust `cleaned_actuals` asset in a dedicated file, backed by inline Dagster asset checks on the raw data. This ensures downstream models (like XGBoost) only train on physically plausible data without breaking temporal continuity. It also introduces strict Patito data contracts, proper partition mapping for rolling windows, and modularizes the cleaning logic into the `nged_data` package.

*Update v3.0:* This version incorporates fixes from the Scientist Code Review Round 1, specifically addressing critical data duplication in `cleaned_actuals`, correcting stuck sensor logic that deleted valid data, enforcing the implementation of missing NWP robustness tests, relaxing schema validation for fully null partitions, and cleaning up duplicated code in the XGBoost training asset.

## Step-by-Step Implementation

### 1. Configurable Thresholds in Settings
*   **Target File:** `packages/contracts/src/contracts/settings.py`
    *   **Action:** Add data quality thresholds to the `Settings` class:
        *   `stuck_std_threshold: float = 0.01`
        *   `max_mw_threshold: float = 100.0`
        *   `min_mw_threshold: float = -20.0`
    *   **Commenting Mandate:** Add comments explaining *why* these specific thresholds were chosen as defaults (e.g., "MW > 100.0 is physically unrealistic for these primary substations", "STD < 0.01 indicates a stuck sensor"). Explain that centralizing these in `Settings` allows them to be configurable per environment while preventing logic drift between asset checks and data cleaning steps.

### 2. ECMWF Bug Fix & Zarr Testing
*   **Target File:** `packages/dynamical_data/src/dynamical_data/scaling.py`
    *   **Action:** Fix the duplication bug in the `download_and_scale_ecmwf` function.
    *   **Commenting Mandate:** You must add comments explaining *why* the duplication was occurring and *why* the chosen fix resolves it. Connect this logic to the broader NWP ingestion strategy.

#### Required Implementation for FLAW-003 (Missing NWP Robustness Tests)
*   **Target File:** `packages/dynamical_data/src/dynamical_data/scripts/create_production_like_test_zarr.py` (New File)
    *   **Action:** Create a comprehensive test Zarr generator script that produces NWP data with the exact same coordinate structure as the production code expects (latitude, longitude, and time coordinates).
    *   **Logic:** Generate synthetic NWP data with dimensions `(latitude, longitude, init_time, lead_time, ensemble_member)` and 13 data variables matching production specifications. Include a CLI for generating both valid and broken test cases (missing coords, wrong dim order, malformed data, etc.).
    *   **Commenting Mandate:** Explain *why* we are creating a production-like Zarr sample (to ensure fast, reliable, and deterministic CI runs that can mathematically test temporal duplication bugs and validation logic without hitting the real API).
    *   **Consolidation:** Remove redundant scripts `create_ecmwf_test_zarr.py` and `create_production_like_ecmwf_zarr.py` to prevent maintenance rot.
*   **Target File:** `tests/conftest.py`
    *   **Action:** Refactor test data generation into `pytest` fixtures (`production_like_zarr_path`, `broken_zarr_factory`) using `tmp_path` for on-the-fly generation.
    *   **Logic:** This avoids committing binary artifacts to git while still testing the full serialization stack (reading from a real filesystem).
*   **Target File:** `tests/test_nwp_ingestion_robustness.py` (New File)
    *   **Action:** Create a new integration test to verify that `download_and_scale_ecmwf` runs without duplication errors when merging multiple forecast steps.
    *   **Action:** Expand the test suite to include edge cases for Zarr ingestion failures. Use the `broken_zarr_factory` fixture to test "broken" Zarr samples (e.g., missing coordinates, missing variables, wrong dtypes).
    *   **Logic:** Update the test to use the `pytest` fixtures instead of static file paths. Ensure the ingestion pipeline fails loudly with informative, specific error messages.
    *   **Commenting Mandate:** Explicitly comment *why* we use on-the-fly Zarr generation for testing (to test the actual xarray/zarr logic and temporal merging without external network dependencies or binary bloat).

### 3. Asset & Check Refactoring
*   **Target File:** `src/nged_substation_forecast/defs/metrics_assets.py`
    *   **Action:** **Remove** the `healthy_substations` asset and the `check_all_zeros` asset check.
*   **Target File:** `src/nged_substation_forecast/defs/nged_assets.py`
    *   **Action:** **Add** a new `AssetCheckSpec` named `substation_data_quality` to the `live_primary_flows` asset.
    *   **Logic:**
        *   Configure the check to run on each daily partition.
        *   Import and use the configurable thresholds from the `Settings` object (`stuck_std_threshold`, `max_mw_threshold`, `min_mw_threshold`).
        *   **Crucially**, include a guard clause to handle empty partitions (e.g., if ingestion failed completely for a day). If the Polars DataFrame is empty, the check should gracefully handle it by returning a `passed=True` result with a warning in the metadata indicating "No data found for partition". Do not attempt to compute statistics like `count` or `std` on an empty DataFrame.
        *   Report the count of affected substations and a sample of their IDs in the `AssetCheckResult` metadata.
    *   **Commenting Mandate:** Add comments explaining *why* we are moving these checks to the source asset (to catch data quality issues immediately upon ingestion) and *why* we guard against empty DataFrames (to prevent pipeline crashes on days with total data loss).

### 4. Modular Cleaning Logic
*   **Target File:** `packages/nged_data/src/nged_data/cleaning.py` (New File)
    *   **Action:** Move the core cleaning and imputation Polars logic into a dedicated function (e.g., `clean_substation_flows`).
    *   **Logic:**
        *   Accept a Polars DataFrame of raw flows and a `Settings` object.
        *   Identify "stuck" or "insane" values using the configurable thresholds. The rolling standard deviation must use a strictly **backward-looking** window.
        *   **Crucially**, all temporal operations (rolling windows) must be performed using `.over("substation_number")` or within a `group_by("substation_number")` context to prevent cross-substation data leakage.
        *   Replace "stuck" or "insane" values with `null` (NaN). **Do NOT remove rows and do NOT impute missing values.** Leaving them as `null` maintains the strict 30-minute temporal grid, which is critical for accurate lag/rolling feature generation.
    *   **Commenting Mandate:** Explain *why* this logic is modularized (to allow unit testing without Dagster overhead), *why* grouping by substation is critical for temporal operations, and *why* we leave bad values as `null` instead of imputing or dropping rows (to preserve the temporal grid for feature engineering).

#### Required Implementation for FLAW-002 (Incorrect "Stuck Sensor" Logic)
*   **Target Files:** `packages/nged_data/src/nged_data/cleaning.py` (around line 62) and `src/nged_substation_forecast/defs/nged_assets.py` (around line 348).
*   **Action:** In the rolling standard deviation calculation for stuck sensors, change the null-filling logic from `.fill_null(0)` to `.fill_null(float('inf'))` (or `.fill_null(stuck_std_threshold + 1.0)`).
*   **Logic:** Because `rolling_std(48)` naturally returns `null` for the first 47 rows of any group due to insufficient data, filling with `0` incorrectly flags the first 23.5 hours of valid data as "stuck" (since 0 < 0.01). Filling with infinity ensures we assume data is valid until we have enough history to mathematically prove it is stuck.

### 5. New `data_cleaning_assets.py` & Patito Contracts
*   **Target File:** `packages/contracts/src/contracts/schemas.py` (or `data_schemas.py`)
    *   **Action:** Ensure a Patito schema exists for the cleaned data (e.g., `CleanedSubstationFlows` or reusing `SubstationFlows`). **Update the schema to explicitly allow `null` values for `MW`, `MVA`, and `MW_or_MVA`**.

#### Required Implementation for FLAW-004 (SubstationFlows Validation Fails)
*   **Target File:** `packages/contracts/src/contracts/data_schemas.py` (around lines 51-55).
*   **Action:** Relax the validation rule that raises `MissingCorePowerVariablesError` if the entire DataFrame has only nulls for `MW` and `MVA`.
*   **Logic:** A fully null DataFrame is a valid (though rare) outcome of the cleaning process if an entire partition's data is flagged as "stuck" or "insane". The validation should allow fully null partitions for these columns without crashing the pipeline.

*   **Target File:** `src/nged_substation_forecast/defs/data_cleaning_assets.py` (New File)
    *   **Action:** Create a new file for data cleaning assets and define the `cleaned_actuals` asset here.
    *   **Logic:**
        *   **Remove** the old `combined_actuals` asset (and update all its dependents, like `metrics`, to use `cleaned_actuals` instead) to prevent redundancy and logic drift.
        *   Read from the `live_primary_flows` Delta table.
        *   **Crucially**, use a `PartitionMapping` (e.g., `TimeWindowPartitionMapping`) to include the previous day's data in its input, or scan a sufficient lookback from the Delta table, to ensure the rolling window calculations at the start of a partition boundary have sufficient history.
        *   Call the `clean_substation_flows` function from `nged_data.cleaning`.
        *   Perform the join with `substation_metadata` (previously done in `combined_actuals`) to ensure metadata is included.
        *   Validate the output DataFrame using the Patito schema (e.g., `CleanedSubstationFlows.validate(df)`).
        *   Save the result to a new Delta table named `cleaned_actuals`.
    *   **Commenting Mandate:** Comment *why* we use a `PartitionMapping` for lookback (to prevent NaN values at the start of daily partitions due to rolling windows), *why* we validate with Patito (to enforce strict data contracts), and *why* `combined_actuals` was removed in favor of this unified asset.

#### Required Implementation for FLAW-001 (Temporal Data Duplication)
*   **Target File:** `src/nged_substation_forecast/defs/data_cleaning_assets.py` (around lines 114-117).
*   **Action:** When using `TimeWindowPartitionMapping` with a lookback, the resulting DataFrame contains both historical and current partition data. You must filter the `validated_df` to only include rows where the `timestamp` falls strictly within the *current* partition's time window *before* appending it to the Delta table.
*   **Logic:** Failure to filter back to the current partition results in the previous day's data being duplicated in the Delta table every time a new partition runs.

### 6. Downstream Updates
*   **Target File:** `src/nged_substation_forecast/defs/xgb_assets.py`
    *   **Action:** Update the `train_xgboost` and `evaluate_xgboost` assets to use `cleaned_actuals` as their primary data source.
    *   **Action:** Update the `_get_target_substations` helper function. Instead of scanning actuals data, it should use `substation_metadata` (filtering for those with `url` not null) as a more efficient fallback when `config.substation_ids` is not provided.
    *   **Action:** Update `XGBoostForecaster` (or the `train_xgboost` asset) to drop rows where the **target variable** is `null`. This must happen *after* feature engineering (so lags are computed correctly using the nulls) but *before* calling `fit()`.
    *   **Commenting Mandate:** Explain *why* the models now consume `cleaned_actuals` (to ensure they only train on physically plausible data, improving model stability). Document the more efficient `_get_target_substations` logic using metadata. Explain *why* null targets are dropped only after feature engineering.

#### Required Implementation for FLAW-005 (Duplicated Code Block)
*   **Target File:** `src/nged_substation_forecast/defs/xgb_assets.py` (around lines 135-149).
*   **Action:** Remove the duplicated code block in the `train_xgboost` asset that computes `healthy_substations` from `substation_power_flows_df`.
*   **Logic:** The block is duplicated exactly twice in a row, causing minor code bloat and confusion.

### 7. Final Verification
*   **Action:** Run `uv run dagster dev` to ensure the Dagster UI starts without issues and all asset definitions are valid.

## Review Responses & Rejections

*   **FLAW-001 [Scientist - Round 1]:** ACCEPTED. The `cleaned_actuals` asset must filter the `validated_df` back to the current partition's time window before appending to the Delta table to prevent data duplication caused by the lookback partition mapping.
*   **FLAW-002 [Scientist - Round 1]:** ACCEPTED. The rolling standard deviation logic for stuck sensors must use `.fill_null(float('inf'))` instead of `.fill_null(0)` to prevent the first 24 hours of valid data from being incorrectly flagged and deleted.
*   **FLAW-003 [Scientist - Round 1]:** ACCEPTED. The missing NWP robustness tests and the Zarr generation script must be implemented exactly as specified in the original plan to ensure deterministic CI testing for temporal duplication bugs.
*   **FLAW-004 [Scientist - Round 1]:** ACCEPTED. The `SubstationFlows` validation rule in `packages/contracts/src/contracts/data_schemas.py` must be relaxed to allow fully null partitions for `MW` and `MVA`, as this is a valid outcome of the cleaning process.
*   **FLAW-005 [Scientist - Round 1]:** ACCEPTED. The duplicated code block in `train_xgboost` (`src/nged_substation_forecast/defs/xgb_assets.py`) must be removed to improve code quality.

*(Legacy Review Responses from v2.7)*
*   **FLAW-001 [Scientist - Iteration 2]:** ACCEPTED. Removing rows or imputing values destroys the temporal grid or introduces artifacts. The plan now strictly mandates replacing bad values with `null` and leaving them as `null` during feature engineering. Rows where the target is `null` will be dropped only *after* feature engineering and right before `fit()`. The Patito schema will be updated to allow `null` for powerflow columns.
*   **FLAW-001 [Reviewer]:** ACCEPTED. The plan now explicitly requires `cleaned_actuals` to use Patito for schema definition and validation (e.g., `CleanedSubstationFlows`).
*   **FLAW-002 [Reviewer]:** ACCEPTED. The `cleaned_actuals` asset is now mandated to use a `PartitionMapping` (e.g., `TimeWindowPartitionMapping`) or scan a sufficient lookback to ensure rolling windows work correctly across partition boundaries.
*   **FLAW-003 [Reviewer]:** ACCEPTED. The plan explicitly mandates that all temporal operations (rolling windows, imputation) must be performed using `.over("substation_number")` or within a `group_by("substation_number")` context.
*   **FLAW-004 [Reviewer]:** ACCEPTED. `combined_actuals` is being removed entirely to prevent redundancy and logic drift. All dependents (like `metrics`) will be updated to use `cleaned_actuals`.
*   **FLAW-005 [Reviewer]:** ACCEPTED. The join with `substation_metadata` is now explicitly included in the `cleaned_actuals` logic.
*   **FLAW-006 [Reviewer]:** ACCEPTED. The `_get_target_substations` fallback is updated to efficiently use `substation_metadata` (filtering for `url` not null) instead of scanning the actuals data.
*   **FLAW-007 [Reviewer]:** ACCEPTED. The core cleaning and imputation Polars logic is moved into a dedicated function in `packages/nged_data/src/nged_data/cleaning.py` for better modularity and testability.
*   **FLAW-001 [Tester]:** ACCEPTED. The plan for `tests/test_nwp_ingestion_robustness.py` now explicitly requires testing edge cases like empty Zarr archives, missing variables, and missing coordinates to ensure the pipeline fails loudly and informatively.
*   **FLAW-002 [Tester]:** ACCEPTED. The `substation_data_quality` asset check in `src/nged_substation_forecast/defs/nged_assets.py` now includes a strict requirement for a guard clause to handle empty DataFrames gracefully, preventing crashes when computing statistics on empty partitions.
*   **FLAW-001 [Scientist - Iteration 1]:** ACCEPTED. Filtering rows breaks the temporal grid and corrupts lag features. The plan now specifies replacing bad values with `null` (NaN) to maintain the 30-minute temporal continuity.
*   **FLAW-002 [Scientist - Iteration 1]:** ACCEPTED. A centered rolling window causes temporal data leakage. The plan now explicitly mandates a strictly backward-looking window (e.g., `center=False` or `closed='right'`) for the 48-period rolling standard deviation check.
*   **FLAW-003 [Scientist - Iteration 1]:** ACCEPTED. A single forecast step cannot test for duplication during merging. The `scripts/create_ecmwf_test_zarr.py` script is updated to download at least two consecutive forecast steps (e.g., `step=0` and `step=1`).
*   **FLAW-004 [Scientist - Iteration 1]:** ACCEPTED. The default thresholds were physically unrealistic for primary substations. They have been updated in `Settings` to `max_mw_threshold: 100.0` and `min_mw_threshold: -20.0`.
*   **Prior User Feedback:**
    *   **FLAW-001 [User]:** ACCEPTED. The ECMWF duplication bug will be tested against a real, but small, local Zarr sample instead of a mocked API response. This provides a higher fidelity test of the `xarray` and `zarr` logic. The script to generate this is placed in `packages/dynamical_data/scripts/` rather than `exploration_scripts/`.
    *   **FLAW-002 [User]:** ACCEPTED. Data quality thresholds are now configurable via the `Settings` class in `packages/contracts/src/contracts/settings.py` rather than hardcoded constants. This allows environment-specific overrides while maintaining a single source of truth.
    *   **FLAW-003 [User]:** ACCEPTED. The `cleaned_actuals` asset is moved to a dedicated `data_cleaning_assets.py` file to better organize the Dagster pipeline and separate cleaning logic from metrics.
    *   **FLAW-004 [User]:** ACCEPTED. Added a final verification step to run `uv run dagster dev` to ensure the refactored assets and checks load correctly in the Dagster UI.
