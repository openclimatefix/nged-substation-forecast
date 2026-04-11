---
status: "reviewed"
version: "v1.1"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["src/nged_substation_forecast", "packages/xgboost_forecaster", "tests"]
---
# Implementation Plan: Refactor Data Pipeline and Remove `live_primary_flows`

## 1. Goal
Refactor the data pipeline to separate raw ingestion from cleaning, and remove all references to the deprecated `live_primary_flows` table. The raw ingestion assets will now save uncleaned data to a `raw_power_time_series` Delta table, and a new `cleaned_power_time_series` asset will handle the cleaning logic.

## 2. Pipeline Changes

### 2.1. Raw Ingestion (`src/nged_substation_forecast/defs/nged_assets.py`)
*   **Action:** Update `nged_json_archive_asset`, `nged_json_live_asset`, and `nged_sharepoint_json_asset`.
*   **Changes:**
    *   Remove the call to `clean_power_data`.
    *   Validate the raw `time_series_df` against the `PowerTimeSeries` contract before appending. Ensure this validation preserves "insane" values as intended for raw data.
    *   Change the `append_to_delta` path to `settings.nged_data_path / "delta" / "raw_power_time_series"`.
    *   Update docstrings to explicitly state that these assets ingest raw data *without* cleaning.
*   **Mandatory Comments:** Add comments explaining *why* we are not cleaning the data here (to separate raw ingestion from business logic/cleaning, allowing us to re-run cleaning without re-ingesting).

### 2.2. Partitioning Strategy (`src/nged_substation_forecast/utils.py`)
*   **Action:** Update `get_partition_window` to support 6-hourly partitions and act as a normalizer.
*   **Changes:**
    *   Modify the function to try parsing the `partition_key` using the 6-hourly format (`%Y-%m-%d-%H:%M`) first. If that fails, fall back to the daily format (`%Y-%m-%d`).
    *   For 6-hourly partitions, set `partition_end = partition_start + timedelta(hours=6)`.
    *   Ensure the function raises a clear, specific error for malformed keys that match neither format.
*   **Mandatory Comments:** Explain *why* we support both formats (to handle both the new 6-hourly live ingestion and legacy daily partitions). Explicitly document this fallback as **temporary technical debt** with a deprecation plan to remove daily partition support once backfilling is complete.

### 2.3. Cleaning Asset (`src/nged_substation_forecast/defs/data_cleaning_assets.py`)
*   **Action:** Rename `cleaned_actuals` to `cleaned_power_time_series`.
*   **Changes:**
    *   Change the `partitions_def` to use the same 6-hourly `TimeWindowPartitionsDefinition` as `nged_json_live_asset` (cron: `0 */6 * * *`, start: `2026-01-26-00:00`).
    *   Change `deps=["live_primary_flows"]` to `deps=["nged_json_live_asset"]`.
    *   Update `_get_delta_path` calls: read from `"raw_power_time_series"` and write to `"cleaned_power_time_series"`.
    *   Rename `get_cleaned_actuals_lazy` to `get_cleaned_power_time_series_lazy` and update the Delta path it reads from.
    *   **Idempotency:** Ensure the write operation to `cleaned_power_time_series` is perfectly idempotent (e.g., using partition overwrite/replace logic) so re-running yields the exact same result.
    *   **Causality:** Ensure the 1-day lookback is strictly causal. It must only pull historical context *up to* the partition boundary and never use data from the future relative to the partition being processed.
*   **Mandatory Comments:** Explain *why* we use a 6-hourly partition (to match live ingestion) and *why* we use a 1-day lookback (to ensure rolling window calculations have sufficient history at partition boundaries, strictly enforcing causality).

### 2.4. Update Downstream Assets (`src/nged_substation_forecast/defs/xgb_assets.py` & `src/nged_substation_forecast/defs/xgb_jobs.py`)
*   **Action:** Update references to the renamed cleaning asset.
*   **Changes:**
    *   In `xgb_assets.py`, change `deps=["cleaned_actuals"]` to `deps=["cleaned_power_time_series"]`.
    *   Update the import and usage of `get_cleaned_actuals_lazy` to `get_cleaned_power_time_series_lazy`.
    *   In `xgb_jobs.py`, update `xgboost_integration_job` selection to use `"cleaned_power_time_series"` instead of `"cleaned_actuals"`. Ensure no hardcoded string literals of the old names remain.

## 3. Cleanup `live_primary_flows`

*   **Action:** Remove all references to `live_primary_flows` across the codebase.
*   **Changes:**
    *   `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`: Change `base_power_path` default to `_SETTINGS.nged_data_path / "delta" / "cleaned_power_time_series"`.
    *   `src/nged_substation_forecast/ingestion/helpers.py`: Change `get_delta_path` to return `raw_power_time_series`.
    *   `tests/test_xgboost_dagster_mocked.py` & `tests/test_data_cleaning_robustness.py`: Update `live_flows_path` to use `raw_power_time_series`.

## 4. Tests

*   **Action:** Update integration tests and add new unit tests.
*   **Changes:**
    *   **Integration (`tests/test_xgboost_dagster_integration.py`):** Change `actuals_path` to `raw_power_time_series` and `cleaned_path` to `cleaned_power_time_series`. Remove `live_primary_flows` from `run_config`. Assert the existence of the new `cleaned_power_time_series` Delta table.
    *   **Unit Tests (`tests/test_utils.py` or similar):** Add specific test cases for `get_partition_window` covering both `%Y-%m-%d-%H:%M` and `%Y-%m-%d` formats, boundary cases, and malformed keys.
    *   **Regression/E2E:** Ensure existing E2E tests run successfully on synthetic datasets representing both old daily data and new 6-hourly data.

## 5. Conflicts
No major unresolvable conflicts were found. The primary point of discussion was the complexity of supporting two partition formats in `get_partition_window`. This was resolved by keeping the logic within `get_partition_window` (acting as a normalizer to a standard datetime tuple) but explicitly documenting it as temporary technical debt with a deprecation plan, and adding rigorous unit tests.

## Review Responses & Rejections

*   **FLAW-001 [Scientist - Lookahead Bias]:** ACCEPTED. Added strict causality requirements to the `cleaned_power_time_series` asset to ensure the 1-day lookback only uses historical data up to the partition boundary.
*   **FLAW-002 [Scientist - Schema Enforcement]:** ACCEPTED. Clarified that validation against `PowerTimeSeries` contract must preserve "insane" values for raw data.
*   **FLAW-003 [Scientist - Idempotency]:** ACCEPTED. Added explicit requirement for the cleaning pipeline to be perfectly idempotent using partition overwrite logic.
*   **FLAW-004 [Tester - Test Coverage]:** ACCEPTED. Added requirement for unit tests covering both partition formats in `get_partition_window`.
*   **FLAW-005 [Tester - Edge Cases]:** ACCEPTED. Added requirement to handle malformed keys with specific errors in `get_partition_window`.
*   **FLAW-006 [Tester - Regression Testing]:** ACCEPTED. Added requirement to ensure E2E tests cover both old and new data formats.
*   **FLAW-007 [Reviewer - Maintainability]:** ACCEPTED. Added requirement to document the fallback logic as temporary technical debt with a deprecation plan.
*   **FLAW-008 [Reviewer - Code Quality]:** ACCEPTED. Added explicit check for hardcoded string literals in `xgb_jobs.py` and downstream assets.
*   **FLAW-009 [Reviewer - Architectural Standards]:** ACCEPTED. Confirmed Dagster asset graph dependencies are correctly updated.
*   **FLAW-010 [Simplicity - Approach Simplicity]:** ACCEPTED. Acknowledged the complexity of dual partitioning schemes.
*   **FLAW-011 [Simplicity - Design Elegance]:** REJECTED (Modified). The suggestion to create a separate "Normalizer" was modified. Since Dagster passes partition keys as strings, `get_partition_window` *is* the appropriate place to normalize these strings into standard `(start, end)` datetime tuples. We will keep the logic there but ensure it acts as a strict normalizer and is well-tested.
*   **FLAW-012 [Simplicity - Reduce Complexity]:** ACCEPTED. Documenting the fallback strategy as temporary technical debt.
