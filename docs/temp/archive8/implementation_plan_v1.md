---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
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
    *   Validate the raw `time_series_df` against the `PowerTimeSeries` contract before appending.
    *   Change the `append_to_delta` path to `settings.nged_data_path / "delta" / "raw_power_time_series"`.
    *   Update docstrings to explicitly state that these assets ingest raw data *without* cleaning (preserving "insane" values).
*   **Mandatory Comments:** Add comments explaining *why* we are not cleaning the data here (to separate raw ingestion from business logic/cleaning, allowing us to re-run cleaning without re-ingesting).

### 2.2. Partitioning Strategy (`src/nged_substation_forecast/utils.py`)
*   **Action:** Update `get_partition_window` to support 6-hourly partitions.
*   **Changes:**
    *   Modify the function to try parsing the `partition_key` using the 6-hourly format (`%Y-%m-%d-%H:%M`) first. If that fails, fall back to the daily format (`%Y-%m-%d`).
    *   For 6-hourly partitions, set `partition_end = partition_start + timedelta(hours=6)`.
*   **Mandatory Comments:** Explain *why* we support both formats (to handle both the new 6-hourly live ingestion and legacy daily partitions).

### 2.3. Cleaning Asset (`src/nged_substation_forecast/defs/data_cleaning_assets.py`)
*   **Action:** Rename `cleaned_actuals` to `cleaned_power_time_series`.
*   **Changes:**
    *   Change the `partitions_def` to use the same 6-hourly `TimeWindowPartitionsDefinition` as `nged_json_live_asset` (cron: `0 */6 * * *`, start: `2026-01-26-00:00`). This handles both small frequent updates and allows Dagster to backfill large archival loads.
    *   Change `deps=["live_primary_flows"]` to `deps=["nged_json_live_asset"]`.
    *   Update `_get_delta_path` calls: read from `"raw_power_time_series"` and write to `"cleaned_power_time_series"`.
    *   Rename `get_cleaned_actuals_lazy` to `get_cleaned_power_time_series_lazy` and update the Delta path it reads from.
*   **Mandatory Comments:** Explain *why* we use a 6-hourly partition (to match the live ingestion frequency) and *why* we use a 1-day lookback (to ensure rolling window calculations have sufficient history at partition boundaries).

### 2.4. Update Downstream Assets (`src/nged_substation_forecast/defs/xgb_assets.py` & `src/nged_substation_forecast/defs/xgb_jobs.py`)
*   **Action:** Update references to the renamed cleaning asset.
*   **Changes:**
    *   In `xgb_assets.py`, change `deps=["cleaned_actuals"]` to `deps=["cleaned_power_time_series"]`.
    *   Update the import and usage of `get_cleaned_actuals_lazy` to `get_cleaned_power_time_series_lazy`.
    *   In `xgb_jobs.py`, update `xgboost_integration_job` selection to use `"cleaned_power_time_series"` instead of `"cleaned_actuals"`.

## 3. Cleanup `live_primary_flows`

*   **Action:** Remove all references to `live_primary_flows` across the codebase.
*   **Changes:**
    *   `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`: Change `base_power_path` default to `_SETTINGS.nged_data_path / "delta" / "cleaned_power_time_series"`.
    *   `src/nged_substation_forecast/ingestion/helpers.py`: Change `get_delta_path` to return `raw_power_time_series`.
    *   `tests/test_xgboost_dagster_mocked.py` & `tests/test_data_cleaning_robustness.py`: Update `live_flows_path` to use `raw_power_time_series`.

## 4. Tests (`tests/test_xgboost_dagster_integration.py`)

*   **Action:** Update the integration test to use the new assets and paths.
*   **Changes:**
    *   Change `actuals_path` to `settings.nged_data_path / "delta" / "raw_power_time_series"`.
    *   Change `cleaned_path` to `settings.nged_data_path / "delta" / "cleaned_power_time_series"`.
    *   Remove the `live_primary_flows` entry from the `run_config` dictionary, as it is no longer a valid op/asset.
    *   Ensure the test asserts the existence of the new `cleaned_power_time_series` Delta table.

## Review Responses & Rejections
(None yet - initial draft)
