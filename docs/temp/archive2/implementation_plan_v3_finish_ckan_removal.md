---
status: "draft"
version: "v3"
after_reviewer: "architect"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast", "packages/nged_json_data", "tests"]
---

# Implementation Plan: Finish CKAN Removal Refactor

This plan addresses the remaining issues from the CKAN removal refactor, bringing the codebase back to a consistent and runnable state.

## Step 1: Fix `PowerTimeSeries` Mapping and Tests
The `PowerTimeSeries` schema requires `start_time`, `end_time`, `time_series_id`, and `value`. The raw JSON data provides `timestamp`, `MW`, and `MVA`.

1. **Update `packages/nged_json_data/tests/test_clean.py`**:
   - Add `start_time` to the dummy data (e.g., `end_time - timedelta(minutes=30)`).
   - Ensure the test passes validation against `PowerTimeSeries`.

2. **Update `src/nged_substation_forecast/defs/nged_assets.py`**:
   - In `nged_json_archive_asset` and `nged_json_live_asset`, transform `time_series_df` before passing it to `clean_power_data`.
   - Map `timestamp` to `end_time` (parse string to UTC datetime).
   - Calculate `start_time` as `end_time - 30m`.
   - Map `value` as `pl.coalesce([pl.col("MW"), pl.col("MVA")])`.
   - Add `time_series_id` from `metadata_df["substation_number"]` (cast to string).
   - Select only the required columns: `["time_series_id", "start_time", "end_time", "value"]`.

3. **Fix Test Imports**:
   - Remove `packages/nged_json_data/tests/__init__.py` to prevent pytest module resolution conflicts.

## Step 2: Create `time_series_metadata` Asset
The XGBoost model and plotting assets require `time_series_metadata`, which was previously provided by CKAN assets.

1. **Update `src/nged_substation_forecast/defs/nged_assets.py`**:
   - In `nged_json_archive_asset` and `nged_json_live_asset`, transform `metadata_df` to match `TimeSeriesMetadata`:
     - Map `substation_name_in_location_table` to `time_series_name`.
     - Map `preferred_power_col` to `units`.
     - Set `time_series_id` to `str(substation_number)`.
   - Use `upsert_metadata` from `nged_json_data.metadata` to save `metadata_df` to a Parquet file (e.g., `settings.nged_data_path / "metadata" / "time_series_metadata.parquet"`).
   - Create a new asset `@asset(group_name="reference_data") def time_series_metadata(context, settings) -> pt.DataFrame[TimeSeriesMetadata]:` that reads this Parquet file and validates it against `TimeSeriesMetadata`.

## Step 3: Update XGBoost Assets
The XGBoost training and evaluation assets need to use the new `time_series_metadata`.

1. **Update `src/nged_substation_forecast/defs/xgb_assets.py`**:
   - Add `time_series_metadata: pl.DataFrame` as an input to `train_xgboost` and `evaluate_xgboost`.
   - Pass `time_series_metadata` to `train_and_log_model` and `evaluate_and_save_model`.
   - Ensure `sub_ids` filtering uses `time_series_metadata` correctly.

2. **Update `src/nged_substation_forecast/defs/xgb_jobs.py`**:
   - Add `time_series_metadata` to the `xgboost_integration_job` asset selection.

## Step 4: Update Plotting Assets
The plotting asset needs to use `time_series_metadata` and `time_series_id`.

1. **Update `src/nged_substation_forecast/defs/plotting_assets.py`**:
   - Rename the input `substation_metadata` to `time_series_metadata`.
   - Update the join logic to use `time_series_id` instead of `substation_number`.
   - Update the column names to match `PowerForecast` (`power_fcst` instead of `MW_or_MVA`).

## Step 5: Fix Integration Tests
The tests need to be updated to reflect the new asset dependencies and schemas.

1. **Update `tests/test_plotting_robustness.py`**:
   - Rename `substation_metadata` to `time_series_metadata` in the test setup.
   - Ensure `predictions` uses `time_series_id` and `power_fcst`.
   - Ensure `actuals` uses `time_series_id`, `start_time`, `end_time`, and `value`.

2. **Update `tests/test_xgboost_dagster_mocked.py`**:
   - Update `sub_meta` to match `TimeSeriesMetadata` schema (add `time_series_id`, `time_series_name`, etc.).
   - Provide `time_series_metadata=sub_meta` to `train_xgboost`.

3. **Update `tests/test_xgboost_dagster_integration.py`**:
   - Ensure the integration test job resolves correctly with the new `time_series_metadata` asset.

## Review Responses & Rejections
None yet.
