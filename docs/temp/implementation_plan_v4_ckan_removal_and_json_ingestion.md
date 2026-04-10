---
status: "draft"
version: "v4"
after_reviewer: "architect"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["packages/contracts", "packages/nged_json_data", "src/nged_substation_forecast", "tests"]
---

# Implementation Plan: JSON Ingestion and Contract Updates

This plan addresses the ingestion of the new NGED JSON data, fixing incorrect assumptions about the data format, and updating the data contracts to use `time_series_id` and period-ending timestamps.

## 1. Update Data Contracts (`packages/contracts/src/contracts/data_schemas.py`)

The data contracts need to be updated to reflect the new JSON data format and the shift from `substation_number` to `time_series_id`.

*   **Update `TimeSeriesMetadata`**:
    *   Ensure all fields are `snake_case` and match the JSON metadata fields: `time_series_id` (primary identifier, `unique=True`), `time_series_name`, `time_series_type`, `units`, `licence_area`, `substation_number`, `substation_type`, `latitude`, `longitude`, `information`, `area_wkt`, `area_center_lat`, `area_center_lon`.
*   **Update `PowerTimeSeries`**:
    *   Remove the `start_time` field. We will only store the end time of each recording (period-ending).
    *   Update the `validate` method to remove the 30-minute span check (since `start_time` is gone). Keep the check that `end_time` is at :00 or :30.

## 2. Update JSON IO (`packages/nged_json_data/src/nged_json_data/io.py`)

The `load_nged_json` function needs to correctly parse the JSON and format it according to the new contracts.

*   **Format Metadata**:
    *   Convert the extracted metadata keys from CamelCase to `snake_case` (e.g., `TimeSeriesID` -> `time_series_id`).
    *   Validate the metadata DataFrame against the new `TimeSeriesMetadata` contract.
*   **Format Time Series**:
    *   Rename `endTime` to `end_time` and parse it as a UTC datetime.
    *   Ensure that every reading in the JSON `data` list covers a time period of exactly half an
        hour. i.e. ensure that `end_time - start_time` is half an hour.
    *   Add the `time_series_id` from the metadata to the `time_series_df` so it can be associated with the correct time series.
    *   Select only the required columns: `["time_series_id", "end_time", "value"]`.
    *   Validate the time series against the `PowerTimeSeries` data contract.
*   **Return pt.DataFrames**:
    * `load_nged_json` should return `tuple[pt.DataFrame[TimeSeriesMetadata], pt.DataFrame[PowerTimeSeries`]]`

## 3. Update Data Cleaning (`packages/nged_json_data/src/nged_json_data/clean.py`)

The cleaning logic needs to be updated to use `time_series_id` and to properly remove bad pre-amble data without dropping genuine zeros later in the time series.

*   **Use `time_series_id`**:
    *   Change the `substation_number` argument in `clean_power_data` to `time_series_id` (type `int32`).
*   **Remove Bad Pre-amble Data**:
    *   Instead of dropping all rows with low variance (which would remove genuine zeros later on), find the *first* index where the rolling variance exceeds the threshold.
    *   Slice the DataFrame from that index onwards. This fully removes the initial bad data (zeros/nulls/tiny values) while preserving the integrity of the time series once it starts producing valid data.

## 4. Update Dagster Assets (`src/nged_substation_forecast/defs/nged_assets.py`)

The assets need to be updated to use the new functions and contracts.

*   **Update `nged_json_archive_asset` and `nged_json_live_asset`**:
    *   Call `load_nged_json`.
    *   Call `upsert_metadata` directly inside the asset with the extracted metadata. **Do NOT create a separate `time_series_metadata` Dagster asset.**
    *   Call `clean_power_data` using `time_series_id`.
    *   Append the cleaned data to the Delta table.

## 5. Update Downstream Code

Any code that relied on `substation_number` or `start_time` needs to be updated.

*   **Update XGBoost Features and Models**:
    *   Update `packages/xgboost_forecaster/src/xgboost_forecaster/features.py` and `model.py` to use `time_series_id` instead of `substation_number`.
    *   Update any logic that used `start_time` to only use `end_time`.
*   **Update Plotting Assets**:
    *   Update `src/nged_substation_forecast/defs/plotting_assets.py` to join on `time_series_id` and use `end_time`.
*   **Update Tests**:
    *   Update all relevant tests in `tests/` and `packages/*/tests/` to reflect the new schemas and logic.

## Review Responses & Rejections
None yet.
