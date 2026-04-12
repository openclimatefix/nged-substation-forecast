# Implementation Plan: Enforce Uniqueness Across All Data Assets

## Problem
The data pipeline currently lacks a unified guarantee that raw and cleaned data assets are free of duplicate rows. This leads to data quality issues, inefficient processing, and potential OOM errors.

## Proposed Solution
We will implement a "deduplication at ingestion" strategy, ensuring that every asset that writes to a Delta table or Parquet file performs a deduplication step first.

### 1. `raw_power_time_series` (Ingestion)
*   **File**: `src/nged_substation_forecast/defs/nged_assets.py`
*   **Action**: Add `.unique(subset=["time_series_id", "period_end_time"])` to the DataFrame before calling `append_to_delta` in `nged_json_archive_asset`, `nged_json_live_asset`, and `nged_sharepoint_json_asset`.

### 2. NWP Data (Ingestion)
*   **File**: `src/nged_substation_forecast/defs/weather_assets.py`
*   **Action**: Add `.unique(subset=["time", "latitude", "longitude", "ensemble_member"])` to the DataFrame before saving to Parquet in the `ecmwf_ens_forecast` asset.

### 3. Cleaned Data Assets
*   **File**: `src/nged_substation_forecast/defs/data_cleaning_assets.py`
*   **Action**: Ensure `cleaned_power_time_series` also includes `.unique(subset=["time_series_id", "period_end_time"])` before writing to Delta, similar to the fix already applied to `cleaned_actuals`.

## Benefits
*   **Data Integrity**: Guarantees that all data assets are free of duplicates.
*   **Pipeline Robustness**: Prevents downstream issues caused by redundant data.
*   **Performance**: Reduces the volume of data processed and stored, improving pipeline speed and memory efficiency.
