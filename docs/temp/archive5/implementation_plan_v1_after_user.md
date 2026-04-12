---
status: "draft"
version: "v1"
after_reviewer: "user"
task_type: "data-ingestion"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["src/nged_substation_forecast/defs/nged_assets.py", "packages/contracts/src/contracts/data_schemas.py"]
---

# Implementation Plan: Ingesting 33 NGED JSON Files

## Overview
This plan outlines the steps to ingest the 33 JSON files containing NGED substation time series data located in `data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/`. The data will be validated against strict Patito data contracts, written to a Delta table (for time series) and a Parquet file (for metadata), and made ready for visualization in the Marimo app.

**Data Resolution Note:** The new NGED data is already provided at a 30-minute resolution. No resampling from 5-minute data to 30-minute data is required or should be performed. The pipeline will ingest the 30-minute data as-is.

## 1. Patito Data Contract
During the exploration phase, we identified that the JSON files contain metadata fields at the root level and a `data` array containing the time series. 

We will leverage the existing, highly rigorous Patito contracts defined in `packages/contracts/src/contracts/data_schemas.py`:
*   **`TimeSeriesMetadata`**: Captures the root-level fields (`time_series_id`, `time_series_name`, `time_series_type`, `units`, `licence_area`, `substation_number`, `substation_type`, `latitude`, `longitude`, `information`, `area_wkt`, `area_center_lat`, `area_center_lon`).
*   **`PowerTimeSeries`**: Captures the time series data (`time_series_id`, `period_end_time`, `power`).

*Architectural Note:* The existing contracts perfectly match the JSON structure. `load_nged_json` in `packages/nged_data/src/nged_data/io.py` already maps the CamelCase JSON keys to our snake_case contract fields and handles the validation. We do not need to write new contracts, but we must ensure the new asset strictly enforces them.

## 2. Creating a Dagster Asset
We will create a new Dagster asset in `src/nged_substation_forecast/defs/nged_assets.py` to ingest these specific SharePoint files.

*   **Asset Name:** `nged_sharepoint_json_asset`
*   **Group Name:** `NGED_JSON`
*   **Implementation:**
    *   Define the source directory: `Path("data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/")`.
    *   Iterate through all `*.json` files in this directory.
    *   Call `load_nged_json(json_file)` for each file to extract and validate the metadata and time series DataFrames.

## 3. Validating the Data
Validation is inherently handled by the `load_nged_json` function, which calls `TimeSeriesMetadata.validate()` and `PowerTimeSeries.validate()`. 
*   *Why:* This ensures that any malformed data (e.g., missing `time_series_id`, invalid `period_end_time` minutes) fails loudly during ingestion rather than silently corrupting the downstream Delta tables.

## 4. Writing Output to Delta and Parquet
For each processed JSON file, the asset will:
1.  **Metadata:** Call `upsert_metadata(metadata_df, settings.nged_data_path / "metadata" / "json_metadata")`. This writes/updates the Parquet file.
2.  **Time Series:** 
    *   Call `clean_power_data` to apply variance thresholds and remove anomalies.
    *   Call `append_to_delta(cleaned_df, settings.nged_data_path / "delta" / "json_data")`. This safely appends the data to the Delta table, partitioned by `time_series_id`, while avoiding duplicates.

## 5. Marimo App Integration
By writing the data to Delta (`json_data`) and Parquet (`json_metadata`), the data is immediately ready for the Marimo app. 
*   *Why:* Polars has native, zero-copy support for reading Delta tables (`pl.read_delta()`) and Parquet files (`pl.read_parquet()`). The Marimo app can simply point to `settings.nged_data_path / "delta" / "json_data"` to load the time series for visualization without any additional transformation steps.

## Required Code Changes

**`src/nged_substation_forecast/defs/nged_assets.py`**
Add the following asset:

```python
@asset(group_name="NGED_JSON")
def nged_sharepoint_json_asset(context: AssetExecutionContext, settings: ResourceParam[Settings]):
    """Ingest the 33 NGED JSON files provided via SharePoint."""
    # Hardcoded path for the specific SharePoint drop
    json_dir = Path("data/NGED/from_sharepoint/OneDrive_1_4-8-2026/1451606400000_1774512000000/")
    
    if not json_dir.exists():
        context.log.warning(f"Directory {json_dir} does not exist. Skipping ingestion.")
        return

    for json_file in json_dir.glob("*.json"):
        context.log.info(f"Processing {json_file.name}")
        metadata_df, time_series_df = load_nged_json(json_file)

        # Upsert metadata to Parquet
        upsert_metadata(metadata_df, settings.nged_data_path / "metadata" / "json_metadata")

        # Clean power data
        time_series_id = int(metadata_df.get_column("time_series_id").item())
        cleaned_df = clean_power_data(
            time_series_df,
            time_series_id=time_series_id,
            variance_thresholds=settings.data_quality.variance_thresholds,
        )

        # Append to Delta table
        append_to_delta(cleaned_df, settings.nged_data_path / "delta" / "json_data")

    context.log.info("Finished processing SharePoint JSON data.")
```
*(Note: Ensure `from pathlib import Path` is imported if not already present).*

## Review Responses & Rejections

*   **FLAW-USER (User):** ACCEPTED. The user noted that the new NGED data is already 30-minutely. We have explicitly added a note to the Overview section to ensure no resampling from 5-minute to 30-minute data is attempted, and the plan reflects the correct 30-minute resolution.
