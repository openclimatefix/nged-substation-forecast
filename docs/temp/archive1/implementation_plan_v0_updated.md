---
status: "draft"
version: "v0_updated"
after_reviewer: "architect"
task_type: "data-ingestion"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["packages/nged_ckan_data", "packages/nged_json_data", "src/nged_substation_forecast/defs", "packages/contracts"]
---

# Implementation Plan: Ingesting NGED's New S3 JSON Data

This plan outlines the steps to deprecate the existing CKAN ingestion pipeline and introduce a new pipeline for NGED's new JSON data format, as requested in `plan_for_ingesting_NGEDs_new_S3_data.md`.

## 1. Deprecating the Existing CKAN Code

We will keep the CKAN code as a fallback but clearly mark it as deprecated.

*   **Rename Package:** Rename the directory `packages/nged_data` to `packages/nged_ckan_data`.
*   **Update Package Metadata:** Update `pyproject.toml` inside `packages/nged_ckan_data` to reflect the new package name (`name = "nged_ckan_data"`).
*   **Update Downstream Imports:** Search and replace all imports of `nged_data` with `nged_ckan_data` across the codebase (specifically in `src/nged_substation_forecast/defs/nged_assets.py`, `data_cleaning_assets.py`, `metrics_assets.py`, `plotting_assets.py`, `xgb_assets.py`, etc.).
*   **Dagster UI Grouping:** In `src/nged_substation_forecast/defs/nged_assets.py`, add `group_name="NGED CKAN"` to all `@asset` decorators to visually separate them in the Dagster UI.

## 2. New Code to Import JSON Data

We will create a new package to handle the JSON data ingestion.

*   **Create New Package:** Create `packages/nged_json_data` with a standard Python package structure (`pyproject.toml`, `src/nged_json_data/`, `tests/`).
*   **Data Contracts (`packages/contracts/src/contracts/data_schemas.py`):**
    *   *Architect's Recommendation:* We should convert the JSON data to a Polars DataFrame first and verify the DataFrame using a Patito data contract. This is much faster and aligns perfectly with our existing Polars-heavy architecture.
    *   Create a new contract `NgedJsonPowerFlows` for the time series data (columns: `time_series_id`, `end_time`, `value`).
    *   Update `SubstationMetadata` to use a single contract for both CKAN and JSON data. Add a new field `time_series_name` as the canonical name (which will eventually replace `substation_name_in_location_table`). Include the new fields: `time_series_id` (Primary Key), `time_series_type`, `units`, `licence_area`, `substation_number`, `substation_type`, `latitude`, `longitude`, `information`, `area_wkt`, `area_center_lat`, `area_center_lon`. Make the old CKAN fields optional to maintain backwards compatibility.
*   **Loading & Processing Logic (`packages/nged_json_data/src/nged_json_data/io.py`):**
    *   Implement `load_nged_json(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]` using `pl.read_json()`. Extract the metadata into one DataFrame and `explode().struct.unnest()` the `data` array into a time series DataFrame.
*   **Data Cleaning (`packages/nged_json_data/src/nged_json_data/clean.py`):**
    *   Implement `clean_power_data(df: pl.DataFrame) -> pl.DataFrame`. This function will:
        1.  Sort by `end_time`.
        2.  Drop any rows where `value` is null.
        3.  Identify "bad pre-amble data" by throwing away contiguous time slices (e.g., days) where the variance is <= 0.1 MW. Stop throwing away days when the variance exceeds 0.1 MW.
*   **Metadata Handling (`packages/nged_json_data/src/nged_json_data/metadata.py`):**
    *   Implement `upsert_metadata(new_metadata: pl.DataFrame, metadata_path: Path)`.
    *   If the local Parquet file doesn't exist, save `new_metadata`.
    *   If it exists, read it, compare with `new_metadata`. If there are differences, update the Parquet file and log a prominent warning message (e.g., using Dagster's `get_dagster_logger().warning`).
*   **Delta Table Storage (`packages/nged_json_data/src/nged_json_data/storage.py`):**
    *   Implement `append_to_delta(df: pl.DataFrame, delta_path: Path)`.
    *   Use `deltalake` to merge the new data, ensuring no duplicate `(time_series_id, end_time)` pairs are inserted.
    *   Partition the Delta table by `time_series_id`.
    *   Save the new Delta table to `data/NGED/delta/timeseries/`.

## 3. Partitioning Considerations for the New NGED JSON Asset

The S3 data consists of a one-off multi-year archive and 6-hourly live updates (containing ~2 weeks of rolling data).

*   **Architect's Recommendation:** Instead of trying to force a single complex partition definition, we should create **two separate Dagster assets** that write to the same underlying Delta table:
    1.  `nged_json_archive_asset`: An unpartitioned asset (or dynamically partitioned by `time_series_id` if memory is an issue). This is run manually *once* to bootstrap the Delta table from the historical S3 files.
    2.  `nged_json_live_asset`: A time-partitioned asset using a `TimeWindowPartitionsDefinition` (e.g., `cron_schedule="0 */6 * * *"`, 6-hourly). This asset will fetch the `<start_unix>_<end_unix>/` directory corresponding to its time window, process the JSONs, update metadata, and merge the new power data into the Delta table.

This approach cleanly separates the one-off historical backfill from the ongoing live ingestion, making the Dagster UI much easier to manage.

## Review Responses & Rejections

*(To be filled after review)*
