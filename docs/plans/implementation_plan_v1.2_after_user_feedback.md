---
status: "draft"
version: "v1.2"
after_reviewer: "user_feedback"
task_type: "data-ingestion"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["packages/nged_data", "packages/nged_json_data", "src/nged_substation_forecast/defs", "packages/contracts"]
---

# Implementation Plan: Ingesting NGED's New S3 JSON Data

This plan outlines the steps to deprecate the existing CKAN ingestion pipeline and introduce a new pipeline for NGED's new JSON data format, as requested in `plan_for_ingesting_NGEDs_new_S3_data.md`.

**Important Note for Builder:** You are strictly forbidden from referencing `FLAW-XXX` IDs in any code comments. Code comments must focus on the *why* (intent and rationale) rather than the *how*, and should "connect the dots" across the codebase to explain how new components relate to existing ones.

## 1. Deprecating the Existing CKAN Code

We will keep the CKAN code as a fallback but clearly mark it as deprecated. To minimize code churn, we will **not** rename the existing package.

*   **Deprecation Warnings:** Add `warnings.warn("...", DeprecationWarning)` and update docstrings in `packages/nged_data` to clearly indicate that this package is deprecated in favor of `nged_json_data`.
*   **Dagster UI Grouping:** In `src/nged_substation_forecast/defs/nged_assets.py`, add `group_name="NGED CKAN (Deprecated)"` to all `@asset` decorators related to the old pipeline to visually separate them in the Dagster UI.

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
    *   Implement `clean_power_data(df: pl.DataFrame, variance_threshold: float = 0.1) -> pl.DataFrame`. This function will:
        1.  Sort by `end_time`.
        2.  Drop any rows where `value` is null.
        3.  Identify "stuck" sensors by calculating variance over daily aggregates.
        4.  Drop entire days where the daily variance falls below the `variance_threshold`.
        5.  The `variance_threshold` must be a single, configurable parameter applied to all time series.
        6.  **Crucial:** Heavily document the rationale for this daily variance threshold in the code comments, explaining *why* this approach is taken (to handle stuck sensors). Ensure the function fails loudly (raises an exception) if *all* rows are removed by this filter.
*   **Metadata Handling (`packages/nged_json_data/src/nged_json_data/metadata.py`):**
    *   Implement `upsert_metadata(new_metadata: pl.DataFrame, metadata_path: Path)`.
    *   **Concurrency Control:** To avoid race conditions and complex conflict resolution, we will designate a single Dagster asset (the `nged_json_archive_asset`) as the *exclusive* owner of metadata updates. The `nged_json_live_asset` will be restricted from updating metadata. Dagster's concurrency limits will be configured to ensure only one instance of the metadata-updating asset can run at a time.
    *   If the local Parquet file doesn't exist, save `new_metadata`.
    *   If it exists, read it, compare with `new_metadata`. If there are differences, update the Parquet file and log a prominent warning message (e.g., using Dagster's `get_dagster_logger().warning`).
*   **Delta Table Storage (`packages/nged_json_data/src/nged_json_data/storage.py`):**
    *   Implement `append_to_delta(df: pl.DataFrame, delta_path: Path)`.
    *   Use `deltalake` to merge the new data, ensuring no duplicate `(time_series_id, end_time)` pairs are inserted.
    *   **Concurrency Control:** Enforce sequential ingestion at the Dagster asset level to avoid concurrency issues entirely. Do not implement complex retry logic for concurrent writes.
    *   Partition the Delta table by `time_series_id`.
    *   Save the new Delta table to `data/NGED/delta/timeseries/`.

## 3. Partitioning Considerations for the New NGED JSON Asset

The S3 data consists of a one-off multi-year archive and 6-hourly live updates (containing ~2 weeks of rolling data).

*   **Architect's Recommendation:** Instead of trying to force a single complex partition definition, we should create **two separate Dagster assets** that write to the same underlying Delta table:
    1.  `nged_json_archive_asset`: An unpartitioned asset (or dynamically partitioned by `time_series_id` if memory is an issue). This is run manually *once* to bootstrap the Delta table from the historical S3 files.
    2.  `nged_json_live_asset`: A time-partitioned asset using a `TimeWindowPartitionsDefinition` (e.g., `cron_schedule="0 */6 * * *"`, 6-hourly). This asset will fetch the `<start_unix>_<end_unix>/` directory corresponding to its time window, process the JSONs, update metadata, and merge the new power data into the Delta table.

This approach cleanly separates the one-off historical backfill from the ongoing live ingestion, making the Dagster UI much easier to manage.

## 4. Testing Requirements

The implementation must include rigorous tests covering the following scenarios:
*   **Data Cleaning:** Empty/malformed input, daily variance logic edge cases (exactly at threshold, slightly above, all rows removed, insufficient rows).
*   **Metadata Handling:** File corruption, conflict resolution, and concurrent access (mocking the file lock).
*   **Delta Table Storage:** Duplicate handling within the new data itself, and correct partitioning.
*   **General:** Ensure all components fail loudly on invalid data.

## Review Responses & Rejections

*   **FLAW-001 (Scientist):** REJECTED. The user explicitly requested to keep the current daily variance cleaning logic, as it is sufficient and simpler than a rolling window approach.
*   **FLAW-002 (Scientist):** ACCEPTED (with modification). The `variance_threshold` will be made configurable, but as a single parameter for all time series rather than per-substation, per the user's request.
*   **FLAW-003 (Scientist):** REJECTED. The user requested to enforce sequential ingestion to avoid concurrency issues entirely, rather than implementing complex retry logic for concurrent writes.
*   **FLAW-001 (Reviewer):** ACCEPTED. The hardcoded 0.1 MW variance threshold in `clean_power_data` has been made a configurable parameter (`variance_threshold`).
*   **FLAW-002 (Reviewer):** ACCEPTED. Potential race conditions in `upsert_metadata` will be mitigated by designating a single asset as the exclusive owner of metadata updates and configuring Dagster to prevent concurrent runs of this asset.
*   **FLAW-001 (Simplicity):** ACCEPTED. The unnecessary and high-churn renaming of `packages/nged_data` to `packages/nged_ckan_data` has been rejected. We will keep the original name and use deprecation warnings instead.
*   **FLAW-002 (Simplicity):** ACCEPTED. The complex data cleaning logic will be retained but made configurable and heavily documented to explain the rationale. We will also ensure it fails loudly if it removes all data.
