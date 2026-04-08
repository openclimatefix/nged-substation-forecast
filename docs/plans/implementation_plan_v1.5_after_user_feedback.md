---
status: "draft"
version: "v1.5"
after_reviewer: "user_feedback"
task_type: "data-ingestion"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["packages/nged_data", "packages/nged_json_data", "src/nged_substation_forecast/defs", "packages/contracts", "src/nged_substation_forecast/ingestion"]
---

# Implementation Plan: Ingesting NGED's New S3 JSON Data

This plan outlines the steps to deprecate the existing CKAN ingestion pipeline and introduce a new pipeline for NGED's new JSON data format, as requested in `plan_for_ingesting_NGEDs_new_S3_data.md`.

**Important Note for Builder:** You are strictly forbidden from referencing `FLAW-XXX` IDs in any code comments. Code comments must focus on the *why* (intent and rationale) rather than the *how*, and should "connect the dots" across the codebase to explain how new components relate to existing ones.

## 1. Deprecating the Existing CKAN Code

We will keep the CKAN code as a fallback but clearly mark it as deprecated. To minimize code churn, we will **not** rename the existing package.

*   **Deprecation Warnings:** Add `warnings.warn("...", DeprecationWarning)` and update docstrings in `packages/nged_data` to clearly indicate that this package is deprecated in favor of `nged_json_data`.
*   **Dagster UI Grouping:** In `src/nged_substation_forecast/defs/nged_assets.py`, add `group_name="NGED CKAN (Deprecated)"` to all `@asset` decorators related to the old pipeline to visually separate them in the Dagster UI.

## 2. Refactoring Dagster Assets

The current `nged_assets.py` file is too large and contains multiple responsibilities. We will refactor it to improve maintainability.

*   **Extract Helpers:** Move helper functions (e.g., `_download_and_process_substation`, `_merge_to_delta`, `_get_processed_substations`) from `src/nged_substation_forecast/defs/nged_assets.py` into a dedicated module, such as `src/nged_substation_forecast/ingestion/helpers.py`. This will keep the asset definitions clean and focused solely on Dagster orchestration.

## 3. New Code to Import JSON Data

We will create a new package to handle the JSON data ingestion.

*   **Create New Package:** Create `packages/nged_json_data` with a standard Python package structure (`pyproject.toml`, `src/nged_json_data/`, `tests/`).
*   **Data Contracts (`packages/contracts/src/contracts/data_schemas.py`):**
    *   *Architect's Recommendation:* We should convert the JSON data to a Polars DataFrame first and verify the DataFrame using a Patito data contract. This is much faster and aligns perfectly with our existing Polars-heavy architecture.
    *   Create a new contract `NgedJsonPowerFlows` for the time series data (columns: `time_series_id`, `end_time`, `value`).
    *   Update `SubstationMetadata` to use a single contract for both CKAN and JSON data. Add a new field `time_series_name` as the canonical name (which will eventually replace `substation_name_in_location_table`). Include the new fields: `time_series_id` (Primary Key), `time_series_type`, `units`, `licence_area`, `substation_number`, `substation_type`, `latitude`, `longitude`, `information`, `area_wkt`, `area_center_lat`, `area_center_lon`. Make the old CKAN fields optional to maintain backwards compatibility.
*   **Loading & Processing Logic (`packages/nged_json_data/src/nged_json_data/io.py`):**
    *   Implement `load_nged_json(file_path: Path) -> tuple[pl.DataFrame, pl.DataFrame]` using `pl.read_json()`. Extract the metadata into one DataFrame and `explode().struct.unnest()` the `data` array into a time series DataFrame.
*   **Data Cleaning (`packages/nged_json_data/src/nged_json_data/clean.py`):**
    *   Break down the data cleaning process into smaller, single-responsibility functions to improve readability and testability: `sort_data`, `calculate_rolling_variance`, `apply_variance_threshold`, and `validate_data`.
    *   The main `clean_power_data(df: pl.DataFrame, variance_threshold: float = 0.1) -> pl.DataFrame` function will orchestrate these smaller functions.
    *   The `variance_threshold` must be a single, configurable parameter applied to all time series.
    *   **Crucial:** Heavily document the rationale for this daily variance threshold in the code comments, explaining *why* this approach is taken (to handle stuck sensors). Ensure the function fails loudly (raises an exception) if *all* rows are removed by this filter.
    *   **Logging:** Replace any `print` statements with proper logging using `dagster.get_dagster_logger()` or the standard `logging` module.
    *   **Error Handling:** Maintain aggressive error handling. `validate_data` and `clean_power_data` must raise a `ValueError` if the DataFrame is empty after filtering, ensuring the pipeline fails loudly on missing data.
*   **Metadata Handling (`packages/nged_json_data/src/nged_json_data/metadata.py`):**
    *   Implement `upsert_metadata(new_metadata: pl.DataFrame, metadata_path: Path)`.
    *   **Concurrency Control:** To avoid race conditions and complex conflict resolution, we will designate a single Dagster asset (the `nged_json_archive_asset`) as the *exclusive* owner of metadata updates. The `nged_json_live_asset` will be restricted from updating metadata. Dagster's concurrency limits will be configured to ensure only one instance of the metadata-updating asset can run at a time.
    *   If the local Parquet file doesn't exist, save `new_metadata`.
    *   If it exists, read it, and compare with `new_metadata` using a more efficient method (e.g., comparing hashes of the DataFrames or checking file metadata) rather than an $O(N)$ element-wise comparison. If there are differences, update the Parquet file and log a prominent warning message (e.g., using Dagster's `get_dagster_logger().warning`).
*   **Delta Table Storage (`packages/nged_json_data/src/nged_json_data/storage.py`):**
    *   Implement `append_to_delta(df: pl.DataFrame, delta_path: Path)`.
    *   Use `deltalake` to merge the new data, ensuring no duplicate `(time_series_id, end_time)` pairs are inserted.
    *   **Concurrency Control:** Enforce sequential ingestion at the Dagster asset level to avoid concurrency issues entirely. Do not implement complex retry logic for concurrent writes. Because we enforce sequential ingestion, we do not need atomic operations in `append_to_delta`.
    *   **Refactoring:** Ensure `append_to_delta` is refactored to remove any redundant code blocks, keeping the logic DRY and maintainable.
    *   Partition the Delta table by `time_series_id`.
    *   Save the new Delta table to `data/NGED/delta/timeseries/`.

## 4. Partitioning Considerations for the New NGED JSON Asset

The S3 data consists of a one-off multi-year archive and 6-hourly live updates (containing ~2 weeks of rolling data).

*   **Architect's Recommendation:** Instead of trying to force a single complex partition definition, we should create **two separate Dagster assets** that write to the same underlying Delta table:
    1.  `nged_json_archive_asset`: An unpartitioned asset (or dynamically partitioned by `time_series_id` if memory is an issue). This is run manually *once* to bootstrap the Delta table from the historical S3 files.
    2.  `nged_json_live_asset`: A time-partitioned asset using a `TimeWindowPartitionsDefinition` (e.g., `cron_schedule="0 */6 * * *"`, 6-hourly). This asset will fetch the `<start_unix>_<end_unix>/` directory corresponding to its time window, process the JSONs, update metadata, and merge the new power data into the Delta table. **Crucially, this asset must collect all cleaned dataframes from the individual JSON files and append them to the Delta table in a single operation** to reduce the number of Delta Lake transactions and improve efficiency.
    *   **Note on File Processing:** We will keep the current sequential file processing within the asset to maintain simplicity, rather than introducing complex parallel processing or batch scanning logic.

This approach cleanly separates the one-off historical backfill from the ongoing live ingestion, making the Dagster UI much easier to manage.

## 5. Testing Requirements

The implementation must include rigorous tests covering the following scenarios:
*   **Data Cleaning:** Empty/malformed input, daily variance logic edge cases (exactly at threshold, slightly above, all rows removed, insufficient rows).
*   **Metadata Handling:** File corruption, conflict resolution, and concurrent access (mocking the file lock).
*   **Delta Table Storage:** Duplicate handling within the new data itself, and correct partitioning.
*   **General:** Ensure all components fail loudly on invalid data.

## Review Responses & Rejections

*   **FLAW-001 (Scientist Code Review 1 - Current):** ACCEPTED. We will replace all `print` statements with proper logging using `dagster.get_dagster_logger()` or the standard `logging` module to prevent log pollution in production.
*   **FLAW-002 (Scientist Code Review 1 - Current):** REJECTED. We will keep the current aggressive error handling (raising `ValueError` for empty DataFrames). Failing loudly on missing data is preferred over silently skipping partitions, as it ensures data completeness issues are immediately flagged.
*   **FLAW-003 (Scientist Code Review 1 - Current):** ACCEPTED. We will optimize `upsert_metadata` to use a more efficient comparison method (e.g., hashing the DataFrames or checking file metadata) instead of an $O(N)$ element-wise comparison to prevent performance bottlenecks as the number of substations grows.
*   **FLAW-001 (Simplicity Review 1):** ACCEPTED. We will refactor `nged_assets.py` by moving helper functions (e.g., `_download_and_process_substation`, `_merge_to_delta`, `_get_processed_substations`) into a dedicated module (`src/nged_substation_forecast/ingestion/helpers.py`) to improve maintainability.
*   **FLAW-002 (Simplicity Review 1):** ACCEPTED. We will break `clean_power_data` into smaller, single-responsibility functions (`sort_data`, `calculate_rolling_variance`, `apply_variance_threshold`, `validate_data`) to improve readability and testability.
*   **FLAW-003 (Simplicity Review 1):** REJECTED. We will keep the current sequential file processing in `nged_json_live_asset` to maintain simplicity, as explicitly requested by the user.
*   **FLAW-001 (Scientist Code Review 1 - Previous):** REJECTED. We will enforce sequential ingestion at the Dagster asset level to prevent race conditions. This avoids the need for complex atomic operations in `append_to_delta`.
*   **FLAW-002 (Scientist Code Review 1 - Previous):** ACCEPTED. The `nged_json_live_asset` will be modified to collect all cleaned dataframes and append them in a single operation to improve Delta Lake efficiency.
*   **FLAW-003 (Scientist Code Review 1 - Previous):** ACCEPTED. The `append_to_delta` function will be refactored to remove redundant code.
*   **FLAW-001 (Scientist Plan Review 0):** REJECTED. The user explicitly requested to keep the current daily variance cleaning logic, as it is sufficient and simpler than a rolling window approach.
*   **FLAW-002 (Scientist Plan Review 0):** ACCEPTED (with modification). The `variance_threshold` will be made configurable, but as a single parameter for all time series rather than per-substation, per the user's request.
*   **FLAW-003 (Scientist Plan Review 0):** REJECTED. The user requested to enforce sequential ingestion to avoid concurrency issues entirely, rather than implementing complex retry logic for concurrent writes.
*   **FLAW-001 (Reviewer Plan Review 0):** ACCEPTED. The hardcoded 0.1 MW variance threshold in `clean_power_data` has been made a configurable parameter (`variance_threshold`).
*   **FLAW-002 (Reviewer Plan Review 0):** ACCEPTED. Potential race conditions in `upsert_metadata` will be mitigated by designating a single asset as the exclusive owner of metadata updates and configuring Dagster to prevent concurrent runs of this asset.
*   **FLAW-001 (Simplicity Plan Review 0):** ACCEPTED. The unnecessary and high-churn renaming of `packages/nged_data` to `packages/nged_ckan_data` has been rejected. We will keep the original name and use deprecation warnings instead.
*   **FLAW-002 (Simplicity Plan Review 0):** ACCEPTED. The complex data cleaning logic will be retained but made configurable and heavily documented to explain the rationale. We will also ensure it fails loudly if it removes all data.
