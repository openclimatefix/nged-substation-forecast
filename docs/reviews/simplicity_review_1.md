---
review_iteration: 1
reviewer: "simplicity_audit"
total_flaws: 3
critical_flaws: 0
---

# Simplicity Audit: NGED JSON Ingestion Pipeline

## FLAW-001: Monolithic Asset File
* **File:** `src/nged_substation_forecast/defs/nged_assets.py`
* **The Issue:** The file is too large (nearly 700 lines) and contains multiple, distinct responsibilities: asset definitions, ingestion logic, data cleaning, and reporting.
* **Required Fix:** Refactor `nged_assets.py` by moving the helper functions (e.g., `_download_and_process_substation`, `_merge_to_delta`, `_get_processed_substations`) into a dedicated module (e.g., `src/nged_substation_forecast/ingestion/helpers.py`). This will make the asset definitions much cleaner and easier to maintain.

## FLAW-002: Redundant Data Cleaning Logic
* **File:** `packages/nged_json_data/src/nged_json_data/clean.py`
* **The Issue:** The `clean_power_data` function is doing too much: sorting, dropping nulls, calculating rolling variance, nullifying, and dropping nulls again. This is complex and hard to test.
* **Required Fix:** Break `clean_power_data` into smaller, single-responsibility functions: `sort_data`, `calculate_rolling_variance`, `apply_variance_threshold`, and `validate_data`. This will improve readability and testability.

## FLAW-003: Inefficient File Processing in `nged_json_live_asset`
* **File:** `src/nged_substation_forecast/defs/nged_assets.py`
* **The Issue:** The `nged_json_live_asset` iterates over files one by one, loading and cleaning them individually. This is inefficient for large numbers of files.
* **Required Fix:** Use `polars` to scan the directory and process all files in parallel or in a single batch operation if possible, rather than a Python-level loop. This will significantly improve performance.
