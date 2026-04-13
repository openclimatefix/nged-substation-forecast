---
status: "draft"
version: "v2.5"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules:
  - "src/nged_substation_forecast/defs/data_cleaning_assets.py"
  - "src/nged_substation_forecast/defs/xgb_assets.py"
---

# Implementation Plan: Address Code Review Flaws

## Overview
This plan addresses the flaws identified in `docs/temp/reviewer_code_review_4.md` and subsequent architectural feedback. The focus is on refactoring large functions to improve readability and maintainability (FLAW-001), optimizing the XGBoost data preparation pipeline to minimize inefficient eager evaluations (FLAW-002), and enforcing strict Patito typing across the codebase.

## 1. Global Rename: `sub_ids` to `time_series_ids`
*   **Action:** Search the entire codebase for the variable name `sub_ids` and rename all instances to `time_series_ids`. This ensures consistent terminology across the project, aligning with the `time_series_id` column name.

## 2. Refactoring `cleaned_power_time_series` (FLAW-001)
**File:** `src/nged_substation_forecast/defs/data_cleaning_assets.py`

*   **Action:** Extract the core data cleaning and validation logic from the `cleaned_power_time_series` asset into a new helper function named `_process_cleaned_partition`.
*   **Implementation Details:**
    *   Create `_process_cleaned_partition(raw_power_time_series: pt.DataFrame[PowerTimeSeries], settings: Settings, partition_start: datetime, partition_end: datetime) -> pt.DataFrame[PowerTimeSeries]`.
    *   Move the calls to `clean_power_time_series`, `.unique()`, `PowerTimeSeries.validate`, and `filter_to_partition_window` into this helper.
    *   Update the `cleaned_power_time_series` asset to call this helper, passing the materialized `df_joined_materialized`.
*   **Commenting Requirements:** Add a docstring to `_process_cleaned_partition` explaining *why* this logic is grouped (to separate pure data transformation/validation from Dagster asset I/O and partitioning logic). Do not reference FLAW IDs.

## 3. Optimizing and Refactoring `_prepare_xgboost_inputs` (FLAW-001 & FLAW-002)
**File:** `src/nged_substation_forecast/defs/xgb_assets.py`

*   **Action 1 (FLAW-002):** Optimize the lazy evaluation pipeline to prevent scanning the entire dataset when only a subset of substations is requested, and eliminate multiple `collect()` calls.
    *   Apply the `config.time_series_ids` filter to `power_time_series` *before* computing healthy substations.
    *   Compute `valid_ids_lf` lazily: `power_time_series.filter(pl.col("power").is_not_null()).select("time_series_id").unique()`.
    *   Filter the main dataset lazily using an inner join: `power_time_series.join(valid_ids_lf, on="time_series_id", how="inner")`.
    *   Perform a single `.collect()` on `valid_ids_lf` to get the `time_series_ids` list for metadata filtering and empty checks.
*   **Action 2 (FLAW-001):** Extract the data loading and filtering logic into a new helper function `_get_filtered_time_series_data`.
    *   Create `_get_filtered_time_series_data(context: dg.AssetExecutionContext, config: XGBoostConfig, settings: dg.ResourceParam[Settings]) -> tuple[pt.LazyFrame[PowerTimeSeries], pt.DataFrame[SubstationMetadata], list[int]]`.
    *   Move the metadata loading, lazy frame filtering (from Action 1), and `time_series_ids` extraction into this helper.
    *   Update `_prepare_xgboost_inputs` to call this helper, leaving it responsible only for MLflow and Hydra setup orchestration.
*   **Action 3 (FLAW-002):** Remove eager `.collect()` calls used purely for logging in downstream assets.
    *   In `evaluate_xgboost`, remove `cast(pl.DataFrame, power_time_series_filtered.collect()).shape` from the logging statement. Log the number of target substations (`len(time_series_ids)`) instead, as the exact row count cannot be known without evaluating the lazy frame.
*   **Commenting Requirements:**
    *   In `_get_filtered_time_series_data`, explicitly comment *why* the join approach is used (to push down filters and avoid materializing the entire dataset just to find healthy substations).
    *   Update the docstring for `_get_filtered_time_series_data` to clearly explain what the `list[int]` in the return type represents (e.g., "Returns a list of valid time series IDs that have sufficient non-null power data for training/evaluation").
    *   Ensure no FLAW-XXX IDs are referenced in any source code comments.

## Review Responses & Rejections

* **FLAW-001 (Reviewer):** ACCEPTED. The functions `cleaned_power_time_series` and `_prepare_xgboost_inputs` will be refactored by extracting core logic into `_process_cleaned_partition` and `_get_filtered_time_series_data` respectively, bringing them well under the 50-line limit.
* **FLAW-002 (Reviewer):** ACCEPTED. The data preparation pipeline will be optimized by applying ID filters *before* checking for healthy substations, chaining operations via a lazy join, and removing unnecessary `.collect()` calls used for logging in downstream assets. This ensures the lazy computation graph is evaluated efficiently.
* **Architectural Feedback:** ACCEPTED. Updated `_process_cleaned_partition` and `_get_filtered_time_series_data` to use strict `pt.DataFrame` and `pt.LazyFrame` types. Renamed `sub_ids` to `time_series_ids` globally for consistency.
