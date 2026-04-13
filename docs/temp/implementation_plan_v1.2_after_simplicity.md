---
status: "draft"
version: "v1.3"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules:
  - "src/nged_substation_forecast/utils.py"
  - "src/nged_substation_forecast/defs/nged_assets.py"
  - "src/nged_substation_forecast/defs/data_cleaning_assets.py"
---
# Implementation Plan: Simplicity and Refactoring

## Overview
This plan addresses the flaws identified in `docs/temp/simplicity_review_1.md` and subsequent feedback. It focuses on refactoring shared logic into simple, reusable functions, improving error handling to prevent silent failures, using concrete Patito types, and consolidating data cleaning logic to correctly handle rolling windows across partition boundaries.

## Step-by-Step Implementation

### Step 1: Add Reusable Utility Functions (`src/nged_substation_forecast/utils.py`)
Add the following utility functions to abstract partition filtering and delta table logic (Addressing FLAW-004). Use concrete Patito types for DataFrames.

1.  **`filter_new_delta_records`**:
    *   **Purpose**: Filters a DataFrame to only include records newer than the maximum timestamp in an existing Delta table.
    *   **Arguments**: `df: pt.DataFrame[PowerTimeSeries]`, `delta_path: Path`
    *   **Return type**: `pt.DataFrame[PowerTimeSeries]`
    *   **Logic**: Reads the `period_end_time` from the Delta table (if it exists and is not empty), finds the max timestamp, and filters `df` where `period_end_time > max_timestamp`.

2.  **`filter_to_partition_window`**:
    *   **Purpose**: Filters a DataFrame to strictly within a partition window `[start, end)`.
    *   **Arguments**: `df: pt.DataFrame[PowerTimeSeries]`, `partition_start: datetime`, `partition_end: datetime`.
    *   **Return type**: `pt.DataFrame[PowerTimeSeries]`

3.  **`get_delta_partition_predicate`**:
    *   **Purpose**: Generates a SQL predicate string for Delta table partition replacement.
    *   **Arguments**: `partition_start: datetime`, `partition_end: datetime`, `time_column: str = "period_end_time"`.
    *   **Returns**: `f"{time_column} >= '{partition_start.isoformat()}' AND {time_column} < '{partition_end.isoformat()}'"`

### Step 2: Consolidate and Correct Data Cleaning (`src/nged_substation_forecast/defs/data_cleaning_assets.py`)
Address FLAW-001 and FLAW-004 by consolidating cleaning logic and fixing the rolling window boundary issue.

1.  **Delete `cleaned_actuals`**:
    *   Remove the `cleaned_actuals` asset entirely. Its logic will be consolidated into `cleaned_power_time_series`.
2.  **Update `cleaned_power_time_series`**:
    *   **Correct Rolling Window Logic**: To prevent values from being incorrectly set to zero or null at midnight (the start of a daily partition), the asset must read a larger window of raw data.
    *   **Logic Flow**:
        1.  Determine the `partition_start` and `partition_end`.
        2.  Calculate an `extended_start` (e.g., `partition_start - timedelta(days=1)`) to provide sufficient context for rolling window calculations.
        3.  Read `raw_power_time_series` from `extended_start` to `partition_end`.
        4.  Apply the cleaning and rolling window logic to this extended DataFrame.
        5.  Use `filter_to_partition_window` to strictly filter the cleaned DataFrame back to the `[partition_start, partition_end)` window.
        6.  Write the strictly filtered DataFrame to the Delta table using `get_delta_partition_predicate`.

### Step 3: Refactor NGED Assets (`src/nged_substation_forecast/defs/nged_assets.py`)
Address FLAW-001, FLAW-003, and FLAW-004.

1.  **Create `process_nged_json_file`**:
    *   Add a helper function in this file to handle the repeated file processing logic: loading JSON, upserting metadata, and validating.
    *   **Arguments**: `json_file: Path`, `metadata_path: Path`, `context: AssetExecutionContext`.
    *   **Error Handling (FLAW-003)**: Wrap the `PowerTimeSeries.validate` call in a `try...except ValueError` block. Instead of swallowing the error with a warning, log an error with `context.log.error` and **raise a `RuntimeError`** chaining the original exception. This ensures the pipeline fails visibly on bad data.
2.  **Update `nged_json_archive_asset`**:
    *   Refactor the loop to use `process_nged_json_file`.
3.  **Update `nged_json_live_asset`**:
    *   Refactor the loop to use `process_nged_json_file`.
    *   Replace the manual Delta table max-timestamp checking logic with the new `filter_new_delta_records` utility from `utils.py`.
4.  **Update `nged_sharepoint_json_asset`**:
    *   Refactor the loop to use `process_nged_json_file`.
    *   Remove the existing silent `try...except` block, as the helper function will now properly raise the error.

## Code Comments Mandate
*   **Focus on the Why**: All new code comments must explain the intent and rationale behind the code, not just repeat what the code does.
*   **Connect the Dots**: Comments should explain how these new utility functions relate to the broader pipeline (e.g., explaining *why* we filter new delta records to maintain idempotency across the ingestion layer, and *why* we read an extended window for cleaning).
*   **No FLAW IDs**: The Builder is strictly forbidden from referencing FLAW-XXX IDs in any source code comments.

## Review Responses & Rejections

*   **FLAW-001 (Simplicity Review):** ACCEPTED. Shared logic in both asset files will be extracted into standard Python helper functions (`process_nged_json_file`). The `cleaned_actuals` asset is deleted and consolidated into `cleaned_power_time_series` to simplify the pipeline.
*   **FLAW-002 (Simplicity Review):** REJECTED. Ignored based on explicit user instructions.
*   **FLAW-003 (Simplicity Review):** ACCEPTED. The silent `try...except` block in `nged_sharepoint_json_asset` will be replaced. Validation errors will now be caught, logged with context, and re-raised as a `RuntimeError` to ensure the pipeline fails visibly.
*   **FLAW-004 (Simplicity Review):** ACCEPTED. Complex partition filtering and Delta table predicate logic will be abstracted into well-tested, reusable utility functions in `utils.py` (`filter_new_delta_records`, `filter_to_partition_window`, `get_delta_partition_predicate`).
