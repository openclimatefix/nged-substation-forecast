---
review_iteration: 0
reviewer: "tester"
total_flaws: 0
critical_flaws: 0
test_status: "fully_tested"
---

# Testability & QA Review of Implementation Plan (v0_updated)

The implementation plan is well-structured for testing. The separation of concerns into distinct modules (`io`, `clean`, `metadata`, `storage`) and the use of `Patito` for data contracts provide a solid foundation for rigorous testing.

## Key Testability Observations

1.  **Data Contracts:** The use of `Patito` for `NgedJsonPowerFlows` is excellent. I will be able to write adversarial tests that deliberately violate these schemas to ensure the pipeline fails loudly.
2.  **Modular Design:** The functions `load_nged_json`, `clean_power_data`, `upsert_metadata`, and `append_to_delta` are well-isolated, which will allow for fast, targeted unit tests.
3.  **Asset Separation:** Separating the archive and live ingestion assets is a good architectural decision that simplifies testing the two distinct ingestion workflows.

## Required Test Coverage & Edge Cases

To ensure robustness, the following scenarios must be covered by the test suite:

### 1. Data Cleaning (`clean_power_data`)
*   **Empty/Malformed Input:** Test with empty DataFrames, DataFrames with missing columns, and DataFrames with all-null values.
*   **Variance Logic:** Test the "bad pre-amble data" logic with:
    *   Data with variance exactly 0.1 MW.
    *   Data with variance slightly above 0.1 MW.
    *   Data where *all* rows are removed by the variance filter (the pipeline must fail loudly, not return an empty DataFrame).
    *   Data with insufficient rows to calculate variance.

### 2. Metadata Handling (`upsert_metadata`)
*   **File Corruption:** Test behavior when the existing metadata Parquet file is corrupted or unreadable.
*   **Conflict Resolution:** Test scenarios where `new_metadata` conflicts with existing metadata (e.g., different `units` for the same `time_series_id`). Ensure the warning is logged and the update is handled correctly.

### 3. Delta Table Storage (`append_to_delta`)
*   **Duplicate Handling:** Test the merge logic with input data that contains duplicate `(time_series_id, end_time)` pairs *within the new data itself*.
*   **Partitioning:** Verify that the data is correctly partitioned by `time_series_id` in the Delta table.

### 4. General Robustness
*   **Loud Failures:** Ensure that all components fail loudly (raise exceptions) when encountering invalid data, rather than silently propagating NaNs or returning empty results.

I am confident that this plan is highly testable. I will proceed with writing the necessary tests as the implementation progresses.
