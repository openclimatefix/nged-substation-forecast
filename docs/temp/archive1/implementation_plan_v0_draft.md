---
status: "draft"
version: "v0"
after_reviewer: "simplicity_audit"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/nged_json_data/src/nged_json_data/storage.py", "packages/nged_json_data/src/nged_json_data/clean.py"]
---

# Implementation Plan: Address Simplicity Audit Findings

This plan addresses the flaws identified in `docs/reviews/simplicity_review_1.md` to improve scalability, reduce cognitive load, and clean up logging.

## 1. Fix Scalability Issue in `append_to_delta` (FLAW-001)

**File:** `packages/nged_json_data/src/nged_json_data/storage.py`

**Changes:**
- Replace the memory-intensive anti-join logic with Delta Lake's native `merge` operation.
- This ensures that as the Delta table grows, we don't load all existing keys into memory.

**Implementation Details:**
```python
    # Load the existing Delta table
    dt = DeltaTable(str(delta_path))

    # Use Delta Lake's merge operation to insert only new records
    # This efficiently prevents duplicate (time_series_id, end_time) entries
    # without loading the full table into memory.
    (
        dt.merge(
            source=df.to_arrow(),
            predicate="s.time_series_id = t.time_series_id AND s.end_time = t.end_time",
            source_alias="s",
            target_alias="t",
        )
        .when_not_matched_insert_all()
        .execute()
    )
```
- Remove the `existing_keys` loading and `df.join(..., how="anti")` logic.

## 2. Remove Unnecessary Wrapper Functions (FLAW-002)

**File:** `packages/nged_json_data/src/nged_json_data/clean.py`

**Changes:**
- Delete the `sort_data` function.
- In `clean_power_data`, replace `df = sort_data(df)` with `df = df.sort("end_time")`.

## 3. Simplify Verbose Cleaning Logic (FLAW-003)

**File:** `packages/nged_json_data/src/nged_json_data/clean.py`

**Changes:**
- Delete the `calculate_rolling_variance` and `apply_variance_threshold` functions.
- Inline the logic into `clean_power_data` using a single, idiomatic Polars chain.
- Instead of nullifying values and then dropping nulls, directly filter the DataFrame.

**Implementation Details:**
Replace the calls to `calculate_rolling_variance` and `apply_variance_threshold` with:
```python
    # 3. Identify and filter out 'bad pre-amble data'
    # We calculate a rolling variance to identify flatlined/bad pre-amble data from the sensors.
    # We filter out any rows where the variance is below the threshold or null.
    df = (
        df.join(
            df.rolling(index_column="end_time", period="6h").agg(
                rolling_variance=pl.col("value").var()
            ),
            on="end_time"
        )
        .filter(
            pl.col("rolling_variance").is_not_null() & (pl.col("rolling_variance") > threshold)
        )
        .drop("rolling_variance")
    )
```

## 4. Remove Noisy Logging (FLAW-004)

**File:** `packages/nged_json_data/src/nged_json_data/clean.py`

**Changes:**
- Remove all `dagster.get_dagster_logger().info(...)` statements that log intermediate DataFrames.
- Specifically, remove the logs previously in `apply_variance_threshold` and the log after dropping nulls in `clean_power_data`.
- If `dagster` is no longer used in `clean.py` after removing the logs, remove the `import dagster` statement.

## Review Responses & Rejections

* **FLAW-001 (Simplicity Audit):** ACCEPTED. The anti-join approach is not scalable. Delta Lake's `merge` operation is the correct, scalable solution.
* **FLAW-002 (Simplicity Audit):** ACCEPTED. `sort_data` adds unnecessary cognitive load.
* **FLAW-003 (Simplicity Audit):** ACCEPTED. The verbose functions will be replaced with a concise Polars chain.
* **FLAW-004 (Simplicity Audit):** ACCEPTED. Intermediate DataFrame logging will be removed to clean up logs.

## Commenting Requirements

- Ensure that the `append_to_delta` function has a clear comment explaining *why* `merge` is used (to efficiently prevent duplicate `(time_series_id, end_time)` entries without loading the full table into memory).
- Ensure that the rolling variance filtering logic in `clean_power_data` has a comment explaining *why* we filter based on variance (to remove flatlined/bad pre-amble data from the sensors).
- **Crucial:** Do not include `FLAW-XXX` IDs in any code comments.
