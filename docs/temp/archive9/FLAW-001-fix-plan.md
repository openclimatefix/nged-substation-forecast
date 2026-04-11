---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/nged_data/src/nged_data/io.py", "packages/nged_data/tests/test_io.py"]
---

# Implementation Plan: Fix FLAW-001 (Incorrect Datetime Parsing)

## 1. Context and Rationale
The current implementation of `load_nged_json` in `packages/nged_data/src/nged_data/io.py` attempts to parse `period_end_time` using a hardcoded format string: `format="%Y-%m-%d %H:%M:%S%z"`. This fails when encountering standard ISO 8601 datetime strings (e.g., `2026-01-01T00:00:00Z`), which use a `T` separator and a `Z` timezone indicator. 

Polars' `str.to_datetime()` method natively supports parsing mixed ISO 8601 formats (both with `T` and space separators, and various timezone indicators like `Z` or `+00:00`) when the `time_zone` argument is provided directly, without needing an explicit `format` string.

## 2. Identify the Failing Test
Run the test suite to confirm the failure:
```bash
pytest packages/nged_data/tests/test_io.py
```
You will see `test_load_nged_json_valid` failing with an `InvalidOperationError` due to the inability to parse `"2026-01-01T00:00:00Z"`.

## 3. Modify the Code
Update `packages/nged_data/src/nged_data/io.py`.

**Target File:** `packages/nged_data/src/nged_data/io.py`
**Location:** Around line 110.

**Current Code:**
```python
    # Try parsing with different formats
    time_series_df = time_series_df.with_columns(
        pl.col("period_end_time")
        .str.to_datetime(format="%Y-%m-%d %H:%M:%S%z")
        .dt.replace_time_zone("UTC")
    )
```

**New Code:**
```python
    # Parse ISO 8601 datetime strings directly to UTC.
    # Polars natively infers ISO 8601 formats (including 'T' separators and 'Z' timezones)
    # when no explicit format string is provided.
    time_series_df = time_series_df.with_columns(
        pl.col("period_end_time")
        .str.to_datetime(time_zone="UTC")
    )
```

*Note: Ensure the comment explains the "why" (Polars natively infers ISO 8601 formats) rather than just the "how". Do not include FLAW-001 in the code comments.*

## 4. Verify the Fix
Run the test suite again to ensure the fix resolves the issue and doesn't break existing functionality:
```bash
pytest packages/nged_data/tests/test_io.py
```
All tests should pass.

## Review Responses & Rejections

* **FLAW-001 (Reviewer):** ACCEPTED. The hardcoded format string `%Y-%m-%d %H:%M:%S%z` was too restrictive and failed on standard ISO 8601 strings containing `T` and `Z`. The fix leverages Polars' native ISO 8601 inference by removing the explicit format string and specifying `time_zone="UTC"`.
