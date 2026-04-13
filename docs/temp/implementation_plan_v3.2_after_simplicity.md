---
status: "draft"
version: "v3.2"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["src/nged_substation_forecast/defs/data_cleaning_assets.py", "packages/nged_data/src/nged_data/clean.py"]
---

# Implementation Plan v3.2: Simplicity Refactoring

This plan addresses the flaws identified in `docs/temp/simplicity_review_3.md`, focusing on removing redundant wrappers and leveraging Patito's native type hints to eliminate unnecessary casting.

## 1. Refactor `data_cleaning_assets.py` (FLAW-001 & FLAW-002)

**File:** `src/nged_substation_forecast/defs/data_cleaning_assets.py`

*   **Remove Redundant Wrapper:** Delete the `_process_cleaned_partition` function entirely. Its responsibilities (duplicate removal and validation) are already handled by `clean_power_time_series` in `packages/nged_data`.
*   **Update Asset Signature:** Change the return type hint of the `cleaned_power_time_series` asset from `pl.DataFrame` to `pt.DataFrame[PowerTimeSeries]`.
*   **Eliminate Casts:** Remove the `cast(pl.DataFrame, ...)` and `cast(pt.DataFrame[PowerTimeSeries], ...)` calls. Rely on the fact that `raw_flows.collect()` natively returns a `pt.DataFrame[PowerTimeSeries]` because `raw_flows` is a `pt.LazyFrame[PowerTimeSeries]`.
*   **Direct Invocation:** In `cleaned_power_time_series`, call `clean_power_time_series` directly with the collected DataFrame, followed by `filter_to_partition_window`.

**Code Changes in `cleaned_power_time_series`:**
```python
    # Materialize the LazyFrame once (no casts needed)
    df_joined_materialized = raw_flows.collect()

    context.log.info(f"Materialized data shape before cleaning: {df_joined_materialized.shape}")

    # Clean the data using the shared cleaning module directly
    df_cleaned = clean_power_time_series(
        df_joined_materialized,
        stuck_std_threshold=settings.data_quality.stuck_std_threshold,
        min_mw_threshold=settings.data_quality.min_mw_threshold,
        max_mw_threshold=settings.data_quality.max_mw_threshold,
    )

    # Filter to current partition's time window
    validated_df = filter_to_partition_window(df_cleaned, partition_start, partition_end)
```

## 2. Refactor `clean.py` (FLAW-002)

**File:** `packages/nged_data/src/nged_data/clean.py`

*   **Update Signatures:** Update `clean_power_time_series` and `validate_data` to accept `df: pt.DataFrame[PowerTimeSeries]` instead of `pl.DataFrame`. Update `calculate_rolling_variance` to return `pt.DataFrame[PowerTimeSeries]`.
*   **Remove Redundant Validation:** Remove the initial `df = PowerTimeSeries.validate(df)` call at the start of `clean_power_time_series`, as the input is now guaranteed to be a validated `pt.DataFrame[PowerTimeSeries]`.
*   **Eliminate Casts:** Remove the `cast` calls in `sort_data` and `clean_power_time_series` (around `df.unique()`). Patito's `sort()` and `unique()` methods natively return the correct `pt.DataFrame[Model]` subclass.

## Review Responses & Rejections

*   **FLAW-001 (Simplicity Review):** ACCEPTED. The `_process_cleaned_partition` wrapper in `data_cleaning_assets.py` is redundant. The core logic (duplicate removal and validation) is already handled by `clean_power_time_series` in `packages/nged_data`. We will remove the wrapper and call the core function directly, making the asset a thin wrapper.
*   **FLAW-002 (Simplicity Review):** ACCEPTED. We will remove the unnecessary `cast` calls in both `data_cleaning_assets.py` and `clean.py`. Patito's `LazyFrame.collect()`, `sort()`, and `unique()` methods natively return the correct `pt.DataFrame[Model]` subclasses, so we can rely on the type system without manual casting. The return type of `cleaned_power_time_series` will be updated to `pt.DataFrame[PowerTimeSeries]`.

## Code Commenting Mandate

*   Ensure that any new or modified comments focus on the *why* (intent and rationale) rather than the *how*.
*   Do NOT include FLAW-XXX IDs in any source code comments.
