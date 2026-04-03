# Simplicity Audit: Implementation Plan v0

## Overall Assessment
The plan is generally a direct set of fixes for existing warnings and exceptions. However, Step 2 (Polars sortedness) is significantly over-engineered and introduces unnecessary complexity.

## Specific Improvements

### FLAW-001: Over-engineered Polars Sort
* **Location:** Section 2 (Polars UserWarning)
* **The Issue:** The plan proposes sorting by `[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX, "available_time"]` on both sides of the join and setting `check_sortedness=False`.
* **Why it is not simple:** `join_asof` only requires the `on` column to be sorted. The `by` columns are used for grouping, not for the asof search. Explicitly sorting by the group keys is overkill and potentially slow.
* **Simplest Approach:**
    1. Keep the sort on `available_time` only.
    2. Simply pass `check_sortedness=False` to suppress the warning.
* **Proposed Simplification:**
    Change the proposed "to" block to:
    ```python
    combined_nwps = (
        combined_nwps.sort("available_time")
        .join_asof(
            other_nwp.sort("available_time"),
            on="available_time",
            by=[NwpColumns.VALID_TIME, NwpColumns.H3_INDEX],
            check_sortedness=False,
        )
    )
    ```
    This removes the redundant sorting of `VALID_TIME` and `H3_INDEX` while still silencing the warning.

### FLAW-002: MLflow URI Redundancy
* **Location:** Section 5 (MLflow FutureWarning)
* **The Issue:** Adding a setting to `Settings` and then calling `mlflow.set_tracking_uri` in every utility function.
* **Simplest Approach:** Use the `MLFLOW_TRACKING_URI` environment variable. MLflow automatically reads this variable, eliminating the need for explicit code changes in multiple files.
* **Recommendation:** Remove the proposed code changes to `packages/ml_core/src/ml_core/utils.py` and instead handle this via infrastructure/environment configuration.

## Summary of Changes to Plan
- **Step 2:** Revert the complex multi-column sort; use only `sort("available_time")` with `check_sortedness=False`.
- **Step 5:** Replace explicit `set_tracking_uri` calls with environment variable configuration.
