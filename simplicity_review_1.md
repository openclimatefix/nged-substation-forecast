---
total_flaws: 4
test_status: pass
---

# Simplicity Review

## FLAW-001: Logic Duplication in `substation_data_quality`
* **File & Line Number:** `src/nged_substation_forecast/defs/nged_assets.py`, lines 511-591
* **The Issue:** The `substation_data_quality` asset check re-implements the stuck sensor and insane value detection logic that already exists in `packages/nged_data/src/nged_data/cleaning.py`.
* **Concrete Failure Mode:** If the cleaning thresholds or logic change in `cleaning.py`, the asset check will become out of sync, leading to false positives or false negatives in data quality monitoring.
* **Required Fix:** Refactor `packages/nged_data/src/nged_data/cleaning.py` to expose the core cleaning expressions or a "diagnostic" version of the cleaning function that returns the masks. Alternatively, have the asset check use the same `clean_substation_flows` function and compare the input and output to identify "bad" rows.

## FLAW-002: Manual Lookback Logic Duplication
* **File & Line Number:** `src/nged_substation_forecast/defs/data_cleaning_assets.py` (lines 129-131) and `src/nged_substation_forecast/defs/nged_assets.py` (lines 487-489)
* **The Issue:** The calculation of `partition_start`, `partition_end`, and `lookback_start` is duplicated.
* **Concrete Failure Mode:** Inconsistency if the lookback period needs to change (e.g., from 1 day to 2 days).
* **Required Fix:** Create a helper function (perhaps in `partitions.py` or a utility module) that takes a `partition_key` and returns the `(start, end, lookback_start)` tuple.

## FLAW-003: Complex Metadata Upsert Logic
* **File & Line Number:** `src/nged_substation_forecast/defs/nged_assets.py`, lines 427-434
* **The Issue:** The manual join and coalesce for upserting metadata is verbose and uses a deprecated `how='outer'` (should be `how='full'`).
* **Concrete Failure Mode:** Harder to maintain and read than idiomatic Polars.
* **Required Fix:** Use `pl.concat([existing_metadata, new_metadata]).unique(subset=["substation_number"], keep="last")`. This is much simpler and idiomatic Polars for an "upsert" where the latest data is preferred.

## FLAW-004: Redundant `ensure_utc_timestamp_lazy` Boilerplate
* **File & Line Number:** Multiple locations in `data_cleaning_assets.py` and `nged_assets.py`.
* **The Issue:** `pl.scan_delta(path).pipe(ensure_utc_timestamp_lazy)` is repeated multiple times.
* **Concrete Failure Mode:** Unnecessary boilerplate and risk of forgetting to call it when scanning Delta tables.
* **Required Fix:** While `get_cleaned_actuals_lazy` exists for one table, a more general `scan_nged_delta(settings, table_name)` helper would centralize the path construction and UTC enforcement.

# Recommendations for Simplicity

1. **Centralize Partition Math**: Move the `partition_start/end/lookback` logic to `partitions.py`.
2. **Share Cleaning Logic**: The `substation_data_quality` check should be the "canary" for the cleaning logic. It should use the exact same code to identify issues.
3. **Idiomatic Polars**: Prefer `unique(keep='last')` for upserts unless there's a specific reason to merge column-by-column.
