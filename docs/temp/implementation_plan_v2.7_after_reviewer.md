---
status: "draft"
version: "v2.7"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules:
  - "src/nged_substation_forecast/utils.py"
---

# Implementation Plan: Address Code Review Flaws

## Overview
This plan addresses FLAW-002 identified in `docs/temp/reviewer_code_review_5.md` regarding a type hinting inconsistency in `src/nged_substation_forecast/utils.py`. FLAW-001 is explicitly rejected based on user instructions.

## 1. Fix Type Hinting Inconsistency in `scan_delta_table` (FLAW-002)
**File:** `src/nged_substation_forecast/utils.py`

*   **Action:** Update the `scan_delta_table` function to correctly return a `pt.LazyFrame[PowerTimeSeries]` at runtime, rather than just casting a `pl.LazyFrame`.
*   **Implementation Details:**
    *   Modify the return statement of `scan_delta_table` to use Patito's `from_existing` and `set_model` methods.
    *   Change: `return cast(pt.LazyFrame[PowerTimeSeries], pl.scan_delta(delta_path))`
    *   To: `return pt.LazyFrame.from_existing(pl.scan_delta(delta_path)).set_model(PowerTimeSeries)`
    *   Keep the `from typing import cast` import as it is still used in `filter_new_delta_records`.
*   **Commenting Requirements:** Add a brief comment explaining *why* `from_existing` and `set_model` are used (to ensure the returned object is strictly a Patito LazyFrame with the correct schema attached, satisfying both runtime and static type checking). Do not reference FLAW IDs.

## Review Responses & Rejections

* **FLAW-001 (Reviewer):** REJECTED. The hardcoded path in Dagster Definitions (`/home/jack/dagster_home/storage`) is intentionally kept as is based on explicit user instructions.
* **FLAW-002 (Reviewer):** ACCEPTED. The `scan_delta_table` function will be updated to use `pt.LazyFrame.from_existing(...).set_model(...)` to ensure the returned object strictly matches the `pt.LazyFrame[PowerTimeSeries]` type hint according to Patito's API.
