---
review_iteration: 1
reviewer: "scientist"
total_flaws: 3
critical_flaws: 1
---

# ML Rigor Review: NGED JSON Ingestion Pipeline

## FLAW-001: Aggressive Data Cleaning (Critical)
* **File & Line Number:** `packages/nged_json_data/src/nged_json_data/clean.py`, lines 48-55
* **The Theoretical Issue:** The `clean_power_data` function calculates variance per day and drops the *entire day* if the variance is below a threshold. This is scientifically unsound as it removes valid low-load periods (e.g., overnight) and fails to handle sensors that are only stuck for a few hours.
* **Concrete Failure Mode:** The model will lose significant amounts of valid, low-load training data, leading to biased forecasts that overestimate load during quiet periods.
* **Required Architectural Fix:** Redesign the cleaning logic to use a rolling window for variance calculation instead of a daily aggregate, and only nullify the specific time slices that are identified as "stuck" rather than dropping the entire day.

## FLAW-002: Hardcoded Variance Threshold
* **File & Line Number:** `packages/nged_json_data/src/nged_json_data/clean.py`, line 7
* **The Theoretical Issue:** The `variance_threshold` is hardcoded as a default (0.1). This threshold is unlikely to be appropriate for all substations, which have different load profiles and noise characteristics.
* **Concrete Failure Mode:** The model will either fail to filter out genuinely stuck sensors (if the threshold is too low) or incorrectly filter out valid data (if the threshold is too high), leading to inconsistent data quality across the grid.
* **Required Architectural Fix:** Make the `variance_threshold` a configurable parameter, ideally per substation, and document the rationale for its selection.

## FLAW-003: Lack of Concurrency Control in Delta Appends
* **File & Line Number:** `packages/nged_json_data/src/nged_json_data/storage.py`, lines 40-43, 73-76
* **The Theoretical Issue:** The `append_to_delta` function lacks the robust retry logic for concurrent writes that is present in `src/nged_substation_forecast/defs/nged_assets.py`.
* **Concrete Failure Mode:** In a distributed or parallel ingestion environment, multiple processes attempting to append to the same Delta table will likely cause race conditions and write failures.
* **Required Architectural Fix:** Implement a retry mechanism for concurrent writes in `append_to_delta`, similar to the one used in `_merge_to_delta` in `nged_assets.py`.
