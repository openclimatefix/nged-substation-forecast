---
review_iteration: 5
reviewer: "review"
total_flaws: 5
critical_flaws: 1
---

# Code Review

## FLAW-001: Inefficient Delta Lake Read for Metadata Filtering
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`, line 44
* **The Issue:** The code uses `pl.read_delta` to eagerly load the entire `live_primary_flows` table into memory just to extract unique `substation_number` values.
* **Concrete Failure Mode:** If the `live_primary_flows` table grows to millions of rows (typical for telemetry data), this operation will become extremely slow and eventually trigger Out-Of-Memory (OOM) errors during a simple metadata loading step.
* **Required Fix:** Use `pl.scan_delta(str(config.base_power_path)).select("substation_number").unique().collect()` to perform a lazy, optimized scan that only reads the necessary column and metadata.

## FLAW-002: Brittle Filename Parsing for Historical Weather
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`, line 114
* **The Issue:** The code assumes that every `.parquet` file in the NWP directory starts with a 10-character date string (`YYYY-MM-DD`).
* **Concrete Failure Mode:** If a temporary file, a hidden file (e.g., `.DS_Store`), or a file with a different naming convention (e.g., `backup_2024-01-01.parquet`) is present in the directory, `datetime.strptime` will raise a `ValueError`, crashing the entire data loading pipeline.
* **Required Fix:** Use a more robust filtering mechanism, such as a regex match on the filename or a `try...except` block within the list comprehension to skip non-conforming files.

## FLAW-003: Hardcoded S3 Paths and Configuration
* **File & Line Number:** `packages/dynamical_data/src/dynamical_data/processing.py`, lines 134-137
* **The Issue:** The S3 bucket name and prefix for the ECMWF Icechunk store are hardcoded directly inside the `download_ecmwf` function.
* **Concrete Failure Mode:** Changing the data source (e.g., for a different model version or a different environment like staging/production) requires modifying the source code. This violates the principle of separating configuration from logic.
* **Required Fix:** Move these strings to the `Settings` class in `contracts.settings` or pass them as arguments to the function.

## FLAW-004: Redundant and Fragile Variable Lists
* **File & Line Number:** `packages/dynamical_data/src/dynamical_data/processing.py`, lines 44-58 and 164-178
* **The Issue:** The list of required NWP variables is defined twice within the same module: once for validation and once for subsetting during download.
* **Concrete Failure Mode:** If a new variable is added to the model requirements, a developer might update one list but forget the other, leading to either validation failures or missing data during processing.
* **Required Fix:** Define a single constant (e.g., `REQUIRED_NWP_VARS`) at the module level and use it in both locations.

## FLAW-005: Silent Data Loss in NWP Interpolation
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`, lines 203-208
* **The Issue:** Groups with only a single row are silently dropped (with a warning) because they cannot be interpolated.
* **Concrete Failure Mode:** If a specific forecast run or substation has a data gap that results in many single-row groups, a significant portion of the dataset could be discarded without the user realizing the scale of the loss. This could lead to biased training or missing predictions in production.
* **Required Fix:** The warning should include the percentage of total rows being dropped. Additionally, consider if these single-row groups can be handled by a different strategy (e.g., constant fill) if they represent valid but isolated data points.
