---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: true
target_modules: ["packages/contracts", "packages/nged_data", "src/nged_substation_forecast"]
---
# Implementation Plan: Add h3_res_5 to TimeSeriesMetadata

## 1. Update `TimeSeriesMetadata` Contract
**File:** `packages/contracts/src/contracts/data_schemas.py`
- Add `h3_res_5: int = pt.Field(dtype=pl.UInt64)` to the `TimeSeriesMetadata` class.
- Add a description explaining that this is the H3 resolution 5 index for the substation location, computed during ingestion.

## 2. Update `load_nged_json` to Compute `h3_res_5`
**File:** `packages/nged_data/src/nged_data/io.py`
- Import `polars_h3 as plh3`.
- In `load_nged_json`, before validating `metadata_df` against `TimeSeriesMetadata`, compute the `h3_res_5` column:
  ```python
  if "h3_res_5" not in metadata_df.columns:
      metadata_df = metadata_df.with_columns(
          h3_res_5=plh3.latlng_to_cell(pl.col("latitude"), pl.col("longitude"), 5).cast(pl.UInt64)
      )
  ```

## 3. Add `polars-h3` Dependency to `nged_data`
**File:** `packages/nged_data/pyproject.toml`
- Add `"polars-h3>=0.6.1"` to the `dependencies` list.

## 4. Remove Redundant `ensure_h3_res_5` Calls
**File:** `src/nged_substation_forecast/utils.py`
- Remove the `ensure_h3_res_5` function entirely.
- Remove the `import polars_h3 as plh3` import.

**File:** `src/nged_substation_forecast/defs/xgb_assets.py`
- Remove `from nged_substation_forecast.utils import ensure_h3_res_5`.
- Remove the `time_series_metadata = ensure_h3_res_5(time_series_metadata)` calls in `train_xgboost` and `evaluate_xgboost`.

**File:** `src/nged_substation_forecast/defs/weather_assets.py`
- Remove `from nged_substation_forecast.utils import ensure_h3_res_5`.
- Remove the `metadata = ensure_h3_res_5(metadata)` call in `processed_nwp_data`.

## 5. Clean Up Untracked Scratchpad Scripts (Optional)
- If any untracked scratchpad scripts (e.g., `test_h3_fix.py`, `check_h3_mismatch_fixed.py`, `test_ensure_h3_fix.py`, `test_h3_function.py`, `inspect_metadata.py`) exist and import `ensure_h3_res_5`, they can be safely ignored or deleted as they are not part of the tracked codebase.

## Review Responses & Rejections
(None yet)
