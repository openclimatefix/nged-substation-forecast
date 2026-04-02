---
status: "draft"
version: "v1.8"
after_reviewer: "scientist_loop3"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/geo", "packages/dynamical_data", "packages/xgboost_forecaster"]
---

# Implementation Plan: Address Scientist Loop 3 Flaws (Missing "Why" Comments)

## Summary of Recent Feedback
The Scientist identified 4 flaws in Loop 3, all related to missing "why" comments that explain the intent and rationale behind specific mathematical and physical assumptions in the codebase. The architecture and code should be easy for a junior engineer to understand by "connecting the dots" across the codebase.

The 4 flaws are:
1. **Categorical Forward-fill**: Explain why linear interpolation is physically meaningless for categories.
2. **Circular Interpolation Assumption**: Document the assumption that 0-255 UInt8 maps perfectly to 0-360 degrees.
3. **Temporal Interpolation & Leakage**: Explain why interpolating over `valid_time` within a single `init_time` is not data leakage.
4. **Grid Snapping Formula**: Explain the mathematical intent behind the half-grid offset binning.

## Step-by-Step Plan

### 1. Add Comment for Categorical Forward-fill (FLAW-1)
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `process_nwp_data`, add a detailed comment above the `forward_fill()` operation for categorical variables.
**Comment Content:** Explain that linear interpolation is physically meaningless for categorical variables (e.g., a value of 1.5 between 'rain' (1) and 'snow' (2) has no physical interpretation). Therefore, we use forward-fill to maintain the discrete state of the weather condition until the next forecast step.

### 2. Add Comment for Circular Interpolation Assumption (FLAW-2)
**Target Files:**
- `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
- `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:**
- In `xgboost_forecaster/data.py` (`process_nwp_data`), add a comment above the sine/cosine decomposition explaining the assumption that the 0-255 UInt8 scale maps linearly to 0-360 degrees (0-2pi radians), where 255 is treated as 2pi (equivalent to 0 degrees).
- In `dynamical_data/processing.py` (`process_ecmwf_dataset`), add a comment near the `scale_to_uint8` call or wind direction calculation, explaining that wind direction (0-360 degrees) is scaled to 0-255 UInt8 for storage, and this assumption is relied upon by downstream interpolation logic.

### 3. Add Comment for Temporal Interpolation & Leakage (FLAW-3)
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `process_nwp_data`, add a comment above the `.over(["init_time", "h3_index", "ensemble_member"])` interpolation logic.
**Comment Content:** Explicitly state why interpolating over `valid_time` within a single `init_time` is *not* data leakage. Explain that all `valid_time` predictions in a single forecast run are generated simultaneously at `init_time`. We are not looking into the future of when the forecast was made (which would be leakage), but merely interpolating the forecast's own future predictions.

### 4. Add Comment for Grid Snapping Formula (FLAW-4)
**Target Files:**
- `packages/geo/src/geo/h3.py`
- `packages/geo/src/geo/assets.py`
**Action:**
- In `geo/h3.py` (`compute_h3_grid_weights`), add a detailed comment above the `nwp_lat` and `nwp_lng` calculation. Explain the mathematical intent behind the half-grid offset binning: `((lat + grid_size/2) / grid_size).floor() * grid_size`. Explain that adding `grid_size/2` before flooring ensures that points are snapped to the *closest* grid center rather than the bottom-left corner of the grid cell.
- In `geo/assets.py` (`gb_h3_grid_weights`), add a brief comment to the docstring or before calling `compute_h3_grid_weights` explaining that the `grid_size` parameter is used to snap high-resolution H3 cells to the nearest regular NWP grid points.

## Review Responses & Rejections

* **FLAW-1 (Categorical Forward-fill):** ACCEPTED. Adding the physical rationale improves code readability and prevents future developers from accidentally changing it to linear interpolation.
* **FLAW-2 (Circular Interpolation Assumption):** ACCEPTED. Documenting the implicit data contract (0-255 = 0-360 degrees) between the data ingestion and forecasting modules is critical for maintainability.
* **FLAW-3 (Temporal Interpolation & Leakage):** ACCEPTED. Clarifying the distinction between `init_time` and `valid_time` is essential for ML engineers to trust that the interpolation does not introduce data leakage.
* **FLAW-4 (Grid Snapping Formula):** ACCEPTED. The half-grid offset formula is a common source of off-by-one errors; explaining its mathematical intent makes the code much easier to understand.
