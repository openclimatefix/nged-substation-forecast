---
status: "draft"
version: "v1.9"
after_reviewer: "tester_reviewer_loop3"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/geo", "packages/dynamical_data", "packages/xgboost_forecaster"]
---

# Implementation Plan: Address Tester & Reviewer Loop 3 Flaws (Missing "Why" Comments)

## Summary of Recent Feedback
The Tester and Reviewer identified 7 flaws in Loop 3, all related to missing "why" comments that explain the intent and rationale behind specific constants, heuristics, and parallelization patterns. The architecture and code should be easy for a junior engineer to understand by "connecting the dots" across the codebase.

The 7 flaws are:
1. **25km Buffer Rationale**: Explain the 25km buffer in `packages/geo/src/geo/assets.py` (relates to NWP grid resolution).
2. **Child Resolution Heuristic**: Explain `child_res = h3_res + 2` in `packages/geo/src/geo/h3.py` and `packages/geo/src/geo/assets.py` (balance between precision and memory).
3. **Parallel I/O Rationale**: Explain the use of `ThreadPoolExecutor` for S3 downloads in `packages/dynamical_data/src/dynamical_data/processing.py`.
4. **NWP Filename Contract**: Document the expected filename format and add validation/error handling in `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`.
5. **H3 Resolution Standard**: Explain why resolution 5 is the standard in `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`.
6. **Lead Time Limit**: Explain the 336-hour (14-day) cap in `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`.
7. **Radiation Interpolation Caveat**: Acknowledge the limitations of linear interpolation for solar radiation in `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`.

## Step-by-Step Plan

### 1. Add Comment for 25km Buffer Rationale (FLAW-001)
**Target File:** `packages/geo/src/geo/assets.py`
**Action:** In `uk_boundary`, update the comment above `shape_osgb.buffer(25000)`.
**Comment Content:** Explain that the 25km buffer ensures that even the most coastal H3 cells will have at least one overlapping NWP grid cell (given the 0.25-degree resolution, which is ~28km at UK latitudes). This prevents coastal substations from losing coverage from the nearest NWP grid points.

### 2. Add Comment for Child Resolution Heuristic (FLAW-002)
**Target Files:**
- `packages/geo/src/geo/h3.py`
- `packages/geo/src/geo/assets.py`
**Action:**
- In `geo/h3.py` (`compute_h3_grid_weights`), add a comment to the docstring or near the `child_res` parameter explaining the trade-off.
- In `geo/assets.py` (`gb_h3_grid_weights`), add a comment above `child_res = config.child_res if config.child_res is not None else h3_res + 2`.
**Comment Content:** Explain that `+2` provides ~49 sample points per H3 cell, which is a sufficient balance between spatial precision (for area-weighting against a 0.25-degree grid) and computation time/memory overhead. Increasing it further could cause an exponential explosion in the number of child cells and potentially trigger OOM errors.

### 3. Add Comment for Parallel I/O Rationale (FLAW-003 / FLAW-005)
**Target File:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** In `download_ecmwf`, add comments near `xr.open_zarr(..., chunks=None)` and `ThreadPoolExecutor`.
**Comment Content:**
- For `chunks=None`: Clarify that this disables Dask because we want to manually control the parallelization of S3 fetches without Dask overhead.
- For `ThreadPoolExecutor`: Explain that the download is I/O bound (S3 network requests) and `ThreadPoolExecutor` is used to parallelize network latency across multiple variables. `ProcessPoolExecutor` would be less efficient due to serialization overhead for Xarray objects.

### 4. Document NWP Filename Contract (FLAW-004)
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `load_nwp_data_for_init_time` and `load_nwp_data_for_date_range`, add a comment explaining the filename format. Improve the error message in `load_nwp_data_for_init_time`.
**Comment Content:** Explain that the filename format (`YYYY-MM-DDTHHZ.parquet`) is a strict contract with the upstream data pipeline. If parsing fails, the error message should indicate the expected format.

### 5. Add Comment for H3 Resolution Standard (FLAW-005)
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `DataConfig`, replace the TODO comment for `h3_res: int = 5`.
**Comment Content:** Explain why resolution 5 is the fixed standard for this model (e.g., it balances spatial precision with feature dimensionality for the XGBoost model).

### 6. Add Comment for Lead Time Limit (FLAW-003)
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `process_nwp_data`, add a comment above the `.filter((pl.col("lead_time_hours") >= 3) & (pl.col("lead_time_hours") <= 336))` line.
**Comment Content:** Explain the choice of 336 hours (14 days). For example, state that ECMWF ENS reliability drops significantly after day 14, or the model is only validated for a 14-day horizon.

### 7. Add Comment for Radiation Interpolation Caveat (FLAW-004)
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `process_nwp_data`, add a comment above the linear interpolation of numeric columns (`pl.col(c).interpolate()`).
**Comment Content:** Acknowledge the physical limitation of using linear interpolation for solar radiation (`downward_short_wave_radiation_flux_surface`). Explain that linear interpolation between 3-hourly NWP points will "cut the corners" of the diurnal solar cycle, potentially underestimating peak solar generation. It is used as a baseline, and future iterations could use a clear-sky model to better preserve the diurnal cycle.

## Review Responses & Rejections

* **FLAW-001 (25km Buffer Rationale):** ACCEPTED. Adding the rationale prevents future maintainers from unknowingly breaking coastal substation coverage.
* **FLAW-002 (Child Resolution Heuristic):** ACCEPTED. Explaining the `+2` heuristic prevents junior engineers from causing OOM errors by arbitrarily increasing the resolution.
* **FLAW-003 (Parallel I/O Rationale):** ACCEPTED. Clarifying the manual parallelization strategy prevents others from incorrectly switching to `ProcessPoolExecutor` or re-enabling Dask without understanding the I/O bound nature of the task.
* **FLAW-004 (NWP Filename Contract):** ACCEPTED. Documenting the strict filename contract and improving error messages makes debugging pipeline failures much easier.
* **FLAW-005 (H3 Resolution Standard):** ACCEPTED. Explaining why resolution 5 is the standard provides necessary context for the model's spatial design.
* **FLAW-003 (Lead Time Limit):** ACCEPTED. Documenting the 14-day cap clarifies whether it's a model limitation or a data reliability choice.
* **FLAW-004 (Radiation Interpolation Caveat):** ACCEPTED. Acknowledging the limitations of linear interpolation for radiation variables shows scientific rigor and provides a clear path for future improvements.
