---
status: "draft"
version: "v5.2"
after_reviewer: "tester"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["packages/geo", "packages/dynamical_data", "packages/xgboost_forecaster"]
---

# Implementation Plan: Address Tester Loop 5 QA Flaws

## Summary of Changes
This plan addresses the 6 robustness and edge-case flaws identified by the Tester in Loop 5. The fixes ensure the geospatial and dynamical data processing pipelines are resilient to empty datasets, anti-meridian crossings, and sparse temporal data.

## Step-by-Step Plan

### 1. Fix FLAW-001: Missing `grid_size` validation
**Target File:** `packages/geo/src/geo/h3.py`
**Action:** In `compute_h3_grid_weights`, add validation to ensure `grid_size` is strictly positive.
**Code Snippet:**
```python
    if df.is_empty():
        raise ValueError("Input DataFrame is empty.")

    if grid_size <= 0:
        raise ValueError("grid_size must be strictly positive")
```

### 2. Fix FLAW-006: Inconsistent H3 resolution handling
**Target File:** `packages/geo/src/geo/h3.py`
**Action:** In `compute_h3_grid_weights`, validate that all H3 indices have the same resolution using `polars_h3.get_resolution`.
**Code Snippet:** Replace the `first_h3` check with:
```python
    # Check resolution
    h3_res_unique = df.select(plh3.get_resolution("h3_index")).unique()
    if h3_res_unique.height > 1:
        raise ValueError("All H3 indices must have the same resolution.")
    h3_res = h3_res_unique.item()
    if child_res <= h3_res:
        raise ValueError(f"child_res ({child_res}) must be strictly greater than h3_res ({h3_res})")
```

### 3. Fix FLAW-003: Lack of robustness to empty coordinates
**Target File:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** In `download_ecmwf`, check if the dataset's spatial coordinates are empty before computing bounds.
**Code Snippet:** Add before `ds.longitude.min()`:
```python
    if ds.longitude.size == 0 or ds.latitude.size == 0:
        raise ValueError("Dataset has empty longitude or latitude coordinates.")
```

### 4. Fix FLAW-002: Anti-meridian slicing bug
**Target File:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** In `download_ecmwf`, implement wrap-around aware slicing for longitudes.
**Code Snippet:** Replace the `min_lng, max_lng` extraction and `ds_cropped` creation with:
```python
    # Find spatial bounds from grid
    min_lat, max_lat = h3_grid.select(
        pl.col("nwp_lat").min(),
        pl.col("nwp_lat").max(),
    ).row(0)

    # Robust slicing: xarray slice(a, b) is sensitive to coordinate direction.
    if len(ds.latitude.values) > 1:
        lat_is_descending = ds.latitude.values[0] > ds.latitude.values[1]
        lat_slice = slice(max_lat, min_lat) if lat_is_descending else slice(min_lat, max_lat)
    else:
        lat_slice = slice(min_lat, max_lat)

    lngs = h3_grid.get_column("nwp_lng").unique().sort()
    diffs = lngs.diff().drop_nulls()
    crosses_antimeridian = len(diffs) > 0 and diffs.max() > 180

    if crosses_antimeridian:
        gap_idx = diffs.arg_max()
        max_neg_lng = lngs[gap_idx]
        min_pos_lng = lngs[gap_idx + 1]
        ds_neg = ds.sel(latitude=lat_slice, longitude=slice(-180, max_neg_lng), init_time=nwp_init_time)
        ds_pos = ds.sel(latitude=lat_slice, longitude=slice(min_pos_lng, 180), init_time=nwp_init_time)
        ds_cropped = xr.concat([ds_pos, ds_neg], dim="longitude")
    else:
        min_lng, max_lng = lngs[0], lngs[-1]
        ds_cropped = ds.sel(latitude=lat_slice, longitude=slice(min_lng, max_lng), init_time=nwp_init_time)
```

### 5. Fix FLAW-004: Fragile spatial intersection handling
**Target File:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** In `download_ecmwf`, explicitly check for an empty spatial intersection after slicing.
**Code Snippet:** Add immediately after `ds_cropped` is created:
```python
    if ds_cropped.longitude.size == 0 or ds_cropped.latitude.size == 0:
        raise ValueError("No spatial overlap found between H3 grid and NWP dataset.")
```

### 6. Fix FLAW-005: Interpolation failure on single-row groups
**Target File:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
**Action:** In `process_nwp_data`, filter out groups with only 1 row before attempting interpolation, and log a warning.
**Code Snippet:** Add before `df = df.sort("valid_time")`:
```python
    # Ensure each group has at least two points for interpolation
    group_counts = df.group_by(["init_time", "h3_index", "ensemble_member"]).len()
    single_row_groups = group_counts.filter(pl.col("len") == 1)
    if single_row_groups.height > 0:
        log.warning(f"Dropping {single_row_groups.height} groups with only 1 row as they cannot be interpolated.")
        df = df.filter(pl.len().over(["init_time", "h3_index", "ensemble_member"]) > 1)

    if df.is_empty():
        return lf.limit(0)
```

## Review Responses & Rejections

* **FLAW-001 (grid_size validation):** ACCEPTED. Added strict positivity check to prevent division by zero or nonsensical snapping.
* **FLAW-002 (Anti-meridian slicing):** ACCEPTED. Implemented gap detection in longitudes to split the slice when crossing the anti-meridian, preventing massive unnecessary downloads.
* **FLAW-003 (Empty coordinates):** ACCEPTED. Added explicit checks for empty coordinates before computing bounds to fail gracefully.
* **FLAW-004 (Empty spatial intersection):** ACCEPTED. Added explicit check after slicing to prevent downstream `KeyError`s during DataFrame conversion.
* **FLAW-005 (Single-row interpolation):** ACCEPTED. Added filtering and logging for single-row groups to prevent violation of the 30-minute temporal resolution contract.
* **FLAW-006 (Inconsistent H3 resolution):** ACCEPTED. Replaced the single-row resolution check with a vectorized uniqueness check using `polars_h3.get_resolution` to ensure all cells are valid.
