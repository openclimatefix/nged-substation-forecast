---
status: "draft"
version: "v5.1"
after_reviewer: "scientist"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: true
target_modules: ["packages/geo", "packages/dynamical_data", "packages/xgboost_forecaster", "packages/contracts", "src/nged_substation_forecast"]
---

# Implementation Plan: Address Scientist Loop 5 Critical Flaws

## Summary of Changes
This plan addresses the 5 critical logical and physical flaws identified by the Scientist in Loop 5. The fixes ensure mathematical correctness for geospatial operations, physical realism for weather interpolation, and eliminate lookahead bias in the ML pipeline.

## Step-by-Step Plan

### 1. Fix FLAW-001: Longitude Wrap-around in Grid Snapping
**Target File:** `packages/geo/src/geo/h3.py`
**Action:** In `compute_h3_grid_weights`, update the `nwp_lng` calculation to normalize longitudes to the strict `[-180, 180)` range.
**Code Snippet:**
```python
            nwp_lng=((((plh3.cell_to_lng("child_h3") + (grid_size / 2)) / grid_size).floor() * grid_size + 180) % 360) - 180,
```
**Rationale Comment:** Add a comment explaining that this normalization prevents silent data loss for H3 cells near the anti-meridian, ensuring they correctly join with the NWP dataset.

### 2. Fix FLAW-002: Weighted Categorical Aggregation
**Target File:** `packages/dynamical_data/src/dynamical_data/processing.py`
**Action:** In `process_ecmwf_dataset`, replace the unweighted mode aggregation (`pl.col(c).mode().first()`) with a weighted mode calculation that respects the H3 cell overlap proportions.
**Code Snippet:**
```python
    # Aggregate numeric variables
    agg_exprs = [
        pl.when(pl.col(f"{x}_weight_sum").sum() > 0)
        .then(pl.col(f"{x}_weighted").sum() / pl.col(f"{x}_weight_sum").sum())
        .otherwise(None)
        .alias(str(x))
        for x in numeric_vars
    ]
    df_numeric = df.group_by(["h3_index", "step"]).agg(agg_exprs)

    # Aggregate categorical variables using weighted mode
    for c in categorical_vars:
        df_cat = (
            df.group_by(["h3_index", "step", c])
            .agg(pl.col("proportion").sum().alias("weight"))
            .sort(["h3_index", "step", "weight"])
            .group_by(["h3_index", "step"])
            .agg(pl.col(c).last().cast(pl.UInt8))
        )
        df_numeric = df_numeric.join(df_cat, on=["h3_index", "step"], how="left")

    df_h3 = df_numeric
```
**Rationale Comment:** Add a comment explaining that this ensures categorical weather features (like precipitation type) accurately reflect the area-weighted majority of the H3 cell, rather than treating a 1% overlap the same as a 99% overlap.

### 3. Fix FLAW-003 & FLAW-004: Wind Vector Interpolation & Scaling
**Target File 1:** `packages/contracts/src/contracts/data_schemas.py`
- Update `Nwp` schema: Replace `wind_speed_10m`, `wind_direction_10m`, `wind_speed_100m`, `wind_direction_100m` (UInt8) with `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m` (Float32).
- Update `SubstationFeatures` schema: Replace `wind_speed_10m_uint8_scaled`, etc. with `wind_speed_10m`, `wind_direction_10m`, `wind_speed_100m`, `wind_direction_100m` (Float32).

**Target File 2:** `packages/dynamical_data/src/dynamical_data/processing.py`
- Remove the `calculate_wind_speed_and_direction` function entirely.
- In `process_ecmwf_dataset`, do NOT drop the raw `wind_u` and `wind_v` components.
- Exclude wind components from empirical min-max scaling so they remain as Float32:
  ```python
  scaling_params = scaling_params.filter(
      ~pl.col("col_name").is_in([
          "categorical_precipitation_type_surface",
          "wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"
      ])
  )
  ```
- Remove the comment `CIRCULAR INTERPOLATION ASSUMPTION (FLAW-2)`.

**Target File 3:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
- Remove the circular variable decomposition (sine/cosine) and reconstruction logic.
- Allow `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m` to be interpolated linearly along with other numeric columns.
- After interpolation, calculate physical wind speed and direction:
  ```python
  processed = processed.with_columns(
      [
          (pl.col("wind_u_10m")**2 + pl.col("wind_v_10m")**2).sqrt().alias("wind_speed_10m"),
          ((pl.arctan2("wind_u_10m", "wind_v_10m") * 180 / math.pi + 180) % 360).alias("wind_direction_10m"),
          (pl.col("wind_u_100m")**2 + pl.col("wind_v_100m")**2).sqrt().alias("wind_speed_100m"),
          ((pl.arctan2("wind_u_100m", "wind_v_100m") * 180 / math.pi + 180) % 360).alias("wind_direction_100m"),
      ]
  ).drop(["wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m"])
  ```
**Rationale Comment:** Add comments explaining that interpolating Cartesian components (`u`, `v`) is physically realistic and avoids phantom high winds during direction shifts. Storing them as Float32 avoids destroying circular topology via min-max scaling.

### 4. Fix FLAW-005: Horizon Leakage in Lead Time Filter
**Target File 1:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
- Update `process_nwp_data` signature to accept `target_horizon_hours: int` and `publication_delay_hours: int = 3`.
- Update the lead time filter:
  ```python
  min_lead_time = target_horizon_hours + publication_delay_hours
  lf = lf.filter((pl.col("lead_time_hours") >= min_lead_time) & (pl.col("lead_time_hours") <= 336))
  ```

**Target File 2:** `src/nged_substation_forecast/defs/weather_assets.py`
- Update `ProcessedNWPConfig` to include `target_horizon_hours: int = 24` and `publication_delay_hours: int = 3`.
- Pass these parameters to `process_nwp_data` in the `processed_nwp_data` asset:
  ```python
  return process_nwp_data(
      all_nwp_data,
      h3_indices,
      target_horizon_hours=config.target_horizon_hours,
      publication_delay_hours=config.publication_delay_hours
  )
  ```
**Rationale Comment:** Add a comment explaining that parameterizing the lead time filter by the target horizon eliminates lookahead bias, ensuring the model is trained on forecasts with the exact same accuracy as those available in production.

## Review Responses & Rejections

* **FLAW-001 (Longitude Wrap-around):** ACCEPTED. Normalizing `nwp_lng` prevents silent data loss near the anti-meridian.
* **FLAW-002 (Unweighted Categorical Aggregation):** ACCEPTED. Implementing a weighted mode ensures physically correct categorical features.
* **FLAW-003 (Wind Vector Interpolation):** ACCEPTED. Interpolating Cartesian `u` and `v` components prevents physically unrealistic phantom winds during direction shifts.
* **FLAW-004 (Circular Variable Scaling):** ACCEPTED. Excluding wind components from empirical min-max scaling and storing them as Float32 avoids destroying circular topology and simplifies downstream processing.
* **FLAW-005 (Horizon Leakage):** ACCEPTED. Parameterizing the lead time filter by the target horizon eliminates lookahead bias and ensures production-realistic training data.
