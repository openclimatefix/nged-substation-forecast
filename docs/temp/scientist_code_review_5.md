---
review_iteration: 5
reviewer: "scientist"
total_flaws: 5
critical_flaws: 5
---

# ML Rigor Review

## FLAW-001: Longitude Wrap-around Data Loss in Grid Snapping
* **File & Line Number:** `packages/geo/src/geo/h3.py`, lines 59-62
* **The Theoretical Issue:** The grid snapping formula `((lng + grid_size/2) / grid_size).floor() * grid_size` does not normalize longitudes. For H3 cells near the anti-meridian, `cell_to_lng` can return values near 180 or -180, resulting in `nwp_lng` bins of both 180.0 and -180.0 for the exact same physical meridian.
* **Concrete Failure Mode:** The NWP dataset in `processing.py` only contains one representation of the anti-meridian (e.g., `[-180, 180)`). The `left_join` will fail for the other representation, causing silent data loss (null weather features) for H3 cells near the anti-meridian.
* **Required Architectural Fix:** Normalize `nwp_lng` to a strict `[-180, 180)` range after binning (e.g., `(((nwp_lng + 180) % 360) - 180)`).

## FLAW-002: Unweighted Categorical Aggregation
* **File & Line Number:** `packages/dynamical_data/src/dynamical_data/processing.py`, line 325
* **The Theoretical Issue:** Categorical variables are aggregated using `pl.col(c).mode().first()`, which completely ignores the `proportion` column calculated during H3 grid weighting.
* **Concrete Failure Mode:** An NWP grid cell that covers only 1% of an H3 cell has the exact same "vote" in the mode calculation as a grid cell covering 99%. This leads to physically incorrect categorical weather features (e.g., assigning 'snow' to a cell that is 99% 'rain').
* **Required Architectural Fix:** Implement a weighted mode for categorical variables by grouping by `h3_index` and the categorical value, summing the `proportion`, and selecting the category with the highest sum.

## FLAW-003: Physically Unrealistic Wind Vector Interpolation
* **File & Line Number:** `packages/dynamical_data/src/dynamical_data/processing.py` (line 356) and `packages/xgboost_forecaster/src/xgboost_forecaster/data.py` (lines 197-220)
* **The Theoretical Issue:** `processing.py` drops the raw `u` and `v` wind components, forcing `data.py` to interpolate wind speed linearly and wind direction circularly. This treats polar coordinates as independent linear variables.
* **Concrete Failure Mode:** During a wind direction shift (e.g., North to South), the wind speed must physically drop to near zero. Independent interpolation keeps the wind speed artificially high while rotating the direction, creating phantom high winds that will corrupt the XGBoost model's predictions.
* **Required Architectural Fix:** Retain `wind_u` and `wind_v` in `processing.py`. In `data.py`, interpolate `u` and `v` linearly, and then calculate wind speed and direction from the interpolated components.

## FLAW-004: Circular Variable Min-Max Scaling Corruption
* **File & Line Number:** `packages/dynamical_data/src/dynamical_data/processing.py` (line 385) and `packages/xgboost_forecaster/src/xgboost_forecaster/data.py` (line 216)
* **The Theoretical Issue:** `scale_to_uint8` applies empirical min-max scaling to wind direction. `data.py` assumes that the resulting UInt8 value of 255 maps exactly to 360 degrees (`2 * math.pi`).
* **Concrete Failure Mode:** If the historical maximum wind direction in the dataset is 355 degrees, it will be scaled to 255. `data.py` will then incorrectly decode this as 360 degrees. Furthermore, min-max scaling a circular variable destroys its topology (0 and 360 are the same).
* **Required Architectural Fix:** Exclude circular variables (wind direction) from empirical min-max scaling. Scale them using fixed theoretical bounds (0 to 360) or store `u` and `v` components instead.

## FLAW-005: Hardcoded Lead Time Filter Causes Horizon Leakage
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`, lines 162-169
* **The Theoretical Issue:** `process_nwp_data` hardcodes a filter of `lead_time_hours >= 3`. When constructing historical weather for training, this retains highly accurate short-term forecasts (e.g., 3-hour ahead).
* **Concrete Failure Mode:** If the XGBoost model is trained for a Day-Ahead (24h) horizon, joining on `valid_time` will cause it to learn from 3-hour ahead weather forecasts. In production, it will only have access to 24-hour ahead forecasts, leading to a massive drop in real-world performance (lookahead bias).
* **Required Architectural Fix:** The lead time filter must be parameterized by the target forecasting horizon (e.g., `lead_time_hours >= target_horizon + publication_delay`), ensuring the model is trained on forecasts with the same accuracy as those available in production.
