---
review_iteration: 3
reviewer: "scientist"
total_flaws: 4
critical_flaws: 3
---

# ML Rigor Review

## FLAW-001: [Critical] Dynamic Seasonal Lag Logic Remains Hardcoded
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 140-150
* **The Theoretical Issue:** The previous review identified that the lag logic was hardcoded to switch between 7 and 14 days, which causes data leakage for lead times > 14 days. The implementation plan required `weeks_ahead = ceil(lead_time / 7_days)` and `lag = weeks_ahead * 7_days`. The current code adds a comment `# lag = ceil(lead_time / 7 days) * 7 days` but the actual code below it is completely unchanged and still uses the hardcoded `pl.when(pl.col("lead_time_days") <= 7).then(...).otherwise(...)`.
* **Concrete Failure Mode:** If the forecast horizon is extended beyond 14 days, the model will leak future data during training and backtesting.
* **Required Architectural Fix:** The Architect must actually implement the dynamic lag logic using Polars expressions (e.g., dynamically generating the required lag or using a join strategy that calculates the exact lag needed), rather than just adding a comment.

## FLAW-002: [Critical] Feature Engineering Broken by NWP Prefixing
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 200-210, and `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 45-50 and 80-85
* **The Theoretical Issue:** The new multi-NWP join strategy correctly prefixes NWP columns with the model name (e.g., `ECMWF_temperature_2m`). However, `add_weather_features` and `add_physical_features` hardcode the original column names (e.g., `"temperature_2m"`). Because the unprefixed columns no longer exist in the joined dataframe, these functions silently return without adding any features or descaling the data.
* **Concrete Failure Mode:** The model is completely missing critical features like `windchill`, `temp_trend_6h`, and lagged weather features. Furthermore, the NWP data remains in `uint8` format instead of physical units.
* **Required Architectural Fix:** The Architect must update `add_weather_features` and `add_physical_features` to dynamically detect prefixed columns, or apply the feature engineering *before* the columns are prefixed and joined.

## FLAW-003: [Critical] Invalid Lead Time Calculation for Backtesting Metrics
* **File & Line Number:** `packages/ml_core/src/ml_core/utils.py`, lines 150-160, and `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 350-360
* **The Theoretical Issue:** In `evaluate_and_save_model`, `inference_params.nwp_init_time` is set to the *maximum* `init_time` in the test set (e.g., the end of the month). The `predict` method then hardcodes this single datetime into the `nwp_init_time` column for all predictions. Finally, `evaluate_and_save_model` calculates `lead_time_hours = valid_time - nwp_init_time`.
* **Concrete Failure Mode:** For a month-long backtest, `lead_time_hours` will be negative for almost all predictions (e.g., `Jan 1st - Jan 31st = -30 days`). The resulting metrics grouped by lead time will be complete garbage and scientifically invalid.
* **Required Architectural Fix:** The `predict` method must retain the actual `init_time` of the primary NWP used for each row (which varies across the backtest period) and output that as `nwp_init_time`, rather than hardcoding the scalar `inference_params.nwp_init_time`.

## FLAW-004: [Standard] Missing NWP Deaccumulation Comments
* **File & Line Number:** `packages/contracts/src/contracts/data_schemas.py` and `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
* **The Theoretical Issue:** The previous review required adding comments to explain that accumulated variables (e.g., precipitation, radiation) are de-accumulated by Dynamical.org before download. These comments were not added.
* **Concrete Failure Mode:** Future developers or agents may attempt to "fix" the accumulation by applying `.diff()`, which would corrupt the data.
* **Required Architectural Fix:** Add the required comments to the NWP data contract and the data loading functions.
