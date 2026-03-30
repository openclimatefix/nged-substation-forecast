---
review_iteration: 2
reviewer: "scientist"
total_flaws: 4
critical_flaws: 2
---

# ML Rigor Review

## FLAW-001: [Critical] Data Leakage in Dynamic Seasonal Lag for Lead Times > 14 Days
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 140-150 and 280-290
* **The Theoretical Issue:** The implementation plan required a dynamic lag calculation: `weeks_ahead = ceil(lead_time / 7_days)` and `lag = weeks_ahead * 7_days`. The actual implementation just switches between a 7-day and 14-day lag based on a 168-hour threshold. For lead times > 14 days (e.g., 15 days), the 14-day lag references a timestamp that is 1 day in the *future* relative to `init_time`.
* **Concrete Failure Mode:** The model will perfectly predict the target for days 14-15 during backtesting because it is accidentally being fed the actual future power flows. In production, this data will not exist, causing catastrophic failure for week-2 forecasts.
* **Required Architectural Fix:** The Architect must implement the dynamic lag logic as originally specified: `weeks_ahead = ceil(lead_time / 7_days)` and use `lag = weeks_ahead * 7_days`. The hardcoded 14-day lag is insufficient for 15-day ECMWF forecasts.

## FLAW-002: [Critical] Lookahead Bias in Multi-NWP Join Strategy
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 205-215
* **The Theoretical Issue:** The implementation plan required defining an `available_time = init_time + publication_delay` to prevent lookahead bias. The actual implementation uses `lf.filter(pl.col("init_time") <= target_init_time)`. This ignores publication delay. A secondary NWP initialized at `target_init_time` will be included even if it wouldn't be available in production until hours later.
* **Concrete Failure Mode:** The model will learn to rely on secondary NWP data that won't actually be available in the production environment at the time the forecast is generated.
* **Required Architectural Fix:** The Architect must add `available_time` to the NWP schemas and use it for filtering/joining secondary NWPs, ensuring that `secondary_NWP.available_time <= primary_NWP.init_time`.

## FLAW-003: [Standard] Missing NWP Deaccumulation Comments
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py` and `packages/contracts/src/contracts/data_schemas.py`
* **The Theoretical Issue:** The implementation plan required adding comments to explain that accumulated variables (e.g., precipitation, radiation) are de-accumulated by Dynamical.org before download. These comments were not added.
* **Concrete Failure Mode:** Future developers or agents may attempt to "fix" the accumulation by applying `.diff()`, which would corrupt the data.
* **Required Architectural Fix:** Add the required comments to the NWP data contract and the data loading functions.

## FLAW-004: [Standard] Missing Backtesting Metrics per Lead Time
* **File & Line Number:** `packages/ml_core/src/ml_core/utils.py`, `evaluate_and_save_model`
* **The Theoretical Issue:** The implementation plan required calculating metrics (MAE, RMSE, MAPE) grouped by `lead_time` in `evaluate_and_save_model`. The actual implementation just returns the raw predictions `results_df` without calculating any metrics.
* **Concrete Failure Mode:** We cannot assess how the model's performance degrades over the forecast horizon, which is critical for operational decision-making.
* **Required Architectural Fix:** Implement the metric calculation grouped by `lead_time` as specified in the implementation plan.
