---
review_iteration: 1
reviewer: "scientist"
total_flaws: 3
critical_flaws: 2
---

# ML Rigor Review

## FLAW-001: Temporal Data Leakage in Weather Lags
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, lines 80-123
* **The Theoretical Issue:** The `add_weather_features` function creates weather lags (e.g., `weather_lag_7d`) by grouping the NWP dataset by `valid_time` and taking the *last* `init_time` (the most recent forecast). It then shifts the `valid_time` by 7 days and joins it back to the main dataset. For a forecast with a long lead time (e.g., predicting Day 10 from Day 1), it joins the weather lag for Day 3. However, the lag value comes from the *latest* forecast for Day 3 (likely initialized on Day 3), which is in the future relative to the `init_time` of Day 1.
* **Concrete Failure Mode:** The model will learn to rely on highly accurate "lagged" weather features that actually contain future knowledge. This will lead to artificially low error during backtesting but poor performance in production where future forecasts are unavailable.
* **Required Architectural Fix:** Weather lags must be calculated dynamically relative to the `init_time` of the row being predicted, similar to how `latest_available_weekly_lag` is calculated for power flows. Alternatively, the join must include `init_time` such that `init_time_history <= init_time_current`.

## FLAW-002: Substation Number Treated as Continuous Feature
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, lines 211-215 and 71-76
* **The Theoretical Issue:** `substation_number` is cast to `pl.Int32` and included in the feature matrix because it is numeric. XGBoost will treat this categorical identifier as a continuous numerical feature, assuming that substation 10 is mathematically "greater than" substation 5.
* **Concrete Failure Mode:** The model will make nonsensical splits on the `substation_number` (e.g., `substation_number < 42.5`), grouping unrelated substations together and failing to learn substation-specific behaviors accurately.
* **Required Architectural Fix:** Cast `substation_number` to `pl.Categorical` and configure XGBoost with `enable_categorical=True`, or explicitly exclude it from the feature matrix and rely on target encoding / static metadata features instead.

## FLAW-003: Loss of Substation-Specific Static Features in Global Model
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`, line 144
* **The Theoretical Issue:** The shift to a global model is architecturally sound, but the current implementation discards all substation metadata except `substation_number` and `h3_res_5` during the join.
* **Concrete Failure Mode:** The global model lacks the necessary context (e.g., `substation_type`, `latitude`, `longitude`) to differentiate between substations, leading to underfitting compared to the previous local models which implicitly learned these differences.
* **Required Architectural Fix:** Include relevant static features from `substation_metadata` (like `substation_type`, `latitude`, `longitude`) in the feature matrix when training the global model.
