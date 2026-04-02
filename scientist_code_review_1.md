---
review_iteration: 1
reviewer: "scientist"
total_flaws: 1
critical_flaws: 1
---

# ML Rigor Review

## FLAW-001: Physical Data Leakage / Meaningless Features due to Missing Descaling
* **File & Line Number:** `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`, `process_nwp_data` function (around line 200) and `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`, `add_weather_features` function (around line 80).
* **The Theoretical Issue:** The NWP data is scaled to a `UInt8` (0-255) range before being saved to disk to save space. However, the Dagster pipeline (`processed_nwp_data` asset) never descales this data back to physical units before passing it to the model. While tree-based models are invariant to monotonic transformations of individual features, the `windchill` feature is calculated using a non-linear physical formula that mixes the 0-255 scaled `temperature_2m` with the unscaled `wind_speed_10m` (which was excluded from scaling).
* **Concrete Failure Mode:** The `windchill` feature will be completely physically meaningless and mathematically corrupted, leading to degraded model performance and incorrect feature importance.
* **Required Architectural Fix:** The Architect must ensure that `process_nwp_data` (or the `processed_nwp_data` asset) calls `uint8_to_physical_unit` to descale the weather variables back to their physical units *before* any feature engineering (like `windchill` or `temp_trend_6h`) is performed.
