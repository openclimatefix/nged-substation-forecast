- In `xgboost_baseline.yaml`:
    - Should we rename `ml_model_name` to `power_fcst_model_name` to be consistent with the
    `PowerForecast` contract?
    - Should we set a default `data_split` somewhere more general than `xgboost_baseline.yaml`?
      I imagine that we'll use the same splits for most ML models.
- In `ml_core/assets.py`:
    - Rename `SUBSTATION_SCADA` to `SUBSTATION_POWER_FLOWS`
    - Add `WEATHER_ECMWF_ENS_0_25` (for 0.25 degree ensemble forecast)
- In `ml_core/model.py`:
    - Should we rename `BasePolarsModel` to `ForecastInference`?
    - Can `predict` return a `pt.DataFrame[PowerForecast]`?
- In `ml_core/scaling.py`:
    - Can `params` be given a type hint of a specific Patito data contract?

