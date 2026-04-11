# Session Handoff: XGBoost RAM Usage Investigation

## Current State
- We have identified that the massive RAM spike occurs **immediately after** the XGBoost inference step finishes.
- We have added extensive logging to `packages/ml_core/src/ml_core/utils.py` to pinpoint the exact post-processing operation causing the spike.
- The logs confirm the spike happens right after `XGBoost inference finished.` is printed.

## Achieved
- Added logging to `tests/test_xgboost_dagster_integration.py` to track step start/end times.
- Confirmed that `CompositeIOManager` is not pickling large DataFrames (it uses Parquet).
- Verified that training and inference are correctly scoped to 4 substations and the control NWP member (training) / all 51 members (inference).
- Confirmed inference is restricted to a single NWP run.
- Configured Dagster to output logs to `stdout` to allow debugging during `pytest` runs.
- Added detailed logging to the post-processing steps in `evaluate_and_save_model` to isolate the memory-intensive operation.

## Next Steps
- Investigate if the `results_df` (predictions) or `actuals` (cleaned power time series) are being collected into memory in a way that causes the spike during the join or aggregation.
- Consider using `polars` lazy evaluation more effectively to avoid eager collection of large DataFrames during post-processing.
- Analyze the logs from the next test run to identify the specific post-processing operation (e.g., join, aggregation) that triggers the OOM. Ask the user to run the test whilst watching RAM usage.
