---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/ml_core/src/ml_core/utils.py"]
---
# Implementation Plan: Extend Forecast Horizon in Evaluation

## Objective
Modify `evaluate_and_save_model` to allow the forecast horizon to extend beyond `test_end` (e.g., by 14 days) for inference, while keeping the evaluation metrics restricted to the `[test_start, test_end]` period.

## Context
Currently, all input data in `evaluate_and_save_model` is sliced to `test_end`. This limits the generated forecast to the test period (which is 1 day in integration tests). By extending the NWP slice to `test_end + 14 days`, the model can generate a 14-day forecast. The evaluation will naturally remain restricted to `test_end` because the actuals (`power_time_series`) will still be sliced to `test_end`, and the evaluation uses an `inner` join between predictions and actuals.

## Proposed Changes

### 1. Fix `time_col` determination for `power_time_series`
In both `evaluate_and_save_model` and `train_and_log_model`, the `time_col` logic currently checks for `"power_flows" in key`. This fails to match the new `"power_time_series"` key, causing actuals to not be sliced at all.
- **Change:** Update the condition to check for `"power" in key` (or explicitly `key in ["power_flows", "power_time_series"]`).
- **Why:** Ensures `power_time_series` is correctly identified and sliced using `period_end_time`.

### 2. Extend `slice_end` for NWPs in `evaluate_and_save_model`
Modify the temporal slicing loop in `evaluate_and_save_model` to extend the end date for NWP data.
- **Change:** Introduce a `slice_end` variable that defaults to `test_end`.
- **Change:** If `"nwps" in key`, set `slice_end = test_end + timedelta(days=getattr(config.model, "forecast_horizon_days", 14))`.
- **Change:** Pass `slice_end` to `_slice_temporal_data` instead of `test_end`.
- **Why:** This provides the model with future weather data, allowing it to generate predictions up to 14 days beyond `test_end`.

### 3. Verify Evaluation Restriction
No changes are needed to the evaluation logic itself.
- **Why:** Because `power_time_series` (actuals) will now be correctly sliced to `test_end`, the `inner` join between `results_lf` (which extends to `test_end + 14 days`) and `actuals_lf` will automatically drop the future predictions from the evaluation dataset (`eval_lf`). The returned dataframe will still contain the full 14-day forecast.

## Code Snippet (for `evaluate_and_save_model`)
```python
    # 1. Universal Temporal Slicing for Test Set
    test_start = config.data_split.test_start
    test_end = config.data_split.test_end

    sliced_data = {}
    for key, val in kwargs.items():
        if key == "time_series_metadata":
            sliced_data[key] = val
            continue

        # Fix time_col to handle power_time_series
        time_col = "period_end_time" if "power" in key else "valid_time"

        # Add a configurable lookback for autoregressive features
        slice_start = test_start
        if "power" in key or "nwps" in key:
            lookback = getattr(config.model, "required_lookback_days", 14)
            slice_start = test_start - timedelta(days=lookback)

        # Extend slice_end for NWPs to allow forecasting beyond test_end
        slice_end = test_end
        if "nwps" in key:
            forecast_horizon = getattr(config.model, "forecast_horizon_days", 14)
            slice_end = test_end + timedelta(days=forecast_horizon)

        sliced_data[key] = _slice_temporal_data(val, slice_start, slice_end, time_col)
```

## Code Snippet (for `train_and_log_model`)
```python
        # Fix time_col to handle power_time_series
        time_col = "period_end_time" if "power" in key else "valid_time"

        # Add a configurable lookback for autoregressive features
        slice_start = train_start
        if "power" in key or "nwps" in key:
            lookback = getattr(config.model, "required_lookback_days", 14)
            slice_start = train_start - timedelta(days=lookback)
```

## Coding Standards & Mandates
- **Comments:** You must add explicit code comments explaining *why* `slice_end` is extended for NWPs (to allow forecasting into the future) and *why* the evaluation remains restricted (because the inner join with actuals naturally drops future predictions). Do not just describe *what* the code is doing.
- **No FLAW IDs:** You are strictly forbidden from referencing any FLAW-XXX IDs in code comments.
