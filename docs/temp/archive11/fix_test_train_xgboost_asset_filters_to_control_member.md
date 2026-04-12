---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["tests"]
---
# Implementation Plan: Fix `test_train_xgboost_asset_filters_to_control_member`

## Context
The test `test_train_xgboost_asset_filters_to_control_member` in `tests/test_xgboost_forecaster.py` is failing with a `FileNotFoundError`. The `train_xgboost` asset reads `time_series_metadata.parquet` from the `nged_data_path` (which is mocked to `tmp_path` in the test), but this file is never created by the test setup. Additionally, the `time_series_id` in the dummy `flows` dataframe is defined as a string (`"123"`), which causes a `polars.exceptions.InvalidOperationError` when filtering the metadata (which expects an `Int32` `time_series_id`).

## Proposed Changes

1. **Update `flows` DataFrame in the Test:**
   - Change the `time_series_id` value from `["123"]` to `[123]`.
   - Add a cast to `pl.Int32` for the `time_series_id` column to match the expected schema.

2. **Create Dummy `time_series_metadata.parquet`:**
   - Create a dummy Polars DataFrame representing the time series metadata. It must include at least `time_series_id` (cast to `pl.Int32`), `substation_number`, and `h3_res_5`.
   - Create the `parquet` directory within `tmp_path`.
   - Write the dummy metadata DataFrame to `tmp_path / "parquet" / "time_series_metadata.parquet"`.

## Implementation Details

Modify `tests/test_xgboost_forecaster.py`:

```python
def test_train_xgboost_asset_filters_to_control_member(tmp_path):
    # ... existing nwp setup ...

    flows = pl.DataFrame(
        {
            "time_series_id": [123], # Changed from ["123"] to [123]
            "start_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "period_end_time": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30)],
            "power": [1.0],
            "MVA": [None],
        }
    ).with_columns(
        pl.col("MVA").cast(pl.Float64),
        pl.col("time_series_id").cast(pl.Int32) # Added cast to Int32
    )

    # Write flows to Delta as the asset now loads from Delta
    delta_dir = tmp_path / "delta"
    cleaned_actuals_path = delta_dir / "cleaned_actuals"
    cleaned_actuals_path.mkdir(parents=True)
    flows.write_delta(str(cleaned_actuals_path))

    # NEW: Create time series metadata
    metadata = pl.DataFrame(
        {
            "time_series_id": [123],
            "substation_number": [1],
            "h3_res_5": [1],
        }
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))
    
    parquet_dir = tmp_path / "parquet"
    parquet_dir.mkdir(parents=True)
    metadata.write_parquet(parquet_dir / "time_series_metadata.parquet")

    settings = Settings(nged_data_path=tmp_path)
    # ... rest of the test ...
```

## Code Comments
- Ensure that any new code added to the test is clear and concise.
- No FLAW-XXX IDs should be referenced in the code comments.
