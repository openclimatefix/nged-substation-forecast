---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/ml_core/src/ml_core/utils.py", "packages/ml_core/src/ml_core/data.py"]
---
# Memory Optimization Plan: Lazy Evaluation in `evaluate_and_save_model`

## 1. Identification of Memory Collection Points

In `packages/ml_core/src/ml_core/utils.py` within the `evaluate_and_save_model` function, data is collected into memory at the following points:

*   **`results_df`**: This is collected and returned as an in-memory Polars `DataFrame` by the `forecaster.predict()` method (e.g., in `XGBoostForecaster.predict()`, the features are collected to pass to the XGBoost model, and the resulting predictions are constructed as an in-memory DataFrame).
*   **`actuals`**: This is explicitly collected into memory at line 178:
    ```python
    actuals = cast(
        pl.DataFrame,
        sliced_data["power_time_series"].collect(),
    )
    ```
*   **`peak_capacity_df`**: The `actuals` dataframe is passed to `calculate_peak_capacity()`, which internally calls `.collect()` and returns an in-memory `DataFrame` (located in `packages/ml_core/src/ml_core/data.py`).
*   **`eval_df`**: Because `results_df`, `actuals`, and `peak_capacity_df` are all in-memory DataFrames, all subsequent joins, filters, and transformations result in a fully materialized `eval_df` in memory. This is highly inefficient for large datasets.

## 2. Proposed Plan for Lazy Evaluation

To drastically reduce peak memory usage, we should keep the evaluation pipeline lazy for as long as possible, only materializing the final aggregated metrics and streaming the granular results directly to disk.

### Step 1: Make `calculate_peak_capacity` Lazy
Modify `calculate_peak_capacity` in `packages/ml_core/src/ml_core/data.py` to return a `pt.LazyFrame` instead of calling `.collect()`. If the collected version is strictly required elsewhere, we should introduce a `calculate_peak_capacity_lazy` variant.

### Step 2: Keep `actuals` Lazy
In `evaluate_and_save_model`, remove the `.collect()` call when assigning `actuals`.
```python
actuals = sliced_data["power_time_series"] # Keep as LazyFrame
```

### Step 3: Convert `results_df` to a LazyFrame
Although `results_df` is returned as an in-memory `DataFrame` from `predict()`, we should immediately convert it to a `LazyFrame` to ensure all subsequent joins and transformations are part of a lazy computation graph.
```python
results_lf = results_df.lazy()
```

### Step 4: Perform Joins and Transformations Lazily
Construct `eval_lf` by performing all joins (`results_lf` with `actuals`, and then with `peak_capacity_lf`), filters, and column additions (like `lead_time_hours`) lazily.

### Step 5: Compute Metrics Lazily
Define the metrics aggregations (grouped by `lead_time_hours` and global metrics) as lazy queries. Collect only these small aggregated dataframes.
```python
metrics = eval_lf.group_by("lead_time_hours").agg(...).sort("lead_time_hours").collect()
```

### Step 6: Stream Granular Results to Disk
Instead of collecting the potentially massive `eval_lf` into memory to write it to a Parquet file, use Polars' `sink_parquet()` method. This streams the data through the lazy computation graph and writes it directly to disk in chunks.
```python
eval_lf.sink_parquet(eval_df_path)
```

### Step 7: Metadata Handling
Since `results_df` is already in memory from the `predict()` call, we can safely continue using `len(results_df)` for the `num_rows` output metadata.

## Review Responses & Rejections
*(None yet)*
