---
status: "draft"
version: "v1.0"
task_type: "standard"
requires_ml_review: false
requires_data_engineer: false
target_modules: ["packages/ml_core/src/ml_core/utils.py"]
---
# Plan: Fix OOM in `evaluate_and_save_model`

## Rationale
The `evaluate_and_save_model` function in `packages/ml_core/src/ml_core/utils.py` currently suffers from multiple redundant evaluations of Polars LazyFrames, which can lead to Out-Of-Memory (OOM) errors and excessive computation time when dealing with large datasets.

Specifically:
1. **Redundant `collect()` calls:** `results_lf.collect()` is called twice at the end of the function (once to get the height for metadata, and once to return the result). This can duplicate memory usage if the underlying DataFrame is large.
2. **Redundant Join Executions:** The `eval_lf` LazyFrame joins the predictions (`results_lf`) with the actuals (`actuals_lf`, which is a lazy scan of a Parquet file). This `eval_lf` is then evaluated **three times**:
   - `metrics_lf.collect()`
   - `global_metrics_lf.collect()`
   - `eval_lf.sink_parquet()`
   This means the Parquet file is scanned and the join is performed three times.

## Implementation Steps

1. **Modify `packages/ml_core/src/ml_core/utils.py`:**
   - **Cache `eval_lf` to disk first:** Move the `eval_lf.sink_parquet()` step to happen *before* calculating the metrics.
   - **Read back lazily:** After sinking to Parquet, use `pl.scan_parquet()` to read the saved file back into a new LazyFrame (`eval_lf_cached`).
   - **Compute metrics from cache:** Update `metrics_lf` and `global_metrics_lf` to use `eval_lf_cached` instead of `eval_lf`. This ensures the expensive join is only executed once.
   - **Single `collect()` for results:** At the end of the function, collect `results_lf` exactly once into a variable `results_df`, use `results_df.height` for the metadata, and return `results_df`.

## Code Changes

```python
# In packages/ml_core/src/ml_core/utils.py

# ... inside evaluate_and_save_model ...

        # Calculate lead_time_hours
        if "nwp_init_time" in eval_lf.collect_schema().names():
            context.log.info("Calculating lead_time_hours (with nwp_init_time).")
            eval_lf = eval_lf.with_columns(
                lead_time_hours=(pl.col("valid_time") - pl.col("nwp_init_time")).dt.total_minutes()
                / 60.0
            )
        else:
            # Fallback if nwp_init_time is not available
            context.log.info("Calculating lead_time_hours (fallback).")
            eval_lf = eval_lf.with_columns(
                lead_time_hours=(pl.col("valid_time") - pl.lit(forecast_time)).dt.total_minutes()
                / 60.0
            )

        # Log metrics to MLflow
        mlflow.set_experiment(model_name)
        with mlflow.start_run(run_name=f"{model_name}_eval"):
            # 1. Save full granular evaluation dataframe as Parquet artifact FIRST
            # This executes the join ONCE and streams the result to disk.
            with tempfile.TemporaryDirectory() as tmpdir:
                eval_df_path = os.path.join(tmpdir, "evaluation_granular.parquet")
                eval_lf.sink_parquet(eval_df_path)
                mlflow.log_artifact(eval_df_path)

                # Now read it back lazily to compute metrics without re-executing the join
                eval_lf_cached = pl.scan_parquet(eval_df_path)

                # Group by lead_time_hours and calculate metrics
                context.log.info("Calculating metrics by lead_time_hours.")
                metrics_lf = (
                    eval_lf_cached.group_by("lead_time_hours")
                    .agg(
                        [
                            (pl.col("power_fcst") - pl.col("actual")).abs().mean().alias("MAE"),
                            ((pl.col("power_fcst") - pl.col("actual")) ** 2).mean().sqrt().alias("RMSE"),
                            ((pl.col("power_fcst") - pl.col("actual")).abs() / pl.col("peak_capacity"))
                            .mean()
                            .alias("nMAE"),
                        ]
                    )
                    .sort("lead_time_hours")
                )

                # 2. Metric Thinning: Log only key operational horizons
                key_horizons = {24.0, 48.0, 72.0, 168.0, 336.0}
                metrics = metrics_lf.collect()
                for row in metrics.iter_rows(named=True):
                    lt = round(float(row["lead_time_hours"]), 1)
                    if lt in key_horizons:
                        if row["MAE"] is not None:
                            mlflow.log_metric(f"MAE_LT_{lt}h", row["MAE"])
                        if row["RMSE"] is not None:
                            mlflow.log_metric(f"RMSE_LT_{lt}h", row["RMSE"])
                        if row["nMAE"] is not None:
                            mlflow.log_metric(f"nMAE_LT_{lt}h", row["nMAE"])

                # 3. Log global aggregate metrics
                global_metrics_lf = eval_lf_cached.select(
                    (pl.col("power_fcst") - pl.col("actual")).abs().mean().alias("mae_global"),
                    ((pl.col("power_fcst") - pl.col("actual")) ** 2).mean().sqrt().alias("rmse_global"),
                    ((pl.col("power_fcst") - pl.col("actual")).abs() / pl.col("peak_capacity"))
                    .mean()
                    .alias("nmae_global"),
                )
                global_metrics = global_metrics_lf.collect().to_dicts()[0]

                if global_metrics["mae_global"] is not None:
                    mlflow.log_metric("mean_mae_all_horizons", global_metrics["mae_global"])
                    mlflow.log_metric("MAE_global", global_metrics["mae_global"])

                if global_metrics["rmse_global"] is not None:
                    mlflow.log_metric("RMSE_global", global_metrics["rmse_global"])

                if global_metrics["nmae_global"] is not None:
                    mlflow.log_metric("nMAE_global", global_metrics["nmae_global"])

    # 4. Trigger Dynamic Partition
    if hasattr(context, "instance") and context.instance:
        context.instance.add_dynamic_partitions("model_partitions", [model_name])

    # Collect results ONCE to avoid redundant memory allocation
    results_df = results_lf.collect()

    if hasattr(context, "add_output_metadata"):
        context.add_output_metadata(
            {
                "num_rows": results_df.height,
                "power_fcst_model_name": model_name,
            }
        )

    return results_df
```

## Review Responses & Rejections
None yet.
