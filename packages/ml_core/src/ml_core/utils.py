import logging
import os
import tempfile
from datetime import date, datetime, timedelta, timezone
from typing import Any, Union, cast

import dagster as dg
import mlflow
import polars as pl
from contracts.data_schemas import InferenceParams
from contracts.hydra_schemas import TrainingConfig
from ml_core.data import calculate_peak_capacity

log = logging.getLogger(__name__)


def _slice_temporal_data(data: Any, start: date | str, end: date | str, time_col: str) -> Any:
    """Recursively slice temporal data (LazyFrames, DataFrames, or dicts thereof)."""
    if isinstance(data, dict):
        return {k: _slice_temporal_data(v, start, end, time_col) for k, v in data.items()}

    if isinstance(data, (pl.LazyFrame, pl.DataFrame)):
        cols = data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.columns
        if time_col not in cols:
            return data

        # If end is a date (but not a datetime), we want to include the entire day.
        # Polars' is_between is inclusive, but if the column is datetime and end is date,
        # it might only include up to 00:00:00 of that date.
        if isinstance(end, date) and not isinstance(end, datetime):
            end_filter = end + timedelta(days=1)
            return data.filter((pl.col(time_col) >= start) & (pl.col(time_col) < end_filter))
        else:
            return data.filter(pl.col(time_col).is_between(start, end))

    return data


def train_and_log_model(
    context: Union[dg.AssetExecutionContext, dg.OpExecutionContext],
    model_name: str,
    trainer,
    config: TrainingConfig,
    **kwargs,
):
    """Universal utility to handle temporal slicing and MLflow logging for training.

    Args:
        context: Dagster execution context (Asset or Op).
        model_name: Name of the model (for MLflow run name).
        trainer: An object with a `train(config, **kwargs)` method.
        config: Training configuration object.
        **kwargs: Input LazyFrames to be temporally sliced.

    Returns:
        The trained model object.
    """
    # 1. Universal Temporal Slicing
    train_start = config.data_split.train_start
    train_end = config.data_split.train_end

    sliced_data = {}
    for key, val in kwargs.items():
        if key == "time_series_metadata":
            sliced_data["time_series_metadata"] = val
            continue

        time_col = "period_end_time" if "power_flows" in key else "valid_time"

        # Add a configurable lookback for autoregressive features
        slice_start = train_start
        if "power_flows" in key or "nwps" in key:
            lookback = getattr(config.model, "required_lookback_days", 14)
            slice_start = train_start - timedelta(days=lookback)

        sliced_data[key] = _slice_temporal_data(val, slice_start, train_end, time_col)

        # 2. Call the Model-Specific Math
        # The trainer is responsible for joining and feature engineering.
        # We ensure power flows are at 30m resolution for consistency across models.
        if "substation_power_flows" in sliced_data:
            flows = sliced_data.pop("substation_power_flows")

            power_time_series = flows
            sliced_data["power_time_series"] = power_time_series

    model = trainer.fit(config=config.model, **sliced_data)

    # 3. Universal MLflow Logging
    mlflow.set_experiment(model_name)
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(config.model_dump(mode="json"))
        trainer.log_model(model_name)
        if hasattr(context, "add_output_metadata"):
            context.add_output_metadata({"mlflow_run_id": run.info.run_id})

    return model


def evaluate_and_save_model(
    context: Union[dg.AssetExecutionContext, dg.OpExecutionContext],
    model_name: str,
    forecaster,
    config: TrainingConfig,
    **kwargs,
):
    """Universal utility to handle temporal slicing, inference, and storage.

    Args:
        context: Dagster execution context (Asset or Op).
        model_name: Name of the model.
        forecaster: An object with a `predict(**kwargs)` method.
        config: Training configuration object.
        **kwargs: Input LazyFrames to be temporally sliced and collected.

    Returns:
        A Polars DataFrame containing the predictions.
    """
    # 1. Universal Temporal Slicing for Test Set
    test_start = config.data_split.test_start
    test_end = config.data_split.test_end

    sliced_data = {}
    for key, val in kwargs.items():
        if key == "time_series_metadata":
            sliced_data[key] = val
            continue

        time_col = "period_end_time" if "power_flows" in key else "valid_time"

        # Add a configurable lookback for autoregressive features
        slice_start = test_start
        if "power_flows" in key or "nwps" in key:
            lookback = getattr(config.model, "required_lookback_days", 14)
            slice_start = test_start - timedelta(days=lookback)

        sliced_data[key] = _slice_temporal_data(val, slice_start, test_end, time_col)

    # 2. Call the Model-Specific Inference
    # Extract the actual init_time from the provided nwps data
    forecast_time = datetime.now(timezone.utc)
    if "nwps" in sliced_data:
        nwps_data = sliced_data["nwps"]
        first_nwp = None
        if isinstance(nwps_data, dict) and nwps_data:
            first_nwp = next(iter(nwps_data.values()))
        elif isinstance(nwps_data, list) and nwps_data:
            first_nwp = nwps_data[0]
        elif isinstance(nwps_data, pl.LazyFrame):
            first_nwp = nwps_data

        if isinstance(first_nwp, pl.LazyFrame):
            df = cast(pl.DataFrame, first_nwp.select(pl.col("init_time").max()).collect())
            if not df.is_empty() and df.item() is not None:
                delay_hours = getattr(config.model, "nwp_availability_delay_hours", 3)
                forecast_time = df.item() + timedelta(hours=delay_hours)

    inference_params = InferenceParams(
        forecast_time=forecast_time,
        power_fcst_model_name=model_name,
    )

    # Downsample power flows to 30m for inference (lags)
    if "substation_power_flows" in sliced_data:
        flows = sliced_data.pop("substation_power_flows")

        power_time_series = flows
        sliced_data["power_time_series"] = power_time_series

    results_lf = forecaster.predict(
        inference_params=inference_params, collapse_lead_times=False, **sliced_data
    ).lazy()
    results_df = results_lf.collect()
    context.log.info("XGBoost inference finished.")

    # 3. Calculate Metrics per lead_time
    if "power_time_series" in sliced_data:
        actuals_lf = sliced_data["power_time_series"]

        # Join predictions with actuals
        context.log.info("Joining predictions with actuals.")
        eval_lf = results_df.lazy().join(
            actuals_lf.rename({"period_end_time": "valid_time", "power": "actual"}),
            on=["valid_time", "time_series_id"],
            how="inner",
        )

        # Join the peak_capacity from the actuals
        context.log.info("Calculating peak capacity.")
        peak_capacity_lf = calculate_peak_capacity(actuals_lf)
        eval_lf = eval_lf.join(
            peak_capacity_lf.select(["time_series_id", "peak_capacity"]),
            on="time_series_id",
            how="left",
        )
        # Fill missing peak_capacity with 1.0 to avoid division by zero
        eval_lf = eval_lf.with_columns(pl.col("peak_capacity").fill_null(1.0))

        # Filter out the lookback period to avoid data leakage in evaluation
        if isinstance(test_start, date) and not isinstance(test_start, datetime):
            test_start_dt = datetime.combine(test_start, datetime.min.time()).replace(
                tzinfo=timezone.utc
            )
        else:
            test_start_dt = test_start

        eval_lf = eval_lf.filter(pl.col("valid_time") >= test_start_dt)

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

        # Sink to temporary file for caching
        tmp_file = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        eval_parquet_path = tmp_file.name
        tmp_file.close()

        try:
            eval_lf.sink_parquet(eval_parquet_path)
            eval_lf_cached = pl.scan_parquet(eval_parquet_path)

            # Group by lead_time_hours and calculate metrics
            context.log.info("Calculating metrics by lead_time_hours.")
            metrics_lf = (
                eval_lf_cached.group_by("lead_time_hours")
                .agg(
                    [
                        (pl.col("power_fcst") - pl.col("actual")).abs().mean().alias("MAE"),
                        ((pl.col("power_fcst") - pl.col("actual")) ** 2)
                        .mean()
                        .sqrt()
                        .alias("RMSE"),
                        ((pl.col("power_fcst") - pl.col("actual")).abs() / pl.col("peak_capacity"))
                        .mean()
                        .alias("nMAE"),
                    ]
                )
                .sort("lead_time_hours")
            )

            # Log metrics to MLflow
            mlflow.set_experiment(model_name)
            with mlflow.start_run(run_name=f"{model_name}_eval"):
                # 1. Metric Thinning: Log only key operational horizons
                key_horizons = {24.0, 48.0, 72.0, 168.0, 336.0}
                metrics = cast(pl.DataFrame, metrics_lf.collect())
                for row in metrics.iter_rows(named=True):
                    lt = round(float(row["lead_time_hours"]), 1)
                    if lt in key_horizons:
                        if row["MAE"] is not None:
                            mlflow.log_metric(f"MAE_LT_{lt}h", row["MAE"])
                        if row["RMSE"] is not None:
                            mlflow.log_metric(f"RMSE_LT_{lt}h", row["RMSE"])
                        if row["nMAE"] is not None:
                            mlflow.log_metric(f"nMAE_LT_{lt}h", row["nMAE"])

                # 2. Log global aggregate metrics
                global_metrics_lf = eval_lf_cached.select(
                    (pl.col("power_fcst") - pl.col("actual")).abs().mean().alias("mae_global"),
                    ((pl.col("power_fcst") - pl.col("actual")) ** 2)
                    .mean()
                    .sqrt()
                    .alias("rmse_global"),
                    ((pl.col("power_fcst") - pl.col("actual")).abs() / pl.col("peak_capacity"))
                    .mean()
                    .alias("nmae_global"),
                )
                global_metrics = cast(pl.DataFrame, global_metrics_lf.collect()).to_dicts()[0]

                if global_metrics["mae_global"] is not None:
                    mlflow.log_metric("mean_mae_all_horizons", global_metrics["mae_global"])
                    mlflow.log_metric("MAE_global", global_metrics["mae_global"])

                if global_metrics["rmse_global"] is not None:
                    mlflow.log_metric("RMSE_global", global_metrics["rmse_global"])

                if global_metrics["nmae_global"] is not None:
                    mlflow.log_metric("nMAE_global", global_metrics["nmae_global"])

                # 3. Save full granular evaluation dataframe as Parquet artifact
                mlflow.log_artifact(eval_parquet_path)
        finally:
            if os.path.exists(eval_parquet_path):
                os.remove(eval_parquet_path)

    # 4. Trigger Dynamic Partition
    if hasattr(context, "instance") and context.instance:
        context.instance.add_dynamic_partitions("model_partitions", [model_name])

    if hasattr(context, "add_output_metadata"):
        context.add_output_metadata(
            {
                "num_rows": results_lf.collect().height,
                "power_fcst_model_name": model_name,
            }
        )

    return results_lf.collect()
