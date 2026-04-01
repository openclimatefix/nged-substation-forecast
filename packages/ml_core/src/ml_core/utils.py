import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Union, cast

import dagster as dg
import mlflow
import polars as pl
from contracts.data_schemas import InferenceParams
from contracts.hydra_schemas import TrainingConfig
from ml_core.data import downsample_power_flows

log = logging.getLogger(__name__)


def _slice_temporal_data(data: Any, start: date | str, end: date | str, time_col: str) -> Any:
    """Recursively slice temporal data (LazyFrames, DataFrames, or dicts thereof)."""
    if isinstance(data, dict):
        return {k: _slice_temporal_data(v, start, end, time_col) for k, v in data.items()}

    if isinstance(data, (pl.LazyFrame, pl.DataFrame)):
        cols = data.collect_schema().names() if isinstance(data, pl.LazyFrame) else data.columns
        if time_col not in cols:
            return data

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
        if key == "substation_metadata":
            sliced_data[key] = val
            continue

        time_col = "timestamp" if "power_flows" in key else "valid_time"

        # Add a configurable lookback for autoregressive features
        slice_start = train_start
        if "power_flows" in key or "nwps" in key:
            lookback = getattr(config.model, "required_lookback_days", 14)
            slice_start = train_start - timedelta(days=lookback)

        sliced_data[key] = _slice_temporal_data(val, slice_start, train_end, time_col)

    # 2. Call the Model-Specific Math
    # The trainer is responsible for joining and feature engineering.
    model = trainer.train(config=config.model, **sliced_data)

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
        if key == "substation_metadata":
            sliced_data[key] = val
            continue

        time_col = "timestamp" if "power_flows" in key else "valid_time"

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
            if not df.is_empty():
                delay_hours = getattr(config.model, "nwp_availability_delay_hours", 3)
                forecast_time = df.item() + timedelta(hours=delay_hours)

    inference_params = InferenceParams(
        forecast_time=forecast_time,
        power_fcst_model_name=model_name,
    )

    results_df = forecaster.predict(
        inference_params=inference_params, collapse_lead_times=False, **sliced_data
    )

    # 3. Calculate Metrics per lead_time
    if "substation_power_flows" in sliced_data:
        actuals = cast(
            pl.DataFrame, downsample_power_flows(sliced_data["substation_power_flows"]).collect()
        )

        # Join predictions with actuals
        eval_df = results_df.join(
            actuals.rename({"timestamp": "valid_time", "MW_or_MVA": "actual"}),
            on=["valid_time", "substation_number"],
            how="inner",
        )

        # Join the peak_capacity_MW_or_MVA from the forecaster's target_map
        if hasattr(forecaster, "target_map") and forecaster.target_map is not None:
            target_map_df = forecaster.target_map
            if isinstance(target_map_df, pl.LazyFrame):
                target_map_df = target_map_df.collect()
            # Cast to normal Polars DataFrame to avoid Patito type mismatch errors
            eval_df = pl.DataFrame(eval_df).join(
                pl.DataFrame(target_map_df).select(
                    ["substation_number", "peak_capacity_MW_or_MVA"]
                ),
                on="substation_number",
                how="left",
            )
            # Fill missing peak_capacity_MW_or_MVA with 1.0 to avoid division by zero
            eval_df = eval_df.with_columns(pl.col("peak_capacity_MW_or_MVA").fill_null(1.0))
        else:
            # Fallback if no target_map is available
            eval_df = eval_df.with_columns(peak_capacity_MW_or_MVA=pl.lit(1.0))

        if not eval_df.is_empty():
            # Filter out the lookback period to avoid data leakage in evaluation
            if isinstance(test_start, date) and not isinstance(test_start, datetime):
                test_start_dt = datetime.combine(test_start, datetime.min.time()).replace(
                    tzinfo=timezone.utc
                )
            else:
                test_start_dt = test_start

            eval_df = eval_df.filter(pl.col("valid_time") >= test_start_dt)

            # Calculate lead_time_hours
            if "nwp_init_time" in eval_df.columns:
                eval_df = eval_df.with_columns(
                    lead_time_hours=(
                        pl.col("valid_time") - pl.col("nwp_init_time")
                    ).dt.total_minutes()
                    / 60.0
                )
            else:
                # Fallback if nwp_init_time is not available
                eval_df = eval_df.with_columns(
                    lead_time_hours=(
                        pl.col("valid_time") - pl.lit(forecast_time)
                    ).dt.total_minutes()
                    / 60.0
                )

            # Group by lead_time_hours and calculate metrics
            metrics = (
                eval_df.group_by("lead_time_hours")
                .agg(
                    [
                        (pl.col("MW_or_MVA") - pl.col("actual")).abs().mean().alias("MAE"),
                        ((pl.col("MW_or_MVA") - pl.col("actual")) ** 2).mean().sqrt().alias("RMSE"),
                        (
                            (pl.col("MW_or_MVA") - pl.col("actual")).abs()
                            / pl.col("peak_capacity_MW_or_MVA")
                        )
                        .mean()
                        .alias("nMAE"),
                    ]
                )
                .sort("lead_time_hours")
            )

            # Log metrics to MLflow
            mlflow.set_experiment(model_name)
            with mlflow.start_run(run_name=f"{model_name}_eval"):
                for row in metrics.iter_rows(named=True):
                    lt = float(row["lead_time_hours"])
                    mlflow.log_metric(f"MAE_LT_{lt}h", row["MAE"])
                    mlflow.log_metric(f"RMSE_LT_{lt}h", row["RMSE"])
                    mlflow.log_metric(f"nMAE_LT_{lt}h", row["nMAE"])

                # Log global metrics
                mlflow.log_metric(
                    "MAE_global",
                    eval_df.select((pl.col("MW_or_MVA") - pl.col("actual")).abs().mean()).item(),
                )
                mlflow.log_metric(
                    "RMSE_global",
                    eval_df.select(
                        ((pl.col("MW_or_MVA") - pl.col("actual")) ** 2).mean().sqrt()
                    ).item(),
                )
                mlflow.log_metric(
                    "nMAE_global",
                    eval_df.select(
                        (
                            (pl.col("MW_or_MVA") - pl.col("actual")).abs()
                            / pl.col("peak_capacity_MW_or_MVA")
                        ).mean()
                    ).item(),
                )

    # 4. Trigger Dynamic Partition
    if hasattr(context, "instance") and context.instance:
        context.instance.add_dynamic_partitions("model_partitions", [model_name])

    if hasattr(context, "add_output_metadata"):
        context.add_output_metadata(
            {
                "num_rows": len(results_df),
                "power_fcst_model_name": model_name,
            }
        )

    return results_df
