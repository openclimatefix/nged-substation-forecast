import logging
from datetime import date, datetime, timezone
from typing import Any

import dagster as dg
import mlflow
import polars as pl
from contracts.data_schemas import InferenceParams
from contracts.hydra_schemas import TrainingConfig

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
    context: dg.AssetExecutionContext,
    model_name: str,
    trainer,
    config: TrainingConfig,
    **kwargs,
):
    """Universal utility to handle temporal slicing and MLflow logging for training.

    Args:
        context: Dagster execution context.
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
        sliced_data[key] = _slice_temporal_data(val, train_start, train_end, time_col)

    # 2. Call the Model-Specific Math
    # The trainer is responsible for joining and feature engineering.
    model = trainer.train(config=config.model, **sliced_data)

    # 3. Universal MLflow Logging
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(config.model_dump(mode="json"))
        trainer.log_model(model_name)
        context.add_output_metadata({"mlflow_run_id": run.info.run_id})

    return model


def evaluate_and_save_model(
    context: dg.AssetExecutionContext,
    model_name: str,
    forecaster,
    config: TrainingConfig,
    **kwargs,
):
    """Universal utility to handle temporal slicing, inference, and storage.

    Args:
        context: Dagster execution context.
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
            sliced_data[key] = val.collect() if isinstance(val, pl.LazyFrame) else val
            continue

        time_col = "timestamp" if "power_flows" in key else "valid_time"
        sliced = _slice_temporal_data(val, test_start, test_end, time_col)

        # Collect LazyFrames into DataFrames for inference
        if isinstance(sliced, dict):
            sliced_data[key] = {
                k: (v.collect() if isinstance(v, pl.LazyFrame) else v) for k, v in sliced.items()
            }
        else:
            sliced_data[key] = sliced.collect() if isinstance(sliced, pl.LazyFrame) else sliced

    # 2. Call the Model-Specific Inference
    # Extract the actual init_time from the provided nwps data
    nwp_init_time = datetime.now(timezone.utc)
    if "nwps" in kwargs:
        # Assuming nwps is a dictionary or list of LazyFrames
        nwps_data = kwargs["nwps"]
        if isinstance(nwps_data, dict) and nwps_data:
            first_nwp = next(iter(nwps_data.values()))
            if isinstance(first_nwp, pl.LazyFrame):
                df = first_nwp.select(pl.col("init_time").max()).collect()
                if isinstance(df, pl.DataFrame):
                    nwp_init_time = df.item()
        elif isinstance(nwps_data, list) and nwps_data:
            first_nwp = nwps_data[0]
            if isinstance(first_nwp, pl.LazyFrame):
                df = first_nwp.select(pl.col("init_time").max()).collect()
                if isinstance(df, pl.DataFrame):
                    nwp_init_time = df.item()
        elif isinstance(nwps_data, pl.LazyFrame):
            df = nwps_data.select(pl.col("init_time").max()).collect()
            if isinstance(df, pl.DataFrame):
                nwp_init_time = df.item()

    inference_params = InferenceParams(
        nwp_init_time=nwp_init_time,
        power_fcst_model_name=model_name,
    )

    results_df = forecaster.predict(inference_params=inference_params, **sliced_data)

    # 3. Trigger Dynamic Partition
    context.instance.add_dynamic_partitions("model_partitions", [model_name])

    context.add_output_metadata(
        {
            "num_rows": len(results_df),
            "power_fcst_model_name": model_name,
        }
    )

    return results_df
