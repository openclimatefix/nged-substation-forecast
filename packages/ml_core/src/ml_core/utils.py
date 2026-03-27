import logging
from datetime import datetime, timezone

import dagster as dg
import mlflow
import polars as pl
from contracts.data_schemas import InferenceParams
from contracts.hydra_schemas import TrainingConfig

log = logging.getLogger(__name__)


def train_and_log_model(
    context: dg.AssetExecutionContext,
    model_name: str,
    trainer,
    config: TrainingConfig,
    flavor: str,
    **kwargs,
):
    """Universal utility to handle temporal slicing and MLflow logging for training.

    Args:
        context: Dagster execution context.
        model_name: Name of the model (for MLflow run name).
        trainer: An object with a `train(config, **kwargs)` method.
        config: Training configuration object.
        flavor: MLflow flavor to use for logging (e.g., 'xgboost', 'pytorch').
        **kwargs: Input LazyFrames to be temporally sliced.

    Returns:
        The trained model object.
    """
    # 1. Universal Temporal Slicing
    train_start = config.data_split.train_start
    train_end = config.data_split.train_end

    sliced_data = {}
    for key, lf in kwargs.items():
        if key == "substation_metadata":
            sliced_data[key] = lf
            continue
        # We assume all feature assets have a time-based column.
        # For power flows it's 'timestamp', for NWP it's 'valid_time'.
        time_col = "timestamp" if "power_flows" in key else "valid_time"
        sliced_data[key] = lf.filter(pl.col(time_col).is_between(train_start, train_end))

    # 2. Call the Model-Specific Math
    # The trainer is responsible for joining and feature engineering.
    model = trainer.train(config=config.model, **sliced_data)

    # 3. Universal MLflow Logging
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(config.model_dump())

        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            raise ValueError(f"Unsupported MLflow flavor: {flavor}")

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
    for key, lf in kwargs.items():
        if key == "substation_metadata":
            sliced_data[key] = lf.collect() if isinstance(lf, pl.LazyFrame) else lf
            continue
        time_col = "timestamp" if "power_flows" in key else "valid_time"
        sliced_data[key] = lf.filter(pl.col(time_col).is_between(test_start, test_end)).collect()

    # 2. Call the Model-Specific Inference
    # Construct InferenceParams. We use the current time for nwp_init_time
    # as a placeholder if it's not available in the data.
    inference_params = InferenceParams(
        nwp_init_time=datetime.now(timezone.utc),
        power_fcst_model=model_name,
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
