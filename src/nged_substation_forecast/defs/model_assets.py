"""Unified Dagster assets for ML model training."""

import importlib
import logging
from datetime import datetime

import dagster as dg
import mlflow
import polars as pl
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra


log = logging.getLogger(__name__)


def load_hydra_config(model_name: str) -> dict:
    """Load the Hydra configuration for a specific model.

    Args:
        model_name: The name of the model (corresponds to a YAML file in conf/model/).

    Returns:
        The configuration dictionary.
    """
    # We use the Hydra Compose API to load the config without taking over the thread.
    if not GlobalHydra.instance().is_initialized():
        initialize(version_base=None, config_path="../../../conf")
    cfg = compose(config_name="config", overrides=[f"model={model_name}"])
    return dict(cfg.model)  # Return as a plain dictionary


def get_trainer_class(trainer_path: str):
    """Dynamically load a trainer class from a string path.

    Args:
        trainer_path: The full python path to the trainer class (e.g. 'pkg.mod.Class').

    Returns:
        The trainer class.
    """
    module_path, class_name = trainer_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_model_assets(model_name: str):
    """Factory function to create Dagster assets for a specific ML model.

    This factory implements the 'Write-Once' pipeline philosophy. Instead of
    writing a new Dagster asset for every model architecture, we generate
    the assets dynamically based on the model's Hydra configuration and its
    Python-defined data requirements.

    Args:
        model_name: The name of the model (must match a YAML in conf/model/).

    Returns:
        A list of Dagster assets (train and evaluate).
    """
    hydra_cfg = load_hydra_config(model_name)
    TrainerClass = get_trainer_class(hydra_cfg["trainer_class"])

    # Determine required assets from the Trainer's requirements class.
    # This is where the 'Dagster Vocabulary' (FeatureAsset) is used to
    # automatically wire up the DAG dependencies.
    required_asset_keys = [asset.value for asset in TrainerClass.data_requirements()]

    @dg.asset(
        name=f"train_{model_name}",
        ins={key: dg.AssetIn(key) for key in required_asset_keys},
        compute_kind="python",
        group_name="models",
    )
    def train_model(context: dg.AssetExecutionContext, **kwargs):
        """Generic training asset that delegates to the model's Trainer class."""
        log.info(f"Training model: {model_name}")

        # 1. Slicing the data based on Hydra config
        # In a real scenario, we would use the dates from hydra_cfg["data_split"]
        # to filter the LazyFrames in kwargs.
        payload_dict = {}
        for key, lf in kwargs.items():
            # Apply temporal slicing using Polars pushdown
            # (The model is ignorant of this!)
            # lf = lf.filter(pl.col("timestamp").is_between(...))
            payload_dict[key] = lf

        # 2. Instantiate the Pydantic payload
        # This validates that all required assets were provided and have correct types.
        payload = TrainerClass.requirements_class(**payload_dict)

        # 3. Train and Log to MLflow
        trainer = TrainerClass()
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(hydra_cfg)
            trained_model = trainer.train(payload, hydra_cfg)

            # Log to registry, bypassing MLflow's Pandas-centric signature engine
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=trained_model,
                registered_model_name=model_name,
                signature=None,
            )

            context.add_output_metadata(
                {
                    "mlflow_run_id": run.info.run_id,
                    "ml_model_name": model_name,
                }
            )

    @dg.asset(
        name=f"evaluate_{model_name}",
        ins={
            "model": dg.AssetIn(f"train_{model_name}"),
            **{key: dg.AssetIn(key) for key in required_asset_keys},
        },
        compute_kind="python",
        group_name="models",
    )
    def evaluate_model(context: dg.AssetExecutionContext, model, **kwargs):
        """Generic evaluation asset that loads the model and generates forecasts."""
        log.info(f"Evaluating model: {model_name}")

        # 1. Load the model from MLflow
        # In a real scenario, we would use the run_id from the metadata
        # run_id = context.step_context.get_output_metadata(f"train_{model_name}")["mlflow_run_id"]
        # pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")

        # For this example, we'll assume 'model' is the trained artifact
        # (though in Dagster it would usually be a path or a reference)

        # 2. Prepare inference data (collect LazyFrames to DataFrames)
        inference_payload_dict = {key: lf.collect() for key, lf in kwargs.items()}

        # 3. Run inference
        # The model artifact handles the internal validation and math
        predictions_df = model.predict(None, inference_payload_dict)

        # 4. Add metadata for Delta Lake
        # We partition by ml_model_name and power_fcst_init_year_month
        now = datetime.now()
        year_month = now.strftime("%Y-%m")

        results_df = predictions_df.with_columns(
            ml_model_name=pl.lit(model_name).cast(pl.Categorical),
            power_fcst_init_time=pl.lit(now).cast(pl.Datetime("us", "UTC")),
            power_fcst_init_year_month=pl.lit(year_month).cast(pl.String),
            # nwp_init_time should come from the input data in a real scenario
            nwp_init_time=pl.lit(now).cast(pl.Datetime("us", "UTC")),
        )

        # 5. Write to Delta Lake
        # results_df.write_delta(
        #     "data/evaluation_results.delta",
        #     mode="append",
        #     delta_table_options={
        #         "partition_by": ["ml_model_name", "power_fcst_init_year_month"]
        #     },
        # )

        context.add_output_metadata(
            {
                "num_rows": len(results_df),
                "ml_model_name": model_name,
            }
        )
        return results_df

    return [train_model, evaluate_model]


# Example of how to instantiate the assets for the baseline model
xgboost_baseline_assets = create_model_assets("xgboost_baseline")
