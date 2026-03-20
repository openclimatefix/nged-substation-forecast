import logging
from datetime import date, datetime, timedelta
from typing import cast

import dagster as dg
import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import InferenceParams, PowerForecast
from contracts.settings import Settings
from dagster import ResourceParam
from mlflow.models import ModelSignature
from mlflow.types import ParamSchema, ParamSpec
from xgboost_forecaster import (
    DataConfig,
    EnsembleSelection,
    XGBoostForecaster,
    XGBoostPyFuncWrapper,
    get_substation_metadata,
    prepare_inference_data,
    prepare_training_data,
    train_local_xgboost_model,
)

from pydantic import BaseModel

log = logging.getLogger(__name__)


# We define both a Pydantic BaseModel (XGBoostTrainingParams) and a Dagster Config
# (XGBoostTrainingConfig) to handle the two ways training parameters enter our pipeline:
#
# 1. XGBoostTrainingParams: Used for *dynamic* data passed between ops. In the
#    cross-validation job, the training windows are generated during the run, making
#    them "Data" rather than "Configuration". Using a pure Pydantic model also keeps
#    our core training logic decoupled from Dagster.
#
# 2. XGBoostTrainingConfig: Used for *static* configuration provided by a user or
#    schedule before the run starts. This allows for type-safe configuration in the
#    Dagster UI.


class XGBoostTrainingParams(BaseModel):
    """Parameters for XGBoost training.

    This is a pure Pydantic model used for passing training parameters as data
    between ops (e.g. in cross-validation).
    """

    train_start_date: str | None = None
    train_end_date: str | None = None
    test_end_date: str | None = None
    substation_numbers: list[int] | None = None


class XGBoostTrainingConfig(dg.Config):
    """Configuration for XGBoost training.

    This is a Dagster Config object used for static configuration provided at
    launch time via the Dagster UI or run configuration.
    """

    train_start_date: str | None = None
    train_end_date: str | None = None
    test_end_date: str | None = None
    substation_numbers: list[int] | None = None

    def to_params(self) -> XGBoostTrainingParams:
        """Convert static config into dynamic params."""
        return XGBoostTrainingParams(
            train_start_date=self.train_start_date,
            train_end_date=self.train_end_date,
            test_end_date=self.test_end_date,
            substation_numbers=self.substation_numbers,
        )


def train_xgboost_models_for_range(
    context: dg.AssetExecutionContext | dg.OpExecutionContext,
    params: XGBoostTrainingParams,
    settings: Settings,
) -> dict[str, str]:
    """Train XGBoost models for a given date range.

    Args:
        context: Dagster context.
        params: Training parameters.
        settings: Application settings.

    Returns:
        A dictionary mapping substation numbers to model paths.
    """
    power_path = settings.nged_data_path / "delta" / "live_primary_flows"
    if not power_path.exists():
        context.log.warning("No Delta table found.")
        return {}

    substation_numbers = params.substation_numbers
    if substation_numbers is None:
        # This fallback is now handled by the `xgb_models` asset, but we keep it
        # for robustness when called directly.
        substation_numbers = (
            pl.read_delta(str(power_path))
            .select("substation_number")
            .unique()
            .to_series()
            .to_list()
        )

    context.log.info(f"Training models for {len(substation_numbers)} substations")

    data_config = DataConfig(
        base_power_path=power_path,
        base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # Resolve training dates
    if params.train_end_date:
        end_date = date.fromisoformat(params.train_end_date)
    else:
        # Default to 7 days ago to ensure data is settled
        end_date = (datetime.now() - timedelta(days=7)).date()

    if params.train_start_date:
        start_date = date.fromisoformat(params.train_start_date)
    else:
        # Default to 1 year before end_date
        start_date = end_date - timedelta(days=365)

    context.log.info(f"Training from {start_date} to {end_date}")

    df_all = prepare_training_data(
        substation_numbers=substation_numbers,
        metadata=metadata,
        start_date=start_date,
        end_date=end_date,
        config=data_config,
        selection=EnsembleSelection.MEAN,
        use_lags=True,
    )

    artifacts = {}
    for substation_number in substation_numbers:
        try:
            df = df_all.filter(pl.col("substation_number") == substation_number)

            if df.is_empty():
                context.log.warning(f"No data available for substation {substation_number}")
                continue

            # Save model to a temporary location for MLflow to pick up
            model_path = (
                settings.trained_ml_model_params_base_path
                / XGBoostForecaster.model_name_and_version()
                / f"{substation_number}.json"
            )

            train_local_xgboost_model(
                substation_number=substation_number,
                df=df,
                output_path=model_path,
            )

            artifacts[str(substation_number)] = str(model_path)

        except Exception as e:
            context.log.error(f"Failed to train model for {substation_number}: {e}")

    return artifacts


@dg.asset(deps=["live_primary_flows", "ecmwf_ens_forecast"])
def xgb_models(
    context: dg.AssetExecutionContext,
    config: XGBoostTrainingConfig,
    healthy_substations: list[int],
    settings: ResourceParam[Settings],
) -> dg.Output[str]:
    """Train XGBoost models for all substations and log to MLflow."""
    params = config.to_params()
    if params.substation_numbers is None:
        params.substation_numbers = healthy_substations

    artifacts = train_xgboost_models_for_range(context, params, settings)

    if not artifacts:
        return dg.Output("", metadata={"n_models": 0})

    # Log to MLflow
    # We omit the MLflow input/output signature to prevent MLflow from forcing the
    # input into a Pandas DataFrame. We rely on Patito for strict schema validation instead.
    # However, we manually define a params schema so that MLflow allows parameters
    # to be passed at inference time.
    # TODO: Log the training data `df_all` to MLflow using `mlflow.data.from_polars()`
    # to establish data lineage for this model run.
    signature = ModelSignature(
        inputs=None,
        outputs=None,
        params=ParamSchema(
            [
                ParamSpec(name="nwp_init_time", dtype="datetime", default=datetime(2000, 1, 1)),
                ParamSpec(name="power_fcst_model", dtype="string", default=""),
            ]
        ),
    )

    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=XGBoostPyFuncWrapper(),
            artifacts=artifacts,
            signature=signature,
        )

    return dg.Output(
        model_info.model_uri,
        metadata={
            "n_models": len(artifacts),
            "substations": dg.MetadataValue.text(", ".join(artifacts.keys())),
            "mlflow_run_id": model_info.run_id,
            "model_uri": model_info.model_uri,
        },
    )


class XGBoostInferenceConfig(dg.Config):
    """Configuration for XGBoost inference."""

    init_time: str | None = None
    substation_numbers: list[int] | None = None


@dg.asset(deps=["ecmwf_ens_forecast"])
def xgb_forecasts(
    context: dg.AssetExecutionContext,
    xgb_models: str,
    config: XGBoostInferenceConfig,
    settings: ResourceParam[Settings],
) -> dg.Output[pl.DataFrame]:
    """Generate forecasts for all substations using the trained XGBoost models via MLflow."""
    if not xgb_models:
        return dg.Output(pl.DataFrame(), metadata={"n_forecasts": 0})

    # Load model from MLflow
    loaded_model = mlflow.pyfunc.load_model(xgb_models)

    data_config = DataConfig(
        base_power_path=settings.nged_data_path / "delta" / "live_primary_flows",
        base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # Resolve init_time
    if config.init_time:
        init_time = datetime.fromisoformat(config.init_time)
    else:
        # Default to the most recent midnight
        init_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    context.log.info(f"Generating forecasts for init_time {init_time}")

    # We need to know which substations we have models for.
    # If substation_numbers is provided in config, use that.
    # Otherwise, try all substations in metadata.
    substation_numbers = config.substation_numbers
    if substation_numbers is None:
        substation_numbers = metadata.select("substation_number").to_series().to_list()

    all_inference_data = []
    for substation_number in substation_numbers:
        try:
            df = prepare_inference_data(
                substation_number=substation_number,
                init_time=init_time,
                metadata=metadata,
                config=data_config,
                use_lags=True,
            )
            if not df.is_empty():
                all_inference_data.append(df)
        except Exception as e:
            context.log.warning(f"Failed to prepare inference data for {substation_number}: {e}")

    if not all_inference_data:
        context.log.error("No inference data prepared for any substation.")
        return dg.Output(pl.DataFrame(), metadata={"n_forecasts": 0})

    inference_df = pl.concat(all_inference_data)
    context.log.info(
        f"Prepared inference data for {inference_df['substation_number'].n_unique()} substations"
    )

    # Make predictions using the model-agnostic MLflow wrapper.
    # We pass an InferenceParams object which MLflow will serialize to a dict.
    params = InferenceParams(
        nwp_init_time=init_time,
        power_fcst_model=XGBoostForecaster.model_name_and_version(),
    )

    preds_df = cast(
        pt.DataFrame[PowerForecast],
        loaded_model.predict(inference_df, params=params.model_dump()),
    )

    # Save combined forecast
    forecast_path = (
        settings.power_forecasts_data_path
        / XGBoostForecaster.model_name_and_version()
        / "all_substations.parquet"
    )
    forecast_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.write_parquet(forecast_path)

    return dg.Output(
        preds_df,
        metadata={
            "path": dg.MetadataValue.path(forecast_path),
            "n_points": len(preds_df),
            "n_substations": preds_df.select("substation_number").n_unique(),
        },
    )
