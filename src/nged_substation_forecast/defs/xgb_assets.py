import logging
from datetime import datetime
from typing import cast

import dagster as dg
import mlflow
import patito as pt
import polars as pl
from contracts.data_schemas import PowerForecast
from contracts.settings import Settings
from dagster import ResourceParam
from xgboost_forecaster import (
    DataConfig,
    EnsembleSelection,
    XGBoostForecaster,
    XGBoostPyFuncWrapper,
    get_substation_metadata,
    prepare_inference_data,
    prepare_training_data,
)

log = logging.getLogger(__name__)


@dg.asset(deps=["live_primary_flows", "ecmwf_ens_forecast"])
def xgb_models(
    context: dg.AssetExecutionContext, settings: ResourceParam[Settings]
) -> dg.Output[str]:
    """Train XGBoost models for all substations and log to MLflow."""
    power_path = settings.nged_data_path / "delta" / "live_primary_flows"
    if not power_path.exists():
        context.log.warning("No Delta table found.")
        return dg.Output("", metadata={"n_models": 0})

    substation_numbers = (
        pl.read_delta(str(power_path)).select("substation_number").unique().to_series().to_list()
    )

    context.log.info(f"Training models for {len(substation_numbers)} substations")

    data_config = DataConfig(
        base_power_path=power_path,
        base_weather_path=settings.nwp_data_path / "ECMWF" / "ENS",
    )
    metadata = get_substation_metadata(data_config)

    # We need a start and end date for training.
    # For now, let's use a fixed range.
    # TODO: This should be configurable so we can do expanding window time series cross-validation
    # (i.e. where we mimic what happens in production) and so that, when we train for production,
    # we use as much historical data as possible.
    start_date = datetime(2026, 2, 1).date()
    end_date = datetime(2026, 2, 28).date()

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
            # TODO: Move all the code in this `try` block into a new function.

            df = df_all.filter(pl.col("substation_number") == substation_number)

            if df.is_empty():
                context.log.warning(f"No data available for substation {substation_number}")
                continue

            # Train model
            forecaster = XGBoostForecaster()

            # Split into train/eval
            df = df.sort("valid_time")
            train_size = int(len(df) * 0.8)
            train_df = df.head(train_size)
            eval_df = df.tail(len(df) - train_size)

            target_col = "MW_or_MVA"
            feature_cols = [
                c for c in df.columns if c not in [target_col, "valid_time", "substation_number"]
            ]

            eval_set = [(eval_df, eval_df[target_col])]

            forecaster.train(
                df=train_df,
                target_col=target_col,
                feature_cols=feature_cols,
                eval_set=eval_set,
            )

            # Save model to a temporary location for MLflow to pick up
            model_path = (
                settings.trained_ml_model_params_base_path
                / XGBoostForecaster.model_name_and_version()
                / f"{substation_number}.json"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            forecaster.save(model_path)
            artifacts[str(substation_number)] = str(model_path)

        except Exception as e:
            context.log.error(f"Failed to train model for {substation_number}: {e}")

    # Log to MLflow
    # We omit the MLflow signature to prevent MLflow from forcing the input into a Pandas DataFrame.
    # We rely on Patito for strict schema validation instead.
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=XGBoostPyFuncWrapper(),
            artifacts=artifacts,
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


@dg.asset(deps=["ecmwf_ens_forecast"])
def xgb_forecasts(
    context: dg.AssetExecutionContext, xgb_models: str, settings: ResourceParam[Settings]
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

    # TODO (p1): This MUST be changed to dynamically use the most recent NWP init!
    init_time = datetime(2026, 2, 17, 0)  # Example init time

    # Prepare inference data for all substations at once
    # Currently prepare_inference_data only takes one substation_number.
    # Let's check if we can loop over them or if we should update prepare_inference_data.
    # For now, let's loop to keep it simple, but we'll pass the whole DF to the model.

    # We need to know which substations we have models for.
    # The pyfunc wrapper knows this, but we don't have easy access to its internal state here.
    # However, we can just try to prepare data for all substations that have metadata.
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
            context.log.error(f"Failed to prepare inference data for {substation_number}: {e}")

    if not all_inference_data:
        return dg.Output(pl.DataFrame(), metadata={"n_forecasts": 0})

    inference_df = pl.concat(all_inference_data)

    # Make predictions using the model-agnostic MLflow wrapper.
    # MLflow's pyfunc.PythonModel.predict expects a list of inputs when using
    # custom types (like Patito DataFrames) to support batching.
    # We wrap our single DataFrame in a list and unwrap the result.
    result_list = loaded_model.predict(
        [inference_df],
        params={
            "nwp_init_time": init_time.isoformat(),
            "power_fcst_model": XGBoostForecaster.model_name_and_version(),
        },
    )
    preds_df = cast(pt.DataFrame[PowerForecast], result_list[0])

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
