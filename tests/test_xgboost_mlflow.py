import pytest
import polars as pl
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from ml_core.utils import train_and_log_model, evaluate_and_save_model
from contracts.hydra_schemas import (
    TrainingConfig,
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
    DataSplitConfig,
)
from xgboost_forecaster.model import XGBoostForecaster
from xgboost_forecaster.config import XGBoostHyperparameters


def test_evaluate_and_save_model_logs_metrics():
    """Test that evaluate_and_save_model correctly logs metrics to MLflow."""

    # Setup dummy data
    sub_meta = pl.DataFrame(
        {
            "time_series_id": ["1"],
            "substation_name": ["Sub1"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [123],
            "last_updated": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
        }
    ).with_columns(pl.col("time_series_id").cast(pl.String))

    valid_time = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    sub_flows = (
        pl.DataFrame(
            {
                "end_time": [valid_time],
                "time_series_id": ["1"],
                "value": [10.0],
                "MVA": [10.0],
                "MVAr": [0.0],
                "ingested_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            }
        )
        .with_columns(pl.col("time_series_id").cast(pl.String))
        .lazy()
    )

    nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pl.DataFrame(
            {
                "valid_time": [valid_time],
                "init_time": [valid_time - timedelta(hours=24)],
                "lead_time_hours": [24.0],
                "h3_index": [123],
                "ensemble_member": [0],
                "temperature_2m": [15.0],
                "dew_point_temperature_2m": [10.0],
                "wind_speed_10m": [5.0],
                "wind_direction_10m": [180.0],
                "wind_speed_100m": [7.0],
                "wind_direction_100m": [185.0],
                "pressure_surface": [100.0],
                "pressure_reduced_to_mean_sea_level": [101.0],
                "geopotential_height_500hpa": [50.0],
                "downward_short_wave_radiation_flux_surface": [100.0],
                "categorical_precipitation_type_surface": [0.0],
            }
        )
        .with_columns(
            [
                pl.col("h3_index").cast(pl.UInt64),
                pl.col("ensemble_member").cast(pl.UInt8),
            ]
        )
        .lazy()
    }

    config = TrainingConfig(
        model=ModelConfig(
            power_fcst_model_name="xgboost",
            hyperparameters=XGBoostHyperparameters().model_dump(),
            features=ModelFeaturesConfig(
                nwps=[NwpModel.ECMWF_ENS_0_25DEG],
                feature_names=[
                    "time_series_id",
                    "lead_time_hours",
                    "latest_available_weekly_power_lag",
                    "temperature_2m",
                    "downward_short_wave_radiation_flux_surface",
                    "wind_speed_10m",
                    "hour_sin",
                    "hour_cos",
                    "day_of_year_sin",
                    "day_of_year_cos",
                    "day_of_week",
                ],
            ),
        ),
        data_split=DataSplitConfig(
            train_start=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            train_end=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            test_start=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
            test_end=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
        ),
    )

    nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pl.DataFrame(
            {
                "valid_time": [valid_time],
                "init_time": [valid_time - timedelta(hours=24)],
                "lead_time_hours": [24.0],
                "h3_index": [123],
                "ensemble_member": [0],
                "temperature_2m": [15.0],
                "dew_point_temperature_2m": [10.0],
                "wind_speed_10m": [5.0],
                "wind_direction_10m": [180.0],
                "wind_speed_100m": [7.0],
                "wind_direction_100m": [185.0],
                "pressure_surface": [100.0],
                "pressure_reduced_to_mean_sea_level": [101.0],
                "geopotential_height_500hpa": [50.0],
                "downward_short_wave_radiation_flux_surface": [100.0],
                "categorical_precipitation_type_surface": [0.0],
            }
        )
        .with_columns(
            [
                pl.col("h3_index").cast(pl.UInt64),
                pl.col("ensemble_member").cast(pl.UInt8),
            ]
        )
        .lazy()
    }

    config = TrainingConfig(
        model=ModelConfig(
            power_fcst_model_name="xgboost",
            hyperparameters=XGBoostHyperparameters().model_dump(),
            features=ModelFeaturesConfig(
                nwps=[NwpModel.ECMWF_ENS_0_25DEG],
                feature_names=[
                    "time_series_id",
                    "lead_time_hours",
                    "latest_available_weekly_power_lag",
                    "temperature_2m",
                    "downward_short_wave_radiation_flux_surface",
                    "wind_speed_10m",
                    "hour_sin",
                    "hour_cos",
                    "day_of_year_sin",
                    "day_of_year_cos",
                    "day_of_week",
                ],
            ),
        ),
        data_split=DataSplitConfig(
            train_start=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            train_end=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            test_start=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
            test_end=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
        ),
    )

    forecaster = MagicMock()
    # Mock predict to return a DataFrame that matches the expected schema
    forecaster.predict.return_value = pl.DataFrame(
        {
            "valid_time": [valid_time],
            "time_series_id": ["1"],
            "ensemble_member": [0],
            "value": [11.0],  # Prediction is 11.0, actual is 10.0
            "nwp_init_time": [valid_time - timedelta(hours=24)],
        }
    ).with_columns(
        [
            pl.col("time_series_id").cast(pl.String),
            pl.col("ensemble_member").cast(pl.UInt8),
            pl.col("value").cast(pl.Float32),
        ]
    )

    context = MagicMock()

    with (
        patch("mlflow.start_run") as mock_start_run,
        patch("mlflow.log_metric") as mock_log_metric,
        patch("mlflow.set_experiment") as mock_set_experiment,
    ):
        evaluate_and_save_model(
            context=context,
            model_name="xgboost",
            forecaster=forecaster,
            config=config,
            nwps=nwps,
            substation_power_flows=sub_flows,
            substation_metadata=sub_meta,
        )

        mock_set_experiment.assert_called_with("xgboost")
        # Check that metrics were logged. MAE should be 1.0 (abs(11-10))
        # RMSE should be 1.0. nMAE should be 0.01 (1.0 / 100.0)

        # We expect multiple calls to log_metric (per lead time and global)
        metric_calls = {call.args[0]: call.args[1] for call in mock_log_metric.call_args_list}

        assert metric_calls["MAE_global"] == pytest.approx(1.0)
        assert metric_calls["RMSE_global"] == pytest.approx(1.0)
        assert metric_calls["nMAE_global"] == pytest.approx(0.1)
        assert metric_calls["MAE_LT_24.0h"] == pytest.approx(1.0)

    # Setup dummy data
    sub_meta = pl.DataFrame(
        {
            "time_series_id": ["1"],
            "substation_name": ["Sub1"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [123],
            "last_updated": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
        }
    ).with_columns(pl.col("time_series_id").cast(pl.String))

    sub_flows = (
        pl.DataFrame(
            {
                "end_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                "time_series_id": ["1"],
                "value": [10.0],
                "MVA": [10.0],
                "MVAr": [0.0],
                "ingested_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            }
        )
        .with_columns(pl.col("time_series_id").cast(pl.String))
        .lazy()
    )

    nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pl.DataFrame(
            {
                "valid_time": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                "init_time": [datetime(2025, 12, 31, 21, tzinfo=timezone.utc)],
                "lead_time_hours": [3.0],
                "h3_index": [123],
                "ensemble_member": [0],
                "temperature_2m": [15.0],
                "dew_point_temperature_2m": [10.0],
                "wind_speed_10m": [5.0],
                "wind_direction_10m": [180.0],
                "wind_speed_100m": [7.0],
                "wind_direction_100m": [185.0],
                "pressure_surface": [100.0],
                "pressure_reduced_to_mean_sea_level": [101.0],
                "geopotential_height_500hpa": [50.0],
                "downward_short_wave_radiation_flux_surface": [100.0],
                "categorical_precipitation_type_surface": [0.0],
            }
        )
        .with_columns(
            [
                pl.col("h3_index").cast(pl.UInt64),
                pl.col("ensemble_member").cast(pl.UInt8),
            ]
        )
        .lazy()
    }

    config = TrainingConfig(
        model=ModelConfig(
            power_fcst_model_name="xgboost",
            hyperparameters=XGBoostHyperparameters().model_dump(),
            features=ModelFeaturesConfig(
                nwps=[NwpModel.ECMWF_ENS_0_25DEG],
                feature_names=[
                    "time_series_id",
                    "lead_time_hours",
                    "latest_available_weekly_power_lag",
                    "temperature_2m",
                    "downward_short_wave_radiation_flux_surface",
                    "wind_speed_10m",
                    "hour_sin",
                    "hour_cos",
                    "day_of_year_sin",
                    "day_of_year_cos",
                    "day_of_week",
                ],
            ),
        ),
        data_split=DataSplitConfig(
            train_start=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            train_end=datetime(2026, 1, 1, tzinfo=timezone.utc).date(),
            test_start=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
            test_end=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
        ),
    )

    trainer = XGBoostForecaster()
    context = MagicMock()

    with (
        patch("mlflow.start_run") as mock_start_run,
        patch("mlflow.log_params") as mock_log_params,
        patch("mlflow.set_experiment") as mock_set_experiment,
        patch("mlflow.xgboost.log_model") as mock_log_model,
    ):
        # Mock the run object returned by start_run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_start_run.return_value.__enter__.return_value = mock_run

        train_and_log_model(
            context=context,
            model_name="xgboost",
            trainer=trainer,
            config=config,
            substation_power_flows=sub_flows,
            time_series_metadata=sub_meta,
            nwps=nwps,
        )

        mock_set_experiment.assert_called_once_with("xgboost")
        mock_start_run.assert_called_once()
        mock_log_params.assert_called_once()
        mock_log_model.assert_called_once()
        context.add_output_metadata.assert_called_once_with({"mlflow_run_id": "test_run_id"})
