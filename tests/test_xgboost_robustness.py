import pytest
import polars as pl
import patito as pt
from datetime import datetime, timezone, timedelta
from typing import Any, cast
from unittest.mock import MagicMock
from patito.exceptions import DataFrameValidationError
from contracts.data_schemas import (
    PowerTimeSeries,
    Nwp,
    ProcessedNwp,
    TimeSeriesMetadata,
    InferenceParams,
)
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from xgboost_forecaster.config import XGBoostHyperparameters
from xgboost_forecaster.model import XGBoostForecaster


def test_substation_power_flows_validation_extreme_values():
    """Test PowerTimeSeries validation with values outside the ge/le range."""
    # MW = 2000 is outside [-1000, 1000]
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "time_series_id": [123],
            "MW": [2000.0],
        }
    ).with_columns(
        [
            pl.col("time_series_id").cast(pl.Int32),
            pl.col("MW").cast(pl.Float32),
        ]
    )

    with pytest.raises(DataFrameValidationError):
        PowerTimeSeries.validate(df)


def test_nwp_validation_missing_accumulated_variables_at_step_1():
    """Test Nwp validation when accumulated variables are missing from the second step onwards."""
    init_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "init_time": [init_time, init_time],
            "valid_time": [init_time, init_time + timedelta(hours=1)],
            "ensemble_member": [0, 0],
            "h3_index": [123, 123],
            "temperature_2m": [10, 11],
            "dew_point_temperature_2m": [5, 6],
            "wind_u_10m": [2.0, 3.0],
            "wind_v_10m": [0.0, 0.0],
            "wind_u_100m": [4.0, 5.0],
            "wind_v_100m": [0.0, 0.0],
            "pressure_surface": [100, 101],
            "pressure_reduced_to_mean_sea_level": [102, 103],
            "geopotential_height_500hpa": [50, 51],
            "categorical_precipitation_type_surface": [0, 0],
            "precipitation_surface": [None, None],  # Step 1 is null!
            "downward_short_wave_radiation_flux_surface": [None, 100],
            "downward_long_wave_radiation_flux_surface": [None, 200],
        }
    ).with_columns(
        [
            pl.col("ensemble_member").cast(pl.UInt8),
            pl.col("h3_index").cast(pl.UInt64),
            pl.col("temperature_2m").cast(pl.Float32),
            pl.col("dew_point_temperature_2m").cast(pl.Float32),
            pl.col("wind_u_10m").cast(pl.Float32),
            pl.col("wind_v_10m").cast(pl.Float32),
            pl.col("wind_u_100m").cast(pl.Float32),
            pl.col("wind_v_100m").cast(pl.Float32),
            pl.col("pressure_surface").cast(pl.Float32),
            pl.col("pressure_reduced_to_mean_sea_level").cast(pl.Float32),
            pl.col("geopotential_height_500hpa").cast(pl.Float32),
            pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
            pl.col("precipitation_surface").cast(pl.Float32),
            pl.col("downward_short_wave_radiation_flux_surface").cast(pl.Float32),
            pl.col("downward_long_wave_radiation_flux_surface").cast(pl.Float32),
        ]
    )

    with pytest.raises(ValueError, match="Column 'precipitation_surface' contains 1 null values"):
        Nwp.validate(df)


def test_xgboost_forecaster_train_with_nans():
    """Test that train raises ValueError if input features contain NaNs."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
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
    )

    # Mock data with NaN in temperature_2m
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = (
        pl.DataFrame(
            {
                "period_end_time": [valid_time, valid_time - timedelta(days=7)],
                "time_series_id": [1, 1],
                "power": [50.0, 50.0],
                "MVA": [50.0, 50.0],
            }
        )
        .with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col("power").cast(pl.Float32),
            ]
        )
        .lazy()
    )

    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("nan")],  # NaN!
            "dew_point_temperature_2m": [5.0],
            "wind_speed_10m": [2.0],
            "wind_direction_10m": [180.0],
            "wind_speed_100m": [3.0],
            "wind_direction_100m": [185.0],
            "pressure_surface": [100.0],
            "pressure_reduced_to_mean_sea_level": [101.0],
            "geopotential_height_500hpa": [50.0],
            "downward_short_wave_radiation_flux_surface": [100.0],
            "categorical_precipitation_type_surface": [0.0],
            "lead_time_hours": [6.0],
        }
    ).lazy()

    # Centralized data preparation

    flows_30m = flows

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.train(
            config=config,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                metadata,
            ),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )


def test_xgboost_forecaster_train_with_infs():
    """Test that train raises ValueError if input features contain Infs."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
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
    )

    # Mock data with Inf in temperature_2m
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = (
        pl.DataFrame(
            {
                "period_end_time": [valid_time, valid_time - timedelta(days=7)],
                "time_series_id": [1, 1],
                "power": [50.0, 50.0],
                "MVA": [50.0, 50.0],
            }
        )
        .with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col("power").cast(pl.Float32),
            ]
        )
        .lazy()
    )

    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("inf")],  # Inf!
            "dew_point_temperature_2m": [5.0],
            "wind_speed_10m": [2.0],
            "wind_direction_10m": [180.0],
            "wind_speed_100m": [3.0],
            "wind_direction_100m": [185.0],
            "pressure_surface": [100.0],
            "pressure_reduced_to_mean_sea_level": [101.0],
            "geopotential_height_500hpa": [50.0],
            "downward_short_wave_radiation_flux_surface": [100.0],
            "categorical_precipitation_type_surface": [0.0],
            "lead_time_hours": [6.0],
        }
    ).lazy()

    # Centralized data preparation

    flows_30m = flows

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.train(
            config=config,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                metadata,
            ),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )


def test_xgboost_forecaster_predict_with_nans():
    """Test that predict raises ValueError if input features contain NaNs."""
    forecaster = XGBoostForecaster()
    mock_model = MagicMock()
    mock_model.feature_names_in_ = [
        "temperature_2m",
        "latest_available_weekly_power_lag",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week",
    ]
    forecaster.model = cast(Any, mock_model)
    forecaster.feature_names = mock_model.feature_names_in_

    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    # NWP with NaN
    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("nan")],  # NaN!
            "dew_point_temperature_2m": [5.0],
            "wind_speed_10m": [2.0],
            "wind_direction_10m": [180.0],
            "wind_speed_100m": [3.0],
            "wind_direction_100m": [185.0],
            "pressure_surface": [100.0],
            "pressure_reduced_to_mean_sea_level": [101.0],
            "geopotential_height_500hpa": [50.0],
            "downward_short_wave_radiation_flux_surface": [100.0],
            "categorical_precipitation_type_surface": [0.0],
        }
    ).lazy()

    flows = (
        pl.DataFrame(
            {
                "period_end_time": [valid_time - timedelta(days=7)],
                "time_series_id": [1],
                "power": [50.0],
                "MVA": [50.0],
            }
        )
        .with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col("power").cast(pl.Float32),
            ]
        )
        .lazy()
    )

    inference_params = InferenceParams(
        forecast_time=valid_time,
        power_fcst_model_name="test",
    )
    # Centralized data preparation

    flows_30m = flows

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.predict(
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                metadata,
            ),
            inference_params=inference_params,
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
            flows_30m=cast(pt.LazyFrame, flows_30m),
        )


def test_xgboost_forecaster_train_empty_data_after_drop_nulls():
    """Test that train raises ValueError if no training data remains after dropping nulls."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
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
    )

    # Mock data with null in temperature_2m (which is a critical column)
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = (
        pl.DataFrame(
            {
                "period_end_time": [valid_time],
                "time_series_id": [1],
                "power": [50.0],
                "MVA": [50.0],
            }
        )
        .with_columns(
            [
                pl.col("time_series_id").cast(pl.Int32),
                pl.col("power").cast(pl.Float32),
            ]
        )
        .lazy()
    )

    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    nwp = (
        pl.DataFrame(
            {
                "init_time": [valid_time - timedelta(hours=6)],
                "valid_time": [valid_time],
                "h3_index": [1],
                "ensemble_member": [0],
                "temperature_2m": [None],  # Null!
                "dew_point_temperature_2m": [5.0],
                "wind_speed_10m": [2.0],
                "wind_direction_10m": [180.0],
                "wind_speed_100m": [3.0],
                "wind_direction_100m": [185.0],
                "pressure_surface": [100.0],
                "pressure_reduced_to_mean_sea_level": [101.0],
                "geopotential_height_500hpa": [50.0],
                "downward_short_wave_radiation_flux_surface": [100.0],
                "categorical_precipitation_type_surface": [0.0],
                "lead_time_hours": [6.0],
            }
        )
        .with_columns(pl.col("temperature_2m").cast(pl.Float32))
        .lazy()
    )

    # Centralized data preparation

    flows_30m = flows

    with pytest.raises(
        ValueError, match="No training data remaining after dropping nulls in critical columns."
    ):
        forecaster.train(
            config=config,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                metadata,
            ),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )

    # Mock data with Inf in temperature_2m
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = (
        pl.DataFrame(
            {
                "timestamp": [valid_time, valid_time - timedelta(days=7)],
                "time_series_id": [1, 1],
                "MW": [50.0, 50.0],
                "MVA": [50.0, 50.0],
            }
        )
        .with_columns(pl.col("time_series_id").cast(pl.Int32))
        .lazy()
    )

    metadata = pl.DataFrame({"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]})

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("inf")],  # Inf!
            "dew_point_temperature_2m": [5.0],
            "wind_speed_10m": [2.0],
            "wind_direction_10m": [180.0],
            "wind_speed_100m": [3.0],
            "wind_direction_100m": [185.0],
            "pressure_surface": [100.0],
            "pressure_reduced_to_mean_sea_level": [101.0],
            "geopotential_height_500hpa": [50.0],
            "downward_short_wave_radiation_flux_surface": [100.0],
            "categorical_precipitation_type_surface": [0.0],
            "lead_time_hours": [6.0],
        }
    ).lazy()

    # Centralized data preparation

    flows_30m = flows.rename({"timestamp": "period_end_time", "MW": "power"})

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.train(
            config=config,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            time_series_metadata=cast(
                pt.DataFrame[TimeSeriesMetadata],
                metadata,
            ),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )
