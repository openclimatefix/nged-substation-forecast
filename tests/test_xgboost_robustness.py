import pytest
import polars as pl
import patito as pt
from datetime import datetime, timezone, timedelta
from typing import cast
from unittest.mock import MagicMock
from patito.exceptions import DataFrameValidationError
from contracts.data_schemas import (
    SubstationFlows,
    Nwp,
    ProcessedNwp,
    SubstationMetadata,
    InferenceParams,
)
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from xgboost_forecaster.config import XGBoostHyperparameters
from xgboost_forecaster.model import XGBoostForecaster


def test_substation_flows_validation_extreme_values():
    """Test SubstationFlows validation with values outside the ge/le range."""
    # MW = 2000 is outside [-1000, 1000]
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [123],
            "MW": [2000.0],
        }
    ).with_columns(
        [
            pl.col("substation_number").cast(pl.Int32),
            pl.col("MW").cast(pl.Float32),
        ]
    )

    with pytest.raises(DataFrameValidationError):
        SubstationFlows.validate(df)


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
            "wind_speed_10m": [2, 3],
            "wind_direction_10m": [180, 190],
            "wind_speed_100m": [4, 5],
            "wind_direction_100m": [200, 210],
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
            pl.col("temperature_2m").cast(pl.UInt8),
            pl.col("dew_point_temperature_2m").cast(pl.UInt8),
            pl.col("wind_speed_10m").cast(pl.UInt8),
            pl.col("wind_direction_10m").cast(pl.UInt8),
            pl.col("wind_speed_100m").cast(pl.UInt8),
            pl.col("wind_direction_100m").cast(pl.UInt8),
            pl.col("pressure_surface").cast(pl.UInt8),
            pl.col("pressure_reduced_to_mean_sea_level").cast(pl.UInt8),
            pl.col("geopotential_height_500hpa").cast(pl.UInt8),
            pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
            pl.col("precipitation_surface").cast(pl.UInt8),
            pl.col("downward_short_wave_radiation_flux_surface").cast(pl.UInt8),
            pl.col("downward_long_wave_radiation_flux_surface").cast(pl.UInt8),
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
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG]),
    )

    # Mock data with NaN in temperature_2m
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = pl.DataFrame(
        {
            "timestamp": [valid_time, valid_time - timedelta(days=7)],
            "substation_number": [1, 1],
            "MW": [50.0, 50.0],
            "MVA": [50.0, 50.0],
        }
    ).lazy()

    metadata = pl.DataFrame({"substation_number": [1], "h3_res_5": [1]})

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("nan")],  # NaN!
            "lead_time_hours": [6.0],
        }
    ).lazy()

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.train(
            config=config,
            substation_power_flows=cast(pt.LazyFrame[SubstationFlows], flows),
            substation_metadata=cast(pt.DataFrame[SubstationMetadata], metadata),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )


def test_xgboost_forecaster_train_with_infs():
    """Test that train raises ValueError if input features contain Infs."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG]),
    )

    # Mock data with Inf in temperature_2m
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = pl.DataFrame(
        {
            "timestamp": [valid_time, valid_time - timedelta(days=7)],
            "substation_number": [1, 1],
            "MW": [50.0, 50.0],
            "MVA": [50.0, 50.0],
        }
    ).lazy()

    metadata = pl.DataFrame({"substation_number": [1], "h3_res_5": [1]})

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("inf")],  # Inf!
            "lead_time_hours": [6.0],
        }
    ).lazy()

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.train(
            config=config,
            substation_power_flows=cast(pt.LazyFrame[SubstationFlows], flows),
            substation_metadata=cast(pt.DataFrame[SubstationMetadata], metadata),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )


def test_xgboost_forecaster_predict_with_nans():
    """Test that predict raises ValueError if input features contain NaNs."""
    forecaster = XGBoostForecaster()
    forecaster.model = MagicMock()
    forecaster.model.feature_names_in_ = [
        "temperature_2m",
        "latest_available_weekly_lag",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week",
    ]

    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    metadata = pl.DataFrame({"substation_number": [1], "h3_res_5": [1]})

    # NWP with NaN
    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [float("nan")],  # NaN!
        }
    ).lazy()

    flows = pl.DataFrame(
        {
            "timestamp": [valid_time - timedelta(days=7)],
            "substation_number": [1],
            "MW": [50.0],
            "MVA": [50.0],
        }
    ).lazy()

    inference_params = InferenceParams(
        nwp_init_time=valid_time,
        power_fcst_model_name="test",
    )

    with pytest.raises(ValueError, match="Input features X contain NaN or Inf values"):
        forecaster.predict(
            substation_metadata=cast(pt.DataFrame[SubstationMetadata], metadata),
            inference_params=inference_params,
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
            substation_power_flows=cast(pt.LazyFrame[SubstationFlows], flows),
        )


def test_xgboost_forecaster_train_empty_data_after_drop_nulls():
    """Test that train raises ValueError if no training data remains after dropping nulls."""
    forecaster = XGBoostForecaster()
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG]),
    )

    # Mock data with null in temperature_2m (which is a critical column)
    valid_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    flows = pl.DataFrame(
        {
            "timestamp": [valid_time],
            "substation_number": [1],
            "MW": [50.0],
            "MVA": [50.0],
        }
    ).lazy()

    metadata = pl.DataFrame({"substation_number": [1], "h3_res_5": [1]})

    nwp = pl.DataFrame(
        {
            "init_time": [valid_time - timedelta(hours=6)],
            "valid_time": [valid_time],
            "h3_index": [1],
            "ensemble_member": [0],
            "temperature_2m": [None],  # Null!
            "lead_time_hours": [6.0],
        }
    ).lazy()

    with pytest.raises(ValueError, match="No training data remaining after dropping nulls"):
        forecaster.train(
            config=config,
            substation_power_flows=cast(pt.LazyFrame[SubstationFlows], flows),
            substation_metadata=cast(pt.DataFrame[SubstationMetadata], metadata),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )
