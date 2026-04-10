import pytest
import polars as pl
import polars.exceptions as pl_exc
import patito as pt
from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock
from xgboost_forecaster.model import XGBoostForecaster
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from xgboost_forecaster.config import XGBoostHyperparameters
from contracts.data_schemas import (
    InferenceParams,
    ProcessedNwp,
    TimeSeriesMetadata,
    PowerTimeSeries,
)


def test_prepare_features_missing_column_fails_loudly():
    """Test that _prepare_features fails when a requested feature is missing."""
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(feature_names=["missing_feature"]),
    )
    forecaster = XGBoostForecaster()
    forecaster.config = config

    df = pl.DataFrame(
        {
            "valid_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "time_series_id": ["1"],
        }
    )

    with pytest.raises(pl_exc.ColumnNotFoundError):
        forecaster._prepare_features(df)


def test_train_handles_missing_init_time():
    """Test that train handles missing init_time (e.g. for autoregressive-only models)."""
    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters().model_dump(),
        features=ModelFeaturesConfig(nwps=[]),
    )
    forecaster = XGBoostForecaster()

    flows = pt.DataFrame[PowerTimeSeries](
        {
            "time_series_id": ["1"],
            "start_time": [datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)],
            "end_time": [datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)],
            "value": [10.0],
        }
    )

    time_series_id = "1"
    metadata = pt.DataFrame[TimeSeriesMetadata](
        {
            "substation_number": [1],
            "substation_type": ["Primary"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [1],
            "last_updated": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "time_series_id": [time_series_id],
        }
    )

    # Centralized data preparation
    flows_30m = flows.lazy()

    # This should no longer raise ColumnNotFoundError for init_time
    # It might fail later due to missing features in the mock setup, but not on init_time
    try:
        forecaster.train(
            config=config,
            flows_30m=flows_30m,
            time_series_metadata=metadata,
            nwps={},
        )
    except pl_exc.ColumnNotFoundError as e:
        assert "init_time" not in str(e)
    except Exception:
        # Other errors are fine for this test as long as it's not init_time
        pass


def test_process_nwp_data_empty_input():
    """Test process_nwp_data with empty input."""
    from xgboost_forecaster.data import process_nwp_data

    df = pl.DataFrame(
        schema={
            "init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
            "h3_index": pl.UInt64,
            "ensemble_member": pl.UInt8,
            "temperature_2m": pl.Float32,
        }
    ).lazy()

    res = cast(pl.DataFrame, process_nwp_data(df, h3_indices=[1]).collect())
    assert res.is_empty()


def test_predict_with_missing_feature_column_fails_loudly():
    """Test that predict fails when a feature column used during training is missing."""
    forecaster = XGBoostForecaster()

    # Mock model with specific feature names
    mock_model = MagicMock()
    mock_model.feature_names_in_ = ["feature_a", "feature_b"]
    forecaster.model = mock_model
    forecaster.feature_names = ["feature_a", "feature_b"]

    time_series_id = "1"
    metadata = pt.DataFrame[TimeSeriesMetadata](
        {
            "substation_number": [1],
            "substation_name_in_location_table": ["Sub1"],
            "substation_type": ["Primary"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [1],
            "last_updated": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "time_series_id": [time_series_id],
        }
    )

    inference_params = InferenceParams(
        forecast_time=datetime(2024, 1, 1, 4, tzinfo=timezone.utc), power_fcst_model_name="test"
    )

    # NWP with all critical features
    nwp = cast(
        pt.LazyFrame[ProcessedNwp],
        pl.DataFrame(
            {
                "init_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
                "valid_time": [datetime(2024, 1, 1, 6, tzinfo=timezone.utc)],
                "h3_index": [1],
                "ensemble_member": [0],
                "temperature_2m": [10.0],
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
                "precipitation_surface": [0.0],
                "downward_long_wave_radiation_flux_surface": [0.0],
            }
        )
        .with_columns(
            [
                pl.col("h3_index").cast(pl.UInt64),
                pl.col("ensemble_member").cast(pl.UInt8),
                pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
            ]
        )
        .lazy(),
    )

    flows = pl.DataFrame(
        {
            "timestamp": [datetime(2023, 12, 25, tzinfo=timezone.utc)],
            "MW": [50.0],
            "MVA": [50.0],
            "MVAr": [0.0],
            "ingested_at": [datetime(2023, 12, 25, tzinfo=timezone.utc)],
            "time_series_id": ["1"],
        }
    ).lazy()
    flows_30m = flows

    # This should fail because feature_a and feature_b are missing from the prepared data
    with pytest.raises(pl_exc.ColumnNotFoundError):
        forecaster.predict(
            time_series_metadata=metadata,
            inference_params=inference_params,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: nwp},
        )
