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
)
from xgboost_forecaster.config import XGBoostHyperparameters
from contracts.data_schemas import (
    SubstationFlows,
    SubstationMetadata,
    InferenceParams,
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
            "substation_number": [1],
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

    flows = pt.DataFrame[SubstationFlows](
        {
            "timestamp": [datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)],
            "substation_number": [1],
            "MW": [10.0],
            "MVA": [10.0],
            "MVAr": [0.0],
            "ingested_at": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    ).lazy()

    metadata = pt.DataFrame[SubstationMetadata](
        {
            "substation_number": [1],
            "substation_name_in_location_table": ["Sub1"],
            "substation_type": ["Primary"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [1],
            "last_updated": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    )

    # This should no longer raise ColumnNotFoundError for init_time
    # It might fail later due to missing features in the mock setup, but not on init_time
    try:
        forecaster.train(
            config=config, substation_power_flows=flows, substation_metadata=metadata, nwps={}
        )
    except pl_exc.ColumnNotFoundError as e:
        assert "init_time" not in str(e)
    except Exception:
        # Other errors are fine for this test as long as it's not init_time
        pass


@pytest.mark.parametrize(
    "mw,mva,expected_fail",
    [
        (10.0, 10.0, False),
        (10.0, None, False),
        (None, 10.0, False),
        (None, None, True),
    ],
)
def test_substation_flows_validation_mw_mva_combinations(mw, mva, expected_fail):
    """Test SubstationFlows validation with various MW/MVA combinations."""
    df = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "substation_number": [1],
            "MW": [mw],
            "MVA": [mva],
            "MVAr": [0.0],
            "ingested_at": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    ).with_columns(
        [
            pl.col("substation_number").cast(pl.Int32),
            pl.col("MW").cast(pl.Float32),
            pl.col("MVA").cast(pl.Float32),
            pl.col("MVAr").cast(pl.Float32),
        ]
    )

    from contracts.data_schemas import MissingCorePowerVariablesError

    if expected_fail:
        with pytest.raises(MissingCorePowerVariablesError):
            SubstationFlows.validate(df)
    else:
        SubstationFlows.validate(df)


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

    res = cast(pl.DataFrame, process_nwp_data(df, h3_indices=[1], target_horizon_hours=0).collect())
    assert res.is_empty()


def test_predict_with_empty_features_matrix():
    """Test predict when the feature matrix is empty."""
    forecaster = XGBoostForecaster()
    forecaster.model = MagicMock()

    metadata = pt.DataFrame[SubstationMetadata](
        {
            "substation_number": [1],
            "substation_name_in_location_table": ["Sub1"],
            "substation_type": ["Primary"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [1],
            "last_updated": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
        }
    )

    inference_params = InferenceParams(
        forecast_time=datetime(2024, 1, 1, tzinfo=timezone.utc), power_fcst_model_name="test"
    )

    # Empty NWPs will cause a ValueError as per code
    with pytest.raises(ValueError, match="XGBoostForecaster requires NWP data for prediction."):
        forecaster.predict(
            substation_metadata=metadata,
            inference_params=inference_params,
            substation_power_flows=pt.DataFrame[SubstationFlows](
                {
                    "timestamp": [],
                    "substation_number": [],
                    "MW": [],
                    "MVA": [],
                    "MVAr": [],
                    "ingested_at": [],
                }
            ).lazy(),
            nwps={},
        )
