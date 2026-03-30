from datetime import datetime, timedelta, timezone
import patito as pt
import polars as pl
import pytest
from contracts.data_schemas import (
    InferenceParams,
    ProcessedNwp,
    SubstationFlows,
    SubstationMetadata,
)
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
    XGBoostHyperparameters,
)
from xgboost_forecaster.model import XGBoostForecaster


def test_xgboost_forecaster_train_and_predict():
    # Setup dummy data
    sub_meta = pt.DataFrame[SubstationMetadata](
        {
            "substation_number": [1],
            "substation_name": ["Sub1"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [123],
            "last_updated": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
        }
    )

    # Create 4 weeks of data to satisfy the dynamic lag
    timestamps = pl.datetime_range(
        datetime(2025, 12, 15, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    sub_flows = pt.DataFrame[SubstationFlows](
        {
            "timestamp": timestamps,
            "substation_number": [1] * len(timestamps),
            "MW": [10.0] * len(timestamps),
            "MVA": [10.0] * len(timestamps),
            "MVAr": [0.0] * len(timestamps),
            "ingested_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * len(timestamps),
        }
    ).lazy()

    # NWPs only for the training period (Jan 1st to Jan 15th)
    nwp_timestamps = pl.datetime_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pt.DataFrame[ProcessedNwp](
            {
                "valid_time": nwp_timestamps,
                "init_time": nwp_timestamps - timedelta(hours=3),
                "lead_time_hours": [3.0] * len(nwp_timestamps),
                "h3_index": [123] * len(nwp_timestamps),
                "ensemble_member": [0] * len(nwp_timestamps),
                "temperature_2m": [15.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters=XGBoostHyperparameters(learning_rate=0.1, n_estimators=10, max_depth=3),
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG]),
    )

    # Initialize Forecaster
    forecaster = XGBoostForecaster()

    # Train
    forecaster.train(
        config=config,
        substation_power_flows=sub_flows,
        substation_metadata=sub_meta,
        nwps=nwps,
    )

    assert forecaster.model is not None

    # Predict
    inference_params = InferenceParams(
        nwp_init_time=datetime(2026, 1, 15, 3, tzinfo=timezone.utc),
        power_fcst_model_name="xgboost",
    )

    # Predict on the last day
    predict_timestamps = pl.datetime_range(
        datetime(2026, 1, 15, 3, tzinfo=timezone.utc),
        datetime(2026, 1, 15, 23, 30, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    predict_nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pt.DataFrame[ProcessedNwp](
            {
                "valid_time": predict_timestamps,
                "init_time": [datetime(2026, 1, 15, tzinfo=timezone.utc)] * len(predict_timestamps),
                "lead_time_hours": [3.0 + float(i) / 2.0 for i in range(len(predict_timestamps))],
                "h3_index": [123] * len(predict_timestamps),
                "ensemble_member": [0] * len(predict_timestamps),
                "temperature_2m": [15.0] * len(predict_timestamps),
            }
        ).lazy()
    }

    preds = forecaster.predict(
        substation_metadata=sub_meta,
        inference_params=inference_params,
        nwps=predict_nwps,
        substation_power_flows=sub_flows,
    )

    assert len(preds) == len(predict_timestamps)
    assert "MW_or_MVA" in preds.columns
    assert "power_fcst_model_name" in preds.columns
    assert preds["power_fcst_model_name"][0] == "xgboost"


def test_xgboost_forecaster_predict_empty():
    # Setup dummy data
    sub_meta = pt.DataFrame[SubstationMetadata](
        {
            "substation_number": [1],
            "substation_name": ["Sub1"],
            "latitude": [51.0],
            "longitude": [-1.0],
            "h3_res_5": [123],
            "last_updated": [datetime(2026, 1, 1, tzinfo=timezone.utc)],
        }
    )

    # Create 4 weeks of data to satisfy the dynamic lag
    timestamps = pl.datetime_range(
        datetime(2025, 12, 15, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    sub_flows = pt.DataFrame[SubstationFlows](
        {
            "timestamp": timestamps,
            "substation_number": [1] * len(timestamps),
            "MW": [10.0] * len(timestamps),
            "MVA": [10.0] * len(timestamps),
            "MVAr": [0.0] * len(timestamps),
            "ingested_at": [datetime(2026, 1, 1, tzinfo=timezone.utc)] * len(timestamps),
        }
    ).lazy()

    # NWPs only for the training period (Jan 1st to Jan 15th)
    nwp_timestamps = pl.datetime_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pt.DataFrame[ProcessedNwp](
            {
                "valid_time": nwp_timestamps,
                "init_time": nwp_timestamps - timedelta(hours=3),
                "lead_time_hours": [3.0] * len(nwp_timestamps),
                "h3_index": [123] * len(nwp_timestamps),
                "ensemble_member": [0] * len(nwp_timestamps),
                "temperature_2m": [15.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters=XGBoostHyperparameters(learning_rate=0.1, n_estimators=10, max_depth=3),
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG]),
    )

    # Initialize Forecaster
    forecaster = XGBoostForecaster()

    # Train
    forecaster.train(
        config=config,
        substation_power_flows=sub_flows,
        substation_metadata=sub_meta,
        nwps=nwps,
    )

    # Predict with nwp_init_time BEFORE any available NWPs
    inference_params = InferenceParams(
        nwp_init_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        power_fcst_model_name="xgboost",
    )

    predict_nwps = {
        NwpModel.ECMWF_ENS_0_25DEG: pt.DataFrame[ProcessedNwp](
            {
                "valid_time": nwp_timestamps,
                "init_time": nwp_timestamps,
                "lead_time_hours": [3.0] * len(nwp_timestamps),
                "h3_index": [123] * len(nwp_timestamps),
                "ensemble_member": [0] * len(nwp_timestamps),
                "temperature_2m": [15.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    with pytest.raises(ValueError, match="No inference data remaining"):
        forecaster.predict(
            substation_metadata=sub_meta,
            inference_params=inference_params,
            nwps=predict_nwps,
            substation_power_flows=sub_flows,
        )
