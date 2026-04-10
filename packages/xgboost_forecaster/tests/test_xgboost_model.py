from datetime import datetime, timedelta, timezone
from typing import cast
import patito as pt
import polars as pl
import pytest
from contracts.data_schemas import (
    InferenceParams,
    ProcessedNwp,
    PowerTimeSeries,
    TimeSeriesMetadata,
)
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from xgboost_forecaster.config import XGBoostHyperparameters
from xgboost_forecaster.model import XGBoostForecaster


def test_xgboost_forecaster_train_and_predict():
    # Setup dummy data
    time_series_id = 1
    time_series_metadata = pt.DataFrame[TimeSeriesMetadata](
        {
            "time_series_id": ["1"],
            "substation_number": [time_series_id],
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

    sub_flows = pt.DataFrame[PowerTimeSeries](
        {
            "start_time": timestamps - timedelta(minutes=30),
            "end_time": timestamps,
            "time_series_id": ["1"] * len(timestamps),
            "value": [10.0] * len(timestamps),
        }
    ).lazy()

    # Centralized data preparation
    flows_30m = sub_flows

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
                "dew_point_temperature_2m": [10.0] * len(nwp_timestamps),
                "wind_speed_10m": [5.0] * len(nwp_timestamps),
                "wind_direction_10m": [180.0] * len(nwp_timestamps),
                "wind_speed_100m": [7.0] * len(nwp_timestamps),
                "wind_direction_100m": [185.0] * len(nwp_timestamps),
                "pressure_surface": [100.0] * len(nwp_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(nwp_timestamps),
                "geopotential_height_500hpa": [50.0] * len(nwp_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(nwp_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters=XGBoostHyperparameters(
            learning_rate=0.1, n_estimators=10, max_depth=3
        ).model_dump(),
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

    # Initialize Forecaster
    forecaster = XGBoostForecaster()

    # Train
    forecaster.train(
        config=config,
        flows_30m=flows_30m,
        time_series_metadata=time_series_metadata,
        nwps=nwps,
    )

    # Create 4 weeks of data to satisfy the dynamic lag
    timestamps = pl.datetime_range(
        datetime(2025, 12, 15, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    sub_flows = pt.DataFrame[PowerTimeSeries](
        {
            "start_time": timestamps - timedelta(minutes=30),
            "end_time": timestamps,
            "time_series_id": ["1"] * len(timestamps),
            "value": [10.0] * len(timestamps),
        }
    ).lazy()

    # Centralized data preparation
    flows_30m = sub_flows

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
                "dew_point_temperature_2m": [10.0] * len(nwp_timestamps),
                "wind_speed_10m": [5.0] * len(nwp_timestamps),
                "wind_direction_10m": [180.0] * len(nwp_timestamps),
                "wind_speed_100m": [7.0] * len(nwp_timestamps),
                "wind_direction_100m": [185.0] * len(nwp_timestamps),
                "pressure_surface": [100.0] * len(nwp_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(nwp_timestamps),
                "geopotential_height_500hpa": [50.0] * len(nwp_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(nwp_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters=XGBoostHyperparameters(
            learning_rate=0.1, n_estimators=10, max_depth=3
        ).model_dump(),
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

    # Initialize Forecaster
    forecaster = XGBoostForecaster()

    # Train
    forecaster.train(
        config=config,
        flows_30m=flows_30m,
        time_series_metadata=time_series_metadata,
        nwps=nwps,
    )

    assert forecaster.model is not None

    # Predict
    inference_params = InferenceParams(
        forecast_time=datetime(2026, 1, 15, 3, tzinfo=timezone.utc),
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
                "dew_point_temperature_2m": [10.0] * len(predict_timestamps),
                "wind_speed_10m": [5.0] * len(predict_timestamps),
                "wind_direction_10m": [180.0] * len(predict_timestamps),
                "wind_speed_100m": [7.0] * len(predict_timestamps),
                "wind_direction_100m": [185.0] * len(predict_timestamps),
                "pressure_surface": [100.0] * len(predict_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(predict_timestamps),
                "geopotential_height_500hpa": [50.0] * len(predict_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(predict_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(predict_timestamps),
            }
        ).lazy()
    }

    preds = forecaster.predict(
        time_series_metadata=time_series_metadata,
        inference_params=inference_params,
        nwps=predict_nwps,
        flows_30m=flows_30m,
    )

    assert len(preds) == len(predict_timestamps)
    assert "power_fcst_model_name" in preds.columns
    assert preds["power_fcst_model_name"][0] == "xgboost"


def test_xgboost_forecaster_predict_empty():
    # Setup dummy data
    time_series_id = 1
    time_series_metadata = pt.DataFrame[TimeSeriesMetadata](
        {
            "time_series_id": ["1"],
            "substation_number": [time_series_id],
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

    sub_flows = pt.DataFrame[PowerTimeSeries](
        {
            "start_time": timestamps - timedelta(minutes=30),
            "end_time": timestamps,
            "time_series_id": ["1"] * len(timestamps),
            "value": [10.0] * len(timestamps),
        }
    ).lazy()

    # Centralized data preparation
    flows_30m = sub_flows

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
                "dew_point_temperature_2m": [10.0] * len(nwp_timestamps),
                "wind_speed_10m": [5.0] * len(nwp_timestamps),
                "wind_direction_10m": [180.0] * len(nwp_timestamps),
                "wind_speed_100m": [7.0] * len(nwp_timestamps),
                "wind_direction_100m": [185.0] * len(nwp_timestamps),
                "pressure_surface": [100.0] * len(nwp_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(nwp_timestamps),
                "geopotential_height_500hpa": [50.0] * len(nwp_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(nwp_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters=XGBoostHyperparameters(
            learning_rate=0.1, n_estimators=10, max_depth=3
        ).model_dump(),
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

    # Initialize Forecaster
    forecaster = XGBoostForecaster()

    # Train
    forecaster.train(
        config=config,
        flows_30m=flows_30m,
        time_series_metadata=time_series_metadata,
        nwps=nwps,
    )

    # Create 4 weeks of data to satisfy the dynamic lag
    timestamps = pl.datetime_range(
        datetime(2025, 12, 15, tzinfo=timezone.utc),
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        "30m",
        eager=True,
    )

    sub_flows = pt.DataFrame[PowerTimeSeries](
        {
            "start_time": timestamps - timedelta(minutes=30),
            "end_time": timestamps,
            "time_series_id": ["1"] * len(timestamps),
            "value": [10.0] * len(timestamps),
        }
    ).lazy()

    # Centralized data preparation
    flows_30m = sub_flows

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
                "dew_point_temperature_2m": [10.0] * len(nwp_timestamps),
                "wind_speed_10m": [5.0] * len(nwp_timestamps),
                "wind_direction_10m": [180.0] * len(nwp_timestamps),
                "wind_speed_100m": [7.0] * len(nwp_timestamps),
                "wind_direction_100m": [185.0] * len(nwp_timestamps),
                "pressure_surface": [100.0] * len(nwp_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(nwp_timestamps),
                "geopotential_height_500hpa": [50.0] * len(nwp_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(nwp_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters=XGBoostHyperparameters(
            learning_rate=0.1, n_estimators=10, max_depth=3
        ).model_dump(),
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

    # Initialize Forecaster
    forecaster = XGBoostForecaster()

    # Train
    forecaster.train(
        config=config,
        flows_30m=flows_30m,
        time_series_metadata=time_series_metadata,
        nwps=nwps,
    )

    # Predict with forecast_time BEFORE any available NWPs
    inference_params = InferenceParams(
        forecast_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
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
                "dew_point_temperature_2m": [10.0] * len(nwp_timestamps),
                "wind_speed_10m": [5.0] * len(nwp_timestamps),
                "wind_direction_10m": [180.0] * len(nwp_timestamps),
                "wind_speed_100m": [7.0] * len(nwp_timestamps),
                "wind_direction_100m": [185.0] * len(nwp_timestamps),
                "pressure_surface": [100.0] * len(nwp_timestamps),
                "pressure_reduced_to_mean_sea_level": [101.0] * len(nwp_timestamps),
                "geopotential_height_500hpa": [50.0] * len(nwp_timestamps),
                "downward_short_wave_radiation_flux_surface": [100.0] * len(nwp_timestamps),
                "categorical_precipitation_type_surface": [0.0] * len(nwp_timestamps),
            }
        ).lazy()
    }

    with pytest.raises(ValueError, match="No inference data remaining"):
        forecaster.predict(
            time_series_metadata=time_series_metadata,
            inference_params=inference_params,
            nwps=predict_nwps,
            flows_30m=flows_30m,
        )


def test_prepare_and_join_nwps_handles_unsorted_input():
    """
    Test that _prepare_and_join_nwps correctly handles unsorted input data.
    """
    # Setup: Create NWP source with unsorted data
    timestamps = pl.datetime_range(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 6, tzinfo=timezone.utc),
        "1h",
        eager=True,
    )

    # Create unsorted data
    nwp_data = {
        "valid_time": timestamps,
        "init_time": timestamps - timedelta(hours=3),
        "lead_time_hours": [3.0] * len(timestamps),
        "h3_index": [123] * len(timestamps),
        "ensemble_member": [0] * len(timestamps),
        "temperature_2m": [15.0] * len(timestamps),
        "dew_point_temperature_2m": [10.0] * len(timestamps),
        "wind_speed_10m": [5.0] * len(timestamps),
        "wind_direction_10m": [180.0] * len(timestamps),
        "wind_speed_100m": [7.0] * len(timestamps),
        "wind_direction_100m": [185.0] * len(timestamps),
        "pressure_surface": [100.0] * len(timestamps),
        "pressure_reduced_to_mean_sea_level": [101.0] * len(timestamps),
        "geopotential_height_500hpa": [50.0] * len(timestamps),
        "downward_short_wave_radiation_flux_surface": [100.0] * len(timestamps),
        "categorical_precipitation_type_surface": [0.0] * len(timestamps),
    }

    # Create NWP source with unsorted valid_time
    nwp1 = pt.DataFrame[ProcessedNwp](nwp_data).lazy().sort("valid_time", descending=True)

    # Initialize Forecaster
    forecaster = XGBoostForecaster()
    # Set dummy config
    forecaster.config = ModelConfig(
        power_fcst_model_name="xgboost",
        hyperparameters={},
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG], feature_names=[]),
        nwp_availability_delay_hours=3,
    )

    # Call _prepare_and_join_nwps
    joined_nwps = forecaster._prepare_and_join_nwps(nwp1)

    # Verify the output is not empty and has the expected columns
    result = cast(pl.DataFrame, joined_nwps.collect())
    assert not result.is_empty()
    assert "temperature_2m" in result.columns
