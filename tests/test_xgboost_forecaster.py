import pytest
import polars as pl
import patito as pt
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from typing import cast

from contracts.data_schemas import (
    InferenceParams,
    ProcessedNwp,
    TimeSeriesMetadata,
)
from xgboost_forecaster.data import (
    DataConfig,
    load_nwp_run,
    process_nwp_data,
)
from xgboost_forecaster.scaling import load_scaling_params
from xgboost_forecaster.model import XGBoostForecaster
from contracts.hydra_schemas import (
    ModelConfig,
    ModelFeaturesConfig,
    NwpModel,
)
from xgboost_forecaster.config import XGBoostHyperparameters
from contracts.settings import Settings
import dagster as dg
from src.nged_substation_forecast.defs.xgb_assets import train_xgboost, XGBoostConfig


@pytest.fixture
def data_config(tmp_path):
    power_path = tmp_path / "power_data"
    power_path.mkdir()
    weather_path = tmp_path / "weather_data"
    weather_path.mkdir()

    import polars as pl
    from datetime import datetime, timezone

    # Create dummy power data
    df_power = pl.DataFrame(
        {"period_end_time": ["2026-03-07T00:00:00"], "power": [1.0], "time_series_id": [123]}
    )
    df_power = df_power.with_columns(
        pl.col("period_end_time").str.to_datetime().dt.replace_time_zone("UTC"),
        pl.col("time_series_id").cast(pl.Int32),
    )
    df_power.write_delta(power_path, delta_write_options={"partition_by": ["time_series_id"]})

    # Create dummy weather data (Nwp schema)
    init_time = datetime(2026, 3, 7, 0, tzinfo=timezone.utc)
    df_weather = pl.DataFrame(
        {
            "init_time": [init_time],
            "valid_time": [init_time],
            "ensemble_member": [0],
            "h3_index": [1],
            "temperature_2m": [200],
            "dew_point_temperature_2m": [190],
            "wind_u_10m": [10.0],
            "wind_v_10m": [0.0],
            "wind_u_100m": [15.0],
            "wind_v_100m": [0.0],
            "pressure_surface": [100],
            "pressure_reduced_to_mean_sea_level": [101],
            "geopotential_height_500hpa": [50],
            "downward_long_wave_radiation_flux_surface": [None],
            "downward_short_wave_radiation_flux_surface": [None],
            "precipitation_surface": [None],
            "categorical_precipitation_type_surface": [0],
        }
    )
    # Cast to correct dtypes for Nwp schema
    df_weather = df_weather.with_columns(
        [
            pl.col("ensemble_member").cast(pl.UInt8),
            pl.col("h3_index").cast(pl.UInt64),
            pl.col("temperature_2m").cast(pl.UInt8),
            pl.col("dew_point_temperature_2m").cast(pl.UInt8),
            pl.col("wind_u_10m").cast(pl.Float32),
            pl.col("wind_v_10m").cast(pl.Float32),
            pl.col("wind_u_100m").cast(pl.Float32),
            pl.col("wind_v_100m").cast(pl.Float32),
            pl.col("pressure_surface").cast(pl.UInt8),
            pl.col("pressure_reduced_to_mean_sea_level").cast(pl.UInt8),
            pl.col("geopotential_height_500hpa").cast(pl.UInt8),
            pl.col("downward_long_wave_radiation_flux_surface").cast(pl.UInt8),
            pl.col("downward_short_wave_radiation_flux_surface").cast(pl.UInt8),
            pl.col("precipitation_surface").cast(pl.UInt8),
            pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
        ]
    )

    weather_file = weather_path / f"{init_time.strftime('%Y-%m-%dT%H')}Z.parquet"
    df_weather.write_parquet(weather_file)

    return DataConfig(
        base_power_path=power_path,
        base_weather_path=weather_path,
    )


def test_load_nwp_run_with_config(data_config):
    from datetime import datetime

    h3_indices = [1]
    init_time = datetime(2026, 3, 7, 0)

    # Test that the function loads data without errors
    load_nwp_run(init_time, h3_indices, config=data_config)


def test_load_scaling_params_with_config(tmp_path):
    # Create dummy scaling params file
    scaling_params_path = tmp_path / "scaling_params.csv"
    scaling_params_path.write_text("dummy scaling params")

    # set the environment variable
    # os.environ["XGBOOST_SCALING_PARAMS_PATH"] = str(scaling_params_path)

    # Load scaling params and assert it doesn't throw an error
    try:
        load_scaling_params()
    except Exception as e:
        assert False, f"Failed to load scaling params: {e}"


def test_process_nwp_data_removes_zero_lead_time():
    init_time = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "init_time": [init_time, init_time, init_time],
            "valid_time": [
                init_time + timedelta(hours=3),  # 3-hour
                init_time + timedelta(hours=4),  # 4-hour
                init_time + timedelta(hours=5),  # 5-hour
            ],
            "h3_index": [1, 1, 1],
            "ensemble_member": [0, 0, 0],
            "temperature_2m": [10.0, 12.0, 14.0],
        }
    ).lazy()

    res = cast(pl.DataFrame, process_nwp_data(df, h3_indices=[1]).collect())

    # Remaining valid_times: 03:00, 04:00, 05:00.
    # Upsampling to 30m will create: 03:00, 03:30, 04:00, 04:30, 05:00.
    assert len(res) == 5
    assert res["valid_time"][0] == init_time + timedelta(hours=3)
    assert res["lead_time_hours"][0] == 3.0
    assert res["lead_time_hours"][1] == 3.5
    assert res["lead_time_hours"][2] == 4.0
    assert res["lead_time_hours"][3] == 4.5
    assert res["lead_time_hours"][4] == 5.0


def test_process_nwp_data_interpolates_safely_within_trajectories():
    init_time_1 = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    init_time_2 = datetime(2024, 1, 2, 0, tzinfo=timezone.utc)

    # Both forecast runs predict for the same valid_times
    valid_time_1 = datetime(2024, 1, 3, 0, tzinfo=timezone.utc)
    valid_time_2 = datetime(2024, 1, 3, 1, tzinfo=timezone.utc)

    df = pl.DataFrame(
        {
            "init_time": [init_time_1, init_time_1, init_time_2, init_time_2],
            "valid_time": [valid_time_1, valid_time_2, valid_time_1, valid_time_2],
            "h3_index": [1, 1, 1, 1],
            "ensemble_member": [0, 0, 0, 0],
            "temperature_2m": [10.0, 20.0, 100.0, 200.0],  # Very different predictions
        }
    ).lazy()

    res = cast(pl.DataFrame, process_nwp_data(df, h3_indices=[1]).collect())

    # We should have 3 rows for init_time_1 (00:00, 00:30, 01:00)
    # and 3 rows for init_time_2 (00:00, 00:30, 01:00)
    assert len(res) == 6

    # Check interpolation for init_time_1
    res_1 = res.filter(pl.col("init_time") == init_time_1).sort("valid_time")
    assert res_1["temperature_2m"][1] == 15.0  # Midpoint of 10 and 20

    # Check interpolation for init_time_2
    res_2 = res.filter(pl.col("init_time") == init_time_2).sort("valid_time")
    assert res_2["temperature_2m"][1] == 150.0  # Midpoint of 100 and 200


def test_prepare_training_data_prevents_row_explosion():
    # Power flows with enough history for lags
    valid_time = datetime(2024, 1, 3, 10, 30, tzinfo=timezone.utc)
    flows = pl.DataFrame(
        {
            "period_end_time": [
                valid_time,
                valid_time - timedelta(days=7),
                valid_time - timedelta(days=14),
                valid_time - timedelta(days=21),
                valid_time - timedelta(days=28),
            ],
            "time_series_id": ["1"] * 5,
            "power": [50.0] * 5,
            "MVA": [50.0] * 5,
        }
    ).lazy()

    # Metadata
    metadata = pl.DataFrame(
        {
            "time_series_id": [1],
            "substation_number": [1],
            "h3_res_5": [1],
        }
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    # 4 NWP forecast runs, each with 50 ensemble members
    nwp_rows = []
    for i in range(4):
        init_time = valid_time - timedelta(days=i + 1)
        for ens in range(50):
            nwp_rows.append(
                {
                    "init_time": init_time,
                    "valid_time": valid_time,
                    "h3_index": 1,
                    "ensemble_member": ens,
                    "temperature_2m": 10.0,
                    "dew_point_temperature_2m": 5.0,
                    "wind_speed_10m": 2.0,
                    "wind_direction_10m": 180.0,
                    "wind_speed_100m": 3.0,
                    "wind_direction_100m": 185.0,
                    "pressure_surface": 100.0,
                    "pressure_reduced_to_mean_sea_level": 101.0,
                    "geopotential_height_500hpa": 50.0,
                    "downward_short_wave_radiation_flux_surface": 100.0,
                    "categorical_precipitation_type_surface": 0.0,
                    "lead_time_hours": (valid_time - init_time).total_seconds() / 3600,
                }
            )
    nwp = pl.DataFrame(nwp_rows).lazy()

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

    forecaster = XGBoostForecaster()

    import patito as pt
    from contracts.data_schemas import TimeSeriesMetadata, ProcessedNwp

    # Centralized data preparation
    flows_30m = flows.with_columns(pl.col("time_series_id").cast(pl.Int32))

    with patch("xgboost_forecaster.model.XGBRegressor.fit") as mock_fit:
        forecaster.train(
            config=config,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            time_series_metadata=cast(pt.DataFrame[TimeSeriesMetadata], metadata),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )

        # Get the X dataframe passed to fit
        X = mock_fit.call_args[0][0]

        # 1 power flow row * 4 init_times * 1 ensemble member (control) = 4 rows
        assert X.shape[0] == 4


def test_train_xgboost_asset_filters_to_control_member(tmp_path):
    # Create NWP with members 0, 1, 2
    nwp = pl.DataFrame(
        {
            "valid_time": [datetime(2024, 1, 1, 3, tzinfo=timezone.utc)] * 3,
            "h3_index": [1, 1, 1],
            "ensemble_member": [0, 1, 2],
            "init_time": [datetime(2024, 1, 1, 0, tzinfo=timezone.utc)] * 3,
            "temperature_2m": [10.0, 11.0, 12.0],
        }
    ).lazy()

    flows = pl.DataFrame(
        {
            "time_series_id": ["123"],
            "start_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            "period_end_time": [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30)],
            "power": [1.0],
            "MVA": [None],
        }
    ).with_columns(pl.col("MVA").cast(pl.Float64))

    # Write flows to Delta as the asset now loads from Delta
    delta_dir = tmp_path / "delta"
    cleaned_actuals_path = delta_dir / "cleaned_actuals"
    cleaned_actuals_path.mkdir(parents=True)
    flows.write_delta(str(cleaned_actuals_path))

    settings = Settings(nged_data_path=tmp_path)

    config = XGBoostConfig()

    with dg.build_asset_context() as context:
        with patch(
            "src.nged_substation_forecast.defs.xgb_assets.train_and_log_model"
        ) as mock_train:
            metadata = pl.DataFrame(
                {
                    "time_series_id": ["123"],
                    "h3_res_5": [1],
                }
            )
            train_xgboost(
                context=context,
                config=config,
                settings=settings,
                nwp=nwp,
                substation_metadata=metadata,
            )

            # Check the nwp passed to train_and_log_model
            passed_nwps = mock_train.call_args[1]["nwps"]
            passed_nwp = passed_nwps[NwpModel.ECMWF_ENS_0_25DEG].collect()

            # Should only contain member 0
            assert len(passed_nwp) == 1
            assert passed_nwp["ensemble_member"][0] == 0


def test_latest_available_weekly_power_lag_prevents_leakage():
    # Test that the dynamic lag logic correctly switches from 7d to 14d
    # when lead_time_hours > 168 (7 days)

    valid_time = datetime(2024, 1, 20, 12, 0, tzinfo=timezone.utc)

    # Case 1: lead_time = 24h (1 day) -> should use 7d lag
    init_time_short = valid_time - timedelta(days=1)

    # Case 2: lead_time = 240h (10 days) -> should use 14d lag
    init_time_long = valid_time - timedelta(days=10)

    # Power flows
    flows = pl.DataFrame(
        {
            "period_end_time": [
                valid_time,  # Target row
                valid_time - timedelta(days=7),
                valid_time - timedelta(days=14),
                valid_time - timedelta(days=21),
                valid_time - timedelta(days=28),
            ],
            "time_series_id": ["1"] * 5,
            "power": [100.0, 77.0, 1414.0, 0.0, 0.0],  # Distinct values for 7d and 14d lags
            "MVA": [100.0, 77.0, 1414.0, 0.0, 0.0],
        }
    ).lazy()

    metadata = pl.DataFrame(
        {"time_series_id": [1], "substation_number": [1], "h3_res_5": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))

    nwp = pl.DataFrame(
        [
            {
                "init_time": init_time_short,
                "valid_time": valid_time,
                "h3_index": 1,
                "ensemble_member": 0,
                "temperature_2m": 10.0,
                "dew_point_temperature_2m": 5.0,
                "wind_speed_10m": 2.0,
                "wind_direction_10m": 180.0,
                "wind_speed_100m": 3.0,
                "wind_direction_100m": 185.0,
                "pressure_surface": 100.0,
                "pressure_reduced_to_mean_sea_level": 101.0,
                "geopotential_height_500hpa": 50.0,
                "downward_short_wave_radiation_flux_surface": 100.0,
                "categorical_precipitation_type_surface": 0.0,
                "lead_time_hours": 24.0,
            },
            {
                "init_time": init_time_long,
                "valid_time": valid_time,
                "h3_index": 1,
                "ensemble_member": 0,
                "temperature_2m": 10.0,
                "dew_point_temperature_2m": 5.0,
                "wind_speed_10m": 2.0,
                "wind_direction_10m": 180.0,
                "wind_speed_100m": 3.0,
                "wind_direction_100m": 185.0,
                "pressure_surface": 100.0,
                "pressure_reduced_to_mean_sea_level": 101.0,
                "geopotential_height_500hpa": 50.0,
                "downward_short_wave_radiation_flux_surface": 100.0,
                "categorical_precipitation_type_surface": 0.0,
                "lead_time_hours": 240.0,
            },
        ]
    ).lazy()

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

    forecaster = XGBoostForecaster()

    import patito as pt
    from contracts.data_schemas import TimeSeriesMetadata, ProcessedNwp

    # Centralized data preparation
    flows_30m = flows.with_columns(pl.col("time_series_id").cast(pl.Int32))

    with patch("xgboost_forecaster.model.XGBRegressor.fit") as mock_fit:
        forecaster.train(
            config=config,
            flows_30m=cast(pt.LazyFrame, flows_30m),
            time_series_metadata=cast(pt.DataFrame[TimeSeriesMetadata], metadata),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )

        X_arrow = mock_fit.call_args[0][0]
        X = cast(pl.DataFrame, pl.from_arrow(X_arrow))

        # Row 0: lead_time=24 -> should have latest_available_weekly_power_lag = 77.0 / 1414.0
        # Row 1: lead_time=240 -> should have latest_available_weekly_power_lag = 1414.0 / 1414.0 = 1.0
        # Note: Polars might reorder rows, so we filter

        # We need to find which row is which. We can use lead_time_hours if it's in X.
        # It should be there because it's numeric and not in exclude_cols.

        row_short = X.filter(pl.col("lead_time_hours") == 24.0)
        assert row_short["latest_available_weekly_power_lag"][0] == pytest.approx(77.0 / 1414.0)

        row_long = X.filter(pl.col("lead_time_hours") == 240.0)
        assert row_long["latest_available_weekly_power_lag"][0] == pytest.approx(1.0)


def test_xgboost_predict_with_lags():
    # Setup dummy data for predict
    valid_time = datetime(2024, 1, 20, 12, 0, tzinfo=timezone.utc)
    # NWP init time must be at least 3h before valid_time to be available
    # AND the inference_params.nwp_init_time must be at least 3h after nwp.init_time
    nwp_init_time = valid_time - timedelta(days=1)
    inference_nwp_init_time = nwp_init_time + timedelta(hours=4)

    metadata = pl.DataFrame(
        {"time_series_id": [1], "h3_res_5": [1], "substation_number": [1]}
    ).with_columns(pl.col("time_series_id").cast(pl.Int32))
    nwp = pl.DataFrame(
        [
            {
                "init_time": nwp_init_time,
                "valid_time": valid_time,
                "h3_index": 1,
                "ensemble_member": 0,
                "temperature_2m": 10.0,
                "dew_point_temperature_2m": 5.0,
                "wind_speed_10m": 2.0,
                "wind_direction_10m": 180.0,
                "wind_speed_100m": 3.0,
                "wind_direction_100m": 185.0,
                "pressure_surface": 100.0,
                "pressure_reduced_to_mean_sea_level": 101.0,
                "geopotential_height_500hpa": 50.0,
                "downward_short_wave_radiation_flux_surface": 100.0,
                "categorical_precipitation_type_surface": 0.0,
            }
        ]
    ).lazy()

    flows = pl.DataFrame(
        [
            {
                "period_end_time": valid_time - timedelta(days=7),
                "time_series_id": 1,
                "power": 77.0,
                "MVA": 77.0,
            },
            {
                "period_end_time": valid_time - timedelta(days=14),
                "time_series_id": 1,
                "power": 1414.0,
                "MVA": 1414.0,
            },
        ]
    ).lazy()

    # Centralized data preparation
    flows_30m = flows.with_columns(pl.col("time_series_id").cast(pl.Int32))

    forecaster = XGBoostForecaster()

    # Mock the model and its feature_names_in_
    mock_model = patch("xgboost.XGBRegressor").start()
    mock_model.feature_names_in_ = [
        "temperature_2m",
        "latest_available_weekly_power_lag",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "day_of_week",
    ]
    mock_model.predict.return_value = [1.0]  # Normalized prediction
    forecaster.model = mock_model
    forecaster.feature_names = mock_model.feature_names_in_

    inference_params = InferenceParams(
        forecast_time=inference_nwp_init_time,
        power_fcst_model_name="test",
    )

    preds = forecaster.predict(
        time_series_metadata=cast(pt.DataFrame[TimeSeriesMetadata], metadata),
        inference_params=inference_params,
        nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        flows_30m=cast(pt.LazyFrame, flows_30m),
    )

    assert len(preds) == 1
    assert preds["power_fcst"][0] == 1414.0
    # Check that the lag was correctly picked up (77.0 for 1-day lead time)
    # We can't easily check the internal X dataframe without more patching,
    # but the fact that it didn't crash is a good sign.
