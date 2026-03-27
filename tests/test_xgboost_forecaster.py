import pytest
import polars as pl
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from typing import cast

from xgboost_forecaster.data import (
    DataConfig,
    load_nwp_run,
    downsample_power_flows,
    process_nwp_data,
)
from xgboost_forecaster.scaling import load_scaling_params
from xgboost_forecaster.model import XGBoostForecaster
from contracts.hydra_schemas import (
    ModelConfig,
    XGBoostHyperparameters,
    ModelFeaturesConfig,
    NwpModel,
)
import dagster as dg
from src.nged_substation_forecast.defs.xgb_assets import train_xgboost


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
        {"timestamp": ["2026-03-07T00:00:00"], "MW": [1.0], "substation_number": [123]}
    )
    df_power = df_power.with_columns(
        pl.col("timestamp").str.to_datetime().dt.replace_time_zone("UTC"),
        pl.col("substation_number").cast(pl.Int32),
    )
    df_power.write_delta(power_path, delta_write_options={"partition_by": ["substation_number"]})

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
            "wind_speed_10m": [10],
            "wind_direction_10m": [180],
            "wind_speed_100m": [15],
            "wind_direction_100m": [185],
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
            pl.col("wind_speed_10m").cast(pl.UInt8),
            pl.col("wind_direction_10m").cast(pl.UInt8),
            pl.col("wind_speed_100m").cast(pl.UInt8),
            pl.col("wind_direction_100m").cast(pl.UInt8),
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


def test_downsample_power_flows_uses_period_ending_semantics():
    # Create power flows at 10:01, 10:15, 10:29
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 10, 1, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 15, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 29, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 31, tzinfo=timezone.utc),  # Next period
            ],
            "substation_number": [1, 1, 1, 1],
            "MW": [10.0, 20.0, 30.0, 40.0],
            "MVA": [10.0, 20.0, 30.0, 40.0],
        }
    ).lazy()

    res = cast(pl.DataFrame, downsample_power_flows(df).collect())

    # The first three should be aggregated into 10:30
    assert len(res) == 2
    assert res["timestamp"][0] == datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    assert res["MW"][0] == 20.0  # (10+20+30)/3

    # The last one should be aggregated into 11:00
    assert res["timestamp"][1] == datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc)
    assert res["MW"][1] == 40.0


def test_process_nwp_data_removes_zero_lead_time():
    init_time = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    df = pl.DataFrame(
        {
            "init_time": [init_time, init_time, init_time],
            "valid_time": [
                init_time,  # 0-hour
                init_time + timedelta(hours=1),  # 1-hour
                init_time + timedelta(hours=2),  # 2-hour
            ],
            "h3_index": [1, 1, 1],
            "ensemble_member": [0, 0, 0],
            "temperature_2m": [10.0, 12.0, 14.0],
        }
    ).lazy()

    res = cast(pl.DataFrame, process_nwp_data(df, h3_indices=[1]).collect())

    # 0-hour is removed. Remaining valid_times: 01:00, 02:00.
    # Upsampling to 30m will create: 01:00, 01:30, 02:00.
    assert len(res) == 3
    assert res["valid_time"][0] == init_time + timedelta(hours=1)
    assert res["lead_time_hours"][0] == 1.0
    assert res["lead_time_hours"][1] == 1.5
    assert res["lead_time_hours"][2] == 2.0


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
    # 1 row of power flow
    flows = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 3, 10, 30, tzinfo=timezone.utc)],
            "substation_number": [1],
            "MW": [50.0],
            "MVA": [50.0],
        }
    ).lazy()

    # Metadata
    metadata = pl.DataFrame(
        {
            "substation_number": [1],
            "h3_res_5": [1],
        }
    )

    # 4 NWP forecast runs, each with 50 ensemble members
    nwp_rows = []
    valid_time = datetime(2024, 1, 3, 10, 30, tzinfo=timezone.utc)
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
                }
            )
    nwp = pl.DataFrame(nwp_rows).lazy()

    config = ModelConfig(
        power_fcst_model_name="test",
        hyperparameters=XGBoostHyperparameters(),
        features=ModelFeaturesConfig(nwps=[NwpModel.ECMWF_ENS_0_25DEG]),
    )

    forecaster = XGBoostForecaster()

    import patito as pt
    from contracts.data_schemas import SubstationFlows, SubstationMetadata, ProcessedNwp

    with patch("xgboost_forecaster.model.XGBRegressor.fit") as mock_fit:
        forecaster.train(
            config=config,
            substation_power_flows=cast(pt.LazyFrame[SubstationFlows], flows),
            substation_metadata=cast(pt.DataFrame[SubstationMetadata], metadata),
            nwps={NwpModel.ECMWF_ENS_0_25DEG: cast(pt.LazyFrame[ProcessedNwp], nwp)},
        )

        # Get the X dataframe passed to fit
        X = mock_fit.call_args[0][0]

        # 1 power flow row * 4 init_times * 50 ensemble members = 200 rows
        assert X.shape[0] == 200


def test_train_xgboost_asset_filters_to_control_member():
    # Create NWP with members 0, 1, 2
    nwp = pl.DataFrame(
        {
            "valid_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 3,
            "h3_index": [1, 1, 1],
            "ensemble_member": [0, 1, 2],
            "init_time": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 3,
            "temperature_2m": [10.0, 11.0, 12.0],
        }
    ).lazy()

    flows = pl.DataFrame().lazy()
    metadata = pl.DataFrame()

    context = dg.build_asset_context()

    with patch("src.nged_substation_forecast.defs.xgb_assets.train_and_log_model") as mock_train:
        train_xgboost(context, nwp, flows, metadata)

        # Check the nwp passed to train_and_log_model
        passed_nwps = mock_train.call_args[1]["nwps"]
        passed_nwp = passed_nwps[NwpModel.ECMWF_ENS_0_25DEG].collect()

        # Should only contain member 0
        assert len(passed_nwp) == 1
        assert passed_nwp["ensemble_member"][0] == 0
