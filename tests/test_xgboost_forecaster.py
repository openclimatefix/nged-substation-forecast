import pytest


from xgboost_forecaster.data import (
    DataConfig,
    load_substation_power,
    load_nwp_run,
)
from xgboost_forecaster.scaling import load_scaling_params


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


def test_load_substation_power_with_config(data_config):
    # Test that the function loads data without errors
    load_substation_power(123, config=data_config)


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
