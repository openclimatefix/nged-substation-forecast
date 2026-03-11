import pytest


from xgboost_forecaster.data import (
    DataConfig,
    load_substation_power,
    load_weather_data,
)
from xgboost_forecaster.scaling import load_scaling_params


@pytest.fixture
def data_config(tmp_path):
    power_path = tmp_path / "power_data"
    power_path.mkdir()
    weather_path = tmp_path / "weather_data"
    weather_path.mkdir()

    # Create dummy data files (replace with actual data creation later)
    import polars as pl

    df = pl.DataFrame(
        {"timestamp": ["2026-03-07T00:00:00"], "MW": [1.0], "substation_number": [123]}
    )
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime(), pl.col("substation_number").cast(pl.Int32)
    )
    df.write_delta(power_path, delta_write_options={"partition_by": ["substation_number"]})
    (weather_path / "weather1.parquet").write_text("dummy weather data")

    return DataConfig(
        base_power_path=power_path,
        base_weather_path=weather_path,
    )


def test_load_substation_power_with_config(data_config):
    # Test that the function loads data without errors
    try:
        load_substation_power(123, config=data_config)
    except Exception as e:
        assert False, f"Failed to load substation power: {e}"


def test_load_weather_data_with_config(data_config):
    # Create dummy data
    h3_indices = [1, 2, 3]
    start_date = "2026-03-01"
    end_date = "2026-03-07"

    # Test that the function loads data without errors
    try:
        load_weather_data(h3_indices, start_date, end_date, config=data_config)
    except Exception as e:
        assert False, f"Failed to load weather data: {e}"


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
