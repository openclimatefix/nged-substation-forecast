import polars as pl
import pytest
from xgboost_forecaster.data import process_nwp_data
from datetime import datetime, timezone
from typing import cast


def test_process_nwp_data_descaling():
    """Test that process_nwp_data correctly descales UInt8 weather variables."""
    # Create mock NWP data with UInt8 columns (scaled)
    # For temperature_2m, let's say buffered_min = -50, buffered_range = 100
    # uint8_value = 127 -> physical_value approx 0

    df = pl.DataFrame(
        {
            "init_time": [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "valid_time": [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "h3_index": [1, 1],
            "ensemble_member": [0, 0],
            "temperature_2m": [127, 127],  # UInt8
            "wind_u_10m": [127, 127],  # UInt8
            "wind_v_10m": [127, 127],  # UInt8
            "wind_u_100m": [127, 127],  # UInt8
            "wind_v_100m": [127, 127],  # UInt8
            "categorical_precipitation_type_surface": [
                0,
                0,
            ],  # UInt8 (categorical, should stay UInt8)
        }
    ).with_columns(
        [
            pl.col("h3_index").cast(pl.UInt64),
            pl.col("ensemble_member").cast(pl.UInt8),
            pl.col("temperature_2m").cast(pl.UInt8),
            pl.col("wind_u_10m").cast(pl.UInt8),
            pl.col("wind_v_10m").cast(pl.UInt8),
            pl.col("wind_u_100m").cast(pl.UInt8),
            pl.col("wind_v_100m").cast(pl.UInt8),
            pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
        ]
    )

    # We need to mock load_scaling_params to return known values
    # But process_nwp_data calls it internally.
    # Let's see if we can just run it and check if the types changed to Float32.

    processed_lf = process_nwp_data(df.lazy(), h3_indices=[1])
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    # Check types
    assert processed_df.schema["temperature_2m"] == pl.Float32
    assert processed_df.schema["wind_speed_10m"] == pl.Float32
    assert processed_df.schema["categorical_precipitation_type_surface"] == pl.UInt8

    # Check that values are no longer 127 (they should be descaled)
    # The actual value depends on the scaling params in the assets, but it definitely shouldn't be 127.0
    assert processed_df["temperature_2m"][0] != 127.0


def test_process_nwp_data_no_double_descaling():
    """Test that process_nwp_data does not descale already descaled (Float32) data."""
    df = pl.DataFrame(
        {
            "init_time": [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
            ],
            "valid_time": [
                datetime(2024, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "h3_index": [1, 1],
            "ensemble_member": [0, 0],
            "temperature_2m": [15.5, 16.0],  # Float32
            "wind_u_10m": [1.0, 1.1],
            "wind_v_10m": [0.5, 0.6],
            "wind_u_100m": [2.0, 2.1],
            "wind_v_100m": [1.0, 1.1],
            "categorical_precipitation_type_surface": [0, 0],
        }
    ).with_columns(
        [
            pl.col("h3_index").cast(pl.UInt64),
            pl.col("ensemble_member").cast(pl.UInt8),
            pl.col("temperature_2m").cast(pl.Float32),
            pl.col("categorical_precipitation_type_surface").cast(pl.UInt8),
        ]
    )

    processed_lf = process_nwp_data(df.lazy(), h3_indices=[1])
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    # Values should remain roughly the same (modulo interpolation if any, but here we have 1h steps and upsample to 30m)
    # At valid_time 0 and 1h, they should be exactly the same.

    val_0 = processed_df.filter(
        pl.col("valid_time") == datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
    )["temperature_2m"][0]
    assert val_0 == pytest.approx(15.5)
