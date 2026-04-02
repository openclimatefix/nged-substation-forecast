from datetime import datetime, timedelta, timezone
from typing import cast

import polars as pl
import pytest
from xgboost_forecaster.data import process_nwp_data


def test_process_nwp_data_interpolation():
    # Create synthetic NWP data with 3-hour steps
    init_time = datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc)
    h3_index = 12345
    ensemble_member = 0

    data = {
        "init_time": [init_time] * 3,
        "valid_time": [
            init_time + timedelta(hours=3),
            init_time + timedelta(hours=6),
            init_time + timedelta(hours=9),
        ],
        "h3_index": [h3_index] * 3,
        "ensemble_member": [ensemble_member] * 3,
        "temperature_2m": [10.0, 20.0, 30.0],
        "wind_u_10m": [0.0, 10.0, 0.0],
        "wind_v_10m": [10.0, 0.0, -10.0],
        "wind_u_100m": [0.0, 10.0, 0.0],
        "wind_v_100m": [10.0, 0.0, -10.0],
        "categorical_precipitation_type_surface": [1, 2, 3],
    }
    lf = pl.LazyFrame(data)

    # Process data (should interpolate to 30m steps)
    processed_lf = process_nwp_data(lf, [h3_index], target_horizon_hours=0)
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    # Check number of rows: 3h to 9h is 6 hours. 6 hours / 30m = 12 steps + 1 = 13 rows.
    assert len(processed_df) == 13

    # Check temperature interpolation (linear)
    # At 4.5h (midway between 3h and 6h), temp should be 15.0
    mid_temp = processed_df.filter(
        pl.col("valid_time") == init_time + timedelta(hours=4, minutes=30)
    )["temperature_2m"][0]
    assert mid_temp == pytest.approx(15.0)

    # Check categorical forward-fill
    # At 4.5h, it should still be 1 (from 3h)
    mid_cat = processed_df.filter(
        pl.col("valid_time") == init_time + timedelta(hours=4, minutes=30)
    )["categorical_precipitation_type_surface"][0]
    assert mid_cat == 1

    # Check physical wind calculation
    # At 4.5h, wind_u should be 5.0, wind_v should be 5.0
    # Speed = sqrt(5^2 + 5^2) = sqrt(50) = 7.071
    # Direction = (atan2(5, 5) * 180 / pi + 180) % 360 = (45 + 180) % 360 = 225
    mid_row = processed_df.filter(
        pl.col("valid_time") == init_time + timedelta(hours=4, minutes=30)
    )
    mid_speed = mid_row["wind_speed_10m"][0]
    mid_dir = mid_row["wind_direction_10m"][0]
    assert mid_speed == pytest.approx(7.071, abs=1e-3)
    assert mid_dir == pytest.approx(225.0)


def test_process_nwp_data_circular_boundary():
    init_time = datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc)
    h3_index = 12345
    ensemble_member = 0

    # Test boundary by interpolating U/V that cross a direction boundary
    # Point 1: Wind from North (U=0, V=10) -> Direction 180
    # Point 2: Wind from East (U=-10, V=0) -> Direction 270
    data = {
        "init_time": [init_time] * 2,
        "valid_time": [
            init_time + timedelta(hours=3),
            init_time + timedelta(hours=6),
        ],
        "h3_index": [h3_index] * 2,
        "ensemble_member": [ensemble_member] * 2,
        "wind_u_10m": [0.0, -10.0],
        "wind_v_10m": [10.0, 0.0],
        "wind_u_100m": [0.0, -10.0],
        "wind_v_100m": [10.0, 0.0],
    }
    lf = pl.LazyFrame(data)

    processed_lf = process_nwp_data(lf, [h3_index], target_horizon_hours=0)
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    # Midway: U=-5, V=5
    # Direction = (atan2(-5, 5) * 180 / pi + 180) % 360 = (-45 + 180) % 360 = 135
    # Wait, atan2(-5, 5) is -45 degrees. -45 + 180 = 135.
    # Let's re-verify atan2(u, v)
    # u=0, v=10 -> atan2(0, 10) = 0. (0 + 180) % 360 = 180. Correct (from North).
    # u=-10, v=0 -> atan2(-10, 0) = -90. (-90 + 180) % 360 = 90. Correct (from East).
    # Midway: u=-5, v=5 -> atan2(-5, 5) = -45. (-45 + 180) % 360 = 135. Correct (from North-East).

    mid_wind = processed_df.filter(
        pl.col("valid_time") == init_time + timedelta(hours=4, minutes=30)
    )["wind_direction_10m"][0]

    assert mid_wind == pytest.approx(135.0)


def test_process_nwp_data_lead_time_filter():
    init_time = datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc)
    h3_index = 12345
    ensemble_member = 0

    data = {
        "init_time": [init_time] * 4,
        "valid_time": [
            init_time + timedelta(hours=0),  # Should be filtered out (< 3h)
            init_time + timedelta(hours=3),
            init_time + timedelta(hours=6),
            init_time + timedelta(hours=400),  # Should be filtered out (> 336h)
        ],
        "h3_index": [h3_index] * 4,
        "ensemble_member": [ensemble_member] * 4,
        "temperature_2m": [10.0, 20.0, 30.0, 40.0],
    }
    lf = pl.LazyFrame(data)

    processed_lf = process_nwp_data(lf, [h3_index], target_horizon_hours=0)
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    # Only 3h and 6h should remain, plus interpolated steps between them.
    # 3h to 6h is 3 hours = 6 steps + 1 = 7 rows.
    assert len(processed_df) == 7
    assert processed_df["valid_time"].min() == init_time + timedelta(hours=3)
    assert processed_df["valid_time"].max() == init_time + timedelta(hours=6)


def test_process_nwp_data_empty_input():
    lf = pl.LazyFrame(
        schema={
            "init_time": pl.Datetime,
            "valid_time": pl.Datetime,
            "h3_index": pl.UInt64,
            "ensemble_member": pl.UInt8,
            "temperature_2m": pl.Float32,
        }
    )

    processed_lf = process_nwp_data(lf, [12345], target_horizon_hours=0)
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    assert len(processed_df) == 0
