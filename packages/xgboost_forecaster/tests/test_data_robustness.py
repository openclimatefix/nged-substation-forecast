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
        "wind_direction_10m": [0.0, 63.75, 127.5],  # 0, 90, 180 degrees in 0-255 scale
        "categorical_precipitation_type_surface": [1, 2, 3],
    }
    lf = pl.LazyFrame(data)

    # Process data (should interpolate to 30m steps)
    processed_lf = process_nwp_data(lf, [h3_index])
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

    # Check circular interpolation for wind direction
    # 0.0 (0 deg) and 63.75 (90 deg)
    # Midway should be 45 degrees.
    # In 0-255 scale, 45 deg is 255 * 45 / 360 = 31.875
    mid_row = processed_df.filter(
        pl.col("valid_time") == init_time + timedelta(hours=4, minutes=30)
    )
    mid_wind = mid_row["wind_direction_10m"][0]
    assert mid_wind == pytest.approx(31.875)

    # Check 0/360 boundary interpolation
    # 255.0 (360 deg) and 0.0 (0 deg) - wait, my input doesn't have this.
    # Let's add another test case for boundary.


def test_process_nwp_data_circular_boundary():
    init_time = datetime(2024, 4, 1, 0, 0, tzinfo=timezone.utc)
    h3_index = 12345
    ensemble_member = 0

    data = {
        "init_time": [init_time] * 2,
        "valid_time": [
            init_time + timedelta(hours=3),
            init_time + timedelta(hours=6),
        ],
        "h3_index": [h3_index] * 2,
        "ensemble_member": [ensemble_member] * 2,
        "wind_direction_10m": [240.0, 15.0],  # Near 360 and near 0
    }
    lf = pl.LazyFrame(data)

    processed_lf = process_nwp_data(lf, [h3_index])
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    # Midway between 240 and 15 (across the boundary)
    # 240 is ~338 deg, 15 is ~21 deg.
    # Distance is (255-240) + 15 = 30 units.
    # Midway is 240 + 15 = 255 (0) or 15 - 15 = 0.
    # Wait, 240 + 15 = 255.
    mid_wind = processed_df.filter(
        pl.col("valid_time") == init_time + timedelta(hours=4, minutes=30)
    )["wind_direction_10m"][0]

    # (240/255 * 360) = 338.82
    # (15/255 * 360) = 21.17
    # Midpoint across boundary: (338.82 + 21.17 + 360) / 2 % 360 = 359.995 or 0.
    # In 0-255 scale, this is 0 or 255.
    assert mid_wind == pytest.approx(0.0, abs=1e-2) or mid_wind == pytest.approx(255.0, abs=1e-2)


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

    processed_lf = process_nwp_data(lf, [h3_index])
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

    processed_lf = process_nwp_data(lf, [12345])
    processed_df = cast(pl.DataFrame, processed_lf.collect())

    assert len(processed_df) == 0
