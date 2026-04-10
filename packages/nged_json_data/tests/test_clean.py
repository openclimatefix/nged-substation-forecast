import pytest
import polars as pl
from datetime import datetime, timedelta, timezone
from nged_json_data.clean import clean_power_data


def test_clean_power_data():
    # Create dummy data
    start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        "time_series_id": ["1"] * 48,
        "end_time": [start_time + timedelta(hours=i) for i in range(48)],
        "value": [1.0] * 48,  # Constant (variance 0)
    }
    df = pl.DataFrame(data).with_columns(
        pl.col("end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("value").cast(pl.Float32),
    )

    # Test with variance threshold 0.1
    # Day 1 variance is 0, Day 2 variance is 0. Both should be removed.
    with pytest.raises(
        ValueError, match="All rows were removed after filtering by variance threshold."
    ):
        clean_power_data(df, default_threshold=0.1)

    # Test with data that should be kept
    data_keep = {
        "time_series_id": ["1"] * 48,
        "end_time": [start_time + timedelta(hours=i) for i in range(48)],
        "value": [1.0, 2.0] * 24,  # Day 1: variance > 0, Day 2: variance > 0
    }
    df_keep = pl.DataFrame(data_keep).with_columns(
        pl.col("end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("value").cast(pl.Float32),
    )
    cleaned_df = clean_power_data(df_keep, default_threshold=0.1)
    assert len(cleaned_df) == 47

    # Test with null values
    data_null = {
        "time_series_id": ["1"] * 48,
        "end_time": [start_time + timedelta(hours=i) for i in range(48)],
        "value": [1.0, 2.0] * 23 + [None, None],
    }
    df_null = pl.DataFrame(data_null).with_columns(
        pl.col("end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("value").cast(pl.Float32),
    )
    cleaned_df_null = clean_power_data(df_null, default_threshold=0.1)
    assert len(cleaned_df_null) == 45

    # Test with time_series_id
    cleaned_df_id = clean_power_data(df_keep, time_series_id=1, default_threshold=0.1)
    assert len(cleaned_df_id) == 47

    # Test with variance_thresholds and time_series_id
    cleaned_df_thresholds = clean_power_data(
        df_keep,
        time_series_id=1,
        variance_thresholds={1: 0.0},
        default_threshold=0.1,
    )
    # If threshold is 0.0, all rows should be kept (variance > 0)
    # The first row has null variance, so it's dropped.
    assert len(cleaned_df_thresholds) == 47
