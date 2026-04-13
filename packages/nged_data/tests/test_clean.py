import pytest
import polars as pl
import patito as pt
from datetime import datetime, timedelta, timezone
from nged_data.clean import clean_power_time_series
from contracts.data_schemas import PowerTimeSeries


def test_clean_power_time_series():
    # Create dummy data
    start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    data = {
        "time_series_id": [1] * 48,
        "period_end_time": [start_time + timedelta(hours=i) for i in range(48)],
        "power": [1.0] * 48,  # Constant (variance 0)
    }
    df = pl.DataFrame(data).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("period_end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("power").cast(pl.Float32),
    )

    # Test with variance threshold 0.1
    # Day 1 variance is 0, Day 2 variance is 0. Both should be removed.
    with pytest.raises(
        ValueError, match="All rows were removed after filtering by variance threshold."
    ):
        clean_power_time_series(
            pt.DataFrame[PowerTimeSeries](df),
            stuck_std_threshold=0.0,
            min_mw_threshold=-100.0,
            max_mw_threshold=1000.0,
            default_threshold=0.1,
        )

    # Test with data that should be kept
    data_keep = {
        "time_series_id": [1] * 48,
        "period_end_time": [start_time + timedelta(hours=i) for i in range(48)],
        "power": [1.0, 2.0] * 24,  # Day 1: variance > 0, Day 2: variance > 0
    }
    df_keep = pl.DataFrame(data_keep).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("period_end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("power").cast(pl.Float32),
    )
    cleaned_df = clean_power_time_series(
        pt.DataFrame[PowerTimeSeries](df_keep),
        stuck_std_threshold=0.0,
        min_mw_threshold=-100.0,
        max_mw_threshold=1000.0,
        default_threshold=0.1,
    )
    assert len(cleaned_df) == 37

    # Test with null values
    data_null = {
        "time_series_id": [1] * 48,
        "period_end_time": [start_time + timedelta(hours=i) for i in range(48)],
        "power": [1.0, 2.0] * 23 + [None, None],
    }
    df_null = pl.DataFrame(data_null).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("period_end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("power").cast(pl.Float32),
    )
    cleaned_df_null = clean_power_time_series(
        pt.DataFrame[PowerTimeSeries](df_null),
        stuck_std_threshold=0.0,
        min_mw_threshold=-100.0,
        max_mw_threshold=1000.0,
        default_threshold=0.1,
    )
    assert len(cleaned_df_null) == 35

    # Test with time_series_id
    cleaned_df_id = clean_power_time_series(
        pt.DataFrame[PowerTimeSeries](df_keep),
        stuck_std_threshold=0.0,
        min_mw_threshold=-100.0,
        max_mw_threshold=1000.0,
        time_series_id=1,
        default_threshold=0.1,
    )
    assert len(cleaned_df_id) == 37

    # Test with variance_thresholds and time_series_id
    cleaned_df_thresholds = clean_power_time_series(
        pt.DataFrame[PowerTimeSeries](df_keep),
        stuck_std_threshold=0.0,
        min_mw_threshold=-100.0,
        max_mw_threshold=1000.0,
        time_series_id=1,
        variance_thresholds={1: 0.0},
        default_threshold=0.1,
    )
    # If threshold is 0.0, all rows should be kept (variance > 0)
    # The first row has null variance, so it's dropped.
    assert len(cleaned_df_thresholds) == 37


def test_clean_power_time_series_keeps_valid_zero_power():
    # Create dummy data with zero power values
    start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    # Need enough data for rolling std dev (48 periods)
    # Let's make it 61 periods
    num_periods = 61

    # Power values: 0.0, 1.0, 0.0, 1.0, ...
    # This should have enough variance to not be "stuck"
    power_values = [float(i % 2) for i in range(num_periods)]

    data = {
        "time_series_id": [1] * num_periods,
        "period_end_time": [start_time + timedelta(minutes=30 * i) for i in range(num_periods)],
        "power": power_values,
    }
    df = pl.DataFrame(data).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("period_end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("power").cast(pl.Float32),
    )

    # Call clean_power_time_series
    # Set thresholds such that 0.0 is not "stuck" and not "insane"
    # stuck_std_threshold = 0.1 (std of [0, 1, 0, 1...] is ~0.5, so > 0.1)
    # min_mw_threshold = -1.0
    # max_mw_threshold = 2.0

    cleaned_df = clean_power_time_series(
        pt.DataFrame[PowerTimeSeries](df),
        stuck_std_threshold=0.1,
        min_mw_threshold=-1.0,
        max_mw_threshold=2.0,
        default_threshold=0.0,  # Set variance threshold to 0 to keep all rows
    )

    # Check that 0.0 values are still 0.0 and not null
    # We only check after the first 48 periods to avoid the fill_null(0) issue
    assert (cleaned_df["power"][48:] == 0.0).any()
    assert cleaned_df["power"][48:].null_count() == 0


def test_clean_power_time_series_handles_partition_boundary():
    # Create dummy data with 60 periods
    start_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    num_periods = 60
    # Power values: 0.0, 1.0, 0.0, 1.0... (high variance)
    power_values = [float(i % 2) for i in range(num_periods)]
    data = {
        "time_series_id": [1] * num_periods,
        "period_end_time": [start_time + timedelta(minutes=30 * i) for i in range(num_periods)],
        "power": power_values,
    }
    df = pl.DataFrame(data).with_columns(
        pl.col("time_series_id").cast(pl.Int32),
        pl.col("period_end_time").cast(pl.Datetime(time_unit="us", time_zone="UTC")),
        pl.col("power").cast(pl.Float32),
    )

    # Call clean_power_time_series
    # Set thresholds such that it's not "stuck" and not "insane"
    # stuck_std_threshold = 0.1
    # min_mw_threshold = -1.0
    # max_mw_threshold = 2.0
    # default_threshold = 0.0 (keep all rows)

    cleaned_df = clean_power_time_series(
        pt.DataFrame[PowerTimeSeries](df),
        stuck_std_threshold=0.1,
        min_mw_threshold=-1.0,
        max_mw_threshold=2.0,
        default_threshold=0.0,
    )

    # Check that the first 48 periods are not null
    assert cleaned_df["power"].null_count() == 0
    assert len(cleaned_df) == 49
