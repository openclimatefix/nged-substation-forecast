from datetime import datetime, timedelta, timezone

from typing import cast

import polars as pl
from xgboost_forecaster.features import add_autoregressive_lags


def test_add_autoregressive_lags_prevents_lookahead():
    """Test that add_autoregressive_lags correctly calculates lags to prevent lookahead bias."""
    # Setup: 30-minute timestamps for a single substation
    substation_number = 1
    init_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)

    # Create a range of valid times (lead times)
    valid_times = [init_time + timedelta(hours=h) for h in range(0, 48, 1)]

    df = pl.LazyFrame(
        {
            "substation_number": [substation_number] * len(valid_times),
            "valid_time": valid_times,
            "init_time": [init_time] * len(valid_times),
        }
    )

    # Create historical flows (30m resolution)
    # We need flows for the target_lag_time
    # For lead_time=0, telemetry_delay=24h, lag_days = ceil((0 + 1)/7)*7 = 7 days
    # target_lag_time = valid_time - 7 days

    flow_timestamps = [t - timedelta(days=7) for t in valid_times] + [
        t - timedelta(days=14) for t in valid_times
    ]

    flows_30m = pl.LazyFrame(
        {
            "substation_number": [substation_number] * len(flow_timestamps),
            "timestamp": flow_timestamps,
            "MW_or_MVA": [float(i) for i in range(len(flow_timestamps))],
        }
    )

    # Apply lags
    telemetry_delay_hours = 24
    result = cast(
        pl.DataFrame,
        add_autoregressive_lags(
            df, flows_30m, telemetry_delay_hours=telemetry_delay_hours
        ).collect(),
    )

    # Verify lag_days calculation
    # lead_time_days = (valid_time - init_time) / 24
    # lag_days = ceil((lead_time_days + 24/24) / 7) * 7

    # For lead_time = 0: lag_days = ceil(1/7)*7 = 7
    # For lead_time = 6 days: lag_days = ceil((6+1)/7)*7 = 7
    # For lead_time = 6.5 days: lag_days = ceil((6.5+1)/7)*7 = 14

    # Check lead_time = 0
    row_0 = result.filter(pl.col("valid_time") == init_time)
    assert row_0["lag_days"][0] == 7
    assert row_0["target_lag_time"][0] == init_time - timedelta(days=7)

    # Check lead_time = 6.5 days (156 hours)
    valid_time_6_5d = init_time + timedelta(hours=156)
    # We need to add this to our valid_times if it's not there
    df_long = pl.LazyFrame(
        {
            "substation_number": [substation_number],
            "valid_time": [valid_time_6_5d],
            "init_time": [init_time],
        }
    )

    result_long = cast(
        pl.DataFrame,
        add_autoregressive_lags(
            df_long, flows_30m, telemetry_delay_hours=telemetry_delay_hours
        ).collect(),
    )
    assert result_long["lag_days"][0] == 14
    assert result_long["target_lag_time"][0] == valid_time_6_5d - timedelta(days=14)


def test_add_autoregressive_lags_handles_missing_flows():
    """Test that add_autoregressive_lags handles cases where historical flows are missing."""
    substation_number = 1
    init_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    valid_time = init_time + timedelta(hours=1)

    df = pl.LazyFrame(
        {
            "substation_number": [substation_number],
            "valid_time": [valid_time],
            "init_time": [init_time],
        }
    )

    # Empty flows
    flows_30m = pl.LazyFrame(
        {
            "substation_number": pl.Series([], dtype=pl.Int32),
            "timestamp": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "MW_or_MVA": pl.Series([], dtype=pl.Float32),
        }
    )

    result = cast(pl.DataFrame, add_autoregressive_lags(df, flows_30m).collect())

    assert result["latest_available_weekly_lag"][0] is None
