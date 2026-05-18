from datetime import datetime

import polars as pl
import pytest
from ml_core.features import (
    STATIC_FEATURE_REGISTRY,
    apply_lag_feature,
    apply_latest_weekly_lag_feature,
    apply_local_time_features,
    apply_rolling_mean_feature,
    calculate_lead_time,
)


def test_calculate_lead_time():
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 13),
            ],
            "init_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 0),
            ],
        }
    )

    lf = df.lazy()
    result_lf = calculate_lead_time(lf)
    result = result_lf.collect()  # type: ignore

    assert "lead_time_hours" in result.columns
    assert result["lead_time_hours"].to_list() == [12.0, 13.0]


def test_calculate_lead_time_no_init_time():
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2020, 1, 1, 12),
                datetime(2020, 1, 1, 13),
            ]
        }
    )

    lf = df.lazy()
    result_lf = calculate_lead_time(lf)
    result = result_lf.collect()  # type: ignore

    assert "lead_time_hours" not in result.columns


def test_windchill_feature():
    df = pl.DataFrame(
        {
            "temperature_2m": [0.0, -10.0],
            "wind_speed_10m": [5.0, 10.0],  # m/s
        }
    )

    lf = df.lazy().with_columns(STATIC_FEATURE_REGISTRY["windchill"])
    result = lf.collect()  # type: ignore

    assert "windchill" in result.columns
    # Formula: 13.12 + 0.6215 * T - 11.37 * (V ** 0.16) + 0.3965 * T * (V ** 0.16)
    # V is in km/h, so V = wind_speed_10m * 3.6
    # For T=0, V=18: 13.12 + 0 - 11.37 * (18 ** 0.16) + 0 = 13.12 - 11.37 * 1.583 = -4.88
    # For T=-10, V=36: 13.12 - 6.215 - 11.37 * (36 ** 0.16) - 3.965 * (36 ** 0.16) = 6.905 - 15.335 * 1.768 = -20.2
    assert result["windchill"][0] == pytest.approx(-4.88, abs=0.1)
    assert result["windchill"][1] == pytest.approx(-20.3, abs=0.1)


def test_temperature_2m_trend_6h_feature():
    df = pl.DataFrame({"temperature_2m": [10.0, 15.0], "temperature_2m_6h_ago": [5.0, 20.0]})

    lf = df.lazy().with_columns(STATIC_FEATURE_REGISTRY["temperature_2m_trend_6h"])
    result = lf.collect()  # type: ignore

    assert "temperature_2m_trend_6h" in result.columns
    assert result["temperature_2m_trend_6h"].to_list() == [5.0, -5.0]


def test_apply_lag_feature():
    # Create a dataframe with a missing row (missing 02:00)
    df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1, 1],
            "valid_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 1),
                # Missing 02:00
                datetime(2020, 1, 1, 3),
                datetime(2020, 1, 1, 4),
            ],
            "power": [10.0, 20.0, 40.0, 50.0],
        }
    )

    lf = df.lazy()
    result_lf = apply_lag_feature(lf, "power", 2)
    result = result_lf.collect()  # type: ignore

    assert "power_lag_2h" in result.columns

    # Expected behavior:
    # 00:00 -> lag 2h is 22:00 (prev day) -> None
    # 01:00 -> lag 2h is 23:00 (prev day) -> None
    # 03:00 -> lag 2h is 01:00 -> 20.0
    # 04:00 -> lag 2h is 02:00 -> None (because 02:00 is missing!)
    # If we used shift(2), 03:00 would get 10.0 (wrong) and 04:00 would get 20.0 (wrong)
    assert result["power_lag_2h"].to_list() == [None, None, 20.0, None]


def test_apply_rolling_mean_feature():
    df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1, 1],
            "valid_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 1),
                datetime(2020, 1, 1, 2),
                datetime(2020, 1, 1, 3),
            ],
            "temperature": [10.0, 20.0, 30.0, 40.0],
        }
    )

    lf = df.lazy()
    result_lf = apply_rolling_mean_feature(lf, "temperature", 2)
    result = result_lf.collect()  # type: ignore

    assert "temperature_rolling_mean_2h" in result.columns
    # Rolling mean over 2 hours (inclusive of current row, so current and previous)
    # 00:00 -> 10.0
    # 01:00 -> (10+20)/2 = 15.0
    # 02:00 -> (20+30)/2 = 25.0
    # 03:00 -> (30+40)/2 = 35.0
    assert result["temperature_rolling_mean_2h"].to_list() == [10.0, 15.0, 25.0, 35.0]


def test_apply_rolling_mean_feature_with_ensemble():
    df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1, 1],
            "ensemble_member": [0, 0, 1, 1],
            "valid_time": [
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 1),
                datetime(2020, 1, 1, 0),
                datetime(2020, 1, 1, 1),
            ],
            "temperature": [10.0, 20.0, 100.0, 200.0],
        }
    )

    lf = df.lazy()
    result_lf = apply_rolling_mean_feature(lf, "temperature", 2)
    result = result_lf.collect()  # type: ignore

    assert "temperature_rolling_mean_2h" in result.columns

    # Sort to ensure consistent order for assertion
    result = result.sort(["ensemble_member", "valid_time"])

    # Ensemble 0: 10.0, 15.0
    # Ensemble 1: 100.0, 150.0
    assert result["temperature_rolling_mean_2h"].to_list() == [10.0, 15.0, 100.0, 150.0]


def test_apply_dynamic_lag_feature():
    # Main dataframe with lead times
    df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "valid_time": [
                datetime(2020, 1, 15, 12),  # Lead time 10h -> lag 168h (7 days)
                datetime(2020, 1, 15, 12),  # Lead time 200h -> lag 336h (14 days)
                datetime(2020, 1, 15, 12),  # Lead time 400h -> lag 504h (21 days)
            ],
            "lead_time_hours": [10.0, 200.0, 400.0],
        }
    )

    # Source dataframe with historical data
    source_df = pl.DataFrame(
        {
            "time_series_id": [1, 1, 1],
            "valid_time": [
                datetime(2020, 1, 8, 12),  # 7 days ago
                datetime(2020, 1, 1, 12),  # 14 days ago
                datetime(2019, 12, 25, 12),  # 21 days ago
            ],
            "power": [100.0, 200.0, 300.0],
        }
    )

    result_lf = apply_latest_weekly_lag_feature(
        df.lazy(), source_df.lazy(), "power", "latest_weekly_lagged_power", ["time_series_id"]
    )
    result = result_lf.collect()  # type: ignore

    assert "latest_weekly_lagged_power" in result.columns
    assert result["latest_weekly_lagged_power"].to_list() == [100.0, 200.0, 300.0]


def test_apply_dynamic_lag_feature_no_lead_time():
    # Main dataframe without lead times
    df = pl.DataFrame(
        {
            "time_series_id": [1],
            "valid_time": [
                datetime(2020, 1, 15, 12),
            ],
        }
    )

    # Source dataframe with historical data
    source_df = pl.DataFrame(
        {
            "time_series_id": [1],
            "valid_time": [
                datetime(2020, 1, 8, 12),  # 7 days ago
            ],
            "power": [100.0],
        }
    )

    result_lf = apply_latest_weekly_lag_feature(
        df.lazy(), source_df.lazy(), "power", "latest_weekly_lagged_power", ["time_series_id"]
    )
    result = result_lf.collect()  # type: ignore

    assert "latest_weekly_lagged_power" in result.columns
    assert result["latest_weekly_lagged_power"].to_list() == [100.0]


def test_apply_local_time_features():
    # Test with a winter date (GMT) and a summer date (BST)
    df = pl.DataFrame(
        {
            "valid_time": [
                datetime(2023, 1, 10, 12, 0),  # Winter, UTC=GMT
                datetime(2023, 7, 10, 12, 0),  # Summer, UTC=BST-1
            ]
        }
    )

    result_lf = apply_local_time_features(df.lazy())
    result = result_lf.collect()  # type: ignore

    assert "local_utc_offset" in result.columns
    assert result["local_utc_offset"].to_list() == [0.0, 1.0]

    # In winter, 12:00 UTC is 12:00 local.
    # In summer, 12:00 UTC is 13:00 local.
    # local_time_of_day_sin for 12:00 is sin(pi) = 0
    # local_time_of_day_sin for 13:00 is sin(13/24 * 2pi) = -0.2588
    assert result["local_time_of_day_sin"][0] == pytest.approx(0.0, abs=1e-5)
    assert result["local_time_of_day_sin"][1] == pytest.approx(-0.258819, abs=1e-5)

    assert "local_day_of_week" in result.columns
    # 2023-01-10 is a Tuesday (2)
    # 2023-07-10 is a Monday (1)
    assert result["local_day_of_week"].to_list() == ["Tuesday", "Monday"]
