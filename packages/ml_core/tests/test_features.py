import polars as pl
from datetime import datetime, timedelta
from ml_core.features import apply_lag_feature, apply_rolling_mean_feature

def test_apply_lag_feature():
    # Create a dataframe with a missing row (missing 02:00)
    df = pl.DataFrame({
        "time_series_id": [1, 1, 1, 1],
        "valid_time": [
            datetime(2020, 1, 1, 0),
            datetime(2020, 1, 1, 1),
            # Missing 02:00
            datetime(2020, 1, 1, 3),
            datetime(2020, 1, 1, 4),
        ],
        "power": [10.0, 20.0, 40.0, 50.0]
    })
    
    lf = df.lazy()
    result_lf = apply_lag_feature(lf, "power", 2)
    result = result_lf.collect()
    
    assert "power_lag_2h" in result.columns
    
    # Expected behavior:
    # 00:00 -> lag 2h is 22:00 (prev day) -> None
    # 01:00 -> lag 2h is 23:00 (prev day) -> None
    # 03:00 -> lag 2h is 01:00 -> 20.0
    # 04:00 -> lag 2h is 02:00 -> None (because 02:00 is missing!)
    # If we used shift(2), 03:00 would get 10.0 (wrong) and 04:00 would get 20.0 (wrong)
    assert result["power_lag_2h"].to_list() == [None, None, 20.0, None]

def test_apply_rolling_mean_feature():
    df = pl.DataFrame({
        "time_series_id": [1, 1, 1, 1],
        "valid_time": [
            datetime(2020, 1, 1, 0),
            datetime(2020, 1, 1, 1),
            datetime(2020, 1, 1, 2),
            datetime(2020, 1, 1, 3),
        ],
        "temperature": [10.0, 20.0, 30.0, 40.0]
    })
    
    lf = df.lazy()
    result_lf = apply_rolling_mean_feature(lf, "temperature", 2)
    result = result_lf.collect()
    
    assert "temperature_rolling_mean_2h" in result.columns
    # Rolling mean over 2 hours (inclusive of current row, so current and previous)
    # 00:00 -> 10.0
    # 01:00 -> (10+20)/2 = 15.0
    # 02:00 -> (20+30)/2 = 25.0
    # 03:00 -> (30+40)/2 = 35.0
    assert result["temperature_rolling_mean_2h"].to_list() == [10.0, 15.0, 25.0, 35.0]

def test_apply_rolling_mean_feature_with_ensemble():
    df = pl.DataFrame({
        "time_series_id": [1, 1, 1, 1],
        "ensemble_member": [0, 0, 1, 1],
        "valid_time": [
            datetime(2020, 1, 1, 0),
            datetime(2020, 1, 1, 1),
            datetime(2020, 1, 1, 0),
            datetime(2020, 1, 1, 1),
        ],
        "temperature": [10.0, 20.0, 100.0, 200.0]
    })
    
    lf = df.lazy()
    result_lf = apply_rolling_mean_feature(lf, "temperature", 2)
    result = result_lf.collect()
    
    assert "temperature_rolling_mean_2h" in result.columns
    
    # Sort to ensure consistent order for assertion
    result = result.sort(["ensemble_member", "valid_time"])
    
    # Ensemble 0: 10.0, 15.0
    # Ensemble 1: 100.0, 150.0
    assert result["temperature_rolling_mean_2h"].to_list() == [10.0, 15.0, 100.0, 150.0]
