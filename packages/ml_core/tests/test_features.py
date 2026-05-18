import polars as pl
from ml_core.features import build_lag_expr, build_rolling_mean_expr

def test_build_lag_expr():
    df = pl.DataFrame({"power": [10.0, 20.0, 30.0, 40.0]})
    expr = build_lag_expr("power", 2)
    result = df.with_columns(expr)
    
    assert "power_lag_2h" in result.columns
    assert result["power_lag_2h"].to_list() == [None, None, 10.0, 20.0]

def test_build_rolling_mean_expr():
    df = pl.DataFrame({"temperature": [10.0, 20.0, 30.0, 40.0]})
    expr = build_rolling_mean_expr("temperature", 2)
    result = df.with_columns(expr)
    
    assert "temperature_rolling_mean_2h" in result.columns
    assert result["temperature_rolling_mean_2h"].to_list() == [None, 15.0, 25.0, 35.0]
