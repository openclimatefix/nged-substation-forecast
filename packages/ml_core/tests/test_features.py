import numpy as np
import polars as pl

from ml_core.features import add_cyclical_temporal_features


def test_add_cyclical_temporal_features():
    df = pl.DataFrame(
        {
            "valid_time": [
                "2023-01-01T00:00:00",
                "2023-01-01T12:00:00",
                "2023-07-02T12:00:00",
            ]
        }
    ).with_columns(pl.col("valid_time").str.to_datetime())

    result = add_cyclical_temporal_features(df)

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "day_of_year_sin" in result.columns
    assert "day_of_year_cos" in result.columns
    assert "day_of_week" in result.columns

    # Check hour 0
    assert np.isclose(result["hour_sin"][0], 0.0, atol=1e-6)
    assert np.isclose(result["hour_cos"][0], 1.0, atol=1e-6)

    # Check hour 12
    assert np.isclose(result["hour_sin"][1], 0.0, atol=1e-6)
    assert np.isclose(result["hour_cos"][1], -1.0, atol=1e-6)

    # Check day of week (2023-01-01 is a Sunday, which is 7 in ISO, but Polars dt.weekday() returns 1-7 where 1 is Monday, 7 is Sunday)
    assert result["day_of_week"][0] == 7
