import polars as pl
from datetime import datetime, timezone
from contracts.data_schemas import PowerTimeSeries


def test_power_time_series_schema_matches_expected():
    # This test verifies that the PowerTimeSeries schema has the expected columns
    # and types, which should be identical to the previous schema.

    # Define a valid DataFrame
    df = pl.DataFrame(
        {
            "time_series_id": [123],
            "period_end_time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
            "power": [10.0],
        }
    ).cast(
        {
            "time_series_id": pl.Int32,
            "period_end_time": pl.Datetime(time_unit="us", time_zone="UTC"),
            "power": pl.Float32,
        }
    )

    # Validate
    validated_df = PowerTimeSeries.validate(df)

    # Assert columns
    expected_columns = ["time_series_id", "period_end_time", "power"]
    assert list(validated_df.columns) == expected_columns

    # Assert types
    assert validated_df.schema["time_series_id"] == pl.Int32
    assert validated_df.schema["period_end_time"] == pl.Datetime(time_unit="us", time_zone="UTC")
    assert validated_df.schema["power"] == pl.Float32
