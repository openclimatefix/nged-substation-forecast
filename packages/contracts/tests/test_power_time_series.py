from datetime import datetime, timezone

import patito as pt
import pytest
from contracts.power_schemas import PowerTimeSeries


def test_power_time_series_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "power": [10.0],
            }
        )
        .set_model(PowerTimeSeries)
        .cast()
    )

    # Should pass
    df.validate()


@pytest.mark.parametrize(
    "data, expected_error",
    [
        # Invalid power (too high)
        (
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "power": [1000.1],
            },
            "power",
        ),
        # Invalid power (too low)
        (
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "power": [-1000.1],
            },
            "power",
        ),
        # Invalid time (not :00 or :30)
        (
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 15, tzinfo=timezone.utc)],
                "power": [10.0],
            },
            "time must be at the top or bottom of the hour",
        ),
        # Invalid time_series_id (string instead of int)
        (
            {
                "time_series_id": ["abc"],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)],
                "power": [10.0],
            },
            "time_series_id",
        ),
        # Duplicate rows
        (
            {
                "time_series_id": [123, 123],
                "time": [
                    datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
                    datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),
                ],
                "power": [10.0, 20.0],
            },
            "Duplicate entries found for",
        ),
    ],
)
def test_power_time_series_invalid_data(data, expected_error):
    # We need to cast to ensure the types are checked
    df = pt.DataFrame(data).set_model(PowerTimeSeries)

    # We expect validation to fail
    with pytest.raises(Exception, match=expected_error):
        df.cast().validate()
