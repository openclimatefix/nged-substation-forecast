from datetime import datetime, timezone

import patito as pt
import pytest
from contracts.common import MAX_PLAUSIBLE_DATETIME, MIN_PLAUSIBLE_DATETIME
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


def _one_row(time: datetime) -> pt.DataFrame:
    """A single otherwise-valid PowerTimeSeries row at the given time."""
    return pt.DataFrame({"time_series_id": [123], "time": [time], "power": [10.0]}).set_model(
        PowerTimeSeries
    )


@pytest.mark.parametrize(
    "time, expected_error",
    [
        # Pre-modern: `Europe/London` ran on local mean time (UTC-0:01:15) until 1847, so a
        # timestamp like this makes every local-time feature nonsensical.
        (
            datetime(1840, 6, 1, 0, 30, tzinfo=timezone.utc),
            "before MIN_PLAUSIBLE_DATETIME",
        ),
        # One microsecond below the lower bound: the bound itself is inclusive.
        (
            datetime(1999, 12, 31, 23, 59, 59, 999_999, tzinfo=timezone.utc),
            "before MIN_PLAUSIBLE_DATETIME",
        ),
        # Far future — how a Unix-epoch value in milliseconds read as seconds shows up.
        (
            datetime(3000, 6, 1, 0, 30, tzinfo=timezone.utc),
            "after MAX_PLAUSIBLE_DATETIME",
        ),
        (
            datetime(2100, 1, 1, 0, 30, tzinfo=timezone.utc),
            "after MAX_PLAUSIBLE_DATETIME",
        ),
    ],
)
def test_power_time_series_rejects_out_of_range_time(time: datetime, expected_error: str) -> None:
    """An out-of-range timestamp must be rejected, naming the column and the bound it broke."""
    with pytest.raises(ValueError, match=expected_error) as exc_info:
        _one_row(time).cast().validate()

    assert "`time` is outside the plausible datetime range" in str(exc_info.value)


@pytest.mark.parametrize(
    "time",
    [
        MIN_PLAUSIBLE_DATETIME,  # The bounds are inclusive at both ends.
        MAX_PLAUSIBLE_DATETIME,
        datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc),  # An ordinary reading.
    ],
)
def test_power_time_series_accepts_in_range_time(time: datetime) -> None:
    _one_row(time).cast().validate()


def test_power_time_series_bounds_span_all_plausible_nged_data() -> None:
    """The bounds must be wide enough never to reject real NGED telemetry.

    NGED's trial-area feed starts in the 2020s and this project runs for a handful of years, so
    the range has to comfortably contain that. It must also start well after 1847, when
    ``Europe/London`` stopped running on local mean time at UTC-0:01:15.
    """
    assert MIN_PLAUSIBLE_DATETIME < datetime(2015, 1, 1, tzinfo=timezone.utc)
    assert MIN_PLAUSIBLE_DATETIME > datetime(1900, 1, 1, tzinfo=timezone.utc)
    assert MAX_PLAUSIBLE_DATETIME > datetime(2050, 1, 1, tzinfo=timezone.utc)


def test_power_time_series_empty_frame_passes_bounds_check() -> None:
    """An empty frame has no timestamps to reject, so the range check must not raise on it."""
    PowerTimeSeries.DataFrame(schema=PowerTimeSeries.dtypes).validate()
