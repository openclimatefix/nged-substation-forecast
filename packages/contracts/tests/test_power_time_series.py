from datetime import UTC, datetime
from typing import Any

import patito as pt
import polars as pl
import pytest
from contracts.common import MAX_PLAUSIBLE_DATETIME, MIN_PLAUSIBLE_DATETIME
from contracts.power_schemas import PowerTimeSeries


def test_power_time_series_validation():
    # Valid
    df = (
        pt.DataFrame(
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
                "power": [10.0],
            }
        )
        .set_model(PowerTimeSeries)
        .cast()
    )

    # Should pass
    df.validate()


@pytest.mark.parametrize(
    ("data", "expected_error"),
    [
        # Invalid power (too high)
        (
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
                "power": [1000.1],
            },
            "power",
        ),
        # Invalid power (too low)
        (
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
                "power": [-1000.1],
            },
            "power",
        ),
        # Invalid time (not :00 or :30)
        (
            {
                "time_series_id": [123],
                "time": [datetime(2026, 1, 1, 0, 15, tzinfo=UTC)],
                "power": [10.0],
            },
            "time must be at the top or bottom of the hour",
        ),
        # Invalid time_series_id (string instead of int)
        (
            {
                "time_series_id": ["abc"],
                "time": [datetime(2026, 1, 1, 0, 30, tzinfo=UTC)],
                "power": [10.0],
            },
            "time_series_id",
        ),
        # Duplicate rows
        (
            {
                "time_series_id": [123, 123],
                "time": [
                    datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
                    datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
                ],
                "power": [10.0, 20.0],
            },
            "Duplicate entries found for",
        ),
    ],
)
def test_power_time_series_invalid_data(data: dict[str, list[Any]], expected_error: str):
    # We need to cast to ensure the types are checked
    df = pt.DataFrame(data).set_model(PowerTimeSeries)

    # We expect validation to fail
    with pytest.raises(Exception, match=expected_error):
        df.cast().validate()


def _one_row(time: datetime) -> pt.DataFrame[PowerTimeSeries]:
    """A single PowerTimeSeries row at the given time, valid in every respect except `time`.

    Every `time` passed in below is on :00 or :30, so the only rule an out-of-range case can break
    is the range rule — the assertion cannot be satisfied by the alignment check firing instead.
    """
    return pt.DataFrame({"time_series_id": [123], "time": [time], "power": [10.0]}).set_model(
        PowerTimeSeries
    )


@pytest.mark.parametrize(
    ("time", "expected_error"),
    [
        # Pre-modern: `Europe/London` ran on local mean time (UTC-0:01:15) until 1847, so a
        # timestamp like this makes every local-time feature nonsensical.
        (
            datetime(1840, 6, 1, 0, 30, tzinfo=UTC),
            "before MIN_PLAUSIBLE_DATETIME",
        ),
        # The last half-hour before the lower bound: the bound itself is inclusive, this is not.
        (
            datetime(1999, 12, 31, 23, 30, tzinfo=UTC),
            "before MIN_PLAUSIBLE_DATETIME",
        ),
        # Far future — how a Unix-epoch value in milliseconds read as seconds shows up.
        (
            datetime(3000, 6, 1, 0, 30, tzinfo=UTC),
            "after MAX_PLAUSIBLE_DATETIME",
        ),
        (
            datetime(2100, 1, 1, 0, 30, tzinfo=UTC),
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
        datetime(2026, 1, 1, 0, 30, tzinfo=UTC),  # An ordinary reading.
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
    assert datetime(2015, 1, 1, tzinfo=UTC) > MIN_PLAUSIBLE_DATETIME
    assert datetime(1900, 1, 1, tzinfo=UTC) < MIN_PLAUSIBLE_DATETIME
    assert datetime(2050, 1, 1, tzinfo=UTC) < MAX_PLAUSIBLE_DATETIME


def test_power_time_series_empty_frame_passes_bounds_check() -> None:
    """An empty frame has no timestamps to reject, so the range check must not raise on it."""
    PowerTimeSeries.DataFrame(schema=PowerTimeSeries.dtypes).validate()


_OK = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
"""A plausible, aligned `time` — the row that must survive every case below."""

_OK2 = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
"""A second survivor, distinct from `_OK` so that assertions on survivor order can fail."""

_OUT_OF_RANGE = datetime(1840, 6, 1, 0, 30, tzinfo=UTC)
_MISALIGNED = datetime(2026, 1, 1, 0, 15, tzinfo=UTC)


def _frame(times: list[datetime | None]) -> pl.DataFrame:
    """A `PowerTimeSeries`-shaped frame carrying `times`, cast but not yet validated."""
    return pl.DataFrame(
        {
            "time_series_id": pl.Series(
                [123] * len(times), dtype=PowerTimeSeries.dtypes["time_series_id"]
            ),
            "time": pl.Series(times, dtype=PowerTimeSeries.dtypes["time"]),
            "power": pl.Series([10.0] * len(times), dtype=PowerTimeSeries.dtypes["power"]),
        }
    )


@pytest.mark.parametrize(
    ("times", "expected_survivors"),
    [
        pytest.param([_OK2, _OK], [_OK2, _OK], id="all_well_formed"),
        pytest.param([_OUT_OF_RANGE, _OK], [_OK], id="out_of_range_time"),
        pytest.param([_MISALIGNED, _OK], [_OK], id="misaligned_time"),
        pytest.param([_OK2, _OUT_OF_RANGE, _MISALIGNED, _OK], [_OK2, _OK], id="both_kinds_in_one"),
        # A null `time` is malformed this early (the schema declares it non-nullable) and must be
        # dropped *and counted*, never silently vanish from the row accounting: `dt.minute()` is
        # null for a null `time`, and `.filter()` drops a row on a null predicate.
        pytest.param([None, _OK], [_OK], id="null_time"),
        pytest.param([], [], id="empty_frame"),
    ],
)
def test_drop_implausible_rows(
    times: list[datetime | None], expected_survivors: list[datetime]
) -> None:
    """Malformed `time`s are dropped and counted; well-formed rows survive, in order.

    `n_dropped` is asserted against the case's own arithmetic on every case, so a row cannot go
    missing without being reported. (Asserting `survivors.height + n_dropped == df.height` instead
    would be a tautology: that difference is how `n_dropped` is computed.)
    """
    df = _frame(times)

    survivors, n_dropped = PowerTimeSeries.drop_implausible_rows(df)

    assert survivors["time"].to_list() == expected_survivors
    assert n_dropped == len(times) - len(expected_survivors)


def test_drop_implausible_rows_leaves_a_validatable_frame() -> None:
    """The survivors are exactly what `validate` — still strict — then accepts."""
    survivors, _ = PowerTimeSeries.drop_implausible_rows(
        _frame([_OUT_OF_RANGE, _MISALIGNED, None, _OK])
    )

    PowerTimeSeries.validate(survivors)
