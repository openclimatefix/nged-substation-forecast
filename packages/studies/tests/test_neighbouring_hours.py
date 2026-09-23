from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from studies.neighbouring_hours import with_neighbouring_hours

START = datetime(2025, 6, 1, tzinfo=UTC)


def _hours(*, site: str, values: list[float], first: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": [site] * len(values),
            "time": [START + timedelta(hours=first + index) for index in range(len(values))],
            "value": values,
        }
    )


def _source() -> pl.DataFrame:
    return pl.concat(
        [_hours(site="A", values=[0.0, 1.0, 2.0, 3.0, 4.0]), _hours(site="B", values=[10.0, 11.0])]
    )


def test_each_offset_reads_the_hour_that_far_away():
    frame = _hours(site="A", values=[2.0], first=2).drop("value")

    added = with_neighbouring_hours(
        frame=frame,
        source=_source(),
        columns={"before_2": ("value", -2), "before": ("value", -1), "after": ("value", 1)},
    )

    assert added.select("before_2", "before", "after").row(0) == (0.0, 1.0, 3.0)


def test_a_neighbour_is_missing_only_where_the_source_is():
    # Site A's scored rows skip hour 2, but the source holds it, so hour 3's previous hour is still
    # read; hour 4's next hour is beyond the source, so it is missing.
    frame = pl.concat(
        [_hours(site="A", values=[1.0], first=1), _hours(site="A", values=[3.0, 4.0], first=3)]
    )

    added = with_neighbouring_hours(
        frame=frame, source=_source(), columns={"before": ("value", -1), "after": ("value", 1)}
    )

    assert added["before"].to_list() == [0.0, 2.0, 3.0]
    assert added["after"].to_list() == [2.0, 4.0, None]


def test_one_sites_values_never_reach_another_site():
    frame = _hours(site="B", values=[10.0, 11.0]).drop("value")

    added = with_neighbouring_hours(frame=frame, source=_source(), columns={"after": ("value", 1)})

    assert added["after"].to_list() == [11.0, None]


def test_the_frames_rows_and_order_are_kept():
    frame = pl.DataFrame(
        {
            "site": ["B", "A", "A"],
            "time": [START + timedelta(hours=hours) for hours in (0, 3, 0)],
            "order": [0, 1, 2],
        }
    )

    added = with_neighbouring_hours(frame=frame, source=_source(), columns={"after": ("value", 1)})

    assert added.drop("after").equals(frame)
    assert added["after"].to_list() == [11.0, 4.0, 1.0]


def test_a_source_with_a_repeated_hour_raises():
    source = pl.concat([_source(), _hours(site="A", values=[9.0])])

    with pytest.raises(ValueError, match="more than one row"):
        with_neighbouring_hours(frame=_source(), source=source, columns={"after": ("value", 1)})


def test_a_column_name_already_on_the_frame_raises():
    frame = _hours(site="A", values=[2.0], first=2).rename({"value": "after"})

    with pytest.raises(ValueError, match="after"):
        with_neighbouring_hours(frame=frame, source=_source(), columns={"after": ("value", 1)})
