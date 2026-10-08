"""Tests for the pure parts of the BMU notification download script. No test touches the network."""

from datetime import UTC, date, datetime

import polars as pl
import pytest
from fetch_bmu_notifications import (
    POINT_SCHEMA,
    check_no_empty_month,
    half_hourly_means,
    parse_points,
    period_start_table,
)
from market_common import IncompleteChunkError


def _row(*, start: str, stop: str, level_from: float, level_to: float, **extra: object) -> dict:
    return {
        "bmUnit": "E_TESTB-1",
        "timeFrom": f"2026-03-10T{start}:00Z",
        "timeTo": f"2026-03-10T{stop}:00Z",
        "levelFrom": level_from,
        "levelTo": level_to,
        "settlementDate": "2026-03-10",
        "settlementPeriod": 1,
        **extra,
    }


def _points(*rows: dict) -> pl.DataFrame:
    return parse_points(rows=list(rows), day=date(2026, 3, 10))


def _means(*rows: dict) -> pl.DataFrame:
    return half_hourly_means(
        points=_points(*rows),
        period_starts=period_start_table(start=date(2026, 3, 10), end=date(2026, 3, 10)),
    )


def test_parse_points_keeps_only_points_starting_in_the_day() -> None:
    previous_day = _row(start="00:00", stop="00:30", level_from=1, level_to=1)
    previous_day["timeFrom"] = "2026-03-09T23:30:00Z"
    next_day = _row(start="00:00", stop="00:30", level_from=1, level_to=1)
    next_day["timeFrom"] = "2026-03-11T00:00:00Z"
    inside = _row(start="12:00", stop="12:30", level_from=2, level_to=2)
    points = _points(previous_day, inside, next_day)
    assert points["time_from"].to_list() == [datetime(2026, 3, 10, 12, tzinfo=UTC)]
    assert points.schema == pl.Schema(POINT_SCHEMA)


def test_parse_points_keeps_the_highest_notification_sequence() -> None:
    old = _row(start="12:00", stop="12:30", level_from=1, level_to=1, notificationSequence=5)
    new = _row(start="12:00", stop="12:30", level_from=9, level_to=9, notificationSequence=7)
    points = _points(new, old)
    assert points["level_from_mw"].to_list() == [9.0]


def test_parse_points_of_no_rows_has_the_point_schema() -> None:
    assert _points().schema == pl.Schema(POINT_SCHEMA)


def test_constant_point_across_a_window_gives_that_level() -> None:
    means = _means(_row(start="01:00", stop="01:30", level_from=20, level_to=20))
    assert means["mean_level_mw"].to_list() == [20.0]
    assert means["covered_seconds"].to_list() == [1800]
    assert means["time"].to_list() == [datetime(2026, 3, 10, 1, tzinfo=UTC)]


def test_ramp_inside_a_window_gives_the_average_of_the_ends() -> None:
    means = _means(_row(start="01:00", stop="01:30", level_from=0, level_to=100))
    assert means["mean_level_mw"].to_list() == [pytest.approx(50.0)]


def test_point_spanning_two_windows_is_split_by_interpolation() -> None:
    # 0 to 120 MW over 01:00 to 02:00: the first window averages 30 MW, the second 90 MW.
    means = _means(_row(start="01:00", stop="02:00", level_from=0, level_to=120))
    assert means["mean_level_mw"].to_list() == [pytest.approx(30.0), pytest.approx(90.0)]
    assert means["covered_seconds"].to_list() == [1800, 1800]


def test_point_starting_mid_window_is_a_partial_cover() -> None:
    means = _means(
        _row(start="01:10", stop="01:20", level_from=10, level_to=10),
        _row(start="01:20", stop="01:30", level_from=30, level_to=30),
    )
    assert means["covered_seconds"].to_list() == [1200]
    assert means["mean_level_mw"].to_list() == [pytest.approx(20.0)]


def test_zero_length_point_adds_nothing() -> None:
    means = _means(
        _row(start="01:00", stop="01:00", level_from=500, level_to=500),
        _row(start="01:00", stop="01:30", level_from=4, level_to=4),
    )
    assert means["mean_level_mw"].to_list() == [4.0]


def test_window_gets_its_settlement_date_and_period() -> None:
    means = _means(_row(start="01:00", stop="01:30", level_from=1, level_to=1))
    assert means["settlement_date"].to_list() == [date(2026, 3, 10)]
    assert means["settlement_period"].to_list() == [3]


def test_period_numbers_follow_uk_clock_time_in_summer() -> None:
    table = period_start_table(start=date(2026, 7, 1), end=date(2026, 7, 1))
    row = table.filter(pl.col("time") == datetime(2026, 7, 1, 12, tzinfo=UTC))
    assert row["settlement_period"].to_list() == [27]


def test_period_table_on_the_46_period_day() -> None:
    table = period_start_table(start=date(2026, 3, 29), end=date(2026, 3, 29))
    day = table.filter(pl.col("settlement_date") == date(2026, 3, 29))
    assert day.height == 46
    assert day["time"].min() == datetime(2026, 3, 29, 0, tzinfo=UTC)
    assert day["time"].max() == datetime(2026, 3, 29, 22, 30, tzinfo=UTC)


def test_period_table_on_the_50_period_day() -> None:
    table = period_start_table(start=date(2025, 10, 26), end=date(2025, 10, 26))
    day = table.filter(pl.col("settlement_date") == date(2025, 10, 26))
    assert day.height == 50
    assert day["time"].min() == datetime(2025, 10, 25, 23, 0, tzinfo=UTC)
    assert day["time"].max() == datetime(2025, 10, 26, 23, 30, tzinfo=UTC)


def test_a_month_of_empty_days_raises_but_one_empty_day_does_not() -> None:
    keys = [f"2026-03-{day:02d}" for day in range(1, 32)]
    check_no_empty_month(label="x", empty=keys[:1], keys=keys)
    with pytest.raises(IncompleteChunkError):
        check_no_empty_month(label="x", empty=keys, keys=keys)
