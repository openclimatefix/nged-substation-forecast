"""Tests for the pure functions of `studies/solar_bmu_census/fetch_sources.py`.

Each test fails on the bug it exists for: a half-hour time read as local rather than UTC, a window
that keeps both its ends and so duplicates a chunk boundary, and a window that ends inside the
period B1610 has not yet published.
"""

from datetime import UTC, date, datetime
from typing import Any

import fetch_sources
import polars as pl

WINDOW = fetch_sources.Window(
    start=datetime(2026, 6, 1, tzinfo=UTC), end=datetime(2026, 6, 3, tzinfo=UTC)
)


def _row(*, end_time: str, quantity: float) -> dict[str, Any]:
    return {"halfHourEndTime": end_time, "quantity": quantity}


def test_half_hour_end_time_is_parsed_as_utc() -> None:
    frame = fetch_sources.parse_b1610(
        rows=[_row(end_time="2026-06-01T12:00:00", quantity=1.0)], window=WINDOW
    )
    assert frame["half_hour_end_time"][0] == datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    assert frame.schema["half_hour_end_time"] == pl.Datetime("us", "UTC")


def test_the_window_is_half_open_so_a_chunk_boundary_does_not_duplicate() -> None:
    frame = fetch_sources.parse_b1610(
        rows=[
            _row(end_time="2026-06-01T00:00:00", quantity=9.0),
            _row(end_time="2026-06-01T00:30:00", quantity=1.0),
            _row(end_time="2026-06-03T00:00:00", quantity=2.0),
            _row(end_time="2026-06-03T00:30:00", quantity=9.0),
        ],
        window=WINDOW,
    )
    assert frame["output_mwh"].to_list() == [1.0, 2.0]


def test_a_repeated_half_hour_keeps_one_row_and_the_last() -> None:
    frame = fetch_sources.parse_b1610(
        rows=[
            _row(end_time="2026-06-01T12:00:00", quantity=1.0),
            _row(end_time="2026-06-01T12:00:00", quantity=2.0),
        ],
        window=WINDOW,
    )
    assert frame["output_mwh"].to_list() == [2.0]


def test_an_empty_response_gives_an_empty_typed_frame() -> None:
    frame = fetch_sources.parse_b1610(rows=[], window=WINDOW)
    assert frame.height == 0
    assert frame.schema["half_hour_end_time"] == pl.Datetime("us", "UTC")


def test_study_window_is_twelve_complete_months_two_weeks_before_the_run() -> None:
    window = fetch_sources.study_window(today=date(2026, 10, 7))
    assert window.start == datetime(2025, 9, 1, tzinfo=UTC)
    assert window.end == datetime(2026, 9, 1, tzinfo=UTC)
    assert window.expected_half_hours == 365 * 48


def test_study_window_across_a_year_end_runs_back_into_the_previous_year() -> None:
    window = fetch_sources.study_window(today=date(2027, 1, 5))
    assert window.end == datetime(2026, 12, 1, tzinfo=UTC)
    assert window.start == datetime(2025, 12, 1, tzinfo=UTC)


def test_interconnectors_and_unregistered_units_are_not_fetched() -> None:
    reference = [
        {"elexonBmUnit": "T_A", "interconnectorId": None},
        {"elexonBmUnit": "I_B", "interconnectorId": "BRITNED"},
        {"elexonBmUnit": None, "interconnectorId": None},
    ]
    assert fetch_sources.b1610_bmu_ids(reference=reference) == ["T_A"]
