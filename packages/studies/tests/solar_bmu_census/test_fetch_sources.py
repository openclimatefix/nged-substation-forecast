"""Tests for the pure functions of `studies/solar_bmu_census/fetch_sources.py`.

Each test fails on the bug it exists for. Among those bugs are a half-hour time read as local rather
than UTC, a window that keeps both its ends and so duplicates a chunk boundary, and a window that
ends inside the period B1610 has not yet published.
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


def test_the_lag_margin_is_fourteen_days() -> None:
    assert fetch_sources.study_window(today=date(2026, 10, 10)).end == datetime(
        2026, 9, 1, tzinfo=UTC
    )
    assert fetch_sources.study_window(today=date(2026, 10, 15)).end == datetime(
        2026, 10, 1, tzinfo=UTC
    )


def test_parse_b1610_sorts_rows_given_out_of_order() -> None:
    frame = fetch_sources.parse_b1610(
        rows=[
            _row(end_time="2026-06-01T13:00:00", quantity=2.0),
            _row(end_time="2026-06-01T12:00:00", quantity=1.0),
        ],
        window=WINDOW,
    )
    assert frame["output_mwh"].to_list() == [1.0, 2.0]


def test_a_bmu_listed_twice_is_fetched_once() -> None:
    reference = [
        {"elexonBmUnit": "T_B", "interconnectorId": None},
        {"elexonBmUnit": "T_A", "interconnectorId": None},
        {"elexonBmUnit": "T_B", "interconnectorId": None},
    ]
    assert fetch_sources.b1610_bmu_ids(reference=reference) == ["T_A", "T_B"]


def _mapping_row(*, cfd_id: str, bmu_id: str, ended: bool = False) -> dict[str, Any]:
    return {
        "CFD_Id": cfd_id,
        "BMU_Id": bmu_id,
        "Effective_From": "2024-12-18 00:00:00.0000000",
        "Effective_date_to": "2025-01-01 00:00:00.0000000" if ended else "",
    }


def _portfolio_row(*, cfd_id: str, name: str, capacity: str = "30.000") -> dict[str, Any]:
    return {
        "CFD_ID": cfd_id,
        "Name_of_CFD_Unit": name,
        "Technology_Type": "Solar PV",
        "Transmission_or_Distribution_connection": "Distribution",
        "Status": "Live (Post-FIC)",
        "Maximum_Contract_Capacity_MW": capacity,
    }


def test_a_c_bmu_with_one_named_cfd_unit_is_a_single_site() -> None:
    units = fetch_sources.single_site_cfd_bmus(
        mapping=[_mapping_row(cfd_id="AR4-X", bmu_id="C__ONE")],
        portfolio=[_portfolio_row(cfd_id="AR4-X", name="Example Solar Farm", capacity="49.900")],
    )
    assert units == {
        "C__ONE": {
            "cfd_id": "AR4-X",
            "name": "Example Solar Farm",
            "technology": "Solar PV",
            "connection": "Distribution",
            "status": "Live (Post-FIC)",
            "capacity_mw": 49.9,
        }
    }


def test_a_c_bmu_that_carries_two_current_cfd_units_is_not_a_single_site() -> None:
    units = fetch_sources.single_site_cfd_bmus(
        mapping=[
            _mapping_row(cfd_id="A", bmu_id="C__POOL"),
            _mapping_row(cfd_id="B", bmu_id="C__POOL"),
        ],
        portfolio=[_portfolio_row(cfd_id="A", name="One"), _portfolio_row(cfd_id="B", name="Two")],
    )
    assert units == {}


def test_an_ended_mapping_row_does_not_count_towards_a_single_site() -> None:
    units = fetch_sources.single_site_cfd_bmus(
        mapping=[
            _mapping_row(cfd_id="OLD", bmu_id="C__MOVED", ended=True),
            _mapping_row(cfd_id="NEW", bmu_id="C__MOVED"),
        ],
        portfolio=[
            _portfolio_row(cfd_id="OLD", name="Old"),
            _portfolio_row(cfd_id="NEW", name="New"),
        ],
    )
    assert units["C__MOVED"]["cfd_id"] == "NEW"


def test_a_cfd_unit_without_a_name_or_a_portfolio_row_is_not_a_single_site() -> None:
    units = fetch_sources.single_site_cfd_bmus(
        mapping=[
            _mapping_row(cfd_id="NONAME", bmu_id="C__BLANK"),
            _mapping_row(cfd_id="MISSING", bmu_id="C__LOST"),
        ],
        portfolio=[_portfolio_row(cfd_id="NONAME", name="  ")],
    )
    assert units == {}


def test_only_c_bmus_are_considered() -> None:
    units = fetch_sources.single_site_cfd_bmus(
        mapping=[_mapping_row(cfd_id="A", bmu_id="T_WIND-1")],
        portfolio=[_portfolio_row(cfd_id="A", name="A wind farm")],
    )
    assert units == {}


def test_a_blank_contract_capacity_is_none() -> None:
    units = fetch_sources.single_site_cfd_bmus(
        mapping=[_mapping_row(cfd_id="A", bmu_id="C__X")],
        portfolio=[_portfolio_row(cfd_id="A", name="Named", capacity="")],
    )
    assert units["C__X"]["capacity_mw"] is None
