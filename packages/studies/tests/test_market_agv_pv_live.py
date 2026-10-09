"""Tests for the AGV and PV_Live download scripts, with no network access."""

from datetime import UTC, date, datetime

import fetch_agv
import fetch_pv_live
import polars as pl
import pytest
from fetch_agv import tidy
from fetch_pv_live import check_pes_list, chunk_keys, fetch_chunk, parse_pes_rows
from market_common import IncompleteChunkError


def _agv_row(
    *, run: str = "SF", day: str = "20260301", period: str = "1", group: str = "_B"
) -> dict[str, str]:
    return {
        "gsp_group": group,
        "settlement_date": day,
        "settlement_run": run,
        "cdca_run_number": "1",
        "settlement_period": period,
        "estimate_indicator": "T",
        "import_export": "I",
        "take_mwh": "100.5",
        "flow_run_date": "20260401",
    }


def test_tidy_keeps_only_the_sf_run_inside_the_window() -> None:
    raw = pl.DataFrame(
        [
            _agv_row(run="SF"),
            _agv_row(run="R3"),
            _agv_row(run="SF", day="20250801"),
            _agv_row(run="SF", day="20260301", period="2"),
        ]
    )
    frame = tidy(raw=raw, start=date(2026, 3, 1), end=date(2026, 3, 31))
    assert frame["settlement_period"].to_list() == [1, 2]
    assert frame["time"].to_list() == [
        datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
        datetime(2026, 3, 1, 0, 30, tzinfo=UTC),
    ]
    assert frame["take_mwh"].to_list() == [100.5, 100.5]


def test_tidy_raises_when_a_group_date_and_period_repeat() -> None:
    raw = pl.DataFrame([_agv_row(), _agv_row()])
    with pytest.raises(ValueError, match="share a group"):
        tidy(raw=raw, start=date(2026, 3, 1), end=date(2026, 3, 31))


def test_agv_groups_are_the_14_elexon_letters() -> None:
    assert len(fetch_agv.GSP_GROUPS) == 14
    assert "_I" not in fetch_agv.GSP_GROUPS
    assert "_O" not in fetch_agv.GSP_GROUPS


def _pes_body(*, labels: list[str]) -> dict[str, object]:
    return {
        "meta": ["pes_id", "datetime_gmt", "generation_mw", "installedcapacity_mwp", "updated_gmt"],
        "data": [[22, label, 1.5, 2400.0, "2026-08-04T18:13:59Z"] for label in labels],
    }


def test_parse_pes_rows_moves_the_end_label_back_to_the_start_of_the_half_hour() -> None:
    frame = parse_pes_rows(body=_pes_body(labels=["2026-03-01T00:30:00Z"]), pes_id=22)
    assert frame["time"].to_list() == [datetime(2026, 3, 1, 0, 0, tzinfo=UTC)]
    assert frame["gsp_group"].to_list() == ["_L"]


def test_parse_pes_rows_rejects_an_error_body() -> None:
    with pytest.raises(TypeError):
        parse_pes_rows(body={"detail": "bad"}, pes_id=22)


def test_chunk_keys_cover_each_area_and_month() -> None:
    keys = chunk_keys(start=date(2026, 1, 15), end=date(2026, 2, 10))
    assert len(keys) == 4 * 2
    assert "22_2026-02-01_2026-02-11" in keys


def test_fetch_chunk_raises_when_the_month_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _pes_body(labels=["2026-03-01T00:30:00Z"])
    monkeypatch.setattr(fetch_pv_live, "get_json", lambda **_: body)
    with pytest.raises(IncompleteChunkError):
        fetch_chunk("22_2026-03-01_2026-03-02")


def test_fetch_chunk_asks_for_labels_from_the_first_half_hour_to_midnight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, str] = {}

    def fake_get_json(*, url: str, params: list[tuple[str, str]]) -> dict[str, object]:
        seen.update(dict(params))
        labels = [
            f"2026-03-01T{hour:02d}:{minute:02d}:00Z" for hour in range(24) for minute in (0, 30)
        ]
        # Labels 00:30 ... 23:30 of the day, then midnight of the next day.
        return _pes_body(labels=[*labels[1:], "2026-03-02T00:00:00Z"])

    monkeypatch.setattr(fetch_pv_live, "get_json", fake_get_json)
    frame = fetch_chunk("22_2026-03-01_2026-03-02")
    assert seen["start"] == "2026-03-01T00:30:00"
    assert seen["end"] == "2026-03-02T00:00:00"
    assert frame.height == 48
    assert frame["time"].min() == datetime(2026, 3, 1, 0, 0, tzinfo=UTC)


def test_check_pes_list_raises_when_a_letter_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = {"data": [[11, "_B", "x"], [14, "_E", "x"], [21, "_K", "x"], [22, "_H", "x"]]}
    monkeypatch.setattr(fetch_pv_live, "get_json", lambda **_: listing)
    with pytest.raises(ValueError, match="22"):
        check_pes_list()
