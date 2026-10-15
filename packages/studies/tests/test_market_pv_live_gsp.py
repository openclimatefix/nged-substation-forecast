"""Tests for the PV_Live GSP download's parsing, windowing, and list check."""

from datetime import UTC, datetime

import fetch_pv_live_gsp
import pytest
from fetch_pv_live_gsp import check_gsp_list, chunk_keys, fetch_chunk
from market_common import IncompleteChunkError


def _body(*, labels: list[str], capacity: str = "100.0") -> dict[str, object]:
    meta = [
        "gsp_id",
        "datetime_gmt",
        "generation_mw",
        "installedcapacity_mwp",
        "capacity_mwp",
        "updated_gmt",
    ]
    rows = [[24, label, "50.0", "110.0", capacity, "2026-05-09T22:50:39Z"] for label in labels]
    return {"meta": meta, "data": rows}


def _day_labels() -> list[str]:
    labels = [f"2019-01-01T{h:02d}:{m:02d}:00Z" for h in range(24) for m in (0, 30)]
    return [*labels[1:], "2019-01-02T00:00:00Z"]


def test_fetch_chunk_shifts_labels_back_and_computes_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch_pv_live_gsp, "get_json", lambda **_: _body(labels=_day_labels()))
    frame = fetch_chunk("24_2019-01-01_2019-01-02")
    assert frame.height == 48
    assert frame["time"].min() == datetime(2019, 1, 1, 0, 0, tzinfo=UTC)
    assert frame["generation_percent_of_capacity"].to_list() == [50.0] * 48


def test_percent_is_null_when_capacity_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fetch_pv_live_gsp, "get_json", lambda **_: _body(labels=_day_labels(), capacity="0")
    )
    frame = fetch_chunk("24_2019-01-01_2019-01-02")
    assert frame["generation_percent_of_capacity"].null_count() == 48


def test_fetch_chunk_raises_when_the_day_is_short(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _body(labels=["2019-01-01T00:30:00Z"])
    monkeypatch.setattr(fetch_pv_live_gsp, "get_json", lambda **_: body)
    with pytest.raises(IncompleteChunkError):
        fetch_chunk("24_2019-01-01_2019-01-02")


def test_chunk_keys_cover_both_gsps_month_by_month() -> None:
    keys = chunk_keys(start=datetime(2019, 1, 15).date(), end=datetime(2019, 2, 10).date())
    assert keys == [
        "298_2019-01-15_2019-02-01",
        "298_2019-02-01_2019-02-11",
        "24_2019-01-15_2019-02-01",
        "24_2019-02-01_2019-02-11",
    ]


def test_check_gsp_list_raises_when_a_name_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = {"data": [[298, "WALP_B", 11], [24, "OTHER", 11]]}
    monkeypatch.setattr(fetch_pv_live_gsp, "get_json", lambda **_: listing)
    with pytest.raises(ValueError, match="24"):
        check_gsp_list()
