"""Tests for the system-series download script, with no network access."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import fetch_system_series as fss
import httpx
import polars as pl
import pytest
from fetch_system_series import (
    SERIES,
    Col,
    add_period_time,
    chunk_bounds,
    day_chunks,
    days_with_missing_issues,
    empty_chunks,
    fetch_bod_month,
    fetch_neso_frequency_month,
    fetch_series_chunk,
    frame_from_rows,
    keep_first_pair,
    neso_frequency_urls,
    period_time_params,
    publish_params,
    settlement_date_params,
    summarise_frequency,
)
from market_common import IncompleteChunkError

UTC_TIME = pl.Datetime(time_unit="us", time_zone="UTC")


def test_frame_from_rows_parses_utc_timestamps_and_dates() -> None:
    columns = (
        Col("t", "time", UTC_TIME, ""),
        Col("d", "day", pl.Date, ""),
        Col("v", "value", pl.Float64, ""),
    )
    rows = [{"t": "2026-03-01T00:30:00Z", "d": "2026-03-01", "v": 3}]
    frame = frame_from_rows(rows=rows, columns=columns, what="test")
    assert frame["time"][0] == datetime(2026, 3, 1, 0, 30, tzinfo=UTC)
    assert frame["day"][0] == date(2026, 3, 1)
    assert frame["value"].dtype == pl.Float64


def test_frame_from_rows_missing_field_is_null() -> None:
    frame = frame_from_rows(rows=[{}], columns=(Col("v", "value", pl.Float64, ""),), what="test")
    assert frame["value"].to_list() == [None]


def test_frame_from_rows_rejects_an_error_object() -> None:
    with pytest.raises(TypeError, match="expected a JSON list"):
        frame_from_rows(rows={"title": "validation"}, columns=(), what="test")


def test_chunk_keys_round_trip() -> None:
    keys = day_chunks(start=date(2026, 2, 28), end=date(2026, 3, 1))
    assert keys == ["2026-02-28_2026-03-01", "2026-03-01_2026-03-02"]
    assert chunk_bounds(key=keys[0]) == (date(2026, 2, 28), date(2026, 3, 1))


def test_publish_params_use_utc_day_bounds() -> None:
    params = publish_params(first=date(2026, 3, 1), after=date(2026, 3, 2), extra=[("a", "b")])
    assert params == [
        ("publishDateTimeFrom", "2026-03-01T00:00Z"),
        ("publishDateTimeTo", "2026-03-02T00:00Z"),
        ("a", "b"),
    ]


def test_settlement_date_params_end_on_the_last_included_date() -> None:
    params = settlement_date_params(first=date(2026, 3, 1), after=date(2026, 4, 1), extra=[])
    assert params == [("settlementDateFrom", "2026-03-01"), ("settlementDateTo", "2026-03-31")]


def test_period_time_params_follow_uk_clock_time() -> None:
    params = period_time_params(first=date(2026, 7, 1), after=date(2026, 7, 2), extra=[])
    assert params == [("from", "2026-06-30T23:00Z"), ("to", "2026-07-01T23:00Z")]


def test_add_period_time_on_a_clock_change_day() -> None:
    frame = pl.DataFrame(
        {"settlement_date": [date(2026, 3, 29)] * 2, "settlement_period": [1, 46]},
        schema={"settlement_date": pl.Date, "settlement_period": pl.Int16},
    )
    times = add_period_time(frame=frame)["time"].to_list()
    assert times == [
        datetime(2026, 3, 29, 0, 0, tzinfo=UTC),
        datetime(2026, 3, 29, 22, 30, tzinfo=UTC),
    ]


def test_add_period_time_rejects_a_period_that_does_not_exist() -> None:
    frame = pl.DataFrame(
        {"settlement_date": [date(2026, 3, 29)], "settlement_period": [48]},
        schema={"settlement_date": pl.Date, "settlement_period": pl.Int16},
    )
    with pytest.raises(ValueError, match="does not exist"):
        add_period_time(frame=frame)


def test_summarise_frequency_statistics() -> None:
    times = [datetime(2026, 3, 1, 0, 0, tzinfo=UTC), datetime(2026, 3, 1, 0, 15, tzinfo=UTC)]
    times += [datetime(2026, 3, 1, 0, 30, tzinfo=UTC)]
    raw = pl.DataFrame(
        {"time": times, "frequency_hz": [49.9, 50.1, 50.0]},
        schema={"time": UTC_TIME, "frequency_hz": pl.Float32},
    )
    summary = summarise_frequency(raw=raw, column="frequency_hz")
    assert summary["time"].to_list() == [times[0], times[2]]
    assert summary["sample_count"].to_list() == [2, 1]
    assert summary["frequency_mean_hz"][0] == pytest.approx(50.0, abs=1e-5)
    assert summary["frequency_min_hz"][0] == pytest.approx(49.9, abs=1e-5)
    assert summary["frequency_max_hz"][0] == pytest.approx(50.1, abs=1e-5)
    assert summary["frequency_std_hz"][0] == pytest.approx(0.141421, abs=1e-5)
    assert summary["frequency_std_hz"][1] is None


def test_keep_first_pair_keeps_only_pairs_one_and_minus_one() -> None:
    frame = pl.DataFrame({"pair_id": [-2, -1, 1, 2, 3]}, schema={"pair_id": pl.Int8})
    assert keep_first_pair(frame=frame)["pair_id"].to_list() == [-1, 1]


def _publish_row(*, publish: str, start: str) -> dict[str, object]:
    return {"publishTime": publish, "startTime": start, "generation": 5}


def test_fetch_series_chunk_drops_rows_published_outside_the_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _publish_row(publish="2026-03-01T00:30:00Z", start="2026-03-01T01:00:00Z"),
        _publish_row(publish="2026-03-02T00:00:00Z", start="2026-03-02T01:00:00Z"),
        _publish_row(publish="2026-03-01T00:30:00Z", start="2026-03-01T01:00:00Z"),
    ]
    monkeypatch.setattr(fss, "get_json", lambda **_: rows)
    frame = fetch_series_chunk("2026-03-01_2026-03-02", spec=SERIES["elexon_windfor"])
    assert frame.height == 1
    assert frame["forecast_mw"].to_list() == [5.0]


def test_fetch_series_chunk_raises_on_an_empty_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fss, "get_json", lambda **_: [])
    with pytest.raises(IncompleteChunkError):
        fetch_series_chunk("2026-03-01_2026-03-02", spec=SERIES["elexon_windfor"])


def test_fetch_series_chunk_accepts_an_empty_chunk_for_a_sparse_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fss, "get_json", lambda **_: [])
    frame = fetch_series_chunk("2026-03-01_2026-03-02", spec=SERIES["elexon_syswarn"])
    assert frame.is_empty()
    assert frame.columns == ["publish_time", "warning_type", "warning_text"]


def test_fetch_series_chunk_computes_time_for_a_dataset_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {"settlementDate": "2026-03-01", "settlementPeriod": 2, "id": 1, "cost": 1.0},
        {"settlementDate": "2026-04-01", "settlementPeriod": 1, "id": 2, "cost": 2.0},
    ]
    monkeypatch.setattr(fss, "get_json", lambda **_: rows)
    frame = fetch_series_chunk("2026-03-01_2026-04-01", spec=SERIES["elexon_disbsad"])
    assert frame["time"].to_list() == [datetime(2026, 3, 1, 0, 30, tzinfo=UTC)]
    assert frame.columns[0] == "time"


def test_fetch_bod_month_keeps_only_prices_inside_the_utc_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def row(time_from: str) -> dict[str, object]:
        return {
            "bmUnit": "E_X-1",
            "timeFrom": time_from,
            "timeTo": time_from,
            "settlementDate": "2026-03-01",
            "settlementPeriod": 1,
            "pairId": 1,
            "levelFrom": 1,
            "levelTo": 1,
            "offer": 10.0,
            "bid": 5.0,
        }

    rows = [row("2026-03-01T00:00:00Z"), row("2026-04-01T00:00:00Z")]
    monkeypatch.setattr(fss, "get_json", lambda **_: rows)
    frame = fetch_bod_month("2026-03-01_2026-04-01", batches=[["E_X-1"]])
    assert frame["time"].to_list() == [datetime(2026, 3, 1, tzinfo=UTC)]
    assert frame["pair_id"].dtype == pl.Int8


def test_neso_frequency_urls_maps_year_month_to_url(monkeypatch: pytest.MonkeyPatch) -> None:
    package = {
        "result": {
            "resources": [
                {"url": "https://x/download/fnew-2025-9.csv"},
                {"url": "https://x/download/fnew-2025-10.csv"},
                {"url": "https://x/download/readme.pdf"},
            ]
        }
    }
    monkeypatch.setattr(fss, "get_json", lambda **_: package)
    assert neso_frequency_urls() == {
        "2025-09": "https://x/download/fnew-2025-9.csv",
        "2025-10": "https://x/download/fnew-2025-10.csv",
    }


def test_fetch_neso_frequency_month_returns_a_summary_not_raw_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv = "dtm,f\n2026-03-01 00:00:00,50.0\n2026-03-01 00:00:01,50.2\n2026-04-01 00:00:00,49.0\n"
    response = httpx.Response(200, content=csv.encode())
    monkeypatch.setattr(fss, "get_response", lambda **_: response)
    summary = fetch_neso_frequency_month("2026-03-01_2026-04-01", urls={"2026-03": "u"})
    assert summary.height == 1
    assert summary["sample_count"].to_list() == [2]
    assert summary["frequency_mean_hz"][0] == pytest.approx(50.1)


def test_fetch_neso_frequency_month_with_no_listed_file_is_incomplete() -> None:
    with pytest.raises(IncompleteChunkError):
        fetch_neso_frequency_month("2026-03-01_2026-04-01", urls={})


def test_fetch_bod_month_flags_placeholder_prices_and_keeps_the_raw_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def row(offer: float, bid: float) -> dict[str, object]:
        return {
            "bmUnit": "E_X-1",
            "timeFrom": "2026-03-01T00:00:00Z",
            "timeTo": "2026-03-01T00:30:00Z",
            "settlementDate": "2026-03-01",
            "settlementPeriod": 1,
            "pairId": 1,
            "levelFrom": 1,
            "levelTo": 1,
            "offer": offer,
            "bid": bid,
        }

    rows = [row(105.0, -20.0), row(99999.0, 5.0), row(10.0, -9999.0)]
    monkeypatch.setattr(fss, "get_json", lambda **_: rows)
    frame = fetch_bod_month("2026-03-01_2026-04-01", batches=[["E_X-1"]]).sort("offer_gbp_per_mwh")
    assert frame["price_is_placeholder"].to_list() == [True, False, True]
    assert frame["offer_gbp_per_mwh"].to_list() == [10.0, 105.0, 99999.0]


def test_fetch_bod_month_with_no_rows_is_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fss, "get_json", lambda **_: [])
    with pytest.raises(IncompleteChunkError):
        fetch_bod_month("2026-03-01_2026-04-01", batches=[["E_X-1"]])


def test_fetch_series_chunk_raises_on_an_empty_disbsad_month(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fss, "get_json", lambda **_: [])
    with pytest.raises(IncompleteChunkError):
        fetch_series_chunk("2026-03-01_2026-04-01", spec=SERIES["elexon_disbsad"])


def test_empty_chunks_lists_only_zero_row_files(tmp_path: Path) -> None:
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "full.parquet")
    pl.DataFrame({"a": []}, schema={"a": pl.Int64}).write_parquet(tmp_path / "empty.parquet")
    assert empty_chunks(cache_dir=tmp_path, keys=["full", "empty", "absent"]) == ["empty"]


@pytest.mark.parametrize(
    ("source", "raw_field", "value_column"),
    [
        ("elexon_inddem", "demand", "demand_mw"),
        ("elexon_indgen", "generation", "generation_mw"),
    ],
)
def test_ind_series_read_the_value_from_the_raw_field_each_dataset_uses(
    monkeypatch: pytest.MonkeyPatch, source: str, raw_field: str, value_column: str
) -> None:
    rows = [
        {
            "publishTime": "2026-03-01T00:17:00Z",
            "startTime": "2026-03-01T01:00:00Z",
            "settlementDate": "2026-03-01",
            "settlementPeriod": 3,
            "boundary": boundary,
            raw_field: value,
        }
        for boundary, value in (("N", -100), ("B1", -7))
    ]
    monkeypatch.setattr(fss, "get_json", lambda **_: rows)
    frame = fetch_series_chunk("2026-03-01_2026-03-02", spec=SERIES[source])
    assert frame["boundary"].to_list() == ["N", "B1"]
    assert frame[value_column].to_list() == [-100.0, -7.0]


def test_days_with_missing_issues_lists_only_short_days() -> None:
    times = [datetime(2026, 3, 1, 0, 17, tzinfo=UTC) + timedelta(minutes=30 * i) for i in range(47)]
    short_day = [
        datetime(2026, 3, 2, 0, 17, tzinfo=UTC) + timedelta(minutes=30 * i) for i in range(46)
    ]
    frame = pl.DataFrame({"publish_time": [*times, *times, *short_day]})
    result = days_with_missing_issues(frame=frame, expected_per_day=47)
    assert result["count"] == 1
    assert result["first"] == [{"day": "2026-03-02", "issues": 46}]


def test_days_with_missing_issues_counts_half_hour_slots_not_publish_times() -> None:
    slots = [datetime(2026, 3, 1, 0, 17, tzinfo=UTC) + timedelta(minutes=30 * i) for i in range(46)]
    extra_issue_in_an_existing_slot = datetime(2026, 3, 1, 0, 16, tzinfo=UTC)
    frame = pl.DataFrame({"publish_time": [*slots, extra_issue_in_an_existing_slot]})
    result = days_with_missing_issues(frame=frame, expected_per_day=47)
    assert result["first"] == [{"day": "2026-03-01", "issues": 46}]
