"""Tests for the pure parts of the GB price and BMU dispatch download scripts.

No test touches the network. The HTTP-facing functions are exercised through `fetch_missing_chunks`
with fake fetchers.
"""

from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx
import polars as pl
import pytest
from fetch_bmu_dispatch import (
    battery_hint,
    bmu_batches,
    build_candidates,
    list_hash,
    month_chunks,
    parse_boalf,
    parse_boav,
    parse_ebocf,
    parse_reference,
    source_gaps,
    status_chunks,
)
from fetch_gb_prices import (
    apx_lag_correlations,
    carbon_chunks,
    check_n2ex_hours,
    parse_carbon,
    parse_mid,
    parse_n2ex,
    parse_system_prices,
)
from market_common import (
    IncompleteChunkError,
    expected_half_hour_starts,
    expected_hour_starts,
    expected_settlement_starts,
    fetch_missing_chunks,
    period_start_utc,
    period_time_mismatches,
    periods_in_settlement_day,
    read_chunks,
    require_complete,
    summarise_gaps,
    write_readme,
)

SPRING_FORWARD = date(2026, 3, 29)
FALL_BACK = date(2025, 10, 26)


def test_period_counts_on_clock_change_days() -> None:
    assert periods_in_settlement_day(settlement_date=date(2025, 10, 1)) == 48
    assert periods_in_settlement_day(settlement_date=SPRING_FORWARD) == 46
    assert periods_in_settlement_day(settlement_date=FALL_BACK) == 50


def test_period_start_in_summer_time_is_an_hour_before_local_midnight() -> None:
    start = period_start_utc(settlement_date=date(2025, 10, 1), settlement_period=1)
    assert start == datetime(2025, 9, 30, 23, 0, tzinfo=UTC)
    last = period_start_utc(settlement_date=date(2025, 10, 1), settlement_period=48)
    assert last == datetime(2025, 10, 1, 22, 30, tzinfo=UTC)


def test_period_start_on_the_46_period_day() -> None:
    first = period_start_utc(settlement_date=SPRING_FORWARD, settlement_period=1)
    last = period_start_utc(settlement_date=SPRING_FORWARD, settlement_period=46)
    assert first == datetime(2026, 3, 29, 0, 0, tzinfo=UTC)
    assert last == datetime(2026, 3, 29, 22, 30, tzinfo=UTC)


def test_period_start_on_the_50_period_day() -> None:
    first = period_start_utc(settlement_date=FALL_BACK, settlement_period=1)
    last = period_start_utc(settlement_date=FALL_BACK, settlement_period=50)
    assert first == datetime(2025, 10, 25, 23, 0, tzinfo=UTC)
    assert last == datetime(2025, 10, 26, 23, 30, tzinfo=UTC)


def test_period_that_does_not_exist_raises() -> None:
    with pytest.raises(ValueError, match="does not exist"):
        period_start_utc(settlement_date=SPRING_FORWARD, settlement_period=47)
    with pytest.raises(ValueError, match="does not exist"):
        period_start_utc(settlement_date=date(2025, 10, 1), settlement_period=0)


def test_expected_settlement_starts_are_unique_and_half_hourly() -> None:
    starts = expected_settlement_starts(start=date(2025, 10, 25), end=date(2025, 10, 27))
    assert len(starts) == 48 + 50 + 48
    assert len(set(starts)) == len(starts)
    assert {b - a for a, b in pairwise(starts)} == {timedelta(minutes=30)}


def test_expected_row_counts_for_the_whole_window() -> None:
    start, end = date(2025, 9, 1), date(2026, 9, 30)
    days = (end - start).days + 1
    assert len(expected_half_hour_starts(start=start, end=end)) == 48 * days
    assert len(expected_hour_starts(start=start, end=end)) == 24 * days
    assert len(expected_settlement_starts(start=start, end=end)) == 48 * days


def test_summarise_gaps_counts_missing_and_unexpected() -> None:
    expected = expected_half_hour_starts(start=date(2025, 10, 1), end=date(2025, 10, 1))
    present = [stamp for stamp in expected if stamp != expected[5]]
    extra = datetime(2030, 1, 1, tzinfo=UTC)
    summary = summarise_gaps(
        expected=expected,
        actual=pl.Series([*present, present[0], extra], dtype=pl.Datetime("us", "UTC")),
    )
    assert summary["expected_rows"] == 48
    assert summary["distinct_rows"] == 48
    assert summary["missing_count"] == 1
    assert summary["unexpected_count"] == 1
    assert summary["missing_first"] == [expected[5].isoformat()]


N2EX_CSV = (
    "Date,Delivery Period,Price\n"
    + "".join(
        f"2025-10-26,{hour:02d}:00 - {(hour + 1) % 24:02d}:00,{hour}.5\n" for hour in range(24)
    )
    + "".join(
        f"2025-10-27,{hour:02d}:00 - {(hour + 1) % 24:02d}:00,{hour}.25\n" for hour in range(24)
    )
    + "2025-10-25,23:00 - 00:00,99.0\n"
)


def test_n2ex_hour_mapping_treats_the_grid_as_utc() -> None:
    frame = parse_n2ex(csv_text=N2EX_CSV, start=date(2025, 10, 26), end=date(2025, 10, 27))
    assert frame.height == 48
    assert frame["time"][0] == datetime(2025, 10, 26, 0, 0, tzinfo=UTC)
    assert frame["time"][23] == datetime(2025, 10, 26, 23, 0, tzinfo=UTC)
    assert frame["time"][24] == datetime(2025, 10, 27, 0, 0, tzinfo=UTC)
    assert frame["price_gbp_per_mwh"][23] == 23.5
    check = check_n2ex_hours(frame=frame, start=date(2025, 10, 26), end=date(2025, 10, 27))
    assert check == {
        "days_without_24_rows": 0,
        "rows_on_clock_change_days": {"2025-10-26": 24},
        "hourly_series_unbroken": True,
        "hours_priced_exactly_zero": 0,
    }


def test_n2ex_check_catches_a_missing_hour() -> None:
    frame = parse_n2ex(csv_text=N2EX_CSV, start=date(2025, 10, 26), end=date(2025, 10, 27))
    check = check_n2ex_hours(frame=frame.slice(1), start=date(2025, 10, 26), end=date(2025, 10, 27))
    assert check["days_without_24_rows"] == 1


def test_n2ex_rejects_a_dump_without_the_expected_columns() -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        parse_n2ex(
            csv_text="Date,Price\n2025-10-26,1.0\n",
            start=date(2025, 10, 26),
            end=date(2025, 10, 26),
        )


def _system_price_row(*, period: int, start: str) -> dict[str, object]:
    return {
        "startTime": start,
        "settlementDate": "2025-10-01",
        "settlementPeriod": period,
        "systemSellPrice": 70.0,
        "systemBuyPrice": 71.0,
        "netImbalanceVolume": -3.5,
        "totalAcceptedOfferVolume": 1800.0,
        "totalAcceptedBidVolume": -1801.0,
        "priceDerivationCode": "N",
    }


def test_parse_system_prices_keeps_utc_start_and_sorts() -> None:
    rows = [
        _system_price_row(period=2, start="2025-09-30T23:30:00Z"),
        _system_price_row(period=1, start="2025-09-30T23:00:00Z"),
    ]
    frame = parse_system_prices(rows=rows)
    assert frame["settlement_period"].to_list() == [1, 2]
    assert frame["time"][0] == datetime(2025, 9, 30, 23, 0, tzinfo=UTC)
    assert frame["system_buy_price_gbp_per_mwh"][0] == 71.0


def test_parse_mid_drops_the_row_at_the_inclusive_end_and_duplicates() -> None:
    def row(start: str) -> dict[str, object]:
        return {
            "startTime": start,
            "settlementDate": "2025-10-01",
            "settlementPeriod": 3,
            "price": 65.5,
            "volume": 1900.0,
        }

    rows = [row("2025-10-01T00:00:00Z"), row("2025-10-01T00:00:00Z"), row("2025-10-02T00:00:00Z")]
    frame = parse_mid(
        rows=rows,
        start=datetime(2025, 10, 1, tzinfo=UTC),
        end=datetime(2025, 10, 2, tzinfo=UTC),
    )
    assert frame.height == 1


def test_parse_carbon_reads_minute_precision_timestamps() -> None:
    rows = [
        {
            "from": "2025-10-01T00:00Z",
            "to": "2025-10-01T00:30Z",
            "intensity": {"forecast": 130, "actual": None, "index": "moderate"},
        }
    ]
    frame = parse_carbon(
        rows=rows, start=datetime(2025, 10, 1, tzinfo=UTC), end=datetime(2025, 10, 2, tzinfo=UTC)
    )
    assert frame["time_end"][0] == datetime(2025, 10, 1, 0, 30, tzinfo=UTC)
    assert frame["actual_gco2_per_kwh"][0] is None


def test_carbon_chunks_cover_the_window_without_overlap() -> None:
    chunks = carbon_chunks(start=date(2025, 9, 1), end=date(2025, 10, 5))
    assert chunks[0] == (date(2025, 9, 1), date(2025, 9, 15))
    assert chunks[-1][1] == date(2025, 10, 6)
    assert all(a[1] == b[0] for a, b in pairwise(chunks))


def test_month_chunks_split_at_month_boundaries() -> None:
    assert month_chunks(start=date(2025, 9, 15), end=date(2025, 11, 10)) == [
        "2025-09-15_2025-10-01",
        "2025-10-01_2025-11-01",
        "2025-11-01_2025-11-11",
    ]


def test_status_chunks_are_at_most_30_days() -> None:
    keys = status_chunks(start=date(2025, 9, 1), end=date(2025, 11, 5))
    assert keys[0] == "2025-09-01_2025-09-30"
    assert keys[-1].endswith("2025-11-05")


def _reference() -> pl.DataFrame:
    def entry(bmu_id: str, *, fuel: str | None, name: str, party: str) -> dict[str, object]:
        return {
            "elexonBmUnit": bmu_id,
            "nationalGridBmUnit": bmu_id[2:],
            "fuelType": fuel,
            "leadPartyName": party,
            "bmUnitName": name,
            "bmUnitType": bmu_id[0],
            "generationCapacity": "50.0",
            "demandCapacity": "-50.0",
        }

    return parse_reference(
        rows=[
            entry("T_PUMP-1", fuel="PS", name="Pump Hydro", party="Hydro Ltd"),
            entry("T_CELLX-1", fuel="OTHER", name="Cell X", party="Cell Co"),
            entry("T_LKSDB-1", fuel="OTHER", name="Lakeside BESS", party="Lakeside Energy Storage"),
            entry("E_NOFUB-1", fuel=None, name="Quiet Site", party="Quiet Co"),
            entry("E_KEYWD-9", fuel="OTHER", name="Big Battery", party="Grid Co"),
            entry("T_WINDY-1", fuel="WIND", name="Windy Battery Farm", party="Wind Co"),
            entry("2__SUPPL1", fuel=None, name="Supplier", party="Supply Co"),
            entry("T_OTHR-1", fuel="OTHER", name="Gas Peaker", party="Peak Co"),
        ]
    )


def test_battery_hint_rules() -> None:
    assert "bess" in battery_hint(bmu_id="X", bmu_name="Lakeside BESS", lead_party=None)
    assert "storage" in battery_hint(bmu_id="X", bmu_name=None, lead_party="Energy Storage Ltd")
    assert "B-<n>" in battery_hint(bmu_id="T_LKSDB-1", bmu_name=None, lead_party=None)
    assert "B-<n>" in battery_hint(bmu_id="E_DOLLB-1", bmu_name=None, lead_party=None)
    assert battery_hint(bmu_id="T_PEAK-1", bmu_name="Gas Peaker", lead_party="Peak Co") == ""
    assert battery_hint(bmu_id="T_X-1", bmu_name="Chess Club", lead_party=None) == ""


def test_candidates_cover_fuel_types_hints_and_named_units() -> None:
    candidates = build_candidates(reference=_reference())
    by_id = {row["elexon_bmu_id"]: row for row in candidates.iter_rows(named=True)}
    assert by_id["T_PUMP-1"]["candidate_class"] == "pumped_storage"
    assert by_id["T_CELLX-1"]["candidate_class"] == "other_fuel_unhinted"
    assert by_id["T_LKSDB-1"]["candidate_class"] == "battery_hint"
    assert by_id["E_KEYWD-9"]["battery_hint"] is True
    assert by_id["E_DOLLB-1"]["candidate_class"] == "named_by_study"
    assert "E_NOFUB-1" in by_id
    assert "T_WINDY-1" not in by_id
    assert "2__SUPPL1" not in by_id
    assert by_id["T_OTHR-1"]["candidate_class"] == "other_fuel_unhinted"
    assert by_id["T_OTHR-1"]["battery_hint"] is False
    assert candidates["elexon_bmu_id"].is_unique().all()


def test_bmu_batches_depend_only_on_each_identifier() -> None:
    ids = [f"T_UNIT{index:03d}-1" for index in range(45)]
    batches = bmu_batches(bmu_ids=ids)
    assert sorted(bmu_id for batch in batches for bmu_id in batch) == ids
    assert len(batches) <= 16
    fewer = bmu_batches(bmu_ids=[bmu_id for bmu_id in ids if bmu_id != ids[7]])
    changed = {tuple(batch) for batch in fewer} ^ {tuple(batch) for batch in batches}
    assert len(changed) <= 2
    with pytest.raises(ValueError, match="URL"):
        bmu_batches(bmu_ids=[f"T_{'X' * 80}{index}" for index in range(1000)])


def test_list_hash_ignores_order_and_tracks_content() -> None:
    assert list_hash(bmu_ids=["B", "A"]) == list_hash(bmu_ids=["A", "B"])
    assert list_hash(bmu_ids=["A"]) != list_hash(bmu_ids=["A", "B"])


def _pairs(*, first: float) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for index in range(1, 7):
        values[f"negative{index}"] = None if index > 2 else 0.0
        values[f"positive{index}"] = None if index > 2 else (first if index == 1 else 0.0)
    return values


def test_parse_boav_and_ebocf_flatten_pairs_and_deduplicate() -> None:
    boav_row = {
        "createdDateTime": "2025-10-02T23:14:46Z",
        "settlementDate": "2025-10-01",
        "settlementPeriod": 48,
        "startTime": "2025-10-01T22:30:00Z",
        "bmUnit": "T_LKSDB-1",
        "acceptanceId": 22018,
        "acceptanceDuration": "S",
        "totalVolumeAccepted": 2.5,
        "pairVolumes": _pairs(first=2.5),
    }
    boav = parse_boav(rows=[boav_row, boav_row], side="offer")
    assert boav.height == 1
    assert boav["pair_volume_pos1"][0] == 2.5
    assert boav["pair_volume_pos3"][0] is None
    assert boav["side"][0] == "offer"
    ebocf_row = {
        "createdDateTime": "2025-10-02T23:14:46Z",
        "settlementDate": "2025-10-01",
        "settlementPeriod": 48,
        "startTime": "2025-10-01T22:30:00Z",
        "bmUnit": "T_LKSDB-1",
        "totalCashflow": 40.5,
        "bidOfferPairCashflows": _pairs(first=40.5),
    }
    ebocf = parse_ebocf(rows=[ebocf_row], side="bid")
    assert ebocf["total_cashflow_gbp"][0] == 40.5
    assert ebocf["pair_cashflow_pos1"][0] == 40.5


def test_parse_boalf_keeps_only_time_from_inside_the_chunk() -> None:
    def row(time_from: str) -> dict[str, object]:
        return {
            "bmUnit": "T_LKSDB-1",
            "acceptanceNumber": 22021,
            "acceptanceTime": "2025-10-01T23:49:00Z",
            "settlementDate": "2025-10-02",
            "settlementPeriodFrom": 2,
            "settlementPeriodTo": 3,
            "timeFrom": time_from,
            "timeTo": "2025-10-02T00:30:00Z",
            "levelFrom": 33,
            "levelTo": 0,
            "deemedBoFlag": False,
            "soFlag": False,
            "amendmentFlag": "ORI",
            "storFlag": False,
            "rrFlag": False,
        }

    frame = parse_boalf(
        rows=[
            row("2025-10-01T23:59:00Z"),
            row("2025-10-02T00:00:00Z"),
            row("2025-11-01T00:00:00Z"),
        ],
        start=datetime(2025, 10, 1, tzinfo=UTC),
        end=datetime(2025, 11, 1, tzinfo=UTC),
    )
    assert frame.height == 2
    assert frame["level_from_mw"].dtype == pl.Float64


def test_source_gaps_find_the_known_gap_and_ignore_gaps_outside_the_window() -> None:
    start = end = date(2025, 10, 14)
    rows = [(start, period, 5) for period in range(1, 49) if period not in (20, 21)]
    status = pl.DataFrame(
        rows,
        schema={
            "settlement_date": pl.Date,
            "settlement_period": pl.Int16,
            "data_point_count": pl.Int64,
        },
        orient="row",
    )
    gaps = source_gaps(status=status, dataset="BOAV", start=start, end=end)
    assert gaps["periods_with_no_data_point"] == ["2025-10-14:20", "2025-10-14:21"]
    assert gaps["known_gaps_still_missing"] == ["2025-10-14:20", "2025-10-14:21"]
    assert gaps["known_gaps_now_present"] == []
    assert gaps["missing_not_in_known_list"] == []


def _chunk(key: str) -> pl.DataFrame:
    return pl.DataFrame({"key": [key]})


def test_fetch_missing_chunks_skips_cached_chunks(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(key: str) -> pl.DataFrame:
        calls.append(key)
        return _chunk(key)

    first = fetch_missing_chunks(cache_dir=tmp_path, keys=["a", "b"], fetch=fetch, label="t")
    assert sorted(first["fetched"]) == ["a", "b"]
    second = fetch_missing_chunks(cache_dir=tmp_path, keys=["a", "b", "c"], fetch=fetch, label="t")
    assert second["fetched"] == ["c"]
    assert sorted(second["cached"]) == ["a", "b"]
    assert sorted(calls) == ["a", "b", "c"]
    combined = read_chunks(cache_dir=tmp_path, keys=["a", "b", "c"], schema={"key": pl.String})
    assert combined["key"].to_list() == ["a", "b", "c"]
    assert list(tmp_path.glob("*.partial")) == []


def test_fetch_missing_chunks_records_404_and_raises_on_other_errors(tmp_path: Path) -> None:
    def fetch(key: str) -> pl.DataFrame:
        status = 404 if key == "gone" else 500
        request = httpx.Request("GET", "https://example.invalid")
        response = httpx.Response(status, request=request)
        if key in ("gone", "broken"):
            raise httpx.HTTPStatusError("boom", request=request, response=response)
        return _chunk(key)

    outcome = fetch_missing_chunks(cache_dir=tmp_path, keys=["ok", "gone"], fetch=fetch, label="t")
    assert outcome["not_published"] == ["gone"]
    assert not (tmp_path / "gone.parquet").exists()
    with pytest.raises(httpx.HTTPStatusError):
        fetch_missing_chunks(cache_dir=tmp_path, keys=["broken"], fetch=fetch, label="t")


def test_incomplete_trailing_chunk_is_skipped_but_not_cached(tmp_path: Path) -> None:
    def fetch(key: str) -> pl.DataFrame:
        return require_complete(
            frame=_chunk(key).head(0 if key == "c" else 1), expected_rows=1, key=key
        )

    outcome = fetch_missing_chunks(cache_dir=tmp_path, keys=["a", "b", "c"], fetch=fetch, label="t")
    assert outcome["not_published"] == ["c"]
    assert not (tmp_path / "c.parquet").exists()


def test_incomplete_chunk_before_a_complete_one_raises(tmp_path: Path) -> None:
    def fetch(key: str) -> pl.DataFrame:
        return require_complete(
            frame=_chunk(key).head(0 if key == "a" else 1), expected_rows=1, key=key
        )

    with pytest.raises(RuntimeError, match="not at the end"):
        fetch_missing_chunks(cache_dir=tmp_path, keys=["a", "b"], fetch=fetch, label="t")


def test_require_complete_raises_on_a_short_frame() -> None:
    with pytest.raises(IncompleteChunkError):
        require_complete(frame=_chunk("a"), expected_rows=2, key="a")


def test_period_time_mismatches_counts_wrong_start_times() -> None:
    good = period_start_utc(settlement_date=date(2025, 10, 1), settlement_period=1)
    frame = pl.DataFrame(
        {
            "settlement_date": [date(2025, 10, 1), date(2025, 10, 1)],
            "settlement_period": [1, 2],
            "time": [good, good],
        },
        schema={
            "settlement_date": pl.Date,
            "settlement_period": pl.Int16,
            "time": pl.Datetime("us", "UTC"),
        },
    )
    assert period_time_mismatches(frame=frame) == 1


def test_apx_lag_correlation_peaks_at_the_true_lag() -> None:
    hours = [datetime(2025, 7, 1, tzinfo=UTC) + timedelta(hours=index) for index in range(72)]
    prices = [float((index * 37) % 11) for index in range(72)]
    n2ex = pl.DataFrame(
        {"time": hours, "price_gbp_per_mwh": prices},
        schema={"time": pl.Datetime("us", "UTC"), "price_gbp_per_mwh": pl.Float64},
    )

    def apx_from(offset: int) -> pl.DataFrame:
        times = [stamp + timedelta(hours=offset) for stamp in hours]
        rows = [
            (t + timedelta(minutes=m), p)
            for t, p in zip(times, prices, strict=True)
            for m in (0, 30)
        ]
        return pl.DataFrame(
            rows,
            schema={"time": pl.Datetime("us", "UTC"), "price_gbp_per_mwh": pl.Float64},
            orient="row",
        )

    same = apx_lag_correlations(n2ex=n2ex, apx=apx_from(0))
    assert same["0"] == pytest.approx(1.0)
    assert same["1"] is not None
    assert same["1"] < 0.9
    shifted = apx_lag_correlations(n2ex=n2ex, apx=apx_from(1))
    assert shifted["1"] == pytest.approx(1.0)
    assert shifted["0"] is not None
    assert shifted["0"] < 0.9


def test_write_readme_names_the_study_it_was_given(tmp_path: Path) -> None:
    kwargs = {
        "product_dir": tmp_path,
        "title": "A series",
        "source_page": "https://example.org",
        "script_path": "studies/x.py",
        "attribution": None,
        "cache_hint": "_cache/",
        "licence": "Open.",
        "timestamp_convention": "UTC.",
        "columns": {"time": "UTC start."},
        "row_summary": "- 1 row.",
        "gotchas": [],
    }
    write_readme(**kwargs, purpose="Public data for the study of X.")
    named = (tmp_path / "README.md").read_text()
    assert "Public data for the study of X." in named
    assert "battery-versus-solar-PV" not in named
    write_readme(**kwargs)
    assert "battery-versus-solar-PV" in (tmp_path / "README.md").read_text()
