"""Tests for the NESO Enduring Auction Capability download script and its unit-to-BMU mapping.

No test touches the network. The HTTP call is replaced with a fake through `monkeypatch`.
"""

from datetime import UTC, date, datetime
from typing import Any

import fetch_neso_eac
import polars as pl
import pytest
from fetch_neso_eac import (
    BALANCING_RESERVE_RESOURCE,
    RESPONSE_RESERVE_RESOURCES,
    EacResource,
    block_lengths,
    build_bmu_index,
    build_mapping,
    candidates_for_unit,
    day_query,
    describe_units,
    efa_block_starts,
    fetch_result_day,
    parse_results,
    party_similarity,
    product_gaps,
    resource_overlaps_day,
    study_coverage,
)
from market_common import IncompleteChunkError


def _record(**overrides: Any) -> dict[str, Any]:
    base = {
        "_id": 1,
        "registeredAuctionParticipant": "EDF ENERGY CUSTOMERS LIMITED",
        "auctionUnit": "BREDB-1",
        "serviceType": "Quick Reserve",
        "auctionProduct": "PQR",
        "executedQuantity": "4",
        "clearingPrice": "0.5",
        "deliveryStart": "2026-10-08T04:00:00",
        "deliveryEnd": "2026-10-08T04:30:00",
        "technologyType": "Batteries",
        "postCode": "SK6 2BS",
        "unitResultID": "2865#||#2674#||#PQR#||#212263",
    }
    return base | overrides


def test_parse_results_casts_numbers_and_reads_times_as_utc() -> None:
    frame = parse_results(rows=[_record()])
    assert frame.columns == list(fetch_neso_eac.RESULT_SCHEMA)
    assert frame["time"][0] == datetime(2026, 10, 8, 4, 0, tzinfo=UTC)
    assert frame["time_end"][0] == datetime(2026, 10, 8, 4, 30, tzinfo=UTC)
    assert frame["executed_quantity_mw"][0] == 4.0
    assert frame["clearing_price_gbp_per_mw_per_h"][0] == 0.5


def test_parse_results_sorts_by_time_then_product_then_unit() -> None:
    later = _record(deliveryStart="2026-10-08T05:00:00", unitResultID="b")
    earlier_unit_b = _record(auctionUnit="B", unitResultID="c")
    earlier_unit_a = _record(auctionUnit="A", unitResultID="a")
    frame = parse_results(rows=[later, earlier_unit_b, earlier_unit_a])
    assert frame["unit_result_id"].to_list() == ["a", "c", "b"]


def test_parse_results_of_no_rows_has_the_schema() -> None:
    frame = parse_results(rows=[])
    assert frame.height == 0
    assert frame.schema == pl.Schema(fetch_neso_eac.RESULT_SCHEMA)


def test_parse_results_rejects_a_non_numeric_price() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        parse_results(rows=[_record(clearingPrice="n/a")])


def test_day_query_selects_one_utc_day_and_the_requested_service() -> None:
    resource = RESPONSE_RESERVE_RESOURCES[0]
    only = day_query(resource=resource, day=date(2025, 10, 30), balancing_reserve_only=True)
    drop = day_query(resource=resource, day=date(2025, 10, 30), balancing_reserve_only=False)
    assert "\"deliveryStart\" >= '2025-10-30T00:00:00'" in only
    assert "\"deliveryStart\" < '2025-10-31T00:00:00'" in only
    assert "\"serviceType\" = 'Balancing Reserve'" in only
    assert "\"serviceType\" <> 'Balancing Reserve'" in drop


def test_dedicated_balancing_reserve_resource_stops_on_the_day_the_other_one_starts() -> None:
    last_day = date(2025, 10, 29)
    assert resource_overlaps_day(resource=BALANCING_RESERVE_RESOURCE, day=last_day)
    assert not resource_overlaps_day(resource=BALANCING_RESERVE_RESOURCE, day=date(2025, 10, 30))
    first, _ = RESPONSE_RESERVE_RESOURCES
    assert resource_overlaps_day(resource=first, day=last_day)
    assert not resource_overlaps_day(resource=first, day=date(2026, 4, 1))


def test_open_ended_resource_overlaps_every_later_day() -> None:
    _, current = RESPONSE_RESERVE_RESOURCES
    assert current.last_start is None
    assert resource_overlaps_day(resource=current, day=date(2026, 9, 30))
    assert not resource_overlaps_day(resource=current, day=date(2026, 3, 30))


def test_efa_block_starts_follow_uk_clock_time() -> None:
    winter = efa_block_starts(start=date(2025, 12, 15), end=date(2025, 12, 15))
    assert [stamp.hour for stamp in winter] == [3, 7, 11, 15, 19, 23]
    summer = efa_block_starts(start=date(2025, 6, 15), end=date(2025, 6, 15))
    assert [stamp.hour for stamp in summer] == [2, 6, 10, 14, 18, 22]
    assert all(stamp.tzinfo == UTC for stamp in winter + summer)


def test_efa_block_starts_stay_unique_across_a_clock_change() -> None:
    starts = efa_block_starts(start=date(2025, 10, 24), end=date(2025, 10, 28))
    assert len(starts) == 30
    assert len(set(starts)) == 30
    assert starts == sorted(starts)


def _result_frame(*, rows: list[tuple[str, datetime, datetime]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "auction_product": [row[0] for row in rows],
            "time": [row[1] for row in rows],
            "time_end": [row[2] for row in rows],
        },
        schema={
            "auction_product": pl.String,
            "time": pl.Datetime("us", "UTC"),
            "time_end": pl.Datetime("us", "UTC"),
        },
    )


def test_product_gaps_skips_sparse_products_and_finds_a_missing_block() -> None:
    day = date(2025, 12, 15)
    starts = efa_block_starts(start=day, end=day)
    kept = [stamp for stamp in starts if stamp.hour != 11]
    rows = [("DCH", stamp, stamp) for stamp in kept]
    rows.append(("NBR", starts[0], starts[0]))
    gaps = product_gaps(frame=_result_frame(rows=rows), start=day, end=day)
    assert set(gaps) == {"DCH"}
    assert gaps["DCH"]["expected_rows"] == 6
    assert gaps["DCH"]["missing_count"] == 1


def test_product_gaps_counts_a_missing_block_between_the_first_and_last() -> None:
    first_day, last_day = date(2025, 12, 15), date(2025, 12, 16)
    starts = efa_block_starts(start=first_day, end=last_day)
    kept = [stamp for stamp in starts if stamp != starts[4]]
    gaps = product_gaps(
        frame=_result_frame(rows=[("DCL", stamp, stamp) for stamp in kept]),
        start=first_day,
        end=last_day,
    )
    assert gaps["DCL"]["missing_count"] == 1
    assert gaps["DCL"]["missing_first"] == [starts[4].isoformat()]


def test_product_gaps_expects_every_half_hour_for_a_reserve_product() -> None:
    day = date(2025, 12, 15)
    first = datetime(2025, 12, 15, 0, 0, tzinfo=UTC)
    last = datetime(2025, 12, 15, 1, 30, tzinfo=UTC)
    middle = datetime(2025, 12, 15, 0, 30, tzinfo=UTC)
    rows = [("PQR", first, first), ("PQR", last, last)]
    gaps = product_gaps(frame=_result_frame(rows=rows), start=day, end=day)
    assert gaps["PQR"]["expected_rows"] == 4
    assert gaps["PQR"]["missing_count"] == 2
    assert gaps["PQR"]["missing_first"][0] == middle.isoformat()


def test_block_lengths_report_the_clock_change_blocks() -> None:
    rows = [
        (
            "DCH",
            datetime(2025, 10, 25, 22, 0, tzinfo=UTC),
            datetime(2025, 10, 26, 3, 0, tzinfo=UTC),
        ),
        ("DCH", datetime(2025, 10, 26, 3, 0, tzinfo=UTC), datetime(2025, 10, 26, 7, 0, tzinfo=UTC)),
        (
            "PQR",
            datetime(2025, 10, 26, 3, 0, tzinfo=UTC),
            datetime(2025, 10, 26, 3, 30, tzinfo=UTC),
        ),
    ]
    assert block_lengths(frame=_result_frame(rows=rows)) == {"DCH": ["4", "5"], "PQR": ["0.5"]}


def test_fetch_result_day_pages_through_a_long_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    records = [_record(_id=index, unitResultID=f"id{index}") for index in range(5)]

    def fake_get_json(*, url: str, params: list[tuple[str, Any]]) -> dict[str, Any]:
        sql = params[0][1]
        calls.append(sql)
        offset = int(sql.rsplit("OFFSET ", maxsplit=1)[1])
        return {"success": True, "result": {"records": records[offset : offset + 2]}}

    monkeypatch.setattr(fetch_neso_eac, "get_json", fake_get_json)
    monkeypatch.setattr(fetch_neso_eac, "PAGE_ROWS", 2)
    frame = fetch_result_day(source="neso_eac_response_reserve", key="2026-10-08")
    assert frame.height == 5
    assert len(calls) == 3
    assert calls[0].endswith("LIMIT 2 OFFSET 0")


def test_fetch_result_day_queries_both_balancing_reserve_resources_on_the_boundary_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resources_asked: list[str] = []

    def fake_get_json(*, url: str, params: list[tuple[str, Any]]) -> dict[str, Any]:
        resources_asked.append(params[0][1].split('"')[1])
        return {"success": True, "result": {"records": [_record()]}}

    monkeypatch.setattr(fetch_neso_eac, "get_json", fake_get_json)
    fetch_result_day(source="neso_eac_balancing_reserve", key="2025-10-29")
    assert resources_asked == [
        BALANCING_RESERVE_RESOURCE.resource_id,
        RESPONSE_RESERVE_RESOURCES[0].resource_id,
    ]


def test_fetch_result_day_with_no_rows_is_an_incomplete_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_json(*, url: str, params: list[tuple[str, Any]]) -> dict[str, Any]:
        return {"success": True, "result": {"records": []}}

    monkeypatch.setattr(fetch_neso_eac, "get_json", fake_get_json)
    with pytest.raises(IncompleteChunkError, match="no resource holds rows"):
        fetch_result_day(source="neso_eac_response_reserve", key="2026-10-08")


def test_fetch_result_day_raises_when_the_reply_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_json(*, url: str, params: list[tuple[str, Any]]) -> dict[str, Any]:
        return {"success": False, "error": {"message": "boom"}}

    monkeypatch.setattr(fetch_neso_eac, "get_json", fake_get_json)
    with pytest.raises(RuntimeError, match="NESO SQL failed"):
        fetch_result_day(source="neso_eac_response_reserve", key="2026-10-08")


def test_resource_tuple_is_well_formed() -> None:
    assert isinstance(BALANCING_RESERVE_RESOURCE, EacResource)
    assert RESPONSE_RESERVE_RESOURCES[0].last_start is not None
    assert RESPONSE_RESERVE_RESOURCES[1].first_start > RESPONSE_RESERVE_RESOURCES[0].last_start


REFERENCE = pl.DataFrame(
    {
        "elexon_bmu_id": ["E_ARBRB-1", "E_BURWB-2", "T_ERSKB-1", None, "2__FBPGM001"],
        "national_grid_bmu_id": ["ARBRB-1", "BURWB-2", "ERSKB-1", "COALB-1", "FBPG01"],
        "lead_party_name": [
            "Octopus Energy Trading Limited",
            "EDF Energy Customers Limited",
            "EDF Energy Customers Limited",
            "Coalition Power plc",
            "BP Gas Marketing Limited",
        ],
    }
)


def test_exact_national_grid_id_matches_with_score_one() -> None:
    index = build_bmu_index(reference=REFERENCE)
    assert candidates_for_unit(unit="ARBRB-1", index=index) == [
        ("E_ARBRB-1", "exact_national_grid_id", 1.0)
    ]
    assert candidates_for_unit(unit="arbrb-1", index=index)[0][0] == "E_ARBRB-1"


def test_bmu_without_an_elexon_id_is_keyed_by_its_national_grid_id() -> None:
    index = build_bmu_index(reference=REFERENCE)
    assert candidates_for_unit(unit="COALB-1", index=index) == [
        ("COALB-1", "exact_national_grid_id", 1.0)
    ]


def test_exact_elexon_id_and_prefix_stripped_id_match() -> None:
    index = build_bmu_index(reference=REFERENCE)
    assert candidates_for_unit(unit="E_BURWB-2", index=index) == [
        ("E_BURWB-2", "exact_elexon_id", 1.0)
    ]
    # `FBPGM001` is the Elexon id once the `2__` prefix is removed.
    assert candidates_for_unit(unit="FBPGM001", index=index) == [
        ("2__FBPGM001", "elexon_id_without_prefix", 0.95)
    ]


def test_an_exact_match_hides_weaker_candidates() -> None:
    index = build_bmu_index(reference=REFERENCE)
    methods = {method for _, method, _ in candidates_for_unit(unit="BURWB-2", index=index)}
    assert methods == {"exact_national_grid_id"}


def test_sibling_battery_b_and_fuzzy_candidates() -> None:
    index = build_bmu_index(reference=REFERENCE)
    assert candidates_for_unit(unit="BURWB-1", index=index) == [
        ("E_BURWB-2", "same_stem_sibling", 0.7)
    ]
    assert candidates_for_unit(unit="ERSK-01", index=index) == [
        ("T_ERSKB-1", "stem_plus_battery_b", 0.8)
    ]
    fuzzy = candidates_for_unit(unit="ARBRC-1", index=index)
    assert [(bmu, method) for bmu, method, _ in fuzzy] == [("E_ARBRB-1", "fuzzy_id")]
    assert 0.85 <= fuzzy[0][2] < 1.0


def test_unit_with_nothing_similar_has_no_candidate() -> None:
    index = build_bmu_index(reference=REFERENCE)
    assert candidates_for_unit(unit="ZZZZ-99", index=index) == []


def test_party_similarity_ignores_legal_words_and_handles_missing_names() -> None:
    assert (
        party_similarity(left="EDF ENERGY CUSTOMERS LIMITED", right="EDF Energy Customers Ltd")
        == 1.0
    )
    assert party_similarity(left="Tesla Motors Limited", right="BGI Trading Limited") == 0.0
    assert party_similarity(left=None, right="BGI Trading Limited") is None
    assert party_similarity(left="Energy Limited", right="BGI Trading Limited") is None


def _units() -> pl.DataFrame:
    results = pl.DataFrame(
        {
            "auction_unit": ["ARBRB-1", "ARBRB-1", "ARBRB-1", "ERSK-01", "ZZZZ-99"],
            "participant": ["Octopus", "Octopus", "Other", "EDF", "Nobody"],
            "technology_type": ["Batteries"] * 4 + ["Diesel"],
            "service_type": ["Response", "Quick Reserve", "Response", "Response", "Response"],
        }
    )
    return describe_units(results=[results])


def test_describe_units_keeps_the_most_common_participant_and_lists_services() -> None:
    units = _units().filter(pl.col("auction_unit") == "ARBRB-1").row(0, named=True)
    assert units["participant"] == "Octopus"
    assert units["services"] == "Quick Reserve, Response"
    assert units["result_rows"] == 3


def test_build_mapping_gives_every_unit_at_least_one_row() -> None:
    mapping = build_mapping(
        units=_units(), reference=REFERENCE, study_bmus={"E_ARBRB-1", "E_BURWB-2"}
    )
    assert mapping["auction_unit"].n_unique() == 3
    none_row = mapping.filter(pl.col("auction_unit") == "ZZZZ-99").row(0, named=True)
    assert none_row["candidate_bmu_id"] is None
    assert none_row["match_method"] == "none"
    assert none_row["match_score"] == 0.0
    assert none_row["candidate_is_study_bmu"] is False
    arbrb = mapping.filter(pl.col("auction_unit") == "ARBRB-1").row(0, named=True)
    assert arbrb["candidate_is_study_bmu"] is True
    assert arbrb["candidate_lead_party"] == "Octopus Energy Trading Limited"


def test_study_coverage_separates_plausible_possible_and_none() -> None:
    mapping = build_mapping(units=_units(), reference=REFERENCE, study_bmus=set())
    coverage = study_coverage(
        mapping=mapping, study_bmus=["E_ARBRB-1", "T_ERSKB-1", "E_BURWB-2"]
    ).sort("elexon_bmu_id")
    verdicts = dict(zip(coverage["elexon_bmu_id"], coverage["eac_match"], strict=True))
    assert verdicts == {
        "E_ARBRB-1": "plausible",
        "T_ERSKB-1": "possible",
        "E_BURWB-2": "no_identifier_match",
    }
    assert coverage.filter(pl.col("elexon_bmu_id") == "E_ARBRB-1")["result_rows"][0] == 3
