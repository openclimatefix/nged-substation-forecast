import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import census
import polars as pl
import pytest

EMPTY = "data not applicable"


def _ecr_frame(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    """A frame with the columns `with_storage_flags` reads, filling unspecified ones."""
    defaults: dict[str, object] = {
        "licence_area": "National Grid Electricity Distribution (South West) Plc",
        "energy_source_1": "Solar",
        "technology_1": "Photovoltaic",
        "energy_source_2": EMPTY,
        "technology_2": EMPTY,
        "energy_source_3": EMPTY,
        "technology_3": EMPTY,
        "connection_status": "Connected",
        "maximum_export_capacity_mw": 1.0,
        "connected_registered_capacity_mw": 1.0,
        "customer_name": "Someone",
        "customer_site": "Somewhere",
        "address_line_1": "",
        "address_line_2": "",
        "town": "",
        "county": "",
        "easting": 0.0,
        "northing": 0.0,
    }
    full = [{**defaults, **row} for row in rows]
    return pl.DataFrame(full, schema_overrides={"easting": pl.Float64, "northing": pl.Float64})


STORAGE = {
    "energy_source_1": "Stored Energy (all stored energy irrespective of the original source)",
    "technology_1": "Storage - Electrochemical  (Batteries)",
}


@pytest.mark.parametrize(
    ("megawatts", "expected"),
    [
        (0.999, "under 1 MW"),
        (1.0, "1 to 10 MW"),
        (9.99, "1 to 10 MW"),
        (10.0, "10 to 50 MW"),
        (50.0, "50 to 100 MW"),
        (99.9, "50 to 100 MW"),
        (100.0, "100 MW or more"),
        (None, None),
        (float("nan"), None),
    ],
)
def test_a_battery_on_a_size_edge_belongs_to_the_class_above(
    megawatts: float | None, expected: str | None
) -> None:
    assert census.size_class_of(megawatts=megawatts) == expected


def test_the_licence_area_is_the_bracketed_name() -> None:
    text = "National Grid Electricity Distribution (East Midlands) Plc"

    assert census.licence_area_of(text=text) == "East Midlands"
    with pytest.raises(ValueError, match="bracketed"):
        census.licence_area_of(text="No brackets here")


def test_every_licence_area_has_a_grid_supply_point_group() -> None:
    assert set(census.GSP_GROUP_OF_LICENCE_AREA) == set(census.LICENCE_AREAS)
    assert census.GSP_GROUP_OF_LICENCE_AREA["West Midlands"] == "Midlands"
    assert census.GSP_GROUP_OF_LICENCE_AREA["South West"] == "South Western"


def test_storage_flags_separate_storage_only_hybrid_and_other_rows() -> None:
    frame = _ecr_frame(
        [
            STORAGE,
            {**STORAGE, "energy_source_2": "Solar", "technology_2": "Photovoltaic"},
            {
                "energy_source_2": STORAGE["energy_source_1"],
                "technology_2": STORAGE["technology_1"],
            },
            {},
        ]
    )

    flagged = census.with_storage_flags(ecr=frame)

    assert flagged["lists_storage"].to_list() == [True, True, True, False]
    assert flagged["hybrid"].to_list() == [False, True, True, False]
    assert flagged["storage_first"].to_list() == [True, True, False, False]
    assert flagged["electrochemical"].to_list() == [True, True, True, False]


def test_an_unfilled_technology_slot_does_not_make_a_row_hybrid() -> None:
    flagged = census.with_storage_flags(ecr=_ecr_frame([STORAGE]))

    assert flagged["hybrid"].to_list() == [False]


def test_export_capacity_falls_back_to_the_registered_capacity() -> None:
    frame = _ecr_frame(
        [
            {
                **STORAGE,
                "maximum_export_capacity_mw": None,
                "connected_registered_capacity_mw": 7.0,
            },
            {**STORAGE, "maximum_export_capacity_mw": 3.0},
        ]
    )

    flagged = census.with_storage_flags(ecr=frame)

    assert flagged["export_mw"].to_list() == [7.0, 3.0]
    assert flagged["size_class"].to_list() == ["1 to 10 MW", "1 to 10 MW"]


def test_connected_storage_keeps_only_connected_rows_that_list_storage() -> None:
    frame = _ecr_frame(
        [STORAGE, {**STORAGE, "connection_status": "Accepted to connect"}, {"technology_1": "x"}]
    )

    assert census.connected_storage(ecr=frame).height == 1


def test_place_tokens_drop_generic_and_short_words() -> None:
    assert census.place_tokens(bmu_name="Wolverhampton West BESS") == ["wolverhampton", "west"]
    assert census.place_tokens(bmu_name="AR0006-BLOX") == ["blox"]
    assert census.place_tokens(bmu_name="Energy Storage Park") == []


def _bmus(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    defaults: dict[str, object] = {
        "elexon_bmu_id": "E_NEWB-1",
        "bmu_name": "Newbury BESS",
        "generation_capacity_mw": 50.0,
        "licence_area": "South West",
        "bmu_type": "E",
        "fpn_flag": True,
        "pn_rows": 100,
    }
    return pl.DataFrame(
        [{**defaults, **row} for row in rows], schema_overrides={"pn_rows": pl.UInt32}
    )


def _connected(rows: Sequence[Mapping[str, object]]) -> pl.DataFrame:
    return census.connected_storage(ecr=_ecr_frame([{**STORAGE, **row} for row in rows]))


def test_a_row_that_agrees_on_text_and_capacity_is_graded_text_and_capacity() -> None:
    ecr_storage = _connected(
        [
            {"customer_site": "Newbury Battery Storage", "maximum_export_capacity_mw": 49.0},
            {"customer_site": "Oakfield", "maximum_export_capacity_mw": 3.0},
        ]
    )
    bmus = _bmus([{"elexon_bmu_id": "E_NEWB-1"}])

    proposals = census.propose_matches(ecr_storage=ecr_storage, bmus=bmus)

    assert proposals["grade"].to_list() == ["text and capacity"]
    assert proposals["ecr_row"].to_list() == [0]


def test_capacity_alone_is_graded_capacity_only_and_text_alone_text_only() -> None:
    ecr_storage = _connected(
        [
            {"customer_site": "Oakfield", "maximum_export_capacity_mw": 51.0},
            {"customer_site": "Newbury Battery Storage", "maximum_export_capacity_mw": 3.0},
        ]
    )
    bmus = _bmus([{"elexon_bmu_id": "E_NEWB-1"}])

    proposals = census.propose_matches(ecr_storage=ecr_storage, bmus=bmus).sort("ecr_row")

    assert proposals["grade"].to_list() == ["capacity only", "text only"]


def test_capacity_beyond_ten_percent_is_not_a_capacity_hit() -> None:
    ecr_storage = _connected([{"customer_site": "Oakfield", "maximum_export_capacity_mw": 56.0}])

    proposals = census.propose_matches(ecr_storage=ecr_storage, bmus=_bmus([{}]))

    assert proposals.height == 0


def test_a_row_in_another_licence_area_is_never_a_candidate() -> None:
    ecr_storage = _connected(
        [
            {
                "licence_area": "National Grid Electricity Distribution (South Wales) Plc",
                "customer_site": "Newbury Battery Storage",
                "maximum_export_capacity_mw": 50.0,
            }
        ]
    )

    assert census.propose_matches(ecr_storage=ecr_storage, bmus=_bmus([{}])).height == 0


def test_the_place_key_reads_the_address_text() -> None:
    ecr_storage = _connected([{"town": "Newbury", "maximum_export_capacity_mw": 0.2}])

    proposals = census.propose_matches(ecr_storage=ecr_storage, bmus=_bmus([{}]))

    assert proposals["place_hit"].to_list() == [True]
    assert proposals["grade"].to_list() == ["text only"]


def test_a_bmu_outside_the_four_licence_areas_proposes_nothing() -> None:
    ecr_storage = _connected([{"customer_site": "Newbury Battery Storage"}])

    proposals = census.propose_matches(
        ecr_storage=ecr_storage, bmus=_bmus([{"licence_area": None}])
    )

    assert proposals.height == 0


def test_nearest_ecr_row_reports_distance_and_leaves_far_batteries_unplaced() -> None:
    repd = pl.DataFrame(
        {
            "repd_row": [0, 1],
            "easting": [1000.0, 90000.0],
            "northing": [2000.0, 90000.0],
        }
    )
    ecr_rows = pl.DataFrame(
        {"ecr_row": [7, 8], "easting": [1300.0, 1000.0], "northing": [2400.0, 4000.0]}
    ).with_columns(pl.col("ecr_row").cast(pl.UInt32))

    result = census.nearest_ecr_row(repd=repd, ecr_rows=ecr_rows)

    assert result["nearest_ecr_row"].to_list() == [7, None]
    assert result["distance_m"].to_list() == [pytest.approx(500.0), None]


def test_nearest_ecr_row_honours_a_tighter_limit() -> None:
    repd = pl.DataFrame({"repd_row": [0], "easting": [0.0], "northing": [0.0]})
    ecr_rows = pl.DataFrame({"ecr_row": [1], "easting": [3000.0], "northing": [0.0]}).with_columns(
        pl.col("ecr_row").cast(pl.UInt32)
    )

    assert census.nearest_ecr_row(repd=repd, ecr_rows=ecr_rows, limit_m=2000.0)[
        "nearest_ecr_row"
    ].to_list() == [None]
    assert census.nearest_ecr_row(repd=repd, ecr_rows=ecr_rows, limit_m=4000.0)[
        "nearest_ecr_row"
    ].to_list() == [1]


def test_repd_route_matches_a_bmu_to_a_battery_by_name_and_carries_its_ecr_row() -> None:
    repd_located = pl.DataFrame(
        {
            "repd_row": [0, 1],
            "repd_site_name": ["Newbury Battery Storage", "Oakfield"],
            "repd_capacity_mw": [49.5, 5.0],
            "nearest_ecr_row": [9, None],
            "distance_m": [120.0, None],
        },
        schema_overrides={"repd_row": pl.UInt32, "nearest_ecr_row": pl.UInt32},
    )
    bmus = _bmus([{"elexon_bmu_id": "E_NEWB-1"}, {"elexon_bmu_id": "E_OTHER-1", "bmu_name": "Zzz"}])

    route = census.propose_repd_route(bmus=bmus, repd_located=repd_located)

    assert route["elexon_bmu_id"].to_list() == ["E_NEWB-1"]
    assert route["nearest_ecr_row"].to_list() == [9]
    assert route["capacity_ratio"].to_list() == [pytest.approx(0.99)]


def test_repd_batteries_keep_operational_batteries_in_england_and_wales(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = pl.DataFrame(
        {
            "Site Name": ["A", "B", "C", "D", "E"],
            "Technology Type": ["Battery", "Battery", "Battery", "Solar Photovoltaics", "Battery"],
            "Development Status (short)": [
                "Operational",
                "Awaiting Construction",
                "Operational",
                "Operational",
                "Operational",
            ],
            "Country": ["England", "England", "Scotland", "England", "Wales"],
            "Installed Capacity (MWelec)": ["5", "6", "7", "8", "9"],
            "X-coordinate": ["100", "200", "300", "400", "500"],
            "Y-coordinate": ["10", "20", "30", "40", "50"],
        }
    )
    buffer = io.StringIO()
    rows.write_csv(buffer)
    path = tmp_path / "repd.json"
    path.write_text(json.dumps({"body": buffer.getvalue()}))
    monkeypatch.setattr(census, "REPD_PATH", path)

    result = census.repd_batteries()

    assert result["repd_site_name"].to_list() == ["A", "E"]
    assert result["repd_capacity_mw"].to_list() == [5.0, 9.0]


def _classified_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    ecr_storage = _connected(
        [
            {"customer_name": "Alpha Ltd", "maximum_export_capacity_mw": 40.0},
            {"customer_name": "Beta Ltd", "maximum_export_capacity_mw": 20.0},
            {"customer_name": "Gamma Ltd", "maximum_export_capacity_mw": 10.0},
            {"customer_name": "Delta Ltd", "maximum_export_capacity_mw": 30.0},
        ]
    )
    bmus = _bmus(
        [
            {"elexon_bmu_id": "E_A-1", "pn_rows": 100},
            {"elexon_bmu_id": "E_B-1", "pn_rows": 0},
        ]
    )
    accepted = pl.DataFrame(
        {"elexon_bmu_id": ["E_A-1", "E_B-1"], "ecr_row": [0, 1]},
        schema_overrides={"ecr_row": pl.UInt32},
    )
    return ecr_storage, bmus, accepted


def test_classes_separate_fpn_submitters_other_bmus_response_providers_and_the_rest() -> None:
    ecr_storage, bmus, accepted = _classified_inputs()

    classified = census.classify_connected_storage(
        ecr_storage=ecr_storage,
        accepted_matches=accepted,
        bmus=bmus,
        participants=frozenset({"GAMMA LTD"}),
    ).sort("ecr_row")

    assert classified["bmu_class"].to_list() == [
        "own BMU, FPN submitted",
        "own BMU, no FPN",
        "response provider, no BMU found",
        "no BMU found",
    ]


def test_a_matched_row_stays_a_bmu_even_if_its_customer_sells_response() -> None:
    ecr_storage, bmus, accepted = _classified_inputs()

    classified = census.classify_connected_storage(
        ecr_storage=ecr_storage,
        accepted_matches=accepted,
        bmus=bmus,
        participants=frozenset({"ALPHA LTD"}),
    ).sort("ecr_row")

    assert classified["bmu_class"][0] == "own BMU, FPN submitted"


def test_two_bmus_matched_to_one_row_give_that_row_one_class() -> None:
    ecr_storage, bmus, _ = _classified_inputs()
    accepted = pl.DataFrame(
        {"elexon_bmu_id": ["E_B-1", "E_A-1"], "ecr_row": [0, 0]},
        schema_overrides={"ecr_row": pl.UInt32},
    )

    classified = census.classify_connected_storage(
        ecr_storage=ecr_storage, accepted_matches=accepted, bmus=bmus, participants=frozenset()
    )

    assert classified.height == ecr_storage.height


def test_the_share_band_adds_the_missed_share_of_the_unknown_rows_to_the_upper_bound() -> None:
    ecr_storage, bmus, accepted = _classified_inputs()
    classified = census.classify_connected_storage(
        ecr_storage=ecr_storage, accepted_matches=accepted, bmus=bmus, participants=frozenset()
    )

    band = census.bmu_share_band(classified=classified, recall=0.5)

    assert band["count_low"] == pytest.approx(2 / 4)
    assert band["count_high"] == pytest.approx((2 + 0.5 * 2) / 4)
    assert band["megawatts_low"] == pytest.approx(60 / 100)
    assert band["megawatts_high"] == pytest.approx((60 + 0.5 * 40) / 100)


def test_a_perfect_recall_leaves_no_unknown_band() -> None:
    ecr_storage, bmus, accepted = _classified_inputs()
    classified = census.classify_connected_storage(
        ecr_storage=ecr_storage, accepted_matches=accepted, bmus=bmus, participants=frozenset()
    )

    band = census.bmu_share_band(classified=classified, recall=1.0)

    assert band["count_low"] == band["count_high"]
    assert band["megawatts_low"] == band["megawatts_high"]
