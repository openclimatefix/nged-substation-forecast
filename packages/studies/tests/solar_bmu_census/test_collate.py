"""Tests for the pure functions of `studies/solar_bmu_census/collate.py`.

Each test fails on the bug it exists for. Among those bugs are a generic word that stops two
spellings of one site name matching, a threshold loose enough to match a different farm, a hybrid
read as pure PV, and a grid position read as degrees.
"""

import collate
import polars as pl
import pytest

PV = "PV Array (Photo Voltaic/solar)"


def test_a_spelling_without_a_space_still_matches() -> None:
    assert collate.best_match(
        site_name="Beechgreen Energy Farm", candidates={"a": "Beechgreen Energyfarm"}
    )


def test_a_different_farm_with_a_similar_name_does_not_match() -> None:
    assert collate.best_match(site_name="Beechgreen", candidates={"a": "Beechgrove"}) is None


def test_the_best_of_several_candidates_wins() -> None:
    match = collate.best_match(
        site_name="Larks Green Solar", candidates={"a": "Larks Green Solar Farm", "b": "Larkfield"}
    )
    assert match is not None
    assert match[0] == "a"


def test_a_name_made_only_of_generic_words_matches_nothing() -> None:
    assert collate.best_match(site_name="Solar Farm", candidates={"a": "Solar Farm"}) is None


def test_pv_with_storage_is_a_hybrid() -> None:
    plant_type = "Energy Storage System;PV Array (Photo Voltaic/solar)"
    assert collate.technology_from_tec_plant_type(plant_type=plant_type) == "hybrid"


def test_pv_with_demand_only_is_pure_pv() -> None:
    plant_type = "Demand;PV Array (Photo Voltaic/solar)"
    assert collate.technology_from_tec_plant_type(plant_type=plant_type) == "pure PV"


def test_pv_alone_is_pure_pv() -> None:
    plant_type = "PV Array (Photo Voltaic/solar)"
    assert collate.technology_from_tec_plant_type(plant_type=plant_type) == "pure PV"


def test_a_plant_type_without_pv_is_unknown() -> None:
    assert collate.technology_from_tec_plant_type(plant_type="Wind Onshore") == "unknown"


def test_osgb_position_converts_to_degrees_in_london() -> None:
    longitude, latitude = collate.osgb_to_lon_lat(easting_m=530000, northing_m=180000)
    assert -0.3 < longitude < 0.1
    assert 51.4 < latitude < 51.6


def test_connection_type_follows_the_prefix() -> None:
    assert collate.connection_type(elexon_bmu_id="T_X") == "transmission-connected"
    assert collate.connection_type(elexon_bmu_id="E_X") == "embedded"
    assert collate.connection_type(elexon_bmu_id="2__X") == "other"


def test_a_project_with_a_later_stage_keeps_its_built_row() -> None:
    """A later stage's cumulative capacity includes capacity not yet built, in either row order."""
    tec = pl.DataFrame(
        {
            "Project ID": ["a", "a", "b", "b"],
            "Project Name": ["Farm", "Farm", "Other", "Other"],
            "Plant Type": ["PV Array (Photo Voltaic/solar)"] * 4,
            "Project Status": ["Built", "Consents Approved", "Consents Approved", "Built"],
            "MW Connected": ["99.4", "0", "0", "50"],
            "Cumulative Total Capacity (MW)": ["99.4", "120", "70", "50"],
        }
    )
    rows = collate.best_tec_rows(tec=tec).sort("Project ID")
    assert rows["tec_mw"].to_list() == [99.4, 50.0]


def test_a_built_project_uses_its_connected_capacity_and_an_unbuilt_one_its_agreed_capacity() -> (
    None
):
    """A built row's cumulative figure can include a later increase, such as 200 MW due in 2036."""
    tec = pl.DataFrame(
        {
            "Project ID": ["built", "unbuilt"],
            "Project Name": ["Built", "Unbuilt"],
            "Plant Type": ["PV Array (Photo Voltaic/solar)"] * 2,
            "Project Status": ["Built", "Under Construction/Commissioning"],
            "MW Connected": ["350", "0"],
            "Cumulative Total Capacity (MW)": ["550", "20.62"],
        }
    )
    rows = collate.best_tec_rows(tec=tec).sort("Project ID")
    assert rows["tec_mw"].to_list() == [350.0, 20.62]


def test_storage_evidence_is_graded_from_the_strongest_first() -> None:
    plant_type = "Energy Storage System;PV Array (Photo Voltaic/solar)"

    def evidence(*, bmu: bool, battery: str | None, plant: str | None) -> tuple[str, str]:
        return collate.site_technology(
            tec_plant_type=plant,
            storage_bmu_with_output=bmu,
            repd_battery_status=battery,
            repd_solar_found=True,
        )

    assert evidence(bmu=True, battery="Operational", plant=plant_type) == (
        "hybrid",
        "storage BMU with output",
    )
    assert (
        evidence(bmu=False, battery="Operational", plant=None)[1] == "operational battery in REPD"
    )
    assert evidence(bmu=False, battery="Under Construction", plant=None) == (
        "hybrid",
        "storage planned or under construction",
    )
    assert evidence(bmu=False, battery=None, plant=plant_type)[0] == "hybrid"
    assert evidence(bmu=False, battery=None, plant="PV Array (Photo Voltaic/solar)") == (
        "pure PV",
        "TEC plant type lists PV only",
    )
    assert evidence(bmu=False, battery=None, plant=None) == (
        "pure PV",
        "REPD solar row, no battery row",
    )


def test_a_near_spelling_at_0_95_matches_and_the_score_is_rounded() -> None:
    match = collate.best_match(
        site_name="Longfield Solar", candidates={"k": "Long Field 2 Solar Farm"}
    )
    assert match == ("k", 0.95)


def test_a_tie_goes_to_the_key_that_sorts_first() -> None:
    match = collate.best_match(
        site_name="Larks Green", candidates={"b": "Larks Green", "a": "Larks Green"}
    )
    assert match == ("a", 1.0)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Upvale Farm", "upvale"),
        ("Hillphotovoltaics", "hill"),
        ("Foo PV Ltd", "foo"),
        ("Beech Green", "beechgreen"),
    ],
)
def test_normalise_on_literal_names(name: str, expected: str) -> None:
    assert collate.normalise(name) == expected


def test_pv_with_reactive_compensation_is_pure_pv() -> None:
    plant_type = "Reactive Compensation;PV Array (Photo Voltaic/solar)"
    assert collate.technology_from_tec_plant_type(plant_type=plant_type) == "pure PV"


def test_display_name_falls_back_to_the_tec_name_then_the_repd_name() -> None:
    def name(*, site_name: str, tec_name: str | None) -> str:
        return collate._display_name(
            site_name=site_name, bmu_id="T_X", tec_name=tec_name, repd_name="Repd"
        )

    assert name(site_name="Real", tec_name="Tec") == "Real"
    assert name(site_name="T_X", tec_name="Tec") == "Tec"
    assert name(site_name="", tec_name=None) == "Repd"


def test_a_cfd_unit_s_connection_sets_the_connection_of_a_c_bmu() -> None:
    assert (
        collate.connection_type(elexon_bmu_id="C__X", cfd_connection="Distribution") == "embedded"
    )
    assert (
        collate.connection_type(elexon_bmu_id="C__X", cfd_connection="Transmission")
        == "transmission-connected"
    )
    assert collate.connection_type(elexon_bmu_id="C__X", cfd_connection=None) == "other"


def test_a_connection_type_prefix_needs_its_underscore() -> None:
    assert collate.connection_type(elexon_bmu_id="TX-1") == "other"
    assert collate.connection_type(elexon_bmu_id="EX-1") == "other"


def test_best_tec_rows_ranks_statuses_and_picks_the_capacity_column() -> None:
    tec = pl.DataFrame(
        {
            "Project ID": ["built", "built", "uc", "uc", "ca", "ca", "odd", "odd", "wind"],
            "Project Name": ["B", "B", "U", "U", "C", "C", "O", "O", "W"],
            "Plant Type": [PV] * 8 + ["Wind Onshore"],
            "Project Status": [
                "Under Construction/Commissioning",
                "Built",
                "Under Construction/Commissioning",
                "Scoping",
                "Awaiting Consents",
                "Consents Approved",
                "Withdrawn",
                "Built",
                "Built",
            ],
            "MW Connected": ["0", "350", "0", "0", "0", "0", "0", "40", "10"],
            "Cumulative Total Capacity (MW)": [
                "600",
                "550",
                "20.62",
                "57",
                "8",
                "9",
                "1",
                "40",
                "10",
            ],
        }
    )
    rows = collate.best_tec_rows(tec=tec).sort("Project ID")
    assert rows.select("Project ID", "Project Status", "tec_mw").to_dicts() == [
        {"Project ID": "built", "Project Status": "Built", "tec_mw": 350.0},
        {"Project ID": "ca", "Project Status": "Consents Approved", "tec_mw": 9.0},
        {"Project ID": "odd", "Project Status": "Built", "tec_mw": 40.0},
        {"Project ID": "uc", "Project Status": "Under Construction/Commissioning", "tec_mw": 20.62},
    ]


def _census(*, rows: list[dict[str, object]]) -> pl.DataFrame:
    columns = [*collate.DISPARITY_FIGURES, "elexon_bmu_id", "display_name", "scope", "basis"]
    return pl.DataFrame(
        [{column: row.get(column) for column in columns} for row in rows],
        schema={
            **dict.fromkeys(collate.DISPARITY_FIGURES, pl.Float64),
            "elexon_bmu_id": pl.String,
            "display_name": pl.String,
            "scope": pl.String,
            "basis": pl.String,
        },
    )


def test_capacity_disparity_ranks_by_highest_over_lowest_of_the_figures_that_exist() -> None:
    census = _census(
        rows=[
            {
                "elexon_bmu_id": "T_A",
                "scope": "single-site",
                "basis": "behaviour only",
                "generation_capacity_mw": 50.0,
                "tec_mw": 100.0,
                "p99_output_mw": 40.0,
            },
            {
                "elexon_bmu_id": "T_B",
                "scope": "single-site",
                "basis": "type and behaviour",
                "generation_capacity_mw": 10.0,
                "largest_mel_mw": 0.0,  # a zero is left out of the ratio
                "repd_installed_capacity_mw": 40.0,
            },
            {
                "elexon_bmu_id": "T_ONE_FIGURE",
                "scope": "single-site",
                "basis": "behaviour only",
                "tec_mw": 90.0,
            },
            {
                "elexon_bmu_id": "T_TYPE_ONLY",
                "scope": "single-site",
                "basis": "type only",
                "generation_capacity_mw": 1.0,
                "tec_mw": 900.0,
            },
            {
                "elexon_bmu_id": "2__AGG",
                "scope": "aggregate",
                "basis": "behaviour only",
                "generation_capacity_mw": 1.0,
                "tec_mw": 900.0,
            },
        ]
    )
    ranked = collate.capacity_disparity(table=census)
    assert ranked["elexon_bmu_id"].to_list() == ["T_B", "T_A"]
    assert ranked["ratio"].to_list() == [4.0, 2.5]
    assert ranked["lowest_mw"].to_list() == [10.0, 40.0]
    assert ranked["highest_mw"].to_list() == [40.0, 100.0]


def test_capacity_disparity_breaks_a_tie_by_identifier() -> None:
    rows: list[dict[str, object]] = [
        {
            "elexon_bmu_id": bmu,
            "scope": "single-site",
            "basis": "behaviour only",
            "generation_capacity_mw": 10.0,
            "tec_mw": 20.0,
        }
        for bmu in ("T_B", "T_A")
    ]
    ranked = collate.capacity_disparity(table=_census(rows=rows))
    assert ranked["elexon_bmu_id"].to_list() == ["T_A", "T_B"]


def test_the_gsp_groups_map_to_the_dno_licence_areas_in_neso_s_geojson() -> None:
    """The expected values are the `Name`, `DNO`, and `Area` of NESO's licence-area GeoJSON.

    The file is `gb-dno-license-areas-20240503-as-geojson.geojson`, at
    <https://neso.energy/data-portal/gis-boundaries-gb-dno-license-areas>. The literals below were
    read from its 14 features, so a mapping edited in `collate.py` alone fails this test.
    """
    assert {group: area for group, (_, area) in collate.GSP_GROUP_AREAS.items()} == {
        "_A": "UKPN",
        "_B": "NGED",
        "_C": "UKPN",
        "_D": "SP Energy Networks",
        "_E": "NGED",
        "_F": "Northern Powergrid",
        "_G": "Electricity North West",
        "_H": "SSEN",
        "_J": "UKPN",
        "_K": "NGED",
        "_L": "NGED",
        "_M": "Northern Powergrid",
        "_N": "SP Energy Networks",
        "_P": "SSEN",
    }


def test_the_gsp_group_area_names_are_those_of_neso_s_geojson() -> None:
    assert {group: area for group, (area, _) in collate.GSP_GROUP_AREAS.items()} == {
        "_A": "East England",
        "_B": "East Midlands",
        "_C": "London",
        "_D": "North Wales, Merseyside and Cheshire",
        "_E": "West Midlands",
        "_F": "North East England",
        "_G": "North West England",
        "_H": "Southern England",
        "_J": "South East England",
        "_K": "South Wales",
        "_L": "South West England",
        "_M": "Yorkshire",
        "_N": "South and Central Scotland",
        "_P": "North Scotland",
    }


def _square(*, west: float, south: float, east: float, north: float) -> list[list[float]]:
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


LICENCE_AREAS = {
    "features": [
        {
            "properties": {"Name": "_X"},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [
                        _square(west=0, south=0, east=10, north=10),
                        _square(west=4, south=4, east=6, north=6),
                    ],
                    [_square(west=20, south=0, east=30, north=10)],
                ],
            },
        },
        {
            "properties": {"Name": "_Y"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [_square(west=10.5, south=0, east=15, north=10)],
            },
        },
    ]
}


@pytest.mark.parametrize(
    ("easting", "northing", "expected"),
    [
        (2, 2, "_X"),
        (25, 5, "_X"),
        (12, 5, "_Y"),
        (5, 5, None),
        (10.2, 5, None),
        (-1, 5, None),
        (2, 11, None),
    ],
)
def test_a_position_is_in_the_licence_area_that_contains_it(
    easting: float, northing: float, expected: str | None
) -> None:
    found = collate.licence_area_at(easting_m=easting, northing_m=northing, areas=LICENCE_AREAS)
    assert (found["Name"] if found else None) == expected


def test_the_distance_to_other_areas_skips_the_areas_own_edges() -> None:
    # (9, 5) is 1 m from _X's east edge, which is skipped, and 1.5 m from _Y's west edge.
    assert collate.distance_to_other_areas_m(
        easting_m=9, northing_m=5, areas=LICENCE_AREAS, own_name="_X"
    ) == pytest.approx(1.5)
    # (12, 5) is 2 m from _X's east edge, and 1.5 m from _Y's own west edge, which is skipped.
    assert collate.distance_to_other_areas_m(
        easting_m=12, northing_m=5, areas=LICENCE_AREAS, own_name="_Y"
    ) == pytest.approx(2.0)


def test_the_distance_to_an_area_past_the_end_of_an_edge_is_to_the_edges_end_point() -> None:
    # (10.2, 12) lies above _X's top edge, which ends at the corner (10, 10), so the distance is
    # to that corner: hypot(0.2, 2).
    assert collate.distance_to_other_areas_m(
        easting_m=10.2, northing_m=12, areas=LICENCE_AREAS, own_name="_Z"
    ) == pytest.approx(2.00998, abs=1e-4)


def test_dno_area_is_empty_for_a_missing_or_unknown_gsp_group() -> None:
    assert collate.dno_area(gsp_group_id="_L") == "NGED"
    assert collate.dno_area(gsp_group_id=None) is None
    assert collate.dno_area(gsp_group_id="_Z") is None
