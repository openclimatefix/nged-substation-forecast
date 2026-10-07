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
