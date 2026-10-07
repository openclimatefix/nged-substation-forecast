"""Tests for the pure functions of `studies/solar_bmu_census/collate.py`.

Each test fails on the bug it exists for. Among those bugs are a generic word that stops two
spellings of one site name matching, a threshold loose enough to match a different farm, a hybrid
read as pure PV, and a grid position read as degrees.
"""

import collate
import polars as pl


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
