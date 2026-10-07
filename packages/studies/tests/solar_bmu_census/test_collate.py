"""Tests for the pure functions of `studies/solar_bmu_census/collate.py`.

Each test fails on the bug it exists for: a generic word that stops two spellings of one site name
matching, a threshold loose enough to match a different farm, a hybrid read as pure PV, and a grid
position read as degrees.
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
    """A later stage's cumulative capacity includes capacity not yet built."""
    tec = pl.DataFrame(
        {
            "Project ID": ["a", "a", "b"],
            "Project Name": ["Farm", "Farm", "Other"],
            "Plant Type": ["PV Array (Photo Voltaic/solar)"] * 3,
            "Project Status": ["Built", "Consents Approved", "Built"],
            "Cumulative Total Capacity (MW)": ["99.4", "120", "50"],
        }
    )
    rows = collate.best_tec_rows(tec=tec)
    assert rows.height == 2
    assert rows.filter(pl.col("Project ID") == "a")["Cumulative Total Capacity (MW)"].to_list() == [
        "99.4"
    ]
