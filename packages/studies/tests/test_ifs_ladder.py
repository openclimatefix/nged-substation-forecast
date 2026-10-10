from itertools import pairwise

import pytest
from studies.ifs_ladder import (
    GROUP_CONTROL_RUNGS,
    MISSING_BY_DESIGN,
    OPEN_DATA_AVAILABILITY,
    PRODUCTION_VARIABLES,
    RUNG_ADDITIONS,
    RUNGS,
    SHARED_FEATURES,
    RungType,
    cloud_cover_control_features,
    cloud_layers_control_features,
    drop_one_group_features,
    era5_comparison_features,
    group_control_features,
    later_groups_columns,
    later_groups_control_features,
    partner_control_features,
    positive_control_features,
    production_features,
    rung_features,
    rung_variables,
    with_partner_features,
)


def test_the_full_set_holds_the_twenty_variables_of_the_plan():
    variables = rung_variables(rung="f6")

    assert len(variables) == 20
    assert len(set(variables)) == 20
    assert set(variables) == set(OPEN_DATA_AVAILABILITY)


def test_each_rung_adds_to_the_rung_below_without_repeating_a_column():
    for lower, upper in pairwise(RUNGS):
        assert rung_features(rung=upper)[: len(rung_features(rung=lower))] == rung_features(
            rung=lower
        )
        assert len(set(rung_features(rung=upper))) == len(rung_features(rung=upper))


def test_every_negative_control_has_as_many_columns_as_the_arm_it_pads():
    assert len(later_groups_control_features()) == len(rung_features(rung="f6"))
    assert len(cloud_cover_control_features()) == len(rung_features(rung="f1"))
    assert len(cloud_layers_control_features()) == len(rung_features(rung="f2"))
    assert len(partner_control_features(rung="f0")) == len(with_partner_features(rung="f0"))
    assert len(partner_control_features(rung="f6")) == len(with_partner_features(rung="f6"))


def test_a_control_keeps_the_base_columns_and_adds_only_permuted_copies():
    control = later_groups_control_features()
    base = rung_features(rung="f2")

    assert control[: len(base)] == base
    assert all(name.endswith("_shuffled") for name in control[len(base) :])
    assert not set(control[len(base) :]) & set(rung_variables(rung="f6"))


def test_the_production_reference_is_the_minimal_set_and_the_four_production_variables():
    assert production_features() == (*rung_features(rung="f0"), *PRODUCTION_VARIABLES)
    assert len(production_features()) == len(rung_features(rung="f0")) + 4


@pytest.mark.parametrize("dropped", RUNGS[1:])
def test_dropping_a_group_removes_exactly_that_groups_variables(dropped: RungType):
    kept = drop_one_group_features(dropped=dropped)
    full = rung_features(rung="f6")

    assert set(full) - set(kept) == set(RUNG_ADDITIONS[dropped])
    assert len(kept) == len(full) - len(RUNG_ADDITIONS[dropped])


def test_the_minimal_set_cannot_be_dropped():
    with pytest.raises(ValueError, match="cannot be dropped"):
        drop_one_group_features(dropped="f0")


def test_the_positive_control_adds_one_column_to_f2():
    assert positive_control_features() == (*rung_features(rung="f2"), "cams_ghi_w_m2")


def test_an_era5_arm_gets_the_shared_columns_and_prefixed_variables_without_derived_indices():
    columns = era5_comparison_features(rung="g1")

    assert columns[: len(SHARED_FEATURES)] == SHARED_FEATURES
    assert columns[len(SHARED_FEATURES) :] == ("era5_ssrd", "era5_t2m", "era5_tcc")
    assert "era5_cape" in era5_comparison_features(rung="g9")
    assert "cape" not in era5_comparison_features(rung="g9")


def test_the_rungs_are_the_plans_table():
    assert RUNG_ADDITIONS == {
        "f0": ("shortwave_radiation", "temperature_2m"),
        "f1": ("cloud_cover",),
        "f2": ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"),
        "f3": ("direct_radiation",),
        "f4": ("dew_point_2m", "total_column_integrated_water_vapour", "boundary_layer_height"),
        "f5": ("cape", "convective_inhibition", "visibility"),
        "f6": (
            "surface_temperature",
            "snow_depth",
            "snowfall",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_gusts_10m",
        ),
    }
    assert PRODUCTION_VARIABLES == (
        "dew_point_2m",
        "surface_pressure",
        "precipitation",
        "wind_speed_10m",
    )
    assert SHARED_FEATURES[-1] == "era_code"
    assert MISSING_BY_DESIGN == ("convective_inhibition",)
    assert {name for name, state in OPEN_DATA_AVAILABILITY.items() if state == "not carried"} == {
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "direct_radiation",
        "boundary_layer_height",
    }


def test_a_group_control_pads_f2_with_only_that_groups_permuted_variables():
    for rung in GROUP_CONTROL_RUNGS:
        control = group_control_features(rung=rung)
        base = rung_features(rung="f2")

        assert control[: len(base)] == base
        assert control[len(base) :] == tuple(f"{name}_shuffled" for name in RUNG_ADDITIONS[rung])
        assert len(control) == len(base) + len(RUNG_ADDITIONS[rung])
        # Every permuted column exists in the later-groups control's permutation.
        assert set(control[len(base) :]) <= {f"{name}_shuffled" for name in later_groups_columns()}


def test_a_rung_without_a_group_control_is_refused():
    with pytest.raises(ValueError, match="no group control"):
        group_control_features(rung="f6")
