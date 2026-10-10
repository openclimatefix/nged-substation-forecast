from itertools import pairwise

import pytest
from studies.ifs_ladder import (
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
