"""Tests for how arms merge their differences from production."""

import arms
import pytest


def test_a_later_part_that_keeps_a_column_does_not_restore_one_an_earlier_part_removed():
    """A1 removes the power lags from PV; the lead-time arm leaves them, but they stay removed."""
    combined = arms.combine(
        name="x",
        parts=(arms.ALL_ARMS["a1_generators_without_lags"], arms.ALL_ARMS["c1_lead_time"]),
        description="",
    )
    pv = combined.columns("PV")
    assert not set(pv) & arms.LAG_COLUMNS
    assert "nwp_lead_time_hours" in pv
    demand = combined.columns("Disaggregated Demand")
    assert set(demand) >= arms.LAG_COLUMNS
    assert "nwp_lead_time_hours" in demand


def test_column_lists_have_no_duplicates_and_keep_production_order_first():
    """Merging three parts yields each column once, production columns first."""
    combined = arms.combine(
        name="x",
        parts=(
            arms.ALL_ARMS["a2_per_type_physics"],
            arms.ALL_ARMS["c1_lead_time"],
            arms.ALL_ARMS["c7_calendar"],
        ),
        description="",
    )
    for series_type in ("PV", "Wind", "Disaggregated Demand"):
        columns = combined.columns(series_type)
        assert len(columns) == len(set(columns))
        kept = [c for c in columns if c in arms.PRODUCTION_COLUMNS]
        assert kept == [c for c in arms.PRODUCTION_COLUMNS if c in kept]
    assert "is_bank_holiday" in combined.columns("Disaggregated Demand")
    assert "is_bank_holiday" not in combined.columns("PV")
    assert "clear_sky_index" in combined.columns("PV")


def test_the_absolute_error_objective_survives_a_later_production_like_part():
    """The objective takes the last non-default value, not the last part's."""
    combined = arms.combine(
        name="x",
        parts=(arms.ALL_ARMS["c4_absolute_error"], arms.ALL_ARMS["c1_lead_time"]),
        description="",
    )
    assert combined.objective == "reg:absoluteerror"


def test_cleaning_rules_of_all_parts_apply_together_without_repeats():
    """Two cleaning parts give the first as `cleaning` and the second as `extra_cleaning`."""
    combined = arms.combine(
        name="x",
        parts=(
            arms.ALL_ARMS["clean_first_2_months"],
            arms.ALL_ARMS["clean_spikes"],
            arms.ALL_ARMS["clean_first_2_months"],
        ),
        description="",
    )
    assert combined.cleaning == "first_2_months"
    assert combined.extra_cleaning == ("spikes",)


def test_every_arm_other_than_the_stored_reference_uses_full_column_sampling():
    """Contrasts need colsample 1.0 so that a wider arm cannot win from column subsampling."""
    for arm in arms.ALL_ARMS.values():
        expected = 0.8 if arm.name == "production_colsample0.8" else 1.0
        assert arm.colsample_bytree == pytest.approx(expected), arm.name
