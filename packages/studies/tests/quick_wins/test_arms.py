"""Tests for how arms merge their differences from production."""

import arms
import polars as pl
import pytest
from fit_arms import feature_matrix


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


def test_an_enum_column_is_coded_by_its_position_and_nulls_stay_null():
    """Monday is 0 and Sunday is 6 in every frame, whatever values the frame holds."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    frame = pl.DataFrame(
        {"local_day_of_week": pl.Series(["Sunday", "Monday"]).cast(pl.Enum(days)), "x": [1, None]}
    )
    matrix = feature_matrix(frame=frame, columns=["local_day_of_week", "x"])
    assert matrix["local_day_of_week"].to_list() == [6.0, 0.0]
    assert matrix["x"].to_list() == [1.0, None]
    assert matrix.dtypes == [pl.Float32, pl.Float32]


def test_lags_removed_only_drops_the_four_lags_and_keeps_the_weekday_columns():
    """A1b differs from production in exactly the four power-lag columns, for every series type."""
    arm = arms.ALL_ARMS["a1b_lags_removed_only"]
    for series_type in ("PV", "Wind", "Disaggregated Demand"):
        columns = set(arm.columns(series_type))
        assert set(arms.PRODUCTION_COLUMNS) - columns == arms.LAG_COLUMNS
        assert columns >= arms.WEEKDAY_COLUMNS
        assert len(columns) == len(arms.PRODUCTION_COLUMNS) - 4


def test_the_padding_control_has_the_lead_time_arms_column_count():
    """The control adds one column, as C1 does, and the new column is the shuffled copy."""
    control = arms.ALL_ARMS["c1p_shuffled_copy_control"]
    lead = arms.ALL_ARMS["c1_lead_time"]
    for series_type in ("PV", "Wind", "Disaggregated Demand"):
        assert len(control.columns(series_type)) == len(lead.columns(series_type))
        assert control.columns(series_type)[-1] == arms.SHUFFLED_COPY_COLUMN
    assert arms.needs_shuffled_copy(control)
    assert not arms.needs_shuffled_copy(lead)


def test_the_shuffled_copy_is_a_fixed_permutation_of_the_source_column():
    """The copy holds the same values in a different order, the same order on every call."""
    from fit_arms import add_shuffled_copy

    frame = pl.DataFrame({arms.SHUFFLED_COPY_SOURCE: [float(i) for i in range(50)]})
    first = add_shuffled_copy(frame=frame)[arms.SHUFFLED_COPY_COLUMN]
    second = add_shuffled_copy(frame=frame)[arms.SHUFFLED_COPY_COLUMN]
    assert first.to_list() == second.to_list()
    assert sorted(first.to_list()) == frame[arms.SHUFFLED_COPY_SOURCE].to_list()
    assert first.to_list() != frame[arms.SHUFFLED_COPY_SOURCE].to_list()
