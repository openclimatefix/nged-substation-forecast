from datetime import datetime

import polars as pl
import pytest
from studies.bootstrap import paired_differences
from studies.ifs_decisions import (
    EvidenceClassType,
    combine_verdicts,
    contrast_verdict,
    display_class,
    drop_one_gain_at_settings,
    drop_one_margin,
    drop_one_matters,
    evidence_class,
    exploratory_gain,
    gain_after_control,
    half_width_allows_no_gain,
    is_near_line,
    mean_absolute_error_on_common_rows,
    mean_paired_difference_by,
    merge_extra_arms,
    percentage_rise,
    priority_order,
    production_scope_sentence,
    second_feed_recommended,
    second_forecast_or_mars_variables,
    stack_lead_days,
)

SMALLEST = 0.001


def test_an_interval_wholly_below_minus_the_smallest_effect_is_a_gain():
    assert contrast_verdict(lower=-0.004, upper=-0.0011, smallest_effect=SMALLEST) == "gain"


def test_an_upper_bound_exactly_on_minus_the_smallest_effect_is_not_a_gain():
    assert contrast_verdict(lower=-0.004, upper=-SMALLEST, smallest_effect=SMALLEST) == "unresolved"


def test_a_lower_bound_exactly_on_minus_the_smallest_effect_is_not_no_gain():
    assert contrast_verdict(lower=-SMALLEST, upper=0.002, smallest_effect=SMALLEST) == "unresolved"


def test_an_interval_that_rules_out_the_smallest_gain_is_no_gain_even_if_it_straddles_zero():
    assert contrast_verdict(lower=-0.0005, upper=0.0003, smallest_effect=SMALLEST) == "no gain"


def test_an_interval_wholly_above_zero_is_no_gain_and_never_a_gain():
    assert contrast_verdict(lower=0.001, upper=0.003, smallest_effect=SMALLEST) == "no gain"


def test_an_interval_that_straddles_minus_the_smallest_effect_is_unresolved():
    assert contrast_verdict(lower=-0.003, upper=0.001, smallest_effect=SMALLEST) == "unresolved"


def test_a_verdict_needs_both_settings_to_return_it():
    assert combine_verdicts(verdicts=["gain", "gain"]) == "gain"
    assert combine_verdicts(verdicts=["no gain", "no gain"]) == "no gain"
    assert combine_verdicts(verdicts=["unresolved", "unresolved"]) == "unresolved"


def test_one_setting_disagreeing_leaves_the_verdict_unresolved():
    assert combine_verdicts(verdicts=["gain", "unresolved"]) == "unresolved"
    assert combine_verdicts(verdicts=["gain", "no gain"]) == "unresolved"
    assert combine_verdicts(verdicts=["no gain", "gain"]) == "unresolved"


def test_a_missing_second_setting_leaves_the_verdict_unresolved():
    assert combine_verdicts(verdicts=["gain"]) == "unresolved"
    assert combine_verdicts(verdicts=[]) == "unresolved"


def test_three_settings_are_not_two_and_leave_the_verdict_unresolved():
    assert combine_verdicts(verdicts=["gain"] * 3) == "unresolved"


def test_a_gain_that_is_not_a_gain_over_the_control_becomes_unresolved():
    assert gain_after_control(versus_reference="gain", versus_control="unresolved") == "unresolved"
    assert gain_after_control(versus_reference="gain", versus_control="no gain") == "unresolved"


def test_a_gain_over_both_the_reference_and_the_control_stands():
    assert gain_after_control(versus_reference="gain", versus_control="gain") == "gain"


def test_the_control_never_turns_a_verdict_that_is_not_a_gain_into_one():
    assert gain_after_control(versus_reference="no gain", versus_control="gain") == "no gain"
    assert gain_after_control(versus_reference="unresolved", versus_control="gain") == "unresolved"


def test_a_second_forecast_gain_recommends_a_second_forecast_whatever_the_mars_interval_says():
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="gain", mars_uppers=[-0.002, -0.002], smallest_effect=SMALLEST
    )

    assert outcome == "second_forecast"


def test_no_second_forecast_gain_and_a_mars_gain_at_both_settings_recommends_mars_variables():
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="no gain", mars_uppers=[-0.0011, -0.0015], smallest_effect=SMALLEST
    )

    assert outcome == "mars_variables"


def test_the_era5_p4_interval_that_reaches_minus_0_08_points_does_not_separate_the_routes():
    # The ERA5 study's P4 at the main setting has an adjusted upper bound of -0.08 points of
    # capacity, which is above minus the smallest effect of 0.1 points.
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="no gain", mars_uppers=[-0.0008, -0.0006], smallest_effect=SMALLEST
    )

    assert outcome == "not_separated"


def test_a_mars_upper_bound_exactly_on_minus_the_smallest_effect_does_not_separate_the_routes():
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="no gain",
        mars_uppers=[-SMALLEST, -SMALLEST],
        smallest_effect=SMALLEST,
    )

    assert outcome == "not_separated"


def test_a_mars_gain_at_only_one_setting_does_not_separate_the_routes():
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="no gain", mars_uppers=[-0.002, -0.0005], smallest_effect=SMALLEST
    )

    assert outcome == "not_separated"


def test_a_mars_gain_at_a_single_setting_does_not_separate_the_routes():
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="no gain", mars_uppers=[-0.002], smallest_effect=SMALLEST
    )

    assert outcome == "not_separated"


def test_an_unresolved_second_forecast_verdict_never_recommends_mars_variables():
    outcome = second_forecast_or_mars_variables(
        second_forecast_verdict="unresolved", mars_uppers=[-0.002, -0.002], smallest_effect=SMALLEST
    )

    assert outcome == "not_separated"


def test_cloud_layers_that_are_a_gain_recommend_a_second_feed():
    assert second_feed_recommended(cloud_layers_verdict="gain", missing_group_drop_one_gains=[])


def test_a_missing_group_that_carries_a_gain_recommends_a_second_feed():
    assert second_feed_recommended(
        cloud_layers_verdict="unresolved", missing_group_drop_one_gains=[False, True]
    )


def test_no_gain_in_the_layers_or_any_missing_group_recommends_no_second_feed():
    assert not second_feed_recommended(
        cloud_layers_verdict="no gain", missing_group_drop_one_gains=[False, False]
    )
    assert not second_feed_recommended(
        cloud_layers_verdict="unresolved", missing_group_drop_one_gains=[]
    )


def test_an_exploratory_gain_needs_every_bound_below_minus_the_smallest_effect_and_the_control():
    assert exploratory_gain(
        uppers=[-0.1, -0.2, -0.05, -0.3],
        difference=-0.002,
        control_difference=0.001,
        smallest_effect=SMALLEST,
    )


def test_a_bound_on_minus_the_smallest_effect_or_above_removes_an_exploratory_gain():
    for bad in (-SMALLEST, -0.0005, 0.0, 0.05):
        assert not exploratory_gain(
            uppers=[-0.1, bad],
            difference=-0.002,
            control_difference=0.001,
            smallest_effect=SMALLEST,
        )


def test_a_gain_smaller_than_the_smallest_effect_is_not_an_exploratory_gain():
    assert not exploratory_gain(
        uppers=[-0.0002] * 6,
        difference=-0.0005,
        control_difference=0.0004,
        smallest_effect=SMALLEST,
    )


def test_no_bounds_is_not_an_exploratory_gain():
    assert not exploratory_gain(
        uppers=[], difference=-0.002, control_difference=0.001, smallest_effect=SMALLEST
    )


def test_a_difference_no_better_than_the_control_is_not_an_exploratory_gain():
    for difference in (-0.0004, -0.0002):
        assert not exploratory_gain(
            uppers=[-0.1],
            difference=difference,
            control_difference=-0.0004,
            smallest_effect=SMALLEST,
        )


def test_removing_a_group_matters_if_the_error_rises_by_more_than_the_control_changed_it():
    assert drop_one_matters(raise_in_error=0.002, control_difference=0.001)
    assert not drop_one_matters(raise_in_error=0.001, control_difference=0.001)
    assert not drop_one_matters(raise_in_error=0.0005, control_difference=0.001)


def test_a_control_that_lowered_the_error_counts_as_a_difference_of_zero():
    assert drop_one_matters(raise_in_error=0.0005, control_difference=-0.001)
    assert not drop_one_matters(raise_in_error=-0.0002, control_difference=-0.001)
    assert not drop_one_matters(raise_in_error=0.0, control_difference=-0.001)


def test_a_planned_gain_gives_the_planned_class_without_needing_a_drop_one_run_for_one_group():
    cls = evidence_class(
        planned_verdict="gain",
        planned_needs_drop_one=False,
        has_exploratory_gain=False,
        has_drop_one_gain=False,
    )

    assert cls == "planned gain"


def test_a_planned_gain_that_covers_several_groups_needs_the_groups_own_drop_one_run():
    with_drop_one = evidence_class(
        planned_verdict="gain",
        planned_needs_drop_one=True,
        has_exploratory_gain=False,
        has_drop_one_gain=True,
    )
    without_drop_one = evidence_class(
        planned_verdict="gain",
        planned_needs_drop_one=True,
        has_exploratory_gain=False,
        has_drop_one_gain=False,
    )

    assert with_drop_one == "planned gain"
    assert without_drop_one == "no gain shown"


def test_a_planned_gain_that_fails_its_drop_one_run_still_earns_a_weaker_class():
    cls = evidence_class(
        planned_verdict="gain",
        planned_needs_drop_one=True,
        has_exploratory_gain=True,
        has_drop_one_gain=False,
    )

    assert cls == "exploratory gain"


def test_the_classes_after_the_planned_one_are_taken_in_order():
    exploratory = evidence_class(
        planned_verdict=None,
        planned_needs_drop_one=False,
        has_exploratory_gain=True,
        has_drop_one_gain=True,
    )
    drop_one = evidence_class(
        planned_verdict="unresolved",
        planned_needs_drop_one=False,
        has_exploratory_gain=False,
        has_drop_one_gain=True,
    )
    nothing = evidence_class(
        planned_verdict="no gain",
        planned_needs_drop_one=False,
        has_exploratory_gain=False,
        has_drop_one_gain=False,
    )

    assert (exploratory, drop_one, nothing) == (
        "exploratory gain",
        "drop-one gain",
        "no gain shown",
    )


def test_the_priority_order_ranks_by_class_before_gain_and_by_gain_within_a_class():
    classes: dict[str, EvidenceClassType] = {
        "small planned": "planned gain",
        "large drop-one": "drop-one gain",
        "large planned": "planned gain",
        "nothing": "no gain shown",
        "exploratory": "exploratory gain",
    }
    gains = {
        "small planned": 0.001,
        "large drop-one": 0.009,
        "large planned": 0.004,
        "nothing": 0.0,
        "exploratory": 0.002,
    }

    order = priority_order(classes=classes, gains=gains)

    assert order == ["large planned", "small planned", "exploratory", "large drop-one", "nothing"]


def _losses(*, sites: list[str], errors: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": sites,
            "time": [datetime(2025, 3, 1, 10)] * len(sites),
            "seed": [0] * len(sites),
            "month": ["2025-03"] * len(sites),
            "arm": ["f1"] * len(sites),
            "error": errors,
        }
    )


def test_stacking_keeps_the_same_farm_hour_at_two_lead_days_as_two_rows():
    stacked = stack_lead_days(
        losses_by_lead_day={
            1: _losses(sites=["A", "B"], errors=[1.0, 2.0]),
            2: _losses(sites=["A", "B"], errors=[3.0, 4.0]),
        }
    )

    assert stacked["site"].to_list() == ["A-L1", "B-L1", "A-L2", "B-L2"]
    assert stacked["lead_day"].to_list() == [1, 1, 2, 2]
    assert stacked.select("site", "time", "seed").is_duplicated().sum() == 0


def test_a_stacked_paired_difference_is_the_equal_weight_mean_of_the_lead_days():
    first = pl.concat(
        [
            _losses(sites=["A", "B"], errors=[1.0, 2.0]),
            _losses(sites=["A", "B"], errors=[0.5, 0.5]).with_columns(arm=pl.lit("f2")),
        ]
    )
    second = pl.concat(
        [
            _losses(sites=["A"], errors=[3.0]),
            _losses(sites=["A"], errors=[1.0]).with_columns(arm=pl.lit("f2")),
        ]
    )
    stacked = stack_lead_days(losses_by_lead_day={1: first, 2: second})

    differences, months = paired_differences(
        losses=stacked, treatment="f2", reference="f1", metric="error"
    )

    # Farm B's hour is missing at lead day 2, so it is dropped at lead day 1 too. Each lead day's
    # difference for farm A is -0.5 and -2.0, and the mean of the two is -1.25. Keeping farm B
    # would weight the lead days by their rows and give -4 / 3.
    assert stacked["site"].unique().sort().to_list() == ["A-L1", "A-L2"]
    assert differences.shape == (1, 2)
    assert differences.mean() == pytest.approx(-1.25)
    assert set(months) == {"2025-03"}


def test_stacking_refuses_nothing_to_stack_and_a_label_that_holds_the_separator():
    with pytest.raises(ValueError, match="no lead days"):
        stack_lead_days(losses_by_lead_day={})
    with pytest.raises(ValueError, match="already holds"):
        stack_lead_days(losses_by_lead_day={1: _losses(sites=["A-L1"], errors=[1.0])})


def test_a_bound_within_a_fifth_of_the_width_from_zero_is_near_the_line():
    assert is_near_line(lower=-0.9, upper=0.1)
    assert is_near_line(lower=0.02, upper=1.0)
    assert is_near_line(lower=-1.0, upper=-0.1)


def test_an_interval_far_from_zero_on_both_sides_is_not_near_the_line():
    assert not is_near_line(lower=-0.5, upper=0.5)
    assert not is_near_line(lower=0.5, upper=1.5)


def test_the_near_line_boundary_is_inclusive():
    assert is_near_line(lower=0.2, upper=1.2)
    assert not is_near_line(lower=0.2001, upper=1.2001)


def test_a_group_counts_as_a_drop_one_gain_only_if_every_setting_agrees():
    assert drop_one_gain_at_settings(raises=[0.0004, 0.0005], control_differences=[0.0001, 0.0001])
    # The main setting passes and the second fails: the group is not a gain.
    assert not drop_one_gain_at_settings(
        raises=[0.0004, 0.0001], control_differences=[0.0001, 0.0003]
    )
    # The main setting fails and the second passes: also not a gain.
    assert not drop_one_gain_at_settings(
        raises=[0.0001, 0.0004], control_differences=[0.0003, 0.0001]
    )


def test_the_drop_one_rule_with_only_the_main_setting_reads_that_setting_alone():
    assert drop_one_gain_at_settings(raises=[0.0004], control_differences=[0.0001])
    assert not drop_one_gain_at_settings(raises=[0.0001], control_differences=[0.0004])


def test_the_drop_one_rule_with_no_setting_fitted_is_not_a_gain():
    assert not drop_one_gain_at_settings(raises=[], control_differences=[])


def test_the_drop_one_rule_refuses_raises_and_controls_of_different_lengths():
    with pytest.raises(ValueError, match="control differences"):
        drop_one_gain_at_settings(raises=[0.0004, 0.0004], control_differences=[0.0001])


def test_the_drop_one_margin_is_the_raise_less_the_control_floored_at_zero():
    assert drop_one_margin(raise_in_error=0.00036, control_difference=0.00035) == pytest.approx(
        0.00001
    )
    # A control that lowered the error counts as zero, so the margin is the whole raise.
    assert drop_one_margin(raise_in_error=0.0003, control_difference=-0.001) == pytest.approx(
        0.0003
    )
    assert drop_one_margin(raise_in_error=0.0001, control_difference=0.0003) < 0


def test_the_drop_one_margin_agrees_with_the_drop_one_rule_at_the_boundary():
    assert drop_one_margin(raise_in_error=0.0003, control_difference=0.0003) == 0
    assert not drop_one_matters(raise_in_error=0.0003, control_difference=0.0003)


def test_total_cloud_wholly_on_the_gain_side_but_short_of_the_smallest_effect_is_unresolved():
    label = display_class(evidence="no gain shown", gain_lower=0.0004, smallest_effect=SMALLEST)

    assert label == "unresolved at the smallest effect"


def test_a_gain_interval_reaching_the_smallest_effect_exactly_is_still_unresolved():
    label = display_class(evidence="no gain shown", gain_lower=SMALLEST, smallest_effect=SMALLEST)

    assert label == "unresolved at the smallest effect"


def test_an_interval_beyond_the_smallest_effect_or_not_wholly_a_gain_keeps_its_class():
    assert (
        display_class(evidence="no gain shown", gain_lower=0.0015, smallest_effect=SMALLEST)
        == "no gain shown"
    )
    assert (
        display_class(evidence="no gain shown", gain_lower=0.0, smallest_effect=SMALLEST)
        == "no gain shown"
    )
    assert (
        display_class(evidence="no gain shown", gain_lower=-0.0004, smallest_effect=SMALLEST)
        == "no gain shown"
    )
    assert (
        display_class(evidence="no gain shown", gain_lower=None, smallest_effect=SMALLEST)
        == "no gain shown"
    )


def test_only_the_no_gain_shown_class_is_relabelled():
    for evidence in ("planned gain", "exploratory gain", "drop-one gain"):
        assert display_class(evidence=evidence, gain_lower=0.0004, smallest_effect=SMALLEST) == (
            evidence
        )


def test_the_production_scope_sentence_differs_for_each_verdict():
    sentences = {
        verdict: production_scope_sentence(verdict=verdict)
        for verdict in ("gain", "no gain", "unresolved", None)
    }

    assert len(set(sentences.values())) == 4
    assert "separates the routes in favour of a second forecast" in sentences["gain"]
    assert "does not separate the routes" in sentences["no gain"]
    assert "not separated" in sentences["unresolved"]
    assert "not yet fitted" in sentences[None]


def test_a_zero_centred_interval_is_no_gain_only_below_the_smallest_effect_half_width():
    assert half_width_allows_no_gain(half_width=0.0009, smallest_effect=SMALLEST)
    assert not half_width_allows_no_gain(half_width=SMALLEST, smallest_effect=SMALLEST)
    assert not half_width_allows_no_gain(half_width=0.0017, smallest_effect=SMALLEST)


def test_the_raw_error_uses_only_rows_where_every_forecast_and_the_truth_are_present():
    frame = pl.DataFrame(
        {
            "a": [110.0, 120.0, None, 500.0],
            "b": [100.0, float("nan"), 130.0, 100.0],
            "truth": [100.0, 100.0, 100.0, 100.0],
        }
    )

    errors = mean_absolute_error_on_common_rows(
        frame=frame, forecast_columns=["a", "b"], truth_column="truth"
    )

    # Only rows 0 and 3 hold everything. Using each column's own rows would give a = 210 / 3.
    assert errors == {"a": pytest.approx((10.0 + 400.0) / 2), "b": pytest.approx(0.0)}


def test_the_percentage_rise_is_relative_to_the_first_value():
    assert percentage_rise(first=50.0, last=63.0) == pytest.approx(26.0)
    assert percentage_rise(first=63.0, last=50.0) < 0


def _fold_losses() -> pl.DataFrame:
    rows = []
    for seed in (0, 1):
        for index, fold in enumerate([0, 0, 0, 1]):
            for arm, value in (("ref", 0.10), ("trt", 0.10 - 0.02 * (fold + 1) - 0.001 * seed)):
                rows.append(
                    {
                        "site": "A",
                        "time": datetime(2025, 3, 1 + index),
                        "seed": seed,
                        "fold": fold,
                        "arm": arm,
                        "metric": value,
                    }
                )
    return pl.DataFrame(rows)


def test_the_difference_by_fold_is_the_mean_over_rows_and_seeds_of_each_fold_alone():
    by_fold = mean_paired_difference_by(
        losses=_fold_losses(), treatment="trt", reference="ref", metric="metric", by="fold"
    )

    assert by_fold["fold"].to_list() == [0, 1]
    # Fold 0 has three rows per seed and fold 1 one, so a mean of the two folds' means would differ.
    assert by_fold["difference"].to_list() == pytest.approx([-0.02 - 0.0005, -0.04 - 0.0005])
    assert by_fold["rows"].to_list() == [6, 2]


def test_extra_arms_are_stacked_under_the_main_columns():
    main = pl.DataFrame({"arm": ["f6"], "setting": ["sensitivity"], "x": [1.0]})
    extra = pl.DataFrame({"x": [2.0], "setting": ["sensitivity"], "arm": ["drop_f2"]})

    merged = merge_extra_arms(main=main, extra=extra)

    assert merged.columns == ["arm", "setting", "x"]
    assert merged["arm"].to_list() == ["f6", "drop_f2"]
    assert merged["x"].to_list() == [1.0, 2.0]


def test_extra_arms_that_repeat_a_fit_already_in_the_main_losses_are_refused():
    main = pl.DataFrame({"arm": ["f6", "drop_f2"], "setting": ["primary", "primary"], "x": [1, 2]})
    extra = pl.DataFrame({"arm": ["drop_f2"], "setting": ["primary"], "x": [3]})

    with pytest.raises(ValueError, match="repeat fits"):
        merge_extra_arms(main=main, extra=extra)
