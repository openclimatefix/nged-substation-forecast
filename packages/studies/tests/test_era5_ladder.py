from datetime import UTC, datetime, timedelta
from itertools import pairwise

import polars as pl
import pytest
from studies.blending import PERMUTED_SUFFIX
from studies.era5_ladder import (
    ACCUMULATED_VARIABLES,
    AEROSOL_COLUMNS,
    AEROSOL_CONDITIONS,
    CLEAR_SKY_INDEX_THRESHOLDS,
    INSTANTANEOUS_VARIABLES,
    MARS_ONLY_VARIABLES,
    MISSING_UNDER_CLEAR_SKY_VARIABLES,
    RUNG_ADDITIONS,
    RUNGS,
    SHARED_FEATURES,
    accumulation_to_hourly_rate,
    aerosol_condition_flags,
    aerosol_hour_ending_mean,
    aerosol_trial_recommendation,
    drop_one_group_features,
    gain_shares,
    mars_fetch_recommendation,
    negative_control_columns,
    negative_control_features,
    raise_unless_same_rows,
    ratio_index,
    rung_features,
    rung_variables,
    season_of_month,
    sky_regime,
    sky_regime_from_cloud_cover,
    without_mars_only_features,
)

LADDER_TABLE_VARIABLES = {
    "ssrd", "t2m", "tcc", "lcc", "mcc", "hcc", "ssrdc", "tclw", "tciw", "tcslw", "cbh", "fdir",
    "cdir", "u10", "v10", "strd", "d2m", "tcwv", "blh", "sd", "sf", "asn", "fal", "tp", "tcrw",
    "tcsw", "cape", "cin", "skt", "tco3", "uvb", "i10fg", "sp", "deg0l",
}  # fmt: skip
"""The 34 variables of the plan's ladder table, typed out again so a drift in the ladder fails."""

START = datetime(2025, 6, 1, tzinfo=UTC)


def test_every_ladder_variable_has_exactly_one_hour_class():
    laddered = [name for additions in RUNG_ADDITIONS.values() for name in additions]

    assert sorted(ACCUMULATED_VARIABLES + INSTANTANEOUS_VARIABLES) == sorted(laddered)
    assert not set(ACCUMULATED_VARIABLES) & set(INSTANTANEOUS_VARIABLES)
    # ECMWF's documentation: these eight are totals over the hour ending at the label.
    assert set(ACCUMULATED_VARIABLES) == {
        "ssrd", "ssrdc", "fdir", "cdir", "strd", "tp", "sf", "uvb",
    }  # fmt: skip


def test_the_ladder_holds_the_34_variables_of_the_plan_once_each():
    laddered = [name for additions in RUNG_ADDITIONS.values() for name in additions]

    assert len(laddered) == len(set(laddered)) == 34
    assert set(laddered) == LADDER_TABLE_VARIABLES
    assert set(rung_variables(rung="g9")) == LADDER_TABLE_VARIABLES


def test_each_rung_strictly_contains_the_rung_below():
    for lower, higher in pairwise(RUNGS):
        lower_features = rung_features(rung=lower)
        higher_features = rung_features(rung=higher)

        assert set(lower_features) < set(higher_features)


def test_a_rung_shows_the_shared_features_and_its_derived_features():
    g0 = rung_features(rung="g0")
    g3 = rung_features(rung="g3")

    assert set(SHARED_FEATURES) <= set(g0)
    assert "clearness_index" in g0
    assert "clear_sky_index" not in g0
    assert "clear_sky_index" in g3


def test_the_columns_that_may_be_missing_are_ladder_variables():
    assert set(MISSING_UNDER_CLEAR_SKY_VARIABLES) <= set(rung_variables(rung="g9"))


def test_dropping_a_group_removes_its_variables_and_derived_feature_only():
    kept = drop_one_group_features(dropped="g3")

    assert "ssrdc" not in kept
    assert "clear_sky_index" not in kept
    assert "ssrd" in kept
    assert "clearness_index" in kept
    assert len(kept) == len(rung_features(rung="g9")) - 2


def test_dropping_a_group_with_no_derived_feature_removes_its_variables():
    kept = drop_one_group_features(dropped="g4")

    assert not set(RUNG_ADDITIONS["g4"]) & set(kept)
    assert len(kept) == len(rung_features(rung="g9")) - 4


def test_the_minimal_set_cannot_be_dropped():
    with pytest.raises(ValueError, match="base every arm keeps"):
        drop_one_group_features(dropped="g0")


def test_the_negative_control_has_as_many_columns_as_g9_and_permutes_only_later_columns():
    control = negative_control_features(suffix=PERMUTED_SUFFIX)
    permuted = negative_control_columns()

    assert len(control) == len(rung_features(rung="g9"))
    assert set(permuted) == set(rung_features(rung="g9")) - set(rung_features(rung="g2"))
    assert control[: len(rung_features(rung="g2"))] == rung_features(rung="g2")
    assert all(name.endswith(PERMUTED_SUFFIX) for name in control[len(rung_features(rung="g2")) :])


def test_an_hours_accumulation_becomes_its_mean_rate():
    frame = pl.DataFrame(
        {
            "ssrd": [3600.0, 7200.0],
            "uvb": [3600.0, 0.0],
            "tp": [0.001, 0.0],
            "sf": [0.002, 0.0],
        }
    )

    converted = frame.select(
        accumulation_to_hourly_rate(variable=name) for name in ("ssrd", "uvb", "tp", "sf")
    )

    assert converted["ssrd"].to_list() == [1.0, 2.0]
    assert converted["uvb"].to_list() == [1.0, 0.0]
    assert converted["tp"].to_list() == pytest.approx([1.0, 0.0])
    assert converted["sf"].to_list() == pytest.approx([2.0, 0.0])


def test_a_slightly_negative_accumulation_is_clipped_to_zero():
    frame = pl.DataFrame({"ssrdc": [-0.25 * 3600.0, 3600.0], "tp": [-1e-9, 0.001]})

    converted = frame.select(
        accumulation_to_hourly_rate(variable="ssrdc"), accumulation_to_hourly_rate(variable="tp")
    )

    assert converted["ssrdc"].to_list() == [0.0, 1.0]
    assert converted["tp"].to_list() == pytest.approx([0.0, 1.0])


def test_an_instantaneous_variable_is_not_converted_as_an_accumulation():
    with pytest.raises(ValueError, match="not an accumulation"):
        accumulation_to_hourly_rate(variable="cape")


def test_a_ratio_is_missing_where_the_denominator_is_zero_or_below_the_minimum():
    frame = pl.DataFrame({"top": [100.0, 0.0, 40.0, 60.0], "ghi": [50.0, 5.0, 20.0, 30.0]})

    ratios = frame.select(
        index=ratio_index(
            numerator=pl.col("ghi"), denominator=pl.col("top"), minimum_denominator=50.0
        )
    )["index"]

    assert ratios.to_list() == [0.5, None, None, 0.5]


def test_a_ratio_with_no_minimum_still_refuses_a_zero_denominator():
    frame = pl.DataFrame({"top": [0.0, 4.0], "ghi": [1.0, 2.0]})

    ratios = frame.select(index=ratio_index(numerator=pl.col("ghi"), denominator=pl.col("top")))

    assert ratios["index"].to_list() == [None, 0.5]


def test_sky_regimes_split_at_the_fixed_thresholds_and_keep_a_missing_index_missing():
    overcast_below, clear_from = CLEAR_SKY_INDEX_THRESHOLDS
    index = pl.Series([overcast_below - 0.01, overcast_below, 0.6, clear_from, None])

    regimes = pl.DataFrame({"index": index}).select(
        regime=sky_regime(index=pl.col("index"), thresholds=CLEAR_SKY_INDEX_THRESHOLDS)
    )["regime"]

    assert regimes.to_list() == ["overcast", "broken", "broken", "clear", None]


def test_cloud_cover_regimes_are_clear_when_the_sky_is_empty():
    cover = pl.DataFrame({"tcc": [0.0, 0.19, 0.2, 0.79, 0.8, 1.0, None]})

    regimes = cover.select(regime=sky_regime_from_cloud_cover(cloud_cover=pl.col("tcc")))["regime"]

    assert regimes.to_list() == ["clear", "clear", "broken", "broken", "overcast", "overcast", None]


def test_every_calendar_month_has_its_meteorological_season():
    months = pl.DataFrame({"month": list(range(1, 13))})

    seasons = months.select(season=season_of_month(month_number=pl.col("month")))["season"]

    assert seasons.to_list() == [
        "winter", "winter", "spring", "spring", "spring", "summer",
        "summer", "summer", "autumn", "autumn", "autumn", "winter",
    ]  # fmt: skip


def _losses(*, rows_by_arm: dict[str, list[int]]) -> pl.DataFrame:
    frames = [
        pl.DataFrame(
            {
                "arm": [arm] * len(hours),
                "site": ["A"] * len(hours),
                "time": [START + timedelta(hours=hour) for hour in hours],
                "seed": [0] * len(hours),
            }
        )
        for arm, hours in rows_by_arm.items()
    ]
    return pl.concat(frames)


def test_arms_with_the_same_rows_pass_the_pairing_guard():
    losses = _losses(rows_by_arm={"g0": [1, 2, 3], "g9": [1, 2, 3]})

    raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def test_an_arm_missing_a_row_the_other_holds_fails_the_pairing_guard():
    losses = _losses(rows_by_arm={"g0": [1, 2, 3], "g9": [1, 2]})

    with pytest.raises(
        ValueError, match=r"1 \(site, time, seed\) rows only in the first and 0 only in the second"
    ):
        raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def test_an_arm_with_an_extra_row_fails_the_pairing_guard():
    losses = _losses(rows_by_arm={"g0": [1, 2], "g9": [1, 2, 3]})

    with pytest.raises(
        ValueError, match=r"0 \(site, time, seed\) rows only in the first and 1 only in the second"
    ):
        raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def test_an_arm_that_does_not_exist_fails_the_pairing_guard():
    losses = _losses(rows_by_arm={"g0": [1, 2]})

    with pytest.raises(ValueError, match="has no rows"):
        raise_unless_same_rows(losses=losses, arms=["g9", "g0"])


def _aerosol_ramp(
    *, site: str, per_hour: float, start_value: float, shift_hours: int = 0
) -> pl.DataFrame:
    # An analysis every 3 hours whose value is `start_value + per_hour * hours since START`.
    times = [START + timedelta(hours=hour) for hour in range(0, 24, 3)]
    return pl.DataFrame(
        {
            "site": [site] * len(times),
            "time": [time + timedelta(hours=shift_hours) for time in times],
            "aod550": [start_value + per_hour * (3 * index) for index in range(len(times))],
            "duaod550": [
                2.0 * (start_value + per_hour * (3 * index)) for index in range(len(times))
            ],
        }
    )


def _labels(*, site: str, hours: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {"site": [site] * len(hours), "time": [START + timedelta(hours=hour) for hour in hours]}
    )


def test_a_straight_rise_gives_the_value_half_an_hour_before_the_label():
    aerosol = _aerosol_ramp(site="A", per_hour=0.01, start_value=0.1)
    labels = _labels(site="A", hours=[1, 2, 3, 4, 5, 10])

    result = aerosol_hour_ending_mean(
        aerosol=aerosol, labels=labels, value_columns=list(AEROSOL_COLUMNS)
    )

    expected = [0.1 + 0.01 * (hour - 0.5) for hour in [1, 2, 3, 4, 5, 10]]
    assert result["aod550"].to_list() == pytest.approx(expected)
    assert result["duaod550"].to_list() == pytest.approx([2.0 * value for value in expected])


def test_each_site_is_interpolated_from_its_own_series():
    aerosol = pl.concat(
        [
            _aerosol_ramp(site="A", per_hour=0.01, start_value=0.1),
            _aerosol_ramp(site="B", per_hour=0.02, start_value=0.3),
        ]
    )
    labels = pl.concat([_labels(site="A", hours=[5]), _labels(site="B", hours=[5])])

    result = aerosol_hour_ending_mean(
        aerosol=aerosol, labels=labels, value_columns=["aod550"]
    ).sort("site")

    assert result["aod550"].to_list() == pytest.approx([0.1 + 0.01 * 4.5, 0.3 + 0.02 * 4.5])


def test_an_input_shifted_by_three_hours_gives_a_different_answer():
    aerosol = _aerosol_ramp(site="A", per_hour=0.01, start_value=0.1, shift_hours=3)
    labels = _labels(site="A", hours=[5])

    result = aerosol_hour_ending_mean(aerosol=aerosol, labels=labels, value_columns=["aod550"])

    assert result["aod550"].item() == pytest.approx(0.1 + 0.01 * 4.5 - 0.03)


def test_an_hour_beyond_the_last_analysis_time_is_missing():
    aerosol = _aerosol_ramp(site="A", per_hour=0.01, start_value=0.1)
    labels = _labels(site="A", hours=[20, 21, 22])

    result = aerosol_hour_ending_mean(aerosol=aerosol, labels=labels, value_columns=["aod550"])

    # The last analysis is at 21:00. Hour 20 reads between 18:00 and 21:00, but the instant 21:00
    # of hour 21 would need the analysis at 24:00, which does not exist.
    assert result["aod550"].is_null().to_list() == [False, True, True]


def test_the_result_is_sorted_by_site_and_time():
    aerosol = _aerosol_ramp(site="A", per_hour=0.01, start_value=0.1)
    labels = _labels(site="A", hours=[9, 5, 7])

    result = aerosol_hour_ending_mean(aerosol=aerosol, labels=labels, value_columns=["aod550"])

    assert result["time"].to_list() == sorted(labels["time"].to_list())


def test_the_mars_free_arm_drops_the_mars_variables_and_the_clear_sky_index_only():
    g9 = set(rung_features(rung="g9"))
    free = set(without_mars_only_features())

    assert g9 - free == {*MARS_ONLY_VARIABLES, "clear_sky_index"}
    assert free < g9
    assert "ssrd" in free
    assert "tcc" in free


def test_the_mars_only_variables_are_the_twelve_the_plan_names():
    plan_names = {
        "ssrdc", "cdir", "tclw", "tciw", "tcslw", "cbh",
        "tcrw", "tcsw", "tco3", "uvb", "fal", "deg0l",
    }  # fmt: skip

    assert set(MARS_ONLY_VARIABLES) == plan_names
    assert len(MARS_ONLY_VARIABLES) == len(plan_names)
    assert plan_names <= set(rung_variables(rung="g9"))


def test_every_arm_is_shown_the_hour_integrated_top_of_atmosphere_flux():
    assert "cams_toa_w_m2" in SHARED_FEATURES
    assert "cams_toa_w_m2" in rung_features(rung="g0")
    assert "cams_toa_w_m2" in without_mars_only_features()


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (-0.5, -0.2, "pilot"),
        (-0.5, -0.1, "unresolved"),
        (-0.5, 0.3, "unresolved"),
        (-0.05, 0.3, "against"),
        (0.1, 0.3, "against"),
        (-0.1, 0.3, "unresolved"),
    ],
)
def test_the_mars_fetch_rule_compares_the_whole_interval_with_minus_the_smallest_effect(
    lower: float, upper: float, expected: str
):
    assert mars_fetch_recommendation(lower=lower, upper=upper, smallest_effect=0.1) == expected


def test_gain_shares_sum_to_one_and_give_a_never_split_column_zero():
    shares = gain_shares(total_gain={"a": 6.0, "b": 2.0}, features=["a", "b", "c"])

    assert shares == {"a": 0.75, "b": 0.25, "c": 0.0}
    assert sum(shares.values()) == pytest.approx(1.0)


def test_gain_shares_are_the_same_whatever_the_scale_of_the_gain():
    small = gain_shares(total_gain={"a": 3.0, "b": 1.0}, features=["a", "b"])
    large = gain_shares(total_gain={"a": 3000.0, "b": 1000.0}, features=["a", "b"])

    assert small == large


def test_a_model_that_made_no_split_has_no_share_in_any_column():
    assert gain_shares(total_gain={}, features=["a", "b"]) == {"a": 0.0, "b": 0.0}


def test_gain_for_a_column_the_model_was_not_shown_raises():
    with pytest.raises(ValueError, match="not shown"):
        gain_shares(total_gain={"a": 1.0, "f3": 2.0}, features=["a", "b"])


def _aerosol_frame() -> pl.DataFrame:
    count = 21
    start = datetime(2024, 6, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "site": ["A"] * count,
            "time": [start + timedelta(hours=hour) for hour in range(count)],
            "tcc": [0.1 if hour % 2 == 0 else 0.9 for hour in range(count)],
            "aod550": [0.01 * (hour + 1) for hour in range(count)],
            "duaod550": [0.001 * (hour + 1) for hour in range(count)],
        }
    )


def test_each_aerosol_condition_applies_its_own_thresholds():
    flags = aerosol_condition_flags(frame=_aerosol_frame())

    assert flags.columns == ["site", "time", *AEROSOL_CONDITIONS]
    # Hour index h has dust 0.001 (h + 1), so the 95th percentile of 21 hours is hour 19's value.
    assert flags.filter(pl.col("dusty"))["time"].dt.hour().to_list() == [19, 20]
    # Of those, only hour 20 is clear (even hours have tcc 0.1).
    assert flags.filter(pl.col("clear_and_dusty"))["time"].dt.hour().to_list() == [20]
    # The median of aod550 is hour 10's value, so clean means hours 0 to 9, and clear means even.
    assert flags.filter(pl.col("clear_and_clean"))["time"].dt.hour().to_list() == [0, 2, 4, 6, 8]
    assert flags["clear"].sum() == 11


def test_aerosol_conditions_refuse_rows_with_a_missing_input():
    frame = _aerosol_frame().with_columns(
        tcc=pl.when(pl.col("time").dt.hour() == 3).then(None).otherwise(pl.col("tcc"))
    )

    with pytest.raises(ValueError, match="non-null"):
        aerosol_condition_flags(frame=frame)


@pytest.mark.parametrize(
    ("uppers", "days", "months", "expected"),
    [
        ([-0.002, -0.0015], 30, 14, "trial_worth_running"),
        ([-0.002, -0.0005], 30, 14, "not_shown"),
        ([-0.001, -0.002], 30, 14, "not_shown"),
        ([-0.002, -0.002], 19, 14, "cannot_be_assessed"),
        ([-0.002, -0.002], 20, 11, "cannot_be_assessed"),
        ([-0.002], 30, 14, "cannot_be_assessed"),
        ([-0.002, -0.002], 20, 12, "trial_worth_running"),
    ],
)
def test_the_aerosol_rule_needs_enough_events_and_both_settings_to_agree(
    uppers: list[float], days: int, months: int, expected: str
):
    assert (
        aerosol_trial_recommendation(uppers=uppers, smallest_effect=0.001, days=days, months=months)
        == expected
    )


def test_arms_that_differ_only_in_seed_fail_the_pairing_guard():
    losses = _losses(rows_by_arm={"g0": [1, 2], "g9": [1, 2]}).with_columns(
        seed=pl.when(pl.col("arm") == "g9").then(1).otherwise(0)
    )

    with pytest.raises(ValueError, match="hold different rows"):
        raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def test_arms_that_differ_only_in_site_fail_the_pairing_guard():
    losses = _losses(rows_by_arm={"g0": [1, 2], "g9": [1, 2]}).with_columns(
        site=pl.when(pl.col("arm") == "g9").then(pl.lit("B")).otherwise(pl.lit("A"))
    )

    with pytest.raises(ValueError, match="hold different rows"):
        raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def _twenty_clear_hours() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "site": ["A"] * 20,
            "time": [START + timedelta(hours=hour) for hour in range(20)],
            "tcc": [0.1] * 20,
            "aod550": [0.01 * (hour + 1) for hour in range(20)],
            "duaod550": [0.001 * (hour + 1) for hour in range(20)],
        }
    )


def test_aerosol_quantiles_interpolate_between_rows():
    flags = aerosol_condition_flags(frame=_twenty_clear_hours())

    # 0.95 of 19 gaps is 18.05, so the dust threshold sits just above hour 18's value.
    assert flags.filter(pl.col("dusty"))["time"].dt.hour().to_list() == [19]
    # 0.5 of 19 gaps is 9.5, so the clean threshold sits between hours 9 and 10.
    assert flags.filter(pl.col("clear_and_clean"))["time"].dt.hour().to_list() == list(range(10))


def test_aerosol_conditions_refuse_a_missing_total_optical_depth():
    frame = _twenty_clear_hours().with_columns(
        aod550=pl.when(pl.col("time").dt.hour() == 3).then(None).otherwise(pl.col("aod550"))
    )

    with pytest.raises(ValueError, match="non-null"):
        aerosol_condition_flags(frame=frame)


def _arm_rows(*, arm: str, hours: list[int], seed: int = 0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "arm": [arm] * len(hours),
            "site": ["A"] * len(hours),
            "time": [START + timedelta(hours=hour) for hour in hours],
            "seed": [seed] * len(hours),
        }
    )


def test_arms_scored_at_different_seeds_fail_the_pairing_guard():
    losses = pl.concat(
        [_arm_rows(arm="g0", hours=[1, 2], seed=0), _arm_rows(arm="g9", hours=[1, 2], seed=1)]
    )

    with pytest.raises(ValueError, match="hold different rows"):
        raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def test_arms_with_as_many_rows_but_different_hours_fail_the_pairing_guard():
    losses = pl.concat([_arm_rows(arm="g0", hours=[1, 2]), _arm_rows(arm="g9", hours=[1, 3])])

    with pytest.raises(ValueError, match=r"1 .* only in the first and 1 only in the second"):
        raise_unless_same_rows(losses=losses, arms=["g0", "g9"])


def test_the_pairing_guard_checks_every_arm_not_only_the_second():
    losses = pl.concat(
        [
            _arm_rows(arm="g0", hours=[1, 2]),
            _arm_rows(arm="g5", hours=[1, 2]),
            _arm_rows(arm="g9", hours=[1]),
        ]
    )

    with pytest.raises(ValueError, match="'g0' and 'g9'"):
        raise_unless_same_rows(losses=losses, arms=["g0", "g5", "g9"])
