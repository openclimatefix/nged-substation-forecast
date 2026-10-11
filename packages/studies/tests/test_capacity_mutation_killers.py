"""Tests that pin the constants and formulas of the capacity estimators, one mutant each.

Each test states the independent expectation (a hand-built reference or a hard-coded number), so a
change to the code under test breaks the test instead of moving both together.
"""

from dataclasses import replace
from typing import Final

import numpy as np
import polars as pl
import pytest
from scipy.optimize import minimize
from studies.capacity_convex import (
    WindowPoints,
    classify_clipped,
    cx2_objective,
    cx2_traces,
)
from studies.capacity_estimators import (
    FALLBACK_NONE,
    FALLBACK_SHORT_HISTORY,
    MIN_SHAPE_SUM,
    REFERENCE_END,
    REFERENCE_START,
    IssueScale,
    RawEstimate,
    _combine_sides,
    _daily_shape_ratio,
    estimate_csq,
    estimate_hm,
    estimate_kf,
    estimate_pr,
    estimate_r30,
    issue_time_estimates,
    reference_day_indices,
    synthetic_record,
    valid_days,
)
from studies.capacity_recovery import (
    PlantedEvent,
    _footprints_clash,
    false_movement_share,
)
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS

from studies import capacity_recovery, capacity_scaling

DAYS: Final[int] = 680


def test_pr_reads_exactly_the_latest_seven_days() -> None:
    record = synthetic_record(seed=30)
    day = 500
    base = estimate_pr(record=record, plant=0, params={}).level[day]
    power = record.power_mw.copy()
    power[0, day - 7] *= 10.0
    assert estimate_pr(record=record.with_power(power_mw=power), plant=0, params={}).level[
        day
    ] == pytest.approx(base)
    power = record.power_mw.copy()
    power[0, day - 6] *= 10.0
    assert estimate_pr(record=record.with_power(power_mw=power), plant=0, params={}).level[
        day
    ] != pytest.approx(base)


def test_pr_needs_four_days_with_data_and_r30_needs_fifteen_valid_days() -> None:
    record = synthetic_record(seed=31)
    day = 500
    power = record.power_mw.copy()
    power[0, day - 6 : day - 3] = np.nan  # leaves days day-3 .. day: 4 days
    assert np.isfinite(
        estimate_pr(record=record.with_power(power_mw=power), plant=0, params={}).level[day]
    )
    power[0, day - 3] = np.nan  # leaves 3 days
    assert np.isnan(
        estimate_pr(record=record.with_power(power_mw=power), plant=0, params={}).level[day]
    )
    power = record.power_mw.copy()
    power[0, day - 29 : day - 14] = np.nan  # 15 valid days remain: day-14 .. day
    assert np.isfinite(
        estimate_r30(record=record.with_power(power_mw=power), plant=0, params={}).level[day]
    )
    power[0, day - 14] = np.nan  # 14 valid days remain
    assert np.isnan(
        estimate_r30(record=record.with_power(power_mw=power), plant=0, params={}).level[day]
    )


def test_r30_is_the_99th_percentile_of_the_valid_days_hourly_power_in_the_window() -> None:
    record = synthetic_record(seed=32)
    day = 500
    valid = valid_days(record=record)[0]
    window = record.power_mw[0, day - 29 : day + 1][valid[day - 29 : day + 1]]
    expected = np.quantile(window.reshape(-1), 0.99)
    assert estimate_r30(record=record, plant=0, params={}).level[day] == pytest.approx(expected)


def test_csq_ignores_hours_below_fifty_watts_per_square_metre_of_clear_sky() -> None:
    record = synthetic_record(seed=33)
    daytime = record.clear_sky_w_m2 > 0
    dim = replace(record, clear_sky_w_m2=np.where(daytime, 49.0, 0.0))
    bright = replace(record, clear_sky_w_m2=np.where(daytime, 51.0, 0.0))
    assert np.isnan(estimate_csq(record=dim, plant=0, params={}).level[500])
    assert np.isfinite(estimate_csq(record=bright, plant=0, params={}).level[500])


def test_the_scale_history_counts_valid_days_up_to_the_latest_whole_day_only() -> None:
    record = synthetic_record(seed=34)
    power = record.power_mw.copy()
    power[0, :100] = np.nan
    gappy = record.with_power(power_mw=power)
    # Valid days start at index 100, so 30 valid days are behind day 129 and the target day 131.
    scale = issue_time_estimates(
        record=gappy, raw=RawEstimate(level=np.full((3, DAYS), 5.0), sd=None), reference=None
    )
    assert scale.fallback_reason[0, 130] == FALLBACK_SHORT_HISTORY
    assert scale.fallback_reason[0, 131] == FALLBACK_NONE


def test_the_relative_sd_is_the_sd_over_the_level_and_empty_on_a_fallback_day() -> None:
    record = synthetic_record(seed=35)
    level = np.full((3, DAYS), 4.0)
    sd = np.full((3, DAYS), 0.5)
    level[0, 400] = np.nan
    scale = issue_time_estimates(record=record, raw=RawEstimate(level=level, sd=sd), reference=None)
    assert scale.relative_sd[0, 500] == pytest.approx(0.125)
    assert np.isnan(scale.relative_sd[0, 402])
    assert np.isnan(scale.relative_sd[0, 5])


def _reference_kf(*, observations: np.ndarray, process_sd: float, observation_sd: float) -> tuple:
    q, r = process_sd**2, observation_sd**2
    state: float | None = None
    variance = 0.0
    levels: list[float] = []
    sds: list[float] = []
    for z in observations:
        if state is None:
            if np.isfinite(z):
                state, variance = z, r
        else:
            variance += q
            if np.isfinite(z):
                s = variance + r
                innovation = np.clip(z - state, -3.0 * np.sqrt(s), 3.0 * np.sqrt(s))
                gain = variance / s
                state += gain * innovation
                variance = (1.0 - gain) * variance
        if state is None:
            levels.append(np.nan)
            sds.append(np.nan)
        else:
            levels.append(np.exp(state))
            sds.append(np.exp(state) * np.sqrt(variance))
    return np.array(levels), np.array(sds)


def test_the_kalman_filter_matches_a_hand_written_filter_with_an_outlier_day() -> None:
    record = synthetic_record(seed=36)
    shape = record.cams_shape.copy()
    shape[0, 300] *= 0.2  # a bad satellite day: the ratio jumps fivefold
    shape[0, 301] *= 3.0
    record = replace(record, cams_shape=shape)
    ratio = _daily_shape_ratio(record=record, plant=0, valid=valid_days(record=record))
    expected_level, expected_sd = _reference_kf(
        observations=np.log(ratio), process_sd=0.01, observation_sd=0.1
    )
    got = estimate_kf(record=record, plant=0, params={"process_sd": 0.01, "observation_sd": 0.1})
    assert got.sd is not None
    assert np.allclose(got.level, expected_level, equal_nan=True, rtol=1e-9)
    assert np.allclose(got.sd, expected_sd, equal_nan=True, rtol=1e-9)


def test_the_daily_ratio_needs_at_least_the_minimum_modelled_output() -> None:
    record = synthetic_record(seed=37)
    shape = record.cams_shape.copy()
    for day, total in ((10, MIN_SHAPE_SUM - 0.01), (11, MIN_SHAPE_SUM + 0.01)):
        shape[0, day] = shape[0, day] * total / shape[0, day].sum()
    record = replace(record, cams_shape=shape)
    ratio = _daily_shape_ratio(record=record, plant=0, valid=np.ones((3, DAYS), dtype=bool))
    assert MIN_SHAPE_SUM == 0.2
    assert np.isnan(ratio[10])
    assert np.isfinite(ratio[11])


def test_hm_calibrates_on_the_median_of_the_reference_year_not_its_mean() -> None:
    record = synthetic_record(seed=38)
    reference = reference_day_indices(record=record)
    power = record.power_mw.copy()
    power[0, reference[: int(0.4 * len(reference))]] *= 6.0  # skews the mean, not the median
    skewed = record.with_power(power_mw=power)
    level = estimate_hm(
        record=skewed,
        plant=0,
        params={"stay_probability": 0.98, "observation_sd": 0.15},
    ).level
    assert level[620] / record.static_p99_mw[0] > 0.97


def test_reference_days_are_every_seventh_day_of_the_year_before_december_2024() -> None:
    record = synthetic_record(seed=39)
    days = reference_day_indices(record=record)
    assert (np.diff(days) == 7).all()
    assert record.dates[days[0]] == np.datetime64(REFERENCE_START)
    assert record.dates[days[-1]] < np.datetime64(REFERENCE_END)
    assert record.dates[days[-1] + 7] >= np.datetime64(REFERENCE_END)


def test_reversing_a_record_reverses_every_array() -> None:
    record = synthetic_record(seed=40, days=60)
    flipped = record.reversed()
    for name in (
        "power_mw",
        "clear_sky_w_m2",
        "cams_ghi_w_m2",
        "cams_shape",
        "clear_shape",
    ):
        assert np.array_equal(getattr(flipped, name), getattr(record, name)[:, ::-1])
    assert np.array_equal(flipped.dates, record.dates[::-1])


def test_inverse_variance_weights_combine_two_sides() -> None:
    sides: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = [
        (np.array([1.0]), np.array([1.0]), np.array([40.0])),
        (np.array([2.0]), np.array([2.0]), np.array([50.0])),
    ]
    level, sd, support = _combine_sides(sides=sides, plants=1)
    assert level[0] == pytest.approx((1.0 / 1.0 + 2.0 / 4.0) / (1.0 / 1.0 + 1.0 / 4.0))
    assert sd[0] == pytest.approx(1.0 / np.sqrt(1.0 + 0.25))
    assert support[0] == 50.0
    unweighted: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = [
        (np.array([1.0]), None, np.array([40.0])),
        (np.array([2.0]), None, np.array([50.0])),
    ]
    plain, plain_sd, _ = _combine_sides(sides=unweighted, plants=1)
    assert plain[0] == pytest.approx(1.5)
    assert np.isnan(plain_sd[0])


def test_footprints_clash_when_either_events_month_meets_the_others_footprint() -> None:
    first = PlantedEvent(0, "step", 0.1, 100, 30)  # month [100, 130), footprint [70, 190)
    second = PlantedEvent(1, "step", 0.1, 180, 30)  # month [180, 210), footprint [150, 270)
    # Only the second event's month meets the first event's footprint, so the test must be
    # symmetric.
    assert _footprints_clash(first=first, second=second)
    assert _footprints_clash(first=second, second=first)
    far = PlantedEvent(1, "step", 0.1, 400, 30)
    assert not _footprints_clash(first=first, second=far)
    assert not _footprints_clash(first=far, second=first)
    same_plant = PlantedEvent(0, "step", 0.1, 200, 30)  # footprint [170, 290) meets [70, 190)
    assert _footprints_clash(first=first, second=same_plant)


def test_drawn_events_keep_apart_by_a_plain_interval_check() -> None:
    record = synthetic_record(
        plants=3, first_day=__import__("datetime").date(2019, 9, 1), days=2500, seed=41
    )
    events = capacity_recovery.draw_events(record=record, seed=5)
    for index, one in enumerate(events):
        for other in events[index + 1 :]:
            one_month = (one.first_day, one.first_day + one.month_days)
            other_month = (other.first_day, other.first_day + other.month_days)
            one_foot = (one.first_day - 30, one.first_day + one.month_days + 60)
            other_foot = (other.first_day - 30, other.first_day + other.month_days + 60)
            if one.plant == other.plant:
                assert one_foot[1] <= other_foot[0] or other_foot[1] <= one_foot[0]
            else:
                assert one_month[1] <= other_foot[0] or other_foot[1] <= one_month[0]
                assert other_month[1] <= one_foot[0] or one_foot[1] <= other_month[0]


def _scale_of(series: np.ndarray, reason: np.ndarray | None = None) -> IssueScale:
    return IssueScale(
        scale_mw=series[None, :],
        relative_sd=np.full((1, len(series)), np.nan),
        fallback_reason=np.zeros((1, len(series)), dtype=np.int8)
        if reason is None
        else reason[None, :],
    )


def test_the_false_movement_median_excludes_the_day_itself() -> None:
    slope = 0.00134
    series = 1.0 + slope * np.arange(300)
    # The 90 days before day 200 have a median 45.5 days behind it: 5.05% below. A window that
    # included the day itself would put the median 44.5 days behind: 4.93%.
    assert false_movement_share(windows=[(0, 200, 1)], original=_scale_of(series)) == 1.0


def test_the_false_movement_threshold_is_strict_and_a_fallback_day_counts_as_moving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capacity_recovery, "FALSE_MOVEMENT_THRESHOLD", 0.0625)
    series = np.ones(300)
    series[200] = 1.0625  # exactly the threshold above a median of 1
    assert false_movement_share(windows=[(0, 200, 1)], original=_scale_of(series)) == 0.0
    series[200] = 1.07
    assert false_movement_share(windows=[(0, 200, 1)], original=_scale_of(series)) == 1.0
    reason = np.zeros(300, dtype=np.int8)
    reason[200:205] = 2
    assert (
        false_movement_share(windows=[(0, 200, 10)], original=_scale_of(np.ones(300), reason))
        == 0.5
    )


def test_the_constant_scale_check_converts_a_fraction_to_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake(offset: float):  # noqa: ANN202
        calls = iter(
            [
                pl.DataFrame(
                    {
                        "absolute_error_fraction_of_capacity": [0.1],
                        capacity_scaling.SCORED_COLUMN: [0.1],
                    }
                ),
                pl.DataFrame(
                    {
                        "absolute_error_fraction_of_capacity": [0.1 + offset],
                        capacity_scaling.SCORED_COLUMN: [0.1 + offset],
                    }
                ),
            ]
        )
        return lambda **_: next(calls)

    site_rows = pl.DataFrame(
        {
            "constrained": [True, False],
            "power_mw": [1.0, 2.0],
            "cap_mw": [1.0, None],
            "effective_capacity_mw": [1.0, 1.0],
        }
    )
    monkeypatch.setattr(capacity_scaling, "out_of_fold_losses", fake(0.0001))
    with pytest.raises(ValueError, match="moved the mean"):
        capacity_scaling.assert_constant_scale_reproduces_baseline(
            site_rows=site_rows,
            features=[],
            hyper_parameters=PRIMARY_HYPER_PARAMETERS,
            scale=0.5,
            tolerance_points=1e-3,
            device="cpu",
        )
    monkeypatch.setattr(capacity_scaling, "out_of_fold_losses", fake(0.0000005))
    worst = capacity_scaling.assert_constant_scale_reproduces_baseline(
        site_rows=site_rows,
        features=[],
        hyper_parameters=PRIMARY_HYPER_PARAMETERS,
        scale=0.5,
        tolerance_points=1e-3,
        device="cpu",
    )
    assert worst == pytest.approx(5e-5)


def test_the_cx2_solution_is_not_beaten_by_a_local_search_when_the_hinge_is_on() -> None:
    rng = np.random.default_rng(3)
    day = np.repeat(np.arange(3), 8)
    unit = rng.uniform(0.4, 1.0, len(day))
    power = np.minimum(0.9 * unit * rng.uniform(0.9, 1.0, len(day)), 0.6)
    points = WindowPoints(day=day, power=power, unit_output=unit)
    assert classify_clipped(points=points).any()
    kwargs = {"quantile": 0.7, "fused_weight": 0.1, "hinge_weight": 3.0}
    dc, ac = cx2_traces(points=points, plant="A", end_day=2, n_days=3, **kwargs)
    found = cx2_objective(dc_trace=dc, ac_trace=ac, points=points, **kwargs)
    best = min(
        minimize(
            lambda z: cx2_objective(dc_trace=z[:3], ac_trace=z[3:], points=points, **kwargs),
            rng.uniform(0.3, 1.2, 6),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 6000},
        ).fun
        for _ in range(12)
    )
    assert found <= best + 1e-7
