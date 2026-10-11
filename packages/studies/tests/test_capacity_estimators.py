"""Tests of the capacity record, the cheap estimators, and the issue-time and hindsight drivers."""

from datetime import UTC, datetime
from typing import Final

import numpy as np
import polars as pl
import pytest
from studies.capacity_estimators import (
    CALIBRATED_ESTIMATORS,
    CHEAP_ESTIMATORS,
    FALLBACK_NONE,
    FALLBACK_NULL_ESTIMATE,
    FALLBACK_SHORT_HISTORY,
    FLOOR_SHARE_OF_STATIC,
    ISSUE_LAG_DAYS,
    CapacityRecord,
    EstimatorNameType,
    RawEstimate,
    assert_scale_ignores_the_issue_day_onward,
    calibrate,
    estimate_kf,
    hindsight_reference_estimates,
    issue_time_estimates,
    neighbour_ratio,
    reference_day_indices,
    reference_levels,
    run_cheap_estimator,
    scale_for_rows,
    scramble_from_day,
    synthetic_record,
    target_day_indices,
    valid_days,
)

DAYS: Final[int] = 680
LOSS: Final[slice] = slice(400, 445)
"""A 45-day, 30% loss at plant 1."""
LOSS_PLANT: Final[int] = 1

ALLOWED_DELAY_DAYS: Final[dict[str, int]] = {
    "R30": 35,
    "CSQ": 35,
    "NR": 10,
    "PR": 10,
    "KF": 25,
    "HM": 12,
}
"""The most days after the loss starts that the scale may take to fall halfway, per estimator."""

RECOVERY_TOLERANCE: Final[dict[str, float]] = {
    "R30": 0.1,
    "CSQ": 0.1,
    "NR": 0.1,
    "PR": 0.08,
    "KF": 0.08,
    "HM": 0.1,
}
"""How far the scale at the end of the loss may be from 0.7 of its level before the loss."""


def _scale(*, record: CapacityRecord, name: str) -> np.ndarray:
    """Return the issue-time scale of one estimator, shape (plants, days)."""
    raw = run_cheap_estimator(name=name, record=record)
    reference = reference_levels(raw=raw, record=record) if name in CALIBRATED_ESTIMATORS else None
    return issue_time_estimates(record=record, raw=raw, reference=reference).scale_mw


@pytest.fixture(scope="module")
def lossy_record() -> CapacityRecord:
    """A record whose plant 1 loses 30% of its capacity on days 400 to 444."""
    factor = np.ones((3, DAYS))
    factor[LOSS_PLANT, LOSS] = 0.7
    return synthetic_record(capacity_factor=factor)


@pytest.mark.parametrize("name", sorted(ALLOWED_DELAY_DAYS))
def test_each_estimator_recovers_a_planted_loss_within_its_tolerance_and_delay(
    name: str, lossy_record: CapacityRecord
) -> None:
    scale = _scale(record=lossy_record, name=name)[LOSS_PLANT]
    before = scale[380:399].mean()
    ratio = scale / before
    assert abs(ratio[440:445].mean() - 0.7) <= RECOVERY_TOLERANCE[name]
    halfway = np.flatnonzero(ratio[400:] <= 0.85)[0]
    assert halfway <= ALLOWED_DELAY_DAYS[name]
    # The plants without a loss stay near their level: the estimator does not chase the weather.
    quiet = _scale(record=lossy_record, name=name)[0]
    assert np.abs(quiet[420:445] / quiet[380:399].mean() - 1.0).max() < 0.2


@pytest.mark.parametrize("name", sorted(ALLOWED_DELAY_DAYS))
def test_an_estimate_for_day_d_ignores_every_input_after_day_d(name: str) -> None:
    record = synthetic_record(seed=3)
    day = 600  # 2025-01-22, after the reference year ends on 2024-11-30
    scrambled_after = scramble_from_day(record=record, first_day=day + 1, seed=1)
    scrambled_from = scramble_from_day(record=record, first_day=day, seed=1)
    original = run_cheap_estimator(name=name, record=record).level[:, day]
    assert np.array_equal(
        original,
        run_cheap_estimator(name=name, record=scrambled_after).level[:, day],
        equal_nan=True,
    )
    # The probe is able to fail: noise from day d itself does change the estimate for day d.
    assert not np.array_equal(
        original,
        run_cheap_estimator(name=name, record=scrambled_from).level[:, day],
        equal_nan=True,
    )


def test_the_scale_for_a_target_day_ignores_its_issue_day_and_the_probe_catches_a_leak() -> None:
    record = synthetic_record(seed=4)
    # The reference constant is computed once, on the original record, and held fixed.
    reference = reference_levels(raw=run_cheap_estimator(name="PR", record=record), record=record)

    def scale(record: CapacityRecord) -> np.ndarray:
        raw = run_cheap_estimator(name="PR", record=record)
        return issue_time_estimates(record=record, raw=raw, reference=reference).scale_mw

    def causal(record: CapacityRecord, days: np.ndarray) -> np.ndarray:
        return scale(record)[:, days]

    def leaky(record: CapacityRecord, days: np.ndarray) -> np.ndarray:
        return scale(record)[:, days + 1]

    assert_scale_ignores_the_issue_day_onward(
        record=record, scale_function=causal, target_days=np.array([420, 500]), seed=0
    )
    with pytest.raises(ValueError, match="issue day onward"):
        assert_scale_ignores_the_issue_day_onward(
            record=record, scale_function=leaky, target_days=np.array([420]), seed=0
        )


def test_the_issue_time_scale_is_the_estimate_from_two_days_earlier() -> None:
    record = synthetic_record(seed=5)
    raw = run_cheap_estimator(name="NR", record=record)
    scale = issue_time_estimates(record=record, raw=raw, reference=None).scale_mw
    day = 500
    assert ISSUE_LAG_DAYS == 2
    assert scale[0, day] == pytest.approx(raw.level[0, day - 2])
    assert scale[0, day] != pytest.approx(raw.level[0, day - 1])


def test_a_null_estimate_and_a_short_history_fall_back_and_are_counted() -> None:
    record = synthetic_record(seed=6)
    level = np.full((3, DAYS), 5.0)
    level[0, 300] = np.nan
    level[1, 301] = -1.0
    scale = issue_time_estimates(
        record=record, raw=RawEstimate(level=level, sd=None), reference=None
    )
    static = record.static_p99_mw
    # Day 302 reads the estimate for day 300, which is null; day 303 reads the negative one.
    assert scale.fallback_reason[0, 302] == FALLBACK_NULL_ESTIMATE
    assert scale.scale_mw[0, 302] == static[0]
    assert scale.fallback_reason[1, 303] == FALLBACK_NULL_ESTIMATE
    # The first days have too few valid days behind them.
    assert (scale.fallback_reason[:, :20] == FALLBACK_SHORT_HISTORY).all()
    assert (scale.scale_mw[:, :20] == static[:, None]).all()
    assert scale.fallback_reason[0, 400] == FALLBACK_NONE
    assert scale.scale_mw[0, 400] == pytest.approx(max(5.0, FLOOR_SHARE_OF_STATIC * static[0]))


def test_the_scale_is_never_below_a_fifth_of_the_static_power() -> None:
    record = synthetic_record(seed=7)
    level = np.full((3, DAYS), 1e-6)
    scale = issue_time_estimates(
        record=record, raw=RawEstimate(level=level, sd=None), reference=None
    )
    floor = FLOOR_SHARE_OF_STATIC * record.static_p99_mw
    assert scale.scale_mw[:, 400] == pytest.approx(floor)
    above = issue_time_estimates(
        record=record,
        raw=RawEstimate(level=np.broadcast_to(floor[:, None] * 2, (3, DAYS)), sd=None),
        reference=None,
    )
    assert above.scale_mw[:, 400] == pytest.approx(2 * floor)


def test_a_day_is_valid_only_if_its_observed_hours_hold_half_its_clear_sky_energy() -> None:
    record = synthetic_record(seed=8)
    power = record.power_mw.copy()
    clear = record.clear_sky_w_m2[0, 100]
    daylight = np.flatnonzero(clear > 0)
    power[0, 100, daylight[: len(daylight) // 4]] = np.nan
    power[0, 101, daylight[: len(daylight) // 2 + 2]] = np.nan
    power[0, 102, :] = np.nan
    valid = valid_days(record=record.with_power(power_mw=power))
    assert valid[0, 100]
    assert not valid[0, 101]
    assert not valid[0, 102]


def test_the_neighbour_ratio_matches_a_plain_loop_over_the_other_plants() -> None:
    record = synthetic_record(plants=4, seed=9)
    power = record.power_mw.copy()
    power[2, 200:230] = np.nan
    power[3, 100:104, :] = 0.0
    record = record.with_power(power_mw=power)
    valid = valid_days(record=record)
    energy = np.nansum(record.power_mw, axis=2) / record.static_p99_mw[:, None]
    relative = np.where(valid, energy, np.nan)

    def rolling(row: np.ndarray, day: int) -> float:
        window = row[max(day - 6, 0) : day + 1]
        finite = window[np.isfinite(window)]
        return float(finite.mean()) if len(finite) >= 5 else float("nan")

    expected = np.full((4, DAYS), np.nan)
    for day in range(DAYS):
        rolled = [rolling(relative[p], day) for p in range(4)]
        for p in range(4):
            others = [v for q, v in enumerate(rolled) if q != p and np.isfinite(v)]
            if np.isfinite(rolled[p]) and others and np.mean(others) > 0:
                expected[p, day] = rolled[p] / np.mean(others)
    actual = neighbour_ratio(record=record)
    assert np.allclose(actual, expected, equal_nan=True, rtol=1e-9)


def test_the_kalman_filters_standard_deviation_grows_while_there_is_no_data() -> None:
    record = synthetic_record(seed=10)
    power = record.power_mw.copy()
    power[0, 300:340] = np.nan
    gap = estimate_kf(
        record=record.with_power(power_mw=power),
        plant=0,
        params={"process_sd": 0.01, "observation_sd": 0.1},
    )
    assert gap.sd is not None
    during = gap.sd[301:339] / gap.level[301:339]
    assert (np.diff(during) > 0).all()
    assert gap.sd[345] / gap.level[345] < during[-1]


def test_the_hindsight_reference_never_reads_the_target_week_but_reads_its_neighbours() -> None:
    record = synthetic_record(seed=11)
    week_first, week_last = 413, 419  # a week starts at a multiple of 7: 413 = 59 * 7
    # The reference constants are computed once on the original record and held fixed.
    references = {
        name: reference_levels(raw=run_cheap_estimator(name=name, record=record), record=record)
        for name in ("PR", "KF")
    }

    def hindsight(candidate: CapacityRecord, name: str) -> np.ndarray:
        forward = run_cheap_estimator(name=name, record=candidate)
        backward = run_cheap_estimator(name=name, record=candidate.reversed())
        reference = references[name]
        return hindsight_reference_estimates(
            record=candidate, forward=forward, backward_on_reversed=backward, reference=reference
        ).scale_mw

    for name in ("PR", "KF"):
        original = hindsight(record, name)
        altered_week = record.power_mw.copy()
        altered_week[:, week_first : week_last + 1] *= 0.3
        altered_before = record.power_mw.copy()
        altered_before[:, week_first - 7 : week_first] *= 0.3
        same = hindsight(record.with_power(power_mw=altered_week), name)
        different = hindsight(record.with_power(power_mw=altered_before), name)
        assert np.allclose(
            original[:, week_first : week_last + 1], same[:, week_first : week_last + 1]
        )
        assert not np.allclose(
            original[:, week_first : week_last + 1], different[:, week_first : week_last + 1]
        )
        # Every day of a week takes the week's value.
        assert (original[:, week_first : week_last + 1] == original[:, [week_first]]).all()


def test_the_hindsight_reference_uses_the_one_side_that_exists_at_the_ends_of_the_record() -> None:
    record = synthetic_record(seed=12)
    forward = run_cheap_estimator(name="NR", record=record)
    backward = run_cheap_estimator(name="NR", record=record.reversed())
    result = hindsight_reference_estimates(
        record=record, forward=forward, backward_on_reversed=backward, reference=None
    )
    # The first weeks have no 30 valid days behind them, so they use the backward side only.
    assert result.fallback_reason[0, 7] == FALLBACK_NONE
    assert result.scale_mw[0, 7] == pytest.approx(
        run_cheap_estimator(name="NR", record=record.reversed()).level[0, DAYS - 1 - 14]
    )


def test_calibration_puts_the_reference_years_median_at_the_static_power() -> None:
    record = synthetic_record(seed=13)
    raw = run_cheap_estimator(name="PR", record=record)
    reference = reference_levels(raw=raw, record=record)
    calibrated = calibrate(raw=raw, record=record, reference=reference)
    days = reference_day_indices(record=record)
    assert len(days) > 10
    assert np.nanmedian(calibrated.level[0, days]) == pytest.approx(record.static_p99_mw[0])
    assert calibrate(raw=raw, record=record, reference=None) is raw


def test_a_row_reads_the_scale_of_the_day_its_hour_midpoint_falls_in() -> None:
    record = synthetic_record(seed=14, days=40)
    start = record.dates[0].astype("datetime64[D]").astype(object)
    midnight = datetime(start.year, start.month, start.day, tzinfo=UTC)
    rows = pl.DataFrame(
        {
            "site": ["B", "A", "A"],
            # The hour ending at 00:00 belongs to the day before; one ending at 01:00 to the day.
            "time": [
                midnight.replace(day=start.day + 2),
                midnight.replace(day=start.day + 2),
                midnight.replace(day=start.day + 2, hour=1),
            ],
        }
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    plants, days = target_day_indices(record=record, rows=rows)
    assert plants.tolist() == [1, 0, 0]
    assert days.tolist() == [1, 1, 2]
    scale = np.arange(3 * 40, dtype=float).reshape(3, 40)
    assert scale_for_rows(record=record, scale_mw=scale, rows=rows).tolist() == [41.0, 1.0, 2.0]


def test_a_row_for_an_unknown_plant_or_a_day_outside_the_record_is_refused() -> None:
    record = synthetic_record(seed=15, days=40)
    stamp = datetime(2023, 6, 5, 12, tzinfo=UTC)
    with pytest.raises(ValueError, match="lacks"):
        target_day_indices(
            record=record,
            rows=pl.DataFrame({"site": ["Z"], "time": [stamp]}).with_columns(
                pl.col("time").dt.replace_time_zone("UTC")
            ),
        )
    far = datetime(2030, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="outside"):
        target_day_indices(
            record=record,
            rows=pl.DataFrame({"site": ["A"], "time": [far]}).with_columns(
                pl.col("time").dt.replace_time_zone("UTC")
            ),
        )


def test_a_record_with_mismatched_arrays_is_refused_and_reversing_twice_restores_it() -> None:
    record = synthetic_record(seed=16, days=30)
    with pytest.raises(ValueError, match="power_mw has shape"):
        CapacityRecord(
            sites=record.sites,
            dates=record.dates,
            power_mw=record.power_mw[:, :10],
            clear_sky_w_m2=record.clear_sky_w_m2,
            cams_ghi_w_m2=record.cams_ghi_w_m2,
            cams_shape=record.cams_shape,
            clear_shape=record.clear_shape,
            static_p99_mw=record.static_p99_mw,
        )
    twice = record.reversed().reversed()
    assert np.array_equal(twice.power_mw, record.power_mw)
    assert np.array_equal(twice.dates, record.dates)
    assert set(CHEAP_ESTIMATORS) == set(ALLOWED_DELAY_DAYS)
    name: EstimatorNameType = "NR"
    assert run_cheap_estimator(name=name, record=record).level.shape == (3, 30)


def test_a_longer_lag_reads_an_older_estimate() -> None:
    record = synthetic_record(seed=17)
    raw = run_cheap_estimator(name="NR", record=record)
    three = issue_time_estimates(record=record, raw=raw, reference=None, lag_days=3).scale_mw
    assert three[0, 500] == pytest.approx(raw.level[0, 497])
