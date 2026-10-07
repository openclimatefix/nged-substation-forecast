"""Tests for the classifier of `studies/solar_bmu_census/classify.py`.

Every series is synthetic, so no test needs `data/`. Each test fails on the bug it exists for.
Among those bugs are months of zero output before commissioning that dilute the correlation, a
commissioning month that is not left out, a unit already running at the start that loses its first
month, meter noise read as generation, and a rule that ignores the sign of the output.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import classify
import numpy as np
import polars as pl
import pytest
from studies.solar import cos_zenith, zenith

START: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
DAYS: Final[int] = 365
HALF_HOURS_PER_DAY: Final[int] = 48


def _stamps(*, days: int = DAYS) -> pl.Series:
    """Return the half-hour end times of `days` whole days from `START`, in UTC."""
    return pl.datetime_range(
        START + timedelta(minutes=30),
        START + timedelta(days=days),
        interval="30m",
        time_zone="UTC",
        time_unit="us",
        eager=True,
    ).alias("half_hour_end_time")


def _sun(*, stamps: pl.Series) -> np.ndarray:
    """Return the cosine of the zenith at the classifier's point, at each half-hour midpoint."""
    midpoints = stamps.dt.offset_by("-15m")
    return cos_zenith(
        zenith_deg=zenith(
            stamps=midpoints,
            latitude=classify.REFERENCE_LATITUDE,
            longitude=classify.REFERENCE_LONGITUDE,
        )
    )


def _frame(*, stamps: pl.Series, values: np.ndarray) -> pl.DataFrame:
    """Return a B1610-shaped frame."""
    return pl.DataFrame({"half_hour_end_time": stamps, "output_mwh": values})


def _solar_like(*, stamps: pl.Series) -> np.ndarray:
    """Return output proportional to the sun, with a small negative reading at night."""
    sun = _sun(stamps=stamps)
    return np.where(sun > 0, 25.0 * sun, -0.05)


def _classify(*, stamps: pl.Series, values: np.ndarray) -> classify.Behaviour:
    return classify.classify_behaviour(
        output=_frame(stamps=stamps, values=values), window_start=START
    )


def test_solar_shaped_output_is_solar() -> None:
    stamps = _stamps()
    result = _classify(stamps=stamps, values=_solar_like(stamps=stamps))
    assert result.behaviour == "solar"
    assert result.correlation is not None
    assert result.correlation > 0.95


def test_constant_output_is_no_output_and_never_solar() -> None:
    stamps = _stamps()
    result = _classify(stamps=stamps, values=np.full(len(stamps), 3.0))
    assert result.correlation is None
    assert result.behaviour == "no_output"


def test_all_zero_output_is_no_output() -> None:
    stamps = _stamps()
    assert _classify(stamps=stamps, values=np.zeros(len(stamps))).behaviour == "no_output"


def test_empty_frame_is_no_output_and_does_not_raise() -> None:
    empty = pl.DataFrame(
        schema={"half_hour_end_time": pl.Datetime("us", "UTC"), "output_mwh": pl.Float64}
    )
    result = classify.classify_behaviour(output=empty, window_start=START)
    assert result.behaviour == "no_output"


def test_meter_noise_is_not_generation() -> None:
    """A unit that never generates still publishes readings of a few thousandths of a MWh."""
    stamps = _stamps()
    noise = np.random.default_rng(0).uniform(0.0, classify.POSITIVE_FLOOR_MWH / 2, len(stamps))
    result = _classify(stamps=stamps, values=noise)
    assert result.positive_half_hours == 0
    assert result.behaviour == "no_output"


def test_a_unit_that_starts_generating_late_is_not_diluted_by_the_silence_before() -> None:
    stamps = _stamps()
    values = _solar_like(stamps=stamps)
    values[: 200 * HALF_HOURS_PER_DAY] = 0.0
    assert _classify(stamps=stamps, values=values).behaviour == "solar"


def test_the_commissioning_month_is_left_out() -> None:
    """Erratic output in the 30 days after first output must not count."""
    stamps = _stamps()
    values = _solar_like(stamps=stamps)
    first = 20 * HALF_HOURS_PER_DAY
    ramp = 30 * HALF_HOURS_PER_DAY
    values[:first] = 0.0
    values[first : first + ramp] = np.random.default_rng(1).uniform(5.0, 25.0, ramp)
    series = classify.analysis_series(
        output=_frame(stamps=stamps, values=values), window_start=START
    )
    assert series["half_hour_end_time"].min() >= stamps[first] + classify.COMMISSIONING_SKIP
    assert _classify(stamps=stamps, values=values).behaviour == "solar"


def test_a_unit_already_running_at_the_window_start_keeps_its_first_month() -> None:
    stamps = _stamps()
    series = classify.analysis_series(
        output=_frame(stamps=stamps, values=_solar_like(stamps=stamps)), window_start=START
    )
    assert series.height == len(stamps)


def test_a_unit_first_generating_within_the_last_month_is_no_output() -> None:
    stamps = _stamps(days=60)
    values = np.zeros(len(stamps))
    values[-20 * HALF_HOURS_PER_DAY :] = _solar_like(stamps=stamps)[-20 * HALF_HOURS_PER_DAY :]
    assert _classify(stamps=stamps, values=values).behaviour == "no_output"


def test_fewer_than_the_minimum_positive_half_hours_is_no_output() -> None:
    """The unit runs from the first day and follows the sun, but only 99 half-hours are positive."""
    stamps = _stamps(days=30)
    values = _solar_like(stamps=stamps)
    values[np.flatnonzero(values > 0)[classify.MIN_POSITIVE_HALF_HOURS - 1 :]] = 0.0
    result = _classify(stamps=stamps, values=values)
    assert result.positive_half_hours == classify.MIN_POSITIVE_HALF_HOURS - 1
    assert result.behaviour == "no_output"


def test_output_that_is_positive_only_at_night_is_not_solar() -> None:
    stamps = _stamps()
    sun = _sun(stamps=stamps)
    assert _classify(stamps=stamps, values=np.where(sun > 0, 0.0, 5.0)).behaviour == "not_solar"


def test_noise_with_no_daily_pattern_is_not_solar() -> None:
    stamps = _stamps()
    noise = np.random.default_rng(0).uniform(0.0, 10.0, len(stamps))
    assert _classify(stamps=stamps, values=noise).behaviour == "not_solar"


def test_zeros_during_daylight_are_dropped_and_zeros_at_night_are_kept() -> None:
    stamps = _stamps()
    sun = _sun(stamps=stamps)
    series = classify.analysis_series(
        output=_frame(stamps=stamps, values=np.where(sun > 0, 25.0 * sun, 0.0)), window_start=START
    )
    cleaned = classify.drop_daytime_zeros(series=series)
    by_day = cleaned.filter(pl.col("cos_zenith") > classify.DAYLIGHT_COS_ZENITH)
    by_night = cleaned.filter(pl.col("cos_zenith") == 0)
    assert (by_day["output_mwh"].to_numpy() > 0).all()
    assert by_night.height > 0
    assert (by_night["output_mwh"].to_numpy() == 0).all()


def test_a_metering_fault_at_midday_does_not_lower_a_solar_units_correlation() -> None:
    stamps = _stamps()
    values = _solar_like(stamps=stamps)
    values[np.flatnonzero(_sun(stamps=stamps) > 0.7)[::3]] = 0.0
    result = _classify(stamps=stamps, values=values)
    assert result.correlation is not None
    assert result.raw_correlation is not None
    assert result.correlation > result.raw_correlation
    assert result.correlation > 0.95


def test_census_basis_names_the_evidence() -> None:
    assert classify.census_basis(by_type=True, by_behaviour=True) == "type and behaviour"
    assert classify.census_basis(by_type=True, by_behaviour=False) == "type only"
    assert classify.census_basis(by_type=False, by_behaviour=True) == "behaviour only"
    assert classify.census_basis(by_type=False, by_behaviour=False) == "neither"


def test_igcpu_solar_ids_keeps_only_solar_rows_with_a_bmu() -> None:
    rows = [
        {"psrType": "Solar", "bmUnit": "T_A"},
        {"psrType": "Solar", "bmUnit": None},
        {"psrType": "Wind Onshore", "bmUnit": "T_B"},
    ]
    assert classify.igcpu_solar_ids(igcpu=rows) == {"T_A"}


def test_cos_zenith_at_a_literal_midsummer_noon_and_midnight() -> None:
    """The reference point and the 15-minute midpoint offset, checked against literal times."""
    stamps = pl.Series(
        "half_hour_end_time",
        [datetime(2026, 6, 21, 0, 15, tzinfo=UTC), datetime(2026, 6, 21, 12, 15, tzinfo=UTC)],
    ).dt.cast_time_unit("us")
    series = classify.analysis_series(
        output=_frame(stamps=stamps, values=np.array([1.0, 1.0])),
        window_start=datetime(2026, 6, 21, tzinfo=UTC),
    )
    midnight, noon = series["cos_zenith"].to_list()
    assert 0.86 < noon < 0.88  # a zenith of 53.0 - 23.4 = 29.6 degrees at 53 N
    assert midnight == 0.0


@pytest.mark.parametrize(("noise_scale", "expected"), [(8.0, "solar"), (11.0, "not_solar")])
def test_the_solar_threshold_sits_at_a_correlation_of_0_6(
    noise_scale: float, expected: str
) -> None:
    stamps = _stamps()
    noise = np.random.default_rng(3).normal(0, noise_scale, len(stamps))
    values = np.clip(25.0 * _sun(stamps=stamps) + noise, 0.5, None)
    result = _classify(stamps=stamps, values=values)
    assert result.correlation is not None
    assert 0.5 < result.correlation < 0.7
    assert result.behaviour == expected


@pytest.mark.parametrize(("positive", "expected"), [(99, "no_output"), (100, "solar")])
def test_one_hundred_positive_half_hours_is_enough(positive: int, expected: str) -> None:
    stamps = _stamps(days=30)
    sun = _sun(stamps=stamps)
    values = np.where(sun > 0, 25.0 * sun, -0.05)
    last = np.flatnonzero(values > 0)[positive - 1] + 1  # end the series at the Nth positive
    result = _classify(stamps=stamps[:last], values=values[:last])
    assert result.positive_half_hours == positive
    assert result.behaviour == expected


def test_readings_of_0_008_mwh_are_noise_and_of_0_3_mwh_are_generation() -> None:
    stamps = _stamps(days=30)
    sun = _sun(stamps=stamps)
    values = np.where(sun > 0.1, 0.3, np.where(sun > 0, 0.008, -0.05))
    result = _classify(stamps=stamps, values=values)
    assert result.positive_half_hours == int((sun > 0.1).sum())


def test_the_commissioning_skip_is_thirty_days_from_the_first_output() -> None:
    stamps = _stamps(days=120)
    values = np.zeros(len(stamps))
    values[20 * HALF_HOURS_PER_DAY :] = 1.0
    series = classify.analysis_series(
        output=_frame(stamps=stamps, values=values), window_start=START
    )
    assert series["half_hour_end_time"][0] == stamps[20 * HALF_HOURS_PER_DAY] + timedelta(days=30)


def test_a_first_output_seven_days_in_counts_as_commissioning() -> None:
    stamps = _stamps(days=60)
    values = np.zeros(len(stamps))
    first = 7 * HALF_HOURS_PER_DAY - 1
    values[first:] = 1.0
    assert stamps[first] == START + timedelta(days=7)
    series = classify.analysis_series(
        output=_frame(stamps=stamps, values=values), window_start=START
    )
    assert series["half_hour_end_time"][0] == START + timedelta(days=37)


def test_a_unit_judged_on_two_months_can_be_solar() -> None:
    stamps = _stamps(days=60)
    assert _classify(stamps=stamps, values=_solar_like(stamps=stamps)).behaviour == "solar"


def test_noise_half_hours_do_not_count_towards_the_minimum() -> None:
    stamps = _stamps(days=30)
    values = np.full(len(stamps), 0.005)
    values[:150] = 5.0
    assert _classify(stamps=stamps, values=values).positive_half_hours == 150


def test_drop_daytime_zeros_on_literal_rows() -> None:
    series = pl.DataFrame(
        {
            "output_mwh": [0.0, -0.05, 0.0, 0.0, 0.0],
            "cos_zenith": [0.5, 0.5, 0.1, 0.05, 0.3],
        }
    )
    kept = classify.drop_daytime_zeros(series=series)
    assert kept.to_dicts() == [
        {"output_mwh": -0.05, "cos_zenith": 0.5},
        {"output_mwh": 0.0, "cos_zenith": 0.1},
        {"output_mwh": 0.0, "cos_zenith": 0.05},
    ]


def test_correlation_is_none_when_the_sun_never_varies() -> None:
    series = pl.DataFrame({"output_mwh": [1.0, 2.0, 3.0], "cos_zenith": [0.0, 0.0, 0.0]})
    assert classify.sun_following_correlation(series=series) is None


def _midday_stamps(*, days: int) -> pl.Series:
    """Return one half-hour end time at 12:15 UTC on each of `days` days, when the sun is up."""
    return pl.Series(
        "half_hour_end_time",
        [START + timedelta(days=day, hours=12, minutes=15) for day in range(days)],
    ).dt.cast_time_unit("us")


def test_p99_skips_the_commissioning_month_and_the_daytime_zeros() -> None:
    """Days 0 to 9 are zero, day 10 is the first output, and days 10 to 39 are the skipped month.

    Days 40 to 139 then hold the readings 1 to 100 MWh, and days 140 to 289 hold daytime zeros.
    Keeping the commissioning month adds readings of 500 and 1000 MWh, and keeping the zeros pulls
    the percentile down to 97.5 MWh.
    """
    values = np.zeros(290)
    values[10] = 1000.0
    values[11:40] = 500.0
    values[40:140] = np.arange(1.0, 101.0)
    output = _frame(stamps=_midday_stamps(days=290), values=values)
    p99 = classify.p99_output_mw(output=output, window_start=START)
    assert p99 == pytest.approx(198.02, abs=1e-6)  # 99.01 MWh at 2 half-hours per hour


def test_p99_is_none_when_the_bmu_never_generates() -> None:
    output = _frame(stamps=_midday_stamps(days=20), values=np.zeros(20))
    assert classify.p99_output_mw(output=output, window_start=START) is None
