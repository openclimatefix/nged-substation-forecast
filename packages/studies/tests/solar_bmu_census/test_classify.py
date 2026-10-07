"""Tests for the classifier of `studies/solar_bmu_census/classify.py`.

Every series is synthetic, so no test needs `data/`. Each test fails on the bug it exists for: a
constant series whose NaN correlation compares as greater than the threshold, silent months before
commissioning that dilute the correlation, a night test that demands exactly zero, and a rule that
ignores the sign of the output.
"""

from datetime import UTC, datetime, timedelta
from typing import Final

import classify
import numpy as np
import polars as pl
from studies.solar import cos_zenith, zenith

START: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
DAYS: Final[int] = 365


def _stamps(*, start: datetime = START, days: int = DAYS) -> pl.Series:
    """Return the half-hour end times of `days` whole days from `start`, in UTC."""
    return pl.datetime_range(
        start + timedelta(minutes=30),
        start + timedelta(days=days),
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


def _solar_like(*, stamps: pl.Series, capacity_mwh: float = 25.0) -> pl.DataFrame:
    """Return output proportional to the sun, with a small negative reading at night."""
    sun = _sun(stamps=stamps)
    return _frame(stamps=stamps, values=np.where(sun > 0, capacity_mwh * sun, -0.05))


def test_solar_shaped_output_is_solar() -> None:
    result = classify.classify_behaviour(output=_solar_like(stamps=_stamps()))
    assert result.behaviour == "solar"
    assert result.correlation is not None
    assert result.correlation > 0.95


def test_constant_output_is_no_output_and_never_solar() -> None:
    """A constant series has an undefined correlation, which Polars would compare as above 0.6."""
    stamps = _stamps()
    result = classify.classify_behaviour(
        output=_frame(stamps=stamps, values=np.full(len(stamps), 3.0))
    )
    assert result.correlation is None
    assert result.behaviour == "no_output"


def test_all_zero_output_is_no_output() -> None:
    stamps = _stamps()
    result = classify.classify_behaviour(output=_frame(stamps=stamps, values=np.zeros(len(stamps))))
    assert result.behaviour == "no_output"


def test_empty_frame_is_no_output_and_does_not_raise() -> None:
    empty = pl.DataFrame(
        schema={"half_hour_end_time": pl.Datetime("us", "UTC"), "output_mwh": pl.Float64}
    )
    assert classify.classify_behaviour(output=empty).behaviour == "no_output"


def test_a_unit_that_starts_generating_late_is_still_solar() -> None:
    """Zero rows before commissioning must not dilute the correlation of a unit that then runs."""
    stamps = _stamps()
    values = _solar_like(stamps=stamps)["output_mwh"].to_numpy().copy()
    values[: len(stamps) * 10 // 12] = 0.0
    result = classify.classify_behaviour(output=_frame(stamps=stamps, values=values))
    assert result.behaviour == "solar"


def test_output_that_is_positive_only_at_night_is_not_solar() -> None:
    stamps = _stamps()
    sun = _sun(stamps=stamps)
    result = classify.classify_behaviour(
        output=_frame(stamps=stamps, values=np.where(sun > 0, 0.0, 5.0))
    )
    assert result.behaviour == "not_solar"


def test_noise_with_no_daily_pattern_is_not_solar() -> None:
    stamps = _stamps()
    noise = np.random.default_rng(0).uniform(0.0, 10.0, len(stamps))
    assert classify.classify_behaviour(output=_frame(stamps=stamps, values=noise)).behaviour == (
        "not_solar"
    )


def test_correlation_ignores_the_unit_of_capacity() -> None:
    stamps = _stamps()
    small = classify.sun_following_correlation(
        series=classify.analysis_series(output=_solar_like(stamps=stamps, capacity_mwh=2.0))
    )
    large = classify.sun_following_correlation(
        series=classify.analysis_series(output=_solar_like(stamps=stamps, capacity_mwh=200.0))
    )
    assert small is not None
    assert large is not None
    assert abs(small - large) < 0.01


def test_a_unit_with_fewer_than_the_minimum_positive_half_hours_is_no_output() -> None:
    stamps = _stamps(days=30)
    values = np.zeros(len(stamps))
    values[: classify.MIN_POSITIVE_HALF_HOURS - 1] = 1.0
    assert classify.classify_behaviour(output=_frame(stamps=stamps, values=values)).behaviour == (
        "no_output"
    )


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


def test_the_commissioning_month_is_left_out() -> None:
    """Erratic output in the first 30 days after first output must not count."""
    stamps = _stamps()
    values = _solar_like(stamps=stamps)["output_mwh"].to_numpy().copy()
    ramp = 30 * 48
    first = int(np.flatnonzero(values > 0)[0])
    values[first : first + ramp] = np.random.default_rng(1).uniform(0.0, 25.0, ramp)
    series = classify.analysis_series(output=_frame(stamps=stamps, values=values))
    assert series["half_hour_end_time"].min() >= stamps[first] + classify.COMMISSIONING_SKIP
    assert classify.classify_behaviour(output=_frame(stamps=stamps, values=values)).behaviour == (
        "solar"
    )


def test_a_unit_first_generating_within_the_last_month_is_no_output() -> None:
    stamps = _stamps(days=60)
    values = np.zeros(len(stamps))
    values[-20 * 48 :] = _solar_like(stamps=stamps)["output_mwh"].to_numpy()[-20 * 48 :]
    result = classify.classify_behaviour(output=_frame(stamps=stamps, values=values))
    assert result.behaviour == "no_output"


def test_zeros_during_daylight_are_dropped_and_zeros_at_night_are_kept() -> None:
    stamps = _stamps()
    sun = _sun(stamps=stamps)
    values = np.where(sun > 0, 25.0 * sun, 0.0)
    series = classify.analysis_series(output=_frame(stamps=stamps, values=values))
    cleaned = classify.drop_daytime_zeros(series=series)
    by_day = cleaned.filter(pl.col("cos_zenith") > classify.DAYLIGHT_COS_ZENITH)
    by_night = cleaned.filter(pl.col("cos_zenith") == 0)
    assert (by_day["output_mwh"].to_numpy() > 0).all()
    assert by_night.height > 0
    assert (by_night["output_mwh"].to_numpy() == 0).all()


def test_a_metering_fault_at_midday_does_not_lower_a_solar_units_correlation() -> None:
    stamps = _stamps()
    values = _solar_like(stamps=stamps)["output_mwh"].to_numpy().copy()
    sun = _sun(stamps=stamps)
    midday = np.flatnonzero(sun > 0.7)
    values[midday[::3]] = 0.0
    result = classify.classify_behaviour(output=_frame(stamps=stamps, values=values))
    assert result.correlation is not None
    assert result.raw_correlation is not None
    assert result.correlation > result.raw_correlation
    assert result.correlation > 0.95
