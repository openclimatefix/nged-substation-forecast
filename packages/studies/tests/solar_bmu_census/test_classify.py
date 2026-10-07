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
