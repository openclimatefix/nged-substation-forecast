"""Tests of the calendar and weather columns of the differentiable-physics series."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.dp_calendar import (
    BAND_NAMES,
    DAY_TYPES,
    HOLIDAY_PERIODS,
    SMOOTHING_HOURS,
    bank_holiday_names,
    calendar_columns,
    lighting_ramps,
    trailing_mean_temperatures,
)


def _columns(*stamps: datetime):  # noqa: ANN202
    return calendar_columns(end_times=pl.Series("t", list(stamps), dtype=pl.Datetime("us", "UTC")))


def _period(*, day: date) -> str | None:
    """Return the holiday period of local midday on `day`."""
    columns = _columns(datetime(day.year, day.month, day.day, 12, 0, tzinfo=UTC))
    index = int(columns.holiday_period[0])
    return None if index < 0 else HOLIDAY_PERIODS[index]


def test_a_half_hour_belongs_to_the_day_and_band_it_starts_in():
    # The half-hour ending 00:00 local on Saturday 2025-01-04 began at 23:30 on Friday, and the one
    # ending 06:00 began at 05:30 in the night band.
    columns = _columns(
        datetime(2025, 1, 4, 0, 0, tzinfo=UTC),
        datetime(2025, 1, 4, 6, 0, tzinfo=UTC),
        datetime(2025, 1, 4, 6, 30, tzinfo=UTC),
    )
    assert [DAY_TYPES[i] for i in columns.day_type] == ["weekday", "saturday", "saturday"]
    assert [BAND_NAMES[i] for i in columns.band] == ["evening", "night", "morning"]
    assert columns.half_hour.tolist() == [47, 11, 12]


def test_the_local_half_hour_follows_daylight_saving_time():
    # In July, 10:30 UTC ends the half-hour that started at 11:00 local (British Summer Time).
    columns = _columns(datetime(2025, 7, 1, 10, 30, tzinfo=UTC))
    assert columns.half_hour.tolist() == [22]
    assert BAND_NAMES[columns.band[0]] == "midday"


def test_bank_holidays_take_the_bank_holiday_day_type_and_their_own_period():
    columns = _columns(
        datetime(2025, 12, 25, 12, tzinfo=UTC), datetime(2025, 4, 21, 12, tzinfo=UTC)
    )
    assert [DAY_TYPES[i] for i in columns.day_type] == ["bank_holiday", "bank_holiday"]
    assert _period(day=date(2025, 12, 25)) == "Christmas Day"
    assert _period(day=date(2025, 4, 21)) == "Easter Monday"


def test_a_one_off_bank_holiday_is_not_one_of_the_recurring_periods():
    assert _period(day=date(2022, 9, 19)) == "one-off bank holiday"
    assert set(bank_holiday_names().values()) <= set(HOLIDAY_PERIODS[:9])


def test_easter_saturday_and_sunday_get_their_own_periods():
    assert _period(day=date(2025, 4, 19)) == "Easter Saturday"
    assert _period(day=date(2025, 4, 20)) == "Easter Sunday"


def test_a_working_day_between_a_bank_holiday_and_a_weekend_is_a_bridge_day():
    # Friday 2024-12-27 follows Boxing Day on Thursday 2024-12-26 and precedes the weekend.
    assert _period(day=date(2024, 12, 27)) == "bridge day"
    assert _period(day=date(2025, 7, 15)) is None


def test_the_christmas_run_is_the_twenty_fourth_and_the_twenty_seventh_to_thirty_first():
    assert _period(day=date(2025, 12, 24)) == "Christmas run"
    assert _period(day=date(2025, 12, 29)) == "Christmas run"
    assert _period(day=date(2025, 12, 23)) is None


def test_the_christmas_tent_is_one_on_the_day_and_zero_beyond_the_window():
    columns = _columns(
        datetime(2025, 12, 25, 12, tzinfo=UTC),
        datetime(2025, 12, 18, 12, tzinfo=UTC),
        datetime(2025, 12, 1, 12, tzinfo=UTC),
        datetime(2026, 1, 15, 12, tzinfo=UTC),
    )
    assert columns.tent[0] == 1.0
    assert 0.0 < columns.tent[1] < 1.0
    assert columns.tent[2] == 0.0
    assert columns.tent[3] == 0.0
    assert columns.signed_tent[1] < 0.0
    assert columns.signed_tent[0] == 0.0


def test_the_season_year_starts_in_july():
    columns = _columns(datetime(2025, 6, 30, 12, tzinfo=UTC), datetime(2025, 7, 1, 12, tzinfo=UTC))
    assert columns.season_year.tolist() == [2024, 2025]
    assert columns.quarter.tolist() == [1, 2]


def test_trailing_means_use_only_earlier_readings():
    stamps = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(minutes=30 * i) for i in range(400)]
    series = pl.DataFrame(
        {"valid_time": stamps, "temperature_c": np.arange(400, dtype=np.float64)}
    ).with_columns(pl.col("valid_time").dt.replace_time_zone("UTC"))
    means = trailing_mean_temperatures(series=series)
    # The 6-hour mean at position 100 averages positions 89 to 100.
    assert means["temperature_c_k6"][100] == pytest.approx(np.mean(np.arange(89, 101)))
    assert means["temperature_c_k6"][10] is None
    assert means["temperature_c_k6"][11] is not None
    assert all(f"temperature_c_k{k}" in means.columns for k in SMOOTHING_HOURS)


def test_trailing_means_refuse_a_series_with_a_gap():
    stamps = [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=h) for h in (0, 1, 2, 4)]
    series = pl.DataFrame({"valid_time": stamps, "temperature_c": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="evenly spaced"):
        trailing_mean_temperatures(series=series)


def test_lighting_ramps_are_zero_at_midday_and_saturate_at_night():
    ramps = lighting_ramps(elevation_deg=np.array([50.0, 12.0, 3.0, -3.0, -60.0]))
    assert ramps[0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert ramps[1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert ramps[2].tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert ramps[4].tolist() == [1.0, 1.0, 1.0, 1.0]
