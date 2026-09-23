from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from studies.solar import cos_zenith, cos_zenith_hour_mean, zenith
from studies.timestamp_checks import (
    best_offset_minutes,
    check_hour_ending,
    correlation_by_offset,
)

LATITUDE = 53.0
LONGITUDE = -0.3


def _stamps() -> pl.Series:
    # Every 7th day for a year, every hour, so each season's day length is represented.
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pl.Series(
        "time",
        [start + timedelta(days=day, hours=hour) for day in range(0, 365, 7) for hour in range(24)],
    ).dt.replace_time_zone("UTC")


def _clouds(*, n: int) -> np.ndarray:
    return np.random.default_rng(11).uniform(0.2, 1.0, size=n)


def test_a_mean_over_the_hour_ending_at_the_label_peaks_half_an_hour_early():
    stamps = _stamps()
    ghi = 900.0 * cos_zenith_hour_mean(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE)
    ghi *= _clouds(n=len(ghi))

    correlations = correlation_by_offset(
        times=stamps, ghi=ghi, latitude=LATITUDE, longitude=LONGITUDE
    )

    assert best_offset_minutes(correlations=correlations) == -30
    check_hour_ending(correlations=correlations, name="synthetic")


def test_a_mean_over_the_hour_beginning_at_the_label_fails_the_check():
    stamps = _stamps()
    ghi = 900.0 * cos_zenith_hour_mean(
        stamps=stamps.dt.offset_by("1h"), latitude=LATITUDE, longitude=LONGITUDE
    )
    ghi *= _clouds(n=len(ghi))

    correlations = correlation_by_offset(
        times=stamps, ghi=ghi, latitude=LATITUDE, longitude=LONGITUDE
    )

    assert best_offset_minutes(correlations=correlations) == 30
    with pytest.raises(ValueError, match=r"\+30 minutes"):
        check_hour_ending(correlations=correlations, name="synthetic")


def test_a_snapshot_read_as_an_hourly_mean_fails_the_check():
    stamps = _stamps()
    ghi = 900.0 * cos_zenith(
        zenith_deg=zenith(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE)
    )
    ghi *= _clouds(n=len(ghi))

    correlations = correlation_by_offset(
        times=stamps, ghi=ghi, latitude=LATITUDE, longitude=LONGITUDE
    )

    assert best_offset_minutes(correlations=correlations) == 0
    with pytest.raises(ValueError, match="settle which window"):
        check_hour_ending(correlations=correlations, name="synthetic")


def test_stamps_and_values_of_different_lengths_raise():
    with pytest.raises(ValueError, match="stamps against"):
        correlation_by_offset(
            times=_stamps()[:3], ghi=np.ones(2), latitude=LATITUDE, longitude=LONGITUDE
        )


def _peaking_at(offset: int) -> dict[int, float]:
    return {candidate: 0.5 - abs(candidate - offset) / 1000 for candidate in range(-60, 65, 5)}


@pytest.mark.parametrize("offset", [-45, -15])
def test_a_peak_15_minutes_from_the_hour_ending_offset_passes(offset: int):
    check_hour_ending(correlations=_peaking_at(offset), name="synthetic")


@pytest.mark.parametrize("offset", [-50, -10])
def test_a_peak_20_minutes_from_the_hour_ending_offset_fails(offset: int):
    with pytest.raises(ValueError, match="settle which window"):
        check_hour_ending(correlations=_peaking_at(offset), name="synthetic")


def test_undefined_correlations_are_ignored_rather_than_read_as_the_first_offset():
    correlations = _peaking_at(-30) | {-60: float("nan")}

    assert best_offset_minutes(correlations=correlations) == -30


def test_a_series_with_no_defined_correlation_fails_the_check():
    stamps = _stamps()[:48]

    correlations = correlation_by_offset(
        times=stamps, ghi=np.zeros(48), latitude=LATITUDE, longitude=LONGITUDE
    )

    with pytest.raises(ValueError, match="no offset"):
        check_hour_ending(correlations=correlations, name="synthetic")


def test_fewer_than_two_hours_raise():
    with pytest.raises(ValueError, match="at least two"):
        correlation_by_offset(
            times=_stamps()[:1], ghi=np.ones(1), latitude=LATITUDE, longitude=LONGITUDE
        )
