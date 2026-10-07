"""Tests for `studies/solar_bmu_census/solar_estimate.py`."""

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
import solar_estimate


def _flat_topped(*, capacity_mw: float, ratio: float, sun: np.ndarray) -> np.ndarray:
    return np.minimum(ratio * capacity_mw * sun, capacity_mw)


SUN = np.tile(np.linspace(0.0, 1.0, 101), 50)


def test_a_flat_topped_series_gives_back_its_capacity() -> None:
    output = _flat_topped(capacity_mw=80.0, ratio=1.4, sun=SUN)
    fitted = solar_estimate.fit_ac_capacity_mw(sun=SUN, output_mw=output, dc_ac_ratio=1.4)
    assert fitted == pytest.approx(80.0, rel=0.02)


def test_clouds_below_the_envelope_do_not_lower_the_fit() -> None:
    # Half the half-hours are cloudy and give 30% of the clear-sky output.
    clear = _flat_topped(capacity_mw=80.0, ratio=1.4, sun=SUN)
    cloudy = np.where(np.arange(SUN.size) % 2 == 0, 0.3 * clear, clear)
    fitted = solar_estimate.fit_ac_capacity_mw(sun=SUN, output_mw=cloudy, dc_ac_ratio=1.4)
    assert fitted == pytest.approx(80.0, rel=0.02)


def test_a_wrong_ratio_changes_the_fit_in_the_expected_direction() -> None:
    output = _flat_topped(capacity_mw=80.0, ratio=1.4, sun=SUN)
    low = solar_estimate.fit_ac_capacity_mw(sun=SUN, output_mw=output, dc_ac_ratio=1.2)
    high = solar_estimate.fit_ac_capacity_mw(sun=SUN, output_mw=output, dc_ac_ratio=1.6)
    assert low is not None
    assert high is not None
    assert low > 80.0 > high


def test_too_little_daylight_gives_none() -> None:
    sun = np.full(10, 0.5)
    assert (
        solar_estimate.fit_ac_capacity_mw(sun=sun, output_mw=np.ones(10), dc_ac_ratio=1.4) is None
    )


def test_a_series_that_never_generates_gives_none() -> None:
    assert (
        solar_estimate.fit_ac_capacity_mw(sun=SUN, output_mw=np.zeros(SUN.size), dc_ac_ratio=1.4)
        is None
    )


def test_the_estimate_from_a_year_of_synthetic_output_is_near_its_capacity() -> None:
    from classify import REFERENCE_LATITUDE, REFERENCE_LONGITUDE
    from studies.solar import cos_zenith, zenith

    start = datetime(2025, 9, 1, tzinfo=UTC)
    ends = [start + timedelta(minutes=30 * (i + 1)) for i in range(48 * 365)]
    mids = pl.Series(ends).dt.offset_by("-15m")
    sun = cos_zenith(
        zenith_deg=zenith(stamps=mids, latitude=REFERENCE_LATITUDE, longitude=REFERENCE_LONGITUDE)
    )
    clear = np.minimum(1.4 * 60.0 * sun / sun.max(), 60.0)
    output = pl.DataFrame({"half_hour_end_time": ends, "output_mwh": clear / 2})
    estimate = solar_estimate.estimate_solar_ac_capacity_mw(
        output=output, window_start=start, dc_ac_ratio=1.4
    )
    assert estimate == pytest.approx(60.0, rel=0.03)


def test_the_regional_sun_averages_reliable_sites_and_scales_to_the_clear_sky_peak() -> None:
    noon = datetime(2026, 6, 21, 12, tzinfo=UTC)
    cams = pl.DataFrame(
        {
            "site": ["A", "B", "C", "A", "B", "C"],
            "time": [
                noon,
                noon,
                noon,
                noon + timedelta(hours=1),
                noon + timedelta(hours=1),
                noon + timedelta(hours=1),
            ],
            "ghi_w_m2": [400.0, 600.0, 900.0, 100.0, 200.0, 300.0],
            "clear_sky_ghi_w_m2": [800.0, 800.0, 800.0, 700.0, 700.0, 700.0],
            "reliability": [1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
        }
    )
    result = solar_estimate.regional_hourly_sun(cams=cams, min_reliability=0.9)
    # At noon site C is dropped: mean ghi 500, mean clear sky 800, which is the peak.
    assert result["sun"].to_list() == pytest.approx([500.0 / 800.0, 200.0 / 800.0])


def test_the_cams_fit_recovers_the_capacity_and_maps_both_half_hours_to_the_hour_end() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    hour_ends = [start + timedelta(hours=h + 1) for h in range(24 * 40)]
    sun_by_hour = np.array([0.1 + 0.9 * ((h * 7) % 10) / 9 for h in range(len(hour_ends))])
    hourly_sun = pl.DataFrame({"hour_end_time": hour_ends, "sun": sun_by_hour})
    half_hour_ends = [start + timedelta(minutes=30 * (i + 1)) for i in range(48 * 40)]
    # The half-hours ending at hh:30 and (hh+1):00 both take the value of the hour ending (hh+1):00.
    sun_per_half_hour = np.repeat(sun_by_hour, 2)
    output_mw = np.minimum(1.4 * 40.0 * sun_per_half_hour, 40.0)
    output = pl.DataFrame({"half_hour_end_time": half_hour_ends, "output_mwh": output_mw / 2})
    estimate = solar_estimate.estimate_solar_ac_capacity_with_cams_mw(
        output=output, window_start=start, dc_ac_ratio=1.4, hourly_sun=hourly_sun
    )
    assert estimate == pytest.approx(40.0, rel=0.01)


def test_the_least_squares_fit_is_none_without_daylight() -> None:
    assert (
        solar_estimate.fit_ac_capacity_least_squares_mw(
            sun=np.zeros(5), output_mw=np.ones(5), dc_ac_ratio=1.4
        )
        is None
    )
