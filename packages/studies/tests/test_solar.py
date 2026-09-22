from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest
from studies.solar import (
    azimuth,
    cos_zenith,
    cos_zenith_hour_mean,
    extraterrestrial_horizontal,
    zenith,
)

# Greenwich, so local solar noon is close to 12:00 UTC and the expected values below can be read
# off the astronomy rather than off a previous run of this code.
LATITUDE = 51.48
LONGITUDE = 0.0


def _stamps(hours: list[int], *, month: int = 6, day: int = 21) -> pl.Series:
    return pl.Series(
        "time", [datetime(2025, month, day, hour, tzinfo=UTC) for hour in hours]
    ).dt.replace_time_zone("UTC")


def test_the_sun_is_highest_at_solar_noon_on_the_solstice():
    angles = zenith(stamps=_stamps(list(range(24))), latitude=LATITUDE, longitude=LONGITUDE)

    # At the June solstice the sun's zenith angle at noon is the latitude minus the 23.44-degree
    # axial tilt, so about 28 degrees at Greenwich.
    assert int(np.argmin(angles)) == 12
    assert angles.min() == pytest.approx(51.48 - 23.44, abs=1.0)


def test_the_cosine_is_clipped_to_zero_below_the_horizon():
    midnight = zenith(stamps=_stamps([0], month=12, day=21), latitude=LATITUDE, longitude=LONGITUDE)

    assert midnight[0] > 90.0
    assert cos_zenith(zenith_deg=midnight)[0] == 0.0


def test_the_hour_mean_cosine_falls_between_the_hours_two_endpoints():
    # The sun is rising through the 08:00 hour, so the mean over (07:00, 08:00] sits between the
    # cosine at 07:00 and the cosine at 08:00. A mean taken over the hour *beginning* at the label
    # would sit above both.
    stamps = _stamps([8])
    hour_mean = cos_zenith_hour_mean(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE)
    at_start = cos_zenith(
        zenith_deg=zenith(stamps=stamps.dt.offset_by("-1h"), latitude=LATITUDE, longitude=LONGITUDE)
    )
    at_end = cos_zenith(zenith_deg=zenith(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE))

    assert at_start[0] < hour_mean[0] < at_end[0]


def test_the_extraterrestrial_flux_is_zero_when_the_sun_is_down():
    stamps = _stamps([0], month=12, day=21)
    angles = zenith(stamps=stamps, latitude=LATITUDE, longitude=LONGITUDE)

    assert extraterrestrial_horizontal(stamps=stamps, zenith_deg=angles)[0] == 0.0


def test_the_extraterrestrial_flux_varies_with_the_earth_sun_distance():
    # Earth is nearest the sun in early January and furthest in early July. The inverse-square
    # law puts perihelion about 3.4% above the annual mean and aphelion about 3.3% below it, so
    # the two dates differ by about 7%. Holding the zenith angle fixed isolates that from geometry.
    fixed_zenith = np.zeros(1)
    january = extraterrestrial_horizontal(
        stamps=_stamps([12], month=1, day=3), zenith_deg=fixed_zenith
    )
    july = extraterrestrial_horizontal(
        stamps=_stamps([12], month=7, day=4), zenith_deg=fixed_zenith
    )

    assert january[0] / july[0] == pytest.approx(1.069, abs=0.005)


@pytest.mark.parametrize(
    ("hour", "expected_deg"), [(6, 74.6), (12, 179.1), (18, 284.8)], ids=["dawn", "noon", "dusk"]
)
def test_the_azimuth_swings_from_east_through_south_to_west(hour: int, expected_deg: float):
    # At Greenwich on the solstice the sun rises north of east and sets north of west. The
    # physical model reads this angle to fit a panel's orientation.
    angles = azimuth(stamps=_stamps([hour]), latitude=LATITUDE, longitude=LONGITUDE)

    assert angles[0] == pytest.approx(expected_deg, abs=0.5)
