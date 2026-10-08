from datetime import UTC, datetime

import numpy as np
import polars as pl
from studies.pv_physics import (
    PlantParameters,
    SunAndSky,
    ac_power_mw,
    daylight,
    half_hour_sun_and_sky,
    plane_of_array_w_m2,
    plant_power_mw,
)

LATITUDE = 51.48
LONGITUDE = 0.0


def _sky(*, zenith: float, sun_azimuth: float, dni: float = 800.0, dhi: float = 100.0) -> SunAndSky:
    ghi = dni * np.cos(np.radians(zenith)) + dhi
    return SunAndSky(
        half_hour_end_time=np.array(["2025-06-21T12:00"], dtype="datetime64[us]"),
        ghi_w_m2=np.array([ghi]),
        dni_w_m2=np.array([dni]),
        dhi_w_m2=np.array([dhi]),
        zenith_deg=np.array([zenith]),
        sun_azimuth_deg=np.array([sun_azimuth]),
    )


def test_half_hours_average_back_to_the_hours_irradiance() -> None:
    hours = pl.Series("time", [datetime(2025, 6, 21, h, tzinfo=UTC) for h in (6, 7, 12, 13)])
    hourly = pl.DataFrame({"time": hours, "ghi_w_m2": [150.0, 300.0, 800.0, 750.0]})

    sky = half_hour_sun_and_sky(hourly=hourly, latitude=LATITUDE, longitude=LONGITUDE)

    assert len(sky.ghi_w_m2) == 8
    np.testing.assert_allclose(
        sky.ghi_w_m2.reshape(4, 2).mean(axis=1), [150, 300, 800, 750], rtol=1e-9
    )


def test_the_hours_irradiance_is_shared_by_the_sun_height_inside_the_hour() -> None:
    # The sun is rising at 06:00 to 07:00 UTC in June, so the later half-hour gets more.
    hourly = pl.DataFrame(
        {"time": [datetime(2025, 6, 21, 7, tzinfo=UTC)], "ghi_w_m2": [300.0]}
    ).with_columns(pl.col("time").dt.replace_time_zone("UTC"))

    sky = half_hour_sun_and_sky(hourly=hourly, latitude=LATITUDE, longitude=LONGITUDE)

    assert sky.ghi_w_m2[1] > sky.ghi_w_m2[0] > 0
    assert sky.half_hour_end_time[1] - sky.half_hour_end_time[0] == np.timedelta64(30, "m")


def test_beam_falls_square_on_a_plane_facing_the_sun() -> None:
    sky = _sky(zenith=60.0, sun_azimuth=180.0, dhi=0.0)

    poa = plane_of_array_w_m2(sky=sky, tilt_deg=60.0, azimuth_deg=180.0, tracker=False)

    # Tilt 60 faces the sun at zenith 60 directly: the beam is the whole DNI. The ground adds
    # a little.
    assert 800.0 < poa[0] < 800.0 + 0.2 * sky.ghi_w_m2[0]


def test_an_east_facing_plane_gets_more_than_a_west_facing_one_when_the_sun_is_in_the_east() -> (
    None
):
    sky = _sky(zenith=60.0, sun_azimuth=90.0)

    east = plane_of_array_w_m2(sky=sky, tilt_deg=30.0, azimuth_deg=90.0, tracker=False)
    west = plane_of_array_w_m2(sky=sky, tilt_deg=30.0, azimuth_deg=270.0, tracker=False)

    assert east[0] > west[0] + 300.0


def test_a_tracker_faces_a_low_eastern_sun_but_a_fixed_south_plane_does_not() -> None:
    sky = _sky(zenith=60.0, sun_azimuth=90.0, dhi=0.0)

    tracker = plane_of_array_w_m2(sky=sky, tilt_deg=0.0, azimuth_deg=180.0, tracker=True)
    fixed = plane_of_array_w_m2(sky=sky, tilt_deg=30.0, azimuth_deg=180.0, tracker=False)

    assert tracker[0] > 800.0
    assert fixed[0] < 0.5 * tracker[0]


def test_power_is_clipped_at_the_ac_capacity() -> None:
    power = ac_power_mw(
        poa_w_m2=np.array([100.0, 500.0, 1000.0]), dc_capacity_mw=60.0, ac_capacity_mw=40.0
    )

    np.testing.assert_allclose(power, [6.0, 30.0, 40.0])


def test_plant_power_applies_the_orientation_and_the_clip() -> None:
    sky = _sky(zenith=30.0, sun_azimuth=180.0)
    plant = PlantParameters(
        tilt_deg=30.0, azimuth_deg=180.0, dc_capacity_mw=100.0, ac_capacity_mw=50.0
    )

    assert plant_power_mw(sky=sky, parameters=plant)[0] == 50.0
    assert plant.dc_ac_ratio == 2.0


def test_daylight_needs_the_sun_more_than_five_degrees_up() -> None:
    sky = SunAndSky(
        half_hour_end_time=np.zeros(3, dtype="datetime64[us]"),
        ghi_w_m2=np.zeros(3),
        dni_w_m2=np.zeros(3),
        dhi_w_m2=np.zeros(3),
        zenith_deg=np.array([80.0, 86.0, 95.0]),
        sun_azimuth_deg=np.zeros(3),
    )

    assert daylight(sky=sky).tolist() == [True, False, False]
