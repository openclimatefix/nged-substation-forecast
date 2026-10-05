import numpy as np
import pytest
from studies.physics_model import GROUND_ALBEDO, Geometry, plane_of_array, power_mw


def _geometry(*, beam: float | None, diffuse: float | None, cos_zenith: float = 1.0) -> Geometry:
    one = np.array([1.0])
    return Geometry(
        cos_zenith=one * cos_zenith,
        sin_zenith=one * np.sqrt(1.0 - cos_zenith**2),
        solar_azimuth_rad=one * 0.0,
        global_horizontal=one * 600.0,
        beam_horizontal=None if beam is None else one * beam,
        diffuse_horizontal=None if diffuse is None else one * diffuse,
        air_temperature_c=one * 25.0,
    )


def test_an_arm_with_no_split_gets_global_irradiance_unchanged():
    geometry = _geometry(beam=None, diffuse=None)

    result = plane_of_array(geometry=geometry, tilt_rad=0.5, azimuth_rad=3.0)

    np.testing.assert_array_equal(result, geometry.global_horizontal)


def test_a_flat_panel_with_the_sun_overhead_receives_beam_plus_diffuse():
    geometry = _geometry(beam=400.0, diffuse=200.0)

    result = plane_of_array(geometry=geometry, tilt_rad=0.0, azimuth_rad=0.0)

    np.testing.assert_allclose(result, [600.0])


def test_a_vertical_panel_gets_half_the_diffuse_and_ground_reflection_and_no_overhead_beam():
    geometry = _geometry(beam=400.0, diffuse=200.0)

    result = plane_of_array(geometry=geometry, tilt_rad=np.pi / 2, azimuth_rad=0.0)

    np.testing.assert_allclose(result, [200.0 * 0.5 + 600.0 * GROUND_ALBEDO * 0.5], atol=1e-9)


def test_power_is_clipped_at_the_inverter_ceiling_and_never_negative():
    geometry = _geometry(beam=400.0, diffuse=200.0)

    kwargs = {"tilt_rad": 0.0, "azimuth_rad": 0.0, "temperature_coefficient": 0.0}
    clipped = power_mw(geometry=geometry, capacity_mw=10.0, clip_mw=3.0, **kwargs)
    negative = power_mw(geometry=geometry, capacity_mw=-10.0, clip_mw=3.0, **kwargs)

    np.testing.assert_allclose(clipped, [3.0])
    np.testing.assert_array_equal(negative, [0.0])


def test_a_hotter_cell_lowers_power_when_the_temperature_coefficient_is_negative():
    geometry = _geometry(beam=400.0, diffuse=200.0)
    kwargs = {"tilt_rad": 0.0, "azimuth_rad": 0.0, "capacity_mw": 1.0, "clip_mw": 10.0}

    neutral = power_mw(geometry=geometry, temperature_coefficient=0.0, **kwargs)
    derated = power_mw(geometry=geometry, temperature_coefficient=-0.004, **kwargs)

    assert derated[0] == pytest.approx(
        neutral[0] * (1.0 - 0.004 * (25.0 + 600.0 * 25.0 / 800.0 - 25.0))
    )
