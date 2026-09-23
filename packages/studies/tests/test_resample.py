import numpy as np
from studies.resample import (
    clear_sky_index_resample,
    hold_flat_outside_daylight,
    interpolate_linear,
    interpolate_pchip,
    wind_components,
    wind_polar,
)

STEPS = np.array([0.0, 3.0, 6.0, 12.0])


def test_linear_interpolates_between_steps_and_holds_the_ends():
    values = np.array([[0.0, 3.0, 9.0, 21.0]])

    result = interpolate_linear(
        values=values, x=STEPS, targets=np.array([-1.0, 1.0, 3.0, 9.0, 13.0])
    )

    np.testing.assert_allclose(result, [[0.0, 1.0, 3.0, 15.0, 21.0]])


def test_linear_treats_each_row_on_its_own():
    values = np.array([[0.0, 3.0, 6.0, 12.0], [10.0, 10.0, 10.0, 10.0]])

    result = interpolate_linear(values=values, x=STEPS, targets=np.array([4.5]))

    np.testing.assert_allclose(result, [[4.5], [10.0]])


def test_pchip_passes_through_every_step_and_does_not_overshoot():
    values = np.array([[0.0, 10.0, 10.0, 0.0]])
    targets = np.linspace(0.0, 12.0, 49)

    result = interpolate_pchip(values=values, x=STEPS, targets=targets)

    np.testing.assert_allclose(result[0, [0, 12, 24, 48]], values[0], atol=1e-9)
    assert result.max() <= 10.0 + 1e-9
    assert result.min() >= 0.0 - 1e-9


def test_pchip_holds_the_end_values_beyond_the_steps():
    values = np.array([[1.0, 2.0, 4.0, 8.0]])

    result = interpolate_pchip(values=values, x=STEPS, targets=np.array([-2.0, 14.0]))

    np.testing.assert_allclose(result, [[1.0, 8.0]])


def test_a_westerly_wind_blows_towards_the_east():
    u, v = wind_components(speed=np.array([10.0]), direction_deg=np.array([270.0]))

    np.testing.assert_allclose(u, [10.0])
    np.testing.assert_allclose(v, [0.0], atol=1e-12)


def test_a_northerly_wind_blows_towards_the_south():
    u, v = wind_components(speed=np.array([10.0]), direction_deg=np.array([0.0]))

    np.testing.assert_allclose(u, [0.0], atol=1e-12)
    np.testing.assert_allclose(v, [-10.0])


def test_components_round_trip_to_speed_and_direction():
    speed = np.array([3.0, 7.5, 12.0, 0.5])
    direction = np.array([10.0, 95.0, 190.0, 350.0])

    u, v = wind_components(speed=speed, direction_deg=direction)
    back_speed, back_direction = wind_polar(u=u, v=v)

    np.testing.assert_allclose(back_speed, speed)
    np.testing.assert_allclose(back_direction, direction)


def test_interpolating_components_crosses_north_the_short_way():
    u, v = wind_components(speed=np.array([10.0, 10.0]), direction_deg=np.array([350.0, 10.0]))
    x = np.array([0.0, 2.0])
    halfway = np.array([1.0])

    _, direction = wind_polar(
        u=interpolate_linear(values=u[None, :], x=x, targets=halfway),
        v=interpolate_linear(values=v[None, :], x=x, targets=halfway),
    )

    np.testing.assert_allclose(direction % 360.0, [[0.0]], atol=1e-9)


def test_a_dawn_step_takes_the_morning_index_and_a_dusk_step_the_evening_index():
    index = np.array([[np.nan, np.nan, 0.5, 0.7, np.nan, np.nan]])
    morning = np.array([True, True, True, False, False, False])

    filled = hold_flat_outside_daylight(index=index, morning=morning)

    np.testing.assert_allclose(filled, [[0.5, 0.5, 0.5, 0.7, 0.7, 0.7]])


def test_a_step_with_no_daylight_on_its_side_takes_the_other_side():
    index = np.array([[0.4, np.nan, np.nan]])
    morning = np.array([True, True, True])

    filled = hold_flat_outside_daylight(index=index, morning=morning)

    np.testing.assert_allclose(filled, [[0.4, 0.4, 0.4]])


def _clear_sky_day() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a symmetric clear-sky day: hourly means, and their 3-hour step means and midpoints."""
    hours = np.arange(24)
    hourly = np.clip(800.0 * np.sin(np.pi * (hours + 0.5 - 6.0) / 12.0), 0.0, None)
    steps = hourly.reshape(8, 3).mean(axis=1)
    step_midpoints = np.arange(8) * 3.0 + 1.5
    target_midpoints = hours + 0.5
    return hourly, steps, step_midpoints, target_midpoints


def test_a_constant_clear_sky_index_comes_back_on_every_hour():
    hourly, steps, step_midpoints, target_midpoints = _clear_sky_day()

    result = clear_sky_index_resample(
        values=0.6 * steps[None, :],
        step_clear_sky=steps[None, :],
        step_midpoints=step_midpoints,
        morning=step_midpoints < 12.0,
        target_clear_sky=hourly[None, :],
        target_midpoints=target_midpoints,
    )

    np.testing.assert_allclose(result, 0.6 * hourly[None, :])


def test_the_index_is_anchored_at_each_step_midpoint():
    _, steps, step_midpoints, _ = _clear_sky_day()
    index = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0])

    result = clear_sky_index_resample(
        values=(index * steps)[None, :],
        step_clear_sky=steps[None, :],
        step_midpoints=step_midpoints,
        morning=step_midpoints < 12.0,
        target_clear_sky=np.full((1, 3), 100.0),
        target_midpoints=np.array([10.5, 12.0, 13.5]),
    )

    np.testing.assert_allclose(result, [[50.0, 75.0, 100.0]])


def test_a_step_below_the_daylight_floor_holds_the_index_flat():
    hourly, steps, step_midpoints, target_midpoints = _clear_sky_day()
    values = 0.8 * steps
    # A dawn step read far brighter than the sky allows: below the floor, it must not anchor.
    values[2] = 5.0 * steps[2]

    result = clear_sky_index_resample(
        values=values[None, :],
        step_clear_sky=steps[None, :],
        step_midpoints=step_midpoints,
        morning=step_midpoints < 12.0,
        target_clear_sky=hourly[None, :],
        target_midpoints=target_midpoints,
        daylight_floor_w_m2=float(steps[2]) + 1.0,
    )

    np.testing.assert_allclose(result, 0.8 * hourly[None, :])


def test_a_dark_target_hour_gets_zero_and_a_row_without_daylight_gets_nan():
    _, steps, step_midpoints, _ = _clear_sky_day()

    result = clear_sky_index_resample(
        values=np.vstack([0.5 * steps, 0.5 * steps]),
        step_clear_sky=np.vstack([steps, np.zeros_like(steps)]),
        step_midpoints=step_midpoints,
        morning=step_midpoints < 12.0,
        target_clear_sky=np.array([[0.0, 100.0], [0.0, 100.0]]),
        target_midpoints=np.array([0.5, 12.5]),
    )

    assert result[0, 0] == 0.0
    assert result[0, 1] == 50.0
    assert result[1, 0] == 0.0
    assert np.isnan(result[1, 1])
