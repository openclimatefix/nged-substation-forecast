import numpy as np
import pytest
from studies.battery_joint_lp import JointFit, fit_joint_solar_battery

_DAYS = 6
_STEPS = 48
_ETA = 0.92
_SOLAR_WEIGHTS = np.array([12.0, 5.0])
_CHARGE_MW = 10.0
_ENERGY_MWH = 18.4 + 0.1
"""Charging at 10 MW for 4 half-hours stores 18.4 MWh; the extra 0.1 is slack."""


def _basis() -> np.ndarray:
    """Two daylight bumps, one peaking in the morning and one in the afternoon."""
    hour = np.tile(np.arange(_STEPS) / 2.0, _DAYS)
    morning = np.where((hour >= 6) & (hour < 14), np.sin(np.pi * (hour - 6) / 8) ** 2, 0.0)
    afternoon = np.where((hour >= 10) & (hour < 18), np.sin(np.pi * (hour - 10) / 8) ** 2, 0.0)
    return np.column_stack([morning, afternoon])


def _battery_schedule() -> np.ndarray:
    """Charge from 02:00 to 04:00 and discharge from 19:00 to 21:00 each day, ending at 0 MWh."""
    day = np.zeros(_STEPS)
    day[4:8] = -_CHARGE_MW
    discharge_mw = 4 * _CHARGE_MW * _ETA * _ETA / 4
    day[38:42] = discharge_mw
    return np.tile(day, _DAYS)


def _fit(*, aggregate: np.ndarray, **overrides: float | np.ndarray | None) -> JointFit:
    arguments = {
        "aggregate_mw": aggregate,
        "solar_basis": _basis(),
        "power_mw": _CHARGE_MW,
        "energy_mwh": _ENERGY_MWH,
        "one_way_efficiency": _ETA,
    } | overrides
    return fit_joint_solar_battery(**arguments)  # ty: ignore[invalid-argument-type]


def test_a_noiseless_solar_plus_battery_series_gives_back_both_parts() -> None:
    truth = _battery_schedule()
    aggregate = _basis() @ _SOLAR_WEIGHTS + truth

    fit = _fit(aggregate=aggregate)

    np.testing.assert_allclose(fit.solar_weights, _SOLAR_WEIGHTS, atol=1e-5)
    np.testing.assert_allclose(fit.battery_mw, truth, atol=1e-5)
    assert np.abs(fit.residual_mw).max() < 1e-5


def test_the_state_of_charge_stays_inside_its_bounds_and_the_power_inside_its_limit() -> None:
    rng = np.random.default_rng(1)
    aggregate = rng.normal(0.0, 8.0, _DAYS * _STEPS)

    fit = _fit(aggregate=aggregate)

    assert fit.soc_mwh.min() >= -1e-6
    assert fit.soc_mwh.max() <= _ENERGY_MWH + 1e-6
    assert np.abs(fit.battery_mw).max() <= _CHARGE_MW + 1e-6
    assert np.abs(fit.battery_mw).max() > 1.0


def test_the_state_of_charge_follows_the_battery_output_through_the_first_day() -> None:
    truth = _battery_schedule()
    fit = _fit(aggregate=_basis() @ _SOLAR_WEIGHTS + truth)

    stored = np.where(truth < 0, -truth * _ETA, -truth / _ETA) * 0.5
    expected = np.cumsum(stored)
    expected = expected - expected[0] + fit.soc_mwh[0] - stored[0]

    np.testing.assert_allclose(fit.soc_mwh[:48], expected[:48], atol=1e-4)


def test_a_battery_with_no_energy_capacity_is_idle() -> None:
    aggregate = _basis() @ _SOLAR_WEIGHTS + _battery_schedule()

    fit = _fit(aggregate=aggregate, energy_mwh=0.0)

    assert np.abs(fit.battery_mw).max() < 1e-6
    assert np.abs(fit.soc_mwh).max() < 1e-6
    assert np.abs(fit.residual_mw).max() > 5.0


def test_a_battery_with_no_power_is_idle() -> None:
    fit = _fit(aggregate=_basis() @ _SOLAR_WEIGHTS + _battery_schedule(), power_mw=0.0)

    assert np.abs(fit.battery_mw).max() < 1e-9


def test_a_free_starting_state_of_charge_per_window_lets_a_window_begin_by_discharging() -> None:
    # Days 1 and 4 each discharge 16 MWh with no charging between them, which one starting state
    # of charge cannot pay for but two can.
    aggregate = np.zeros(_DAYS * _STEPS)
    aggregate[:4] = 8.0
    aggregate[3 * _STEPS : 3 * _STEPS + 4] = 8.0

    no_sun = np.zeros((_DAYS * _STEPS, 2))
    one_window = _fit(aggregate=aggregate, solar_basis=no_sun, window_half_hours=None)
    windows = _fit(aggregate=aggregate, solar_basis=no_sun, window_half_hours=3 * _STEPS)

    assert len(windows.window_starts) == 2
    assert np.abs(windows.residual_mw).max() < 1e-5
    assert np.abs(one_window.residual_mw).max() > 1.0


def test_a_smoothness_penalty_reduces_the_battery_output_variation() -> None:
    rng = np.random.default_rng(2)
    aggregate = rng.normal(0.0, 8.0, _DAYS * _STEPS)

    plain = _fit(aggregate=aggregate)
    smooth = _fit(aggregate=aggregate, smoothness_penalty=0.5)

    assert np.abs(np.diff(smooth.battery_mw)).sum() < np.abs(np.diff(plain.battery_mw)).sum()


def test_solar_weights_are_not_negative() -> None:
    aggregate = -(_basis() @ _SOLAR_WEIGHTS)

    fit = _fit(aggregate=aggregate, power_mw=0.0)

    assert fit.solar_weights.min() >= 0.0
    assert fit.solar_weights.sum() < 1e-6


def test_a_solar_weight_penalty_above_the_residual_saving_leaves_no_solar() -> None:
    aggregate = _basis() @ _SOLAR_WEIGHTS

    free = _fit(aggregate=aggregate, power_mw=0.0)
    penalised = _fit(aggregate=aggregate, power_mw=0.0, solar_weight_penalty=1e4)

    assert free.solar_weights.sum() == pytest.approx(_SOLAR_WEIGHTS.sum(), abs=1e-5)
    assert penalised.solar_weights.sum() < 1e-6


def test_half_hours_without_an_aggregate_add_no_residual_and_the_battery_idles() -> None:
    aggregate = _basis() @ _SOLAR_WEIGHTS + _battery_schedule()
    aggregate[10:20] = np.nan

    fit = _fit(aggregate=aggregate)

    assert np.isnan(fit.residual_mw[10:20]).all()
    assert np.abs(fit.battery_mw[10:20]).max() < 1e-6
    np.testing.assert_allclose(fit.solar_weights, _SOLAR_WEIGHTS, atol=1e-5)


def test_a_mismatched_aggregate_length_raises() -> None:
    with pytest.raises(ValueError, match="basis has"):
        _fit(aggregate=np.zeros(5))


def test_an_efficiency_outside_zero_to_one_raises() -> None:
    with pytest.raises(ValueError, match="one_way_efficiency"):
        _fit(aggregate=np.zeros(_DAYS * _STEPS), one_way_efficiency=1.2)


def test_a_negative_limit_raises() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        _fit(aggregate=np.zeros(_DAYS * _STEPS), energy_mwh=-1.0)


def test_signed_columns_get_free_signed_coefficients_alongside_solar_and_battery() -> None:
    rng = np.random.default_rng(3)
    columns = rng.normal(size=(_DAYS * _STEPS, 2))
    coefficients = np.array([3.0, -2.0])
    aggregate = _basis() @ _SOLAR_WEIGHTS + _battery_schedule() + columns @ coefficients

    fit = _fit(aggregate=aggregate, signed_columns=columns, signed_column_penalty=1e-6)

    np.testing.assert_allclose(fit.signed_coefficients, coefficients, atol=1e-3)
    np.testing.assert_allclose(fit.signed_mw, columns @ coefficients, atol=1e-2)
    np.testing.assert_allclose(fit.solar_weights, _SOLAR_WEIGHTS, atol=1e-2)
    assert np.abs(fit.residual_mw).max() < 1e-2


def test_a_calendar_baseline_stops_the_solar_weights_absorbing_the_daily_shape() -> None:
    hour = np.tile(np.arange(_STEPS) / 2.0, _DAYS)
    slot_indicator = np.zeros((_DAYS * _STEPS, _STEPS))
    slot_indicator[np.arange(_DAYS * _STEPS), np.tile(np.arange(_STEPS), _DAYS)] = 1.0
    daily_shape = 4.0 * np.cos(2 * np.pi * hour / 24.0)
    cloud = np.tile(np.linspace(0.2, 1.0, _DAYS).repeat(_STEPS), 1)
    aggregate = (_basis() @ _SOLAR_WEIGHTS) * cloud + daily_shape
    basis = _basis() * cloud[:, None]

    without = _fit(aggregate=aggregate, solar_basis=basis, power_mw=0.0)
    with_baseline = _fit(
        aggregate=aggregate, solar_basis=basis, power_mw=0.0, signed_columns=slot_indicator
    )

    np.testing.assert_allclose(with_baseline.solar_weights, _SOLAR_WEIGHTS, atol=0.05)
    assert np.abs(without.solar_weights - _SOLAR_WEIGHTS).max() > 0.5


def test_signed_columns_with_the_wrong_number_of_rows_raise() -> None:
    with pytest.raises(ValueError, match="signed_columns has"):
        _fit(aggregate=np.zeros(_DAYS * _STEPS), signed_columns=np.zeros((5, 1)))


def test_without_signed_columns_the_contribution_is_zero() -> None:
    fit = _fit(aggregate=_basis() @ _SOLAR_WEIGHTS)

    assert fit.signed_coefficients.shape == (0,)
    assert np.abs(fit.signed_mw).max() == 0.0


def test_windows_have_near_equal_lengths_rounded_to_the_nearest_whole_number_of_windows() -> None:
    # 288 half-hours at a target of 100 is 2.88 windows, which rounds to 3 of 96.
    fit = _fit(aggregate=np.zeros(_DAYS * _STEPS), window_half_hours=100)

    assert fit.window_starts.tolist() == [0, 96, 192]


def test_a_window_starts_at_its_own_state_of_charge_not_the_previous_windows_end() -> None:
    # Window 1 ends nearly full. Window 2 charges again at once, which only fits if its starting
    # state of charge is free to be empty.
    aggregate = np.zeros(_DAYS * _STEPS)
    aggregate[2 * _STEPS - 4 : 2 * _STEPS] = -_CHARGE_MW
    aggregate[2 * _STEPS : 2 * _STEPS + 4] = -_CHARGE_MW
    no_sun = np.zeros((_DAYS * _STEPS, 2))

    fit = _fit(aggregate=aggregate, solar_basis=no_sun, window_half_hours=2 * _STEPS)

    assert len(fit.window_starts) == 3
    assert np.abs(fit.residual_mw).max() < 1e-5


def test_the_starting_state_of_charge_of_a_window_cannot_exceed_the_energy_capacity() -> None:
    # Discharging 10 MW for 4 half-hours needs 21.7 MWh at the start, above the 18.5 MWh
    # capacity. The first half-hour alone would fit if the start were allowed 5.4 MWh over.
    aggregate = np.zeros(_DAYS * _STEPS)
    aggregate[:4] = _CHARGE_MW

    fit = _fit(
        aggregate=aggregate,
        solar_basis=np.zeros((_DAYS * _STEPS, 2)),
        window_half_hours=_STEPS,
    )

    assert np.abs(fit.residual_mw).max() > 1.0


def test_a_smoothness_penalty_counts_a_plateau_as_two_steps_not_one_per_half_hour() -> None:
    aggregate = np.zeros(_DAYS * _STEPS)
    aggregate[100:104] = 5.0
    no_sun = np.zeros((_DAYS * _STEPS, 2))

    fit = _fit(aggregate=aggregate, solar_basis=no_sun, smoothness_penalty=1.0)

    np.testing.assert_allclose(fit.battery_mw, aggregate, atol=1e-5)


@pytest.mark.parametrize("penalty", [0.1, 0.5])
def test_a_smoothness_penalty_removes_a_one_half_hour_pulse_only_once_it_costs_more_than_it_saves(
    penalty: float,
) -> None:
    # The pulse saves 5 MW of residual for 0.5 MW of throughput cost. Following it costs two steps
    # of 5 MW each, so a penalty of 0.1 follows it (1.0 < 4.5) and 0.5 does not (5.0 > 4.5).
    aggregate = np.zeros(_DAYS * _STEPS)
    aggregate[100] = 5.0
    no_sun = np.zeros((_DAYS * _STEPS, 2))

    fit = _fit(aggregate=aggregate, solar_basis=no_sun, smoothness_penalty=penalty)

    if penalty < 0.25:
        np.testing.assert_allclose(fit.battery_mw[100], 5.0, atol=1e-5)
    else:
        assert np.abs(fit.battery_mw).max() < 1e-5


def test_nan_in_the_solar_basis_counts_as_zero() -> None:
    basis = _basis()
    basis[5, 0] = np.nan
    aggregate = np.nan_to_num(basis) @ _SOLAR_WEIGHTS

    fit = _fit(aggregate=aggregate, solar_basis=basis, power_mw=0.0)

    assert np.isfinite(fit.solar_mw).all()
    np.testing.assert_allclose(fit.solar_weights, _SOLAR_WEIGHTS, atol=1e-5)


@pytest.mark.parametrize("efficiency", [0.0, 1.0001])
def test_an_efficiency_of_zero_or_above_one_raises(efficiency: float) -> None:
    with pytest.raises(ValueError, match="one_way_efficiency"):
        _fit(aggregate=np.zeros(_DAYS * _STEPS), one_way_efficiency=efficiency)


def test_an_efficiency_of_exactly_one_is_accepted() -> None:
    fit = _fit(aggregate=np.zeros(_DAYS * _STEPS), one_way_efficiency=1.0)

    assert np.abs(fit.battery_mw).max() < 1e-6


def test_a_negative_power_limit_raises() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        _fit(aggregate=np.zeros(_DAYS * _STEPS), power_mw=-1.0)


def test_a_large_signed_column_penalty_zeroes_coefficients_of_either_sign() -> None:
    rng = np.random.default_rng(4)
    columns = rng.normal(size=(_DAYS * _STEPS, 2))
    aggregate = columns @ np.array([2.0, -2.0])

    fit = _fit(
        aggregate=aggregate,
        solar_basis=np.zeros((_DAYS * _STEPS, 2)),
        power_mw=0.0,
        signed_columns=columns,
        signed_column_penalty=1e3,
    )

    assert np.abs(fit.signed_coefficients).max() < 1e-6


def test_the_method_is_passed_to_the_solver() -> None:
    with pytest.raises(ValueError, match="method"):
        _fit(aggregate=np.zeros(_DAYS * _STEPS), method="not-a-method")  # ty: ignore[invalid-argument-type]


def test_the_default_efficiency_is_ninety_two_percent() -> None:
    rng = np.random.default_rng(5)
    aggregate = rng.normal(0.0, 8.0, _DAYS * _STEPS)
    arguments = {"aggregate_mw": aggregate, "solar_basis": _basis(), "power_mw": _CHARGE_MW}

    default = fit_joint_solar_battery(energy_mwh=_ENERGY_MWH, **arguments)  # ty: ignore[invalid-argument-type]
    explicit = fit_joint_solar_battery(
        energy_mwh=_ENERGY_MWH,
        one_way_efficiency=0.92,
        **arguments,  # ty: ignore[invalid-argument-type]
    )

    np.testing.assert_allclose(default.battery_mw, explicit.battery_mw, atol=1e-9)
