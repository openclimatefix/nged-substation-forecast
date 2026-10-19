"""Tests of the per-row observation operators of the differentiable-physics series."""

import pytest

torch = pytest.importorskip("torch")

from studies.dp_operator import (  # noqa: E402
    ApparentPowerParameters,
    apparent_power,
    observe,
    power_factor_from_tan_phi,
    real_power,
    tan_phi_from_power_factor,
)


def _parameters(*, tan_phi: float, q0: float = 0.0, n: int = 1) -> ApparentPowerParameters:
    return ApparentPowerParameters(
        tan_phi=torch.full((n,), tan_phi, dtype=torch.float64),
        q0=torch.full((n,), q0, dtype=torch.float64),
    )


def test_identity_returns_real_power_exactly():
    real = torch.tensor([-3.0, 0.0, 2.5], dtype=torch.float64)
    reading = observe(kind="identity", real=real, demand=real + 1.0)
    assert torch.equal(reading, real)


def test_apparent_with_no_reactive_power_is_the_absolute_value_even_when_real_power_is_negative():
    real = torch.tensor([-3.0, 0.0, 2.5], dtype=torch.float64)
    parameters = ApparentPowerParameters(
        tan_phi=torch.zeros(3, dtype=torch.float64), q0=torch.zeros(3, dtype=torch.float64)
    )
    reading = apparent_power(
        real=real, demand=torch.tensor([1.0, 1.0, 1.0]).double(), parameters=parameters
    )
    assert torch.allclose(reading, real.abs(), atol=parameters.epsilon**0.5)


def test_apparent_matches_hand_worked_values_at_power_factor_0_97():
    tan_phi = tan_phi_from_power_factor(power_factor=0.97)
    assert tan_phi == pytest.approx(0.25062, abs=1e-4)
    demand = torch.tensor([4.0], dtype=torch.float64)
    real = torch.tensor([3.0], dtype=torch.float64)
    # Q = 0.25062 * 4 = 1.00248, so |S| = sqrt(9 + 1.00497) = 3.16307.
    reading = apparent_power(real=real, demand=demand, parameters=_parameters(tan_phi=tan_phi))
    assert float(reading) == pytest.approx(3.16307, abs=1e-4)


def test_apparent_gradient_is_finite_at_zero_real_and_reactive_power():
    real = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    demand = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    parameters = ApparentPowerParameters(
        tan_phi=torch.zeros(1, dtype=torch.float64), q0=torch.zeros(1, dtype=torch.float64)
    )
    reading = apparent_power(real=real, demand=demand, parameters=parameters)
    reading.sum().backward()
    assert real.grad is not None
    assert torch.isfinite(real.grad).all()
    assert torch.autograd.gradcheck(
        lambda r, d: apparent_power(real=r, demand=d, parameters=parameters),
        (real.detach().clone().requires_grad_(), demand.detach().clone().requires_grad_()),
    )


def test_dark_rows_with_no_reactive_power_read_demand():
    demand = torch.tensor([2.0, 5.0], dtype=torch.float64)
    solar = torch.tensor([1.0, 1.0], dtype=torch.float64)
    gate = torch.tensor([False, False])
    real = real_power(demand=demand, solar=solar, solar_gate=gate)
    reading = observe(
        kind="apparent",
        real=real,
        demand=demand,
        parameters=ApparentPowerParameters(
            tan_phi=torch.zeros(2, dtype=torch.float64), q0=torch.zeros(2, dtype=torch.float64)
        ),
    )
    assert torch.allclose(reading, demand, atol=1e-3)


def test_real_power_applies_the_solar_term_only_where_gated_on():
    real = real_power(
        demand=torch.tensor([5.0, 5.0]),
        solar=torch.tensor([2.0, 2.0]),
        solar_gate=torch.tensor([True, False]),
    )
    assert real.tolist() == [3.0, 5.0]


def test_a_fit_of_tan_phi_alone_recovers_the_power_factor_of_synthetic_apparent_readings():
    generator = torch.Generator().manual_seed(3)
    demand = 2.0 + 3.0 * torch.rand(2000, generator=generator, dtype=torch.float64)
    solar = 4.0 * torch.rand(2000, generator=generator, dtype=torch.float64)
    real = real_power(demand=demand, solar=solar)
    truth = tan_phi_from_power_factor(power_factor=0.97)
    reading = apparent_power(
        real=real, demand=demand, parameters=_parameters(tan_phi=truth, n=2000)
    )
    tan_phi = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.Adam([tan_phi], lr=0.02)
    for _ in range(400):
        optimiser.zero_grad()
        parameters = ApparentPowerParameters(
            tan_phi=tan_phi.expand(2000), q0=torch.zeros(2000, dtype=torch.float64)
        )
        loss = (
            (apparent_power(real=real, demand=demand, parameters=parameters) - reading) ** 2
        ).mean()
        loss.backward()
        optimiser.step()
    assert float(power_factor_from_tan_phi(tan_phi=tan_phi.detach())) == pytest.approx(
        0.97, abs=0.005
    )


def test_asking_for_apparent_power_without_parameters_raises():
    with pytest.raises(ValueError, match="tan_phi"):
        observe(kind="apparent", real=torch.ones(1), demand=torch.ones(1))


def test_sum_of_nodes_is_a_typed_stub():
    with pytest.raises(NotImplementedError, match="sum_of_nodes"):
        observe(kind="sum_of_nodes", real=torch.ones(1), demand=torch.ones(1))
