"""Tests of the differentiable state-space battery estimator, all on the CPU in float64."""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.battery_dispatch import lp_schedule  # noqa: E402
from studies.battery_recurrence import recurrence  # noqa: E402
from studies.battery_state_space import (  # noqa: E402
    Estimator,
    Layout,
    Priors,
    Problems,
    Sharpness,
    _node_weights,
    integrated_autocorrelation_time,
    make_problems,
    make_signals,
    natural_draws,
    natural_point,
    simulate,
)
from studies.template_posterior import tempering_from_residual  # noqa: E402

SLOTS: Final[int] = 48
SHARP: Final[Sharpness] = Sharpness(smoothing=1e-6)
DEVICE: Final = torch.device("cpu")
DURATION_NODES: Final = np.geomspace(0.5, 6.0, 9)
EFFICIENCY_NODES: Final = np.array([0.7, 0.8, 0.9, 0.97])
LAYOUT: Final = Layout(unit_class=(0, 1), n_classes=2, n_stack=1)
"""Unit 0 is a price taker (class 0) and unit 1 follows a fixed window (class 1)."""


def _prices(*, days: int, seed: int) -> np.ndarray:
    """Daily prices with a morning and an evening peak and a different level every day."""
    rng = np.random.default_rng(seed)
    slot = np.arange(SLOTS)
    shape = 40 + 25 * np.exp(-((slot - 36) ** 2) / 20) + 15 * np.exp(-((slot - 16) ** 2) / 30)
    return np.concatenate(
        [shape * rng.uniform(0.8, 1.2) + rng.normal(0, 4, SLOTS) for _ in range(days)]
    )


def _stack(*, days: int) -> np.ndarray:
    """The linear-programme schedules of the price taker at every node, shape (nodes, T)."""
    prices = _prices(days=days, seed=1)
    columns = [
        lp_schedule(
            prices=prices,
            energy_hours=duration / 0.9,
            eta_one_way=float(np.sqrt(efficiency)),
            cycles_per_day_cap=cap,
        )
        for duration in DURATION_NODES
        for efficiency in EFFICIENCY_NODES
        for cap in (1.0, 2.0)
    ]
    return np.stack(columns)


def _window(
    *, days: int, charge: tuple[int, int], discharge: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Charge and discharge indicators repeating daily, in half-hour slot ranges."""
    one_day_charge = np.zeros(SLOTS)
    one_day_discharge = np.zeros(SLOTS)
    one_day_charge[charge[0] : charge[1]] = 1.0
    one_day_discharge[discharge[0] : discharge[1]] = 1.0
    return np.tile(one_day_charge, days), np.tile(one_day_discharge, days)


def _signals(*, days: int) -> dict[str, np.ndarray]:
    """The price taker's stack and one fixed-window unit."""
    charge, discharge = _window(days=days, charge=(2, 12), discharge=(34, 40))
    return {
        "stacks": _stack(days=days)[None],
        "duration_nodes": DURATION_NODES,
        "efficiency_nodes": EFFICIENCY_NODES,
        "fixed_charge": charge[None],
        "fixed_discharge": discharge[None],
    }


def _theta(
    *, power: tuple[float, float], duration: float, efficiency: float, cap_weight: float = 0.5
) -> np.ndarray:
    """Parameters of the two-unit layout in the unconstrained space."""

    def logit(p: float) -> float:
        return float(np.log(p / (1 - p)))

    return np.array(
        [
            np.log(power[0]),
            np.log(power[1]),
            np.log(duration),
            np.log(duration),
            logit(efficiency),
            logit(efficiency),
            logit(cap_weight),
        ]
    )


def test_the_state_of_charge_stays_within_the_usable_energy_over_a_week_of_random_policies() -> (
    None
):
    signals = make_signals(**_signals(days=7), device=DEVICE, dtype=torch.float64)
    rng = np.random.default_rng(0)
    theta = torch.as_tensor(
        np.stack(
            [
                _theta(
                    power=(rng.uniform(0.5, 4), rng.uniform(0.5, 4)),
                    duration=rng.uniform(0.6, 5),
                    efficiency=rng.uniform(0.72, 0.96),
                    cap_weight=rng.uniform(0.05, 0.95),
                )
                for _ in range(16)
            ]
        )
    )
    _, state, _ = simulate(
        theta=theta, layout=LAYOUT, signals=signals, sharpness=SHARP, return_state=True
    )
    assert float(state.min()) >= -1e-3
    assert float(state.max()) <= 1.0 + 1e-3
    assert float(state.max()) > 0.5, "the policy never charged, so the bound was not exercised"


def test_a_full_charge_and_discharge_returns_the_charged_energy_times_the_efficiency() -> None:
    days = 2
    signals = make_signals(**_signals(days=days), device=DEVICE, dtype=torch.float64)
    efficiency = 0.81
    theta = torch.as_tensor(_theta(power=(1e-9, 2.0), duration=1.5, efficiency=efficiency)[None])
    export = simulate(theta=theta, layout=LAYOUT, signals=signals, sharpness=SHARP)[0].numpy()
    charged = -export[export < 0].sum() * 0.5
    discharged = export[export > 0].sum() * 0.5
    assert charged == pytest.approx(2.0 * 1.5 / np.sqrt(efficiency) * days, rel=1e-3)
    assert discharged / charged == pytest.approx(efficiency, rel=2e-3)


def test_the_node_weights_are_one_at_a_node_and_blend_two_nodes_between_them() -> None:
    signals = make_signals(**_signals(days=1), device=DEVICE, dtype=torch.float64)
    at_node = _node_weights(
        duration=torch.tensor([DURATION_NODES[3]], dtype=torch.float64),
        efficiency=torch.tensor([EFFICIENCY_NODES[2]], dtype=torch.float64),
        cap_weight=torch.tensor([0.0], dtype=torch.float64),
        signals=signals,
    )[0]
    index = (3 * len(EFFICIENCY_NODES) + 2) * 2
    assert float(at_node[index]) == pytest.approx(1.0)
    assert float(at_node.sum()) == pytest.approx(1.0)
    midpoint = float(np.sqrt(DURATION_NODES[3] * DURATION_NODES[4]))
    between = _node_weights(
        duration=torch.tensor([midpoint], dtype=torch.float64),
        efficiency=torch.tensor([EFFICIENCY_NODES[2]], dtype=torch.float64),
        cap_weight=torch.tensor([0.0], dtype=torch.float64),
        signals=signals,
    )[0]
    assert float(between[index]) == pytest.approx(0.5)
    assert float(between[index + len(EFFICIENCY_NODES) * 2]) == pytest.approx(0.5)


def test_interpolating_between_duration_nodes_beats_the_nearest_node() -> None:
    days = 20
    prices = _prices(days=days, seed=1)
    signals = make_signals(**_signals(days=days), device=DEVICE, dtype=torch.float64)
    duration = float(np.sqrt(DURATION_NODES[4] * DURATION_NODES[5]))
    efficiency = 0.86
    exact = lp_schedule(
        prices=prices,
        energy_hours=duration / 0.9,
        eta_one_way=float(np.sqrt(efficiency)),
        cycles_per_day_cap=1.0,
    )
    theta = torch.as_tensor(
        _theta(power=(1.0, 1e-9), duration=duration, efficiency=efficiency, cap_weight=1e-9)[None]
    )
    interpolated = simulate(theta=theta, layout=LAYOUT, signals=signals, sharpness=SHARP)[0].numpy()
    nearest = [
        simulate(
            theta=torch.as_tensor(
                _theta(power=(1.0, 1e-9), duration=float(d), efficiency=e, cap_weight=1e-9)[None]
            ),
            layout=LAYOUT,
            signals=signals,
            sharpness=SHARP,
        )[0].numpy()
        for d in DURATION_NODES[4:6]
        for e in (0.8, 0.9)
    ]
    error = np.sqrt(np.mean((interpolated - exact) ** 2))
    assert error < 0.7 * min(np.sqrt(np.mean((n - exact) ** 2)) for n in nearest)


def test_the_gradient_of_the_export_matches_central_finite_differences() -> None:
    signals = make_signals(**_signals(days=2), device=DEVICE, dtype=torch.float64)
    sharpness = Sharpness(smoothing=1e-3)
    theta = torch.as_tensor(
        _theta(power=(1.7, 2.3), duration=1.37, efficiency=0.853, cap_weight=0.4)[None],
        dtype=torch.float64,
    ).requires_grad_(True)
    weights = torch.linspace(0.5, 1.5, 96, dtype=torch.float64)

    def objective(vector: torch.Tensor) -> torch.Tensor:
        export = simulate(theta=vector, layout=LAYOUT, signals=signals, sharpness=sharpness)
        return (export**2 * weights).sum()

    assert torch.autograd.gradcheck(objective, (theta,), eps=1e-6, atol=1e-5, rtol=1e-5)


def test_the_autocorrelation_time_agrees_with_the_numpy_tempering_function() -> None:
    rng = np.random.default_rng(3)
    n = 6000
    white = rng.normal(size=n)
    correlated = np.zeros(n)
    for t in range(1, n):
        correlated[t] = 0.9 * correlated[t - 1] + rng.normal()
    valid = torch.ones(1, n, dtype=torch.bool)
    times = [
        float(integrated_autocorrelation_time(residual=torch.as_tensor(x[None]), valid=valid))
        for x in (white, correlated)
    ]
    expected = [1.0 / tempering_from_residual(residual=x, rho=0.0) for x in (white, correlated)]
    assert times[0] < 1.5
    assert times[1] > 8
    assert times == pytest.approx(expected, rel=0.01)


def _estimator(*, days: int, durations_sd: float = 0.5, pin_efficiency: bool = False) -> Estimator:
    priors = Priors(
        log_duration_mean=np.log([2.0, 2.0]),
        log_duration_sd=np.array([durations_sd, durations_sd]),
        efficiency_alpha=np.array([1e4, 1e4]) if pin_efficiency else np.array([30.0, 30.0]),
        efficiency_beta=np.array([1.2e3, 1.2e3]) if pin_efficiency else np.array([5.0, 5.0]),
    )
    return Estimator(layout=LAYOUT, signals_numpy=_signals(days=days), priors=priors, device=DEVICE)


def _problems(
    *, estimator: Estimator, theta_true: np.ndarray, days: int, noise: float, memory: float = 0.0
) -> Problems:
    """One group: a calendar-like free part, the battery subtracted, and white noise."""
    n_time = days * SLOTS
    slot = np.tile(np.arange(SLOTS), days)
    free = np.column_stack(
        [
            np.ones(n_time),
            np.sin(2 * np.pi * slot / SLOTS),
            np.cos(2 * np.pi * slot / SLOTS),
            np.sin(4 * np.pi * slot / SLOTS),
            np.cos(4 * np.pi * slot / SLOTS),
        ]
    )
    base = free @ np.array([30.0, 8.0, -5.0, 2.0, 1.0])
    export = simulate(
        theta=torch.as_tensor(theta_true[None]),
        layout=LAYOUT,
        signals=estimator.signals64,
        sharpness=FINAL,
    )[0].numpy()
    rng = np.random.default_rng(5)
    innovations = rng.normal(0, noise, n_time)
    errors = np.zeros(n_time)
    for t in range(n_time):
        errors[t] = innovations[t] + (memory * errors[t - 1] if t else 0.0)
    aggregate = base - export + errors
    valid = np.ones(n_time, dtype=bool)
    return make_problems(
        aggregate=aggregate[None, None, :],
        valid=valid[None],
        free=[free],
        power_scale=np.array([[2.0]]),
    )


FINAL: Final[Sharpness] = Sharpness(smoothing=1e-5)
STARTS: Final = np.array(
    [
        [-0.5, -3.0, np.log(1.5), np.log(2.5), 1.0, 1.0, 0.0],
        [0.5, -3.0, np.log(3.0), np.log(2.0), 2.0, 1.5, 0.0],
    ]
)
STAGES: Final = [(80, 0.08, Sharpness(smoothing=1e-2)), (80, 0.03, FINAL)]


def test_a_noise_free_sum_built_by_the_model_gives_back_its_power_duration_and_efficiency() -> None:
    days = 7
    estimator = _estimator(days=days)
    truth = _theta(power=(3.0, 1e-3), duration=1.7, efficiency=0.86, cap_weight=0.3)
    problems = _problems(estimator=estimator, theta_true=truth, days=days, noise=1e-3)
    fit = estimator.fit(problems=problems, starts=STARTS, stages=STAGES)
    best = fit.best_start[0, 0]
    point = natural_point(theta=fit.theta[0, 0, best], layout=LAYOUT)
    assert point["power"][0] == pytest.approx(3.0, rel=0.02)
    assert point["duration"][0] == pytest.approx(1.7, rel=0.02)
    assert point["efficiency"][0] == pytest.approx(0.86, rel=0.02)
    assert point["energy"][0] == pytest.approx(3.0 * 1.7, rel=0.02)


def test_with_shape_parameters_pinned_the_laplace_interval_matches_least_squares() -> None:
    days = 7
    estimator = _estimator(days=days, durations_sd=1e-4, pin_efficiency=True)
    truth = _theta(power=(3.0, 1e-3), duration=2.0, efficiency=0.88, cap_weight=0.5)
    problems = _problems(estimator=estimator, theta_true=truth, days=days, noise=0.5)
    # A very wide power prior, so that the prior does not add to the interval.
    problems = replace(problems, power_scale=np.array([[500.0]]))
    fit = estimator.fit(problems=problems, starts=STARTS[:1], stages=STAGES)
    assert fit.has_interval[0, 0, 0]
    theta = fit.theta[0, 0, 0]
    # The sd of the power with every other parameter held at the optimum is 1 / sqrt(Hessian[0, 0]).
    precision = np.linalg.inv(fit.covariance[0, 0, 0])
    sd_power = np.exp(theta[0]) / np.sqrt(precision[0, 0])
    # Ordinary least squares: sd = sigma / norm(M t), t the unit's per-megawatt export.
    stack_only = theta.copy()
    stack_only[
        1
    ] = -50.0  # switch the window unit off, so that the export is the stack unit's alone
    export = simulate(
        theta=torch.as_tensor(stack_only[None]),
        layout=LAYOUT,
        signals=estimator.signals64,
        sharpness=FINAL,
    )[0].numpy()
    basis = problems.basis[0]
    template = export / np.exp(theta[0])
    template = template - basis @ (basis.T @ template)
    sigma = np.sqrt(fit.rss[0, 0, 0] / (problems.valid.sum() - problems.rank[0]))
    ols_sd = sigma / np.linalg.norm(template)
    assert sd_power == pytest.approx(ols_sd * fit.tau[0, 0, 0] ** 0.5, rel=0.01)


def test_the_log_bayes_factor_is_large_with_a_battery_in_the_sum_and_small_without() -> None:
    days = 7
    estimator = _estimator(days=days)
    log_bayes_factors = []
    for power in (3.0, 1e-9):
        truth = _theta(power=(power, 1e-9), duration=1.7, efficiency=0.86, cap_weight=0.3)
        problems = _problems(estimator=estimator, theta_true=truth, days=days, noise=0.3)
        fit = estimator.fit(problems=problems, starts=STARTS, stages=STAGES)
        best = fit.best_start[0, 0]
        log_bayes_factors.append(
            float(fit.log_evidence[0, 0, best] - fit.null_log_evidence[0, 0, best])
        )
    with_battery, without_battery = log_bayes_factors
    assert with_battery > 50.0
    assert without_battery < 5.0


def test_the_log_bayes_factor_does_not_change_with_the_unit_of_power() -> None:
    days = 7
    estimator = _estimator(days=days)
    truth = _theta(power=(3.0, 1e-9), duration=1.7, efficiency=0.86, cap_weight=0.3)
    problems = _problems(estimator=estimator, theta_true=truth, days=days, noise=0.2, memory=0.9)
    kilowatts = replace(
        problems,
        aggregate=problems.aggregate * 1000.0,
        power_scale=problems.power_scale * 1000.0,
    )
    log_bayes_factors = []
    for candidate in (problems, kilowatts):
        fit = estimator.fit(problems=candidate, starts=STARTS, stages=STAGES)
        best = fit.best_start[0, 0]
        log_bayes_factors.append(
            float(fit.log_evidence[0, 0, best] - fit.null_log_evidence[0, 0, best])
        )
        # The battery model's residual is more autocorrelated than the null's here, so the two
        # models' autocorrelation times differ, which made the old statistic depend on the unit.
        assert fit.tau[0, 0, best] > 3.0
    assert log_bayes_factors[1] == pytest.approx(log_bayes_factors[0], abs=1e-3)


def test_draws_from_the_laplace_approximation_are_positive_and_centred_on_the_optimum() -> None:
    theta = _theta(power=(3.0, 1.0), duration=2.0, efficiency=0.85)
    covariance = np.eye(LAYOUT.n_parameters) * 1e-4
    draws = natural_draws(
        theta=theta,
        covariance=covariance,
        layout=LAYOUT,
        n_draws=2000,
        rng=np.random.default_rng(0),
    )
    assert (draws["power"] > 0).all()
    assert np.median(draws["power"][:, 0]) == pytest.approx(3.0, rel=0.01)
    assert np.median(draws["energy"][:, 0]) == pytest.approx(6.0, rel=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_gpu_kernel_matches_the_reference_loop_in_value_and_in_every_gradient() -> None:
    generator = torch.Generator().manual_seed(0)
    n_time, n_lanes = 700, 37
    charge = (torch.rand(n_time, n_lanes, generator=generator, dtype=torch.float64) > 0.6).double()
    discharge = torch.rand(n_time, n_lanes, generator=generator, dtype=torch.float64)
    duration = torch.rand(n_lanes, generator=generator, dtype=torch.float64) * 3 + 0.5
    eta = torch.rand(n_lanes, generator=generator, dtype=torch.float64) * 0.2 + 0.8
    weights = torch.randn(n_time, n_lanes, generator=generator, dtype=torch.float64)
    results = []
    for device in ("cpu", "cuda"):
        inputs = [
            t.to(device).clone().requires_grad_(True) for t in (charge, discharge, duration, eta)
        ]
        output, _ = recurrence(
            charge=inputs[0],
            discharge=inputs[1],
            duration=inputs[2],
            eta_one_way=inputs[3],
            smoothing=1e-3,
        )
        gradients = torch.autograd.grad((output * weights.to(device)).sum(), inputs)
        results.append([output.detach().cpu(), *[g.cpu() for g in gradients]])
    for reference, kernel in zip(results[0], results[1], strict=True):
        assert float((reference - kernel).abs().max()) <= 1e-6 * float(reference.abs().max())
