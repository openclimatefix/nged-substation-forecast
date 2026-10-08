"""Tests of the differentiable state-space battery estimator, all on the CPU in float64."""

from dataclasses import replace
from typing import Final

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from studies.battery_recurrence import recurrence  # noqa: E402
from studies.battery_state_space import (  # noqa: E402
    Estimator,
    Layout,
    Priors,
    Problems,
    Sharpness,
    daily_rank_fraction,
    integrated_autocorrelation_time,
    make_problems,
    make_signals,
    natural_draws,
    natural_point,
    simulate,
)
from studies.template_posterior import tempering_from_residual  # noqa: E402

SLOTS: Final[int] = 48
SHARP: Final[Sharpness] = Sharpness(rank=150.0, smoothing=1e-6)
DEVICE: Final = torch.device("cpu")


def _prices(*, days: int, seed: int) -> np.ndarray:
    """Daily prices with a morning and an evening peak and a different level every day."""
    rng = np.random.default_rng(seed)
    slot = np.arange(SLOTS)
    shape = 40 + 25 * np.exp(-((slot - 36) ** 2) / 20) + 15 * np.exp(-((slot - 16) ** 2) / 30)
    return np.concatenate(
        [shape * rng.uniform(0.8, 1.2) + rng.normal(0, 4, SLOTS) for _ in range(days)]
    )


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
    """One rank unit (merchant) and one window unit, in separate classes."""
    rank, valid = daily_rank_fraction(prices=_prices(days=days, seed=1))
    charge, discharge = _window(days=days, charge=(2, 12), discharge=(34, 40))
    return {
        "rank_fraction": rank[None],
        "rank_valid": valid[None],
        "fixed_charge": charge[None],
        "fixed_discharge": discharge[None],
    }


LAYOUT: Final = Layout(unit_class=(0, 1), n_classes=2, n_rank=1)


def _theta(*, power: tuple[float, float], duration: float, efficiency: float) -> np.ndarray:
    """Parameters of the two-unit layout in the unconstrained space."""
    logit = lambda p: float(np.log(p / (1 - p)))  # noqa: E731
    return np.array(
        [
            np.log(power[0]),
            np.log(power[1]),
            np.log(duration),
            np.log(duration),
            logit(efficiency),
            logit(efficiency),
            logit(0.25 / 0.5),
            logit(0.25 / 0.5),
        ]
    )


def test_the_state_of_charge_stays_within_the_usable_energy_over_a_week_of_random_policies() -> (
    None
):
    days = 7
    signals = make_signals(**_signals(days=days), device=DEVICE, dtype=torch.float64)
    rng = np.random.default_rng(0)
    theta = torch.as_tensor(
        np.stack(
            [
                _theta(
                    power=(rng.uniform(0.5, 4), rng.uniform(0.5, 4)),
                    duration=rng.uniform(0.5, 6),
                    efficiency=rng.uniform(0.6, 0.98),
                )
                + rng.normal(0, 0.8, LAYOUT.n_parameters) * np.array([0, 0, 0, 0, 0, 0, 1, 1])
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


def test_the_gradient_of_the_export_matches_central_finite_differences() -> None:
    signals = make_signals(**_signals(days=2), device=DEVICE, dtype=torch.float64)
    sharpness = Sharpness(rank=20.0, smoothing=1e-3)
    theta = torch.as_tensor(
        _theta(power=(1.7, 2.3), duration=1.8, efficiency=0.83)[None] + 0.03,
        dtype=torch.float64,
    ).requires_grad_(True)

    def objective(vector: torch.Tensor) -> torch.Tensor:
        weights = torch.linspace(0.5, 1.5, 96, dtype=torch.float64)
        return (
            simulate(theta=vector, layout=LAYOUT, signals=signals, sharpness=sharpness) ** 2
            * weights
        ).sum()

    assert torch.autograd.gradcheck(objective, (theta,), eps=1e-6, atol=1e-5, rtol=1e-5)


def test_rank_fractions_count_ties_as_half_and_mark_days_with_a_missing_price() -> None:
    day_one = np.arange(SLOTS, dtype=float)
    day_two = np.zeros(SLOTS)
    day_three = np.arange(SLOTS, dtype=float)
    day_three[5] = np.nan
    fraction, valid = daily_rank_fraction(prices=np.concatenate([day_one, day_two, day_three]))
    assert fraction[0] == pytest.approx(0.5 / SLOTS)
    assert fraction[SLOTS - 1] == pytest.approx((SLOTS - 0.5) / SLOTS)
    assert fraction[SLOTS + 3] == pytest.approx(0.5)
    assert valid.reshape(3, SLOTS).all(axis=1).tolist() == [True, True, False]


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


def _estimator(
    *, days: int, durations_sd: float = 0.5, sharp_efficiency: bool = False
) -> Estimator:
    priors = Priors(
        log_duration_mean=np.log([2.0, 2.0]),
        log_duration_sd=np.array([durations_sd, durations_sd]),
        efficiency_alpha=np.array([30.0, 30.0]) if not sharp_efficiency else np.array([1e4, 1e4]),
        efficiency_beta=np.array([5.0, 5.0]) if not sharp_efficiency else np.array([1.2e3, 1.2e3]),
    )
    return Estimator(layout=LAYOUT, signals_numpy=_signals(days=days), priors=priors, device=DEVICE)


def _problems(*, estimator: Estimator, theta_true: np.ndarray, days: int, noise: float) -> Problems:
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
        sharpness=SHARP,
    )[0].numpy()
    rng = np.random.default_rng(5)
    aggregate = base - export + rng.normal(0, noise, n_time)
    valid = np.ones(n_time, dtype=bool)
    return make_problems(
        aggregate=aggregate[None, None, :],
        valid=valid[None],
        free=[free],
        power_scale=np.array([[2.0]]),
    )


STARTS: Final = np.array(
    [
        [-0.5, -3.0, np.log(1.5), np.log(2.5), 1.0, 1.0, -0.5, -0.5],
        [0.5, -3.0, np.log(3.0), np.log(2.0), 2.0, 1.5, 0.5, 0.0],
    ]
)
STAGES: Final = [
    (80, 0.08, Sharpness(rank=30.0, smoothing=1e-2)),
    (80, 0.03, Sharpness(rank=150.0, smoothing=1e-6)),
]


def test_a_noise_free_sum_built_by_the_model_gives_back_its_power_duration_and_efficiency() -> None:
    days = 7
    estimator = _estimator(days=days)
    truth = _theta(power=(3.0, 1e-3), duration=2.0, efficiency=0.85)
    problems = _problems(estimator=estimator, theta_true=truth, days=days, noise=1e-3)
    fit = estimator.fit(problems=problems, starts=STARTS, stages=STAGES)
    best = fit.best_start[0, 0]
    point = natural_point(theta=fit.theta[0, 0, best], layout=LAYOUT)
    assert point["power"][0] == pytest.approx(3.0, rel=0.02)
    assert point["duration"][0] == pytest.approx(2.0, rel=0.02)
    assert point["efficiency"][0] == pytest.approx(0.85, rel=0.02)
    assert point["energy"][0] == pytest.approx(6.0, rel=0.02)


def test_with_shape_parameters_pinned_the_laplace_interval_matches_least_squares() -> None:
    days = 7
    estimator = _estimator(days=days, durations_sd=1e-4, sharp_efficiency=True)
    truth = _theta(power=(3.0, 1e-3), duration=2.0, efficiency=0.88)
    noise = 0.5
    problems = _problems(estimator=estimator, theta_true=truth, days=days, noise=noise)
    # A very wide power prior, so that the prior does not add to the interval.
    problems = replace(problems, power_scale=np.array([[500.0]]))
    fit = estimator.fit(problems=problems, starts=STARTS[:1], stages=STAGES)
    assert fit.has_interval[0, 0, 0]
    theta = fit.theta[0, 0, 0]
    log_power_sd = np.sqrt(fit.covariance[0, 0, 0][0, 0])
    sd_power = np.exp(theta[0]) * log_power_sd
    # Ordinary least squares: sd = sigma / norm(M t), t the unit's per-megawatt export.
    export = simulate(
        theta=torch.as_tensor(theta[None]),
        layout=LAYOUT,
        signals=estimator.signals64,
        sharpness=SHARP,
    )[0].numpy()
    basis = problems.basis[0]
    # The window unit's power is negligible, so the export is the rank unit's alone.
    template = export / np.exp(theta[0])
    template = template - basis @ (basis.T @ template)
    sigma = np.sqrt(fit.rss[0, 0, 0] / (problems.valid.sum() - problems.rank[0]))
    ols_sd = sigma / np.linalg.norm(template)
    assert sd_power == pytest.approx(ols_sd * fit.tau[0, 0, 0] ** 0.5, rel=0.01)


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
