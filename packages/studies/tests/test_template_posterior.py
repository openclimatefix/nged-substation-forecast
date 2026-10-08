import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.stats import multivariate_normal, truncnorm
from studies.template_posterior import (
    ComboGrid,
    estimate_noise,
    evaluate_combinations,
    fit_aggregate,
    log_orthant_probability,
    posterior_draws,
    prewhiten,
    project_out_free,
    sample_truncated_gaussian,
)

from studies import template_posterior


def _single_combo_grid(*, columns: list[int]) -> ComboGrid:
    return ComboGrid(combos=np.array([columns]), log_prior=np.array([0.0]), axes=(1,))


def test_the_evidence_ratio_matches_a_brute_force_integral_on_a_three_column_problem() -> None:
    rng = np.random.default_rng(0)
    n_rows = 5  # the filter drops the first row, so four equations remain
    free = np.ones((n_rows, 1))
    candidates = rng.normal(size=(n_rows, 2))
    target = 1.0 + candidates @ np.array([1.5, 0.5]) + 0.5 * rng.normal(size=n_rows)
    sigma2, tau, scale = 1.0, 3.0, 2.0
    system = prewhiten(
        free=free, candidates=candidates, target=target, valid=np.ones(n_rows, bool), rho=0.0
    )
    projected = project_out_free(system=system, sigma2=sigma2, free_prior_sd=tau)

    evaluation = evaluate_combinations(
        projected=projected,
        grid=_single_combo_grid(columns=[0, 1]),
        sigma2=sigma2,
        power_prior_scale=scale,
        rng=np.random.default_rng(1),
        maxpts=200_000,
    )

    y, f, g = target[1:], free[1:], candidates[1:]
    beta = np.arange(-20.0, 20.0, 0.1)
    power = np.arange(0.05, 14.0, 0.1)  # midpoints, so the boundary at zero is not over-counted
    bb, a1, a2 = np.meshgrid(beta, power, power, indexing="ij")
    mean = bb[..., None] * f[:, 0] + a1[..., None] * g[:, 0] + a2[..., None] * g[:, 1]
    log_likelihood = -0.5 * ((y - mean) ** 2).sum(axis=-1) / sigma2 - 0.5 * len(y) * np.log(
        2 * np.pi * sigma2
    )
    log_prior = (
        -0.5 * bb**2 / tau**2
        - 0.5 * np.log(2 * np.pi * tau**2)
        + np.log(2.0) * 2
        - 0.5 * (a1**2 + a2**2) / scale**2
        - np.log(2 * np.pi * scale**2)
    )
    integral = np.exp(log_likelihood + log_prior).sum() * 0.1**3
    covariance = sigma2 * np.eye(len(y)) + tau**2 * np.outer(f[:, 0], f[:, 0])
    log_null = multivariate_normal.logpdf(y, mean=np.zeros(len(y)), cov=covariance)
    assert log_null + evaluation.log_weight[0] == pytest.approx(np.log(integral), abs=0.02)


def test_the_orthant_probability_of_two_correlated_zero_mean_variables_is_a_closed_form() -> None:
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    value = log_orthant_probability(mean=np.zeros(2), cov=cov, rng=np.random.default_rng(0))

    assert np.exp(value) == pytest.approx(0.25 + np.arcsin(0.5) / (2 * np.pi), abs=2e-3)


def test_the_orthant_probability_of_one_variable_is_the_normal_cdf() -> None:
    value = log_orthant_probability(
        mean=np.array([0.7]), cov=np.array([[4.0]]), rng=np.random.default_rng(0)
    )

    assert value == pytest.approx(log_ndtr(0.7 / 2.0))


def test_the_orthant_probability_is_skipped_when_every_mean_is_far_above_zero() -> None:
    value = log_orthant_probability(
        mean=np.array([50.0, 60.0]), cov=np.eye(2), rng=np.random.default_rng(0)
    )

    assert value == 0.0


def test_six_dimensional_orthant_probability_at_the_default_budget_is_close_to_a_large_budget() -> (
    None
):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 6))
    cov = a @ a.T / 6 + 0.3 * np.eye(6)
    mean = rng.normal(0.5, 0.5, size=6)

    cheap = log_orthant_probability(mean=mean, cov=cov, rng=np.random.default_rng(4))
    precise = log_orthant_probability(
        mean=mean, cov=cov, rng=np.random.default_rng(5), maxpts=400_000
    )

    assert cheap == pytest.approx(precise, abs=0.05)


def test_the_truncated_gibbs_sampler_never_returns_a_negative_power() -> None:
    draws = sample_truncated_gaussian(
        mean=np.array([-2.0, 0.1, 1.0]),
        cov=np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.3], [0.0, 0.3, 2.0]]),
        n_draws=500,
        rng=np.random.default_rng(0),
    )

    assert draws.min() >= 0.0


def test_the_truncated_gibbs_sampler_recovers_the_mean_of_independent_truncated_normals() -> None:
    mean = np.array([0.5, -1.0])
    sd = np.array([1.0, 2.0])

    draws = sample_truncated_gaussian(
        mean=mean, cov=np.diag(sd**2), n_draws=20_000, rng=np.random.default_rng(1)
    )

    expected = [
        truncnorm.mean(a=(0 - m) / s, b=np.inf, loc=m, scale=s)
        for m, s in zip(mean, sd, strict=True)
    ]
    assert draws.mean(axis=0) == pytest.approx(expected, rel=0.03)


def test_the_truncated_gibbs_sampler_matches_rejection_sampling_on_a_correlated_case() -> None:
    mean = np.array([0.3, 0.2])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    rng = np.random.default_rng(2)
    everything = rng.multivariate_normal(mean, cov, size=400_000)
    accepted = everything[(everything >= 0).all(axis=1)]

    draws = sample_truncated_gaussian(mean=mean, cov=cov, n_draws=20_000, rng=rng)

    assert draws.mean(axis=0) == pytest.approx(accepted.mean(axis=0), rel=0.03)
    assert np.corrcoef(draws.T)[0, 1] == pytest.approx(np.corrcoef(accepted.T)[0, 1], abs=0.04)


def test_prewhitening_filters_valid_rows_and_leaves_the_row_after_a_gap_unfiltered() -> None:
    free = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    candidates = np.array([[1.0], [0.0], [2.0], [1.0], [3.0]])
    target = np.array([2.0, 3.0, np.nan, 5.0, 7.0])
    valid = np.array([True, True, False, True, True])
    rho = 0.5

    system = prewhiten(
        free=free, candidates=candidates, target=np.nan_to_num(target), valid=valid, rho=rho
    )

    # Rows 1.. are filtered: row 1 uses row 0, row 2 is invalid (0 = 0), and row 3 follows an
    # invalid row, so it is left unfiltered. Row 4 uses row 3.
    f = np.array([2 - 0.5 * 1, 0.0, 4.0, 5 - 0.5 * 4])
    g = np.array([0 - 0.5 * 1, 0.0, 1.0, 3 - 0.5 * 1])
    y = np.array([3 - 0.5 * 2, 0.0, 5.0, 7 - 0.5 * 5])
    assert system.free_gram[0, 0] == pytest.approx(f @ f)
    assert system.cross_gram[0, 0] == pytest.approx(f @ g)
    assert system.candidate_target[0] == pytest.approx(g @ y)
    assert system.target_norm == pytest.approx(y @ y)
    assert system.effective_rows == 3


def test_the_noise_estimate_recovers_an_autoregression() -> None:
    rng = np.random.default_rng(5)
    innovations = 0.5 * rng.standard_normal(20_000)
    residual = np.zeros(20_000)
    for t in range(1, len(residual)):
        residual[t] = 0.8 * residual[t - 1] + innovations[t]

    rho, sigma2 = estimate_noise(residual=residual, n_parameters=10, effective_rows=19_999)

    assert rho == pytest.approx(0.8, abs=0.02)
    assert sigma2 == pytest.approx(0.25, rel=0.05)


def _boxcar_columns(*, n_days: int, widths: tuple[int, ...]) -> np.ndarray:
    """One column per width: -1 for 4 half-hours from 02:00, then +1 for `width` from 17:00."""
    columns = np.zeros((n_days * 48, len(widths)))
    for k, width in enumerate(widths):
        day = np.zeros(48)
        day[4:8] = -1.0
        day[34 : 34 + width] = 1.0
        columns[:, k] = np.tile(day, n_days)
    return columns


def test_on_a_noise_free_sum_the_posterior_puts_most_mass_on_the_true_duration() -> None:
    n_days = 20
    templates = _boxcar_columns(n_days=n_days, widths=(1, 2, 4, 6))
    true_column = 2
    rng = np.random.default_rng(0)
    baseline = 10.0 + np.tile(2.0 * np.sin(np.linspace(0, 2 * np.pi, 48)), n_days)
    target = baseline - 3.0 * templates[:, true_column] + 0.01 * rng.standard_normal(len(baseline))
    free = np.column_stack(
        [np.ones(len(target)), np.tile(np.sin(np.linspace(0, 2 * np.pi, 48)), n_days)]
    )
    grid = ComboGrid(combos=np.arange(4)[:, None], log_prior=np.full(4, np.log(0.25)), axes=(4,))

    posterior = fit_aggregate(
        free=free,
        candidates=-templates,
        target=target,
        valid=np.ones(len(target), bool),
        grid=grid,
        power_prior_scale=5.0,
        rng=np.random.default_rng(1),
    )

    weights = np.exp(posterior.log_posterior)
    assert weights[true_column] > 0.95
    assert posterior.log_bayes_factor > 50.0


def test_pruning_changes_the_log_bayes_factor_by_less_than_the_margin_allows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_days = 12
    templates = _boxcar_columns(n_days=n_days, widths=(1, 2, 4, 6, 8, 10))
    rng = np.random.default_rng(2)
    target = 10.0 - 2.0 * templates[:, 3] + rng.standard_normal(n_days * 48)
    free = np.ones((len(target), 1))
    grid = ComboGrid(combos=np.arange(6)[:, None], log_prior=np.full(6, np.log(1 / 6)), axes=(6,))

    def fit() -> template_posterior.SumPosterior:
        return fit_aggregate(
            free=free,
            candidates=-templates,
            target=target,
            valid=np.ones(len(target), bool),
            grid=grid,
            power_prior_scale=4.0,
            rng=np.random.default_rng(3),
        )

    pruned = fit()
    monkeypatch.setattr(template_posterior, "PRUNE_LOG_MARGIN", 1e9)
    full = fit()

    assert pruned.evaluation.pruned.sum() > 0
    assert full.evaluation.pruned.sum() == 0
    assert pruned.log_bayes_factor == pytest.approx(full.log_bayes_factor, abs=1e-5)


def _ar_noise(*, n: int, rho: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros(n)
    innovations = sd * rng.standard_normal(n)
    for t in range(1, n):
        noise[t] = rho * noise[t - 1] + innovations[t]
    return noise


def test_a_battery_in_correlated_noise_gives_a_large_bayes_factor_and_a_good_interval() -> None:
    n_days = 56
    templates = _boxcar_columns(n_days=n_days, widths=(2, 4, 6))
    day = np.linspace(0, 2 * np.pi, 48)
    free = np.column_stack(
        [np.ones(n_days * 48), np.tile(np.sin(day), n_days), np.tile(np.cos(day), n_days)]
    )
    grid = ComboGrid(combos=np.arange(3)[:, None], log_prior=np.full(3, np.log(1 / 3)), axes=(3,))

    def fit(power: float) -> template_posterior.SumPosterior:
        noise = _ar_noise(n=n_days * 48, rho=0.8, sd=0.6, rng=np.random.default_rng(12))
        target = 20.0 + 3 * np.sin(np.tile(day, n_days)) - power * templates[:, 1] + noise
        return fit_aggregate(
            free=free,
            candidates=-templates,
            target=target,
            valid=np.ones(len(target), bool),
            grid=grid,
            power_prior_scale=4.0,
            rng=np.random.default_rng(13),
        )

    with_battery = fit(power=3.0)
    without = fit(power=0.0)

    assert with_battery.rho == pytest.approx(0.8, abs=0.1)
    assert with_battery.log_bayes_factor > without.log_bayes_factor + 10.0
    assert without.log_bayes_factor < 3.0
    _, powers = posterior_draws(posterior=with_battery, n_draws=400, rng=np.random.default_rng(14))
    low, high = np.quantile(powers[:, 0], [0.05, 0.95])
    assert powers.min() >= 0.0
    assert low < 3.0 < high
