"""A Bayesian fit of battery templates to an aggregate's half-hourly flow.

The aggregate in one block of time is modelled as

    y(t) = F(t) beta + G(t) a + noise,

where the columns of `F` (a calendar baseline, the solar curves, and charge-only nuisance
templates) have signed coefficients `beta` under a wide Gaussian prior, and the columns of `G` are
the negated one-megawatt templates of the batteries, with powers `a >= 0` under a half-normal
prior. A *combination* picks one column per battery class from a grid of durations and round-trip
efficiencies. The posterior is a sum over combinations: within a combination the model is
linear-Gaussian, so `beta` is integrated out exactly, and the powers' integral over the positive
orthant is a multivariate-normal orthant probability. The noise is serially correlated, so the
columns and the aggregate are prewhitened with a first-order autoregression whose coefficient and
variance are estimated from the scored aggregate's own residual.

Log Bayes factor of the model with batteries against the model with none:

    log BF = log sum_c w_c * 2^p * Z_c * P_c(a >= 0)

with `w_c` the combination's prior weight, `p` the number of power columns, `Z_c` the
unconstrained Gaussian evidence ratio against the no-battery model, and `P_c` the orthant
probability of the unconstrained posterior of the powers.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import multivariate_normal, truncnorm

MAX_ORTHANT_POINTS: Final[int] = 2000
"""The most integrand evaluations of one multivariate-normal orthant probability. scipy's default
is a million points per dimension; a few thousand holds the relative error of the probabilities
this fit meets well below the differences between combinations."""
PRUNE_LOG_MARGIN: Final[float] = 14.0
"""A combination whose upper bound on its log weight lies this far below the best combination's
log weight is skipped: it carries less than 1e-6 of the best combination's weight."""
SURE_ORTHANT_Z: Final[float] = 8.0
"""When every power's posterior mean is this many standard deviations above zero, the orthant
probability is 1 to within 1e-15 and is not computed."""
LOG_FLOOR: Final[float] = -1e6
MAX_AUTOCORRELATION: Final[float] = 0.99
FREE_PRIOR_SD_OVER_SIGNAL_SD: Final[float] = 100.0
"""The prior standard deviation of a free coefficient, as a multiple of the aggregate's standard
deviation. It is wide enough to be uninformative, and it cancels from the Bayes factor because the
no-battery model carries the same free columns."""
SIGMA2_FLOOR_RATIO: Final[float] = 1e-8
"""The noise variance never falls below this fraction of the aggregate's variance, so that an
aggregate that is exactly zero for long stretches does not give a singular fit."""
GIBBS_BURN_IN_SWEEPS: Final[int] = 60
POSTERIOR_MASS_SAMPLED: Final[float] = 0.99


@dataclass(frozen=True)
class ComboGrid:
    """The combinations of battery columns that the posterior sums over.

    Attributes:
        combos: Shape (combinations, power columns): the candidate-column index of each power
            column in each combination.
        log_prior: Shape (combinations,): the log prior weight of each combination, summing to 1
            after exponentiation.
        axes: The size of each grid axis, in the order that `combos` was built with the last axis
            varying fastest, so that a flat combination index reshapes to `axes`.
    """

    combos: np.ndarray
    log_prior: np.ndarray
    axes: tuple[int, ...]


@dataclass(frozen=True)
class Prewhitened:
    """Columns and aggregate after the autoregressive filter, and their Gram products.

    Attributes:
        free_gram: `F'F` of the free columns.
        cross_gram: `F'G` between the free and candidate columns.
        candidate_gram: `G'G` of the candidate columns.
        free_target: `F'y`.
        candidate_target: `G'y`.
        target_norm: `y'y`.
        effective_rows: How many equations contributed.
    """

    free_gram: np.ndarray
    cross_gram: np.ndarray
    candidate_gram: np.ndarray
    free_target: np.ndarray
    candidate_target: np.ndarray
    target_norm: float
    effective_rows: int


def prewhiten(
    *,
    free: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    rho: float,
) -> Prewhitened:
    """Filter the columns and the aggregate by `x_t - rho * x_{t-1}` and form the Gram products.

    A half-hour that is not valid has its column values and its aggregate set to zero, so its own
    equation reads 0 = 0 and the next half-hour's equation is left unfiltered. The first half-hour
    has no predecessor and is dropped.

    Args:
        free: Shape (rows, free columns).
        candidates: Shape (rows, candidate columns).
        target: The aggregate, shape (rows,).
        valid: True where the aggregate and every column are finite.
        rho: The autoregressive coefficient.

    Returns:
        The Gram products of the filtered system.
    """
    stacked = np.where(valid[:, None], np.hstack([free, candidates, target[:, None]]), 0.0)
    filtered = np.where(valid[1:, None], stacked[1:] - rho * stacked[:-1], 0.0)
    n_free = free.shape[1]
    n_candidates = candidates.shape[1]
    f = filtered[:, :n_free]
    g = filtered[:, n_free : n_free + n_candidates]
    y = filtered[:, -1]
    return Prewhitened(
        free_gram=f.T @ f,
        cross_gram=f.T @ g,
        candidate_gram=g.T @ g,
        free_target=f.T @ y,
        candidate_target=g.T @ y,
        target_norm=float(y @ y),
        effective_rows=int(valid[1:].sum()),
    )


@dataclass(frozen=True)
class Projected:
    """The candidate columns after the free columns are integrated out.

    Attributes:
        gram: `H = G'G - G'F S^-1 F'G` where `S = F'F + (sigma^2 / tau^2) I`.
        target: `h = G'y - G'F S^-1 F'y`.
        free_solution_matrix: `S^-1`, kept to recover the free coefficients.
    """

    gram: np.ndarray
    target: np.ndarray
    free_solution_matrix: np.ndarray


def project_out_free(*, system: Prewhitened, sigma2: float, free_prior_sd: float) -> Projected:
    """Integrate the free coefficients out of the candidate columns.

    Args:
        system: The filtered system.
        sigma2: The noise variance after filtering.
        free_prior_sd: The prior standard deviation `tau` of every free coefficient.

    Returns:
        The projected Gram matrix and target of the candidate columns.
    """
    n_free = system.free_gram.shape[0]
    stabilised = system.free_gram + (sigma2 / free_prior_sd**2) * np.eye(n_free)
    inverse = np.linalg.inv(stabilised)
    cross = system.cross_gram
    return Projected(
        gram=system.candidate_gram - cross.T @ inverse @ cross,
        target=system.candidate_target - cross.T @ inverse @ system.free_target,
        free_solution_matrix=inverse,
    )


def log_orthant_probability(
    *, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, maxpts: int = MAX_ORTHANT_POINTS
) -> float:
    """Return the log probability that every coordinate of a Gaussian is non-negative.

    Args:
        mean: The mean, shape (p,).
        cov: The covariance, shape (p, p).
        rng: The random generator the quasi-Monte-Carlo integration draws from.
        maxpts: The most integrand evaluations.

    Returns:
        `log P(X >= 0)`, floored at `LOG_FLOOR`.
    """
    sd = np.sqrt(np.diag(cov))
    z = mean / sd
    if z.min() > SURE_ORTHANT_Z:
        return 0.0
    if len(mean) == 1:
        return float(max(log_ndtr(z[0]), LOG_FLOOR))
    # P(X >= 0) = P(-X <= 0), and -X has mean -mean and the same covariance.
    value = multivariate_normal.cdf(
        np.zeros(len(mean)), mean=-mean, cov=cov, maxpts=maxpts, rng=rng
    )
    return float(max(np.log(value), LOG_FLOOR)) if value > 0 else LOG_FLOOR


@dataclass(frozen=True)
class Evaluation:
    """The evidence of every combination under one noise setting.

    Attributes:
        log_weight: Shape (combinations,): log prior weight plus log evidence ratio against the
            no-battery model, with the orthant probability included. `LOG_FLOOR` where pruned.
        pruned: True for combinations skipped because their upper bound was far below the best.
        mean: Shape (combinations, p): the unconstrained posterior mean of the powers.
        cov: Shape (combinations, p, p): the unconstrained posterior covariance.
        log_bayes_factor: `logsumexp` of `log_weight` over the evaluated combinations.
        orthant_evaluations: How many orthant probabilities were computed numerically.
    """

    log_weight: np.ndarray
    pruned: np.ndarray
    mean: np.ndarray
    cov: np.ndarray
    log_bayes_factor: float
    orthant_evaluations: int


def evaluate_combinations(
    *,
    projected: Projected,
    grid: ComboGrid,
    sigma2: float,
    power_prior_scale: float,
    rng: np.random.Generator,
    maxpts: int = MAX_ORTHANT_POINTS,
) -> Evaluation:
    """Compute every combination's log evidence ratio, skipping combinations that cannot matter.

    For a combination with power columns `c`: `Lambda = H[c, c] / sigma2`, `b = h[c] / sigma2`, the
    unconstrained posterior of the powers is `N(mu, Sigma)` with `Sigma = (Lambda + I/s^2)^-1` and
    `mu = Sigma b`, and the log evidence ratio against the no-battery model is
    `p log 2 - 0.5 log det(I + s^2 Lambda) + 0.5 b' Sigma b + log P(a >= 0)`.
    The first three terms bound the whole from above, because a probability is at most 1.

    Args:
        projected: The candidate columns with the free columns integrated out.
        grid: The combinations.
        sigma2: The noise variance after filtering.
        power_prior_scale: The scale `s` of the half-normal prior of every power, in megawatts.
        rng: The random generator for the orthant integrations.
        maxpts: The most integrand evaluations per orthant probability.

    Returns:
        The evaluation.
    """
    combos = grid.combos
    n_combos, n_power = combos.shape
    lam = projected.gram[combos[:, :, None], combos[:, None, :]] / sigma2
    b = projected.target[combos] / sigma2
    precision = lam + np.eye(n_power) / power_prior_scale**2
    cov = np.linalg.inv(precision)
    cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
    mean = np.einsum("kij,kj->ki", cov, b)
    _, log_det_precision = np.linalg.slogdet(precision)
    log_det_term = log_det_precision + 2.0 * n_power * np.log(power_prior_scale)
    quadratic = np.einsum("ki,ki->k", b, mean)
    log_unconstrained = n_power * np.log(2.0) - 0.5 * log_det_term + 0.5 * quadratic
    sd = np.sqrt(np.einsum("kii->ki", cov))
    log_single = log_ndtr(mean / sd).min(axis=1)
    upper = grid.log_prior + log_unconstrained + np.minimum(log_single, 0.0)
    order = np.argsort(-upper)
    log_weight = np.full(n_combos, LOG_FLOOR)
    pruned = np.ones(n_combos, dtype=bool)
    best = -np.inf
    evaluations = 0
    for k in order:
        if upper[k] < best - PRUNE_LOG_MARGIN:
            break
        orthant = log_orthant_probability(mean=mean[k], cov=cov[k], rng=rng, maxpts=maxpts)
        evaluations += int((mean[k] / sd[k]).min() <= SURE_ORTHANT_Z)
        log_weight[k] = grid.log_prior[k] + log_unconstrained[k] + orthant
        pruned[k] = False
        best = max(best, log_weight[k])
    evaluated = ~pruned
    return Evaluation(
        log_weight=log_weight,
        pruned=pruned,
        mean=mean,
        cov=cov,
        log_bayes_factor=float(logsumexp(log_weight[evaluated])),
        orthant_evaluations=evaluations,
    )


def sample_truncated_gaussian(
    *,
    mean: np.ndarray,
    cov: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    burn_in: int = GIBBS_BURN_IN_SWEEPS,
) -> np.ndarray:
    """Draw from a Gaussian truncated to the positive orthant, by Gibbs sampling.

    Each draw is its own chain, started at the mean clipped to be positive and run for `burn_in`
    sweeps over the coordinates, so the draws are independent of one another.

    Args:
        mean: The mean of the untruncated Gaussian, shape (p,).
        cov: Its covariance, shape (p, p).
        n_draws: How many draws.
        rng: The random generator.
        burn_in: The sweeps each chain runs.

    Returns:
        Shape (n_draws, p), every entry non-negative.
    """
    p = len(mean)
    precision = np.linalg.inv(cov)
    sd_conditional = 1.0 / np.sqrt(np.diag(precision))
    x = np.tile(np.maximum(mean, 1e-9 * np.sqrt(np.diag(cov))), (n_draws, 1))
    for _ in range(burn_in):
        for i in range(p):
            others = [j for j in range(p) if j != i]
            shift = (x[:, others] - mean[others]) @ precision[i, others]
            centre = mean[i] - shift / precision[i, i]
            x[:, i] = truncnorm.rvs(
                a=(0.0 - centre) / sd_conditional[i],
                b=np.inf,
                loc=centre,
                scale=sd_conditional[i],
                random_state=rng,
            )
    return x


@dataclass(frozen=True)
class SumPosterior:
    """The posterior of one aggregate over the combination grid and the powers.

    Attributes:
        evaluation: The final evaluation, under the final noise setting.
        log_posterior: Shape (combinations,): normalised log posterior weights, `LOG_FLOOR` where
            pruned.
        rho: The autoregressive coefficient used.
        sigma2: The filtered noise variance used.
        log_bayes_factor: Batteries against none.
        best_combination: The combination of highest posterior weight.
    """

    evaluation: Evaluation
    log_posterior: np.ndarray
    rho: float
    sigma2: float
    log_bayes_factor: float
    best_combination: int


def _residual(
    *,
    free: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    system: Prewhitened,
    projected: Projected,
    columns: np.ndarray,
    powers: np.ndarray,
) -> np.ndarray:
    """Return the unfiltered residual, NaN at invalid half-hours."""
    free_part = system.free_target - system.cross_gram[:, columns] @ powers
    beta = projected.free_solution_matrix @ free_part
    fitted = free @ beta + candidates[:, columns] @ powers
    return np.where(valid, target - fitted, np.nan)


def estimate_noise(
    *, residual: np.ndarray, n_parameters: int, effective_rows: int
) -> tuple[float, float]:
    """Estimate a first-order autoregression's coefficient and innovation variance.

    Args:
        residual: The unfiltered residual, NaN where invalid.
        n_parameters: How many coefficients were fitted, to correct the variance's degrees of
            freedom.
        effective_rows: How many equations the fit used.

    Returns:
        The coefficient (clipped to `[0, MAX_AUTOCORRELATION]`) and the innovation variance.
    """
    now, before = residual[1:], residual[:-1]
    pair = np.isfinite(now) & np.isfinite(before)
    rho = float((now[pair] * before[pair]).sum() / (before[pair] ** 2).sum())
    rho = float(np.clip(rho, 0.0, MAX_AUTOCORRELATION))
    innovations = now[pair] - rho * before[pair]
    degrees = max(effective_rows - n_parameters, 1)
    return rho, float((innovations**2).sum() / degrees)


def fit_aggregate(
    *,
    free: np.ndarray,
    candidates: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    grid: ComboGrid,
    power_prior_scale: float,
    rng: np.random.Generator,
    noise_iterations: int = 2,
    maxpts: int = MAX_ORTHANT_POINTS,
) -> SumPosterior:
    """Fit one aggregate: estimate the noise, then compute the posterior over the combinations.

    The noise starts white with the variance of the free-columns-only residual. Each iteration
    evaluates every combination, takes the combination of highest evidence, forms the residual at
    its posterior mean (powers clipped to be non-negative), and re-estimates the autoregressive
    coefficient and variance from that residual. The final evaluation uses the last estimate.

    Args:
        free: The free columns, shape (rows, free columns).
        candidates: The candidate battery columns, already negated (`-template`), shape
            (rows, candidate columns).
        target: The aggregate in megawatts.
        valid: True where the aggregate and every column are finite.
        grid: The combinations.
        power_prior_scale: The half-normal scale of every power, in megawatts.
        rng: The random generator.
        noise_iterations: How many times the noise is re-estimated.
        maxpts: The most integrand evaluations per orthant probability.

    Returns:
        The posterior.
    """
    free_sd = float(np.std(target[valid]))
    tau = FREE_PRIOR_SD_OVER_SIGNAL_SD * max(free_sd, 1e-12)
    sigma2_floor = max(SIGMA2_FLOOR_RATIO * free_sd**2, 1e-24)
    rho = 0.0
    n_parameters = free.shape[1] + grid.combos.shape[1]
    start = prewhiten(free=free, candidates=candidates, target=target, valid=valid, rho=0.0)
    ols = np.linalg.lstsq(start.free_gram, start.free_target, rcond=None)[0]
    sigma2 = max(
        float(
            (start.target_norm - 2 * ols @ start.free_target + ols @ start.free_gram @ ols)
            / max(start.effective_rows - free.shape[1], 1)
        ),
        sigma2_floor,
    )
    evaluation = None
    for iteration in range(noise_iterations + 1):
        system = prewhiten(free=free, candidates=candidates, target=target, valid=valid, rho=rho)
        projected = project_out_free(system=system, sigma2=sigma2, free_prior_sd=tau)
        evaluation = evaluate_combinations(
            projected=projected,
            grid=grid,
            sigma2=sigma2,
            power_prior_scale=power_prior_scale,
            rng=rng,
            maxpts=maxpts,
        )
        if iteration == noise_iterations:
            break
        best = int(np.argmax(evaluation.log_weight))
        columns = grid.combos[best]
        powers = np.maximum(evaluation.mean[best], 0.0)
        residual = _residual(
            free=free,
            candidates=candidates,
            target=target,
            valid=valid,
            system=system,
            projected=projected,
            columns=columns,
            powers=powers,
        )
        rho, sigma2 = estimate_noise(
            residual=residual, n_parameters=n_parameters, effective_rows=system.effective_rows
        )
        sigma2 = max(sigma2, sigma2_floor)
    assert evaluation is not None
    kept = ~evaluation.pruned
    log_posterior = np.full(len(kept), LOG_FLOOR)
    log_posterior[kept] = evaluation.log_weight[kept] - logsumexp(evaluation.log_weight[kept])
    return SumPosterior(
        evaluation=evaluation,
        log_posterior=log_posterior,
        rho=rho,
        sigma2=sigma2,
        log_bayes_factor=evaluation.log_bayes_factor,
        best_combination=int(np.argmax(log_posterior)),
    )


def posterior_draws(
    *, posterior: SumPosterior, n_draws: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw combinations and powers from the posterior.

    Combinations are drawn by their posterior weight from those that together hold
    `POSTERIOR_MASS_SAMPLED` of it, and the powers from each combination's truncated Gaussian.

    Args:
        posterior: The fitted posterior.
        n_draws: How many draws.
        rng: The random generator.

    Returns:
        The combination index of each draw, shape (n_draws,), and the powers, shape (n_draws, p).
    """
    weights = np.exp(posterior.log_posterior)
    weights[posterior.evaluation.pruned] = 0.0
    order = np.argsort(-weights)
    cumulative = np.cumsum(weights[order])
    keep = order[: int(np.searchsorted(cumulative, POSTERIOR_MASS_SAMPLED)) + 1]
    probabilities = weights[keep] / weights[keep].sum()
    counts = rng.multinomial(n_draws, probabilities)
    combos = []
    powers = []
    for combination, count in zip(keep, counts, strict=True):
        if count == 0:
            continue
        powers.append(
            sample_truncated_gaussian(
                mean=posterior.evaluation.mean[combination],
                cov=posterior.evaluation.cov[combination],
                n_draws=int(count),
                rng=rng,
            )
        )
        combos.append(np.full(int(count), combination))
    return np.concatenate(combos), np.vstack(powers)
