"""Random-effects pooling of per-series steps for experiment 123.

The pooled mean uses the DerSimonian-Laird between-series variance, a Hartung-Knapp interval (t
with `k - 1` degrees of freedom), a percentile interval from resampling whole series, and a
profile-likelihood interval for the between-series standard deviation `tau`. A set of fewer than
`MIN_SERIES` series gets the plain mean and range only.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

MIN_SERIES: Final[int] = 5
N_SERIES_RESAMPLES: Final[int] = 2000
LEVEL: Final[float] = 0.90


@dataclass(frozen=True)
class Pooled:
    """The pooled step of one set of series.

    Attributes:
        k: The number of series.
        mean: The pooled mean (the plain mean when `k` is under `MIN_SERIES`).
        low_hk: Lower end of the Hartung-Knapp interval (NaN when `k` is under `MIN_SERIES`).
        high_hk: Upper end of the Hartung-Knapp interval.
        low_boot: Lower end of the series-bootstrap percentile interval.
        high_boot: Upper end of the series-bootstrap percentile interval.
        tau: DerSimonian-Laird between-series standard deviation.
        tau_low: Lower end of the profile-likelihood interval for `tau`.
        tau_high: Upper end of the profile-likelihood interval for `tau`.
        minimum: The smallest per-series value.
        maximum: The largest per-series value.
        label: `ok`, `cannot tell` (the two intervals disagree about 0), or `plain mean and range`.
    """

    k: int
    mean: float
    low_hk: float
    high_hk: float
    low_boot: float
    high_boot: float
    tau: float
    tau_low: float
    tau_high: float
    minimum: float
    maximum: float
    label: str


def _random_effects(*, values: np.ndarray, variances: np.ndarray) -> tuple[float, float, float]:
    """Return the pooled mean, its Hartung-Knapp variance, and tau squared (DerSimonian-Laird)."""
    k = len(values)
    weights = 1.0 / variances
    fixed = float(np.sum(weights * values) / np.sum(weights))
    q = float(np.sum(weights * (values - fixed) ** 2))
    scale = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau2 = max(0.0, (q - (k - 1)) / scale)
    random_weights = 1.0 / (variances + tau2)
    mean = float(np.sum(random_weights * values) / np.sum(random_weights))
    hk_variance = float(
        np.sum(random_weights * (values - mean) ** 2) / ((k - 1) * np.sum(random_weights))
    )
    return mean, hk_variance, tau2


def _profile_log_likelihood(*, values: np.ndarray, variances: np.ndarray, tau2: float) -> float:
    total = variances + tau2
    weights = 1.0 / total
    mean = float(np.sum(weights * values) / np.sum(weights))
    return float(-0.5 * np.sum(np.log(total) + (values - mean) ** 2 / total))


def _tau_interval(*, values: np.ndarray, variances: np.ndarray) -> tuple[float, float, float]:
    """Return tau and its profile-likelihood interval (maximum likelihood, grid search)."""
    spread = float(np.ptp(values)) + float(np.sqrt(variances.max())) + 1e-9
    grid = np.concatenate([[0.0], np.linspace(1e-4, 2.0 * spread, 2000) ** 2 / (2.0 * spread)])
    log_likelihood = np.array(
        [_profile_log_likelihood(values=values, variances=variances, tau2=t) for t in grid]
    )
    best = log_likelihood.max()
    inside = grid[2.0 * (best - log_likelihood) <= stats.chi2.ppf(LEVEL, df=1)]
    tau2_hat = float(grid[int(np.argmax(log_likelihood))])
    return float(np.sqrt(tau2_hat)), float(np.sqrt(inside.min())), float(np.sqrt(inside.max()))


def pool(*, values: np.ndarray, standard_errors: np.ndarray, rng: np.random.Generator) -> Pooled:
    """Pool per-series steps with their standard errors.

    Args:
        values: One step per series.
        standard_errors: One standard error per series (the bootstrap standard deviation).
        rng: The random generator for the series bootstrap.

    Returns:
        The pooled summary.
    """
    k = len(values)
    nan = float("nan")
    if k < MIN_SERIES:
        return Pooled(
            k=k, mean=float(values.mean()), low_hk=nan, high_hk=nan, low_boot=nan, high_boot=nan,
            tau=nan, tau_low=nan, tau_high=nan, minimum=float(values.min()),
            maximum=float(values.max()), label="plain mean and range",
        )  # fmt: skip
    variances = np.maximum(standard_errors**2, 1e-12)
    mean, hk_variance, _ = _random_effects(values=values, variances=variances)
    margin = float(stats.t.ppf(0.5 + LEVEL / 2.0, df=k - 1) * np.sqrt(hk_variance))
    means = np.empty(N_SERIES_RESAMPLES)
    for draw in range(N_SERIES_RESAMPLES):
        pick = rng.integers(0, k, size=k)
        means[draw] = _random_effects(values=values[pick], variances=variances[pick])[0]
    low_boot, high_boot = np.quantile(means, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    tau, tau_low, tau_high = _tau_interval(values=values, variances=variances)
    hk_has_zero = mean - margin <= 0.0 <= mean + margin
    boot_has_zero = low_boot <= 0.0 <= high_boot
    return Pooled(
        k=k, mean=mean, low_hk=mean - margin, high_hk=mean + margin, low_boot=float(low_boot),
        high_boot=float(high_boot), tau=tau, tau_low=tau_low, tau_high=tau_high,
        minimum=float(values.min()), maximum=float(values.max()),
        label="cannot tell" if hk_has_zero != boot_has_zero else "ok",
    )  # fmt: skip
