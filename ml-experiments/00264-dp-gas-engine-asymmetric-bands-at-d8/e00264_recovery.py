# ruff: noqa: E501
"""Known-answer test of the mixture fit: simulate readings with known unit probabilities.

The base distribution is a normal with standard deviation 6 points (noise scale 1 against a
reference whose quantiles are those of that normal), engine units of 19.7 points, and a unit
probability that is higher in the evening hour block. The fit must recover the noise scale and the
evening/other contrast. A second run uses no engine and must give a mean P(k >= 1) near zero.

    uv run --no-sync python ml-experiments/00264-*/e00264_recovery.py
"""

import e00264_common as common
import e00264_mixture as mixture
import numpy as np
from scipy.stats import norm

N_ROWS = 6000


def simulate(*, engine: bool, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (y, quantiles, features, true k) for a simulated site."""
    rng = np.random.default_rng(seed)
    hours = rng.integers(0, 4, size=N_ROWS)
    sigma = 6.0
    quantiles = np.tile(norm.ppf(common.LEVELS) * sigma, (N_ROWS, 1))
    p_run = np.where(hours == 3, 0.5, 0.1) if engine else np.zeros(N_ROWS)
    k = np.where(rng.random(N_ROWS) < p_run, rng.integers(1, 4, size=N_ROWS), 0)
    delta = common.UNIT_POINTS * np.where(hours == 3, 0.5 * 2.0, 0.1 * 2.0) if engine else np.zeros(N_ROWS)
    y = rng.normal(0.0, sigma, size=N_ROWS) - k * common.UNIT_POINTS + delta
    features = np.zeros((N_ROWS, 9))
    features[:, 0] = 1.0
    features[:, 1:4] = np.eye(4)[hours][:, 1:]
    return y, quantiles, features, k


def main() -> None:
    """Fit the simulated engine and no-engine sites and print the recovered quantities."""
    for engine in (True, False):
        y, quantiles, features, k = simulate(engine=engine, seed=1)
        fit = mixture.fit_mixture(y=y, quantiles=quantiles, features=features)
        probs = mixture.softmax_probabilities(features=features, weights=fit.weights)
        evening = features[:, 3] == 1
        print(f"engine={engine}: noise scale {fit.scale} (truth 1.0)")
        print(f"  true P(k>=1): evening {(k[evening] > 0).mean():.3f}, other {(k[~evening] > 0).mean():.3f}")
        print(f"  fitted P(k>=1): evening {(1 - probs[evening, 0]).mean():.3f}, other {(1 - probs[~evening, 0]).mean():.3f}")


if __name__ == "__main__":
    main()
