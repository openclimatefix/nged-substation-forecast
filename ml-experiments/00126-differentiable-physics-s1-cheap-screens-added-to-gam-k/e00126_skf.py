"""G-skf of experiment 126: a Kalman filter with three hidden regimes, mixed by the GPB1 collapse.

**The model.** The residual stream of the frozen GAM-K goes through one two-state filter (a level
that follows a random walk and a transient that follows an AR(1)) per hour band, exactly as
`studies.dp_assimilation.run_kalman` does. G-skf runs three such filters side by side at every
reading, one per regime, and mixes them by the regime probabilities:

- **normal:** the plain filter's level step variance and observation variance;
- **shift** (a load transfer or level shift): the level step variance is 100 times the normal one;
- **outlier** (a bad meter): the observation variance is 100 times the normal one, and the reading
  does not update the state.

After each reading the three posterior states collapse into one Gaussian, weighted by the posterior
regime probabilities (the first-order generalised pseudo-Bayesian approximation, GPB1). The regime
changes between half-hours with a 3 by 3 transition matrix, raised to the gap in half-hours between
two readings.

**The transition matrix is fitted by a forward-only expectation-maximisation.** Each iteration
runs the filter, accumulates the expected one-step transition counts from the filtered
probabilities of consecutive half-hours (the readings that follow a reading by exactly one
half-hour), and sets each row to the counts plus a Dirichlet prior of 200 pseudo-counts per row,
divided by their sum. There is no backward pass, so the counts use the filtered and not the
smoothed regime probabilities.

**Every function takes streams on a leading axis**, so that the injection grid can run hundreds of
streams of one substation together; the real stream is one row. All streams share one set of reading
positions, and a stream with a missing reading at a position skips that reading.
"""

from dataclasses import dataclass
from math import pi
from typing import Final

import numpy as np
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    CorrectionTrace,
    FilterHyperparameters,
    TraceBuilder,
)
from studies.dp_gam import N_BANDS

HALF_HOURS_PER_DAY: Final[int] = 48
N_REGIMES: Final[int] = 3
NORMAL, SHIFT, OUTLIER = 0, 1, 2
SHIFT_MULTIPLIER: Final[float] = 100.0
OUTLIER_MULTIPLIER: Final[float] = 100.0
LEVEL_VARIANCE_MULTIPLIER: Final[np.ndarray] = np.array([1.0, SHIFT_MULTIPLIER, 1.0])
NOISE_MULTIPLIER: Final[np.ndarray] = np.array([1.0, 1.0, OUTLIER_MULTIPLIER])
PRIOR_PSEUDO_COUNTS: Final[float] = 200.0
PRIOR_TRANSITION: Final[np.ndarray] = np.array(
    [
        [1.0 - 1.0 / 2000.0 - 1.0 / 2000.0, 1.0 / 2000.0, 1.0 / 2000.0],
        [1.0 / 96.0, 1.0 - 1.0 / 96.0, 0.0],
        [1.0 / 6.0, 0.0, 1.0 - 1.0 / 6.0],
    ]
)
"""Mean of the Dirichlet prior on each row: normal to shift 1 in 2,000 half-hours, shift to normal
1 in 96, normal to outlier 1 in 2,000, outlier to normal 1 in 6."""
EM_ITERATIONS: Final[int] = 20
MAX_GAP: Final[int] = 4 * HALF_HOURS_PER_DAY
"""Gaps longer than this many half-hours use the stationary mix of the transition matrix."""
VARIANCE_FLOOR: Final[float] = 1e-12


@dataclass(frozen=True)
class SwitchingResult:
    """The output of one band's regime filter on a batch of streams.

    Attributes:
        level: The collapsed level after each reading, shape (streams, readings).
        anomaly: The collapsed transient after each reading.
        p_bad: The posterior probability of the shift or outlier regime after each reading.
        log_likelihood: The marginal log likelihood of each stream's readings, shape (streams,).
        counts: The expected one-step transition counts, shape (3, 3), summed over streams.
    """

    level: np.ndarray
    anomaly: np.ndarray
    p_bad: np.ndarray
    log_likelihood: np.ndarray
    counts: np.ndarray


def transition_powers(*, transition: np.ndarray) -> np.ndarray:
    """Return `transition ** gap` for every gap from 0 to `MAX_GAP`, shape (MAX_GAP + 1, 3, 3)."""
    powers = np.empty((MAX_GAP + 1, N_REGIMES, N_REGIMES))
    powers[0] = np.eye(N_REGIMES)
    for gap in range(1, MAX_GAP + 1):
        powers[gap] = powers[gap - 1] @ transition
    return powers


def gpb1_band(  # noqa: PLR0915 (one loop body: the three-regime update)
    *,
    positions: np.ndarray,
    values: np.ndarray,
    hyper: FilterHyperparameters,
    transition: np.ndarray,
    multipliers: tuple[float, float] = (SHIFT_MULTIPLIER, OUTLIER_MULTIPLIER),
) -> SwitchingResult:
    """Run the three-regime filter over one band's readings for a batch of streams.

    Args:
        positions: The grid positions of the band's readings, ascending, shape (readings,).
        values: The residual of each stream at each position, shape (streams, readings); `NaN`
            where a stream has no reading.
        hyper: The site's plain-filter hyperparameters (the normal regime).
        transition: The per-half-hour transition matrix, shape (3, 3).
        multipliers: The shift regime's level-step multiplier and the outlier regime's
            observation-variance multiplier; (1, 1) makes every regime the normal one, and a
            multiplier above 1 on the outlier regime also stops its readings updating the state.

    Returns:
        The collapsed states, shift-or-outlier probabilities, likelihoods, and transition counts.
    """
    streams, readings = values.shape
    level_mult = np.array([1.0, multipliers[0], 1.0])
    noise_mult = np.array([1.0, 1.0, multipliers[1]])
    updates = np.array([1.0, 1.0, 0.0 if multipliers[1] > 1.0 else 1.0])
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    powers = transition_powers(transition=transition)
    level = np.zeros(streams)
    anomaly = np.zeros(streams)
    p_ll = np.full(streams, INITIAL_LEVEL_VARIANCE)
    p_lz = np.zeros(streams)
    p_zz = np.full(streams, stationary)
    regime = np.tile(np.array([1.0, 0.0, 0.0]), (streams, 1))
    previous = np.full(streams, int(positions[0]) if readings else 0)
    out_level = np.empty((streams, readings))
    out_anomaly = np.empty((streams, readings))
    out_bad = np.empty((streams, readings))
    log_likelihood = np.zeros(streams)
    counts = np.zeros((N_REGIMES, N_REGIMES))
    for k in range(readings):
        valid = np.isfinite(values[:, k])
        gap = np.where(valid, int(positions[k]) - previous, 0)
        decay = phi**gap
        anomaly_p = anomaly * decay
        p_ll_p = p_ll + q_step * gap
        p_lz_p = p_lz * decay
        p_zz_p = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        prior = np.einsum("si,sij->sj", regime, powers[np.minimum(gap, MAX_GAP)])
        p_ll_r = p_ll_p[:, None] + (level_mult[None, :] - 1.0) * q_step * np.maximum(gap, 1)[
            :, None
        ] * (gap[:, None] > 0)
        innovation = np.where(valid, values[:, k], 0.0) - level - anomaly_p
        variance = p_ll_r + (2.0 * p_lz_p + p_zz_p + noise)[:, None] * 1.0
        variance = variance + (noise_mult[None, :] - 1.0) * noise
        lik = np.exp(-0.5 * innovation[:, None] ** 2 / variance) / np.sqrt(2.0 * pi * variance)
        joint = prior * lik
        total = np.maximum(joint.sum(axis=1), 1e-300)
        posterior = joint / total[:, None]
        gain_l = (p_ll_r + p_lz_p[:, None]) / variance * updates[None, :]
        gain_z = (p_lz_p[:, None] + p_zz_p[:, None]) / variance * updates[None, :]
        level_r = level[:, None] + gain_l * innovation[:, None]
        anomaly_r = anomaly_p[:, None] + gain_z * innovation[:, None]
        new_ll_r = np.maximum(p_ll_r - gain_l * gain_l * variance, VARIANCE_FLOOR)
        new_lz_r = p_lz_p[:, None] - gain_l * gain_z * variance
        new_zz_r = np.maximum(p_zz_p[:, None] - gain_z * gain_z * variance, VARIANCE_FLOOR)
        new_level = (posterior * level_r).sum(axis=1)
        new_anomaly = (posterior * anomaly_r).sum(axis=1)
        d_l = level_r - new_level[:, None]
        d_z = anomaly_r - new_anomaly[:, None]
        new_p_ll = (posterior * (new_ll_r + d_l * d_l)).sum(axis=1)
        new_p_lz = (posterior * (new_lz_r + d_l * d_z)).sum(axis=1)
        new_p_zz = (posterior * (new_zz_r + d_z * d_z)).sum(axis=1)
        one_step = valid & (gap == 1)
        if one_step.any():
            pair = regime[one_step, :, None] * transition[None, :, :] * lik[one_step, None, :]
            counts += (pair / np.maximum(pair.sum(axis=(1, 2), keepdims=True), 1e-300)).sum(axis=0)
        level = np.where(valid, new_level, level)
        anomaly = np.where(valid, new_anomaly, anomaly)
        p_ll = np.where(valid, new_p_ll, p_ll)
        p_lz = np.where(valid, new_p_lz, p_lz)
        p_zz = np.where(valid, new_p_zz, p_zz)
        regime = np.where(valid[:, None], posterior, regime)
        previous = np.where(valid, int(positions[k]), previous)
        log_likelihood += np.where(valid, np.log(total), 0.0)
        out_level[:, k] = level
        out_anomaly[:, k] = anomaly
        out_bad[:, k] = regime[:, SHIFT] + regime[:, OUTLIER]
    return SwitchingResult(
        level=out_level,
        anomaly=out_anomaly,
        p_bad=out_bad,
        log_likelihood=log_likelihood,
        counts=counts,
    )


def run_gpb1(
    *,
    residual: np.ndarray,
    band: np.ndarray,
    hyper: FilterHyperparameters,
    transition: np.ndarray,
    multipliers: tuple[float, float] = (SHIFT_MULTIPLIER, OUTLIER_MULTIPLIER),
) -> tuple[list[SwitchingResult], list[np.ndarray]]:
    """Run the filter over a batch of residual streams, one hour band at a time.

    Args:
        residual: The streams on the grid, shape (streams, grid), `NaN` where there is no reading.
        band: The hour band of each grid row.
        hyper: The plain filter's hyperparameters.
        transition: The per-half-hour transition matrix.
        multipliers: As in `gpb1_band`.

    Returns:
        Each band's result and each band's reading positions.
    """
    results: list[SwitchingResult] = []
    positions: list[np.ndarray] = []
    for band_index in range(N_BANDS):
        rows = np.flatnonzero((band == band_index) & np.isfinite(residual).any(axis=0))
        positions.append(rows)
        results.append(
            gpb1_band(
                positions=rows,
                values=residual[:, rows],
                hyper=hyper,
                transition=transition,
                multipliers=multipliers,
            )
        )
    return results, positions


def switching_builder(*, hyper: FilterHyperparameters, transition: np.ndarray) -> TraceBuilder:
    """Return a trace builder for `pipeline.score_year`: the three-regime filter on one stream."""

    def build(*, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        results, positions = run_gpb1(
            residual=residual[None, :], band=band, hyper=hyper, transition=transition
        )
        # The plain filter keeps only the grid rows a stream has readings at; so does this one.
        kept = [np.isfinite(residual[rows]) for rows in positions]
        return CorrectionTrace(
            kind="kalman",
            positions=tuple(rows[mask] for rows, mask in zip(positions, kept, strict=True)),
            level=tuple(r.level[0][mask] for r, mask in zip(results, kept, strict=True)),
            anomaly=tuple(r.anomaly[0][mask] for r, mask in zip(results, kept, strict=True)),
            phi=hyper.phi,
            log_likelihood=float(sum(r.log_likelihood[0] for r in results)),
        )

    return build


def fit_transition(
    *, residual: np.ndarray, band: np.ndarray, hyper: FilterHyperparameters
) -> np.ndarray:
    """Fit the transition matrix by forward-only expectation-maximisation with a Dirichlet prior.

    Args:
        residual: The out-of-sample residual streams to fit on, shape (streams, grid), with the
            hand-filtered event rows set to `NaN`.
        band: The hour band of each grid row.
        hyper: The plain filter's hyperparameters.

    Returns:
        The fitted per-half-hour transition matrix.
    """
    transition = PRIOR_TRANSITION.copy()
    prior_counts = PRIOR_PSEUDO_COUNTS * PRIOR_TRANSITION
    for _ in range(EM_ITERATIONS):
        results, _ = run_gpb1(residual=residual, band=band, hyper=hyper, transition=transition)
        counts = sum(r.counts for r in results) + prior_counts
        transition = counts / counts.sum(axis=1, keepdims=True)
    return transition
