"""The two-regime Kalman filter of S-KF: GAM-K's level and transient, with a switched regime.

**Each hour band runs GAM-K's two-state filter (a random-walk level and an AR(1) transient). S-KF
adds a second regime, "switched", in which the level's random-walk variance is a multiple of the
fitted one.** Before each reading the filter predicts under both regimes from the previous mixed
posterior, scores the reading under each, and weights the two updated states by

    w = p * L_switched / (p * L_switched + (1 - p) * L_normal),

where `p` is the prior probability of the switched regime at that reading (the detector's signal)
and `L` is the regime's innovation likelihood. The mixed posterior has the weighted mean state and
the moment-matched covariance, so the next prediction starts from one state. With `p = 0` at every
reading the weight is 0 and the filter is exactly `studies.dp_assimilation.run_kalman`'s.

Nothing here reads a switching log or a detector: the caller supplies `p` per grid position.
"""

from math import pi

import numpy as np
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    VARIANCE_FLOOR,
    CorrectionTrace,
    FilterHyperparameters,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_gam import N_BANDS


def _band(
    *,
    positions: np.ndarray,
    values: np.ndarray,
    prior: np.ndarray,
    hyper: FilterHyperparameters,
    multiplier: float,
    update_regime: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the two-regime filter over one band's readings.

    Args:
        positions: The grid positions of the band's readings, ascending.
        values: The residual at each position, in points.
        prior: The prior probability of the switched regime at each reading, in [0, 1].
        hyper: The site's hyperparameters.
        multiplier: The switched regime's level variance as a multiple of the fitted `q`.
        update_regime: False to use the prior as the weight (no likelihood update).

    Returns:
        The mixed level, the mixed transient, and the posterior probability of the switched
        regime, after each reading.
    """
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    level, anomaly = 0.0, 0.0
    p_ll, p_lz, p_zz = INITIAL_LEVEL_VARIANCE, 0.0, stationary
    levels = np.empty(len(values))
    anomalies = np.empty(len(values))
    posterior = np.empty(len(values))
    previous = int(positions[0]) if len(positions) else 0
    for k in range(len(values)):
        gap = int(positions[k]) - previous
        previous = int(positions[k])
        decay = phi**gap
        anomaly *= decay
        base_lz = p_lz * decay
        base_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        updated = []
        for regime_multiplier in (1.0, multiplier):
            m_ll = p_ll + regime_multiplier * q_step * gap
            innovation_variance = m_ll + 2.0 * base_lz + base_zz + noise
            innovation = values[k] - level - anomaly
            likelihood = np.exp(-0.5 * innovation * innovation / innovation_variance) / np.sqrt(
                2.0 * pi * innovation_variance
            )
            gain_l = (m_ll + base_lz) / innovation_variance
            gain_z = (base_lz + base_zz) / innovation_variance
            updated.append(
                (
                    likelihood,
                    level + gain_l * innovation,
                    anomaly + gain_z * innovation,
                    max(m_ll - gain_l * gain_l * innovation_variance, VARIANCE_FLOOR),
                    base_lz - gain_l * gain_z * innovation_variance,
                    max(base_zz - gain_z * gain_z * innovation_variance, VARIANCE_FLOOR),
                )
            )
        p = float(prior[k])
        weight_s = p * updated[1][0]
        weight_n = (1.0 - p) * updated[0][0]
        total = weight_s + weight_n
        w = weight_s / total if (total > 0.0 and update_regime) else p
        n_state, s_state = updated[0], updated[1]
        level = (1.0 - w) * n_state[1] + w * s_state[1]
        anomaly = (1.0 - w) * n_state[2] + w * s_state[2]
        d_n = (n_state[1] - level, n_state[2] - anomaly)
        d_s = (s_state[1] - level, s_state[2] - anomaly)
        p_ll = (1.0 - w) * (n_state[3] + d_n[0] ** 2) + w * (s_state[3] + d_s[0] ** 2)
        p_lz = (1.0 - w) * (n_state[4] + d_n[0] * d_n[1]) + w * (s_state[4] + d_s[0] * d_s[1])
        p_zz = (1.0 - w) * (n_state[5] + d_n[1] ** 2) + w * (s_state[5] + d_s[1] ** 2)
        levels[k], anomalies[k], posterior[k] = level, anomaly, w
    return levels, anomalies, posterior


def run_two_regime(
    *,
    residual: np.ndarray,
    band: np.ndarray,
    prior: np.ndarray,
    hyper: FilterHyperparameters,
    multiplier: float,
    update_regime: bool = True,
) -> tuple[CorrectionTrace, tuple[np.ndarray, ...]]:
    """Run the two-regime filter over a residual stream, one band at a time.

    Args:
        residual: Observed minus forecast on the grid, in points, `NaN` where there is no reading.
        band: The hour band of each grid row.
        prior: The prior probability of the switched regime at each grid row.
        hyper: The site's hyperparameters.
        multiplier: The switched regime's level variance as a multiple of the fitted `q`.
        update_regime: False to use the prior as the weight (no likelihood update).

    Returns:
        The trace (as `run_kalman` returns, with the likelihood left at zero) and, per band, the
        posterior probability of the switched regime after each reading.
    """
    positions, levels, anomalies, posteriors = [], [], [], []
    for band_index in range(N_BANDS):
        chosen = np.flatnonzero((band == band_index) & np.isfinite(residual))
        band_levels, band_anomalies, band_posterior = _band(
            positions=chosen,
            values=residual[chosen],
            prior=prior[chosen],
            hyper=hyper,
            multiplier=multiplier,
            update_regime=update_regime,
        )
        positions.append(chosen)
        levels.append(band_levels)
        anomalies.append(band_anomalies)
        posteriors.append(band_posterior)
    trace = CorrectionTrace(
        kind="kalman",
        positions=tuple(positions),
        level=tuple(levels),
        anomaly=tuple(anomalies),
        phi=hyper.phi,
        log_likelihood=0.0,
    )
    return trace, tuple(posteriors)
