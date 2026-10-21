"""The latent (signed-flow) Kalman filter of experiment 152, and the folded quantiles.

The state is the two-state level and AR(1) transient of `studies.dp_assimilation`, but it corrects
the signed flow, not the reading. The frozen forecast gives the signed flow `p_hat`, the filtered
flow is `f = p_hat + level + anomaly`, and the predicted reading is `sqrt(f**2 + EPSILON)`. The
observation row is `H = f / y_hat` on both states. Where `|f|` is under `GATE_SIGMAS` predictive
standard deviations of the flow's state (the observation noise is left out, because it holds the
forecast model's own error), the sign is ambiguous and the update is skipped, exactly as a
missing reading is.

This is an experiment-local copy of `studies.dp_assimilation._kalman_band` with a time-varying
observation row; it moves into `packages/studies` only if the arm wins.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    VARIANCE_FLOOR,
    FilterHyperparameters,
)
from studies.dp_gam import N_BANDS

EPSILON: Final[float] = 0.01
"""Added under the square root, in points squared, so the slope is defined at zero flow."""
GATE_SIGMAS: Final[float] = 2.0
HALF_HOURS_PER_DAY: Final[int] = 48


@dataclass(frozen=True)
class LatentTrace:
    """The latent filter's state after each reading, per hour band.

    Attributes:
        positions: For each band, the grid positions of its readings, ascending.
        level: For each band, the level after each reading.
        anomaly: For each band, the transient after each reading.
        updated: For each band, True where the reading updated the state (not gated).
        phi: The transient's AR(1) factor per half-hour.
    """

    positions: tuple[np.ndarray, ...]
    level: tuple[np.ndarray, ...]
    anomaly: tuple[np.ndarray, ...]
    updated: tuple[np.ndarray, ...]
    phi: float


def _band(
    *,
    positions: np.ndarray,
    readings: np.ndarray,
    p_hat: np.ndarray,
    q_hat: np.ndarray,
    hyper: FilterHyperparameters,
    gate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    level, anomaly = 0.0, 0.0
    p_ll, p_lz, p_zz = INITIAL_LEVEL_VARIANCE, 0.0, stationary
    levels = np.empty(len(readings))
    anomalies = np.empty(len(readings))
    updated = np.zeros(len(readings), dtype=bool)
    previous = int(positions[0]) if len(positions) else 0
    for k in range(len(readings)):
        gap = int(positions[k]) - previous
        previous = int(positions[k])
        if gap:
            decay = phi**gap
            anomaly *= decay
            p_ll += q_step * gap
            p_lz *= decay
            p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        flow = p_hat[k] + level + anomaly
        predicted = float(np.sqrt(flow * flow + q_hat[k] ** 2 + EPSILON))
        slope = flow / predicted
        flow_variance = p_ll + 2.0 * p_lz + p_zz
        if not gate or abs(flow) >= GATE_SIGMAS * np.sqrt(flow_variance):
            innovation_variance = slope * slope * flow_variance + noise
            innovation = readings[k] - predicted
            gain_l = slope * (p_ll + p_lz) / innovation_variance
            gain_z = slope * (p_lz + p_zz) / innovation_variance
            level += gain_l * innovation
            anomaly += gain_z * innovation
            p_ll, p_lz, p_zz = (
                max(p_ll - gain_l * gain_l * innovation_variance, VARIANCE_FLOOR),
                p_lz - gain_l * gain_z * innovation_variance,
                max(p_zz - gain_z * gain_z * innovation_variance, VARIANCE_FLOOR),
            )
            updated[k] = True
        levels[k] = level
        anomalies[k] = anomaly
    return levels, anomalies, updated


def run_latent(
    *,
    reading: np.ndarray,
    p_hat: np.ndarray,
    q_hat: np.ndarray,
    band: np.ndarray,
    hyper: FilterHyperparameters,
    gate: bool = True,
) -> LatentTrace:
    """Run the latent filter over a site's readings, one band at a time.

    Args:
        reading: The measured `|S|` on the grid, in points, `NaN` where there is none.
        p_hat: The frozen signed-flow forecast on the grid, in points, `NaN` where there is none.
        q_hat: The frozen reactive power on the grid, in points; zero when `Q = 0`.
        band: The hour band of each grid row, from 0 to 3.
        hyper: The filter's hyperparameters.
        gate: Skip the update where the sign is ambiguous; False gives an ungated filter.

    Returns:
        The trace.
    """
    positions, levels, anomalies, updates = [], [], [], []
    for index in range(N_BANDS):
        rows = np.flatnonzero((band == index) & np.isfinite(reading) & np.isfinite(p_hat))
        lv, an, up = _band(
            positions=rows,
            readings=reading[rows],
            p_hat=p_hat[rows],
            q_hat=q_hat[rows],
            hyper=hyper,
            gate=gate,
        )
        positions.append(rows)
        levels.append(lv)
        anomalies.append(an)
        updates.append(up)
    return LatentTrace(
        positions=tuple(positions),
        level=tuple(levels),
        anomaly=tuple(anomalies),
        updated=tuple(updates),
        phi=hyper.phi,
    )


def correction_at(
    *, trace: LatentTrace, cutoff: np.ndarray, target: np.ndarray, band: np.ndarray
) -> np.ndarray:
    """Return the signed-flow correction `level + decayed anomaly` of each forecast case.

    Args:
        trace: The latent trace.
        cutoff: Each case's newest usable grid position.
        target: Each case's grid position of the valid time.
        band: Each case's hour band.

    Returns:
        The correction in points; zero where the band has no reading at or before the cutoff.
    """
    total = np.zeros(len(target))
    for index in range(N_BANDS):
        selected = np.flatnonzero(band == index)
        positions = trace.positions[index]
        found = np.searchsorted(positions, cutoff[selected], side="right") - 1
        known = found >= 0
        rows, state = selected[known], found[known]
        ahead = target[rows] - positions[state]
        total[rows] = trace.level[index][state] + trace.phi**ahead * trace.anomaly[index][state]
    return total


def folded_quantiles(
    *,
    flow: np.ndarray,
    reactive: np.ndarray,
    residual_pool: dict[tuple[int, int], np.ndarray],
    band: np.ndarray,
    quarter: np.ndarray,
    levels: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Return quantiles of `hypot(flow + r, reactive)`, with `r` drawn from the row's residual pool.

    Args:
        flow: The corrected signed flow of each row, in points.
        reactive: The frozen reactive power of each row, in points.
        residual_pool: Signed residuals of the signed flow, keyed by (band, quarter).
        band: Each row's hour band.
        quarter: Each row's quarter.
        levels: The quantile levels, shape (11,).
        n_samples: Samples per row.
        seed: The generator's seed.

    Returns:
        Shape (rows, len(levels)), ascending along the second axis.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((len(flow), len(levels)))
    for (b, q), pool in residual_pool.items():
        rows = np.flatnonzero((band == b) & (quarter == q))
        if not len(rows):
            continue
        draws = rng.choice(pool, size=(len(rows), n_samples))
        out[rows] = np.quantile(
            np.hypot(flow[rows, None] + draws, reactive[rows, None]), levels, axis=1
        ).T
    return out
