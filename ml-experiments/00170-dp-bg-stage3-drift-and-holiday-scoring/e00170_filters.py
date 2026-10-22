"""The filters of experiment 136, copied for experiment 170.

They are GAM-K's two-state filter with an optional online variance.

`filter_band` runs the same two-state filter as `studies.dp_assimilation`'s `_kalman_band` over one
hour band's readings, and adds the scalar variant of VIKING ([de Vilmarest and Wintenberger
(2022)](https://arxiv.org/abs/2104.10777)) for the observation variance. The log observation
variance `a` is a Gaussian random walk with a variational posterior `N(a_hat, s)`. At each reading
the filter runs `N_ROUNDS` rounds. Each round makes the Kalman update from the predicted state with
observation variance `exp(a_hat - s / 2)`, then takes `s` from Equation 6 and `a_hat` from Equation
7 of the paper, both on the updated residual and the updated state variance. With `rho_a` and the
start `s` both 0, the filter has a constant observation variance and reproduces
`studies.dp_assimilation.run_kalman`.

`filter_stream` runs the four hour bands of a residual stream and returns a `CorrectionTrace` plus
the per-reading log observation variance and the trailing root mean square of the one-step
innovations, which the tail scalings of the VK and TRAIL arms read.
"""

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Final

import numpy as np
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    MIN_SIGMA_EPS,
    VARIANCE_FLOOR,
    CorrectionTrace,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_gam import N_BANDS

N_ROUNDS: Final[int] = 2
"""The paper's default number of alternating rounds per reading."""
A_CLIP_MULTIPLE: Final[float] = 3.0
"""`Ma` of Equation 7 is this multiple of the posterior variance of the log variance."""
TRAILING_READINGS: Final[int] = 84
"""A trailing window of 7 days of readings in a band (12 half-hours a day per band)."""


@dataclass(frozen=True)
class FilterSettings:
    """One setting of the two-state filter.

    Attributes:
        q: The level's random-walk variance per day, in points squared.
        phi: The transient's AR(1) factor per half-hour.
        sigma_z: The transient's innovation standard deviation, in points.
        sigma_eps: The observation standard deviation, in points (the start value when
            `rho_a` is above 0).
        rho_a: The variance of the log observation variance's random walk per half-hour; 0 holds
            the observation variance constant.
    """

    q: float
    phi: float
    sigma_z: float
    sigma_eps: float
    rho_a: float = 0.0


@dataclass(frozen=True)
class BandRun:
    """The filter's output after each reading of one band.

    Attributes:
        level: The level after each reading.
        anomaly: The transient after each reading.
        log_variance: The posterior mean of the log observation variance after each reading.
        innovation_rms: The root mean square of the one-step innovations over the newest
            `TRAILING_READINGS` readings, up to and including each reading.
    """

    level: np.ndarray
    anomaly: np.ndarray
    log_variance: np.ndarray
    innovation_rms: np.ndarray


def _update(
    *, covariance: tuple[float, float, float], noise: float, innovation: float
) -> tuple[float, float, tuple[float, float, float]]:
    """Return the level and transient increments and the posterior covariance of one Kalman update.

    Args:
        covariance: The predicted (level, cross, transient) variances.
        noise: The observation variance.
        innovation: The reading minus the predicted level plus transient.

    Returns:
        The level increment, the transient increment, and the posterior covariance.
    """
    p_ll, p_lz, p_zz = covariance
    variance = p_ll + 2.0 * p_lz + p_zz + noise
    gain_l = (p_ll + p_lz) / variance
    gain_z = (p_lz + p_zz) / variance
    posterior = (
        max(p_ll - gain_l * gain_l * variance, VARIANCE_FLOOR),
        p_lz - gain_l * gain_z * variance,
        max(p_zz - gain_z * gain_z * variance, VARIANCE_FLOOR),
    )
    return gain_l * innovation, gain_z * innovation, posterior


def _variance_step(
    *,
    a_hat: float,
    s: float,
    rho_a: float,
    gap: int,
    covariance: tuple[float, float, float],
    innovation: float,
) -> tuple[float, float, float, float, tuple[float, float, float]]:
    """Run the `N_ROUNDS` rounds of one reading and return the new variance posterior and update.

    Args:
        a_hat: The posterior mean of the log observation variance before the reading.
        s: Its posterior variance before the reading.
        rho_a: The random-walk variance per half-hour.
        gap: The half-hours since the band's previous reading.
        covariance: The predicted state covariance.
        innovation: The reading minus the predicted level plus transient.

    Returns:
        The new `a_hat` and `s`, the level and transient increments, and the posterior covariance.
    """
    s_prior = s + rho_a * max(gap, 1)
    clip = A_CLIP_MULTIPLE * s
    a_round, s_round = a_hat, s_prior
    for _ in range(N_ROUNDS):
        noise = max(exp(a_round - s_round / 2.0), MIN_SIGMA_EPS**2)
        d_level, d_anomaly, posterior = _update(
            covariance=covariance, noise=noise, innovation=innovation
        )
        if s_prior <= 0.0:
            return a_hat, s, d_level, d_anomaly, posterior
        energy = (innovation - d_level - d_anomaly) ** 2 + posterior[0] + 2.0 * posterior[1]
        energy += posterior[2]
        s_new = 1.0 / (1.0 / s_prior + 0.5 * energy * exp(-a_round))
        gradient = energy * exp(-a_hat + s_new / 2.0) - 1.0
        curvature = 1.0 / s_prior + 0.5 * energy * exp(-a_hat + s_new / 2.0 + clip)
        a_round = a_hat + max(min(0.5 * gradient / curvature, clip), -clip)
        s_round = s_new
    return a_round, s_round, d_level, d_anomaly, posterior


def filter_band(*, positions: np.ndarray, values: np.ndarray, settings: FilterSettings) -> BandRun:
    """Run the filter over one band's readings.

    Args:
        positions: The grid positions of the band's readings, ascending.
        values: The residual at each position, in points.
        settings: The filter's setting.

    Returns:
        The state after each reading.
    """
    phi = settings.phi
    stationary = settings.sigma_z**2 / (1.0 - phi**2)
    level, anomaly = 0.0, 0.0
    covariance = (INITIAL_LEVEL_VARIANCE, 0.0, stationary)
    a_hat = log(settings.sigma_eps**2)
    s = 1.0 if settings.rho_a > 0.0 else 0.0
    n = len(values)
    levels, anomalies = np.empty(n), np.empty(n)
    log_variance, rms, squares = np.empty(n), np.empty(n), np.empty(n)
    previous = int(positions[0]) if n else 0
    for k in range(n):
        gap = int(positions[k]) - previous
        previous = int(positions[k])
        p_ll, p_lz, p_zz = covariance
        if gap:
            decay = phi**gap
            anomaly *= decay
            p_ll += settings.q / HALF_HOURS_PER_DAY * gap
            p_lz *= decay
            p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        innovation = float(values[k]) - level - anomaly
        squares[k] = innovation * innovation
        a_hat, s, d_level, d_anomaly, covariance = _variance_step(
            a_hat=a_hat,
            s=s,
            rho_a=settings.rho_a,
            gap=gap,
            covariance=(p_ll, p_lz, p_zz),
            innovation=innovation,
        )
        level += d_level
        anomaly += d_anomaly
        levels[k], anomalies[k], log_variance[k] = level, anomaly, a_hat
        rms[k] = sqrt(squares[max(0, k - TRAILING_READINGS + 1) : k + 1].mean())
    return BandRun(level=levels, anomaly=anomalies, log_variance=log_variance, innovation_rms=rms)


@dataclass(frozen=True)
class StreamRun:
    """The filter's run over a stream: the correction trace and the tail-scaling inputs.

    Attributes:
        trace: The state after each reading, per band.
        log_variance: Per band, the log observation variance after each reading.
        innovation_rms: Per band, the trailing root mean square of the innovations.
    """

    trace: CorrectionTrace
    log_variance: tuple[np.ndarray, ...]
    innovation_rms: tuple[np.ndarray, ...]


def filter_stream(*, residual: np.ndarray, band: np.ndarray, settings: FilterSettings) -> StreamRun:
    """Run the filter over a site's residual stream, one band at a time.

    Args:
        residual: Observed minus forecast on the grid, in points, `NaN` where there is no reading.
        band: The hour band of each grid row.
        settings: The filter's setting.

    Returns:
        The run.
    """
    positions, runs = [], []
    for index in range(N_BANDS):
        rows = np.flatnonzero((band == index) & np.isfinite(residual))
        positions.append(rows)
        runs.append(filter_band(positions=rows, values=residual[rows], settings=settings))
    trace = CorrectionTrace(
        kind="kalman",
        positions=tuple(positions),
        level=tuple(run.level for run in runs),
        anomaly=tuple(run.anomaly for run in runs),
        phi=settings.phi,
        log_likelihood=0.0,
    )
    return StreamRun(
        trace=trace,
        log_variance=tuple(run.log_variance for run in runs),
        innovation_rms=tuple(run.innovation_rms for run in runs),
    )


def value_at_cutoffs(
    *,
    positions: tuple[np.ndarray, ...],
    values: tuple[np.ndarray, ...],
    cutoff: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """Return each case's per-reading value from its band's newest reading at or before the cutoff.

    Args:
        positions: Per band, the grid positions of the readings.
        values: Per band, the value after each reading.
        cutoff: Each case's newest usable grid position.
        band: Each case's hour band (the target's).

    Returns:
        One value per case, `NaN` where the band has no reading yet.
    """
    out = np.full(len(cutoff), np.nan)
    for index in range(N_BANDS):
        selected = np.flatnonzero(band == index)
        found = np.searchsorted(positions[index], cutoff[selected], side="right") - 1
        known = found >= 0
        out[selected[known]] = values[index][found[known]]
    return out
