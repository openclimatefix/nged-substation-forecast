"""F-mix of experiment 126: a Kalman filter whose noise differs by day regime.

**Each grid half-hour belongs to one of three regimes**, fixed from the calendar and the weather
before any fit: a holiday (any half-hour inside a named holiday period of `studies.dp_calendar`),
a cold-snap day (a local day whose mean ERA5 temperature is below the 10th percentile of the
2023-24 winter's daily means at the substation's cell, and not a holiday), and a mild day (every
other half-hour). The 2023-24 winter is the tuning winter, so no later weather sets the cut-off.

The mild regime uses the plain filter's level step variance and observation variance. The cold-snap
and holiday regimes multiply both by their own factor, so F-mix adds four factors to the plain
filter's four hyperparameters. The factors are fitted by marginal likelihood on the tuning year's
out-of-sample residuals, under a standard-normal prior on their logarithms, with the plain
hyperparameters held at T-all's. Equal factors of one reproduce the plain filter.
"""

from dataclasses import dataclass
from math import log, pi
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import minimize
from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    CorrectionTrace,
    FilterHyperparameters,
    TraceBuilder,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS

HALF_HOURS_PER_DAY: Final[int] = 48
MILD, COLD_SNAP, HOLIDAY = 0, 1, 2
COLD_SNAP_PERCENTILE: Final[float] = 10.0
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
TUNING_WINTER: Final[tuple[str, str]] = ("2023-11-01", "2024-03-31")
VARIANCE_FLOOR: Final[float] = 1e-12
FACTOR_PRIOR_SPREAD: Final[float] = 1.0
"""Standard deviation of the normal prior on each log factor."""
N_FACTORS: Final[int] = 4
"""Cold-snap level step, cold-snap observation, holiday level step, holiday observation."""


@dataclass(frozen=True)
class RegimeMap:
    """The regime of every grid half-hour at one substation.

    Attributes:
        regime: `MILD`, `COLD_SNAP`, or `HOLIDAY` per grid row, int8.
        cold_threshold_c: The daily-mean temperature below which a day is a cold snap.
    """

    regime: np.ndarray
    cold_threshold_c: float


def regime_map(*, cache: FeatureCache, site: str) -> RegimeMap:
    """Return the regime of each grid half-hour at one substation.

    Args:
        cache: The feature cache.
        site: The substation's label.

    Returns:
        The regime map.
    """
    features = cache.sites[site]
    dates = pl.Series(cache.calendar.local_date)
    frame = pl.DataFrame({"date": dates, "temperature": features.temperature_c})
    daily = frame.group_by("date").agg(pl.col("temperature").mean()).sort("date")
    winter = daily.filter(
        pl.col("date").is_between(
            pl.lit(TUNING_WINTER[0]).str.to_date(), pl.lit(TUNING_WINTER[1]).str.to_date()
        )
    )
    threshold = float(
        np.percentile(winter["temperature"].drop_nulls().to_numpy(), COLD_SNAP_PERCENTILE)
    )
    cold_days = set(daily.filter(pl.col("temperature") < threshold)["date"].to_list())
    cold = np.array([d in cold_days for d in dates.to_list()])
    regime = np.full(len(cache.end_times), MILD, dtype=np.int8)
    regime[cold] = COLD_SNAP
    regime[cache.calendar.holiday_period >= 0] = HOLIDAY
    return RegimeMap(regime=regime, cold_threshold_c=threshold)


def _kalman_band(
    *,
    positions: np.ndarray,
    values: np.ndarray,
    q_multiplier: np.ndarray,
    noise_multiplier: np.ndarray,
    hyper: FilterHyperparameters,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the two-state filter over one band's readings with a noise multiplier per reading.

    The code follows `studies.dp_assimilation`'s band filter; the level step variance over a gap
    is multiplied by the multiplier of the reading the gap ends at, and the observation variance
    by the same reading's second multiplier.
    """
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    level, anomaly = 0.0, 0.0
    p_ll, p_lz, p_zz = INITIAL_LEVEL_VARIANCE, 0.0, stationary
    levels = np.empty(len(values))
    anomalies = np.empty(len(values))
    log_likelihood = 0.0
    previous = int(positions[0]) if len(positions) else 0
    for k in range(len(values)):
        gap = int(positions[k]) - previous
        previous = int(positions[k])
        if gap:
            decay = phi**gap
            anomaly *= decay
            p_ll += q_step * q_multiplier[k] * gap
            p_lz *= decay
            p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        variance = p_ll + 2.0 * p_lz + p_zz + noise * noise_multiplier[k]
        innovation = values[k] - level - anomaly
        log_likelihood -= 0.5 * (log(2.0 * pi * variance) + innovation * innovation / variance)
        gain_l = (p_ll + p_lz) / variance
        gain_z = (p_lz + p_zz) / variance
        level += gain_l * innovation
        anomaly += gain_z * innovation
        p_ll, p_lz, p_zz = (
            max(p_ll - gain_l * gain_l * variance, VARIANCE_FLOOR),
            p_lz - gain_l * gain_z * variance,
            max(p_zz - gain_z * gain_z * variance, VARIANCE_FLOOR),
        )
        levels[k] = level
        anomalies[k] = anomaly
    return levels, anomalies, log_likelihood


def multipliers_by_row(*, regimes: np.ndarray, log_factors: np.ndarray) -> np.ndarray:
    """Return the level step and observation multipliers of each row, shape (rows, 2)."""
    table = np.ones((3, 2))
    table[COLD_SNAP] = np.exp(log_factors[0:2])
    table[HOLIDAY] = np.exp(log_factors[2:4])
    return table[regimes]


def run_regime_kalman(
    *,
    residual: np.ndarray,
    band: np.ndarray,
    regimes: np.ndarray,
    log_factors: np.ndarray,
    hyper: FilterHyperparameters,
) -> CorrectionTrace:
    """Run the regime-dependent filter over a residual stream, one band at a time."""
    multipliers = multipliers_by_row(regimes=regimes, log_factors=log_factors)
    positions, levels, anomalies = [], [], []
    log_likelihood = 0.0
    for band_index in range(N_BANDS):
        rows = np.flatnonzero((band == band_index) & np.isfinite(residual))
        band_levels, band_anomalies, band_likelihood = _kalman_band(
            positions=rows,
            values=residual[rows],
            q_multiplier=multipliers[rows, 0],
            noise_multiplier=multipliers[rows, 1],
            hyper=hyper,
        )
        positions.append(rows)
        levels.append(band_levels)
        anomalies.append(band_anomalies)
        log_likelihood += band_likelihood
    return CorrectionTrace(
        kind="kalman",
        positions=tuple(positions),
        level=tuple(levels),
        anomaly=tuple(anomalies),
        phi=hyper.phi,
        log_likelihood=log_likelihood,
    )


def fit_factors(
    *,
    residual: np.ndarray,
    band: np.ndarray,
    regimes: np.ndarray,
    hyper: FilterHyperparameters,
) -> np.ndarray:
    """Fit the four log factors by maximum a posteriori on one substation's tuning-year residuals.

    Args:
        residual: The out-of-sample residual stream of the tuning year on the grid.
        band: The hour band of each grid row.
        regimes: The regime of each grid row.
        hyper: The plain filter's hyperparameters.

    Returns:
        The four log factors.
    """

    def negative(log_factors: np.ndarray) -> float:
        trace = run_regime_kalman(
            residual=residual, band=band, regimes=regimes, log_factors=log_factors, hyper=hyper
        )
        prior = 0.5 * float(np.sum((log_factors / FACTOR_PRIOR_SPREAD) ** 2))
        return -trace.log_likelihood + prior

    simplex = np.vstack([np.zeros(N_FACTORS), 0.5 * np.eye(N_FACTORS)])
    result = minimize(
        negative,
        np.zeros(N_FACTORS),
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 0.02, "fatol": 0.05, "maxiter": 120},
    )
    return np.asarray(result.x)


def regime_builder(
    *, regimes: np.ndarray, log_factors: np.ndarray, hyper: FilterHyperparameters
) -> TraceBuilder:
    """Return the trace builder for `pipeline.score_year`."""

    def build(*, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        return run_regime_kalman(
            residual=residual, band=band, regimes=regimes, log_factors=log_factors, hyper=hyper
        )

    return build
