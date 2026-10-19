"""Recent-level assimilation for the differentiable-physics (DP) series: the DP's analogue of lags.

**XGBoost sees the latest observed power through its lag columns. A frozen DP forecast cannot.**
Between refits the DP misses a permanent transfer, a new large load, and a behaviour change, and
it misses a period when its weather response is off. This module corrects a frozen forecast with
the residuals (observed minus forecast) that are old enough to be known at the initialisation.

**The Kalman filter.** Per site and per hour band, the residual stream is the sum of two hidden
states plus noise:

- a persistent level `l`, a random walk with variance `q` per day (drift, permanent transfers, new
  loads);
- a transient anomaly `z`, an AR(1) process with factor `phi` per half-hour and innovation variance
  `sigma_z**2` (temporary switching, a misjudged cold spell);
- an observation `l + z + noise` with noise variance `sigma_eps**2`.

Time is wall-clock half-hours, so `z` decays across the band's daily gap and across the delay, and
a missing reading skips the update. The forecast for a target `h` half-hours after the band's
newest usable reading is the frozen forecast `+ l + phi**h * z`.

**The information cut.** At initialisation `tau` the filter sees only residuals whose half-hour
ends no later than `tau` minus `DEFAULT_DELAY_HOURS`. The filter is causal, so its state after each
reading depends on that reading and the earlier ones only, and a forecast looks up the state after
the newest usable reading. No switching mask enters the filter, because the hand-filtered labels
are hindsight; the masks apply to fitting the DP and to scoring.

**The simple fix.** `run_ewma` keeps an exponentially weighted mean of the raw residuals per band,
held flat over every lead, under the same cut. It is a normalised weighted mean with decay by
wall-clock gap, so it equals a local-level Kalman filter only asymptotically on a regular grid, and
not exactly.

**Across a refit,** `stitch_out_of_sample_residuals` joins each fit's residuals after its own cut
into one stream, so the filter's state carries forward and never re-runs on a fit's in-sample
residuals, which would pull the level towards zero at the cut. The replica's filter
(`post_filter_lead_spread`) starts cold on its year and leaves out the first `REPLICA_BURN_IN_DAYS`
days, whereas the scored filter is warm.

**The hyperparameters** (log `q`, logit `phi`, log `sigma_z`, log `sigma_eps`) are fitted per site
by marginal likelihood under a normal prior shared across sites, whose mean and spread are updated
from the sites' own fits (empirical Bayes).

**The spread of a corrected arm** is the replica's post-filter residual quantiles per hour band,
calendar quarter, and lead band, centred on their median (`LeadSpread`).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import exp, log, pi
from typing import Final, Literal, Protocol

import numpy as np
from scipy.optimize import minimize

from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_gam import N_BANDS, ReplicaResiduals, ResidualSpread

DEFAULT_DELAY_HOURS: Final[float] = 3.5
"""The telemetry delay of experiments 87 and 104: 3 hours plus the half-hour a reading covers.

It is an assumption, not a measured value.
"""

INITIALISATION_HOUR_UTC: Final[int] = 9
"""The hour of the daily initialisation, 09:00 UTC."""

REPLICA_BURN_IN_DAYS: Final[int] = 14
"""The first days of a replica year are left out of the spread, while the filter converges."""

INITIAL_LEVEL_VARIANCE: Final[float] = 100.0
"""The prior variance of the level at the first reading, in points squared."""

EWMA_HALF_LIFE_DAYS: Final[float] = 14.0
"""The half-life of the exponentially weighted mean, roughly BG2's 14-day correction."""

LEAD_BAND_UPPER_DAYS: Final[tuple[int, ...]] = (1, 3, 7)
"""The last day ahead of each lead band: day 1, days 2 to 3, and days 4 to 7."""

N_LEAD_BANDS: Final[int] = len(LEAD_BAND_UPPER_DAYS)
"""The number of lead bands."""

PRIOR_START: Final[tuple[float, float, float, float]] = (log(0.05), 4.2, log(0.35), log(1.0))
"""The start of the shared prior mean: a level step of 0.22 points a day, a transient with a
half-life of about a day (phi = 0.985 per half-hour), a transient innovation of 0.35 points, and
observation noise of 1 point."""

PRIOR_SPREAD: Final[tuple[float, float, float, float]] = (1.5, 1.0, 1.0, 1.0)
"""The starting standard deviation of the shared prior, which later rounds replace."""

MIN_PRIOR_SPREAD: Final[float] = 0.25
"""The shared prior's spread never shrinks below this, so the prior stays a prior with few sites."""

MIN_SIGMA_EPS: Final[float] = 0.05
"""The smallest observation noise standard deviation in points, which keeps the update stable."""

BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (-12.0, 6.0),
    (-2.0, 8.0),
    (-12.0, 6.0),
    (log(MIN_SIGMA_EPS), 6.0),
)
"""Bounds on the four unconstrained hyperparameters, which keep the optimiser off degenerate
corners: log `q`, logit `phi`, log `sigma_z`, and log `sigma_eps`."""

VARIANCE_FLOOR: Final[float] = 1e-12
"""A state variance is never allowed below this after an update, so rounding cannot make it
negative."""

N_EMPIRICAL_BAYES_ROUNDS: Final[int] = 3
"""How many times the shared prior is refitted from the sites' own fits."""

CorrectionKindType = Literal["kalman", "ewma"]
"""Which correction a trace holds."""


@dataclass(frozen=True)
class FilterHyperparameters:
    """The four hyperparameters of one site's filter, on the unconstrained scale.

    Attributes:
        log_q: Log of the level's random-walk variance per day, in points squared.
        logit_phi: Logit of the transient's AR(1) factor per half-hour.
        log_sigma_z: Log of the transient's innovation standard deviation, in points.
        log_sigma_eps: Log of the observation noise standard deviation, in points.
    """

    log_q: float
    logit_phi: float
    log_sigma_z: float
    log_sigma_eps: float

    @property
    def q(self) -> float:
        """The level's random-walk variance per day, in points squared."""
        return exp(self.log_q)

    @property
    def phi(self) -> float:
        """The transient's AR(1) factor per half-hour."""
        return 1.0 / (1.0 + exp(-self.logit_phi))

    @property
    def sigma_z(self) -> float:
        """The transient's innovation standard deviation, in points."""
        return exp(self.log_sigma_z)

    @property
    def sigma_eps(self) -> float:
        """The observation noise standard deviation, in points."""
        return exp(self.log_sigma_eps)

    def as_array(self) -> np.ndarray:
        """Return the four unconstrained values as an array."""
        return np.array([self.log_q, self.logit_phi, self.log_sigma_z, self.log_sigma_eps])

    @staticmethod
    def from_array(*, values: np.ndarray) -> FilterHyperparameters:
        """Build the hyperparameters from four unconstrained values."""
        return FilterHyperparameters(
            log_q=float(values[0]),
            logit_phi=float(values[1]),
            log_sigma_z=float(values[2]),
            log_sigma_eps=float(values[3]),
        )


@dataclass(frozen=True)
class CorrectionTrace:
    """The state of a correction after each reading, per hour band.

    Attributes:
        kind: Which correction the trace holds.
        positions: For each band, the grid positions of its readings, ascending.
        level: For each band, the level after each reading (the weighted mean for the EWMA).
        anomaly: For each band, the transient after each reading (zeros for the EWMA).
        phi: The transient's AR(1) factor per half-hour (1 for the EWMA, which is held flat).
        log_likelihood: The marginal log likelihood of the readings (0 for the EWMA).
    """

    kind: CorrectionKindType
    positions: tuple[np.ndarray, ...]
    level: tuple[np.ndarray, ...]
    anomaly: tuple[np.ndarray, ...]
    phi: float
    log_likelihood: float


class TraceBuilder(Protocol):
    """A function that builds a correction trace from a residual stream.

    `run_ewma` and `functools.partial(run_kalman, hyper=hyper)` both satisfy it.
    """

    def __call__(self, *, residual: np.ndarray, band: np.ndarray) -> CorrectionTrace:
        """Build the trace.

        Args:
            residual: Observed minus forecast on the grid, in points, `NaN` where there is none.
            band: The hour band of each grid row.

        Returns:
            The trace.
        """
        ...


@dataclass(frozen=True)
class Correction:
    """The correction at each forecast case.

    Attributes:
        total: The correction to add to the frozen forecast, in points.
        level: The persistent part (the EWMA value for the EWMA).
        anomaly: The transient part after decay (zeros for the EWMA).
    """

    total: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray


def _band_readings(
    *, residual: np.ndarray, band: np.ndarray, band_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the grid positions and values of a band's finite readings."""
    positions = np.flatnonzero((band == band_index) & np.isfinite(residual))
    return positions, residual[positions]


def _kalman_band(
    *, positions: np.ndarray, values: np.ndarray, hyper: FilterHyperparameters
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the two-state filter over one band's readings.

    The state moves by the exact discretisation of the random walk and the AR(1) over the gap in
    half-hours between readings, so a band's daily gap needs no placeholder rows.

    Args:
        positions: The grid positions of the band's readings, ascending.
        values: The residual at each position, in points.
        hyper: The site's hyperparameters.

    Returns:
        The level and the transient after each reading, and the log likelihood.
    """
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    sigma_z_sq = hyper.sigma_z**2
    stationary = sigma_z_sq / (1.0 - phi**2)
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
            p_ll += q_step * gap
            p_lz *= decay
            p_zz = decay * decay * p_zz + stationary * (1.0 - decay * decay)
        innovation_variance = p_ll + 2.0 * p_lz + p_zz + noise
        innovation = values[k] - level - anomaly
        log_likelihood -= 0.5 * (
            log(2.0 * pi * innovation_variance) + innovation * innovation / innovation_variance
        )
        gain_l = (p_ll + p_lz) / innovation_variance
        gain_z = (p_lz + p_zz) / innovation_variance
        level += gain_l * innovation
        anomaly += gain_z * innovation
        p_ll, p_lz, p_zz = (
            max(p_ll - gain_l * gain_l * innovation_variance, VARIANCE_FLOOR),
            p_lz - gain_l * gain_z * innovation_variance,
            max(p_zz - gain_z * gain_z * innovation_variance, VARIANCE_FLOOR),
        )
        levels[k] = level
        anomalies[k] = anomaly
    return levels, anomalies, log_likelihood


def run_kalman(
    *, residual: np.ndarray, band: np.ndarray, hyper: FilterHyperparameters
) -> CorrectionTrace:
    """Run the filter over a site's residual stream, one band at a time.

    Args:
        residual: Observed minus forecast on the grid, in points, `NaN` where there is no reading.
            The caller includes the switching rows: the filter has no mask.
        band: The hour band of each grid row, from 0 to 3.
        hyper: The site's hyperparameters.

    Returns:
        The trace: for each band, the state after each reading.
    """
    positions, levels, anomalies = [], [], []
    log_likelihood = 0.0
    for band_index in range(N_BANDS):
        band_positions, values = _band_readings(residual=residual, band=band, band_index=band_index)
        band_levels, band_anomalies, band_likelihood = _kalman_band(
            positions=band_positions, values=values, hyper=hyper
        )
        positions.append(band_positions)
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


def run_ewma(
    *, residual: np.ndarray, band: np.ndarray, half_life_days: float = EWMA_HALF_LIFE_DAYS
) -> CorrectionTrace:
    """Run an exponentially weighted mean of the raw residuals, one band at a time.

    The weight of a reading falls by half every `half_life_days` of wall-clock time, and the
    value after a reading is the weighted mean of all readings up to it.

    Args:
        residual: Observed minus forecast on the grid, in points, `NaN` where there is no reading.
        band: The hour band of each grid row.
        half_life_days: The half-life of the weights.

    Returns:
        The trace, with zero anomaly.
    """
    per_half_hour = 0.5 ** (1.0 / (half_life_days * HALF_HOURS_PER_DAY))
    positions, levels, anomalies = [], [], []
    for band_index in range(N_BANDS):
        band_positions, values = _band_readings(residual=residual, band=band, band_index=band_index)
        value_after = np.empty(len(values))
        weighted_sum, weight = 0.0, 0.0
        previous = int(band_positions[0]) if len(band_positions) else 0
        for k in range(len(values)):
            decay = per_half_hour ** (int(band_positions[k]) - previous)
            previous = int(band_positions[k])
            weighted_sum = weighted_sum * decay + values[k]
            weight = weight * decay + 1.0
            value_after[k] = weighted_sum / weight
        positions.append(band_positions)
        levels.append(value_after)
        anomalies.append(np.zeros(len(values)))
    return CorrectionTrace(
        kind="ewma",
        positions=tuple(positions),
        level=tuple(levels),
        anomaly=tuple(anomalies),
        phi=1.0,
        log_likelihood=0.0,
    )


def usable_cutoff_positions(
    *, end_times: np.ndarray, initialisation: np.ndarray, delay_hours: float
) -> np.ndarray:
    """Return the grid position of the newest half-hour a forecast may see.

    Args:
        end_times: The grid's half-hour ends as `datetime64[us]`, ascending.
        initialisation: Each case's initialisation time as `datetime64[us]`.
        delay_hours: The telemetry delay; a half-hour is usable when it ends at least this long
            before the initialisation.

    Returns:
        Grid positions, or -1 where no half-hour is usable.
    """
    delay = np.timedelta64(round(delay_hours * 3600 * 1_000_000), "us")
    return np.searchsorted(end_times, initialisation - delay, side="right") - 1


def day_ahead_initialisation(*, end_times: np.ndarray, days_ahead: int) -> np.ndarray:
    """Return the initialisation of each valid time `days_ahead` days ahead.

    The initialisation is 09:00 UTC on the UTC day of the half-hour's start, minus `days_ahead`
    days, so `days_ahead=1` gives leads of 15.5 to 39 hours.

    Args:
        end_times: Valid times as the half-hour ends, `datetime64[us]`.
        days_ahead: How many days before the valid day the forecast is made.

    Returns:
        `datetime64[us]` initialisation times.
    """
    start_day = (end_times - np.timedelta64(30, "m")).astype("datetime64[D]")
    return (start_day - np.timedelta64(days_ahead, "D")).astype("datetime64[us]") + np.timedelta64(
        INITIALISATION_HOUR_UTC, "h"
    )


def lead_band_of_days_ahead(*, days_ahead: int) -> int:
    """Return the lead band of a number of days ahead.

    Args:
        days_ahead: How many days before the valid day the forecast is made.

    Returns:
        0 for day 1, 1 for days 2 to 3, and 2 for days 4 to 7.

    Raises:
        ValueError: If `days_ahead` is outside 1 to 7.
    """
    for band_index, upper in enumerate(LEAD_BAND_UPPER_DAYS):
        if 1 <= days_ahead <= upper:
            return band_index
    msg = f"days_ahead must be between 1 and {LEAD_BAND_UPPER_DAYS[-1]}, not {days_ahead}"
    raise ValueError(msg)


def correction_at(
    *,
    trace: CorrectionTrace,
    cutoff: np.ndarray,
    target: np.ndarray,
    band: np.ndarray,
) -> Correction:
    """Look up the correction for forecast cases.

    Each case uses the state after its band's newest reading at or before `cutoff`. A case with
    no such reading gets a correction of zero.

    Args:
        trace: The correction's trace.
        cutoff: Each case's newest usable grid position.
        target: Each case's grid position of the valid time.
        band: Each case's hour band (the target's).

    Returns:
        The correction of each case.
    """
    level = np.zeros(len(target))
    anomaly = np.zeros(len(target))
    for band_index in range(N_BANDS):
        selected = np.flatnonzero(band == band_index)
        positions = trace.positions[band_index]
        found = np.searchsorted(positions, cutoff[selected], side="right") - 1
        known = found >= 0
        rows = selected[known]
        state = found[known]
        level[rows] = trace.level[band_index][state]
        half_hours_ahead = target[rows] - positions[state]
        anomaly[rows] = trace.phi**half_hours_ahead * trace.anomaly[band_index][state]
    return Correction(total=level + anomaly, level=level, anomaly=anomaly)


def site_log_likelihood(
    *, residual: np.ndarray, band: np.ndarray, hyper: FilterHyperparameters
) -> float:
    """Return the marginal log likelihood of a site's residual stream."""
    return run_kalman(residual=residual, band=band, hyper=hyper).log_likelihood


def fit_site_hyperparameters(
    *,
    residual: np.ndarray,
    band: np.ndarray,
    prior_mean: np.ndarray,
    prior_spread: np.ndarray,
    start: np.ndarray,
) -> FilterHyperparameters:
    """Fit one site's hyperparameters by maximum a posteriori under a normal prior.

    Args:
        residual: The site's residual stream on the grid, in points.
        band: The hour band of each grid row.
        prior_mean: The shared prior mean of the four unconstrained values.
        prior_spread: The shared prior standard deviation of each.
        start: Where the optimiser starts.

    Returns:
        The hyperparameters that maximise the marginal likelihood plus the log prior.

    Raises:
        ValueError: If the optimiser does not report success, since a poor fit would pull the
            shared prior of every site.
    """

    def negative_posterior(values: np.ndarray) -> float:
        hyper = FilterHyperparameters.from_array(values=values)
        likelihood = site_log_likelihood(residual=residual, band=band, hyper=hyper)
        prior = -0.5 * float((((values - prior_mean) / prior_spread) ** 2).sum())
        return -(likelihood + prior)

    result = minimize(negative_posterior, x0=start, method="L-BFGS-B", bounds=BOUNDS)
    if not result.success:
        msg = f"the hyperparameter fit did not converge: {result.message}"
        raise ValueError(msg)
    return FilterHyperparameters.from_array(values=result.x)


def fit_filter_hyperparameters(
    *,
    residuals: Mapping[str, np.ndarray],
    band: np.ndarray,
    end_times: np.ndarray | None = None,
    window: tuple[np.datetime64, np.datetime64] | None = None,
) -> dict[str, FilterHyperparameters]:
    """Fit every site's hyperparameters under a prior shared across sites.

    Each round fits every site under the current prior, then replaces the prior's mean and
    spread with the sites' own fits (the spread never below `MIN_PRIOR_SPREAD`). The first round
    uses `PRIOR_START` and `PRIOR_SPREAD`.

    Args:
        residuals: Each site's residual stream on the grid, in points, `NaN` where there is no
            reading; the tuning year's out-of-sample residuals.
        band: The hour band of each grid row.
        end_times: The grid's half-hour ends as `datetime64[us]`, needed with `window`.
        window: If given, the half-open interval (start, end] that every finite residual must lie
            in, which makes "fitted on the tuning year only" a guarantee of this function.

    Returns:
        Each site's fitted hyperparameters.

    Raises:
        ValueError: If there are no sites, `window` is given without `end_times`, or a residual
            lies outside `window`.
    """
    if not residuals:
        msg = "no residual streams to fit the filter on"
        raise ValueError(msg)
    if window is not None and end_times is None:
        msg = "a fitting window needs the grid's end_times"
        raise ValueError(msg)
    if window is not None and end_times is not None:
        for site, stream in residuals.items():
            outside = np.isfinite(stream) & ~((end_times > window[0]) & (end_times <= window[1]))
            if outside.any():
                msg = f"{site} has {int(outside.sum())} residuals outside the fitting window"
                raise ValueError(msg)
    prior_mean = np.array(PRIOR_START)
    prior_spread = np.array(PRIOR_SPREAD)
    fitted: dict[str, FilterHyperparameters] = {}
    for _ in range(N_EMPIRICAL_BAYES_ROUNDS):
        fitted = {
            site: fit_site_hyperparameters(
                residual=stream,
                band=band,
                prior_mean=prior_mean,
                prior_spread=prior_spread,
                start=fitted[site].as_array() if site in fitted else prior_mean,
            )
            for site, stream in residuals.items()
        }
        stacked = np.stack([hyper.as_array() for hyper in fitted.values()])
        prior_mean = stacked.mean(axis=0)
        if len(fitted) > 1:
            prior_spread = np.maximum(stacked.std(axis=0, ddof=1), MIN_PRIOR_SPREAD)
    return fitted


@dataclass(frozen=True)
class LeadSpread:
    """Post-filter residual quantiles per hour band, calendar quarter, and lead band.

    Attributes:
        by_lead_band: One centred `ResidualSpread` per lead band.
    """

    by_lead_band: tuple[ResidualSpread, ...]

    @staticmethod
    def fit(
        *,
        residuals: Sequence[np.ndarray],
        band: Sequence[np.ndarray],
        quarter: Sequence[np.ndarray],
    ) -> LeadSpread:
        """Fit the spread from post-filter residuals, one array per lead band.

        Args:
            residuals: For each lead band, the actual minus corrected forecast of the replica.
            band: For each lead band, the hour band of each residual.
            quarter: For each lead band, the calendar quarter of each residual.

        Returns:
            The spread, centred on each group's median.

        Raises:
            ValueError: If there is not one array per lead band.
        """
        if not len(residuals) == len(band) == len(quarter) == N_LEAD_BANDS:
            msg = f"the spread needs one array per lead band, {N_LEAD_BANDS} in all"
            raise ValueError(msg)
        return LeadSpread(
            by_lead_band=tuple(
                ResidualSpread.fit(residuals=r, band=b, quarter=q, centre=True)
                for r, b, q in zip(residuals, band, quarter, strict=True)
            )
        )

    def quantiles(
        self, *, mean: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_band: int
    ) -> np.ndarray:
        """Return the 11 quantiles around each corrected mean for one lead band.

        Args:
            mean: The corrected mean forecast in points.
            band: The hour band of each row.
            quarter: The calendar quarter of each row.
            lead_band: The lead band of every row.

        Returns:
            An array of shape (len(mean), 11), non-decreasing along the last axis.
        """
        return self.by_lead_band[lead_band].quantiles(mean=mean, band=band, quarter=quarter)


def stitch_out_of_sample_residuals(
    *, cut_positions: Sequence[int], residuals: Sequence[np.ndarray]
) -> np.ndarray:
    """Join each fit's out-of-sample residuals into one stream across refits.

    Fit `i` supplies the grid positions after its own cut and up to the next fit's cut (the last
    fit supplies every position after its cut). A fit's in-sample residuals, at or before its own
    cut, never enter the stream, so the filter's state carries forward across a refit and switches
    to the new fit's residuals at the cut. Positions at or before the first cut stay `NaN`.

    Args:
        cut_positions: Each fit's cut as a grid position, strictly ascending.
        residuals: Each fit's observed-minus-forecast residuals on the grid, in points; only the
            positions after its cut need be meaningful.

    Returns:
        One stream on the grid, with `NaN` where no fit supplies a residual.

    Raises:
        ValueError: If there are no fits, the arrays and cuts differ in number or length, or the
            cuts are not strictly ascending.
    """
    if not cut_positions or len(cut_positions) != len(residuals):
        msg = "stitching needs one residual array per cut, and at least one"
        raise ValueError(msg)
    if any(later <= earlier for earlier, later in pairwise(cut_positions)):
        msg = "the cuts must be strictly ascending"
        raise ValueError(msg)
    if len({len(r) for r in residuals}) != 1:
        msg = "every fit's residuals must cover the same grid"
        raise ValueError(msg)
    stream = np.full(len(residuals[0]), np.nan)
    ends = [*cut_positions[1:], len(stream) - 1]
    for cut, end, fit_residuals in zip(cut_positions, ends, residuals, strict=True):
        stream[cut + 1 : end + 1] = fit_residuals[cut + 1 : end + 1]
    return stream


def post_filter_lead_spread(
    *,
    end_times: np.ndarray,
    replica: ReplicaResiduals,
    run: TraceBuilder,
    delay_hours: float = DEFAULT_DELAY_HOURS,
    burn_in_days: int = REPLICA_BURN_IN_DAYS,
) -> LeadSpread:
    """Fit the spread of a corrected arm from its replica run through the same correction.

    The replica's residuals form the residual stream, so the correction sees the same staleness a
    scored forecast does. For each number of days ahead from 1 to 7, every replica row is
    corrected from the readings that were usable at its day-ahead initialisation, and the
    post-correction residuals are grouped by lead band. Rows inside the switching masks and the
    first `burn_in_days` days of the replica year do not enter the spread.

    Args:
        end_times: The grid's half-hour ends as `datetime64[us]`.
        replica: The replica's residuals, fitted with `drop_switching=False` so the stream holds
            the switching rows, as a live filter would see them.
        run: Builds the correction's trace from the residual stream and the hour band of each
            grid row, for example `run_ewma` or `functools.partial(run_kalman, hyper=hyper)`.
        delay_hours: The telemetry delay.
        burn_in_days: How many days at the start of the replica year to leave out.

    Returns:
        The centred spread per lead band.
    """
    stream = np.full(len(end_times), np.nan)
    stream[replica.index] = replica.residuals
    grid_band = np.zeros(len(end_times), dtype=np.int64)
    grid_band[replica.index] = replica.band
    trace = run(residual=stream, band=grid_band)
    valid_times = end_times[replica.index]
    settled = valid_times > valid_times.min() + np.timedelta64(burn_in_days, "D")
    kept = settled & ~replica.switching
    residuals: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    bands: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    quarters: list[list[np.ndarray]] = [[] for _ in range(N_LEAD_BANDS)]
    for days_ahead in range(1, LEAD_BAND_UPPER_DAYS[-1] + 1):
        cutoff = usable_cutoff_positions(
            end_times=end_times,
            initialisation=day_ahead_initialisation(end_times=valid_times, days_ahead=days_ahead),
            delay_hours=delay_hours,
        )
        correction = correction_at(
            trace=trace, cutoff=cutoff, target=replica.index, band=replica.band
        )
        lead_band = lead_band_of_days_ahead(days_ahead=days_ahead)
        residuals[lead_band].append((replica.residuals - correction.total)[kept])
        bands[lead_band].append(replica.band[kept])
        quarters[lead_band].append(replica.quarter[kept])
    return LeadSpread.fit(
        residuals=[np.concatenate(r) for r in residuals],
        band=[np.concatenate(b) for b in bands],
        quarter=[np.concatenate(q) for q in quarters],
    )
