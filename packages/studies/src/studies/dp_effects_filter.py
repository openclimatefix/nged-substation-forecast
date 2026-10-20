"""A Kalman filter on the weights of the convex model's frozen effects, for the DP series.

**This module re-implements the method of [de Vilmarest et al.
(2024)](https://doi.org/10.1109/TPWRS.2023.3310280) on our convex model (`studies.dp_gam`).** The
convex model is a sum of effects, and the filter lets a weight on each effect drift.
`studies.dp_assimilation` corrects a frozen forecast with one level and one transient anomaly per
hour band. The filter here freezes the fitted effects, puts a random walk on one weight per effect,
and fits a diagonal noise matrix by likelihood, so that the heating response, the PV capacity, the
profile scale, and the holiday effects can change between refits.

**Where the design differs from the paper and from its R package
[viking](https://CRAN.R-project.org/package=viking)** (checked against viking 1.0.1):

- The state holds 14 weights on groups of effects (a group is several of the model's columns), not
  one coefficient per column, and optionally an AR(1) anomaly per hour band.
- viking adds the noise matrix after every reading, whereas the state here moves by the exact
  discretisation over the gap in half-hours since the previous reading, so a missing half-hour needs
  no placeholder.
- viking's `iterative_grid_search` maximises the likelihood over `Q` divided by the observation
  variance, with the observation variance profiled out, and each iteration applies the one grid
  change that improves the likelihood most. Here the observation noise is a parameter, each weight
  in turn takes its best grid value in two sweeps (cheaper), and an L-BFGS polish follows.
- The estimate is a maximum a posteriori one: every `log q_k` carries a normal prior shared across
  sites, and a weight with too few active half-hours takes the prior mean. viking has no prior.

**The 14 weights.** `effect_regressors` splits a fit's prediction into five groups of frozen
effects, each with its contribution `c_k(t)` at a weight of 1:

| Group | Weights | Regressor |
|---|---|---|
| level, per hour band | 4 | 1 in the band, else 0 (the weight is a shift in points) |
| profile scale, per day type | 4 | the day type's profile with its mean over the day removed |
| heating, per hour band | 4 | the band's fitted heating contribution |
| PV capacity | 1 | minus the pinned capacity times the derated unit PV shape |
| holiday | 1 | the named-holiday and Christmas-tent contributions |

Everything else the fit holds (day-of-year harmonics, lighting ramps, the profile means) is the
`remainder`, which stays frozen. The weights multiply the effects, so the forecast is the fit's
prediction plus `sum_k delta_k c_k(t)`, where `delta_k` is the weight minus 1.

**The state.** Each regressor is divided by its root mean square over the half-hours where it is
non-zero (`regressor_scale`), so that every random-walk variance `q_k` is in points squared per day
of a typical contribution and the `q_k` are comparable. The state is the 14 standardised weight
deviations and, optionally, one AR(1) anomaly per hour band (as in `studies.dp_assimilation`). One
reading per half-hour updates the state by a rank-one Kalman step, and the state moves by the exact
discretisation of the random walks and the AR(1) over the gap since the previous reading, so a
missing reading needs no placeholder. With the 10 non-level weights held at 1 (`moving` false) the
filter equals the per-band filter of `studies.dp_assimilation` (the nesting control).

**The information cut.** `state_at` returns the state after the newest reading at or before a
cutoff position, so a forecast sees only readings that are old enough. The filter is causal and has
no switching mask. The module does not know which months are tuning months: the likelihood mask, the
rows `regressor_scale` and `active_counts` look at, and the sites' streams in
`fit_effects_noise_sites` come from the caller, so keeping them to the tuning months is the
caller's duty.

**The noise matrix** is fitted by `fit_effects_noise` as a maximum a posteriori estimate: a batched
greedy search over a grid of `log q_k` for each weight in turn, then an L-BFGS polish. All `log q_k`
carry a normal prior whose mean and spread `fit_effects_noise_sites` updates from the sites' own
fits. A weight with too few active half-hours in the likelihood has its `q_k` set to the prior mean
instead of fitted.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import log, pi
from typing import Final

import numpy as np
from scipy.optimize import minimize

from studies.dp_assimilation import INITIAL_LEVEL_VARIANCE, VARIANCE_FLOOR
from studies.dp_calendar import DAY_TYPES, HALF_HOURS_PER_DAY
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import N_BANDS, GamFit, pinned_solar_points, predict_gam

N_DAY_TYPES: Final[int] = len(DAY_TYPES)
N_WEIGHTS: Final[int] = N_BANDS + N_DAY_TYPES + N_BANDS + 1 + 1
"""The weights: level per band, profile per day type, heating per band, PV, and holiday."""
N_STATES: Final[int] = N_WEIGHTS + N_BANDS
"""The weights followed by the anomaly of each hour band."""
LEVEL: Final[slice] = slice(0, N_BANDS)
PROFILE: Final[slice] = slice(N_BANDS, N_BANDS + N_DAY_TYPES)
HEATING: Final[slice] = slice(N_BANDS + N_DAY_TYPES, 2 * N_BANDS + N_DAY_TYPES)
PV_INDEX: Final[int] = 2 * N_BANDS + N_DAY_TYPES
HOLIDAY_INDEX: Final[int] = PV_INDEX + 1
GROUP_SLICES: Final[dict[str, slice]] = {
    "level": LEVEL,
    "profile": PROFILE,
    "heating": HEATING,
    "pv": slice(PV_INDEX, PV_INDEX + 1),
    "holiday": slice(HOLIDAY_INDEX, HOLIDAY_INDEX + 1),
}
"""The weights of each group."""
WEIGHT_NAMES: Final[tuple[str, ...]] = (
    *(f"level_band{b}" for b in range(N_BANDS)),
    *(f"profile_{name}" for name in DAY_TYPES),
    *(f"heating_band{b}" for b in range(N_BANDS)),
    "pv",
    "holiday",
)
"""A name for each weight."""
FLOORED_WEIGHTS: Final[np.ndarray] = np.array(
    [index in range(PROFILE.start, PV_INDEX + 1) for index in range(N_WEIGHTS)]
)
"""The weights floored at 0 in a forecast: profile scale, heating, and PV (not level or holiday)."""
NON_LEVEL: Final[np.ndarray] = np.arange(N_WEIGHTS) >= N_BANDS
"""True for the 10 weights that are not a level."""
NON_LEVEL_INITIAL_SD: Final[float] = 0.1
"""The prior standard deviation of a non-level weight's deviation from 1 at the first reading."""
ACTIVE_THRESHOLD: Final[float] = 1e-9
"""A regressor is active at a half-hour when its absolute value is above this."""
MIN_ACTIVE_HALF_HOURS: Final[int] = 200
"""A weight with fewer active half-hours in the likelihood has its `q` set by the prior."""
LOG_Q_BOUNDS: Final[tuple[float, float]] = (-20.0, 2.0)
"""Bounds on `log q`, which keep the optimiser off degenerate corners."""
LOG_SIGMA_EPS_BOUNDS: Final[tuple[float, float]] = (log(0.05), 6.0)
"""Bounds on `log sigma_eps` in points, which keep the update stable."""
GREEDY_GRID: Final[tuple[float, ...]] = (-18.0, -14.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0)
"""The `log q` values the greedy search tries for each weight in turn."""
FINITE_DIFFERENCE_STEP: Final[float] = 1e-4
"""The forward-difference step in `log q` of the polish's gradient."""
PRIOR_LOG_Q_START: Final[float] = log(0.05)
"""The first round's prior mean of `log q`: 0.05 points squared per day."""
PRIOR_LOG_Q_SPREAD_START: Final[float] = 2.0
"""The first round's prior standard deviation of `log q`."""
MIN_PRIOR_SPREAD: Final[float] = 0.5
"""The shared prior's spread never shrinks below this, so it stays a prior with few sites."""
SIGMA_EPS_PRIOR_SPREAD: Final[float] = 1.0
"""The standard deviation of the normal prior on `log sigma_eps`, centred on its start."""
N_EMPIRICAL_BAYES_ROUNDS: Final[int] = 3
"""How many times the shared prior is refitted from the sites' own fits."""
MAX_POLISH_ITERATIONS: Final[int] = 40
"""The most L-BFGS iterations of the polish."""
SYMMETRISE_EVERY: Final[int] = 512
"""The covariance is replaced by the mean of itself and its transpose after this many readings."""


@dataclass(frozen=True)
class EffectRegressors:
    """The frozen effects of a fit, split into the filter's regressors.

    Attributes:
        contributions: Shape (rows, 14), float64. Columns 0 to 3 are the hour-band indicators of the
            level weights; the others are each group's contribution at a weight of 1, in points.
        remainder: Shape (rows,). The fit's prediction minus the contributions of the 10 non-level
            weights, so `remainder + contributions[:, 4:].sum(axis=1)` is `predict_gam`.
    """

    contributions: np.ndarray
    remainder: np.ndarray


@dataclass(frozen=True)
class FilterNoise:
    """The noise of one filter run.

    Attributes:
        q: Random-walk variance per day of each standardised weight, shape (14,); 0 holds a weight
            at 1.
        phi: The AR(1) factor per half-hour of the anomaly.
        sigma_z: The anomaly's innovation standard deviation, in points.
        sigma_eps: The observation noise standard deviation, in points.
        initial_sd: The prior standard deviation of each standardised weight at the first reading,
            shape (14,); 0 for a held weight.
        use_anomaly: Whether the four anomaly states are in use.
    """

    q: np.ndarray
    phi: float
    sigma_z: float
    sigma_eps: float
    initial_sd: np.ndarray
    use_anomaly: bool


def initial_sd_for(*, scale: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Return the starting standard deviations: the level's 10 points and 0.1 of a weight otherwise.

    Args:
        scale: The root mean square of each regressor, shape (14,).
        moving: True for the weights that follow a random walk.

    Returns:
        Shape (14,); 0 for a weight that does not move.
    """
    sd = np.where(NON_LEVEL, NON_LEVEL_INITIAL_SD * scale, INITIAL_LEVEL_VARIANCE**0.5)
    return np.where(moving, sd, 0.0)


@dataclass(frozen=True)
class EffectsTrace:
    """The filter's state after each reading.

    Attributes:
        positions: The grid position of each reading, ascending.
        state: Shape (readings, 18): the 14 standardised weight deviations, then the anomaly of
            each hour band, after each reading.
        sd: Shape (readings, 14): the posterior standard deviation of each weight deviation.
        log_likelihood: The marginal log likelihood of the readings in the likelihood mask.
        noise: The noise the run used.
        final_covariance: Shape (18, 18): the state's covariance after the last reading.
    """

    positions: np.ndarray
    state: np.ndarray
    sd: np.ndarray
    log_likelihood: float
    noise: FilterNoise
    final_covariance: np.ndarray


@dataclass(frozen=True)
class FilterState:
    """The state a forecast starts from.

    Attributes:
        delta: The 14 standardised weight deviations.
        anomaly: The anomaly of each hour band, as of `position`.
        sd: The posterior standard deviation of each weight deviation.
        position: The grid position of the newest reading used, or -1 for none.
    """

    delta: np.ndarray
    anomaly: np.ndarray
    sd: np.ndarray
    position: int


def _complement_mask(*, n_columns: int, slices: Sequence[slice]) -> np.ndarray:
    """Return a boolean mask over columns that is True on the given slices."""
    mask = np.zeros(n_columns, dtype=bool)
    for chosen in slices:
        mask[chosen] = True
    return mask


def effect_regressors(
    *, cache: FeatureCache, fit: GamFit, index: np.ndarray, member: int = 0
) -> EffectRegressors:
    """Split a fit's prediction at the grid rows `index` into the filter's regressors.

    Args:
        cache: The cache whose weather drives the prediction (the proxy cache, an ensemble member's
            cache, or the estimates of past weather).
        fit: A fit with a pinned capacity.
        index: The grid rows.
        member: Which fit of the batch.

    Returns:
        The contributions and the remainder.

    Raises:
        ValueError: If the fit has no pinned capacity.
    """
    if fit.pinned is None:
        msg = "the effects filter needs a fit with a pinned capacity"
        raise ValueError(msg)
    layout = fit.layout
    n_columns = layout.n_columns
    total = predict_gam(cache=cache, fit=fit, index=index, member=member)
    profile_total = predict_gam(
        cache=cache,
        fit=fit,
        index=index,
        member=member,
        columns=_complement_mask(n_columns=n_columns, slices=[layout.slices["profile"]]),
    )
    heating_total = predict_gam(
        cache=cache,
        fit=fit,
        index=index,
        member=member,
        columns=_complement_mask(n_columns=n_columns, slices=[layout.slices["heating"]]),
    )
    holiday_total = predict_gam(
        cache=cache,
        fit=fit,
        index=index,
        member=member,
        columns=_complement_mask(
            n_columns=n_columns, slices=[layout.slices["holiday"], layout.slices["tent"]]
        ),
    )
    features = cache.sites[fit.site]
    solar = pinned_solar_points(
        cache=cache, features=features, pinned=fit.pinned, index=index, clip=layout.spec.clip
    )
    band = cache.calendar.band[index].astype(np.int64)
    day_type = cache.calendar.day_type[index].astype(np.int64)
    profile_mean = (
        fit.theta[member][layout.slices["profile"]].reshape(N_DAY_TYPES, HALF_HOURS_PER_DAY).mean(1)
    )
    contributions = np.zeros((len(index), N_WEIGHTS))
    contributions[np.arange(len(index)), band] = 1.0
    profile_centred = profile_total - profile_mean[day_type]
    contributions[np.arange(len(index)), PROFILE.start + day_type] = profile_centred
    contributions[np.arange(len(index)), HEATING.start + band] = heating_total
    contributions[:, PV_INDEX] = -solar
    contributions[:, HOLIDAY_INDEX] = holiday_total
    remainder = total - contributions[:, N_BANDS:].sum(axis=1)
    return EffectRegressors(contributions=contributions, remainder=remainder)


def stitched_effect_regressors(
    *, cache: FeatureCache, fits: Sequence[GamFit], site: str
) -> EffectRegressors:
    """Return the regressors of the chained fits on a cache, `NaN` where no residual can be made.

    Fit `i` supplies the rows after its cut and up to the next cut, as
    `studies.dp_ens.proxy_residual_stream` does, and the same rows are usable, so the regressors
    line up with that stream.

    Args:
        cache: The cache whose weather arrays drive the fits.
        fits: The monthly fits, strictly ascending by cut.
        site: The substation's label.

    Returns:
        Contributions of shape (grid rows, 14) and a remainder of shape (grid rows,).

    Raises:
        ValueError: If there is no fit or the cuts are not strictly ascending.
    """
    if not fits or any(later.cut <= earlier.cut for earlier, later in pairwise(fits)):
        msg = "the regressors need at least one fit, with strictly ascending cuts"
        raise ValueError(msg)
    features = cache.sites[site]
    n = len(cache.end_times)
    contributions = np.full((n, N_WEIGHTS), np.nan)
    remainder = np.full(n, np.nan)
    cut_positions = [
        int(np.searchsorted(cache.end_times, naive_utc(moment=fit.cut), side="right")) - 1
        for fit in fits
    ]
    ends = [*cut_positions[1:], n - 1]
    for fit, cut, end in zip(fits, cut_positions, ends, strict=True):
        rows = np.arange(cut + 1, end + 1)
        usable = (
            np.isfinite(features.smoothed_temperature_c[rows]).all(axis=1)
            & np.isfinite(features.pv_unit_shape[rows])
            & np.isfinite(features.temperature_c[rows])
            & np.isfinite(features.y[rows])
        )
        rows = rows[usable]
        if len(rows):
            pieces = effect_regressors(cache=cache, fit=fit, index=rows)
            contributions[rows] = pieces.contributions
            remainder[rows] = pieces.remainder
    return EffectRegressors(contributions=contributions, remainder=remainder)


def regressor_scale(*, contributions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the root mean square of each regressor over its active half-hours in `mask`.

    The level weights have a scale of 1, because their regressor is an indicator. A regressor with
    no active half-hour has a scale of 1.

    Args:
        contributions: Shape (rows, 14).
        mask: True for the rows the scale is computed from (the tuning months' readings).

    Returns:
        Shape (14,).
    """
    scale = np.ones(N_WEIGHTS)
    chosen = contributions[mask]
    for k in range(N_BANDS, N_WEIGHTS):
        column = chosen[:, k]
        active = np.isfinite(column) & (np.abs(column) > ACTIVE_THRESHOLD)
        if active.any():
            scale[k] = float(np.sqrt(np.mean(column[active] ** 2)))
    return scale


def active_counts(*, contributions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return how many rows in `mask` each regressor is active at, shape (14,)."""
    column = contributions[mask]
    return (np.isfinite(column) & (np.abs(column) > ACTIVE_THRESHOLD)).sum(axis=0)


def _filter_core(
    *,
    residual: np.ndarray,
    regressors: np.ndarray,
    band: np.ndarray,
    q: np.ndarray,
    sigma_eps: np.ndarray,
    phi: float,
    sigma_z: float,
    initial_sd: np.ndarray,
    use_anomaly: bool,
    likelihood_mask: np.ndarray | None,
    keep_trace: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    """Run the filter for a batch of noise settings that share the data.

    Args:
        residual: Observed minus the frozen forecast on the grid, `NaN` where there is no reading.
        regressors: Standardised regressors on the grid, shape (grid rows, 14).
        band: The hour band of each grid row.
        q: Random-walk variance per day, shape (batch, 14).
        sigma_eps: Observation noise standard deviation, shape (batch,).
        phi: The anomaly's AR(1) factor per half-hour.
        sigma_z: The anomaly's innovation standard deviation.
        initial_sd: Starting standard deviations of the weights, shape (14,).
        use_anomaly: Whether the anomaly states take part.
        likelihood_mask: True on the grid rows whose readings add to the likelihood; `None` for all.
        keep_trace: Whether to keep the state (only for a batch of one).

    Returns:
        The log likelihood of each batch member, the state after each reading and the posterior
        standard deviations (both `None` unless `keep_trace`), the readings' positions, and the
        covariance after the last reading, shape (batch, 18, 18). A weight whose `initial_sd` is 0
        is held at 1 exactly: its regressor is left out of the update.
    """
    positions = np.flatnonzero(np.isfinite(residual) & np.isfinite(regressors).all(axis=1))
    n_batch = len(sigma_eps)
    design = np.zeros((len(positions), N_STATES))
    design[:, :N_WEIGHTS] = regressors[positions] * (initial_sd > 0.0)
    if use_anomaly:
        design[np.arange(len(positions)), N_WEIGHTS + band[positions].astype(np.int64)] = 1.0
    values = residual[positions]
    counted = (
        np.ones(len(positions), dtype=bool)
        if likelihood_mask is None
        else likelihood_mask[positions]
    )
    stationary = sigma_z**2 / (1.0 - phi**2)
    x = np.zeros((n_batch, N_STATES))
    covariance = np.zeros((n_batch, N_STATES, N_STATES))
    diagonal = covariance.reshape(n_batch, -1)[:, :: N_STATES + 1]
    diagonal[:] = np.r_[initial_sd**2, np.full(N_BANDS, stationary)]
    noise = sigma_eps**2
    q_step = q / HALF_HOURS_PER_DAY
    grows = np.ones(N_WEIGHTS)
    grows[:N_BANDS] = 0.0
    log_likelihood = np.zeros(n_batch)
    state = np.empty((len(positions), N_STATES)) if keep_trace else None
    sd = np.empty((len(positions), N_WEIGHTS)) if keep_trace else None
    previous = int(positions[0]) if len(positions) else 0
    for k in range(len(positions)):
        position = int(positions[k])
        band_k = int(band[position])
        gap = position - previous
        previous = position
        if gap:
            if use_anomaly:
                decay = phi**gap
                covariance[:, N_WEIGHTS:, :] *= decay
                covariance[:, :, N_WEIGHTS:] *= decay
                x[:, N_WEIGHTS:] *= decay
                diagonal[:, N_WEIGHTS:] += stationary * (1.0 - decay * decay)
            diagonal[:, :N_WEIGHTS] += q_step * (gap * grows)
        grows[band_k] = 1.0
        h = design[k]
        p_h = covariance @ h
        innovation_variance = p_h @ h + noise
        innovation = values[k] - x @ h
        if counted[k]:
            log_likelihood -= 0.5 * (
                np.log(2.0 * pi * innovation_variance) + innovation**2 / innovation_variance
            )
        gain = p_h / innovation_variance[:, None]
        x += gain * innovation[:, None]
        covariance -= gain[:, :, None] * p_h[:, None, :]
        np.maximum(diagonal, VARIANCE_FLOOR, out=diagonal)
        if k % SYMMETRISE_EVERY == SYMMETRISE_EVERY - 1:
            covariance[:] = 0.5 * (covariance + covariance.transpose(0, 2, 1))
        if state is not None and sd is not None:
            state[k] = x[0]
            sd[k] = np.sqrt(diagonal[0, :N_WEIGHTS])
    return log_likelihood, state, sd, positions, covariance


def run_effects_filter(
    *,
    residual: np.ndarray,
    regressors: np.ndarray,
    band: np.ndarray,
    noise: FilterNoise,
    likelihood_mask: np.ndarray | None = None,
) -> EffectsTrace:
    """Run the filter over a site's residual stream and keep the state after every reading.

    Args:
        residual: Observed minus the frozen forecast on the grid, in points, `NaN` where there is no
            reading. The caller includes switching rows: the filter has no mask.
        regressors: The standardised regressors on the grid, shape (grid rows, 14); a row with a
            non-finite entry is not a reading.
        band: The hour band of each grid row, from 0 to 3.
        noise: The noise.
        likelihood_mask: True on the grid rows whose readings count towards the likelihood; the
            state is updated by every reading regardless. `None` counts all.

    Returns:
        The trace.
    """
    log_likelihood, state, sd, positions, covariance = _filter_core(
        residual=residual,
        regressors=regressors,
        band=band,
        q=noise.q[None, :],
        sigma_eps=np.array([noise.sigma_eps]),
        phi=noise.phi,
        sigma_z=noise.sigma_z,
        initial_sd=noise.initial_sd,
        use_anomaly=noise.use_anomaly,
        likelihood_mask=likelihood_mask,
        keep_trace=True,
    )
    return EffectsTrace(
        positions=positions,
        state=np.zeros((0, N_STATES)) if state is None else state,
        sd=np.zeros((0, N_WEIGHTS)) if sd is None else sd,
        log_likelihood=float(log_likelihood[0]),
        noise=noise,
        final_covariance=covariance[0],
    )


def state_at(*, trace: EffectsTrace, cutoff: int) -> FilterState:
    """Return the state after the newest reading at or before the grid position `cutoff`.

    Args:
        trace: The filter's trace.
        cutoff: The newest grid position a forecast may see.

    Returns:
        The state; all zeros (position -1) if no reading is that old.
    """
    found = int(np.searchsorted(trace.positions, cutoff, side="right")) - 1
    if found < 0:
        return FilterState(
            delta=np.zeros(N_WEIGHTS),
            anomaly=np.zeros(N_BANDS),
            sd=np.zeros(N_WEIGHTS),
            position=-1,
        )
    return FilterState(
        delta=trace.state[found, :N_WEIGHTS].copy(),
        anomaly=trace.state[found, N_WEIGHTS:].copy(),
        sd=trace.sd[found].copy(),
        position=int(trace.positions[found]),
    )


@dataclass(frozen=True)
class Correction:
    """The filter's correction to the members of one run.

    Attributes:
        shift: The correction of each member, shape (members, targets), in points.
        level: The level weight of each target's hour band, in points.
        anomaly: The decayed anomaly of each target's hour band, in points.
    """

    shift: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray


def correction_from_state(
    *,
    state: FilterState,
    standardised_contributions: np.ndarray,
    scale: np.ndarray,
    band: np.ndarray,
    target: np.ndarray,
    phi: float,
) -> Correction:
    """Return the correction of each member: its own regressors times the held weights, plus decay.

    The profile-scale, heating, and PV weights are floored at 0 (a weight below 0 would flip an
    effect's sign), and the weights are held at their value at the cutoff for every lead.

    A target at or before the state's position gets the state's anomaly undecayed.

    Args:
        state: The state at the cutoff.
        standardised_contributions: Each member's regressors divided by `scale`, shape
            (members, targets, 14).
        scale: The regressor scales, shape (14,).
        band: The hour band of each target.
        target: The grid position of each target.
        phi: The anomaly's AR(1) factor per half-hour.

    Returns:
        The correction.
    """
    lower = np.where(FLOORED_WEIGHTS, -scale, -np.inf)
    delta = np.maximum(state.delta, lower)
    weighted = standardised_contributions @ delta
    anomaly = np.zeros(len(target))
    if state.position >= 0:
        anomaly = phi ** np.maximum(target - state.position, 0) * state.anomaly[band]
    level = delta[LEVEL][band]
    return Correction(shift=weighted + anomaly[None, :], level=level, anomaly=anomaly)


@dataclass(frozen=True)
class FilterVariant:
    """Which weights move and how the noise is shared, for one arm.

    Attributes:
        moving: True for the weights that follow a random walk, shape (14,).
        use_anomaly: Whether the anomaly states are in use.
        shared_q: Whether one `q` is fitted for every moving weight (the identity noise matrix).
    """

    moving: np.ndarray
    use_anomaly: bool
    shared_q: bool = False


@dataclass(frozen=True)
class FittedNoise:
    """A fitted noise matrix.

    Attributes:
        noise: The noise.
        log_q: `log q` of each weight (`-inf` for a weight that does not move).
        set_by_prior: True for the moving weights whose `q` was set to the prior mean.
        n_active: The active half-hours of each weight in the likelihood.
        log_posterior: The objective at the optimum (log likelihood plus log prior).
        converged: Whether the polish reported success.
    """

    noise: FilterNoise
    log_q: np.ndarray
    set_by_prior: np.ndarray
    n_active: np.ndarray
    log_posterior: float
    converged: bool


def _expand_q(*, theta_q: np.ndarray, free: np.ndarray, base_log_q: np.ndarray) -> np.ndarray:
    """Return `q` for each row of `theta_q` (batch, free count), other weights at `base_log_q`."""
    log_q = np.broadcast_to(base_log_q, (theta_q.shape[0], N_WEIGHTS)).copy()
    log_q[:, free] = theta_q
    return np.exp(log_q)


def fit_effects_noise(
    *,
    residual: np.ndarray,
    regressors: np.ndarray,
    band: np.ndarray,
    scale: np.ndarray,
    variant: FilterVariant,
    phi: float,
    sigma_z: float,
    sigma_eps_start: float,
    likelihood_mask: np.ndarray,
    prior_mean: np.ndarray,
    prior_spread: np.ndarray,
    start_log_q: np.ndarray | None = None,
    greedy: bool = True,
    sigma_eps_spread: float = SIGMA_EPS_PRIOR_SPREAD,
) -> FittedNoise:
    """Fit one site's noise matrix by maximum a posteriori.

    The search varies `log q` of the fitted weights and `log sigma_eps`. A batch of noise settings
    that share the data runs in one pass, so a greedy sweep and a finite-difference gradient each
    cost about one pass of the filter.

    Args:
        residual: The site's residual stream on the grid, in points.
        regressors: The standardised regressors, shape (grid rows, 14).
        band: The hour band of each grid row.
        scale: The regressor scales, shape (14,).
        variant: Which weights move and whether `q` is shared.
        phi: The anomaly's AR(1) factor (fixed).
        sigma_z: The anomaly's innovation standard deviation (fixed).
        sigma_eps_start: Where `sigma_eps` starts; its prior is centred here.
        likelihood_mask: True on the grid rows whose readings count in the likelihood.
        prior_mean: The shared prior mean of each weight's `log q`, shape (14,).
        prior_spread: The shared prior standard deviation of each, shape (14,).

        start_log_q: Where the polish starts (a previous fit); this skips the greedy search.
        greedy: Whether to run the greedy search before the polish.
        sigma_eps_spread: The standard deviation of the normal prior on `log sigma_eps`, centred
            on `sigma_eps_start`; a small value holds the observation noise near its start.

    Returns:
        The fitted noise.
    """
    counts = active_counts(contributions=regressors, mask=likelihood_mask)
    set_by_prior = variant.moving & (counts < MIN_ACTIVE_HALF_HOURS)
    free = variant.moving & ~set_by_prior
    free_index = np.flatnonzero(free)
    n_free = min(len(free_index), 1) if variant.shared_q else len(free_index)
    base_log_q = np.where(variant.moving, prior_mean, -np.inf)
    initial_sd = initial_sd_for(scale=scale, moving=variant.moving)
    shared_mean = float(prior_mean[free_index].mean()) if len(free_index) else PRIOR_LOG_Q_START
    shared_spread = float(prior_spread[free_index].mean()) if len(free_index) else 1.0

    def to_q(theta: np.ndarray) -> np.ndarray:
        theta_q = theta[:, :n_free]
        if variant.shared_q:
            theta_q = np.repeat(theta_q, len(free_index), axis=1)
        return _expand_q(theta_q=theta_q, free=free_index, base_log_q=base_log_q)

    def log_prior(theta: np.ndarray) -> np.ndarray:
        theta_q = theta[:, :n_free]
        if n_free == 0:
            penalty = np.zeros(theta.shape[0])
        elif variant.shared_q:
            penalty = ((theta_q[:, 0] - shared_mean) / shared_spread) ** 2
        else:
            penalty = (((theta_q - prior_mean[free_index]) / prior_spread[free_index]) ** 2).sum(1)
        eps = ((theta[:, -1] - log(sigma_eps_start)) / sigma_eps_spread) ** 2
        return -0.5 * (penalty + eps)

    def evaluate(theta: np.ndarray) -> np.ndarray:
        q = to_q(theta)
        q = np.where(variant.moving[None, :], q, 0.0)
        log_likelihood, _, _, _, _ = _filter_core(
            residual=residual,
            regressors=regressors,
            band=band,
            q=q,
            sigma_eps=np.exp(theta[:, -1]),
            phi=phi,
            sigma_z=sigma_z,
            initial_sd=initial_sd,
            use_anomaly=variant.use_anomaly,
            likelihood_mask=likelihood_mask,
            keep_trace=False,
        )
        return log_likelihood + log_prior(theta)

    start = np.full(n_free + 1, LOG_Q_BOUNDS[0] + 2.0)
    if start_log_q is not None and n_free:
        start[:n_free] = (
            start_log_q[free_index[:1]] if variant.shared_q else start_log_q[free_index]
        )
    start[-1] = log(sigma_eps_start)
    theta = start.copy()
    best = float(evaluate(theta[None, :])[0]) if n_free else 0.0
    if greedy and start_log_q is None and n_free:
        for _ in range(2):
            for j in range(n_free):
                candidates = np.tile(theta, (len(GREEDY_GRID), 1))
                candidates[:, j] = GREEDY_GRID
                scores = evaluate(candidates)
                if scores.max() > best:
                    best = float(scores.max())
                    theta = candidates[int(scores.argmax())]
    bounds = [LOG_Q_BOUNDS] * n_free + [LOG_SIGMA_EPS_BOUNDS]

    def objective(point: np.ndarray) -> tuple[float, np.ndarray]:
        batch = np.tile(point, (len(point) + 1, 1))
        batch[1:] += FINITE_DIFFERENCE_STEP * np.eye(len(point))
        scores = evaluate(batch)
        return -float(scores[0]), -(scores[1:] - scores[0]) / FINITE_DIFFERENCE_STEP

    result = minimize(
        objective,
        x0=theta,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": MAX_POLISH_ITERATIONS},
    )
    theta = result.x
    q_full = np.where(variant.moving, to_q(theta[None, :])[0], 0.0)
    log_q = np.where(variant.moving, np.log(np.maximum(q_full, np.exp(-745.0))), -np.inf)
    noise = FilterNoise(
        q=q_full,
        phi=phi,
        sigma_z=sigma_z,
        sigma_eps=float(np.exp(theta[-1])),
        initial_sd=initial_sd,
        use_anomaly=variant.use_anomaly,
    )
    return FittedNoise(
        noise=noise,
        log_q=log_q,
        set_by_prior=set_by_prior,
        n_active=counts,
        log_posterior=-float(result.fun),
        converged=bool(result.success),
    )


@dataclass(frozen=True)
class SiteStream:
    """One site's inputs to a noise fit.

    Attributes:
        residual: Observed minus the frozen forecast on the grid, in points.
        regressors: The standardised regressors on the grid, shape (grid rows, 14).
        scale: The regressor scales, shape (14,).
        likelihood_mask: True on the grid rows that count in the likelihood.
        phi: The site's anomaly AR(1) factor.
        sigma_z: The site's anomaly innovation standard deviation.
        sigma_eps_start: The site's starting observation noise.
    """

    residual: np.ndarray
    regressors: np.ndarray
    scale: np.ndarray
    likelihood_mask: np.ndarray
    phi: float
    sigma_z: float
    sigma_eps_start: float


def fit_effects_noise_sites(
    *, streams: Mapping[str, SiteStream], band: np.ndarray, variant: FilterVariant
) -> tuple[dict[str, FittedNoise], np.ndarray, np.ndarray]:
    """Fit every site's noise matrix under a prior shared across sites.

    Each round fits every site under the current prior, then replaces the prior's mean and spread,
    weight by weight, with the sites' own fits (the spread never below `MIN_PRIOR_SPREAD`). A
    weight that no site fitted keeps its prior. The first round runs the greedy search, and later
    rounds polish from the previous fit.

    Args:
        streams: Each site's inputs.
        band: The hour band of each grid row.
        variant: Which weights move and whether `q` is shared.

    Returns:
        Each site's fit, and the final prior mean and spread of `log q` per weight.

    Raises:
        ValueError: If there are no sites.
    """
    if not streams:
        msg = "no sites to fit the effects filter on"
        raise ValueError(msg)
    prior_mean = np.full(N_WEIGHTS, PRIOR_LOG_Q_START)
    prior_spread = np.full(N_WEIGHTS, PRIOR_LOG_Q_SPREAD_START)
    fitted: dict[str, FittedNoise] = {}
    for round_index in range(N_EMPIRICAL_BAYES_ROUNDS):
        fitted = {
            site: fit_effects_noise(
                residual=stream.residual,
                regressors=stream.regressors,
                band=band,
                scale=stream.scale,
                variant=variant,
                phi=stream.phi,
                sigma_z=stream.sigma_z,
                sigma_eps_start=stream.sigma_eps_start,
                likelihood_mask=stream.likelihood_mask,
                prior_mean=prior_mean,
                prior_spread=prior_spread,
                start_log_q=fitted[site].log_q if site in fitted else None,
                greedy=round_index == 0,
            )
            for site, stream in streams.items()
        }
        prior_mean, prior_spread = _updated_prior(
            fits=list(fitted.values()), prior_mean=prior_mean, prior_spread=prior_spread
        )
    return fitted, prior_mean, prior_spread


def _updated_prior(
    *, fits: Sequence[FittedNoise], prior_mean: np.ndarray, prior_spread: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the prior refitted from the sites that fitted each weight."""
    mean = prior_mean.copy()
    spread = prior_spread.copy()
    for k in range(N_WEIGHTS):
        values = np.array(
            [fit.log_q[k] for fit in fits if np.isfinite(fit.log_q[k]) and not fit.set_by_prior[k]]
        )
        if len(values) == 0:
            continue
        mean[k] = values.mean()
        if len(values) > 1:
            spread[k] = max(float(values.std(ddof=1)), MIN_PRIOR_SPREAD)
    return mean, spread


def relative_step_per_year(*, noise: FilterNoise, scale: np.ndarray) -> np.ndarray:
    """Return each weight's random-walk step over a year as a share of its fitted effect.

    Args:
        noise: The fitted noise.
        scale: The regressor scales, shape (14,).

    Returns:
        `sqrt(365 q) / scale` for each weight, in weight units (for a level, in points).
    """
    return np.sqrt(365.0 * noise.q) / scale
