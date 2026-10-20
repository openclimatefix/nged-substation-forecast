"""The recent-error filter with a diurnal-shape state, for experiment 115 (W1).

**GAM-K's filter (`studies.dp_assimilation`) keeps, per hour band, a persistent level and a decaying
transient. This filter adds a shape state `s`: one value per local half-hour of the day, updated
from every reading and held flat over every lead.** The residual model is

    r(t) = level[g(t)] + transient[g(t)] + s[h(t)] + noise

where `g` is the group of the half-hour (its hour band, or one group for `KS1`), `h` is its local
half-hour of day (0 to 47), the level is a random walk, the transient is an AR(1) process, and `s`
is a random walk per day whose steps and initial state lie in a subspace chosen by the variant:

| Variant | Groups of level and transient | `s` has zero mean |
|---|---|---|
| `KW` | the four hour bands | inside each hour band |
| `KS` | the four hour bands | over the whole day |
| `KS1` | one | over the whole day |

With `q_s = 0` and no initial shape variance the filter is exactly GAM-K's, because the shape then
never leaves zero.

**One joint filter with a global clock replaces the four independent band filters.** The state moves
by the exact discretisation over the gap in half-hours between consecutive readings, so a transient
decays across a band's daily gap whether or not another band was read meanwhile. The mean of each
group's level and transient equals the independent filters' values, so the two forms agree to
rounding.

**Information cut.** `lookup` returns, for each case, the state after the newest reading at or before
the case's cutoff, and the transient decays by `phi` per half-hour from that reading to the target.
The cutoff is the caller's; the filter has no mask.
"""

from dataclasses import dataclass
from math import exp, log, pi
from typing import Final, Literal

import numpy as np
from scipy.optimize import minimize
from studies.dp_assimilation import (
    BOUNDS,
    INITIAL_LEVEL_VARIANCE,
    MIN_SIGMA_EPS,
    VARIANCE_FLOOR,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY

N_GROUPS_BANDS: Final[int] = 4
N_HALF_HOURS: Final[int] = HALF_HOURS_PER_DAY
INITIAL_SHAPE_DAYS: Final[float] = 30.0
"""The initial shape variance is this many days of random-walk steps."""
SHAPE_BOUNDS: Final[tuple[tuple[float, float], ...]] = (
    (log(1e-6), log(1.0)),
    (log(MIN_SIGMA_EPS), 6.0),
)
"""Bounds on log `q_s` (points squared per day) and log `sigma_eps`."""
MIN_SHAPE_PRIOR_SPREAD: Final[float] = 0.25
N_EMPIRICAL_BAYES_ROUNDS: Final[int] = 3

VariantType = Literal["KW", "KS", "KS1"]


@dataclass(frozen=True)
class ShapeStructure:
    """Which half-hours share a level, and which shapes are allowed.

    Attributes:
        group: The group (0 to n_groups - 1) of each local half-hour of day, shape (48,).
        projection: The 48 by 48 projection onto the allowed shapes (zero mean inside each block).
    """

    group: np.ndarray
    projection: np.ndarray

    @property
    def n_groups(self) -> int:
        """The number of level-and-transient groups."""
        return int(self.group.max()) + 1


def build_structure(*, variant: VariantType, band_of_half_hour: np.ndarray) -> ShapeStructure:
    """Return the structure of a variant.

    Args:
        variant: `KW`, `KS`, or `KS1`.
        band_of_half_hour: The hour band of each local half-hour of day, shape (48,).

    Returns:
        The structure.
    """
    projection = np.zeros((N_HALF_HOURS, N_HALF_HOURS))
    if variant == "KW":
        for band in range(N_GROUPS_BANDS):
            members = np.flatnonzero(band_of_half_hour == band)
            projection[np.ix_(members, members)] = np.eye(len(members)) - 1.0 / len(members)
        return ShapeStructure(group=band_of_half_hour.astype(np.int64), projection=projection)
    projection[:] = np.eye(N_HALF_HOURS) - 1.0 / N_HALF_HOURS
    group = band_of_half_hour.astype(np.int64) if variant == "KS" else np.zeros(N_HALF_HOURS, int)
    return ShapeStructure(group=group, projection=projection)


@dataclass(frozen=True)
class ShapeHyper:
    """The filter's hyperparameters.

    Attributes:
        log_q: Log of the level's random-walk variance per day, in points squared.
        logit_phi: Logit of the transient's AR(1) factor per half-hour.
        log_sigma_z: Log of the transient's innovation standard deviation, in points.
        log_sigma_eps: Log of the observation noise standard deviation, in points.
        log_q_s: Log of the shape's random-walk variance per day, in points squared; `None` for no
            shape (`q_s = 0`, initial shape variance 0).
    """

    log_q: float
    logit_phi: float
    log_sigma_z: float
    log_sigma_eps: float
    log_q_s: float | None

    @property
    def q(self) -> float:
        """The level's random-walk variance per day."""
        return exp(self.log_q)

    @property
    def phi(self) -> float:
        """The transient's AR(1) factor per half-hour."""
        return 1.0 / (1.0 + exp(-self.logit_phi))

    @property
    def sigma_z(self) -> float:
        """The transient's innovation standard deviation."""
        return exp(self.log_sigma_z)

    @property
    def sigma_eps(self) -> float:
        """The observation noise standard deviation."""
        return exp(self.log_sigma_eps)

    @property
    def q_s(self) -> float:
        """The shape's random-walk variance per day (0 for no shape)."""
        return 0.0 if self.log_q_s is None else exp(self.log_q_s)

    def scaled(self, *, multiplier: float) -> ShapeHyper:
        """Return the hyperparameters with every state step variance multiplied.

        The level step `q`, the transient innovation variance `sigma_z**2`, and the shape step
        `q_s` all scale, and the observation noise stays.
        """
        shift = log(multiplier)
        return ShapeHyper(
            log_q=self.log_q + shift,
            logit_phi=self.logit_phi,
            log_sigma_z=self.log_sigma_z + 0.5 * shift,
            log_sigma_eps=self.log_sigma_eps,
            log_q_s=None if self.log_q_s is None else self.log_q_s + shift,
        )

    def with_shape(
        self, *, log_q_s: float | None, log_sigma_eps: float | None = None
    ) -> ShapeHyper:
        """Return the hyperparameters with another shape step and, optionally, noise."""
        return ShapeHyper(
            log_q=self.log_q,
            logit_phi=self.logit_phi,
            log_sigma_z=self.log_sigma_z,
            log_sigma_eps=self.log_sigma_eps if log_sigma_eps is None else log_sigma_eps,
            log_q_s=log_q_s,
        )


@dataclass(frozen=True)
class ShapeTrace:
    """The joint state after each reading.

    Attributes:
        positions: The grid position of each reading, ascending.
        state: The state after each reading, shape (readings, 2 * groups + 48): the levels, the
            transients, then the shape.
        n_groups: The number of level-and-transient groups.
        phi: The transient's AR(1) factor per half-hour.
        log_likelihood: The marginal log likelihood of the readings.
        shape_variance: The posterior variance of each shape value after each reading, shape
            (readings, 48), or `None` when the run did not keep it.
    """

    positions: np.ndarray
    state: np.ndarray
    n_groups: int
    phi: float
    log_likelihood: float
    shape_variance: np.ndarray | None = None


@dataclass(frozen=True)
class Lookup:
    """The correction at each case, split into its three parts.

    Attributes:
        level: The persistent level of the target's group.
        anomaly: The decayed transient of the target's group.
        shape: The shape value at the target's local half-hour.
    """

    level: np.ndarray
    anomaly: np.ndarray
    shape: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """The correction to add to the frozen forecast."""
        return self.level + self.anomaly + self.shape


def run_shape_filter(
    *,
    residual: np.ndarray,
    half_hour: np.ndarray,
    structure: ShapeStructure,
    hyper: ShapeHyper,
    keep_states: bool = True,
    keep_variance: bool = False,
) -> ShapeTrace:
    """Run the joint filter over a site's residual stream.

    Args:
        residual: Observed minus forecast on the grid, in points, `NaN` where there is no reading.
            The caller includes the switching rows: the filter has no mask.
        half_hour: The local half-hour of day of each grid row, 0 to 47.
        structure: The variant's structure.
        hyper: The hyperparameters.
        keep_states: False to return only the likelihood (positions and states are empty).
        keep_variance: True to keep the shape's posterior variance after each reading.

    Returns:
        The trace.
    """
    n_groups = structure.n_groups
    dim = 2 * n_groups + N_HALF_HOURS
    shape_slice = slice(2 * n_groups, dim)
    positions = np.flatnonzero(np.isfinite(residual))
    values = residual[positions]
    half_hours = half_hour[positions].astype(np.int64)
    groups = structure.group[half_hours]
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    q_step = hyper.q / N_HALF_HOURS
    shape_step = hyper.q_s / N_HALF_HOURS
    projection = structure.projection
    x = np.zeros(dim)
    p = np.zeros((dim, dim))
    level_index = np.arange(n_groups)
    transient_index = n_groups + level_index
    p[level_index, level_index] = INITIAL_LEVEL_VARIANCE
    p[transient_index, transient_index] = stationary
    p[shape_slice, shape_slice] = INITIAL_SHAPE_DAYS * hyper.q_s * projection
    decay_vector = np.ones(dim)
    seen = np.zeros(n_groups)
    states = np.empty((len(values), dim)) if keep_states else np.empty((0, dim))
    variances = np.empty((len(values), N_HALF_HOURS)) if keep_variance else None
    log_likelihood = 0.0
    previous = int(positions[0]) if len(positions) else 0
    for k in range(len(values)):
        gap = int(positions[k]) - previous
        previous = int(positions[k])
        if gap:
            decay = phi**gap
            decay_vector[transient_index] = decay
            x *= decay_vector
            p *= decay_vector[:, None]
            p *= decay_vector[None, :]
            p[level_index, level_index] += (q_step * gap) * seen
            p[transient_index, transient_index] += stationary * (1.0 - decay * decay)
            if shape_step:
                p[shape_slice, shape_slice] += (shape_step * gap) * projection
        group = groups[k]
        seen[group] = 1.0
        i_level = group
        i_transient = n_groups + group
        i_shape = 2 * n_groups + half_hours[k]
        gain_numerator = p[:, i_level] + p[:, i_transient] + p[:, i_shape]
        innovation_variance = (
            gain_numerator[i_level] + gain_numerator[i_transient] + gain_numerator[i_shape] + noise
        )
        innovation = values[k] - x[i_level] - x[i_transient] - x[i_shape]
        log_likelihood -= 0.5 * (
            log(2.0 * pi * innovation_variance) + innovation * innovation / innovation_variance
        )
        gain = gain_numerator / innovation_variance
        x += gain * innovation
        p -= innovation_variance * np.outer(gain, gain)
        diagonal = np.arange(2 * n_groups)
        p[diagonal, diagonal] = np.maximum(p[diagonal, diagonal], VARIANCE_FLOOR)
        if hyper.q_s:
            x[shape_slice] = projection @ x[shape_slice]
        if keep_states:
            states[k] = x
        if variances is not None:
            variances[k] = np.diag(p)[shape_slice]
    return ShapeTrace(
        positions=positions if keep_states else np.empty(0, dtype=np.int64),
        state=states,
        n_groups=n_groups,
        phi=phi,
        log_likelihood=log_likelihood,
        shape_variance=variances,
    )


def lookup(
    *,
    trace: ShapeTrace,
    structure: ShapeStructure,
    cutoff: np.ndarray,
    target: np.ndarray,
    half_hour: np.ndarray,
) -> Lookup:
    """Look up the correction at forecast cases.

    Each case uses the state after the newest reading at or before its cutoff. A case with no such
    reading gets a correction of zero.

    Args:
        trace: The filter's trace.
        structure: The variant's structure.
        cutoff: Each case's newest usable grid position.
        target: Each case's grid position of the valid time.
        half_hour: Each case's local half-hour of day at the target, 0 to 47.

    Returns:
        The level, the decayed transient, and the shape of each case.
    """
    n = len(target)
    level, anomaly, shape = np.zeros(n), np.zeros(n), np.zeros(n)
    found = np.searchsorted(trace.positions, cutoff, side="right") - 1
    known = found >= 0
    state = trace.state[found[known]]
    rows = np.flatnonzero(known)
    group = structure.group[half_hour[rows]]
    level[rows] = state[np.arange(len(rows)), group]
    ahead = target[rows] - trace.positions[found[known]]
    anomaly[rows] = trace.phi**ahead * state[np.arange(len(rows)), trace.n_groups + group]
    shape[rows] = state[np.arange(len(rows)), 2 * trace.n_groups + half_hour[rows]]
    return Lookup(level=level, anomaly=anomaly, shape=shape)


def fit_shape_hyper(
    *,
    residual: np.ndarray,
    half_hour: np.ndarray,
    structure: ShapeStructure,
    base: ShapeHyper,
    prior_mean: np.ndarray,
    prior_spread: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    """Fit log `q_s` and log `sigma_eps` by maximum a posteriori under a normal prior.

    The level and transient settings stay at `base`'s values.

    Args:
        residual: The stream with every reading outside the tuning months set to `NaN`.
        half_hour: The local half-hour of each grid row.
        structure: The variant's structure.
        base: The settings that stay fixed.
        prior_mean: The shared prior mean of (log `q_s`, log `sigma_eps`).
        prior_spread: The shared prior's standard deviation of each.
        start: Where the optimiser starts.

    Returns:
        The two fitted values.

    Raises:
        ValueError: If the optimiser does not report success.
    """

    def negative_posterior(values: np.ndarray) -> float:
        trial = base.with_shape(log_q_s=float(values[0]), log_sigma_eps=float(values[1]))
        likelihood = run_shape_filter(
            residual=residual,
            half_hour=half_hour,
            structure=structure,
            hyper=trial,
            keep_states=False,
        ).log_likelihood
        prior = -0.5 * float((((values - prior_mean) / prior_spread) ** 2).sum())
        return -(likelihood + prior)

    result = minimize(negative_posterior, x0=start, method="L-BFGS-B", bounds=SHAPE_BOUNDS)
    if not result.success:
        # A flat likelihood in log q_s (no shape in the data) can stop the line search; Powell
        # polishes from where it stopped.
        result = minimize(negative_posterior, x0=result.x, method="Powell", bounds=SHAPE_BOUNDS)
    if not result.success:
        msg = f"the shape hyperparameter fit did not converge: {result.message}"
        raise ValueError(msg)
    return result.x


def fit_shape_hypers_shared(
    *,
    residuals: dict[str, np.ndarray],
    half_hour: np.ndarray,
    structure: ShapeStructure,
    bases: dict[str, ShapeHyper],
    start_log_q_s: float,
    initial_spread: tuple[float, float] = (2.0, 0.5),
) -> dict[str, np.ndarray]:
    """Fit every site's (log `q_s`, log `sigma_eps`) under a prior shared across sites.

    Each round fits every site under the current prior, then replaces the prior's mean and spread
    with the sites' own fits (the spread never below `MIN_SHAPE_PRIOR_SPREAD`), as
    `studies.dp_assimilation.fit_filter_hyperparameters` does for GAM-K's four settings. The first
    prior is centred on `start_log_q_s` and the mean of the sites' GAM-K `sigma_eps`.

    Args:
        residuals: Each site's tuning-month stream.
        half_hour: The local half-hour of each grid row.
        structure: The variant's structure.
        bases: Each site's fixed settings (GAM-K's).
        start_log_q_s: The starting log `q_s` and the first prior's centre.
        initial_spread: The first prior's standard deviations.

    Returns:
        Each site's two fitted values.
    """
    mean_noise = float(np.mean([base.log_sigma_eps for base in bases.values()]))
    prior_mean = np.array([start_log_q_s, mean_noise])
    prior_spread = np.array(initial_spread)
    fitted: dict[str, np.ndarray] = {}
    for _ in range(N_EMPIRICAL_BAYES_ROUNDS):
        fitted = {
            site: fit_shape_hyper(
                residual=stream,
                half_hour=half_hour,
                structure=structure,
                base=bases[site],
                prior_mean=prior_mean,
                prior_spread=prior_spread,
                start=fitted[site] if site in fitted else prior_mean,
            )
            for site, stream in residuals.items()
        }
        stacked = np.stack(list(fitted.values()))
        prior_mean = stacked.mean(axis=0)
        prior_spread = np.maximum(stacked.std(axis=0, ddof=1), MIN_SHAPE_PRIOR_SPREAD)
    return fitted


def fit_free_hyper(
    *,
    residual: np.ndarray,
    half_hour: np.ndarray,
    structure: ShapeStructure,
    prior_mean: np.ndarray,
    prior_spread: np.ndarray,
    start: np.ndarray,
) -> np.ndarray:
    """Fit all five settings (log `q`, logit `phi`, log `sigma_z`, log `sigma_eps`, log `q_s`).

    For the variant with one level and one transient (`KS1`), whose four GAM-K settings do not
    carry over from the four-band filter.

    Args:
        residual: The stream with every reading outside the tuning months set to `NaN`.
        half_hour: The local half-hour of each grid row.
        structure: The variant's structure.
        prior_mean: The prior mean of the five unconstrained values.
        prior_spread: The prior standard deviation of each.
        start: Where the optimiser starts.

    Returns:
        The five fitted values.

    Raises:
        ValueError: If the optimiser does not report success.
    """
    bounds = (*BOUNDS, SHAPE_BOUNDS[0])

    def negative_posterior(values: np.ndarray) -> float:
        trial = ShapeHyper(*(float(v) for v in values))
        likelihood = run_shape_filter(
            residual=residual,
            half_hour=half_hour,
            structure=structure,
            hyper=trial,
            keep_states=False,
        ).log_likelihood
        prior = -0.5 * float((((values - prior_mean) / prior_spread) ** 2).sum())
        return -(likelihood + prior)

    result = minimize(negative_posterior, x0=start, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        result = minimize(negative_posterior, x0=result.x, method="Powell", bounds=bounds)
    if not result.success:
        msg = f"the free hyperparameter fit did not converge: {result.message}"
        raise ValueError(msg)
    return result.x
