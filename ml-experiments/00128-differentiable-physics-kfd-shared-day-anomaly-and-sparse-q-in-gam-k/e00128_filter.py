"""The joint level, anomaly, and day-level Kalman filter of experiment 128 (KFD), in plain NumPy.

**The state has nine entries: four band levels, four band anomalies, and one day-level state d.**
A reading in band `b` at half-hour `t` is `r = level_b + z_b + d + eps`, in points of effective
capacity.

- The levels `level_b` are random walks with variance `q_b` per day. A band's level starts to move
  at the band's first reading, as in `studies.dp_assimilation`, whose per-band filter this one
  matches when d is switched off (`sigma_d = 0`).
- The anomalies `z_b` are AR(1) with factor `phi` per half-hour and innovation standard deviation
  `sigma_z`.
- The day-level state `d` is constant within a local (Europe/London) calendar day and follows an
  AR(1) with factor `phi_d` per day and innovation standard deviation `sigma_d`. Across a gap that
  crosses `m` local midnights, d decays by `phi_d**m`, its variance moves towards the stationary
  value `sigma_d**2 / (1 - phi_d**2)`, and its covariance with every other state is scaled by
  `phi_d**m`.
- The observation noise has standard deviation `sigma_eps`.

The filter runs over all readings in time order, so every band's reading updates d and d's estimate
updates the other bands. A missing reading is skipped. A forecast uses the state after the newest
reading at least 3.5 hours before the origin and holds it: the level without decay, the anomaly
decayed by `phi` per half-hour from the state's reading, and d decayed by `phi_d` per local day from
the state's local day (`correction_from_state`).

The module also holds the day-ahead likelihood, a simulator of streams from the model, and the
helpers the gate reads (end-of-day series, day-ahead absolute error).
"""

from dataclasses import dataclass, replace
from math import log, pi
from typing import Final

import numpy as np
from studies.dp_effects_filter import DayAheadTargets

N_BANDS: Final[int] = 4
N_STATE: Final[int] = 2 * N_BANDS + 1
LEVEL: Final[slice] = slice(0, N_BANDS)
ANOMALY: Final[slice] = slice(N_BANDS, 2 * N_BANDS)
D_INDEX: Final[int] = 2 * N_BANDS
HALF_HOURS_PER_DAY: Final[int] = 48
INITIAL_LEVEL_VARIANCE: Final[float] = 100.0
"""The prior variance of every level at the start, in points squared (GAM-K's value)."""
VARIANCE_FLOOR: Final[float] = 1e-12
SYMMETRISE_EVERY: Final[int] = 512
N_PARAMETERS: Final[int] = 6
"""Unconstrained hyperparameters of KD: log q, logit phi, log sigma_z, log sigma_eps, logit phi_d,
log sigma_d."""


@dataclass(frozen=True)
class JointParams:
    """The filter's constants.

    Attributes:
        q: Level random-walk variance per day for each band, in points squared, shape (4,).
        phi: The anomaly's AR(1) factor per half-hour.
        sigma_z: The anomaly's innovation standard deviation, in points.
        sigma_eps: The observation noise standard deviation, in points.
        phi_d: The day-level state's AR(1) factor per local day.
        sigma_d: The day-level state's innovation standard deviation, in points; 0 switches d off.
    """

    q: np.ndarray
    phi: float
    sigma_z: float
    sigma_eps: float
    phi_d: float
    sigma_d: float

    def with_q(self, *, q: np.ndarray) -> JointParams:
        """Return the parameters with other level variances."""
        return replace(self, q=np.asarray(q, dtype=np.float64))

    def without_d(self) -> JointParams:
        """Return the parameters with the day-level state switched off."""
        return replace(self, sigma_d=0.0)

    @property
    def stationary_z(self) -> float:
        """The stationary variance of an anomaly."""
        return self.sigma_z**2 / (1.0 - self.phi**2)

    @property
    def stationary_d(self) -> float:
        """The stationary variance of the day-level state."""
        return self.sigma_d**2 / (1.0 - self.phi_d**2)

    def to_json(self) -> dict:
        """Return the parameters as JSON-ready values."""
        return {
            "q": self.q.tolist(),
            "phi": self.phi,
            "sigma_z": self.sigma_z,
            "sigma_eps": self.sigma_eps,
            "phi_d": self.phi_d,
            "sigma_d": self.sigma_d,
        }

    @staticmethod
    def from_json(*, content: dict) -> JointParams:
        """Rebuild parameters from `to_json`'s output."""
        return JointParams(
            q=np.array(content["q"], dtype=np.float64),
            phi=float(content["phi"]),
            sigma_z=float(content["sigma_z"]),
            sigma_eps=float(content["sigma_eps"]),
            phi_d=float(content["phi_d"]),
            sigma_d=float(content["sigma_d"]),
        )


def logistic(value: float) -> float:
    """Return the logistic function of a number."""
    return float(1.0 / (1.0 + np.exp(-value)))


def params_from_vector(*, values: np.ndarray) -> JointParams:
    """Build KD's parameters (one q shared by the four bands) from six unconstrained values."""
    return JointParams(
        q=np.full(N_BANDS, float(np.exp(values[0]))),
        phi=logistic(values[1]),
        sigma_z=float(np.exp(values[2])),
        sigma_eps=float(np.exp(values[3])),
        phi_d=logistic(values[4]),
        sigma_d=float(np.exp(values[5])),
    )


def vector_from_params(*, params: JointParams) -> np.ndarray:
    """Return KD's six unconstrained values from parameters whose four q's are equal."""
    return np.array(
        [
            log(float(params.q[0])),
            log(params.phi / (1.0 - params.phi)),
            log(params.sigma_z),
            log(params.sigma_eps),
            log(params.phi_d / (1.0 - params.phi_d)),
            log(max(params.sigma_d, 1e-12)),
        ]
    )


def gamk_params(*, hyper: list[float]) -> JointParams:
    """Return GAM-K's parameters from experiment 111's four unconstrained values.

    The four levels share one variance and d is off. The values are log q, logit phi, log sigma_z,
    and log sigma_eps.
    """
    return JointParams(
        q=np.full(N_BANDS, float(np.exp(hyper[0]))),
        phi=logistic(hyper[1]),
        sigma_z=float(np.exp(hyper[2])),
        sigma_eps=float(np.exp(hyper[3])),
        phi_d=0.5,
        sigma_d=0.0,
    )


@dataclass(frozen=True)
class JointTrace:
    """The filter's state after each reading.

    Attributes:
        positions: Grid rows of the readings, ascending.
        day: The local day of each reading.
        state: The state after each reading, shape (readings, 9); empty if not kept.
        sd: The posterior standard deviation of each state entry, shape (readings, 9).
        cov_level_d: The posterior covariance of each level with d, shape (readings, 4).
        innovation: The one-step innovation of each reading.
        innovation_variance: The one-step innovation variance of each reading.
        d_share: The share of each reading's innovation that goes to d: d's Kalman gain over the
            sum of the gains of the reading's level, anomaly, and d.
        log_likelihood: The one-step log likelihood of the counted readings.
        day_ahead_log_likelihood: The day-ahead log likelihood of the targets, 0 with no targets.
        params: The constants the filter ran with.
    """

    positions: np.ndarray
    day: np.ndarray
    state: np.ndarray
    sd: np.ndarray
    cov_level_d: np.ndarray
    innovation: np.ndarray
    innovation_variance: np.ndarray
    d_share: np.ndarray
    log_likelihood: float
    day_ahead_log_likelihood: float
    params: JointParams


def _day_ahead_groups(
    *, targets: DayAheadTargets | None, positions: np.ndarray
) -> dict[int, np.ndarray]:
    """Group target indices by the reading whose state their forecast uses."""
    if targets is None or len(targets.target_rows) == 0:
        return {}
    found = np.searchsorted(positions, targets.cutoff_rows, side="right") - 1
    groups: dict[int, np.ndarray] = {}
    valid = np.flatnonzero(found >= 0)
    for reading in np.unique(found[valid]):
        groups[int(reading)] = valid[found[valid] == reading]
    return groups


def _predictive_log_likelihood(
    *,
    x: np.ndarray,
    covariance: np.ndarray,
    grow: np.ndarray,
    params: JointParams,
    reading_position: int,
    reading_day: int,
    target_rows: np.ndarray,
    target_band: np.ndarray,
    target_day: np.ndarray,
    values: np.ndarray,
) -> float:
    """Return the summed log density of targets predicted from one state, each on its own."""
    gap = (target_rows - reading_position).astype(np.float64)
    midnights = np.maximum(target_day - reading_day, 0).astype(np.float64)
    decay_z = params.phi**gap
    decay_d = params.phi_d**midnights
    design = np.zeros((len(gap), N_STATE))
    rows = np.arange(len(gap))
    design[rows, target_band] = 1.0
    design[rows, N_BANDS + target_band] = decay_z
    design[rows, D_INDEX] = decay_d
    mean = design @ x
    variance = (
        np.einsum("ij,ij->i", design @ covariance, design)
        + grow[target_band] * gap
        + params.stationary_z * (1.0 - decay_z**2)
        + params.stationary_d * (1.0 - decay_d**2)
        + params.sigma_eps**2
    )
    return float(-0.5 * np.sum(np.log(2.0 * pi * variance) + (values - mean) ** 2 / variance))


def run_joint_filter(  # noqa: PLR0915 - one flat hot loop, kept whole for speed
    *,
    residual: np.ndarray,
    band: np.ndarray,
    day: np.ndarray,
    params: JointParams,
    likelihood_mask: np.ndarray | None = None,
    keep_trace: bool = True,
    day_ahead: DayAheadTargets | None = None,
) -> JointTrace:
    """Run the filter over a residual stream.

    Args:
        residual: Observed minus the frozen forecast on the grid, in points, `NaN` where there is
            no reading. The caller includes the switching rows: the filter has no mask.
        band: The hour band of each grid row, 0 to 3.
        day: The local calendar day of each grid row, an integer that rises by 1 at each local
            midnight.
        params: The filter's constants.
        likelihood_mask: True on the grid rows whose readings count towards the one-step
            likelihood; the state is updated by every reading regardless. `None` counts all.
        keep_trace: Whether to keep the state, the posterior standard deviations, and the
            innovations of every reading.
        day_ahead: If given, the day-ahead log likelihood of these targets is computed from the
            states at their cutoffs and returned in `day_ahead_log_likelihood`. The one-step
            likelihood is still returned.

    Returns:
        The trace.
    """
    positions = np.flatnonzero(np.isfinite(residual))
    n = len(positions)
    values = residual[positions]
    bands = band[positions].astype(np.int64)
    days = day[positions].astype(np.int64)
    counted = (
        np.ones(n, dtype=bool) if likelihood_mask is None else likelihood_mask[positions].copy()
    )
    groups = _day_ahead_groups(targets=day_ahead, positions=positions)
    stationary_z, stationary_d = params.stationary_z, params.stationary_d
    noise = params.sigma_eps**2
    q_step = params.q / HALF_HOURS_PER_DAY
    phi, phi_d = params.phi, params.phi_d
    x = np.zeros(N_STATE)
    covariance = np.zeros((N_STATE, N_STATE))
    diagonal = covariance.reshape(-1)[:: N_STATE + 1]
    diagonal[LEVEL] = INITIAL_LEVEL_VARIANCE
    diagonal[ANOMALY] = stationary_z
    diagonal[D_INDEX] = stationary_d
    grow = np.zeros(N_STATE)
    state = np.empty((n, N_STATE)) if keep_trace else np.empty((0, N_STATE))
    sd = np.empty((n, N_STATE)) if keep_trace else np.empty((0, N_STATE))
    cov_level_d = np.empty((n, N_BANDS)) if keep_trace else np.empty((0, N_BANDS))
    innovations = np.empty(n) if keep_trace else np.empty(0)
    innovation_variances = np.empty(n) if keep_trace else np.empty(0)
    d_shares = np.empty(n) if keep_trace else np.empty(0)
    log_likelihood = 0.0
    day_ahead_ll = 0.0
    scale = np.ones(N_STATE)
    extra = np.zeros(N_STATE)
    previous_position = int(positions[0]) if n else 0
    previous_day = int(days[0]) if n else 0
    seen = np.zeros(N_BANDS, dtype=bool)
    for k in range(n):
        position = int(positions[k])
        gap = position - previous_position
        if gap:
            midnights = max(int(days[k]) - previous_day, 0)
            decay_z = phi**gap
            decay_d = phi_d**midnights
            scale[ANOMALY] = decay_z
            scale[D_INDEX] = decay_d
            covariance *= np.multiply.outer(scale, scale)
            extra[LEVEL] = grow[LEVEL] * gap
            extra[ANOMALY] = stationary_z * (1.0 - decay_z * decay_z)
            extra[D_INDEX] = stationary_d * (1.0 - decay_d * decay_d)
            diagonal += extra
            x *= scale
        previous_position = position
        previous_day = int(days[k])
        b = int(bands[k])
        z = N_BANDS + b
        p_h = covariance[:, b] + covariance[:, z] + covariance[:, D_INDEX]
        innovation_variance = p_h[b] + p_h[z] + p_h[D_INDEX] + noise
        innovation = values[k] - (x[b] + x[z] + x[D_INDEX])
        if counted[k]:
            log_likelihood -= 0.5 * (
                log(2.0 * pi * innovation_variance) + innovation * innovation / innovation_variance
            )
        gain = p_h / innovation_variance
        x += gain * innovation
        covariance -= np.multiply.outer(gain, p_h)
        np.maximum(diagonal, VARIANCE_FLOOR, out=diagonal)
        if not seen[b]:
            seen[b] = True
            grow[b] = q_step[b]
        if k % SYMMETRISE_EVERY == SYMMETRISE_EVERY - 1:
            covariance[:] = 0.5 * (covariance + covariance.T)
        if keep_trace:
            state[k] = x
            sd[k] = np.sqrt(diagonal)
            cov_level_d[k] = covariance[D_INDEX, :N_BANDS]
            innovations[k] = innovation
            innovation_variances[k] = innovation_variance
            d_shares[k] = gain[D_INDEX] / (gain[b] + gain[z] + gain[D_INDEX])
        if k in groups:
            chosen = groups[k]
            assert day_ahead is not None
            rows = day_ahead.target_rows[chosen]
            day_ahead_ll += _predictive_log_likelihood(
                x=x,
                covariance=covariance,
                grow=grow,
                params=params,
                reading_position=position,
                reading_day=int(days[k]),
                target_rows=rows,
                target_band=band[rows].astype(np.int64),
                target_day=day[rows].astype(np.int64),
                values=residual[rows],
            )
    return JointTrace(
        positions=positions,
        day=days,
        state=state,
        sd=sd,
        cov_level_d=cov_level_d,
        innovation=innovations,
        innovation_variance=innovation_variances,
        d_share=d_shares,
        log_likelihood=log_likelihood,
        day_ahead_log_likelihood=day_ahead_ll,
        params=params,
    )


@dataclass(frozen=True)
class JointState:
    """The state a forecast holds.

    Attributes:
        x: The state, shape (9,); zeros if no reading is old enough.
        position: The grid row of the reading the state follows, -1 if none.
        day: The local day of that reading, -1 if none.
    """

    x: np.ndarray
    position: int
    day: int


def state_at(*, trace: JointTrace, cutoff: int) -> JointState:
    """Return the state after the newest reading at or before the grid row `cutoff`."""
    found = int(np.searchsorted(trace.positions, cutoff, side="right")) - 1
    if found < 0:
        return JointState(x=np.zeros(N_STATE), position=-1, day=-1)
    return JointState(
        x=trace.state[found].copy(), position=int(trace.positions[found]), day=int(trace.day[found])
    )


@dataclass(frozen=True)
class JointCorrection:
    """The correction of each target of one run, in points.

    Attributes:
        level: The held level of each target's band.
        anomaly: The decayed anomaly of each target's band.
        day_term: The decayed day-level state.
    """

    level: np.ndarray
    anomaly: np.ndarray
    day_term: np.ndarray

    @property
    def total(self) -> np.ndarray:
        """The whole correction."""
        return self.level + self.anomaly + self.day_term


def correction_from_state(
    *,
    state: JointState,
    params: JointParams,
    band: np.ndarray,
    target: np.ndarray,
    target_day: np.ndarray,
    day_value: float | None = None,
) -> JointCorrection:
    """Return the correction for targets: held level, decayed anomaly, decayed day-level state.

    Args:
        state: The state at the cutoff.
        params: The filter's constants.
        band: The hour band of each target.
        target: The grid row of each target.
        target_day: The local day of each target.
        day_value: If given, replaces d in the day term (the shuffled-day check). `None` uses the
            state's own d.

    Returns:
        The correction; zeros if the state follows no reading.
    """
    n = len(target)
    if state.position < 0:
        return JointCorrection(level=np.zeros(n), anomaly=np.zeros(n), day_term=np.zeros(n))
    d = float(state.x[D_INDEX]) if day_value is None else day_value
    half_hours = np.maximum(target - state.position, 0)
    midnights = np.maximum(target_day - state.day, 0)
    return JointCorrection(
        level=state.x[LEVEL][band],
        anomaly=params.phi**half_hours * state.x[ANOMALY][band],
        day_term=params.phi_d**midnights * d,
    )


def day_ahead_corrections(
    *,
    trace: JointTrace,
    rows: np.ndarray,
    cutoffs: np.ndarray,
    band: np.ndarray,
    day: np.ndarray,
) -> JointCorrection:
    """Return the correction of many grid rows, each from the state at its own cutoff.

    A row whose cutoff precedes every reading gets a correction of zero.

    Args:
        trace: The filter's trace with its state kept.
        rows: The grid rows being forecast.
        cutoffs: The newest grid row each forecast may see.
        band: The hour band of every grid row.
        day: The local day of every grid row.

    Returns:
        The corrections, one per row.
    """
    found = np.searchsorted(trace.positions, cutoffs, side="right") - 1
    known = found >= 0
    index = np.where(known, found, 0)
    x = trace.state[index]
    half_hours = np.maximum(rows - trace.positions[index], 0)
    midnights = np.maximum(day[rows] - trace.day[index], 0)
    pick = np.arange(len(rows))
    params = trace.params
    return JointCorrection(
        level=np.where(known, x[pick, band[rows]], 0.0),
        anomaly=np.where(known, params.phi**half_hours * x[pick, N_BANDS + band[rows]], 0.0),
        day_term=np.where(known, params.phi_d**midnights * x[:, D_INDEX], 0.0),
    )


def end_of_day_series(*, trace: JointTrace, values: np.ndarray, n_days: int) -> np.ndarray:
    """Return a per-reading quantity at the end of each local day, holding it over a day with none.

    Args:
        trace: The filter's trace.
        values: The quantity after each reading, shape (readings,).
        n_days: The number of days to return; day `i` ends after local day `i + 1` starts at 0.

    Returns:
        The value after the last reading of local day `i` (counting from 0), shape (n_days,); `NaN`
        before the first reading.
    """
    out = np.full(n_days, np.nan)
    last = np.searchsorted(trace.day, np.arange(n_days), side="right") - 1
    known = last >= 0
    out[known] = values[last[known]]
    return out


def simulate_stream(
    *,
    params: JointParams,
    band: np.ndarray,
    day: np.ndarray,
    present: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Simulate a residual stream from the model on the rows where a reading is present.

    The levels start at 0 and walk with variance `q_b / 48` per half-hour, the anomalies and d start
    from their stationary distributions, and the readings add the observation noise.

    Args:
        params: The model's constants; `sigma_d = 0` leaves d out.
        band: The hour band of each row.
        day: The local day of each row.
        present: True on the rows that hold a reading.
        generator: The random generator.

    Returns:
        The stream, `NaN` where `present` is false.
    """
    n = len(band)
    out = np.full(n, np.nan)
    rows = np.flatnonzero(present)
    stationary_z = params.stationary_z
    total = np.zeros(len(rows))
    for b in range(N_BANDS):
        index = np.flatnonzero(band[rows] == b)
        if len(index) == 0:
            continue
        positions = rows[index]
        gaps = np.diff(positions, prepend=positions[0]).astype(np.float64)
        level = np.cumsum(
            generator.normal(size=len(index)) * np.sqrt(params.q[b] / HALF_HOURS_PER_DAY * gaps)
        )
        decay = params.phi**gaps
        shocks = generator.normal(size=len(index))
        z = np.empty(len(index))
        previous = np.sqrt(stationary_z) * generator.normal()
        for k in range(len(index)):
            previous = (
                decay[k] * previous + np.sqrt(stationary_z * (1.0 - decay[k] ** 2)) * shocks[k]
            )
            z[k] = previous
        total[index] = level + z
    if params.sigma_d > 0.0 and len(rows):
        first, last = int(day[rows].min()), int(day[rows].max())
        d = np.empty(last - first + 1)
        previous = np.sqrt(params.stationary_d) * generator.normal()
        for k in range(len(d)):
            previous = params.phi_d * previous + params.sigma_d * generator.normal()
            d[k] = previous
        total += d[day[rows] - first]
    out[rows] = total + params.sigma_eps * generator.normal(size=len(rows))
    return out
