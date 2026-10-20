"""Flows on the interconnection edges between Primary substations, for the DP series' NRA model.

**A transfer shows up as matched, opposite steps in the residuals of two connected substations.**
When the network operator moves part of one Primary substation's load onto a neighbour, the donor's
metered power falls by the moved megawatts and the receiver's rises by the same amount, for hours
or days. Growth, a new connection, or a changed behaviour shows at one substation only. This module
fits one flow path per interconnection edge to the residuals (observed minus forecast) of a frozen
convex model, with a penalty that keeps the flows sparse, so a conserved transfer is explained by
the flow on its edge, and a one-sided change is left in the residual for the level state of the
Kalman filter (`studies.dp_assimilation`).

**The model.** Each node `i` has residual `r_i(t)` in megawatts. Each path `p` carries a flow
`f_p(t)`, and touches one or two nodes: on an edge between `i` and `j` the flow adds `+f` to
node `i` and `-f` to node `j`, and the outside path of a node adds `+g` to that node alone and
stands for transfers to unmetered partners. The fit minimises, over all paths,

    sum over nodes and rows:  w_i * huber(r_i - sum of the path flows at i; delta_i)
    + sum over paths:  step_penalty * sum |f_t - f_(t-1)|  +  level_penalty * sum |f_t|

where `huber` is the Huber loss, `w_i` is `1 / sigma_i**2`, and `delta_i` is 1.5 `sigma_i`. The
step penalty keeps flows piecewise constant, and the level penalty keeps the network at its normal
running arrangement (NRA) unless the data insist.

**Each path is solved exactly by a dynamic programme** in the style of
[Johnson (2013)](https://doi.org/10.1080/10618600.2012.681238), who states it for squared error.
The derivative of the value function is kept as a piecewise-linear function, clipped to the step
penalty at each row, and the flow is recovered by clipping backwards. The extension to the Huber
loss, per-row weights, and a level penalty is this module's; the tests check it against a
quadratic-programme reference. A flow constrained to be non-negative, and the tests, use ADMM (the
alternating direction method of multipliers) with a banded Cholesky factor instead.
**Paths are fitted by block-coordinate descent:** each path is re-solved on the
residual left after the other paths at its nodes are removed, until the objective and the largest
path change stop moving. A row where a path's nodes have no reading is dropped from that path's
chain, so a gap neither pays the level penalty nor ends a flow.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from scipy.linalg import cho_solve_banded, cholesky_banded

from studies.dp_assimilation import (
    INITIAL_LEVEL_VARIANCE,
    VARIANCE_FLOOR,
    CorrectionTrace,
    FilterHyperparameters,
)
from studies.dp_calendar import HALF_HOURS_PER_DAY
from studies.dp_gam import N_BANDS, ResidualSpread

MAD_TO_STANDARD_DEVIATION: Final[float] = 1.4826
"""The factor that turns a median absolute deviation into a normal standard deviation."""

HUBER_THRESHOLD_IN_SIGMAS: Final[float] = 1.5
"""The Huber loss is quadratic within this many robust residual standard deviations of zero."""

DEFAULT_TOLERANCE: Final[float] = 1e-8
"""The ADMM stops when its primal and dual residuals are both under this, in megawatts."""

DEFAULT_MAX_ITERATIONS: Final[int] = 20_000
"""The most ADMM iterations of one path solve."""

BALANCE_RATIO: Final[float] = 10.0
"""The penalty parameter is rescaled when one residual is this many times the other."""

BALANCE_FACTOR: Final[float] = 2.0
"""The penalty parameter is multiplied or divided by this when the residuals are unbalanced."""

BALANCE_INTERVAL: Final[int] = 25
"""The penalty parameter is considered for rescaling every this many iterations."""

INITIAL_PENALTY_PARAMETER: Final[float] = 1.0
"""The starting ADMM penalty parameter."""

SolverMethodType = Literal["dynamic_programme", "admm"]
"""Which algorithm solves a path problem."""

DEFAULT_MAX_SWEEPS: Final[int] = 20
"""The most block-coordinate sweeps over all paths."""

SWEEP_TOLERANCE: Final[float] = 1e-6
"""A sweep converges when the objective's relative change and the largest path change in
megawatts are both under this."""


@dataclass(frozen=True)
class PathProblem:
    """One path's problem: fit a flow to the residuals at its one or two ends.

    The flow `s` adds `+gain_a * s` to end a and `-gain_b * s` to end b, so a transfer of load onto
    end a from end b is a positive flow. Missing readings are `NaN` and carry no weight.

    Attributes:
        residual_a: End a's residual in megawatts, `NaN` where there is no reading.
        noise_a: End a's robust residual standard deviation in megawatts.
        residual_b: End b's residual, or `None` for an outside path with one end.
        noise_b: End b's, ignored when there is no end b.
        reliability_a: Optional weight per row for end a in 0 to 1 (for example the share of a
            block that holds a reading); `None` means 1 wherever there is a reading.
        reliability_b: The same for end b.
        gain_a: Optional non-negative gain per row at end a (a shape in megawatts per unit of
            share); `None` means 1.
        gain_b: The same for end b.
        step_penalty: Penalty per megawatt of change between consecutive rows.
        level_penalty: Penalty per megawatt of flow on a row with a reading.
        non_negative: Whether the flow is constrained to be at least zero.
    """

    residual_a: np.ndarray
    noise_a: float
    residual_b: np.ndarray | None = None
    noise_b: float = 1.0
    reliability_a: np.ndarray | None = None
    reliability_b: np.ndarray | None = None
    gain_a: np.ndarray | None = None
    gain_b: np.ndarray | None = None
    step_penalty: float = 0.0
    level_penalty: float = 0.0
    non_negative: bool = False


@dataclass(frozen=True)
class PathSolution:
    """The solution of one path problem.

    Attributes:
        flow: The flow on every row of the problem's grid, in megawatts, held flat across rows
            with no reading and zero before the first reading.
        iterations: How many ADMM iterations ran.
        converged: Whether both residuals fell under the tolerance.
        objective: The problem's objective at `flow`.
    """

    flow: np.ndarray
    iterations: int
    converged: bool
    objective: float


@dataclass(frozen=True)
class _End:
    """One end of a compressed path: weighted residual, Huber threshold, and gain."""

    value: np.ndarray
    weight: np.ndarray
    threshold: float
    gain: np.ndarray


def huber_loss(*, x: np.ndarray, threshold: float) -> np.ndarray:
    """Return the Huber loss: `x**2 / 2` within the threshold, linear beyond.

    Args:
        x: The residuals.
        threshold: The half-width of the quadratic region.

    Returns:
        The loss of each residual.
    """
    magnitude = np.abs(x)
    return np.where(magnitude <= threshold, 0.5 * x * x, threshold * (magnitude - 0.5 * threshold))


def _row_weight(
    *, residual: np.ndarray, noise: float, reliability: np.ndarray | None
) -> np.ndarray:
    """Return the weight of each row: `1 / noise**2` where there is a reading, else 0."""
    weight = np.where(np.isfinite(residual), 1.0 / noise**2, 0.0)
    return weight if reliability is None else weight * reliability


def _compress(*, problem: PathProblem) -> tuple[np.ndarray, list[_End]]:
    """Drop the rows where no end has a reading, and package the ends.

    Args:
        problem: The path problem.

    Returns:
        The grid positions of the kept rows, and the kept rows' ends (one or two).
    """
    ends_raw = [(problem.residual_a, problem.noise_a, problem.reliability_a, problem.gain_a)]
    if problem.residual_b is not None:
        ends_raw.append(
            (problem.residual_b, problem.noise_b, problem.reliability_b, problem.gain_b)
        )
    weights = [
        _row_weight(residual=residual, noise=noise, reliability=reliability)
        for residual, noise, reliability, _ in ends_raw
    ]
    kept = np.flatnonzero(np.any([w > 0.0 for w in weights], axis=0))
    ends = [
        _End(
            value=np.nan_to_num(residual[kept]),
            weight=weight[kept],
            threshold=HUBER_THRESHOLD_IN_SIGMAS * noise,
            gain=np.ones(len(kept)) if gain is None else gain[kept],
        )
        for (residual, noise, _, gain), weight in zip(ends_raw, weights, strict=True)
    ]
    return kept, ends


def _signs(*, ends: Sequence[_End]) -> list[float]:
    """Return the sign the flow takes at each end: plus at end a, minus at end b."""
    return [1.0, -1.0][: len(ends)]


def _objective(
    *, flow: np.ndarray, ends: Sequence[_End], step_penalty: float, level_penalty: float
) -> float:
    """Return the objective of a compressed path at a flow."""
    total = 0.0
    for end, sign in zip(ends, _signs(ends=ends), strict=True):
        total += float(
            np.sum(
                end.weight
                * huber_loss(x=end.value - sign * end.gain * flow, threshold=end.threshold)
            )
        )
    total += step_penalty * float(np.sum(np.abs(np.diff(flow))))
    total += level_penalty * float(np.sum(np.abs(flow)))
    return total


def path_objective(*, problem: PathProblem, flow: np.ndarray) -> float:
    """Return a path problem's objective at a flow given on the problem's whole grid.

    Rows with no reading are dropped before the step penalty is applied, as the solver does.

    Args:
        problem: The path problem.
        flow: The flow on the whole grid, in megawatts.

    Returns:
        The objective: the weighted Huber losses plus the step and level penalties.
    """
    kept, ends = _compress(problem=problem)
    return _objective(
        flow=flow[kept],
        ends=ends,
        step_penalty=problem.step_penalty,
        level_penalty=problem.level_penalty,
    )


def _difference_gram_banded(*, n: int) -> np.ndarray:
    """Return `D^T D` of the first-difference matrix in upper banded form, shape (2, n)."""
    diagonal = np.full(n, 2.0)
    diagonal[0] = 1.0
    diagonal[-1] = 1.0
    banded = np.zeros((2, n))
    banded[0, 1:] = -1.0
    banded[1] = diagonal
    return banded


def _huber_prox(*, m: np.ndarray, weight: np.ndarray, threshold: float, rho: float) -> np.ndarray:
    """Return argmin over u of `weight * huber(u) + rho / 2 * (u - m)**2`, elementwise."""
    quadratic = rho * m / (rho + weight)
    linear = m - np.sign(m) * weight * threshold / rho
    return np.where(np.abs(m) <= threshold * (1.0 + weight / rho), quadratic, linear)


def _soft_threshold(*, x: np.ndarray, level: float) -> np.ndarray:
    """Return the soft-thresholding of `x` at `level`."""
    return np.sign(x) * np.maximum(np.abs(x) - level, 0.0)


def _prox_level(*, x: np.ndarray, level: float, non_negative: bool) -> np.ndarray:
    """Return the proximal map of the level penalty, with the optional bound at zero."""
    shrunk = _soft_threshold(x=x, level=level)
    return np.maximum(x - level, 0.0) if non_negative else shrunk


@dataclass
class _AdmmState:
    """The variables of one ADMM solve: the flow, the splitting variables, and scaled duals."""

    s: np.ndarray
    u: list[np.ndarray]
    dual_u: list[np.ndarray]
    p: np.ndarray
    dual_p: np.ndarray
    q: np.ndarray
    dual_q: np.ndarray


def _admm(
    *,
    ends: Sequence[_End],
    step_penalty: float,
    level_penalty: float,
    non_negative: bool,
    tolerance: float,
    max_iterations: int,
    start: np.ndarray | None,
) -> tuple[np.ndarray, int, bool]:
    """Solve a compressed path problem by ADMM.

    The splitting is `u_k = value_k - sign_k * gain_k * s` for each end, `p = D s`, and `q = s`.
    All constraints share one penalty parameter, so `(sum of gain**2 + D^T D + I) s = rhs` has a
    fixed matrix, factorised once.

    Args:
        ends: The compressed ends.
        step_penalty: Penalty per megawatt of step.
        level_penalty: Penalty per megawatt of level.
        non_negative: Whether the flow is constrained to be at least zero.
        tolerance: The stopping tolerance on the primal and dual residuals in megawatts.
        max_iterations: The iteration limit.
        start: A warm start for the flow, or `None` for zero.

    Returns:
        The flow, the iteration count, and whether both residuals fell under the tolerance.
    """
    n = len(ends[0].value)
    signs = _signs(ends=ends)
    system = _difference_gram_banded(n=n)
    system[1] += 1.0 + sum(end.gain**2 for end in ends)
    factor = cholesky_banded(system)
    s = np.zeros(n) if start is None else start.copy()
    u = [end.value - sign * end.gain * s for end, sign in zip(ends, signs, strict=True)]
    dual_u = [np.zeros(n) for _ in ends]
    p = np.diff(s)
    dual_p = np.zeros(n - 1)
    q = s.copy()
    dual_q = np.zeros(n)
    rho = INITIAL_PENALTY_PARAMETER
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        rhs = _right_hand_side(
            ends=ends, signs=signs, u=u, dual_u=dual_u, p=p, dual_p=dual_p, q=q, dual_q=dual_q
        )
        s = cho_solve_banded((factor, False), rhs)
        ds = np.diff(s)
        change = np.zeros(n)
        new_u = []
        for end, sign, u_k, eta in zip(ends, signs, u, dual_u, strict=True):
            m = end.value - sign * end.gain * s - eta
            u_new = _huber_prox(m=m, weight=end.weight, threshold=end.threshold, rho=rho)
            new_u.append(u_new)
            change += sign * end.gain * (u_new - u_k)
        p_new = _soft_threshold(x=ds - dual_p, level=step_penalty / rho)
        q_new = _prox_level(x=s - dual_q, level=level_penalty / rho, non_negative=non_negative)
        delta_p = p_new - p
        transposed = np.concatenate(([0.0], delta_p)) - np.concatenate((delta_p, [0.0]))
        dual_residual = float(np.max(np.abs(change - transposed - (q_new - q))))
        primal_residual = max(
            *(
                float(np.max(np.abs(u_new + sign * end.gain * s - end.value)))
                for end, sign, u_new in zip(ends, signs, new_u, strict=True)
            ),
            float(np.max(np.abs(p_new - ds), initial=0.0)),
            float(np.max(np.abs(q_new - s))),
        )
        for k, (end, sign) in enumerate(zip(ends, signs, strict=True)):
            dual_u[k] = dual_u[k] + new_u[k] + sign * end.gain * s - end.value
        dual_p = dual_p + p_new - ds
        dual_q = dual_q + q_new - s
        u, p, q = new_u, p_new, q_new
        if primal_residual < tolerance and dual_residual < tolerance:
            converged = True
            break
        if iteration % BALANCE_INTERVAL == 0:
            rho, scale = _rebalance(rho=rho, primal=primal_residual, dual=dual_residual)
            if scale != 1.0:
                dual_u = [eta * scale for eta in dual_u]
                dual_p = dual_p * scale
                dual_q = dual_q * scale
    return s, iteration, converged


def _right_hand_side(
    *,
    ends: Sequence[_End],
    signs: Sequence[float],
    u: Sequence[np.ndarray],
    dual_u: Sequence[np.ndarray],
    p: np.ndarray,
    dual_p: np.ndarray,
    q: np.ndarray,
    dual_q: np.ndarray,
) -> np.ndarray:
    """Return the right-hand side of the ADMM flow update."""
    rhs = q + dual_q
    for end, sign, u_k, eta in zip(ends, signs, u, dual_u, strict=True):
        rhs = rhs + sign * end.gain * (end.value - u_k - eta)
    target = p + dual_p
    rhs[:-1] -= target
    rhs[1:] += target
    return rhs


def _rebalance(*, rho: float, primal: float, dual: float) -> tuple[float, float]:
    """Return the new penalty parameter and the factor to scale the scaled duals by."""
    if primal > BALANCE_RATIO * dual:
        return rho * BALANCE_FACTOR, 1.0 / BALANCE_FACTOR
    if dual > BALANCE_RATIO * primal:
        return rho / BALANCE_FACTOR, BALANCE_FACTOR
    return rho, 1.0


def solve_path(
    *,
    problem: PathProblem,
    start: np.ndarray | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    method: SolverMethodType = "dynamic_programme",
) -> PathSolution:
    """Solve one path problem.

    Args:
        problem: The path problem.
        start: A warm start for the flow on the problem's whole grid, or `None`.
        tolerance: The ADMM stopping tolerance in megawatts.
        max_iterations: The ADMM iteration limit.
        method: `dynamic_programme` (exact) or `admm`. A non-negative problem always uses ADMM.

    Returns:
        The flow on the whole grid, held flat across rows with no reading and zero before the
        first reading, with the iteration count, convergence flag, and objective.
    """
    kept, ends = _compress(problem=problem)
    grid_length = len(problem.residual_a)
    if len(kept) == 0:
        return PathSolution(flow=np.zeros(grid_length), iterations=0, converged=True, objective=0.0)
    if (method == "dynamic_programme" or len(kept) == 1) and not problem.non_negative:
        flow_kept = _dynamic_programme(
            ends=ends, step_penalty=problem.step_penalty, level_penalty=problem.level_penalty
        )
        iterations, converged = 0, True
    elif len(kept) == 1:
        # A convex objective in one variable, bounded below by zero, is minimised at the larger of
        # zero and the unconstrained minimum.
        flow_kept = np.maximum(
            _dynamic_programme(ends=ends, step_penalty=0.0, level_penalty=problem.level_penalty),
            0.0,
        )
        iterations, converged = 0, True
    else:
        flow_kept, iterations, converged = _admm(
            ends=ends,
            step_penalty=problem.step_penalty,
            level_penalty=problem.level_penalty,
            non_negative=problem.non_negative,
            tolerance=tolerance,
            max_iterations=max_iterations,
            start=None if start is None else start[kept],
        )
        if problem.non_negative:
            flow_kept = np.maximum(flow_kept, 0.0)
    return PathSolution(
        flow=expand_held_flat(kept=kept, values=flow_kept, length=grid_length),
        iterations=iterations,
        converged=converged,
        objective=_objective(
            flow=flow_kept,
            ends=ends,
            step_penalty=problem.step_penalty,
            level_penalty=problem.level_penalty,
        ),
    )


def _ramp_parameters(*, end: _End, sign: float) -> tuple[np.ndarray, ...]:
    """Return the ramp that an end adds to the derivative of the data term, for every row.

    The derivative with respect to the flow of `weight * huber(value - c * s)`, with `c` equal to
    `sign * gain`, rises from `-A` to `+A` along a ramp of slope `B` between two knots, and is
    flat outside them.

    Args:
        end: The compressed end.
        sign: Plus one at end a and minus one at end b.

    Returns:
        The ramp's lower knot, upper knot, height `A`, and slope `B`, each with one entry per row.
        A row with a zero gain or zero weight has `A = B = 0`.
    """
    c = sign * end.gain
    active = (np.abs(c) > 0.0) & (end.weight > 0.0)
    safe = np.where(active, c, 1.0)
    first = (end.value - end.threshold) / safe
    second = (end.value + end.threshold) / safe
    height = np.where(active, np.abs(c) * end.weight * end.threshold, 0.0)
    slope = np.where(active, c * c * end.weight, 0.0)
    return np.minimum(first, second), np.maximum(first, second), height, slope


@dataclass
class _Derivative:
    """A non-decreasing piecewise-linear function, flat at both ends, with jumps allowed.

    The value is `left_value` before the first knot. At and after knot `i` it is
    `value[i] + slope[i] * (s - knot[i])` up to the next knot.

    Attributes:
        knot: The ascending knot positions.
        value: The right-hand limit at each knot.
        slope: The slope to the right of each knot.
        left_value: The value to the left of the first knot.
    """

    knot: np.ndarray
    value: np.ndarray
    slope: np.ndarray
    left_value: float

    def right_limit(self, *, at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the right-hand limit and the right-hand slope at positions."""
        index = np.searchsorted(self.knot, at, side="right") - 1
        before = index < 0
        safe = np.where(before, 0, index)
        value = np.where(
            before, self.left_value, self.value[safe] + self.slope[safe] * (at - self.knot[safe])
        )
        return value, np.where(before, 0.0, self.slope[safe])

    def left_limits(self) -> np.ndarray:
        """Return the left-hand limit at each knot."""
        lefts = np.empty(len(self.knot))
        lefts[0] = self.left_value
        lefts[1:] = self.value[:-1] + self.slope[:-1] * np.diff(self.knot)
        return lefts

    def first_at_least(self, *, level: float) -> float:
        """Return the smallest position where the function reaches `level`, or `-inf`."""
        if self.left_value >= level:
            return -np.inf
        i = int(np.searchsorted(self.value, level, side="left"))
        if i == len(self.knot):
            return np.inf
        if i > 0 and self.left_limits()[i] >= level:
            return float(self.knot[i - 1] + (level - self.value[i - 1]) / self.slope[i - 1])
        return float(self.knot[i])

    def last_at_most(self, *, level: float) -> float:
        """Return the largest position where the function is at most `level`, or `inf`."""
        if self.value[-1] <= level:
            return np.inf
        j = int(np.searchsorted(self.value, level, side="right"))
        if j > 0 and self.left_limits()[j] > level:
            return float(self.knot[j - 1] + (level - self.value[j - 1]) / self.slope[j - 1])
        return float(self.knot[j])

    def clipped(self, *, limit: float, lower: float, upper: float) -> _Derivative:
        """Return the function clipped to `[-limit, limit]`, given where it reaches those levels."""
        inside = (self.knot > lower) & (self.knot < upper)
        knots: list[np.ndarray] = []
        values: list[np.ndarray] = []
        slopes: list[np.ndarray] = []
        if np.isfinite(lower):
            value, slope = self.right_limit(at=np.array([lower]))
            knots.append(np.array([lower]))
            values.append(np.maximum(value, -limit))
            slopes.append(slope)
        knots.append(self.knot[inside])
        values.append(self.value[inside])
        slopes.append(self.slope[inside])
        if np.isfinite(upper) and upper > lower:
            knots.append(np.array([upper]))
            values.append(np.array([limit]))
            slopes.append(np.zeros(1))
        elif np.isfinite(upper):
            values[0] = np.array([limit])
            slopes[0] = np.zeros(1)
        return _Derivative(
            knot=np.concatenate(knots),
            value=np.concatenate(values),
            slope=np.concatenate(slopes),
            left_value=-limit if np.isfinite(lower) else self.left_value,
        )


def _row_derivative(
    *, ramps: Sequence[tuple[np.ndarray, ...]], row: int, level_penalty: float
) -> _Derivative:
    """Return one row's data-term and level-penalty derivative as a piecewise-linear function."""
    knots, slopes = [], []
    left_value = -level_penalty
    for lower, upper, height, slope in ramps:
        if height[row] > 0.0:
            left_value -= height[row]
            knots += [lower[row], upper[row]]
            slopes += [slope[row], 0.0]
    if level_penalty > 0.0:
        knots.append(0.0)
        slopes.append(0.0)
    order = np.argsort(knots, kind="stable")
    knot = np.asarray(knots)[order]
    derivative = _Derivative(
        knot=knot, value=np.zeros(len(knot)), slope=np.zeros(len(knot)), left_value=left_value
    )
    return _with_values(derivative=derivative, ramps=ramps, row=row, level_penalty=level_penalty)


def _ramp_right_limit(
    *, ramps: Sequence[tuple[np.ndarray, ...]], row: int, at: np.ndarray, level_penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the right-hand limit and slope of one row's derivative at positions."""
    value = np.where(at >= 0.0, level_penalty, -level_penalty)
    slope = np.zeros(len(at))
    for lower, upper, height, ramp_slope in ramps:
        if height[row] > 0.0:
            value = (
                value
                - height[row]
                + ramp_slope[row] * (np.clip(at, lower[row], upper[row]) - lower[row])
            )
            slope = slope + ramp_slope[row] * ((at >= lower[row]) & (at < upper[row]))
    return value, slope


def _with_values(
    *,
    derivative: _Derivative,
    ramps: Sequence[tuple[np.ndarray, ...]],
    row: int,
    level_penalty: float,
) -> _Derivative:
    """Fill a row derivative's knot values and slopes."""
    if len(derivative.knot) == 0:
        return derivative
    value, slope = _ramp_right_limit(
        ramps=ramps, row=row, at=derivative.knot, level_penalty=level_penalty
    )
    return _Derivative(
        knot=derivative.knot, value=value, slope=slope, left_value=derivative.left_value
    )


def _add_row(
    *,
    clipped: _Derivative,
    ramps: Sequence[tuple[np.ndarray, ...]],
    row: int,
    level_penalty: float,
) -> _Derivative:
    """Add one row's derivative to the clipped value-function derivative."""
    row_derivative = _row_derivative(ramps=ramps, row=row, level_penalty=level_penalty)
    union = np.unique(np.concatenate((clipped.knot, row_derivative.knot)))
    old_value, old_slope = clipped.right_limit(at=union)
    new_value, new_slope = row_derivative.right_limit(at=union)
    return _Derivative(
        knot=union,
        value=old_value + new_value,
        slope=old_slope + new_slope,
        left_value=clipped.left_value + row_derivative.left_value,
    )


def _dynamic_programme(
    *, ends: Sequence[_End], step_penalty: float, level_penalty: float
) -> np.ndarray:
    """Solve a compressed path problem exactly, with no non-negativity constraint.

    Args:
        ends: The compressed ends.
        step_penalty: Penalty per megawatt of step.
        level_penalty: Penalty per megawatt of level.

    Returns:
        The minimising flow.
    """
    ramps = [
        _ramp_parameters(end=end, sign=sign)
        for end, sign in zip(ends, _signs(ends=ends), strict=True)
    ]
    n = len(ends[0].value)
    lowers = np.empty(n)
    uppers = np.empty(n)
    derivative = _row_derivative(ramps=ramps, row=0, level_penalty=level_penalty)
    for row in range(1, n):
        lowers[row - 1] = derivative.first_at_least(level=-step_penalty)
        uppers[row - 1] = derivative.last_at_most(level=step_penalty)
        derivative = _add_row(
            clipped=derivative.clipped(
                limit=step_penalty, lower=lowers[row - 1], upper=uppers[row - 1]
            ),
            ramps=ramps,
            row=row,
            level_penalty=level_penalty,
        )
    flow = np.empty(n)
    flow[-1] = derivative.first_at_least(level=0.0)
    for row in range(n - 2, -1, -1):
        flow[row] = min(max(flow[row + 1], lowers[row]), uppers[row])
    return flow


def expand_held_flat(*, kept: np.ndarray, values: np.ndarray, length: int) -> np.ndarray:
    """Put values on a grid, holding each flat until the next kept row and zero before the first.

    Args:
        kept: The ascending grid positions that carry a value.
        values: The value at each kept position.
        length: The length of the grid.

    Returns:
        The grid of length `length`.
    """
    grid = np.zeros(length)
    if len(kept) == 0:
        return grid
    owner = np.searchsorted(kept, np.arange(length), side="right") - 1
    held = owner >= 0
    grid[held] = values[owner[held]]
    return grid


@dataclass(frozen=True)
class PathSpec:
    """A path in the graph: its name, nodes, and penalties.

    Attributes:
        name: The path's name, unique within the graph.
        node_a: The node that gains the flow.
        node_b: The node that loses the flow, or `None` for an outside path.
        step_penalty: Penalty per megawatt of step.
        level_penalty: Penalty per megawatt of level.
        gain_a: Optional gain per row at node a.
        gain_b: Optional gain per row at node b.
        non_negative: Whether the flow is constrained to be at least zero.
    """

    name: str
    node_a: str
    node_b: str | None = None
    step_penalty: float = 0.0
    level_penalty: float = 0.0
    gain_a: np.ndarray | None = None
    gain_b: np.ndarray | None = None
    non_negative: bool = False


def edge_name(*, node_a: str, node_b: str) -> str:
    """Return the name of the edge between two nodes, the same whichever order they come in."""
    first, second = sorted((node_a, node_b))
    return f"{first}-{second}"


def build_paths(
    *,
    edges: Sequence[tuple[str, str]],
    nodes: Sequence[str],
    step_penalty_by_edge: Mapping[str, float],
    level_penalty_by_edge: Mapping[str, float],
    outside_factor: float,
) -> tuple[PathSpec, ...]:
    """Build the paths of a graph: one per edge, and one outside path per node.

    An outside path carries transfers to partners that are not metered. Its step and level
    penalties are `outside_factor` times those of the node's first edge by name, so a conserved
    transfer between two metered nodes is cheaper on their edge than through two outside paths.

    Args:
        edges: Each interconnection as a pair of node names; either order, duplicates ignored.
        nodes: Every node's name.
        step_penalty_by_edge: The step penalty of each edge, by `edge_name`.
        level_penalty_by_edge: The level penalty of each edge, by `edge_name`.
        outside_factor: The multiple of the edge penalties charged on an outside path, at least 1.

    Returns:
        The edge paths in name order, then the outside paths in node order.

    Raises:
        ValueError: If an edge joins a node to itself, names an unknown node, or
            `outside_factor` is under 1.
    """
    if outside_factor < 1.0:
        msg = f"outside_factor must be at least 1, not {outside_factor}"
        raise ValueError(msg)
    known = set(nodes)
    unique = sorted({edge_name(node_a=a, node_b=b) for a, b in edges})
    paths: list[PathSpec] = []
    first_edge_of: dict[str, str] = {}
    for a, b in edges:
        if a == b or a not in known or b not in known:
            msg = f"edge ({a}, {b}) joins a node to itself or names an unknown node"
            raise ValueError(msg)
    for name in unique:
        node_a, node_b = name.split("-")
        first_edge_of.setdefault(node_a, name)
        first_edge_of.setdefault(node_b, name)
        paths.append(
            PathSpec(
                name=name,
                node_a=node_a,
                node_b=node_b,
                step_penalty=step_penalty_by_edge[name],
                level_penalty=level_penalty_by_edge[name],
            )
        )
    for node in nodes:
        reference = first_edge_of.get(node)
        step = outside_factor * (step_penalty_by_edge[reference] if reference else 0.0)
        level = outside_factor * (level_penalty_by_edge[reference] if reference else 0.0)
        paths.append(
            PathSpec(name=f"outside:{node}", node_a=node, step_penalty=step, level_penalty=level)
        )
    return tuple(paths)


@dataclass(frozen=True)
class FlowFit:
    """The result of a block-coordinate fit.

    Attributes:
        flows: Each path's flow on the grid, in megawatts, by path name.
        sweeps: How many sweeps over the paths ran.
        converged: Whether the objective and the largest path change both fell under the
            tolerance, and every path solve converged in the last sweep.
        objective: The joint objective at the flows.
    """

    flows: dict[str, np.ndarray]
    sweeps: int
    converged: bool
    objective: float


def _node_contribution(
    *, paths: Sequence[PathSpec], flows: Mapping[str, np.ndarray], node: str, skip: str, length: int
) -> np.ndarray:
    """Return what every path except `skip` adds to a node, in megawatts."""
    total = np.zeros(length)
    for path in paths:
        if path.name == skip:
            continue
        if path.node_a == node:
            gain = 1.0 if path.gain_a is None else path.gain_a
            total = total + gain * flows[path.name]
        if path.node_b == node:
            gain = 1.0 if path.gain_b is None else path.gain_b
            total = total - gain * flows[path.name]
    return total


def joint_objective(
    *,
    paths: Sequence[PathSpec],
    flows: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
    noise: Mapping[str, float],
    reliability: Mapping[str, np.ndarray] | None = None,
) -> float:
    """Return the objective of the whole graph at a set of flows.

    The data term is summed over nodes, and the penalties over paths. Step penalties apply over
    the rows where the path's nodes have a reading, as the path solver applies them.

    Args:
        paths: The graph's paths.
        flows: Each path's flow on the grid.
        residuals: Each node's residual in megawatts, `NaN` where there is no reading.
        noise: Each node's robust residual standard deviation in megawatts.
        reliability: Optional per-row weight in 0 to 1 for each node.

    Returns:
        The joint objective.
    """
    total = 0.0
    for node, residual in residuals.items():
        modelled = _node_contribution(
            paths=paths, flows=flows, node=node, skip="", length=len(residual)
        )
        weight = _row_weight(
            residual=residual,
            noise=noise[node],
            reliability=None if reliability is None else reliability.get(node),
        )
        total += float(
            np.sum(
                weight
                * huber_loss(
                    x=np.nan_to_num(residual) - modelled,
                    threshold=HUBER_THRESHOLD_IN_SIGMAS * noise[node],
                )
            )
        )
    for path in paths:
        nodes = [path.node_a] + ([] if path.node_b is None else [path.node_b])
        observed = np.any([np.isfinite(residuals[node]) for node in nodes], axis=0)
        flow = flows[path.name][observed]
        total += path.step_penalty * float(np.sum(np.abs(np.diff(flow))))
        total += path.level_penalty * float(np.sum(np.abs(flow)))
    return total


def fit_flows(
    *,
    paths: Sequence[PathSpec],
    residuals: Mapping[str, np.ndarray],
    noise: Mapping[str, float],
    reliability: Mapping[str, np.ndarray] | None = None,
    start: Mapping[str, np.ndarray] | None = None,
    max_sweeps: int = DEFAULT_MAX_SWEEPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> FlowFit:
    """Fit every path's flow by block-coordinate descent.

    Each sweep re-solves every path on the residual left at its nodes after the other paths'
    flows are removed. The sweeps stop when the joint objective's relative change and the largest
    path change in megawatts are both under `SWEEP_TOLERANCE`.

    Args:
        paths: The graph's paths.
        residuals: Each node's residual in megawatts on one common grid, `NaN` where there is no
            reading.
        noise: Each node's robust residual standard deviation in megawatts.
        reliability: Optional per-row weight in 0 to 1 for each node.
        start: Optional warm-start flow for each path.
        max_sweeps: The sweep limit.
        tolerance: The ADMM tolerance of each path solve in megawatts.

    Returns:
        The fitted flows and how the fit ended.

    Raises:
        ValueError: If a path names a node with no residual.
    """
    length = len(next(iter(residuals.values())))
    for path in paths:
        for node in (path.node_a, path.node_b):
            if node is not None and node not in residuals:
                msg = f"path {path.name} names node {node}, which has no residual"
                raise ValueError(msg)
    flows = {
        path.name: (
            np.zeros(length) if start is None or path.name not in start else start[path.name].copy()
        )
        for path in paths
    }
    reliability = reliability or {}
    previous_objective = np.inf
    all_converged = True
    sweeps = 0
    converged = False
    for sweep in range(1, max_sweeps + 1):
        sweeps = sweep
        largest_change = 0.0
        all_converged = True
        for path in paths:
            solution = solve_path(
                problem=_partial_problem(
                    path=path,
                    paths=paths,
                    flows=flows,
                    residuals=residuals,
                    noise=noise,
                    reliability=reliability,
                    length=length,
                ),
                start=flows[path.name],
                tolerance=tolerance,
            )
            all_converged &= solution.converged
            largest_change = max(
                largest_change, float(np.max(np.abs(solution.flow - flows[path.name])))
            )
            flows[path.name] = solution.flow
        objective = joint_objective(
            paths=paths, flows=flows, residuals=residuals, noise=noise, reliability=reliability
        )
        relative = abs(previous_objective - objective) / max(abs(objective), 1.0)
        previous_objective = objective
        if relative < SWEEP_TOLERANCE and largest_change < SWEEP_TOLERANCE:
            converged = all_converged
            break
    return FlowFit(flows=flows, sweeps=sweeps, converged=converged, objective=previous_objective)


def _partial_problem(
    *,
    path: PathSpec,
    paths: Sequence[PathSpec],
    flows: Mapping[str, np.ndarray],
    residuals: Mapping[str, np.ndarray],
    noise: Mapping[str, float],
    reliability: Mapping[str, np.ndarray],
    length: int,
) -> PathProblem:
    """Build one path's problem on the residual left by the other paths."""
    partial_a = residuals[path.node_a] - _node_contribution(
        paths=paths, flows=flows, node=path.node_a, skip=path.name, length=length
    )
    partial_b = None
    noise_b = 1.0
    if path.node_b is not None:
        partial_b = residuals[path.node_b] - _node_contribution(
            paths=paths, flows=flows, node=path.node_b, skip=path.name, length=length
        )
        noise_b = noise[path.node_b]
    return PathProblem(
        residual_a=partial_a,
        noise_a=noise[path.node_a],
        residual_b=partial_b,
        noise_b=noise_b,
        reliability_a=reliability.get(path.node_a),
        reliability_b=None if path.node_b is None else reliability.get(path.node_b),
        gain_a=path.gain_a,
        gain_b=path.gain_b,
        step_penalty=path.step_penalty,
        level_penalty=path.level_penalty,
        non_negative=path.non_negative,
    )


def robust_residual_scale(*, residual: np.ndarray) -> float:
    """Return the robust standard deviation of a residual stream, ignoring `NaN`.

    The scale is 1.4826 times the median absolute deviation from the median.

    Args:
        residual: The residuals in megawatts.

    Returns:
        The scale in megawatts.

    Raises:
        ValueError: If there is no finite residual.
    """
    finite = residual[np.isfinite(residual)]
    if len(finite) == 0:
        msg = "no finite residual to take a scale from"
        raise ValueError(msg)
    return MAD_TO_STANDARD_DEVIATION * float(np.median(np.abs(finite - np.median(finite))))


def block_mean_residuals(
    *, residuals: Mapping[str, np.ndarray], block_size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Average each node's residual over aligned blocks of rows.

    A block's mean is taken over the readings it holds, and its reliability is the share of the
    block's rows that hold a reading, so a block with one reading counts for little. The last block
    is padded with missing rows.

    Args:
        residuals: Each node's residual on one common grid, `NaN` where there is no reading.
        block_size: Rows per block; 6 half-hours make the 3-hour block.

    Returns:
        Each node's block means (`NaN` for an empty block), and each node's reliability in 0 to 1.
    """
    means: dict[str, np.ndarray] = {}
    reliability: dict[str, np.ndarray] = {}
    for node, residual in residuals.items():
        n_blocks = -(-len(residual) // block_size)
        padded = np.full(n_blocks * block_size, np.nan)
        padded[: len(residual)] = residual
        blocks = padded.reshape(n_blocks, block_size)
        counts = np.sum(np.isfinite(blocks), axis=1)
        with np.errstate(invalid="ignore"):
            means[node] = np.where(
                counts > 0, np.nansum(blocks, axis=1) / np.maximum(counts, 1), np.nan
            )
        reliability[node] = counts / block_size
    return means, reliability


def expand_blocks(*, block_values: np.ndarray, block_size: int, length: int) -> np.ndarray:
    """Repeat each block's value over its rows and trim to the grid's length.

    Args:
        block_values: One value per block.
        block_size: Rows per block.
        length: The grid's length.

    Returns:
        The values on the grid.
    """
    return np.repeat(block_values, block_size)[:length]


@dataclass(frozen=True)
class SurvivalCurve:
    """A Kaplan-Meier estimate of how long a flow lasts.

    Attributes:
        times: The distinct observed run lengths in rows, ascending.
        survival: The estimated share of runs still going just after each time.
    """

    times: np.ndarray
    survival: np.ndarray

    def at(self, *, age: np.ndarray | float) -> np.ndarray:
        """Return the survival at an age: one before the shortest run ends."""
        index = np.searchsorted(self.times, age, side="right") - 1
        return np.where(index < 0, 1.0, self.survival[np.maximum(index, 0)])

    def held_share(self, *, age: np.ndarray | float, ahead: np.ndarray | float) -> np.ndarray:
        """Return the chance a run of a given age is still going `ahead` rows later.

        Args:
            age: How long the current run has lasted, in rows.
            ahead: How many rows after now.

        Returns:
            `S(age + ahead) / S(age)`, or zero where no observed run lasted as long as `age`.
        """
        now = self.at(age=age)
        later = self.at(age=np.asarray(age) + np.asarray(ahead))
        return np.where(now > 0.0, later / np.where(now > 0.0, now, 1.0), 0.0)


def run_lengths(*, flow: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the length of every run where the flow's magnitude exceeds a threshold.

    Args:
        flow: A flow on the grid in megawatts.
        threshold: A row is in a run when `|flow|` is over this.

    Returns:
        The run lengths in rows, and whether each run is censored because it reaches the end of
        the flow.
    """
    active = np.abs(flow) > threshold
    padded = np.concatenate(([False], active, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = changes[0::2], changes[1::2]
    return stops - starts, stops == len(flow)


def fit_survival(*, lengths: np.ndarray, censored: np.ndarray) -> SurvivalCurve:
    """Fit the Kaplan-Meier product-limit estimate of the run-length distribution.

    Args:
        lengths: Observed run lengths in rows.
        censored: Whether each run was cut short by the end of the data.

    Returns:
        The survival curve; one everywhere if no run ended.

    Raises:
        ValueError: If `lengths` and `censored` differ in length.
    """
    if len(lengths) != len(censored):
        msg = "lengths and censored must have the same length"
        raise ValueError(msg)
    ended_times = np.unique(lengths[~censored])
    survival = np.empty(len(ended_times))
    running = 1.0
    for k, time in enumerate(ended_times):
        at_risk = int(np.sum(lengths >= time))
        ended = int(np.sum((lengths == time) & ~censored))
        running *= 1.0 - ended / at_risk
        survival[k] = running
    return SurvivalCurve(times=ended_times, survival=survival)


@dataclass(frozen=True)
class BandState:
    """The Kalman filter's state of one hour band just after a reading.

    Attributes:
        position: The grid position of the reading.
        level: The persistent level in megawatts.
        anomaly: The transient anomaly in megawatts.
        variance_level: The level's variance.
        covariance: The covariance of level and anomaly.
        variance_anomaly: The anomaly's variance.
    """

    position: int
    level: float
    anomaly: float
    variance_level: float
    covariance: float
    variance_anomaly: float


@dataclass(frozen=True)
class BandFilterRun:
    """The filter's state after each reading of one hour band.

    Attributes:
        positions: The readings' grid positions, ascending.
        level: The level after each reading.
        anomaly: The anomaly after each reading.
        variance_level: The level's variance after each reading.
        covariance: The covariance after each reading.
        variance_anomaly: The anomaly's variance after each reading.
    """

    positions: np.ndarray
    level: np.ndarray
    anomaly: np.ndarray
    variance_level: np.ndarray
    covariance: np.ndarray
    variance_anomaly: np.ndarray

    def state_at(self, *, position: int) -> BandState | None:
        """Return the state after the newest reading at or before a position, or `None`."""
        index = int(np.searchsorted(self.positions, position, side="right")) - 1
        if index < 0:
            return None
        return BandState(
            position=int(self.positions[index]),
            level=float(self.level[index]),
            anomaly=float(self.anomaly[index]),
            variance_level=float(self.variance_level[index]),
            covariance=float(self.covariance[index]),
            variance_anomaly=float(self.variance_anomaly[index]),
        )


def run_band_filter(
    *,
    positions: np.ndarray,
    values: np.ndarray,
    hyper: FilterHyperparameters,
    initial: BandState | None,
) -> BandFilterRun:
    """Run the two-state filter of `studies.dp_assimilation` over one band's readings.

    This repeats the update of `dp_assimilation.run_kalman`, which always starts cold, so that a
    run can start from the state an earlier run held. Started cold, it returns the same levels and
    anomalies as `run_kalman`.

    Args:
        positions: The grid positions of the readings, ascending.
        values: The residual at each position in megawatts.
        hyper: The site's hyperparameters, whose variances are in points squared; the caller
            passes residuals in the same unit.
        initial: The state to start from, or `None` for the cold start.

    Returns:
        The state after each reading.
    """
    q_step = hyper.q / HALF_HOURS_PER_DAY
    phi = hyper.phi
    stationary = hyper.sigma_z**2 / (1.0 - phi**2)
    noise = hyper.sigma_eps**2
    if initial is None:
        level, anomaly = 0.0, 0.0
        p_ll, p_lz, p_zz = INITIAL_LEVEL_VARIANCE, 0.0, stationary
        previous = int(positions[0]) if len(positions) else 0
    else:
        level, anomaly = initial.level, initial.anomaly
        p_ll, p_lz, p_zz = initial.variance_level, initial.covariance, initial.variance_anomaly
        previous = initial.position
    n = len(values)
    out = np.empty((5, n))
    for k in range(n):
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
        gain_l = (p_ll + p_lz) / innovation_variance
        gain_z = (p_lz + p_zz) / innovation_variance
        level += gain_l * innovation
        anomaly += gain_z * innovation
        p_ll, p_lz, p_zz = (
            max(p_ll - gain_l * gain_l * innovation_variance, VARIANCE_FLOOR),
            p_lz - gain_l * gain_z * innovation_variance,
            max(p_zz - gain_z * gain_z * innovation_variance, VARIANCE_FLOOR),
        )
        out[:, k] = (level, anomaly, p_ll, p_lz, p_zz)
    return BandFilterRun(
        positions=positions,
        level=out[0],
        anomaly=out[1],
        variance_level=out[2],
        covariance=out[3],
        variance_anomaly=out[4],
    )


FilterCarry = tuple[BandState | None, ...]
"""The state of every hour band's filter, `None` for a band that has seen no reading."""


def run_known_input_filter(
    *,
    residual: np.ndarray,
    flow: np.ndarray,
    band: np.ndarray,
    first: int,
    last: int,
    hyper: FilterHyperparameters,
    carry: FilterCarry | None,
) -> tuple[BandFilterRun, ...]:
    """Run the filter over a window of the residual with a known flow taken out.

    The flow is a known input: the filter sees `residual - flow`, so a transfer that the flow
    explains does not reach the level or the anomaly. The window starts from `carry`, the state the
    previous run held at the window's start.

    Args:
        residual: The residual stream on the whole grid in megawatts, `NaN` where there is none.
        flow: The node's flow over the window, one value per row from `first` to `last`.
        band: The hour band of each grid row.
        first: The first grid position of the window.
        last: The last grid position of the window, inclusive.
        hyper: The node's filter hyperparameters.
        carry: The state of each band at the window's start, or `None` for the cold start.

    Returns:
        One run per hour band, with absolute grid positions.

    Raises:
        ValueError: If the flow does not have one value per row of the window.
    """
    if len(flow) != last - first + 1:
        msg = f"flow needs {last - first + 1} values, one per row of the window"
        raise ValueError(msg)
    window = residual[first : last + 1] - flow
    runs = []
    for band_index in range(N_BANDS):
        local = np.flatnonzero((band[first : last + 1] == band_index) & np.isfinite(window))
        runs.append(
            run_band_filter(
                positions=local + first,
                values=window[local],
                hyper=hyper,
                initial=None if carry is None else carry[band_index],
            )
        )
    return tuple(runs)


def carry_at(
    *, runs: Sequence[BandFilterRun], position: int, fallback: FilterCarry | None
) -> FilterCarry:
    """Return each band's state at a position, from the runs or else from the fallback.

    Args:
        runs: One run per hour band.
        position: The grid position to read the state at.
        fallback: The state of a band that has no reading in its run at or before `position`.

    Returns:
        The state of every band.
    """
    states = []
    for band_index, run in enumerate(runs):
        state = run.state_at(position=position)
        if state is None and fallback is not None:
            state = fallback[band_index]
        states.append(state)
    return tuple(states)


def runs_to_trace(
    *, runs: Sequence[BandFilterRun], hyper: FilterHyperparameters
) -> CorrectionTrace:
    """Package filter runs as a `CorrectionTrace`, for `dp_assimilation.correction_at`.

    Args:
        runs: One run per hour band.
        hyper: The hyperparameters the runs used.

    Returns:
        The trace, with a log likelihood of zero (the runs do not track it).
    """
    return CorrectionTrace(
        kind="kalman",
        positions=tuple(run.positions for run in runs),
        level=tuple(run.level for run in runs),
        anomaly=tuple(run.anomaly for run in runs),
        phi=hyper.phi,
        log_likelihood=0.0,
    )


def node_flow(
    *, paths: Sequence[PathSpec], flows: Mapping[str, np.ndarray], node: str
) -> np.ndarray:
    """Return what every path adds to a node, in megawatts.

    Args:
        paths: The graph's paths.
        flows: Each path's flow.
        node: The node's name.

    Returns:
        The node's total flow on the grid.
    """
    length = len(next(iter(flows.values())))
    return _node_contribution(paths=paths, flows=flows, node=node, skip="", length=length)


def causal_window_fit(
    *,
    paths: Sequence[PathSpec],
    residuals: Mapping[str, np.ndarray],
    noise: Mapping[str, float],
    cutoff: int,
    window_length: int,
    start: Mapping[str, np.ndarray] | None = None,
) -> tuple[FlowFit, int]:
    """Fit the flows on the trailing window of residuals that a forecast may see.

    Only rows at or before `cutoff` enter, so the fit cannot read anything newer. The caller sets
    `cutoff` from the telemetry delay and passes residuals built out of sample.

    Args:
        paths: The paths to fit.
        residuals: Each node's residual stream on the whole grid in megawatts.
        noise: Each node's robust residual standard deviation, from rows before the cut.
        cutoff: The newest usable grid position.
        window_length: How many rows the window holds.
        start: Optional warm start for each path, on the window's rows.

    Returns:
        The fit, whose flows have `window_length` rows (fewer near the start of the stream), and
        the grid position of the window's first row.

    Raises:
        ValueError: If `cutoff` is negative.
    """
    if cutoff < 0:
        msg = "cutoff must be a grid position of at least 0"
        raise ValueError(msg)
    first = max(cutoff - window_length + 1, 0)
    window = {node: residual[first : cutoff + 1] for node, residual in residuals.items()}
    return fit_flows(paths=paths, residuals=window, noise=noise, start=start), first


FLOW_AS_MOVED_FRACTION: Final[float] = 0.2
"""A flow has been released when its magnitude is under this share of the moved megawatts."""


def moved_mw_ratio(*, fitted: np.ndarray, true: np.ndarray) -> float:
    """Return the fitted mean flow over the true mean flow, on the transfer's rows.

    Args:
        fitted: The fitted flow on the edge over the transfer's rows, with the true flow's sign
            convention.
        true: The injected flow over the same rows.

    Returns:
        The ratio of means; `NaN` if the true mean is zero.
    """
    true_mean = float(np.mean(true))
    return float(np.mean(fitted)) / true_mean if true_mean != 0.0 else float("nan")


def mw_weighted_recall(*, fitted: np.ndarray, true: np.ndarray) -> float:
    """Return the share of the true flow's megawatt-hours that the fitted flow recovers.

    Each row counts the smaller of the fitted flow with the true flow's sign, floored at zero, and
    the true flow's magnitude, so a flow on the wrong edge, of the wrong sign, or too large earns
    nothing extra.

    Args:
        fitted: The fitted flow on the true edge, with the true flow's sign convention.
        true: The injected flow.

    Returns:
        The recall in 0 to 1; `NaN` if the true flow is zero everywhere.
    """
    total = float(np.sum(np.abs(true)))
    if total == 0.0:
        return float("nan")
    matched = np.minimum(np.maximum(fitted * np.sign(true), 0.0), np.abs(true))
    return float(np.sum(matched)) / total


def release_delay(*, fitted: np.ndarray, moved_mw: float, end: int) -> int | None:
    """Return how many rows after a transfer's end the flow first falls under 20% of the moved MW.

    Args:
        fitted: The fitted flow on the transfer's edge, from before the transfer to well after it.
        moved_mw: The transfer's mean moved megawatts.
        end: The row of the first row after the transfer ends.

    Returns:
        The number of rows after `end` (0 if already under the level at `end`), or `None` if the
        flow never falls under it within the given rows.
    """
    below = np.abs(fitted[end:]) < FLOW_AS_MOVED_FRACTION * abs(moved_mw)
    if not np.any(below):
        return None
    return int(np.argmax(below))


LEAD_HOURS_EDGES: Final[tuple[float, ...]] = (0.0, 3.0, 6.0, 12.0, 24.0, 36.0, 48.0)
"""The lead-time bins of `LeadHoursSpread`, in hours."""


@dataclass(frozen=True)
class LeadHoursSpread:
    """Post-filter residual quantiles per lead-hours bin, hour band, and calendar quarter.

    Attributes:
        by_bin: One centred `ResidualSpread` per bin of `LEAD_HOURS_EDGES`.
    """

    by_bin: tuple[ResidualSpread, ...]

    @staticmethod
    def bin_of(*, lead_hours: np.ndarray) -> np.ndarray:
        """Return each lead's bin index, with leads beyond the last edge in the last bin."""
        inner = np.asarray(LEAD_HOURS_EDGES[1:-1])
        return np.searchsorted(inner, lead_hours, side="right")

    @staticmethod
    def fit(
        *,
        residuals: np.ndarray,
        band: np.ndarray,
        quarter: np.ndarray,
        lead_hours: np.ndarray,
    ) -> LeadHoursSpread:
        """Fit the spread from the replica's residuals (actual minus corrected mean).

        Args:
            residuals: The replica residuals in points.
            band: The hour band of each residual.
            quarter: The calendar quarter of each residual.
            lead_hours: The lead of each residual in hours.

        Returns:
            The centred spread. A bin with no residuals uses all residuals.
        """
        bins = LeadHoursSpread.bin_of(lead_hours=lead_hours)
        fitted = []
        for k in range(len(LEAD_HOURS_EDGES) - 1):
            chosen = bins == k
            use = chosen if np.any(chosen) else np.ones(len(residuals), dtype=bool)
            fitted.append(
                ResidualSpread.fit(
                    residuals=residuals[use], band=band[use], quarter=quarter[use], centre=True
                )
            )
        return LeadHoursSpread(by_bin=tuple(fitted))

    def quantiles(
        self, *, mean: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
    ) -> np.ndarray:
        """Return the 11 quantiles around each corrected mean.

        Args:
            mean: The corrected mean forecast in points.
            band: The hour band of each row.
            quarter: The calendar quarter of each row.
            lead_hours: The lead of each row in hours.

        Returns:
            An array of shape (len(mean), 11), non-decreasing along the last axis.
        """
        bins = self.bin_of(lead_hours=lead_hours)
        out = np.empty((len(mean), 11))
        for k, spread in enumerate(self.by_bin):
            chosen = bins == k
            if np.any(chosen):
                out[chosen] = spread.quantiles(
                    mean=mean[chosen], band=band[chosen], quarter=quarter[chosen]
                )
        return out


def edge_pair(*, pair: tuple[str, str]) -> tuple[str, str]:
    """Return a pair of node names with the smaller name first."""
    first, second = sorted(pair)
    return first, second


def undirected_edges(*, pairs: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    """Return the distinct undirected edges of a list of pairs, each listed once in name order.

    Args:
        pairs: Interconnected pairs, in either order, possibly listed twice.

    Returns:
        The distinct edges, sorted, with the smaller name first. A pair of a node with itself is
        dropped.
    """
    return tuple(sorted({edge_pair(pair=pair) for pair in pairs if pair[0] != pair[1]}))


def decision_edges(
    *, pairs: Sequence[tuple[str, str]], metered: set[str], excluded: set[str]
) -> tuple[tuple[str, str], ...]:
    """Return the edges that have a meter at both ends and touch no excluded node.

    Args:
        pairs: Interconnected pairs, in either order.
        metered: The nodes that have a usable meter.
        excluded: The nodes whose edges are left out of the decision graph.

    Returns:
        The distinct edges, sorted.
    """
    return tuple(
        edge
        for edge in undirected_edges(pairs=pairs)
        if set(edge) <= metered and not set(edge) & excluded
    )
