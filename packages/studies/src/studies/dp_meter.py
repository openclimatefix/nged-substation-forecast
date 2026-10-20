"""The convex model of the differentiable-physics (DP) series, fitted through an MVA meter.

**An MVA meter reads apparent power `|S|`, never signed real power.** The model predicts signed real
power `P` (demand minus solar), so reading `|S|` as `P` puts a mirror image of every midday dip
that crosses zero into the target. This module fits the convex model of `studies.dp_gam` to `|S|`
through the observation operator of `studies.dp_operator`:

    P = D - PV,    Q = tan(phi) * D + q0,    |S| = sqrt(P^2 + Q^2 + epsilon)

where `D` is the demand part of the design (every column except the solar terms and the node).

**The loss is not convex, so the fit is Gauss-Newton with several starts.** A mirror basin exists
wherever `P` can change sign, and `tan(phi) * D(theta)` is bilinear in the parameters. Each step
linearises `sqrt` at the current `P` and `Q` and solves a bounded quadratic program with
`studies.dp_gam.solve_bounded_quadratic`, which keeps the bounds exact. A step is accepted only if
it lowers the penalised loss, and otherwise a damping term pulls it towards the current
coefficients (Levenberg-Marquardt). Six starts are tried, each is flagged if it stopped without
converging, and the lowest loss on a common scale is kept:

- the naive fit, which reads `|S|` as `P`, with the capacity increments multiplied by 0.5, 1, and 2;
- the naive demand with an absolute capacity of 50 and of 120 points, for strong reverse flow, where
  the naive capacity is zero and its multiples are too;
- the demand-only fit on the dark rows, with no solar (the naive fit with its capacity set to zero
  when the substation has no dark row).

**`qmode` sets the reactive power, and the data decide whether it can be learned.** `zero` fixes
`tan(phi) = q0 = 0`, NGED's convention. `learned` fits `tan(phi)` and `q0` together with the
coefficients, but only where the data can identify them. On dark rows `|S| = D * sec(phi)`, so
`sec(phi)` folds into the demand scale and `tan(phi) * D + q0` is nearly collinear in its two
unknowns. Without reverse flow the pair absorbs any structural misfit instead of measuring a
reactive load: on synthetic data it moved the power factor to 0.83 and the capacity 23% high. So
`learned` first fits with `qmode="zero"` and counts the training rows where the signed power is
negative, or, if it is the larger, under the pair held at the prior (the zero fit can sit in the
mirror basin and count none). If that share is under `MIN_REVERSE_FLOW_SHARE`, the reactive pair is
**not learned**: it
is held at the prior (power factor 0.97, `q0` = 0), the fit records `reactive="fixed"`, and the
power factor it reports is an assumption, not an estimate. Where the pair is learned, `tan(phi)` is
bounded between zero (a lagging load; this removes the `(tan(phi), q0) -> (-tan(phi), -q0)` mirror
that leaves `|S|` unchanged) and the value at a power factor of 0.90, and `q0` lies within
`Q0_MAX_POINTS`. Both carry a Gaussian prior, `tan(phi)` centred on its value at a power factor of
0.97 with 95% of the mass on power factors 0.95 to 0.99, and `q0` centred on zero with scale 2
points. Each prior is weighted by the residual variance times `PRIOR_STRENGTH_ROWS`, the rows over
which the residual is correlated, so that the data's thousands of correlated rows do not outvote
it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Literal

import numpy as np
import torch

from studies.dp_features import FeatureCache, SiteFeatures
from studies.dp_gam import (
    LAST_TRAINED_MIN_SHAPE,
    GamFit,
    GamLayout,
    GamSpec,
    ReplicaResiduals,
    TrainingWindow,
    derated_unit_shape,
    design_matrix,
    orthogonal_penalty_matrix,
    penalty_matrix,
    predict_gam,
    prepare_fit,
    replica_cut,
    replica_rows,
    row_chunks,
    solve_bounded_quadratic,
)
from studies.dp_operator import (
    DEFAULT_EPSILON,
    ApparentPowerParameters,
    apparent_power,
    power_factor_from_tan_phi,
    tan_phi_from_power_factor,
)

QModeType = Literal["zero", "learned"]
"""Whether the reactive power is zero (NGED's convention) or learned from the readings."""

ReactiveModeType = Literal["zero", "fixed", "learned"]
"""How a fit treated the reactive pair: zero, held at the prior, or learned."""

TAN_PHI_PRIOR_MEAN: Final[float] = tan_phi_from_power_factor(power_factor=0.97)
"""The centre of the prior on `tan(phi)`: its value at a power factor of 0.97."""
TAN_PHI_PRIOR_SD: Final[float] = (
    tan_phi_from_power_factor(power_factor=0.95) - tan_phi_from_power_factor(power_factor=0.99)
) / (2.0 * 1.96)
"""95% of the prior's mass lies on power factors 0.95 to 0.99 (a half-width of 1.96 deviations)."""
TAN_PHI_MAX: Final[float] = tan_phi_from_power_factor(power_factor=0.90)
"""The most a learned `tan(phi)` can be: its value at a power factor of 0.90."""
Q0_PRIOR_SD_POINTS: Final[float] = 2.0
"""The scale of the prior on the reactive power at zero demand, in points of effective capacity."""
Q0_MAX_POINTS: Final[float] = 5.0
"""A learned `q0` lies within this many points of zero."""
PRIOR_STRENGTH_ROWS: Final[float] = 336.0
"""Each reactive prior counts as this many rows.

It is a week of half-hours, over which a residual from a missing structural term stays correlated.
"""
MIN_REVERSE_FLOW_SHARE: Final[float] = 0.005
"""The reactive pair is learned only if at least this share of the training rows has negative
signed power under the zero-reactive fit."""
CAPACITY_START_MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
"""The naive fit's capacity increments are multiplied by these to make three starts."""
ABSOLUTE_CAPACITY_STARTS_POINTS: Final[tuple[float, ...]] = (50.0, 120.0)
"""Starts with this capacity, in points, in the first season-year and nothing added later.

The naive fit reads `|S|` as `P`, so under strong reverse flow it can put the capacity at zero, and
every multiple of zero is zero. Absolute starts reach the basin with the true capacity.
"""
MAX_ITERATIONS: Final[int] = 100
"""The most Gauss-Newton iterations of one start."""
MAX_DAMPING_TRIES: Final[int] = 8
"""The most times one iteration raises its damping before it gives up and stops."""
RELATIVE_TOLERANCE: Final[float] = 1e-7
"""A start has converged when an iteration lowers its loss by less than this share."""
INITIAL_DAMPING_SHARE: Final[float] = 1e-3
"""The first damping, as a share of the mean diagonal of the coefficient penalty plus one."""
DAMPING_GROWTH: Final[float] = 10.0
"""Each failed step multiplies the damping by this."""

NON_DEMAND_COLUMNS: Final[tuple[str, ...]] = (
    "capacity",
    "climatology",
    "node_level",
    "node_up",
    "node_down",
)
"""The design's blocks that are solar or generation, so not part of the demand `D`."""


@dataclass(frozen=True)
class StartSummary:
    """How one start of the multi-start fit ended.

    Attributes:
        name: The start's name.
        loss: The penalised loss it ended at, on the scale common to every start.
        final_capacity: The capacity of the last season-year at its end, in points.
        tan_phi: The ratio of reactive power to demand it ended at.
        iterations: The Gauss-Newton iterations it took.
        converged: False if it stopped at `MAX_ITERATIONS` or could not lower its loss.
    """

    name: str
    loss: float
    final_capacity: float
    tan_phi: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class MeterFit:
    """The best start of a fit through the meter.

    Attributes:
        fit: The fitted coefficients, a one-member `GamFit` whose `predict_gam` gives signed real
            power `P`.
        qmode: What the caller asked for: zero or learned reactive power.
        reactive: What the fit did: `zero`, `fixed` (asked to learn, but the data hold too little
            reverse flow to identify the pair, so it is held at the prior), or `learned`.
        tan_phi: The ratio of reactive power to demand.
        q0: The reactive power at zero demand, in points.
        loss: The best start's penalised loss.
        reverse_flow_share: The weighted share of training rows with negative signed power under
            the zero-reactive fit; `reactive="learned"` needs `MIN_REVERSE_FLOW_SHARE`.
        starts: Every start's summary, so the spread across starts can be reported.
        demand_columns: True for the design columns that make up the demand `D`.
    """

    fit: GamFit
    qmode: QModeType
    reactive: ReactiveModeType
    tan_phi: float
    q0: float
    loss: float
    reverse_flow_share: float
    starts: tuple[StartSummary, ...]
    demand_columns: np.ndarray

    @property
    def power_factor(self) -> float:
        """The power factor: an estimate if `reactive` is `learned`, else an assumption."""
        return float(power_factor_from_tan_phi(tan_phi=torch.as_tensor(self.tan_phi)))

    @property
    def capacity_spread(self) -> float:
        """The range of the starts' final capacities, in points."""
        capacities = [start.final_capacity for start in self.starts]
        return float(max(capacities) - min(capacities))

    @property
    def converged(self) -> bool:
        """True if every start converged."""
        return all(start.converged for start in self.starts)


@dataclass(frozen=True)
class MeterPrediction:
    """A meter fit's predictions at some rows, in points of effective capacity.

    Attributes:
        real_power: Signed real power `P = D - PV`.
        demand: The demand `D`.
        reactive_power: `Q = tan(phi) * D + q0`.
        apparent_power: The reading `|S|`.
    """

    real_power: np.ndarray
    demand: np.ndarray
    reactive_power: np.ndarray
    apparent_power: np.ndarray


def demand_column_mask(*, layout: GamLayout) -> np.ndarray:
    """Return True for the design columns that make up the demand.

    Args:
        layout: The columns.

    Returns:
        A boolean array of length `layout.n_columns`, False for the solar and node blocks.
    """
    mask = np.ones(layout.n_columns, dtype=bool)
    for name in NON_DEMAND_COLUMNS:
        if name in layout.slices:
            mask[layout.slices[name]] = False
    return mask


def predict_meter(*, cache: FeatureCache, meter: MeterFit, index: np.ndarray) -> MeterPrediction:
    """Predict real, demand, reactive, and apparent power at the grid rows `index`.

    Args:
        cache: The cache.
        meter: The fit.
        index: Grid rows.

    Returns:
        The predictions.
    """
    real = predict_gam(cache=cache, fit=meter.fit, index=index)
    demand = predict_gam(cache=cache, fit=meter.fit, index=index, columns=meter.demand_columns)
    reactive = meter.tan_phi * demand + meter.q0
    apparent = apparent_power(
        real=torch.as_tensor(real),
        demand=torch.as_tensor(demand),
        parameters=ApparentPowerParameters(
            tan_phi=torch.as_tensor(meter.tan_phi, dtype=torch.float64),
            q0=torch.as_tensor(meter.q0, dtype=torch.float64),
        ),
    ).numpy()
    return MeterPrediction(
        real_power=real, demand=demand, reactive_power=reactive, apparent_power=apparent
    )


def fit_gam_through_meter(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
    qmode: QModeType,
    weights: np.ndarray | None = None,
) -> MeterFit:
    """Fit the convex model to a substation's `|S|` readings through the meter operator.

    Args:
        cache: The cache; `features.y` holds the `|S|` reading in points.
        site: The substation's label.
        spec: The penalties; the capacity must be fitted (an MVA substation has no vintage), and
            there is no clip or node.
        cut: The cut; training rows end no later.
        window: The window rule.
        season_years: The season-years with a capacity increment.
        qmode: `zero` for NGED's convention, `learned` to fit the power factor and `q0` where the
            data hold enough reverse flow (see the module docstring).
        weights: Row weights over the training rows (one per row `prepare_fit` returns, in order),
            such as the counts of a month-block bootstrap resample; ones if not given.

    Returns:
        The start with the lowest loss, with every start's summary.

    Raises:
        ValueError: If the spec pins the capacity or has a clip or node, no row carries solar
            signal, the weights have the wrong length or a negative entry, or a start's quadratic
            program does not converge.
    """
    if spec.capacity != "fitted" or spec.clip is not None or spec.node is not None:
        msg = "the meter fit needs a fitted capacity and no clip or node"
        raise ValueError(msg)
    features = cache.sites[site]
    layout, index = prepare_fit(
        cache=cache, site=site, spec=spec, cut=cut, window=window, season_years=season_years
    )
    signal = derated_unit_shape(features=features, index=index) > LAST_TRAINED_MIN_SHAPE
    if not signal.any():
        msg = f"{site} has no training row that carries solar signal before {cut}"
        raise ValueError(msg)
    if weights is None:
        weights = np.ones(len(index))
    if weights.shape != (len(index),) or (weights < 0.0).any():
        msg = f"the weights need one non-negative entry per training row, {len(index)} in all"
        raise ValueError(msg)
    problem = _Problem.build(
        cache=cache, features=features, layout=layout, index=index, weights=weights
    )
    starts = problem.starts(features=features, index=index, layout=layout)
    reactive: ReactiveModeType = "zero"
    runs = [problem.minimise(theta=theta, mode="zero") for _, theta in starts]
    reverse_flow_share = problem.reverse_flow_share(theta=min(runs, key=lambda r: r.squares).theta)
    if qmode == "learned":
        if reverse_flow_share >= MIN_REVERSE_FLOW_SHARE:
            reactive = "learned"
            runs = [problem.minimise(theta=theta, mode="learned") for _, theta in starts]
        else:
            reactive = "fixed"
            runs = [problem.minimise(theta=theta, mode="fixed") for _, theta in starts]
            reverse_flow_share = max(
                reverse_flow_share,
                problem.reverse_flow_share(theta=min(runs, key=lambda r: r.squares).theta),
            )
            if reverse_flow_share >= MIN_REVERSE_FLOW_SHARE:
                reactive = "learned"
                runs = [problem.minimise(theta=theta, mode="learned") for _, theta in starts]
    common_weight = problem.prior_weight(state=min(runs, key=lambda r: r.squares).state)
    losses = [
        problem.loss(state=run.state, weight=common_weight, learned=reactive == "learned")
        for run in runs
    ]
    summaries = tuple(
        StartSummary(
            name=name,
            loss=loss,
            final_capacity=float(np.cumsum(run.theta[layout.slices["capacity"]])[-1]),
            tan_phi=run.tan_phi,
            iterations=run.iterations,
            converged=run.converged,
        )
        for (name, _), run, loss in zip(starts, runs, losses, strict=True)
    )
    best = runs[int(np.argmin(losses))]
    fit = GamFit(
        layout=layout,
        theta=best.theta[None, :],
        cut=cut,
        n_rows=len(index),
        site=site,
        last_season_year=int(cache.calendar.season_year[index[signal]].max()),
        violation=np.zeros(1),
        pinned=None,
    )
    return MeterFit(
        fit=fit,
        qmode=qmode,
        reactive=reactive,
        tan_phi=best.tan_phi,
        q0=best.q0,
        loss=float(min(losses)),
        reverse_flow_share=reverse_flow_share,
        starts=summaries,
        demand_columns=problem.demand_mask,
    )


def meter_replica_residuals(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    cut: datetime,
    window: TrainingWindow,
    season_years: Sequence[int],
    qmode: QModeType,
    drop_switching: bool = True,
) -> ReplicaResiduals:
    """Return the residuals of a replica of a meter fit: `|S|` minus the predicted `|S|`.

    The replica is the same fit 12 months before the cut, frozen and used to predict the 12 months
    that follow, exactly as `studies.dp_gam.replica_residuals` does for a signed series.

    Args:
        cache: The cache.
        site: The substation's label.
        spec: The penalties.
        cut: The cut of the final fit.
        window: The window rule.
        season_years: The season-years with a capacity increment.
        qmode: `zero` or `learned`.
        drop_switching: True to leave out the rows inside the switching masks.

    Returns:
        The residuals, with the bands, quarters, and rows of the replica year.
    """
    fit_cut = replica_cut(cut=cut)
    meter = fit_gam_through_meter(
        cache=cache,
        site=site,
        spec=spec,
        cut=fit_cut,
        window=window,
        season_years=season_years,
        qmode=qmode,
    )
    index = replica_rows(
        cache=cache, site=site, fit_cut=fit_cut, cut=cut, drop_switching=drop_switching
    )
    features = cache.sites[site]
    predicted = predict_meter(cache=cache, meter=meter, index=index)
    return ReplicaResiduals(
        residuals=features.y[index] - predicted.apparent_power,
        band=cache.calendar.band[index],
        quarter=cache.calendar.quarter[index],
        index=index,
        fit_cut=fit_cut,
        switching=features.switching_mask[index],
    )


@dataclass(frozen=True)
class _Run:
    """The end state of one start.

    Attributes:
        state: The coefficients, then `tan(phi)` and `q0`.
        squares: The weighted sum of squared residuals at the end.
        iterations: The iterations taken.
        converged: False if the start stopped without converging.
    """

    state: np.ndarray
    squares: float
    iterations: int
    converged: bool

    @property
    def theta(self) -> np.ndarray:
        """The coefficients."""
        return self.state[:-2]

    @property
    def tan_phi(self) -> float:
        """The ratio of reactive power to demand."""
        return float(self.state[-2])

    @property
    def q0(self) -> float:
        """The reactive power at zero demand, in points."""
        return float(self.state[-1])


@dataclass(frozen=True)
class _Problem:
    """The fixed arrays of one meter fit: the design, the readings, and the penalty."""

    design: np.ndarray
    demand_design: np.ndarray
    demand_mask: np.ndarray
    readings: np.ndarray
    weights: np.ndarray
    penalty: np.ndarray
    cost: np.ndarray
    constrained: torch.Tensor

    @staticmethod
    def build(
        *,
        cache: FeatureCache,
        features: SiteFeatures,
        layout: GamLayout,
        index: np.ndarray,
        weights: np.ndarray,
    ) -> _Problem:
        """Build the design once, in memory, and the penalty."""
        design = np.empty((len(index), layout.n_columns))
        for chunk in row_chunks(index=index):
            design[chunk] = design_matrix(
                cache=cache, features=features, layout=layout, index=index[chunk]
            )
        mask = demand_column_mask(layout=layout)
        penalty, cost = penalty_matrix(layout=layout, n_rows=len(index))
        if layout.spec.orthogonal_penalty > 0.0:
            raw = orthogonal_penalty_matrix(
                cache=cache, features=features, layout=layout, index=index, weights=weights[None]
            )[0]
            gram_diagonal = float(np.einsum("ij,i,ij->j", design, weights, design).mean())
            penalty = (
                penalty + layout.spec.orthogonal_penalty * gram_diagonal / np.diag(raw).mean() * raw
            )
        return _Problem(
            design=design,
            demand_design=design * mask[None, :],
            demand_mask=mask,
            readings=features.y[index],
            weights=weights,
            penalty=penalty,
            cost=cost,
            constrained=torch.as_tensor(layout.constrained),
        )

    def starts(
        self, *, features: SiteFeatures, index: np.ndarray, layout: GamLayout
    ) -> list[tuple[str, np.ndarray]]:
        """Return the named starting coefficients."""
        naive = self._solve_plain(weights=self.weights)
        capacity = layout.slices["capacity"]
        result: list[tuple[str, np.ndarray]] = []
        for multiplier in CAPACITY_START_MULTIPLIERS:
            theta = naive.copy()
            theta[capacity] *= multiplier
            result.append((f"naive capacity x{multiplier:g}", theta))
        for points in ABSOLUTE_CAPACITY_STARTS_POINTS:
            theta = naive.copy()
            theta[capacity] = 0.0
            theta[capacity.start] = points
            result.append((f"capacity {points:g} points", theta))
        dark = features.dark[index]
        if dark.any():
            theta = self._solve_plain(weights=self.weights * dark)
            name = "dark-row demand, no solar"
        else:
            theta = naive.copy()
            name = "naive demand, no solar"
        for block in ("capacity", "climatology"):
            if block in layout.slices:
                theta[layout.slices[block]] = 0.0
        result.append((name, theta))
        return result

    def _solve_plain(self, *, weights: np.ndarray) -> np.ndarray:
        """Return the bounded least-squares fit of the readings read as signed power."""
        weighted = self.design * weights[:, None]
        gram = weighted.T @ self.design + self.penalty
        rhs = weighted.T @ self.readings - 0.5 * self.cost
        return _solve(matrix=gram, rhs=rhs, constrained=self.constrained)

    def reverse_flow_share(self, *, theta: np.ndarray) -> float:
        """Return the weighted share of training rows whose signed power is negative."""
        return float((self.weights * ((self.design @ theta) < 0.0)).sum() / self.weights.sum())

    def squares(self, *, state: np.ndarray) -> float:
        """Return the weighted sum of squared residuals at `state`."""
        return float(
            (self.weights * np.square(self.readings - self._apparent(state=state)[3])).sum()
        )

    def minimise(self, *, theta: np.ndarray, mode: ReactiveModeType) -> _Run:
        """Run Levenberg-Marquardt from `theta`.

        The unknowns are the coefficients and, for `learned`, `tan(phi)` and `q0`, stepped
        together: the linearised problem has two extra columns, and the priors on `tan(phi)` and
        `q0` are two extra pseudo-observations. A step is projected onto the bounds of the reactive
        pair and accepted only if it lowers the penalised loss (with the priors weighted by the
        residual variance of the iteration's start), and otherwise the damping grows. `zero` keeps
        the pair at zero and `fixed` at the prior.

        Args:
            theta: The starting coefficients.
            mode: How to treat the reactive pair.

        Returns:
            The end state.
        """
        learned = mode == "learned"
        tan0 = 0.0 if mode == "zero" else TAN_PHI_PRIOR_MEAN
        state = np.concatenate([theta, [tan0, 0.0]])
        damping = 0.0
        converged = False
        iterations = 0
        for iterations in range(1, MAX_ITERATIONS + 1):  # noqa: B007
            weight = self.prior_weight(state=state)
            loss = self.loss(state=state, weight=weight, learned=learned)
            candidate = self._step(state=state, weight=weight, learned=learned, damping=damping)
            new_loss = self.loss(state=candidate, weight=weight, learned=learned)
            tries = 0
            while new_loss > loss and tries < MAX_DAMPING_TRIES:
                damping = (
                    INITIAL_DAMPING_SHARE * float(np.diag(self.penalty).mean() + 1.0)
                    if damping == 0.0
                    else damping * DAMPING_GROWTH
                )
                candidate = self._step(state=state, weight=weight, learned=learned, damping=damping)
                new_loss = self.loss(state=candidate, weight=weight, learned=learned)
                tries += 1
            if new_loss > loss:
                converged = new_loss - loss <= RELATIVE_TOLERANCE * abs(loss)
                break
            improvement = loss - new_loss
            state = candidate
            damping = damping / DAMPING_GROWTH if damping > 0.0 else 0.0
            if improvement <= RELATIVE_TOLERANCE * abs(loss):
                converged = True
                break
        return _Run(
            state=state,
            squares=self.squares(state=state),
            iterations=iterations,
            converged=converged,
        )

    def prior_weight(self, *, state: np.ndarray) -> float:
        """Return the residual variance at `state` times `PRIOR_STRENGTH_ROWS`."""
        return PRIOR_STRENGTH_ROWS * self.squares(state=state) / float(self.weights.sum())

    def loss(self, *, state: np.ndarray, weight: float, learned: bool) -> float:
        """Return the penalised loss: squared residuals, coefficient penalty, and the priors."""
        theta = state[:-2]
        loss = self.squares(state=state) + float(theta @ self.penalty @ theta + self.cost @ theta)
        if learned:
            loss += weight * (
                ((state[-2] - TAN_PHI_PRIOR_MEAN) / TAN_PHI_PRIOR_SD) ** 2
                + (state[-1] / Q0_PRIOR_SD_POINTS) ** 2
            )
        return loss

    def _apparent(
        self, *, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(P, D, Q, |S|)` at the state `(theta, tan_phi, q0)`.

        This is `studies.dp_operator.apparent_power` in numpy: the inner loop runs thousands of
        times, so it avoids torch. A change to the operator's formula or epsilon must be made here
        too.
        """
        theta, tan_phi, q0 = state[:-2], state[-2], state[-1]
        real = self.design @ theta
        demand = self.demand_design @ theta
        reactive = tan_phi * demand + q0
        return real, demand, reactive, np.sqrt(real * real + reactive * reactive + DEFAULT_EPSILON)

    def jacobian(self, *, state: np.ndarray, learned: bool) -> np.ndarray:
        """Return the derivative of `|S|` with respect to the coefficients (and `tan(phi)`, `q0`).

        Args:
            state: The coefficients, then `tan(phi)` and `q0`.
            learned: True to include the two reactive columns.

        Returns:
            An array of shape (rows, unknowns).
        """
        tan_phi = state[-2]
        real, demand, reactive, apparent = self._apparent(state=state)
        n_coefficients = len(state) - 2
        jacobian = np.empty((len(self.readings), n_coefficients + (2 if learned else 0)))
        jacobian[:, :n_coefficients] = (real / apparent)[:, None] * self.design
        if tan_phi != 0.0:
            jacobian[:, :n_coefficients] += (reactive * tan_phi / apparent)[:, None] * (
                self.demand_design
            )
        if learned:
            jacobian[:, n_coefficients] = reactive * demand / apparent
            jacobian[:, n_coefficients + 1] = reactive / apparent
        return jacobian

    def _step(
        self, *, state: np.ndarray, weight: float, learned: bool, damping: float
    ) -> np.ndarray:
        """Return the state that minimises the linearised, damped, bounded problem."""
        theta = state[:-2]
        apparent = self._apparent(state=state)[3]
        n_coefficients = len(theta)
        n_unknowns = n_coefficients + (2 if learned else 0)
        jacobian = self.jacobian(state=state, learned=learned)
        root_weights = np.sqrt(self.weights)
        jacobian *= root_weights[:, None]
        target = root_weights * (self.readings - apparent) + jacobian @ state[:n_unknowns]
        gram = np.zeros((n_unknowns, n_unknowns))
        rhs = np.zeros(n_unknowns)
        gram[:n_coefficients, :n_coefficients] = self.penalty
        rhs[:n_coefficients] = -0.5 * self.cost
        gram += jacobian.T @ jacobian
        rhs += jacobian.T @ target
        constrained = self.constrained
        if learned:
            tan_slope = float(np.sqrt(weight)) / TAN_PHI_PRIOR_SD
            gram[n_coefficients, n_coefficients] += tan_slope**2
            rhs[n_coefficients] += tan_slope**2 * TAN_PHI_PRIOR_MEAN
            q0_slope = float(np.sqrt(weight)) / Q0_PRIOR_SD_POINTS
            gram[n_coefficients + 1, n_coefficients + 1] += q0_slope**2
            constrained = torch.cat([constrained, torch.tensor([True, False])])
        if damping > 0.0:
            gram += damping * np.eye(n_unknowns)
            rhs += damping * state[:n_unknowns]
        if not learned:
            solved = _solve(matrix=gram, rhs=rhs, constrained=constrained)
            return np.concatenate([solved, state[-2:]])
        return _solve_within_bounds(
            gram=gram, rhs=rhs, constrained=constrained, n_coefficients=n_coefficients
        )


def _solve_within_bounds(
    *, gram: np.ndarray, rhs: np.ndarray, constrained: torch.Tensor, n_coefficients: int
) -> np.ndarray:
    """Solve the step with `tan(phi)` and `q0` inside their bounds.

    The step is solved with only the lower bound on `tan(phi)` that the program expresses. Any
    reactive unknown beyond its upper bound (or `q0` beyond either) is then fixed at the bound and
    the rest re-solved, so the other unknowns respond to the bound rather than being left where an
    unbounded step put them.

    Args:
        gram: The step's normal matrix over the coefficients, `tan(phi)`, and `q0`.
        rhs: The step's right-hand side.
        constrained: True for the unknowns that must be non-negative.
        n_coefficients: The number of coefficients, which come first.

    Returns:
        The solution.
    """
    tan_at, q0_at = n_coefficients, n_coefficients + 1
    bounds = {tan_at: (0.0, TAN_PHI_MAX), q0_at: (-Q0_MAX_POINTS, Q0_MAX_POINTS)}
    fixed: dict[int, float] = {}
    while True:
        free = np.array([i for i in range(len(rhs)) if i not in fixed])
        solved = np.zeros(len(rhs))
        for index, value in fixed.items():
            solved[index] = value
        reduced_rhs = rhs[free] - gram[np.ix_(free, list(fixed))] @ np.array(list(fixed.values()))
        solved[free] = _solve(
            matrix=gram[np.ix_(free, free)], rhs=reduced_rhs, constrained=constrained[free]
        )
        broken = {
            index: float(np.clip(solved[index], low, high))
            for index, (low, high) in bounds.items()
            if index not in fixed and not low <= solved[index] <= high
        }
        if not broken:
            return solved
        fixed.update(broken)


def _solve(*, matrix: np.ndarray, rhs: np.ndarray, constrained: torch.Tensor) -> np.ndarray:
    """Solve one bounded quadratic program with the series' exact solver."""
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(matrix[None]),
        rhs=torch.as_tensor(rhs[None]),
        constrained=constrained,
    )
    return theta[0].numpy()
