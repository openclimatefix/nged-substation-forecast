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
coefficients (Levenberg-Marquardt). Four starts are tried and the lowest loss is kept:

- the naive fit, which reads `|S|` as `P`, with the capacity increments multiplied by 0.5, 1, and 2;
- the demand-only fit on the dark rows, with no solar (the naive fit with its capacity set to zero
  when the substation has no dark row).

**`qmode` sets the reactive power.** `zero` fixes `tan(phi) = q0 = 0`, NGED's convention. `learned`
alternates the coefficient steps with a step in `(tan(phi), q0)`. `tan(phi)` is bounded below by
zero (a lagging load), which removes the `(tan(phi), q0) -> (-tan(phi), -q0)` mirror that leaves
`|S|` unchanged, and `q0` is free in sign because cable capacitance makes it negative. The power
factor has a Gaussian prior centred on 0.97 with 95% of its mass on 0.95 to 0.99, and `q0` has a
Gaussian prior with scale 2 points. The priors are weighted by the residual variance, so they act
as pseudo-observations beside the data.
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
    solve_bounded_quadratic,
)
from studies.dp_operator import (
    DEFAULT_EPSILON,
    ApparentPowerParameters,
    apparent_power,
    tan_phi_from_power_factor,
)

QModeType = Literal["zero", "learned"]
"""Whether the reactive power is zero (NGED's convention) or learned from the readings."""

POWER_FACTOR_PRIOR_MEAN: Final[float] = 0.97
"""The centre of the power-factor prior."""
POWER_FACTOR_PRIOR_SD: Final[float] = 0.02 / 1.96
"""95% of the prior's mass lies on 0.95 to 0.99, a half-width of 1.96 standard deviations."""
Q0_PRIOR_SD_POINTS: Final[float] = 2.0
"""The scale of the prior on the reactive power at zero demand, in points of effective capacity."""
CAPACITY_START_MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 1.0, 2.0)
"""The naive fit's capacity increments are multiplied by these to make three of the four starts."""
MAX_ITERATIONS: Final[int] = 100
"""The most Gauss-Newton iterations of one start."""
MAX_DAMPING_TRIES: Final[int] = 8
"""The most times one iteration raises its damping before it gives up and stops."""
RELATIVE_TOLERANCE: Final[float] = 1e-9
"""A start has converged when an iteration lowers its loss by less than this share."""
INITIAL_DAMPING_SHARE: Final[float] = 1e-3
"""The first damping, as a share of the mean diagonal of the system."""
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
        loss: The penalised loss it ended at, including the reactive-power priors.
        final_capacity: The capacity of the last season-year at its end, in points.
        tan_phi: The ratio of reactive power to demand it ended at.
        iterations: The Gauss-Newton iterations it took.
    """

    name: str
    loss: float
    final_capacity: float
    tan_phi: float
    iterations: int


@dataclass(frozen=True)
class MeterFit:
    """The best start of a fit through the meter.

    Attributes:
        fit: The fitted coefficients, a one-member `GamFit` whose `predict_gam` gives signed real
            power `P`.
        qmode: Whether the reactive power was zero or learned.
        tan_phi: The ratio of reactive power to demand.
        q0: The reactive power at zero demand, in points.
        loss: The best start's penalised loss.
        starts: Every start's summary, so the spread across starts can be reported.
        demand_columns: True for the design columns that make up the demand `D`.
    """

    fit: GamFit
    qmode: QModeType
    tan_phi: float
    q0: float
    loss: float
    starts: tuple[StartSummary, ...]
    demand_columns: np.ndarray

    @property
    def power_factor(self) -> float:
        """The power factor `1 / sqrt(1 + tan_phi^2)` of the fit."""
        return float(1.0 / np.sqrt(1.0 + self.tan_phi**2))

    @property
    def capacity_spread(self) -> float:
        """The range of the starts' final capacities, in points."""
        capacities = [start.final_capacity for start in self.starts]
        return float(max(capacities) - min(capacities))


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
        qmode: `zero` for NGED's convention, `learned` to fit the power factor and `q0`.

    Returns:
        The start with the lowest loss, with every start's summary.

    Raises:
        ValueError: If the spec pins the capacity or has a clip or node, no row carries solar
            signal, or a start's quadratic program does not converge.
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
    problem = _Problem.build(cache=cache, features=features, layout=layout, index=index)
    summaries: list[StartSummary] = []
    best: tuple[float, np.ndarray, float, float] | None = None
    for name, start in problem.starts(features=features, index=index, layout=layout):
        theta, tan_phi, q0, loss, iterations = problem.minimise(theta=start, qmode=qmode)
        summaries.append(
            StartSummary(
                name=name,
                loss=loss,
                final_capacity=float(np.cumsum(theta[layout.slices["capacity"]])[-1]),
                tan_phi=tan_phi,
                iterations=iterations,
            )
        )
        if best is None or loss < best[0]:
            best = (loss, theta, tan_phi, q0)
    if best is None:
        msg = "the meter fit has no start"
        raise ValueError(msg)
    loss, theta, tan_phi, q0 = best
    fit = GamFit(
        layout=layout,
        theta=theta[None, :],
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
        tan_phi=tan_phi,
        q0=q0,
        loss=loss,
        starts=tuple(summaries),
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
class _Problem:
    """The fixed arrays of one meter fit: the design, the readings, and the penalty."""

    design: np.ndarray
    demand_design: np.ndarray
    demand_mask: np.ndarray
    readings: np.ndarray
    penalty: np.ndarray
    cost: np.ndarray
    constrained: torch.Tensor
    orthogonal: np.ndarray | None

    @staticmethod
    def build(
        *, cache: FeatureCache, features: SiteFeatures, layout: GamLayout, index: np.ndarray
    ) -> _Problem:
        """Build the design once, in memory, and the penalty."""
        design = np.empty((len(index), layout.n_columns))
        for start in range(0, len(index), 4000):
            rows = index[start : start + 4000]
            design[start : start + len(rows)] = design_matrix(
                cache=cache, features=features, layout=layout, index=rows
            )
        mask = demand_column_mask(layout=layout)
        penalty, cost = penalty_matrix(layout=layout, n_rows=len(index))
        orthogonal = None
        if layout.spec.orthogonal_penalty > 0.0:
            raw = orthogonal_penalty_matrix(
                cache=cache, features=features, layout=layout, index=index
            )
            gram_diagonal = float(np.einsum("ij,ij->j", design, design).mean())
            orthogonal = layout.spec.orthogonal_penalty * gram_diagonal / np.diag(raw).mean() * raw
        return _Problem(
            design=design,
            demand_design=design * mask[None, :],
            demand_mask=mask,
            readings=features.y[index],
            penalty=penalty if orthogonal is None else penalty + orthogonal,
            cost=cost,
            constrained=torch.as_tensor(layout.constrained),
            orthogonal=orthogonal,
        )

    def starts(
        self, *, features: SiteFeatures, index: np.ndarray, layout: GamLayout
    ) -> list[tuple[str, np.ndarray]]:
        """Return the named starting coefficients."""
        naive = self._solve_plain(weights=np.ones(len(index)))
        capacity = layout.slices["capacity"]
        result: list[tuple[str, np.ndarray]] = []
        for multiplier in CAPACITY_START_MULTIPLIERS:
            theta = naive.copy()
            theta[capacity] *= multiplier
            result.append((f"naive capacity x{multiplier:g}", theta))
        dark = features.dark[index]
        if dark.any():
            theta = self._solve_plain(weights=dark.astype(np.float64))
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

    def minimise(
        self, *, theta: np.ndarray, qmode: QModeType
    ) -> tuple[np.ndarray, float, float, float, int]:
        """Run Levenberg-Marquardt from `theta`; return the end state, its loss, and iterations.

        The unknowns are the coefficients and, for `learned`, `tan(phi)` and `q0`, stepped together:
        the linearised problem has two extra columns, and the priors on the power factor and `q0`
        are two extra pseudo-observations. A step is accepted only if it lowers the penalised loss
        (with the priors weighted by the residual variance of the iteration's start), and
        otherwise the damping grows.
        """
        learned = qmode == "learned"
        state = np.concatenate(
            [theta, [tan_phi_from_power_factor(power_factor=POWER_FACTOR_PRIOR_MEAN), 0.0]]
            if learned
            else [theta, [0.0, 0.0]]
        )
        damping = 0.0
        iterations = 0
        for iterations in range(1, MAX_ITERATIONS + 1):  # noqa: B007
            weight = self._prior_weight(state=state)
            loss = self._loss(state=state, weight=weight, learned=learned)
            candidate = self._step(state=state, weight=weight, learned=learned, damping=damping)
            new_loss = self._loss(state=candidate, weight=weight, learned=learned)
            tries = 0
            while new_loss > loss and tries < MAX_DAMPING_TRIES:
                damping = (
                    INITIAL_DAMPING_SHARE * float(np.diag(self.penalty).mean() + 1.0)
                    if damping == 0.0
                    else damping * DAMPING_GROWTH
                )
                candidate = self._step(state=state, weight=weight, learned=learned, damping=damping)
                new_loss = self._loss(state=candidate, weight=weight, learned=learned)
                tries += 1
            if new_loss > loss:
                break
            improvement = loss - new_loss
            state = candidate
            damping = damping / DAMPING_GROWTH if damping > 0.0 else 0.0
            if improvement <= RELATIVE_TOLERANCE * abs(loss):
                break
        weight = self._prior_weight(state=state)
        final_loss = self._loss(state=state, weight=weight, learned=learned)
        return state[:-2], float(state[-2]), float(state[-1]), final_loss, iterations

    def _prior_weight(self, *, state: np.ndarray) -> float:
        """Return the residual variance at `state`, which weights the priors."""
        squares = float(np.square(self.readings - self._apparent(state=state)[3]).sum())
        return squares / len(self.readings)

    def _loss(self, *, state: np.ndarray, weight: float, learned: bool) -> float:
        """Return the penalised loss: squared residuals, coefficient penalty, and the priors."""
        theta = state[:-2]
        apparent = self._apparent(state=state)[3]
        loss = float(
            np.square(self.readings - apparent).sum()
            + theta @ self.penalty @ theta
            + self.cost @ theta
        )
        if learned:
            power_factor = 1.0 / np.sqrt(1.0 + state[-2] ** 2)
            loss += weight * (
                ((power_factor - POWER_FACTOR_PRIOR_MEAN) / POWER_FACTOR_PRIOR_SD) ** 2
                + (state[-1] / Q0_PRIOR_SD_POINTS) ** 2
            )
        return loss

    def _apparent(
        self, *, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return `(P, D, Q, |S|)` at the state `(theta, tan_phi, q0)`."""
        theta, tan_phi, q0 = state[:-2], state[-2], state[-1]
        real = self.design @ theta
        demand = self.demand_design @ theta
        reactive = tan_phi * demand + q0
        return real, demand, reactive, np.sqrt(real * real + reactive * reactive + DEFAULT_EPSILON)

    def _step(
        self, *, state: np.ndarray, weight: float, learned: bool, damping: float
    ) -> np.ndarray:
        """Return the state that minimises the linearised, damped, bounded problem."""
        theta, tan_phi = state[:-2], state[-2]
        real, demand, reactive, apparent = self._apparent(state=state)
        n_coefficients = len(theta)
        n_unknowns = n_coefficients + (2 if learned else 0)
        jacobian = np.empty((len(self.readings), n_unknowns))
        jacobian[:, :n_coefficients] = (real / apparent)[:, None] * self.design
        if learned:
            jacobian[:, :n_coefficients] += (reactive * tan_phi / apparent)[:, None] * (
                self.demand_design
            )
            jacobian[:, n_coefficients] = reactive * demand / apparent
            jacobian[:, n_coefficients + 1] = reactive / apparent
        target = self.readings - apparent + jacobian @ state[:n_unknowns]
        gram = np.zeros((n_unknowns, n_unknowns))
        rhs = np.zeros(n_unknowns)
        gram[:n_coefficients, :n_coefficients] = self.penalty
        rhs[:n_coefficients] = -0.5 * self.cost
        gram += jacobian.T @ jacobian
        rhs += jacobian.T @ target
        constrained = self.constrained
        if learned:
            root = float(np.sqrt(weight))
            factor = (1.0 + tan_phi**2) ** 1.5
            power_factor = 1.0 / np.sqrt(1.0 + tan_phi**2)
            prior_slope = -root * tan_phi / factor / POWER_FACTOR_PRIOR_SD
            prior_residual = root * (power_factor - POWER_FACTOR_PRIOR_MEAN) / POWER_FACTOR_PRIOR_SD
            gram[n_coefficients, n_coefficients] += prior_slope**2
            rhs[n_coefficients] += prior_slope * (prior_slope * tan_phi - prior_residual)
            q0_slope = root / Q0_PRIOR_SD_POINTS
            gram[n_coefficients + 1, n_coefficients + 1] += q0_slope**2
            constrained = torch.cat([constrained, torch.tensor([True, False])])
        if damping > 0.0:
            gram += damping * np.eye(n_unknowns)
            rhs += damping * state[:n_unknowns]
        solved = _solve(matrix=gram, rhs=rhs, constrained=constrained)
        return np.concatenate([solved, [0.0, 0.0]]) if not learned else solved


def _solve(*, matrix: np.ndarray, rhs: np.ndarray, constrained: torch.Tensor) -> np.ndarray:
    """Solve one bounded quadratic program with the series' exact solver."""
    theta, _ = solve_bounded_quadratic(
        matrix=torch.as_tensor(matrix[None]),
        rhs=torch.as_tensor(rhs[None]),
        constrained=constrained,
    )
    return theta[0].numpy()
