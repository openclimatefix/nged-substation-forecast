"""Variable projection (VP) fit of the pooled differentiable-physics (DP) model.

`studies.dp_pooled.fit_pooled` moves every parameter by gradient descent. **Most of the pooled
model's parameters enter its prediction linearly once the non-linear ones are fixed.** The daily
shapes, the holiday, tent, harmonic, and lighting coefficients, the heating slopes, the capacity
increments, the levels, the deviations, and the last layer of the interaction network all multiply
a column that depends only on the non-linear parameters `phi`. This module solves for those linear
coefficients `theta` exactly at every step, with `studies.dp_gam.solve_bounded_quadratic`, and
moves only `phi` by gradient steps.

**The non-linear parameters `phi`:** the style loadings (`site_logits`), the heating threshold of
each substation and its population mean, the thermal-inertia logits, and the interaction network's
hidden layers. For fixed `phi` the problem in `theta` is one convex quadratic, so `theta*(phi)` is
unique. The projected objective `v(phi) = J(phi, theta*(phi))` therefore has the gradient
`dJ/dphi` taken at `theta*` held constant (Danskin's theorem): the bounds on `theta` do not depend
on `phi`, and every linear column carries a ridge or a prior.

**One objective, in nats.** `vp_objective` is the deterministic maximum a posteriori (MAP)
objective of the model for any parameter set, whether a gradient fit or this module produced the
parameters: the squared error over twice the fixed noise variance, plus the weight decay and the
site-level priors on the natural scale (the slopes and capacity increments themselves, not their
pre-softplus values). Mean-field posteriors are not used, and the MVA operator is fixed at
`tan(phi) = q0 = 0`, which makes it linear wherever the demand is positive.

**What is built.** `fit_pooled_vp` fits a pooled model (or one substation, as a pooled model of one
site). A capacity override may pin some substation-years and leave the rest learned (`NaN`).
`vp_site_fitter` plugs into `studies.dp_onboarding.onboard_site` to fit a new substation's
site-level parameters with every shared parameter frozen. **What is not built:** a pre-softplus
slope prior, slopes and the noise scale among `phi`, a heating basis of one slope per smoothing, and
the batching of several fits in one solve.
"""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from typing import Final, Literal

import torch
from torch import nn

from studies.dp_calendar import HALF_HOURS_PER_DAY, HEATING_THRESHOLD_C
from studies.dp_gam import HEATING_SCALE_C, N_BANDS, solve_bounded_quadratic
from studies.dp_pooled import (
    HOLIDAY_DECAY,
    N_PERIODS,
    N_PROFILE,
    NETWORK_DECAY,
    SHAPE_DECAY,
    THRESHOLD_PRIOR_SCALE_C,
    PooledBatch,
    PooledDemandModel,
    PooledFitResult,
    fit_pooled,
)

SharedBlockType = Literal["shapes", "holiday", "tent", "harmonics", "lighting", "network"]
"""The shared linear blocks, in the order of their columns."""
LocalBlockType = Literal["level", "deviation", "slope", "capacity"]
"""A substation's own linear blocks, in the order of their columns."""
MeanType = Literal["level", "slope", "first", "growth"]
"""The population means that are linear coefficients."""
SHARED_BLOCKS: Final[tuple[SharedBlockType, ...]] = (
    "shapes",
    "holiday",
    "tent",
    "harmonics",
    "lighting",
    "network",
)
LOCAL_BLOCKS: Final[tuple[LocalBlockType, ...]] = ("level", "deviation", "slope", "capacity")
MEANS: Final[tuple[MeanType, ...]] = ("level", "slope", "first", "growth")
CALENDAR_RIDGE: Final[float] = 1e-6
"""A ridge on the tent, harmonic, and lighting coefficients, so that each linear column has one.

The gradient fits put no penalty on these coefficients. The ridge makes the projected problem
strictly convex and is small enough (in nats) to leave the fit unchanged.
"""
SOFTPLUS_FLOOR: Final[float] = 1e-12
"""A coefficient below this maps to the softplus pre-image of this value (the bound at zero)."""
HISTORY_SIZE: Final[int] = 20
"""The number of curvature pairs L-BFGS keeps."""
MAX_EVALUATIONS: Final[int] = 300
"""The most objective evaluations of one L-BFGS fit, line search included."""
FALL_WINDOW: Final[int] = 10
"""L-BFGS stops when the best objective fell under `RELATIVE_FALL` over this many evaluations.

The window counts objective evaluations, line-search evaluations included, where a count of
iterations would not bound the work of one iteration.
"""
RELATIVE_FALL: Final[float] = 1e-7
"""The relative fall in the best objective that counts as progress."""
SOLVER_TOLERANCE: Final[float] = 1e-10
"""The bounded solver's optimality tolerance, relative to the largest entry of the right-hand side.

The solver's default of 1e-7 leaves coefficients of weakly determined columns off by about 1e-3,
which feeds an error of the same order into the gradient on `phi`.
"""
WARM_START_RATE: Final[float] = 0.05
"""Adam's learning rate in the optional warm start of `fit_pooled_vp`."""
ADAM_STEPS: Final[int] = 1200
"""Adam's steps on `phi`, with an exact solve each step, when L-BFGS fails its convergence test."""
ADAM_RATE: Final[float] = 0.02
"""Adam's learning rate for the fallback."""


@dataclass(frozen=True)
class VpSettings:
    """The fixed hyperparameters of the variable-projection objective.

    The population scales are the MAP scales of experiment 105's P-map fit, written on the natural
    scale: the pre-softplus scales of the slope and the growth increment (both 1.0) times the
    softplus slope at P-map's population means (0.985 and 0.946), and the first increment's 8.0
    times 1.00.

    Attributes:
        noise_sd: The standard deviation of the Gaussian noise, in points of effective capacity,
            fixed within a fit.
        level: The scale of the substations' levels around their population mean.
        site_logits: The scale of the style loadings around zero.
        deviation: The scale of the per-site deviation from the shared shapes.
        threshold: The scale of the heating thresholds around their population mean, and of the
            population mean's prior around 15.5 degrees.
        slope: The scale of the heating slopes around their population mean.
        growth_increment: The scale of the capacity increments after a substation's first trained
            year.
        first_increment: The scale of the first trained season-year's capacity increment.
    """

    noise_sd: float
    level: float = 5.0
    site_logits: float = 1.0
    deviation: float = 1.0
    threshold: float = THRESHOLD_PRIOR_SCALE_C
    slope: float = 0.985
    growth_increment: float = 0.946
    first_increment: float = 8.0


@dataclass(frozen=True)
class VpLayout:
    """The columns of the linear problem, in the order of the coefficient vector `theta`.

    Attributes:
        sites: The substations that have training rows, ascending.
        rows: The first and last-plus-one row of each substation in the batch.
        n_shapes: The number of shared daily shapes `K`.
        frozen_shared: True when the shared coefficients and the population means are held at the
            model's values (onboarding), so the problem holds one substation's own block only.
        shared: The slice of each shared block (empty slices when `frozen_shared`).
        n_shared: The number of shared columns.
        means: The column of each population mean that has site columns (none when
            `frozen_shared`).
        local_start: The first column of each substation's own block.
        local: Each substation's slices within its block, relative to its start.
        capacity_years: Each substation's season-years that have a capacity increment column.
        first_year: Each substation's first trained season-year, which holds its whole capacity.
        n_columns: The number of columns.
        constrained: True for the columns that must be non-negative.
    """

    sites: tuple[int, ...]
    rows: dict[int, tuple[int, int]]
    n_shapes: int
    frozen_shared: bool
    shared: dict[SharedBlockType, slice]
    n_shared: int
    means: dict[MeanType, int]
    local_start: dict[int, int]
    local: dict[int, dict[LocalBlockType, slice]]
    capacity_years: dict[int, tuple[int, ...]]
    first_year: dict[int, int]
    n_columns: int
    constrained: torch.Tensor


@dataclass(frozen=True)
class VpProblem:
    """A model, its rows, the columns, and the parts of the problem that do not depend on `phi`.

    Attributes:
        model: The pooled model, whose current `phi` sets the columns.
        batch: The rows, sorted by substation.
        layout: The columns.
        settings: The noise standard deviation and the prior scales.
        penalty: The quadratic prior matrix `R`, so the prior in nats is `theta^T R theta -
            2 shift^T theta` plus a constant.
        shift: The linear part of the prior, non-zero only for the frozen population means.
    """

    model: PooledDemandModel
    batch: PooledBatch
    layout: VpLayout
    settings: VpSettings
    penalty: torch.Tensor
    shift: torch.Tensor


@dataclass(frozen=True)
class SiteDesign:
    """One substation's part of the linear problem, `prediction = shared @ w + local @ v + offset`.

    Attributes:
        shared: The columns of the shared coefficients, shape (rows, shared columns). The
            calendar, lighting, and network columns are centred per season-year.
        local: The substation's own columns, shape (rows, width of its block).
        offset: The part of the prediction that does not depend on `theta`: the solar output of
            a pinned capacity, negative, shape (rows,).
    """

    shared: torch.Tensor
    local: torch.Tensor
    offset: torch.Tensor


@dataclass(frozen=True)
class VpFitResult:
    """What a variable-projection fit reports.

    Attributes:
        objective: The final objective in nats (`vp_objective`).
        objective_per_row: The same, divided by the number of rows.
        lbfgs_objective: The objective in nats at the end of L-BFGS.
        adam_objective: The objective in nats after the Adam fallback, or `None` when it did not
            run.
        warm_start_objective: The objective in nats after the optional Adam warm start, or `None`
            when there was none.
        evaluations: The number of objective evaluations L-BFGS spent, line search included.
        converged: True only if the best objective fell by under a relative 1e-7 over the last 10
            evaluations; a fit that reaches the evaluation cap or whose L-BFGS step fails is not.
        gradient_max_abs: The largest absolute entry of the final per-row objective gradient on
            `phi`.
        kkt_violation: The solver's violation of the optimality conditions at the final `phi`,
            relative to the largest entry of the right-hand side.
        seconds: Wall-clock seconds.
        peak_memory_bytes: The peak GPU memory allocated, or 0 on the CPU.
        history: The objective per row at every evaluation of L-BFGS.
    """

    objective: float
    objective_per_row: float
    lbfgs_objective: float
    adam_objective: float | None
    warm_start_objective: float | None
    evaluations: int
    converged: bool
    gradient_max_abs: float
    kkt_violation: float
    seconds: float
    peak_memory_bytes: int
    history: list[float]


class _StopFitError(Exception):
    """Raised from the L-BFGS closure to end the fit once the objective has stopped falling."""


def inverse_softplus(*, value: torch.Tensor) -> torch.Tensor:
    """Return `x` with `softplus(x) = value`, flooring `value` at `SOFTPLUS_FLOOR`."""
    floor = torch.full_like(value, SOFTPLUS_FLOOR)
    clamped = torch.maximum(value, floor)
    return torch.log(torch.expm1(clamped))


def freeze_operator_for_vp(*, model: PooledDemandModel) -> None:
    """Set the MVA operator to `tan(phi) = q0 = 0` and stop its parameters from training.

    With `Q = 0` the apparent-power operator reads `|P|`, which is `P` wherever the demand is
    positive, so the observation operator is linear in the demand.
    """
    with torch.no_grad():
        model.tan_phi.zero_()
        model.q0.zero_()
    model.tan_phi.requires_grad = False
    model.q0.requires_grad = False


def phi_parameters(*, model: PooledDemandModel) -> list[nn.Parameter]:
    """Return the non-linear parameters: loadings, thresholds, inertia, and hidden layers."""
    parameters: list[nn.Parameter] = []
    if model.config.shared_shapes:
        parameters.append(model.site_logits)
    parameters += [model.threshold, model.inertia_logits, model.population_mean["threshold"]]
    if model.interaction is not None:
        parameters += list(model.interaction[:-1].parameters())
    return parameters


@contextmanager
def trainable_only(*, model: PooledDemandModel, parameters: list[nn.Parameter]) -> Generator[None]:
    """Let only `parameters` train inside the block, and restore every flag afterwards."""
    saved = {name: p.requires_grad for name, p in model.named_parameters()}
    try:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in parameters:
            parameter.requires_grad = True
        yield
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = saved[name]


def pinned_capacity(*, model: PooledDemandModel) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the pinned capacities and where they are pinned, each of shape (sites, years).

    A `NaN` in the model's `capacity_override` means the capacity is learned there. The values are
    0 where the capacity is learned.
    """
    shape = (model.n_sites, model.n_season_years)
    if model.capacity_override is None:
        return torch.zeros(shape, dtype=model.level.dtype), torch.zeros(shape, dtype=torch.bool)
    override = model.capacity_override.to(model.level.dtype)
    pinned = ~torch.isnan(override)
    return torch.where(pinned, override, torch.zeros_like(override)), pinned


def learned_increment_mask(*, model: PooledDemandModel) -> torch.Tensor:
    """Return where a capacity increment is a coefficient of the fit, shape (sites, years).

    An increment of a trained season-year at a solar substation is one when some trained
    season-year at or after it has a learned (not pinned) capacity, because only then does the
    increment reach the prediction. With nothing pinned this is every trained season-year of a
    solar substation.

    Args:
        model: The pooled model, with `trained` and `solar_sites` set.

    Returns:
        A boolean tensor of shape (sites, years).

    Raises:
        ValueError: If a solar substation has a learned capacity in a season-year but none in a
            trained season-year, so no row could fit the increments that capacity rests on.
    """
    trained = model.trained > 0
    solar = model.solar_sites[:, None] > 0
    if model.capacity_override is None:
        learned_year = trained
    else:
        unpinned = torch.isnan(model.capacity_override)
        learned_year = trained & unpinned
        orphan = solar[:, 0] & unpinned.any(dim=1) & ~learned_year.any(dim=1)
        if bool(orphan.any()):
            msg = (
                "a solar substation has a learned capacity only in season-years without training "
                "rows: pin it, or give it a learned trained season-year"
            )
            raise ValueError(msg)
    reach = torch.flip(
        torch.cummax(torch.flip(learned_year.to(torch.int64), dims=[1]), dim=1).values, dims=[1]
    )
    return trained & solar & (reach > 0)


def vp_objective(
    *, model: PooledDemandModel, batch: PooledBatch, settings: VpSettings
) -> torch.Tensor:
    """Return the deterministic MAP objective of `batch` in nats, for any parameter set.

    The data term is the squared error over twice the noise variance. The penalty is the weight
    decay of `PooledDemandModel.penalty` plus a small ridge on the tent, harmonic, and lighting
    coefficients. The site priors are Gaussian on the natural scale around learned population
    means (the heating slope and the capacity increments are the softplus outputs themselves),
    with the heating threshold's population mean centred on 15.5 degrees. The MVA operator must
    be fixed at zero reactive power (`freeze_operator_for_vp`). The model's stored centring means
    are left as they were.

    Args:
        model: The pooled model, in any state; it is put in training mode for the call so the
            calendar terms are centred on the batch's own rows.
        batch: The rows.
        settings: The noise standard deviation and the prior scales.

    Returns:
        The objective, a scalar that carries the autograd graph.

    Raises:
        ValueError: If the model is variational, has a per-site capacity prior, or has a non-zero
            MVA operator.
    """
    if model.config.variational:
        msg = "the variable-projection objective is the MAP objective, not the variational one"
        raise ValueError(msg)
    if not bool(torch.isnan(model.capacity_prior).all()):
        msg = "the variable-projection objective has no per-site capacity prior"
        raise ValueError(msg)
    if bool((model.tan_phi != 0).any()) or bool((model.q0 != 0).any()):
        msg = "the MVA operator must have tan(phi) = q0 = 0; call freeze_operator_for_vp"
        raise ValueError(msg)
    was_training = model.training
    saved_means = model.centre_means.clone()
    model.train()
    try:
        residual = model(batch) - batch.y
    finally:
        model.train(was_training)
        with torch.no_grad():
            model.centre_means.copy_(saved_means)
    total = (residual**2).sum() / (2.0 * settings.noise_sd**2)
    total = total + _weight_decay(model=model)
    return total + _site_priors(model=model, settings=settings)


def _weight_decay(*, model: PooledDemandModel) -> torch.Tensor:
    """Return the weight decay and ridge terms of the objective."""
    term = HOLIDAY_DECAY * (model.holiday**2).sum()
    term = term + CALENDAR_RIDGE * (
        (model.tent_coef**2).sum() + (model.harmonic_coef**2).sum() + (model.lighting_coef**2).sum()
    )
    if model.config.shared_shapes:
        term = term + SHAPE_DECAY * (model.shapes**2).sum()
    if model.interaction is not None:
        term = term + NETWORK_DECAY * sum((p**2).sum() for p in model.interaction.parameters())
    return term


def _tie(*, values: torch.Tensor, centre: torch.Tensor | float, scale: float) -> torch.Tensor:
    """Return the Gaussian prior term `(values - centre)^2 / (2 scale^2)`, elementwise."""
    return (values - centre) ** 2 / (2.0 * scale**2)


def _site_priors(*, model: PooledDemandModel, settings: VpSettings) -> torch.Tensor:
    """Return the site-level prior terms on the natural scale, over the substations with rows."""
    active = model.site_active.to(model.level.dtype)
    both = active[:, None]
    mean_threshold = model.population_mean["threshold"]
    term = _tie(values=mean_threshold, centre=HEATING_THRESHOLD_C, scale=settings.threshold)
    level_term = _tie(
        values=model.level, centre=model.population_mean["level"], scale=settings.level
    )
    threshold_term = _tie(values=model.threshold, centre=mean_threshold, scale=settings.threshold)
    term = term + (level_term * active).sum() + (threshold_term * active).sum()
    term = term + (_tie(values=model.deviation, centre=0.0, scale=settings.deviation) * both).sum()
    if model.config.shared_shapes:
        logit_term = _tie(values=model.site_logits, centre=0.0, scale=settings.site_logits)
        term = term + (logit_term * both).sum()
    slope = nn.functional.softplus(model.slope_raw)
    slope_mean = nn.functional.softplus(model.population_mean["slope_raw"])
    term = term + (_tie(values=slope, centre=slope_mean, scale=settings.slope) * both).sum()
    increment = nn.functional.softplus(model.capacity_raw)
    first, growth = model.capacity_masks()
    learned = learned_increment_mask(model=model).to(increment.dtype)
    first_mean = nn.functional.softplus(model.population_mean["capacity_first"])
    growth_mean = nn.functional.softplus(model.population_mean["capacity_raw"])
    first_term = _tie(values=increment, centre=first_mean, scale=settings.first_increment)
    growth_term = _tie(values=increment, centre=growth_mean, scale=settings.growth_increment)
    return term + (first_term * first * learned).sum() + (growth_term * growth * learned).sum()


def vp_loss_function(
    *, settings: VpSettings
) -> Callable[[PooledDemandModel, PooledBatch], torch.Tensor]:
    """Return the per-row variable-projection objective as a loss for `fit_pooled`.

    Passing it as `loss_function` runs experiment 105's Adam fit on this module's objective, which
    isolates the fitting method from the objective.

    Args:
        settings: The noise standard deviation and the prior scales.

    Returns:
        A function of the model and the batch that gives `vp_objective` divided by the rows.
    """

    def loss(model: PooledDemandModel, batch: PooledBatch) -> torch.Tensor:
        return vp_objective(model=model, batch=batch, settings=settings) / batch.n_rows

    return loss


def fit_pooled_adam_on_vp_objective(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    settings: VpSettings,
    steps: int,
    learning_rate: float = 0.05,
    final_rate_fraction: float = 1.0,
) -> PooledFitResult:
    """Fit by experiment 105's Adam loop on `vp_objective`, with the MVA operator frozen at zero.

    Args:
        model: The pooled model, with the MAP prior (`DpConfig.population_scales` set) so the
            population means train.
        batch: The rows.
        settings: The noise standard deviation and the prior scales.
        steps: The number of Adam steps.
        learning_rate: Adam's starting learning rate.
        final_rate_fraction: The fraction of the rate left at the last step.

    Returns:
        The fit result of `fit_pooled`.
    """
    freeze_operator_for_vp(model=model)
    return fit_pooled(
        model=model,
        batch=batch,
        steps=steps,
        learning_rate=learning_rate,
        final_rate_fraction=final_rate_fraction,
        loss_function=vp_loss_function(settings=settings),
    )


def site_bounds(*, batch: PooledBatch, n_sites: int) -> list[tuple[int, int]]:
    """Return each substation's first and last-plus-one row of `batch`.

    Args:
        batch: The rows, which must be sorted by substation (as `build_pooled_batch` makes them).
        n_sites: The number of substations in the model.

    Returns:
        One `(start, stop)` per substation; an empty range for a substation with no rows.

    Raises:
        ValueError: If the rows are not sorted by substation.
    """
    if bool((batch.site[1:] < batch.site[:-1]).any()):
        msg = "the batch's rows must be sorted by substation"
        raise ValueError(msg)
    counts = torch.bincount(batch.site, minlength=n_sites).tolist()
    bounds = []
    start = 0
    for count in counts:
        bounds.append((start, start + int(count)))
        start += int(count)
    return bounds


def _slice_batch(*, batch: PooledBatch, start: int, stop: int) -> PooledBatch:
    """Return the rows `start` to `stop` of `batch`."""
    rows = {
        field.name: getattr(batch, field.name)[start:stop]
        for field in fields(batch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
    }
    return replace(batch, **rows)


def _last_layer(*, model: PooledDemandModel) -> nn.Linear | None:
    """Return the interaction network's last layer, or `None` when the network is off."""
    if model.interaction is None:
        return None
    last = model.interaction[-1]
    if not isinstance(last, nn.Linear):
        msg = "the interaction network must end in a linear layer"
        raise TypeError(msg)
    return last


def make_vp_layout(
    *, model: PooledDemandModel, batch: PooledBatch, frozen_shared: bool = False
) -> VpLayout:
    """Lay out the linear problem's columns for the substations that have rows in `batch`.

    The model's `trained` and `solar_sites` buffers must be set (`mark_trained_years`). A
    substation gets capacity columns for its trained season-years that reach a learned capacity
    (`learned_increment_mask`).

    Args:
        model: The pooled model.
        batch: The rows.
        frozen_shared: Hold every shared coefficient and population mean at the model's values, so
            the layout has only the substations' own blocks. For onboarding.

    Returns:
        The layout.
    """
    rows = {
        site: (start, stop)
        for site, (start, stop) in enumerate(site_bounds(batch=batch, n_sites=model.n_sites))
        if stop > start
    }
    sites = tuple(rows)
    last_layer = _last_layer(model=model)
    hidden = 0 if last_layer is None else last_layer.in_features
    n_shapes = model.config.shared_shapes
    sizes: dict[SharedBlockType, int] = {
        "shapes": n_shapes * N_PROFILE,
        "holiday": (N_PERIODS + 1) * N_BANDS,
        "tent": 2 * N_BANDS,
        "harmonics": 4 * N_BANDS,
        "lighting": 4,
        "network": hidden,
    }
    shared: dict[SharedBlockType, slice] = {}
    position = 0
    for name in SHARED_BLOCKS:
        size = 0 if frozen_shared else sizes[name]
        shared[name] = slice(position, position + size)
        position += size
    n_shared = position
    learned = learned_increment_mask(model=model)
    capacity_years = {
        site: tuple(int(y) for y in torch.nonzero(learned[site]).flatten()) for site in sites
    }
    first_year = {
        site: int(torch.nonzero(model.trained[site] > 0).flatten()[0])
        for site in sites
        if bool((model.trained[site] > 0).any())
    }
    means: dict[MeanType, int] = {}
    if not frozen_shared:
        wanted: dict[MeanType, bool] = {
            "level": True,
            "slope": True,
            "first": any(first_year.get(s) in capacity_years[s] for s in sites),
            "growth": any(y != first_year.get(s) for s in sites for y in capacity_years[s]),
        }
        for name in MEANS:
            if wanted[name]:
                means[name] = position
                position += 1
    local_start: dict[int, int] = {}
    local: dict[int, dict[LocalBlockType, slice]] = {}
    for site in sites:
        local_start[site] = position
        widths: dict[LocalBlockType, int] = {
            "level": 1,
            "deviation": N_PROFILE,
            "slope": N_BANDS,
            "capacity": len(capacity_years[site]),
        }
        slices: dict[LocalBlockType, slice] = {}
        offset = 0
        for name in LOCAL_BLOCKS:
            slices[name] = slice(offset, offset + widths[name])
            offset += widths[name]
        local[site] = slices
        position += offset
    constrained = torch.zeros(position, dtype=torch.bool)
    for site in sites:
        for name in ("slope", "capacity"):
            block = local[site][name]
            constrained[local_start[site] + block.start : local_start[site] + block.stop] = True
    return VpLayout(
        sites=sites,
        rows=rows,
        n_shapes=n_shapes,
        frozen_shared=frozen_shared,
        shared=shared,
        n_shared=n_shared,
        means=means,
        local_start=local_start,
        local=local,
        capacity_years=capacity_years,
        first_year=first_year,
        n_columns=position,
        constrained=constrained,
    )


def _centre_per_slot(*, values: torch.Tensor, slot: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Subtract each column's mean over the rows of each slot (a substation and season-year)."""
    counts = torch.bincount(slot, minlength=n_slots).clamp(min=1).to(values.dtype)
    sums = torch.zeros((n_slots, values.shape[1]), dtype=values.dtype, device=values.device)
    means = sums.index_add(0, slot, values) / counts[:, None]
    return values - means[slot]


def _capacity_terms(
    *, model: PooledDemandModel, layout: VpLayout, rows: PooledBatch, site: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a substation's capacity-increment columns and the solar offset of pinned years.

    Args:
        model: The pooled model.
        layout: The columns, which name the substation's increment season-years.
        rows: The substation's rows.
        site: The substation.

    Returns:
        The columns, shape (rows, increments), and the offset, shape (rows,): minus the pinned
        capacity times the solar shape on the rows of a pinned season-year, 0 elsewhere. A
        pinned season-year's rows have no increment columns.
    """
    dtype = rows.y.dtype
    slot = rows.season_slot - site * model.n_season_years
    value, pinned = pinned_capacity(model=model)
    row_pinned = pinned.to(rows.y.device)[site][slot]
    solar = rows.solar_gate.to(dtype) * rows.pv_shape
    offset = -solar * torch.where(row_pinned, value.to(rows.y.device)[site][slot], 0.0)
    years = layout.capacity_years[site]
    after = slot[:, None] >= torch.tensor(years, device=rows.y.device)[None, :]
    columns = -(solar * (~row_pinned).to(dtype))[:, None] * after.to(dtype)
    return columns, offset


def linear_design(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout, site: int
) -> SiteDesign:
    """Build the columns of one substation's rows for the model's current `phi`.

    This is the dense reference of the design: the prediction `shared @ w + local @ v + offset`
    equals `PooledDemandModel.forward` when `w` and `v` hold the model's linear parameters on the
    natural scale (the heating slope and the capacity increments as softplus outputs) and the MVA
    operator has zero reactive power. The shared columns are all built, whether or not
    `layout.frozen_shared` is set.

    Args:
        model: The pooled model, whose non-linear parameters `phi` set the columns.
        batch: The rows of the whole fit, sorted by substation.
        layout: The columns.
        site: The substation.

    Returns:
        The substation's columns and offset.
    """
    start, stop = layout.rows[site]
    rows = _slice_batch(batch=batch, start=start, stop=stop)
    n, dtype, device = rows.n_rows, rows.y.dtype, rows.y.device
    profile = rows.day_type * HALF_HOURS_PER_DAY + rows.half_hour
    band_hot = nn.functional.one_hot(rows.band, N_BANDS).to(dtype)
    mixed = rows.smoothed_temperature @ torch.softmax(model.inertia_logits, dim=0)
    parts: list[torch.Tensor] = []
    if layout.n_shapes:
        mix = torch.softmax(model.site_logits[site], dim=0)
        index = torch.arange(layout.n_shapes, device=device)[None, :] * N_PROFILE + profile[:, None]
        shapes = torch.zeros((n, layout.n_shapes * N_PROFILE), dtype=dtype, device=device)
        shapes.scatter_(1, index, mix[None, :].expand(n, -1).to(dtype))
        parts.append(shapes)
    calendar = [
        nn.functional.one_hot(rows.holiday * N_BANDS + rows.band, (N_PERIODS + 1) * N_BANDS).to(
            dtype
        ),
        (rows.tent[:, :, None] * band_hot[:, None, :]).reshape(n, -1),
        (rows.harmonics[:, :, None] * band_hot[:, None, :]).reshape(n, -1),
        rows.lighting,
    ]
    if model.interaction is not None:
        inputs = model.interaction_inputs(batch=rows, mixed=mixed)
        calendar.append(model.interaction[:-1](inputs))
    slot = rows.season_slot - site * model.n_season_years
    centred = _centre_per_slot(
        values=torch.cat(calendar, dim=1), slot=slot, n_slots=model.n_season_years
    )
    shared = torch.cat([*parts, centred], dim=1)
    hinge = torch.relu(model.threshold[site] - mixed) / HEATING_SCALE_C
    capacity, offset = _capacity_terms(model=model, layout=layout, rows=rows, site=site)
    local = torch.cat(
        [
            torch.ones((n, 1), dtype=dtype, device=device),
            nn.functional.one_hot(profile, N_PROFILE).to(dtype),
            hinge[:, None] * band_hot,
            capacity,
        ],
        dim=1,
    )
    return SiteDesign(shared=shared, local=local, offset=offset)


def _sparse_columns(
    *,
    model: PooledDemandModel,
    layout: VpLayout,
    rows: PooledBatch,
    site: int,
    profile: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the one-hot columns (shapes, holiday indicators, deviation) as row-wise entries.

    Every row has the same number of non-zero entries among these columns: one per shared shape,
    one holiday indicator, and one deviation indicator.

    Args:
        model: The pooled model, whose style loadings set the shapes' values.
        layout: The columns.
        rows: The substation's rows.
        site: The substation.
        profile: The daily-profile index of each row.

    Returns:
        The column and the value of each row's entries, both of shape (rows, entries per row), and
        the number of these columns.
    """
    n, dtype, device = rows.n_rows, rows.y.dtype, rows.y.device
    shapes_width, holiday_width = layout.n_shapes * N_PROFILE, (N_PERIODS + 1) * N_BANDS
    cols, vals = [], []
    if layout.n_shapes:
        mix = torch.softmax(model.site_logits[site], dim=0)
        shape_columns = torch.arange(layout.n_shapes, device=device)[None, :] * N_PROFILE
        cols.append(shape_columns + profile[:, None])
        vals.append(mix[None, :].expand(n, -1))
    cols.append((shapes_width + rows.holiday * N_BANDS + rows.band)[:, None])
    cols.append((shapes_width + holiday_width + profile)[:, None])
    vals += [torch.ones((n, 1), dtype=dtype, device=device)] * 2
    return torch.cat(cols, dim=1), torch.cat(vals, dim=1), shapes_width + holiday_width + N_PROFILE


def _dense_columns(
    *, model: PooledDemandModel, layout: VpLayout, rows: PooledBatch, site: int
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Return the continuous columns, how many of their first columns are centred, and the offset.

    The order is tent, harmonics, lighting, network (all centred), then level, heating, and
    capacity (not centred).
    """
    n, dtype, device = rows.n_rows, rows.y.dtype, rows.y.device
    band_hot = nn.functional.one_hot(rows.band, N_BANDS).to(dtype)
    mixed = rows.smoothed_temperature @ torch.softmax(model.inertia_logits, dim=0)
    dense = [
        (rows.tent[:, :, None] * band_hot[:, None, :]).reshape(n, -1),
        (rows.harmonics[:, :, None] * band_hot[:, None, :]).reshape(n, -1),
        rows.lighting,
    ]
    if model.interaction is not None:
        dense.append(model.interaction[:-1](model.interaction_inputs(batch=rows, mixed=mixed)))
    n_centred = sum(block.shape[1] for block in dense)
    hinge = torch.relu(model.threshold[site] - mixed) / HEATING_SCALE_C
    capacity, offset = _capacity_terms(model=model, layout=layout, rows=rows, site=site)
    dense += [torch.ones((n, 1), dtype=dtype, device=device), hinge[:, None] * band_hot, capacity]
    return torch.cat(dense, dim=1), n_centred, offset


def _site_normal_equations(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout, site: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return one substation's columns, `X^T X` block, and `X^T y` block of the normal equations.

    This is the fast twin of `linear_design`: the one-hot columns (the shared shapes, the holiday
    indicators, and the deviation) are handled row-wise from their few non-zero entries, so their
    products with every other column cost a few operations per row, and only the continuous
    columns (tent, harmonics, lighting, network, level, heating, capacity) form dense products.
    The calendar, lighting, and network columns are centred per season-year by subtracting the
    slot means from the products afterwards, which is exact: for a centred column `i`,
    `x_i^T x_j` loses `sum_slots n m_i m_j` with `m` the slot means of the raw columns, and
    `x_i^T y` loses `sum_slots n m_i mean(y)`.

    Args:
        model: The pooled model, whose current `phi` sets the columns.
        batch: The rows of the whole fit, sorted by substation.
        layout: The columns.
        site: The substation.

    Returns:
        The system's column index of each of the substation's raw columns (sparse columns first,
        then dense), the block of `X^T X`, and the block of `X^T y`.
    """
    start, stop = layout.rows[site]
    rows = _slice_batch(batch=batch, start=start, stop=stop)
    dtype, device = rows.y.dtype, rows.y.device
    n_slots = model.n_season_years
    profile = rows.day_type * HALF_HOURS_PER_DAY + rows.half_hour
    slot = rows.season_slot - site * n_slots
    column, value, width_sparse = _sparse_columns(
        model=model, layout=layout, rows=rows, site=site, profile=profile
    )
    dense, n_centred, offset = _dense_columns(model=model, layout=layout, rows=rows, site=site)
    target = rows.y - offset
    entries = column.shape[1]
    g_ss = torch.zeros((width_sparse, width_sparse), dtype=dtype, device=device)
    g_sd = torch.zeros((width_sparse, dense.shape[1]), dtype=dtype, device=device)
    xty_s = torch.zeros(width_sparse, dtype=dtype, device=device)
    for a in range(entries):
        g_sd.index_add_(0, column[:, a], value[:, a, None] * dense)
        xty_s.index_add_(0, column[:, a], value[:, a] * target)
        for b in range(entries):
            g_ss.index_put_(
                (column[:, a], column[:, b]), value[:, a] * value[:, b], accumulate=True
            )
    gram = torch.cat(
        [torch.cat([g_ss, g_sd], dim=1), torch.cat([g_sd.T, dense.T @ dense], dim=1)], dim=0
    )
    xty = torch.cat([xty_s, dense.T @ target])
    counts = torch.bincount(slot, minlength=n_slots).to(dtype)
    sums_sparse = torch.zeros((n_slots, width_sparse), dtype=dtype, device=device)
    sums_sparse.index_put_(
        (slot[:, None].expand_as(column).reshape(-1), column.reshape(-1)),
        value.reshape(-1),
        accumulate=True,
    )
    sums_dense = torch.zeros((n_slots, dense.shape[1]), dtype=dtype, device=device)
    sums_dense.index_add_(0, slot, dense)
    safe = counts.clamp(min=1)
    means = torch.cat([sums_sparse, sums_dense], dim=1) / safe[:, None]
    mean_target = (
        torch.zeros(n_slots, dtype=dtype, device=device).index_add_(0, slot, target) / safe
    )
    centred = torch.zeros(width_sparse + dense.shape[1], dtype=torch.bool, device=device)
    shapes_width = layout.n_shapes * N_PROFILE
    centred[shapes_width : shapes_width + (N_PERIODS + 1) * N_BANDS] = True
    centred[width_sparse : width_sparse + n_centred] = True
    correction = means.T @ (counts[:, None] * means)
    pair = centred[:, None] | centred[None, :]
    gram = gram - torch.where(pair, correction, torch.zeros_like(gram))
    xty = xty - torch.where(centred, means.T @ (counts * mean_target), torch.zeros_like(xty))
    return _system_columns(layout=layout, site=site, device=device), gram, xty


def _system_columns(*, layout: VpLayout, site: int, device: torch.device) -> torch.Tensor:
    """Return the system column of every raw column of `_site_normal_equations`, in its order."""
    shared, block, begin = layout.shared, layout.local[site], layout.local_start[site]

    def span(window: slice, offset: int = 0) -> torch.Tensor:
        return torch.arange(window.start + offset, window.stop + offset, device=device)

    return torch.cat(
        [
            span(shared["shapes"]),
            span(shared["holiday"]),
            span(block["deviation"], begin),
            span(shared["tent"]),
            span(shared["harmonics"]),
            span(shared["lighting"]),
            span(shared["network"]),
            span(block["level"], begin),
            span(block["slope"], begin),
            span(block["capacity"], begin),
        ]
    )


def _tie_columns(
    *,
    matrix: torch.Tensor,
    diagonal: torch.Tensor,
    shift: torch.Tensor,
    columns: list[int],
    scale: float,
    mean_column: int | None = None,
    mean_value: float = 0.0,
) -> None:
    """Add the prior `(theta_j - mean)^2 / (2 scale^2)` of each column `j` to the prior matrix.

    A `mean_column` is a free population mean: the prior couples the columns to that column of
    `theta`. Without one the mean is the number `mean_value` (0 by default), which enters the
    linear part `shift`.
    """
    weight = 1.0 / (2.0 * scale**2)
    index = torch.tensor(columns, dtype=torch.long, device=matrix.device)
    diagonal[index] += weight
    if mean_column is not None:
        diagonal[mean_column] += weight * index.numel()
        matrix[index, mean_column] -= weight
        matrix[mean_column, index] -= weight
    else:
        shift[index] += weight * mean_value


def make_vp_problem(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    settings: VpSettings,
    frozen_shared: bool = False,
) -> VpProblem:
    """Lay out the columns and build the prior matrix for a model and its rows.

    The model's `trained` and `solar_sites` buffers must be set before the call.

    Args:
        model: The pooled model.
        batch: The rows, sorted by substation.
        settings: The noise standard deviation and the prior scales.
        frozen_shared: Hold the shared coefficients and population means at the model's values
            (see `make_vp_layout`); the population means then centre the priors as constants.

    Returns:
        The problem.
    """
    layout = make_vp_layout(model=model, batch=batch, frozen_shared=frozen_shared)
    dtype, device = batch.y.dtype, batch.y.device
    p = layout.n_columns
    matrix = torch.zeros((p, p), dtype=dtype, device=device)
    diagonal = torch.zeros(p, dtype=dtype, device=device)
    shift = torch.zeros(p, dtype=dtype, device=device)
    decay: dict[SharedBlockType, float] = {
        "shapes": SHAPE_DECAY,
        "holiday": HOLIDAY_DECAY,
        "tent": CALENDAR_RIDGE,
        "harmonics": CALENDAR_RIDGE,
        "lighting": CALENDAR_RIDGE,
        "network": NETWORK_DECAY,
    }
    for name, weight in decay.items():
        diagonal[layout.shared[name]] = weight
    frozen: dict[MeanType, float] = {
        "level": float(model.population_mean["level"]),
        "slope": float(nn.functional.softplus(model.population_mean["slope_raw"])),
        "first": float(nn.functional.softplus(model.population_mean["capacity_first"])),
        "growth": float(nn.functional.softplus(model.population_mean["capacity_raw"])),
    }

    scales: dict[MeanType, float] = {
        "level": settings.level,
        "slope": settings.slope,
        "first": settings.first_increment,
        "growth": settings.growth_increment,
    }
    for site in layout.sites:
        start = layout.local_start[site]
        block = layout.local[site]
        first = [
            start + block["capacity"].start + i
            for i, year in enumerate(layout.capacity_years[site])
            if year == layout.first_year[site]
        ]
        growth = [
            start + block["capacity"].start + i
            for i, year in enumerate(layout.capacity_years[site])
            if year != layout.first_year[site]
        ]
        ties: list[tuple[list[int], MeanType]] = [
            ([start + block["level"].start], "level"),
            (list(range(start + block["slope"].start, start + block["slope"].stop)), "slope"),
            (first, "first"),
            (growth, "growth"),
        ]
        for columns, name in ties:
            if columns:
                _tie_columns(
                    matrix=matrix,
                    diagonal=diagonal,
                    shift=shift,
                    columns=columns,
                    scale=scales[name],
                    mean_column=None if frozen_shared else layout.means[name],
                    mean_value=frozen[name] if frozen_shared else 0.0,
                )
        deviation = list(range(start + block["deviation"].start, start + block["deviation"].stop))
        _tie_columns(
            matrix=matrix,
            diagonal=diagonal,
            shift=shift,
            columns=deviation,
            scale=settings.deviation,
        )
    return VpProblem(
        model=model,
        batch=batch,
        layout=layout,
        settings=settings,
        penalty=matrix + torch.diag(diagonal),
        shift=shift,
    )


def _shared_theta(*, model: PooledDemandModel, n_shapes: int) -> torch.Tensor:
    """Return the model's shared linear coefficients in the order of the shared columns."""
    parts = []
    if n_shapes:
        parts.append(model.shapes.reshape(-1))
    parts += [
        model.holiday.reshape(-1),
        model.tent_coef.reshape(-1),
        model.harmonic_coef.reshape(-1),
        model.lighting_coef.reshape(-1),
    ]
    last = _last_layer(model=model)
    if last is not None:
        parts.append(last.weight.reshape(-1))
    return torch.cat(parts)


@torch.no_grad()
def assemble_system(*, problem: VpProblem) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `A` and `b` of the problem: minimise `theta^T A theta - 2 b^T theta`, bounds aside.

    The squared error and the prior are both multiplied by twice the noise variance, so `A` is the
    Gram matrix plus `2 sigma^2 R`. The Gram matrix is built one substation at a time, so no dense
    matrix of all the rows is formed; a substation's block touches the shared columns and its own.
    With frozen shared coefficients the shared columns move into the offset and only the
    substation's own block is solved.

    Args:
        problem: The model, the rows, the columns, and the prior.

    Returns:
        `(A, b)` of shapes (columns, columns) and (columns,).
    """
    model, batch, layout = problem.model, problem.batch, problem.layout
    dtype, device = batch.y.dtype, batch.y.device
    p = layout.n_columns
    gram = torch.zeros((p, p), dtype=dtype, device=device)
    rhs = torch.zeros(p, dtype=dtype, device=device)
    if layout.frozen_shared:
        for site in layout.sites:
            design = linear_design(model=model, batch=batch, layout=layout, site=site)
            start, stop = layout.rows[site]
            frozen = design.shared @ _shared_theta(model=model, n_shapes=layout.n_shapes)
            target = batch.y[start:stop] - design.offset - frozen
            columns = slice(
                layout.local_start[site], layout.local_start[site] + design.local.shape[1]
            )
            gram[columns, columns] = design.local.T @ design.local
            rhs[columns] = design.local.T @ target
    else:
        for site in layout.sites:
            index, block, vector = _site_normal_equations(
                model=model, batch=batch, layout=layout, site=site
            )
            gram[index[:, None], index[None, :]] += block
            rhs[index] += vector
    scale = 2.0 * problem.settings.noise_sd**2
    return gram + scale * problem.penalty, rhs + scale * problem.shift


def solve_theta(*, problem: VpProblem) -> tuple[torch.Tensor, float]:
    """Return the exact linear coefficients for the model's current `phi`, and the KKT violation.

    Args:
        problem: The model, the rows, the columns, and the prior.

    Returns:
        `theta*` of shape (columns,) and the solver's relative violation of the optimality
        conditions.
    """
    matrix, rhs = assemble_system(problem=problem)
    with torch.no_grad():
        theta, violation = solve_bounded_quadratic(
            matrix=matrix[None],
            rhs=rhs[None],
            constrained=problem.layout.constrained,
            tolerance=SOLVER_TOLERANCE,
        )
    return theta[0], float(violation.max())


def load_theta(*, model: PooledDemandModel, layout: VpLayout, theta: torch.Tensor) -> None:
    """Write the linear coefficients into the model's parameters, on the model's own scales.

    The slopes and the capacity increments go through the softplus pre-image, so the model's
    softplus gives the coefficient back (to `SOFTPLUS_FLOOR` at a bound of zero). The last layer's
    bias is set to zero, because the centred columns carry no constant. With `frozen_shared` only
    the substations' own blocks are written.

    Args:
        model: The pooled model.
        layout: The columns.
        theta: The coefficients, shape (columns,).
    """
    with torch.no_grad():
        if not layout.frozen_shared:
            _load_shared(model=model, layout=layout, theta=theta)
        for site in layout.sites:
            start = layout.local_start[site]
            block = layout.local[site]
            model.level[site] = theta[start + block["level"].start]
            deviation = theta[start + block["deviation"].start : start + block["deviation"].stop]
            model.deviation[site] = deviation
            slope = theta[start + block["slope"].start : start + block["slope"].stop]
            model.slope_raw[site] = inverse_softplus(value=slope)
            years = layout.capacity_years[site]
            if years:
                increments = theta[start + block["capacity"].start : start + block["capacity"].stop]
                model.capacity_raw[site, list(years)] = inverse_softplus(value=increments)


def _load_shared(*, model: PooledDemandModel, layout: VpLayout, theta: torch.Tensor) -> None:
    """Write the shared coefficients and the population means into the model."""
    shared = layout.shared
    if layout.n_shapes:
        model.shapes.copy_(theta[shared["shapes"]].reshape(layout.n_shapes, N_PROFILE))
    model.holiday.copy_(theta[shared["holiday"]].reshape(N_PERIODS + 1, N_BANDS))
    model.tent_coef.copy_(theta[shared["tent"]].reshape(2, N_BANDS))
    model.harmonic_coef.copy_(theta[shared["harmonics"]].reshape(4, N_BANDS))
    model.lighting_coef.copy_(theta[shared["lighting"]])
    last = _last_layer(model=model)
    if last is not None:
        last.weight.copy_(theta[shared["network"]].reshape(1, -1))
        last.bias.zero_()
    families: dict[MeanType, str] = {
        "level": "level",
        "slope": "slope_raw",
        "first": "capacity_first",
        "growth": "capacity_raw",
    }
    for name, column in layout.means.items():
        value = theta[column]
        model.population_mean[families[name]].copy_(
            value if name == "level" else inverse_softplus(value=value)
        )


def _evaluate(
    *, problem: VpProblem, phi: list[nn.Parameter]
) -> tuple[float, list[torch.Tensor], float]:
    """Solve `theta*` at the current `phi`, and return the objective per row and its gradient.

    The objective is evaluated at `(phi, theta*)` with `theta*` held constant, and the centring
    means are recomputed inside the autograd graph, so the gradient is the envelope gradient
    `dJ/dphi` of the projected objective. The last value is the solver's KKT violation.
    """
    theta, violation = solve_theta(problem=problem)
    load_theta(model=problem.model, layout=problem.layout, theta=theta)
    objective = (
        vp_objective(model=problem.model, batch=problem.batch, settings=problem.settings)
        / problem.batch.n_rows
    )
    gradients = torch.autograd.grad(objective, phi, allow_unused=True)
    filled = [torch.zeros_like(p) if g is None else g for p, g in zip(phi, gradients, strict=True)]
    return float(objective.detach()), filled, violation


@dataclass
class _BestPoint:
    """The lowest objective per row seen during a fit, and the `phi` that gave it."""

    objective: float
    phi: list[torch.Tensor]


def _run_lbfgs(
    *,
    phi: list[nn.Parameter],
    evaluate: Callable[[], tuple[float, list[torch.Tensor]]],
    max_evaluations: int,
) -> tuple[list[float], _BestPoint, bool]:
    """Run L-BFGS on `phi`.

    Args:
        phi: The parameters to move.
        evaluate: Gives the objective per row and its gradient on `phi` at the current values.
        max_evaluations: The most evaluations, line search included.

    Returns:
        The objective per row at every evaluation, the best point, and whether the best objective
        stopped falling by a relative `RELATIVE_FALL` over `FALL_WINDOW` evaluations.
    """
    history: list[float] = []
    best = _BestPoint(objective=float("inf"), phi=[p.detach().clone() for p in phi])

    def closure() -> torch.Tensor:
        objective, gradients = evaluate()
        for parameter, gradient in zip(phi, gradients, strict=True):
            parameter.grad = gradient
        history.append(objective)
        if objective < best.objective:
            best.objective = objective
            best.phi = [p.detach().clone() for p in phi]
        if len(history) > FALL_WINDOW:
            reference = min(history[: len(history) - FALL_WINDOW])
            if reference - best.objective <= RELATIVE_FALL * abs(reference):
                raise _StopFitError
        # L-BFGS reads the loss with `float`, so a float32 tensor would round it to 7 digits.
        return torch.tensor(objective, dtype=torch.float64)

    optimiser = torch.optim.LBFGS(
        phi,
        lr=1.0,
        max_iter=max_evaluations,
        max_eval=max_evaluations,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        history_size=HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    try:
        optimiser.step(closure)
    except _StopFitError:
        return history, best, True
    return history, best, False


def _restore_phi(*, phi: list[nn.Parameter], values: list[torch.Tensor]) -> None:
    """Copy saved values back into the non-linear parameters."""
    with torch.no_grad():
        for parameter, value in zip(phi, values, strict=True):
            parameter.copy_(value)


def _adam_on_phi(
    *,
    phi: list[nn.Parameter],
    evaluate: Callable[[], tuple[float, list[torch.Tensor]]],
    steps: int,
) -> float:
    """Run Adam on `phi`, solving exactly each step; keep the best `phi`, return its objective."""
    optimiser = torch.optim.Adam(phi, lr=ADAM_RATE)
    best_objective, _ = evaluate()
    best_phi = [p.detach().clone() for p in phi]
    for _ in range(steps):
        objective, gradients = evaluate()
        if objective < best_objective:
            best_objective = objective
            best_phi = [p.detach().clone() for p in phi]
        for parameter, gradient in zip(phi, gradients, strict=True):
            parameter.grad = gradient
        optimiser.step()
    final, _ = evaluate()
    if final < best_objective:
        best_objective = final
        best_phi = [p.detach().clone() for p in phi]
    _restore_phi(phi=phi, values=best_phi)
    return best_objective


def _check_mva_demand(*, model: PooledDemandModel, batch: PooledBatch) -> None:
    """Raise if a modelled MVA demand is not positive, where the linear operator would be wrong.

    Args:
        model: The fitted pooled model, with its centring means finalised.
        batch: The training rows.

    Raises:
        ValueError: If any MVA row's modelled demand is zero or negative.
    """
    if not bool(batch.is_apparent.any()):
        return
    was_training = model.training
    model.eval()
    with torch.no_grad():
        demand = model.demand(batch=batch)[batch.is_apparent]
    model.train(was_training)
    if float(demand.min()) <= 0.0:
        msg = (
            f"an MVA row has modelled demand {float(demand.min()):.3g} <= 0: the apparent-power "
            "operator reads |P|, which is not the linear operator the fit assumes"
        )
        raise ValueError(msg)


def fit_pooled_vp(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    settings: VpSettings,
    max_evaluations: int = MAX_EVALUATIONS,
    adam_steps: int = ADAM_STEPS,
    warm_start_steps: int = 0,
) -> VpFitResult:
    """Fit the pooled model by variable projection.

    The non-linear parameters `phi` start at the model's current values, so the caller sets a
    start by editing the model (a different seed, or thresholds at 17 degrees). L-BFGS moves `phi`
    (strong Wolfe line search, history 20). Each evaluation builds the Gram matrix, solves
    `theta*(phi)` with the bounded convex solver, and back-propagates the objective at `theta*` to
    `phi`. L-BFGS ends when the best objective has not fallen by a relative 1e-7 over 10
    evaluations (converged) or after `max_evaluations` evaluations, line search included (not
    converged). Only a failed convergence runs Adam on `phi` for `adam_steps` steps, with an exact
    solve at each step; the result keeps the lower objective and reports both.

    Training uses only `batch`: the centring means, the population means, and the starting
    levels come from its rows (`finalise_centres`, `start_from_batch`). The caller checks that no
    row of `batch` ends after the training cut (`build_pooled_batch` does).

    Args:
        model: The pooled model with a MAP or decay prior and no variational posterior; it is
            moved to the batch's device and dtype.
        batch: The training rows, sorted by substation, in float64.
        settings: The noise standard deviation and the prior scales.
        max_evaluations: The most L-BFGS objective evaluations.
        adam_steps: The Adam fallback's steps.
        warm_start_steps: Steps of Adam on every parameter of the model's variable-projection
            objective before the first exact solve (`fit_pooled_adam_on_vp_objective`), which
            moves the non-linear parameters cheaply towards a good region; 0 gives a cold start.
            The model must then carry a MAP prior (`DpConfig.population_scales`) so the
            population means train.

    Returns:
        What the fit reports. The model holds the fitted parameters and the final centring means.

    Raises:
        ValueError: If the batch is not float64, or a modelled MVA demand is not positive.
    """
    if batch.y.dtype != torch.float64:
        msg = "the variable-projection fit needs float64: the Gram matrix is badly conditioned"
        raise ValueError(msg)
    device = batch.y.device
    model.to(device=device, dtype=batch.y.dtype)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    freeze_operator_for_vp(model=model)
    warm_start_objective: float | None = None
    if warm_start_steps:
        fit_pooled_adam_on_vp_objective(
            model=model,
            batch=batch,
            settings=settings,
            steps=warm_start_steps,
            learning_rate=WARM_START_RATE,
            final_rate_fraction=0.1,
        )
        with torch.no_grad():
            warm_start_objective = float(vp_objective(model=model, batch=batch, settings=settings))
    problem = make_vp_problem(model=model, batch=batch, settings=settings)
    phi = phi_parameters(model=model)

    def evaluate() -> tuple[float, list[torch.Tensor]]:
        objective, gradients, _ = _evaluate(problem=problem, phi=phi)
        return objective, gradients

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with trainable_only(model=model, parameters=phi):
        history, best, converged = _run_lbfgs(
            phi=phi, evaluate=evaluate, max_evaluations=max_evaluations
        )
        _restore_phi(phi=phi, values=best.phi)
        adam_objective: float | None = None
        if not converged and adam_steps > 0:
            adam_objective = _adam_on_phi(phi=phi, evaluate=evaluate, steps=adam_steps)
        _, gradients, violation = _evaluate(problem=problem, phi=phi)
    model.finalise_centres(batch=batch)
    _check_mva_demand(model=model, batch=batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        final = float(vp_objective(model=model, batch=batch, settings=settings))
    n_rows = batch.n_rows
    return VpFitResult(
        objective=final,
        objective_per_row=final / n_rows,
        lbfgs_objective=best.objective * n_rows,
        adam_objective=None if adam_objective is None else adam_objective * n_rows,
        warm_start_objective=warm_start_objective,
        evaluations=len(history),
        converged=converged,
        gradient_max_abs=max(float(g.abs().max()) for g in gradients),
        kkt_violation=violation,
        seconds=seconds,
        peak_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        history=history,
    )


def vp_site_fitter(
    *, settings: VpSettings, max_evaluations: int = MAX_EVALUATIONS
) -> Callable[[PooledDemandModel, PooledBatch, int], float]:
    """Return a fitter for `studies.dp_onboarding.onboard_site` that solves the site exactly.

    The fitter freezes every shared coefficient, every other substation, and the population means
    at the model's values. It moves only the new substation's heating threshold and style loadings
    by L-BFGS, solving its level, deviation, heating slopes, and capacity increments exactly at
    each step, with the population means as fixed prior centres. The shared calendar terms are
    centred on the substation's own rows, as in the gradient onboarding. A fit that does not meet
    the convergence test keeps its best point.

    Args:
        settings: The noise standard deviation and the prior scales.
        max_evaluations: The most objective evaluations, line search included.

    Returns:
        A function of the model, the substation's rows, and its index that fits the substation in
        place and returns the final loss per row.
    """

    def fitter(model: PooledDemandModel, batch: PooledBatch, site: int) -> float:
        freeze_operator_for_vp(model=model)
        problem = make_vp_problem(model=model, batch=batch, settings=settings, frozen_shared=True)
        owned: list[nn.Parameter] = [model.threshold]
        if model.config.shared_shapes:
            owned.append(model.site_logits)
        coordinates = nn.Parameter(
            torch.cat(
                [model.threshold[site : site + 1], *_logits_of(model=model, site=site)]
            ).detach()
        )

        def evaluate() -> tuple[float, list[torch.Tensor]]:
            with torch.no_grad():
                model.threshold[site] = coordinates[0]
                if model.config.shared_shapes:
                    model.site_logits[site] = coordinates[1:]
            objective, gradients, _ = _evaluate(problem=problem, phi=owned)
            rows = [gradients[0][site : site + 1]]
            if model.config.shared_shapes:
                rows.append(gradients[1][site])
            return objective, [torch.cat(rows)]

        with trainable_only(model=model, parameters=owned):
            _, best, _ = _run_lbfgs(
                phi=[coordinates], evaluate=evaluate, max_evaluations=max_evaluations
            )
            _restore_phi(phi=[coordinates], values=best.phi)
            final, _ = evaluate()
        return final

    return fitter


def _logits_of(*, model: PooledDemandModel, site: int) -> list[torch.Tensor]:
    """Return the substation's style loadings as a list of one tensor, or none without shapes."""
    return [model.site_logits[site]] if model.config.shared_shapes else []
