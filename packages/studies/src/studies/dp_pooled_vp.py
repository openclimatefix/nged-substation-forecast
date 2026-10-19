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
`tan(phi) = q0 = 0`, which makes it linear.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Final

import numpy as np
import torch
from torch import nn

from studies.dp_calendar import HALF_HOURS_PER_DAY, HEATING_THRESHOLD_C
from studies.dp_gam import HEATING_SCALE_C, N_BANDS, solve_bounded_quadratic
from studies.dp_pooled import (
    N_PERIODS,
    N_PROFILE,
    THRESHOLD_PRIOR_SCALE_C,
    PooledBatch,
    PooledDemandModel,
    PooledFitResult,
    fit_pooled,
)

SHAPE_DECAY: Final[float] = 1e-4
"""Weight decay on the shared daily shapes, in nats per unit squared."""
HOLIDAY_DECAY: Final[float] = 1e-3
"""Weight decay on the holiday coefficients."""
NETWORK_DECAY: Final[float] = 1e-5
"""Weight decay on every parameter of the interaction network."""
CALENDAR_RIDGE: Final[float] = 1e-6
"""A ridge on the tent, harmonic, and lighting coefficients, so that each linear column has one.

The gradient fits put no penalty on these coefficients. The ridge makes the projected problem
strictly convex and is small enough (in nats) to leave the fit unchanged.
"""
SOFTPLUS_FLOOR: Final[float] = 1e-12
"""A coefficient below this maps to the softplus pre-image of this value (the bound at zero)."""
MAX_LINE_SEARCH_EVALUATIONS: Final[int] = 25
"""The most objective evaluations one L-BFGS step may spend."""
HISTORY_SIZE: Final[int] = 20
"""The number of curvature pairs L-BFGS keeps."""
MAX_EVALUATIONS: Final[int] = 300
"""The most objective evaluations of one L-BFGS fit, line search included."""
FALL_WINDOW: Final[int] = 10
"""L-BFGS stops when the best objective fell under `RELATIVE_FALL` over this many evaluations."""
RELATIVE_FALL: Final[float] = 1e-7
"""The relative fall in the best objective that counts as progress."""
SOLVER_TOLERANCE: Final[float] = 1e-10
"""The bounded solver's optimality tolerance, relative to the largest entry of the right-hand side.

The solver's default of 1e-7 leaves coefficients of weakly determined columns off by about 1e-3,
which feeds an error of the same order into the gradient on `phi`.
"""
ADAM_STEPS: Final[int] = 1200
"""Adam's steps on `phi`, with an exact solve each step, when L-BFGS fails its convergence test."""
ADAM_RATE: Final[float] = 0.02
"""Adam's learning rate for the fallback."""


@dataclass(frozen=True)
class VpSettings:
    """The fixed hyperparameters of the variable-projection objective.

    The population scales are the MAP scales of experiment 105's P-map fit, written on the natural
    scale: the pre-softplus scales of the slope and growth (both 1.0) times the softplus slope at
    P-map's population means (0.985 and 0.946), and the first increment's 8.0 times 1.00.

    Attributes:
        noise_sd: The standard deviation of the Gaussian noise, in points of effective capacity,
            fixed within a fit.
        level: The scale of the substations' levels around their population mean.
        site_logits: The scale of the style loadings around zero.
        deviation: The scale of the per-site deviation from the shared shapes.
        threshold: The scale of the heating thresholds around their population mean, and of the
            population mean's prior around 15.5 degrees.
        slope: The scale of the heating slopes around their population mean.
        growth: The scale of the capacity increments after a substation's first trained year.
        first: The scale of the first trained season-year's capacity increment.
    """

    noise_sd: float
    level: float = 5.0
    site_logits: float = 1.0
    deviation: float = 1.0
    threshold: float = THRESHOLD_PRIOR_SCALE_C
    slope: float = 0.985
    growth: float = 0.946
    first: float = 8.0


@dataclass(frozen=True)
class VpLayout:
    """The columns of the linear problem, in the order of the coefficient vector `theta`.

    Attributes:
        sites: The substations that have training rows, ascending.
        n_shapes: The number of shared daily shapes `K`.
        shared: The slice of each shared block: `shapes`, `holiday`, `tent`, `harmonics`,
            `lighting`, and `network` (the last layer's weights).
        n_shared: The number of shared columns.
        means: The column of each population mean that has site columns: any of `level`, `slope`,
            `first`, and `growth`.
        local_start: The first column of each substation's own block.
        local: Each substation's slices within its block, relative to its start: `level`,
            `deviation`, `slope`, and `capacity`.
        capacity_years: Each substation's season-years that have a capacity increment column;
            empty when the capacity is pinned or the substation has no solar rows.
        n_columns: The number of columns.
        constrained: True for the columns that must be non-negative.
    """

    sites: tuple[int, ...]
    n_shapes: int
    shared: dict[str, slice]
    n_shared: int
    means: dict[str, int]
    local_start: dict[int, int]
    local: dict[int, dict[str, slice]]
    capacity_years: dict[int, tuple[int, ...]]
    n_columns: int
    constrained: torch.Tensor


@dataclass(frozen=True)
class SiteDesign:
    """One substation's part of the linear problem, `prediction = shared @ w + local @ v + offset`.

    Attributes:
        shared: The columns of the shared coefficients, shape (rows, n_shared). The calendar,
            lighting, and network columns are centred per season-year.
        local: The substation's own columns, shape (rows, width of its block).
        offset: The part of the prediction that does not depend on `theta` (a pinned capacity's
            solar output, negative), shape (rows,).
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
        lbfgs_objective: The objective at the end of L-BFGS.
        adam_objective: The objective after the Adam fallback, or `None` when it did not run.
        evaluations: The number of objective evaluations L-BFGS spent, line search included.
        converged: True if L-BFGS stopped on its tolerance and not on the evaluation cap.
        gradient_norm: The largest absolute entry of the final objective-per-row gradient on
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
    evaluations: int
    converged: bool
    gradient_norm: float
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


def vp_objective(
    *, model: PooledDemandModel, batch: PooledBatch, settings: VpSettings
) -> torch.Tensor:
    """Return the deterministic MAP objective of `batch` in nats, for any parameter set.

    The data term is the squared error over twice the noise variance. The penalty is the weight
    decay of `PooledDemandModel.penalty` plus a small ridge on the tent, harmonic, and lighting
    coefficients. The site priors are Gaussian on the natural scale around learned population
    means (the heating slope and the capacity increments are the softplus outputs themselves),
    with the heating threshold's population mean centred on 15.5 degrees. The MVA operator must
    be fixed at zero reactive power (`freeze_operator_for_vp`).

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
    model.train()
    try:
        residual = model(batch) - batch.y
    finally:
        model.train(was_training)
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
    term = (
        term
        + (
            _tie(values=model.level, centre=model.population_mean["level"], scale=settings.level)
            * active
        ).sum()
    )
    term = (
        term
        + (
            _tie(values=model.threshold, centre=mean_threshold, scale=settings.threshold) * active
        ).sum()
    )
    term = term + (_tie(values=model.deviation, centre=0.0, scale=settings.deviation) * both).sum()
    if model.config.shared_shapes:
        term = (
            term
            + (_tie(values=model.site_logits, centre=0.0, scale=settings.site_logits) * both).sum()
        )
    slope = nn.functional.softplus(model.slope_raw)
    slope_mean = nn.functional.softplus(model.population_mean["slope_raw"])
    term = term + (_tie(values=slope, centre=slope_mean, scale=settings.slope) * both).sum()
    if model.capacity_override is None:
        increment = nn.functional.softplus(model.capacity_raw)
        first, growth = model.capacity_masks()
        first_mean = nn.functional.softplus(model.population_mean["capacity_first"])
        growth_mean = nn.functional.softplus(model.population_mean["capacity_raw"])
        term = (
            term + (_tie(values=increment, centre=first_mean, scale=settings.first) * first).sum()
        )
        term = (
            term
            + (_tie(values=increment, centre=growth_mean, scale=settings.growth) * growth).sum()
        )
    return term


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


def make_vp_layout(*, model: PooledDemandModel, batch: PooledBatch) -> VpLayout:
    """Lay out the linear problem's columns for the substations that have rows in `batch`.

    The model's `trained` and `solar_sites` buffers must be set (`mark_trained_years`). A
    substation gets capacity columns for its trained season-years, unless it has no solar rows
    or the model pins the capacity (`capacity_override`).

    Args:
        model: The pooled model.
        batch: The rows.

    Returns:
        The layout.
    """
    n_shapes = model.config.shared_shapes
    pinned = model.capacity_override is not None
    last_layer = _last_layer(model=model)
    hidden = 0 if last_layer is None else last_layer.in_features
    sizes = {
        "shapes": n_shapes * N_PROFILE,
        "holiday": (N_PERIODS + 1) * N_BANDS,
        "tent": 2 * N_BANDS,
        "harmonics": 4 * N_BANDS,
        "lighting": 4,
        "network": hidden,
    }
    shared: dict[str, slice] = {}
    position = 0
    for name, size in sizes.items():
        shared[name] = slice(position, position + size)
        position += size
    n_shared = position
    sites = tuple(
        site
        for site, (start, stop) in enumerate(site_bounds(batch=batch, n_sites=model.n_sites))
        if stop > start
    )
    first_years: dict[int, int] = {}
    capacity_years: dict[int, tuple[int, ...]] = {}
    for site in sites:
        solar = bool(model.solar_sites[site] > 0) and not pinned
        years = (
            tuple(int(y) for y in torch.nonzero(model.trained[site] > 0).flatten()) if solar else ()
        )
        capacity_years[site] = years
        if years:
            first_years[site] = years[0]
    wanted = {"level": True, "slope": True}
    wanted["first"] = any(capacity_years[s] for s in sites)
    wanted["growth"] = any(len(capacity_years[s]) > 1 for s in sites)
    means: dict[str, int] = {}
    for name, needed in wanted.items():
        if needed:
            means[name] = position
            position += 1
    local_start: dict[int, int] = {}
    local: dict[int, dict[str, slice]] = {}
    for site in sites:
        local_start[site] = position
        widths = {
            "level": 1,
            "deviation": N_PROFILE,
            "slope": N_BANDS,
            "capacity": len(capacity_years[site]),
        }
        slices: dict[str, slice] = {}
        offset = 0
        for name, width in widths.items():
            slices[name] = slice(offset, offset + width)
            offset += width
        local[site] = slices
        position += offset
    constrained = torch.zeros(position, dtype=torch.bool)
    for site in sites:
        start = local_start[site]
        for name in ("slope", "capacity"):
            block = local[site][name]
            constrained[start + block.start : start + block.stop] = True
    return VpLayout(
        sites=sites,
        n_shapes=n_shapes,
        shared=shared,
        n_shared=n_shared,
        means=means,
        local_start=local_start,
        local=local,
        capacity_years=capacity_years,
        n_columns=position,
        constrained=constrained,
    )


def _centre_per_slot(*, values: torch.Tensor, slot: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Subtract each column's mean over the rows of each slot (a substation and season-year)."""
    counts = torch.bincount(slot, minlength=n_slots).clamp(min=1).to(values.dtype)
    sums = torch.zeros((n_slots, values.shape[1]), dtype=values.dtype, device=values.device)
    means = sums.index_add(0, slot, values) / counts[:, None]
    return values - means[slot]


def _one_hot(*, index: torch.Tensor, width: int, dtype: torch.dtype) -> torch.Tensor:
    """Return the (rows, width) one-hot matrix of `index`."""
    out = torch.zeros((index.shape[0], width), dtype=dtype, device=index.device)
    out[torch.arange(index.shape[0], device=index.device), index] = 1.0
    return out


def linear_design(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout, site: int
) -> SiteDesign:
    """Build the columns of one substation's rows for the model's current `phi`.

    The prediction `shared @ w + local @ v + offset` equals `PooledDemandModel.forward` when `w`
    and `v` hold the model's linear parameters on the natural scale (the heating slope and the
    capacity increments as softplus outputs) and the MVA operator has zero reactive power.

    Args:
        model: The pooled model, whose non-linear parameters `phi` set the columns.
        batch: The rows of the whole fit, sorted by substation.
        layout: The columns.
        site: The substation.

    Returns:
        The substation's columns and offset.
    """
    start, stop = site_bounds(batch=batch, n_sites=model.n_sites)[site]
    rows = _slice_batch(batch=batch, start=start, stop=stop)
    n = rows.n_rows
    dtype = rows.y.dtype
    profile = rows.day_type * HALF_HOURS_PER_DAY + rows.half_hour
    band_hot = _one_hot(index=rows.band, width=N_BANDS, dtype=dtype)
    inertia = torch.softmax(model.inertia_logits, dim=0)
    mixed = rows.smoothed_temperature @ inertia
    parts: list[torch.Tensor] = []
    if layout.n_shapes:
        mix = torch.softmax(model.site_logits[site], dim=0)
        index = (
            torch.arange(layout.n_shapes, device=profile.device)[None, :] * N_PROFILE
            + profile[:, None]
        )
        shapes = torch.zeros((n, layout.n_shapes * N_PROFILE), dtype=dtype, device=profile.device)
        shapes.scatter_(1, index, mix[None, :].expand(n, -1).to(dtype))
        parts.append(shapes)
    holiday = _one_hot(
        index=rows.holiday * N_BANDS + rows.band, width=(N_PERIODS + 1) * N_BANDS, dtype=dtype
    )
    calendar = [
        holiday,
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
    gate = rows.solar_gate.to(dtype)
    blocks = [
        torch.ones((n, 1), dtype=dtype, device=profile.device),
        _one_hot(index=profile, width=N_PROFILE, dtype=dtype),
        hinge[:, None] * band_hot,
    ]
    years = layout.capacity_years[site]
    if years:
        after = slot[:, None] >= torch.tensor(years, device=slot.device)[None, :]
        blocks.append(-(gate * rows.pv_shape)[:, None] * after.to(dtype))
    offset = torch.zeros(n, dtype=dtype, device=profile.device)
    if model.capacity_override is not None:
        pinned = torch.nan_to_num(model.capacity_override.to(dtype))
        offset = -gate * rows.pv_shape * pinned.reshape(-1)[rows.season_slot]
    return SiteDesign(shared=shared, local=torch.cat(blocks, dim=1), offset=offset)


def penalty_system(
    *, layout: VpLayout, settings: VpSettings, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Return the matrix `R` of the quadratic prior, so that the prior in nats is `theta^T R theta`.

    The population means are columns of `theta`, and a site-level family is tied to its mean by
    `(value - mean)^2 / (2 scale^2)`.

    Args:
        layout: The columns.
        settings: The prior scales.
        dtype: The floating-point type.
        device: Where the matrix lives.

    Returns:
        The symmetric matrix, shape (columns, columns).
    """
    p = layout.n_columns
    matrix = torch.zeros((p, p), dtype=dtype, device=device)
    diagonal = torch.zeros(p, dtype=dtype, device=device)
    decay = {
        "shapes": SHAPE_DECAY,
        "holiday": HOLIDAY_DECAY,
        "tent": CALENDAR_RIDGE,
        "harmonics": CALENDAR_RIDGE,
        "lighting": CALENDAR_RIDGE,
        "network": NETWORK_DECAY,
    }
    for name, weight in decay.items():
        diagonal[layout.shared[name]] = weight

    def tie(*, columns: list[int], mean: str | None, scale: float) -> None:
        weight = 1.0 / (2.0 * scale**2)
        index = torch.tensor(columns, dtype=torch.long, device=device)
        if index.numel() == 0:
            return
        diagonal[index] += weight
        if mean is not None:
            centre = layout.means[mean]
            diagonal[centre] += weight * index.numel()
            matrix[index, centre] -= weight
            matrix[centre, index] -= weight

    for site in layout.sites:
        start = layout.local_start[site]
        block = layout.local[site]
        tie(columns=[start + block["level"].start], mean="level", scale=settings.level)
        tie(
            columns=list(range(start + block["deviation"].start, start + block["deviation"].stop)),
            mean=None,
            scale=settings.deviation,
        )
        tie(
            columns=list(range(start + block["slope"].start, start + block["slope"].stop)),
            mean="slope",
            scale=settings.slope,
        )
        capacity = list(range(start + block["capacity"].start, start + block["capacity"].stop))
        tie(columns=capacity[:1], mean="first", scale=settings.first)
        tie(columns=capacity[1:], mean="growth", scale=settings.growth)
    return matrix + torch.diag(diagonal)


@torch.no_grad()
def assemble_system(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout, settings: VpSettings
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `A` and `b` of the problem: minimise `theta^T A theta - 2 b^T theta`, bounds aside.

    The squared error and the prior are both multiplied by twice the noise variance, so `A` is the
    Gram matrix plus `2 sigma^2 R`. The Gram matrix is built one substation at a time, so no dense
    matrix of all the rows is formed; a substation's block touches the shared columns and its own.

    Args:
        model: The pooled model, whose current `phi` sets the columns.
        batch: The rows.
        layout: The columns.
        settings: The noise standard deviation and the prior scales.

    Returns:
        `(A, b)` of shapes (columns, columns) and (columns,).
    """
    dtype, device = batch.y.dtype, batch.y.device
    p, q = layout.n_columns, layout.n_shared
    gram = torch.zeros((p, p), dtype=dtype, device=device)
    rhs = torch.zeros(p, dtype=dtype, device=device)
    bounds = site_bounds(batch=batch, n_sites=model.n_sites)
    for site in layout.sites:
        design = linear_design(model=model, batch=batch, layout=layout, site=site)
        target = batch.y[bounds[site][0] : bounds[site][1]] - design.offset
        columns = slice(layout.local_start[site], layout.local_start[site] + design.local.shape[1])
        gram[:q, :q] += design.shared.T @ design.shared
        cross = design.shared.T @ design.local
        gram[:q, columns] = cross
        gram[columns, :q] = cross.T
        gram[columns, columns] = design.local.T @ design.local
        rhs[:q] += design.shared.T @ target
        rhs[columns] = design.local.T @ target
    penalty = penalty_system(layout=layout, settings=settings, dtype=dtype, device=device)
    return gram + 2.0 * settings.noise_sd**2 * penalty, rhs


def solve_theta(
    *, model: PooledDemandModel, batch: PooledBatch, layout: VpLayout, settings: VpSettings
) -> tuple[torch.Tensor, float]:
    """Return the exact linear coefficients for the model's current `phi`, and the KKT violation.

    Args:
        model: The pooled model.
        batch: The rows.
        layout: The columns.
        settings: The noise standard deviation and the prior scales.

    Returns:
        `theta*` of shape (columns,) and the solver's relative violation of the optimality
        conditions.
    """
    with torch.no_grad():
        matrix, rhs = assemble_system(model=model, batch=batch, layout=layout, settings=settings)
        theta, violation = solve_bounded_quadratic(
            matrix=matrix[None],
            rhs=rhs[None],
            constrained=layout.constrained,
            tolerance=SOLVER_TOLERANCE,
        )
    return theta[0], float(violation.max())


def load_theta(*, model: PooledDemandModel, layout: VpLayout, theta: torch.Tensor) -> None:
    """Write the linear coefficients into the model's parameters, on the model's own scales.

    The slopes and the capacity increments go through the softplus pre-image, so the model's
    softplus gives the coefficient back (to `SOFTPLUS_FLOOR` at a bound of zero). The last layer's
    bias is set to zero, because the centred columns carry no constant.

    Args:
        model: The pooled model.
        layout: The columns.
        theta: The coefficients, shape (columns,).
    """
    with torch.no_grad():
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
        for name, family in (
            ("level", "level"),
            ("slope", "slope_raw"),
            ("first", "capacity_first"),
            ("growth", "capacity_raw"),
        ):
            if name in layout.means:
                value = theta[layout.means[name]]
                if name == "level":
                    model.population_mean[family].copy_(value)
                else:
                    model.population_mean[family].copy_(inverse_softplus(value=value))
        for site in layout.sites:
            start = layout.local_start[site]
            block = layout.local[site]
            model.level[site] = theta[start + block["level"].start]
            model.deviation[site] = theta[
                start + block["deviation"].start : start + block["deviation"].stop
            ]
            slope = theta[start + block["slope"].start : start + block["slope"].stop]
            model.slope_raw[site] = inverse_softplus(value=slope)
            years = layout.capacity_years[site]
            if years:
                increments = theta[start + block["capacity"].start : start + block["capacity"].stop]
                model.capacity_raw[site, list(years)] = inverse_softplus(value=increments)


def _evaluate(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    layout: VpLayout,
    settings: VpSettings,
    phi: list[nn.Parameter],
) -> tuple[float, list[torch.Tensor], float]:
    """Solve `theta*` at the current `phi`, and return the objective per row and its `phi` gradient.

    The objective is evaluated at `(phi, theta*)` with `theta*` held constant, and the centring
    means are recomputed inside the autograd graph, so the gradient is the envelope gradient
    `dJ/dphi` of the projected objective.
    """
    theta, violation = solve_theta(model=model, batch=batch, layout=layout, settings=settings)
    load_theta(model=model, layout=layout, theta=theta)
    objective = vp_objective(model=model, batch=batch, settings=settings) / batch.n_rows
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
    model: PooledDemandModel,
    batch: PooledBatch,
    layout: VpLayout,
    settings: VpSettings,
    phi: list[nn.Parameter],
    max_evaluations: int,
) -> tuple[list[float], _BestPoint, bool]:
    """Run L-BFGS on `phi`.

    Returns:
        The objective per row at every evaluation, the best point, and whether the objective
        stopped falling (as against the evaluation cap or L-BFGS's own stop).
    """
    history: list[float] = []
    best = _BestPoint(objective=float("inf"), phi=[p.detach().clone() for p in phi])

    def closure() -> torch.Tensor:
        objective, gradients, _ = _evaluate(
            model=model, batch=batch, layout=layout, settings=settings, phi=phi
        )
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
        return torch.tensor(objective)

    optimiser = torch.optim.LBFGS(
        phi,
        lr=1.0,
        max_iter=max_evaluations,
        max_eval=max_evaluations,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        history_size=HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )
    try:
        optimiser.step(closure)
    except _StopFitError:
        return history, best, True
    return history, best, False


def fit_pooled_vp(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    settings: VpSettings,
    max_evaluations: int = MAX_EVALUATIONS,
    adam_steps: int = ADAM_STEPS,
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
    levels come from its rows (`finalise_centres`, `start_from_batch`).

    Args:
        model: The pooled model with a MAP or decay prior and no variational posterior, already
            on the batch's device and in float64.
        batch: The training rows, sorted by substation, in float64.
        settings: The noise standard deviation and the prior scales.
        max_evaluations: The most L-BFGS objective evaluations.
        adam_steps: The Adam fallback's steps.

    Returns:
        What the fit reports. The model holds the fitted parameters and the final centring means.

    Raises:
        ValueError: If the batch is not float64.
    """
    if batch.y.dtype != torch.float64:
        msg = "the variable-projection fit needs float64: the Gram matrix is badly conditioned"
        raise ValueError(msg)
    device = batch.y.device
    model.to(device=device, dtype=batch.y.dtype)
    model.mark_trained_years(batch=batch)
    model.start_from_batch(batch=batch)
    freeze_operator_for_vp(model=model)
    layout = make_vp_layout(model=model, batch=batch)
    phi = phi_parameters(model=model)
    saved_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in phi:
        parameter.requires_grad = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    history, best, stopped_on_tolerance = _run_lbfgs(
        model=model,
        batch=batch,
        layout=layout,
        settings=settings,
        phi=phi,
        max_evaluations=max_evaluations,
    )
    evaluations = len(history)
    # L-BFGS also ends by itself when the gradient or the step vanishes, before the cap.
    converged = stopped_on_tolerance or evaluations < max_evaluations
    _restore_phi(phi=phi, values=best.phi)
    lbfgs_objective = best.objective
    adam_objective: float | None = None
    if not converged and adam_steps > 0:
        adam_objective = _adam_on_phi(
            model=model, batch=batch, layout=layout, settings=settings, phi=phi, steps=adam_steps
        )
    _, gradients, violation = _evaluate(
        model=model, batch=batch, layout=layout, settings=settings, phi=phi
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad = saved_flags[name]
    model.finalise_centres(batch=batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        final = float(vp_objective(model=model, batch=batch, settings=settings))
    return VpFitResult(
        objective=final,
        objective_per_row=final / batch.n_rows,
        lbfgs_objective=lbfgs_objective * batch.n_rows,
        adam_objective=None if adam_objective is None else adam_objective * batch.n_rows,
        evaluations=evaluations,
        converged=converged,
        gradient_norm=max(float(g.abs().max()) for g in gradients),
        kkt_violation=violation,
        seconds=seconds,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        history=history,
    )


def _restore_phi(*, phi: list[nn.Parameter], values: list[torch.Tensor]) -> None:
    """Copy saved values back into the non-linear parameters."""
    with torch.no_grad():
        for parameter, value in zip(phi, values, strict=True):
            parameter.copy_(value)


def _adam_on_phi(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    layout: VpLayout,
    settings: VpSettings,
    phi: list[nn.Parameter],
    steps: int,
) -> float:
    """Run Adam on `phi`, solving exactly each step; keep the best `phi`, return its objective."""
    optimiser = torch.optim.Adam(phi, lr=ADAM_RATE)
    start_objective, _, _ = _evaluate(
        model=model, batch=batch, layout=layout, settings=settings, phi=phi
    )
    best_objective = start_objective
    best_phi = [p.detach().clone() for p in phi]
    for _ in range(steps):
        objective, gradients, _ = _evaluate(
            model=model, batch=batch, layout=layout, settings=settings, phi=phi
        )
        if objective < best_objective:
            best_objective = objective
            best_phi = [p.detach().clone() for p in phi]
        for parameter, gradient in zip(phi, gradients, strict=True):
            parameter.grad = gradient
        optimiser.step()
    final, _, _ = _evaluate(model=model, batch=batch, layout=layout, settings=settings, phi=phi)
    if final < best_objective:
        best_objective = final
        best_phi = [p.detach().clone() for p in phi]
    _restore_phi(phi=phi, values=best_phi)
    return best_objective


def assert_batch_before_cut(
    *, batch: PooledBatch, end_times: np.ndarray, cut_end: np.datetime64
) -> None:
    """Raise if any row of `batch` ends after `cut_end`.

    Args:
        batch: The training rows; `batch.row` indexes the feature cache's time grid.
        end_times: The cache's `end_times` (a NumPy datetime array).
        cut_end: The instant training ends, in the same dtype.

    Raises:
        ValueError: If a training row ends after the cut (lookahead).
    """
    rows = batch.row.cpu().numpy()
    latest = end_times[rows].max()
    if latest > cut_end:
        msg = f"a training row ends at {latest}, after the cut at {cut_end}: lookahead"
        raise ValueError(msg)
