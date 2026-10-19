"""Many fits of the pooled DP from one batch, onboarding a new substation, and the pooled spread.

`studies.dp_pooled` fits one pooled model. Part 3 of the differentiable-physics series needs many
fits that differ only in which rows or targets they see: a leave-one-site-out fit drops one
substation's rows, a short-history fit drops one substation's early rows, and an injected-farm fit
changes one substation's targets. This module builds those fits from one `PooledBatch`, onboards a
new substation into a fitted model without refitting the shared modules, and turns the pooled
model's residuals on a replica year into a forecast spread.

**Row selection is exact.** A fit on a subset of rows is a fit on a smaller `PooledBatch`, so a
dropped substation's rows influence nothing: not the shared network, not the population scales, not
the centring means of another substation. A substation left with no rows is inactive, and the site
prior leaves its parameters alone.

**The copies run one after another.** The full batch already fills the GPU, so stacking copies
would add memory and no speed. `fit_many` records each copy's peak memory.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from studies.dp_features import FeatureCache
from studies.dp_gam import ResidualSpread
from studies.dp_pooled import (
    SITE_FAMILIES,
    DpConfig,
    PooledBatch,
    PooledDemandModel,
    PooledFitResult,
    fit_pooled,
)

_ROW_FIELDS: tuple[str, ...] = (
    "site",
    "day_type",
    "half_hour",
    "band",
    "holiday",
    "tent",
    "harmonics",
    "smoothed_temperature",
    "lighting",
    "elevation",
    "pv_shape",
    "season_slot",
    "solar_gate",
    "is_apparent",
    "y",
    "quarter",
    "row",
)


def select_rows(*, batch: PooledBatch, keep: torch.Tensor) -> PooledBatch:
    """Return the rows of `batch` where `keep` is true, with the same site and year layout.

    Args:
        batch: The rows.
        keep: A boolean tensor with one entry per row.

    Returns:
        A new batch holding only the kept rows.
    """
    return replace(batch, **{name: getattr(batch, name)[keep] for name in _ROW_FIELDS})


def with_targets(*, batch: PooledBatch, y: torch.Tensor) -> PooledBatch:
    """Return `batch` with the readings replaced by `y` (an injected-farm fit)."""
    if y.shape != batch.y.shape:
        msg = f"targets have shape {tuple(y.shape)} but the batch has {tuple(batch.y.shape)}"
        raise ValueError(msg)
    return replace(batch, y=y)


def site_mask(*, batch: PooledBatch, label: str) -> torch.Tensor:
    """Return a boolean tensor marking the rows of one substation."""
    return batch.site == batch.site_labels.index(label)


def after_mask(*, batch: PooledBatch, cache: FeatureCache, after: np.datetime64) -> torch.Tensor:
    """Return a boolean tensor marking the rows that end after `after`."""
    ends = cache.end_times[batch.row.cpu().numpy()] > after
    return torch.as_tensor(ends, device=batch.y.device)


@dataclass(frozen=True)
class FitCopy:
    """One copy of a batched fit: which rows it sees, and optionally other targets.

    Attributes:
        name: A label for the copy.
        keep: A boolean tensor over the batch's rows, or `None` to keep every row.
        y: Replacement readings for every row of the batch, or `None` to keep the real ones.
    """

    name: str
    keep: torch.Tensor | None = None
    y: torch.Tensor | None = None


def fit_many(
    *,
    batch: PooledBatch,
    copies: Sequence[FitCopy],
    config: DpConfig,
    initial_level: torch.Tensor,
    steps: int,
    learning_rate: float = 0.05,
    final_rate_fraction: float = 1.0,
    seed: int = 0,
) -> tuple[list[PooledDemandModel], list[PooledFitResult]]:
    """Fit one independent pooled model per copy, every one from the same seed.

    Args:
        batch: The rows every copy starts from.
        copies: The copies: row subsets and replacement targets.
        config: The model configuration shared by the copies.
        initial_level: The starting level of each substation, in points.
        steps: Optimiser steps per copy.
        learning_rate: Adam's starting learning rate.
        final_rate_fraction: The fraction of the rate left at the last step (see `fit_pooled`).
        seed: Seeds each copy's initialisation and posterior draws, so a copy's result does not
            depend on the others or on their order.

    Returns:
        The fitted models and their fit reports, in the order of `copies`.
    """
    models: list[PooledDemandModel] = []
    results: list[PooledFitResult] = []
    for copy in copies:
        rows = batch if copy.y is None else with_targets(batch=batch, y=copy.y)
        if copy.keep is not None:
            rows = select_rows(batch=rows, keep=copy.keep)
        torch.manual_seed(seed)
        model = PooledDemandModel(
            config=config,
            n_sites=len(batch.site_labels),
            n_season_years=batch.n_season_years,
            initial_level=initial_level,
        )
        results.append(
            fit_pooled(
                model=model,
                batch=rows,
                steps=steps,
                learning_rate=learning_rate,
                final_rate_fraction=final_rate_fraction,
            )
        )
        models.append(model)
    return models, results


def predict_pooled(*, model: PooledDemandModel, batch: PooledBatch) -> np.ndarray:
    """Return the posterior-mean (or point) reading of each row, in points."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        reading = model(batch).detach().cpu().numpy()
    model.train(was_training)
    return reading


def _site_parameters(*, model: PooledDemandModel) -> list[torch.nn.Parameter]:
    """Return the parameters with a leading substation dimension."""
    named = [getattr(model, name) for name in SITE_FAMILIES] + [model.tan_phi, model.q0]
    if model.config.variational:
        named += [model.posterior_log_sd[name] for name in SITE_FAMILIES]
    return named


def onboard_site(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    label: str,
    forecast_season_year: int,
    steps: int,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a new substation's site-level parameters into a fitted pooled model.

    The shared modules, the population means and scales, and every other substation's parameters
    stay as they are. The new substation starts at the population mean and fits only its own
    site-level parameters on `batch`, its rows, which may be empty (zero weeks of history). Its
    capacity has one increment, in the season-year `forecast_season_year`, so a site with no
    history of its own holds the population's capacity increment.

    Args:
        model: A fitted `map` or `variational` pooled model, in which `label` held no rows.
        batch: The new substation's rows only, with the full model's site and year layout.
        label: The new substation's label.
        forecast_season_year: The season-year of the forecast, which carries the capacity.
        steps: Optimiser steps.
        learning_rate: Adam's learning rate.
        seed: Seeds the posterior draws.

    Returns:
        The rows used and the final loss, for the report.

    Raises:
        ValueError: If the model's site prior is `decay`, or `batch` holds another substation.
    """
    if model.config.site_prior == "decay":
        msg = "onboarding needs a population prior: fit with the map or variational site prior"
        raise ValueError(msg)
    site = batch.site_labels.index(label)
    if batch.n_rows and not bool((batch.site == site).all()):
        msg = f"the onboarding rows must all belong to {label}"
        raise ValueError(msg)
    n_years = model.n_season_years
    year_index = forecast_season_year - batch.first_season_year
    if not 0 <= year_index < n_years:
        msg = f"season-year {forecast_season_year} is outside the capacity table"
        raise ValueError(msg)
    saved_trained = model.trained.clone()
    saved_centres = model.centre_means.clone()
    saved_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    device = batch.y.device
    model.to(device=device, dtype=batch.y.dtype)
    _reset_to_population(model=model, site=site)
    with torch.no_grad():
        model.trained[site] = 0.0
        model.trained[site, year_index] = 1.0
        model.site_active[site] = 1.0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mask = torch.zeros(model.n_sites, device=device, dtype=batch.y.dtype)
    mask[site] = 1.0
    trainable = _site_parameters(model=model)
    handles = []
    for parameter in trainable:
        parameter.requires_grad_(True)
        handles.append(
            parameter.register_hook(lambda grad: grad * mask.reshape(-1, *([1] * (grad.dim() - 1))))
        )
    optimiser = torch.optim.Adam(trainable, lr=learning_rate)
    torch.manual_seed(seed)
    last = float("nan")
    if batch.n_rows:
        model.train()
        for _ in range(steps):
            optimiser.zero_grad(set_to_none=True)
            loss = model.loss(batch=batch)
            loss.backward()
            optimiser.step()
            last = float(loss.detach())
    for handle in handles:
        handle.remove()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(saved_flags[name])
    _restore_other_sites(
        model=model,
        batch=batch,
        site=site,
        saved_trained=saved_trained,
        saved_centres=saved_centres,
    )
    return {"rows": float(batch.n_rows), "loss": last}


def _reset_to_population(*, model: PooledDemandModel, site: int) -> None:
    """Set one substation's site-level parameters to the population mean (and prior spread)."""
    with torch.no_grad():
        for name in SITE_FAMILIES:
            values: torch.Tensor = getattr(model, name)
            if name in model.population_mean:
                values[site] = model.population_mean[name]
            else:
                values[site] = 0.0
            if model.config.variational:
                model.posterior_log_sd[name][site] = model.population_log_scale[name]


def _restore_other_sites(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    site: int,
    saved_trained: torch.Tensor,
    saved_centres: torch.Tensor,
) -> None:
    """Refresh the new substation's centring means and give every other substation its own back."""
    n_years = model.n_season_years
    with torch.no_grad():
        trained_now = model.trained.clone()
        model.trained.copy_(saved_trained)
        model.trained[site] = trained_now[site]
        if batch.n_rows:
            was_training = model.training
            model.train()
            model(batch)
            model.train(was_training)
        fresh = model.centre_means.reshape(model.n_sites, n_years).clone()
        previous = saved_centres.reshape(model.n_sites, n_years)
        counts = torch.bincount(batch.season_slot, minlength=model.n_sites * n_years)
        has_rows = counts.reshape(model.n_sites, n_years)[site] > 0
        others = [i for i in range(model.n_sites) if i != site and bool(saved_trained[i].any())]
        reference = previous[others].mean(dim=0) if others else torch.zeros_like(previous[0])
        own = fresh[site].clone()
        carry = reference.clone()
        seen = False
        last = reference[0]
        for year in range(n_years):
            if bool(has_rows[year]):
                seen, last = True, own[year]
            own[year] = last if seen else carry[year]
        combined = previous.clone()
        combined[site] = own
        model.centre_means.copy_(combined.reshape(-1))


@dataclass(frozen=True)
class PooledResiduals:
    """Residuals of a pooled replica fit, per row.

    Attributes:
        site: The substation label of each residual.
        residual: Actual minus predicted mean, in points.
        band: The hour band of each residual.
        quarter: The calendar quarter of each residual.
    """

    site: np.ndarray
    residual: np.ndarray
    band: np.ndarray
    quarter: np.ndarray


def pooled_residuals(*, model: PooledDemandModel, batch: PooledBatch) -> PooledResiduals:
    """Return the residuals of `model` on `batch`, the rows of a replica year."""
    predicted = predict_pooled(model=model, batch=batch)
    labels = np.array(batch.site_labels)[batch.site.cpu().numpy()]
    return PooledResiduals(
        site=labels,
        residual=batch.y.detach().cpu().numpy() - predicted,
        band=batch.band.cpu().numpy(),
        quarter=batch.quarter.cpu().numpy(),
    )


def cross_fitted_spread(*, residuals: PooledResiduals, held_out: str) -> ResidualSpread:
    """Fit a spread on every substation's replica residuals except the held-out one's.

    A forecast for a new substation then never uses residuals that the substation produced.

    Args:
        residuals: The replica residuals of all substations.
        held_out: The label of the substation whose forecast the spread is for.

    Returns:
        The spread, with the quantile levels of `studies.dp_gam.QUANTILE_LEVELS`.

    Raises:
        ValueError: If no other substation has residuals.
    """
    others = residuals.site != held_out
    if not others.any():
        msg = f"no residuals from a substation other than {held_out}"
        raise ValueError(msg)
    return ResidualSpread.fit(
        residuals=residuals.residual[others],
        band=residuals.band[others],
        quarter=residuals.quarter[others],
    )


__all__ = [
    "FitCopy",
    "PooledResiduals",
    "after_mask",
    "cross_fitted_spread",
    "fit_many",
    "onboard_site",
    "pooled_residuals",
    "predict_pooled",
    "select_rows",
    "site_mask",
    "with_targets",
]
