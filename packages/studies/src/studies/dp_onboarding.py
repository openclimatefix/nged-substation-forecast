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
from studies.dp_gam import MIN_SPREAD_GROUP_ROWS, ResidualSpread
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


def _check_onboarding_inputs(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    label: str,
    cache: FeatureCache,
    history_end: np.datetime64,
    forecast_season_year: int,
) -> tuple[int, int]:
    """Raise unless the model, the rows, and the dates can be onboarded; return site and year."""
    if model.config.site_prior == "decay":
        msg = "onboarding needs a population prior: fit with the map or variational site prior"
        raise ValueError(msg)
    site = batch.site_labels.index(label)
    if bool(model.site_active[site] > 0):
        msg = f"{label} had rows in the fit: onboard a substation that was held out of it"
        raise ValueError(msg)
    if batch.n_season_years != model.n_season_years or len(batch.site_labels) != model.n_sites:
        msg = "the batch's sites and season-years differ from the model's"
        raise ValueError(msg)
    if batch.n_rows and not bool((batch.site == site).all()):
        msg = f"the onboarding rows must all belong to {label}"
        raise ValueError(msg)
    if batch.n_rows and cache.end_times[batch.row.cpu().numpy()].max() > history_end:
        msg = f"an onboarding row of {label} ends after its history end {history_end}"
        raise ValueError(msg)
    n_years = model.n_season_years
    year_index = forecast_season_year - batch.first_season_year
    if not 0 <= year_index < n_years:
        msg = f"season-year {forecast_season_year} is outside the capacity table"
        raise ValueError(msg)
    return site, year_index


def onboard_site(
    *,
    model: PooledDemandModel,
    batch: PooledBatch,
    label: str,
    cache: FeatureCache,
    history_end: np.datetime64,
    forecast_season_year: int,
    steps: int,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Fit a new substation's site-level parameters into a fitted pooled model.

    The shared modules, the population means and scales, and every other substation's parameters
    stay as they are. The new substation starts at the population mean and fits only its own
    site-level parameters on `batch`, its rows, which may be empty (zero weeks of history). Its
    capacity has an increment in every season-year its rows touch and in `forecast_season_year`:
    the first is drawn around the population's first-year increment, the rest around the growth
    increment. A site with no history of its own therefore holds the population's capacity.

    Args:
        model: A fitted `map` or `variational` pooled model, in which `label` held no rows.
        batch: The new substation's rows only, with the full model's site and year layout.
        label: The new substation's label.
        cache: The feature cache the batch was built from, which dates the rows.
        history_end: The substation's history ends here; no row may end after it.
        forecast_season_year: The season-year of the forecast.
        steps: Optimiser steps.
        learning_rate: Adam's learning rate.
        seed: Seeds the posterior draws.

    Returns:
        The rows used and the final loss, for the report.

    Raises:
        ValueError: If the model's site prior is `decay`, the substation was not held out of the
            fit, the batch's layout differs from the model's, `batch` holds another substation,
            or a row ends after `history_end`.
    """
    site, year_index = _check_onboarding_inputs(
        model=model,
        batch=batch,
        label=label,
        cache=cache,
        history_end=history_end,
        forecast_season_year=forecast_season_year,
    )
    n_years = model.n_season_years
    saved_trained = model.trained.clone()
    saved_centres = model.centre_means.clone()
    saved_flags = {name: p.requires_grad for name, p in model.named_parameters()}
    device = batch.y.device
    model.to(device=device, dtype=batch.y.dtype)
    touched = torch.bincount(batch.season_slot, minlength=model.n_sites * n_years)
    touched = touched.reshape(model.n_sites, n_years)[site] > 0
    with torch.no_grad():
        model.trained[site] = touched.to(model.trained.dtype)
        model.trained[site, year_index] = 1.0
        model.site_active[site] = 1.0
        model.solar_sites[site] = 1.0
    _reset_to_population(model=model, site=site)
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
    """Set one substation's site-level parameters to the population mean (and prior spread).

    The capacity row takes the population's first-year increment at the substation's first
    trained season-year and the growth increment at its other trained season-years.
    """
    with torch.no_grad():
        first, _ = model.capacity_masks()
        for name in SITE_FAMILIES:
            values: torch.Tensor = getattr(model, name)
            if name == "capacity_raw":
                values[site] = model.population_mean["capacity_raw"]
                values[site] = torch.where(
                    first[site] > 0, model.population_mean["capacity_first"], values[site]
                )
            elif name in model.population_mean:
                values[site] = model.population_mean[name]
            else:
                values[site] = 0.0
            if model.config.variational:
                scale_name = "capacity_raw" if name == "capacity_raw" else name
                model.posterior_log_sd[name][site] = model.population_log_scale[scale_name]


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
        fit_excluded: The substations that had no rows in the fit that made the residuals.
        weeks_of_history: The weeks of history the held-out substation had in that fit's
            onboarding, or `None` when the fit is not an onboarding arm.
    """

    site: np.ndarray
    residual: np.ndarray
    band: np.ndarray
    quarter: np.ndarray
    fit_excluded: tuple[str, ...] = ()
    weeks_of_history: int | None = None


def pooled_residuals(
    *, model: PooledDemandModel, batch: PooledBatch, weeks_of_history: int | None = None
) -> PooledResiduals:
    """Return the residuals of `model` on `batch`, the rows of a replica year.

    Args:
        model: The replica fit.
        batch: The replica year's rows.
        weeks_of_history: The weeks of history of the onboarding arm, if the model was onboarded.

    Returns:
        The residuals, tagged with the substations the fit excluded and the weeks of history.
    """
    predicted = predict_pooled(model=model, batch=batch)
    labels = np.array(batch.site_labels)[batch.site.cpu().numpy()]
    excluded = tuple(
        label
        for label, active in zip(batch.site_labels, model.site_active.tolist(), strict=True)
        if active == 0
    )
    return PooledResiduals(
        site=labels,
        residual=batch.y.detach().cpu().numpy() - predicted,
        band=batch.band.cpu().numpy(),
        quarter=batch.quarter.cpu().numpy(),
        fit_excluded=excluded,
        weeks_of_history=weeks_of_history,
    )


def cross_fitted_spread(
    *, residuals: PooledResiduals, held_out: str, weeks_of_history: int | None = None
) -> ResidualSpread:
    """Fit a spread on the other substations' residuals from a fit that never saw `held_out`.

    The residuals must come from a replica fit in which `held_out` had no rows, and from the
    onboarding arm with the same weeks of history, so the spread for a new substation carries
    neither its own history nor another arm's staleness. The residuals are centred per group
    first (see `_centre_by_group`), so the spread is a spread around the mean forecast and carries
    no shift.

    Args:
        residuals: The replica residuals of all substations.
        held_out: The label of the substation whose forecast the spread is for.
        weeks_of_history: The weeks of history of the arm the spread is for.

    Returns:
        The spread, with the quantile levels of `studies.dp_gam.QUANTILE_LEVELS`.

    Raises:
        ValueError: If the replica fit saw `held_out`, the weeks of history differ, or no other
            substation has residuals.
    """
    if held_out not in residuals.fit_excluded:
        msg = f"the replica fit saw {held_out}: its residuals cannot make {held_out}'s spread"
        raise ValueError(msg)
    if residuals.weeks_of_history != weeks_of_history:
        msg = (
            f"the residuals are for {residuals.weeks_of_history} weeks of history, "
            f"not {weeks_of_history}"
        )
        raise ValueError(msg)
    others = residuals.site != held_out
    if not others.any():
        msg = f"no residuals from a substation other than {held_out}"
        raise ValueError(msg)
    band, quarter = residuals.band[others], residuals.quarter[others]
    centred = _centre_by_group(residual=residuals.residual[others], band=band, quarter=quarter)
    return ResidualSpread.fit(residuals=centred, band=band, quarter=quarter, centre=False)


def _centre_by_group(*, residual: np.ndarray, band: np.ndarray, quarter: np.ndarray) -> np.ndarray:
    """Subtract the mean residual of each (band, quarter) group, so the spread has mean zero.

    The forecast is a mean forecast, so a group's mean residual in the replica year is the replica
    year's bias, not a spread. Adding raw residual quantiles to the scored year's forecast would
    carry that bias into the scored year as a shift. A group with fewer than
    `MIN_SPREAD_GROUP_ROWS` residuals uses its band's mean, and a band with fewer uses the mean of
    all residuals.
    """
    centred = residual - residual.mean()
    for b in np.unique(band):
        in_band = band == b
        band_mean = (
            residual[in_band].mean() if in_band.sum() >= MIN_SPREAD_GROUP_ROWS else residual.mean()
        )
        centred[in_band] = residual[in_band] - band_mean
        for q in np.unique(quarter):
            group = in_band & (quarter == q)
            if group.sum() >= MIN_SPREAD_GROUP_ROWS:
                centred[group] = residual[group] - residual[group].mean()
    return centred


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
