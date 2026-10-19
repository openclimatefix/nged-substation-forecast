"""The fitting recipes shared by the scripts of experiment 105.

An arm is a model configuration, the substations it trains on, a training window, and the
substations it forecasts. `arm_forecast` fits it at a cut, fits a replica of it 12 months earlier,
turns the replica's residuals into a per-substation spread centred on the mean forecast, and returns
the quantiles of every scored row. Onboarding and short-history arms build on the same pieces in
`e00105_onboard.py`.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import e00105_common as common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import MIN_SPREAD_GROUP_ROWS, QUANTILE_COLUMNS
from studies.dp_onboarding import (
    FitCopy,
    fit_centred_spread,
    fit_many,
    pooled_residuals,
    predict_pooled,
)
from studies.dp_pooled import DpConfig, PooledBatch, PooledDemandModel, PopulationScales

TUNED_PATH: Final[Path] = common.WORK_DIR / "tuned.json"
DEFAULT_TUNED: Final[dict[str, float]] = {"k": 4, "map_deviation": 0.3, "site_deviation": 0.3}


def tuned() -> dict[str, float]:
    """Return the tuned settings, or the defaults before tuning has run."""
    if TUNED_PATH.exists():
        return {**DEFAULT_TUNED, **json.loads(TUNED_PATH.read_text())}
    return dict(DEFAULT_TUNED)


def config_for(*, kind: str, settings: dict[str, float]) -> DpConfig:
    """Return the model configuration of an arm kind.

    Args:
        kind: `p` (variational, network on), `p_map` (MAP, fixed scales), `p_nonet` (variational,
            network off), or `s` (the per-site model: MAP with the per-site deviation scale).
        settings: The tuned settings.

    Returns:
        The configuration.

    Raises:
        ValueError: If `kind` is unknown.
    """
    k = int(settings["k"])
    net = common.INTERACTION
    if kind == "p":
        return DpConfig(variational=True, shared_shapes=k, interaction_hidden=net)
    if kind == "p_nonet":
        return DpConfig(variational=True, shared_shapes=k)
    if kind == "p_map":
        scales = PopulationScales(deviation=settings["map_deviation"])
        return DpConfig(population_scales=scales, shared_shapes=k, interaction_hidden=net)
    if kind == "s":
        scales = PopulationScales(deviation=settings["site_deviation"])
        return DpConfig(population_scales=scales, shared_shapes=k, interaction_hidden=net)
    msg = f"unknown arm kind {kind}"
    raise ValueError(msg)


def fit(
    *, batch: PooledBatch, config: DpConfig, copy: FitCopy | None = None
) -> tuple[PooledDemandModel, float, float]:
    """Fit one model on `batch` and return it, its seconds, and its peak GPU memory in GB."""
    models, results = fit_many(
        batch=batch,
        copies=[copy or FitCopy(name="fit")],
        config=config,
        initial_level=torch.full((len(batch.site_labels),), 50.0),
        steps=common.FIT_STEPS,
        learning_rate=common.FIT_RATE,
        final_rate_fraction=common.FINAL_RATE_FRACTION,
    )
    return models[0], results[0].seconds, results[0].peak_memory_bytes / 1e9


@dataclass(frozen=True)
class SiteForecast:
    """One substation's forecast of the scored rows.

    Attributes:
        index: The grid rows scored, in the batch's order.
        mean: The mean forecast in points.
        quantiles: The 11 quantiles in points, shape (rows, 11).
    """

    index: np.ndarray
    mean: np.ndarray
    quantiles: np.ndarray


def forecasts_by_site(
    *,
    model: PooledDemandModel,
    score: PooledBatch,
    replica: tuple[PooledDemandModel, PooledBatch] | None,
    spread_for: dict[str, Any] | None = None,
) -> dict[str, SiteForecast]:
    """Return each scored substation's forecast.

    Args:
        model: The fitted model.
        score: The rows to score.
        replica: The replica fit and its year of rows, whose residuals give each substation's own
            spread; ignored where `spread_for` has the substation.
        spread_for: A ready spread (`ResidualSpread`) per substation label.

    Returns:
        A forecast per substation.
    """
    mean_all = predict_pooled(model=model, batch=score)
    site_of_row = score.site.cpu().numpy()
    band = score.band.cpu().numpy()
    quarter = score.quarter.cpu().numpy()
    grid_row = score.row.cpu().numpy()
    residuals = pooled_residuals(model=replica[0], batch=replica[1]) if replica else None
    out: dict[str, SiteForecast] = {}
    for position, label in enumerate(score.site_labels):
        rows = np.flatnonzero(site_of_row == position)
        if len(rows) == 0:
            continue
        if spread_for is not None and label in spread_for:
            spread = spread_for[label]
        else:
            assert residuals is not None
            own = residuals.site == label
            if int(own.sum()) < MIN_SPREAD_GROUP_ROWS:
                own = np.ones_like(own)
            spread = fit_centred_spread(
                residual=residuals.residual[own],
                band=residuals.band[own],
                quarter=residuals.quarter[own],
            )
        order = np.argsort(grid_row[rows])
        rows = rows[order]
        mean = mean_all[rows]
        quantiles = mean[:, None] + spread.table[band[rows], quarter[rows]]
        out[label] = SiteForecast(index=grid_row[rows], mean=mean, quantiles=quantiles)
    return out


def frames_from(
    *, cache: FeatureCache, forecasts: dict[str, SiteForecast], extra: dict[str, Any] | None = None
) -> pl.DataFrame:
    """Return the prediction frame of several substations' forecasts."""
    frames = []
    for site, forecast in forecasts.items():
        frame = common.prediction_frame(
            cache=cache, site=site, index=forecast.index, quantiles=forecast.quantiles
        ).with_columns(mean=pl.Series(forecast.mean.astype(np.float32)))
        for name, value in (extra or {}).items():
            frame = frame.with_columns(pl.lit(value).alias(name))
        frames.append(frame)
    return pl.concat(frames)


def arm_forecast(
    *,
    cache: FeatureCache,
    config: DpConfig,
    fit_sites: tuple[str, ...],
    score_sites: tuple[str, ...],
    months: int | None,
    cut: datetime,
    through: datetime,
    all_present_rows: bool = True,
    save_as: str | None = None,
) -> tuple[dict[str, SiteForecast], dict[str, float]]:
    """Fit an arm at `cut`, fit its replica 12 months earlier, and forecast the scored rows.

    Args:
        cache: The feature cache.
        config: The model configuration.
        fit_sites: The substations the model trains on.
        score_sites: The substations scored (a subset of `fit_sites`).
        months: The training window in months before the cut, or `None` for all years.
        cut: The cut.
        through: The last scored instant.
        all_present_rows: Also score the rows inside switching masks.
        save_as: If given, the fitted model's state dict is saved under this name.

    Returns:
        The forecasts, and a record of the fit seconds, rows, and peak GPU memory.
    """
    train = common.pooled_batch_for(cache=cache, sites=fit_sites, cut=cut, months=months)
    model, seconds, peak = fit(batch=train, config=config)
    cut_r = common.months_before(moment=cut, months=common.REPLICA_MONTHS)
    train_r = common.pooled_batch_for(cache=cache, sites=fit_sites, cut=cut_r, months=months)
    model_r, seconds_r, _ = fit(batch=train_r, config=config)
    replica_year = common.scoring_batch_for(
        cache=cache, sites=score_sites, after=cut_r, through=cut
    )
    score = common.scoring_batch_for(
        cache=cache,
        sites=score_sites,
        after=cut,
        through=through,
        all_present_rows=all_present_rows,
    )
    # The scoring batches index their sites by `score_sites`, but the models index them by
    # `fit_sites`, so the site column is re-mapped before predicting.
    forecasts = forecasts_by_site(
        model=model,
        score=remap_sites(batch=score, model_sites=fit_sites),
        replica=(model_r, remap_sites(batch=replica_year, model_sites=fit_sites)),
    )
    if save_as:
        common.WORK_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), common.WORK_DIR / f"model_{save_as}.pt")
    record = {
        "seconds": seconds + seconds_r,
        "rows": float(train.n_rows),
        "peak_gpu_gb": peak,
    }
    return forecasts, record


def remap_sites(*, batch: PooledBatch, model_sites: tuple[str, ...]) -> PooledBatch:
    """Return `batch` with its site column and slots indexed by the model's site order."""
    positions = torch.tensor(
        [model_sites.index(label) for label in batch.site_labels], device=batch.site.device
    )
    site = positions[batch.site]
    year = batch.season_slot % batch.n_season_years
    return replace(
        batch,
        site=site,
        season_slot=site * batch.n_season_years + year,
        site_labels=model_sites,
    )


def quantile_score_of(*, frame: pl.DataFrame, headline: bool = True) -> float:
    """Return the mean quantile score of a prediction frame, in points.

    Args:
        frame: Rows with the 11 quantile columns and `actual`.
        headline: Keep only rows outside the switching masks.

    Returns:
        The mean of twice the mean pinball loss over the 11 levels.
    """
    rows = frame.filter(~pl.col("switching_mask")) if headline else frame
    quantiles = rows.select(list(QUANTILE_COLUMNS))
    return float(
        common.quantile_score(
            actual=rows["actual"].to_numpy().astype(np.float64),
            quantiles=quantiles.to_numpy().astype(np.float64),
        ).mean()
    )


def per_site_forecasts(
    *,
    cache: FeatureCache,
    config: DpConfig,
    sites: tuple[str, ...],
    months: int | None,
    cut: datetime,
    through: datetime,
    all_present_rows: bool = True,
    override_for: Callable[[str, datetime, PooledBatch], torch.Tensor | None] | None = None,
) -> tuple[dict[str, SiteForecast], dict[str, float]]:
    """Fit the model one substation at a time and forecast each substation's scored rows.

    A substation with no rows in the replica fit's window (it began after the replica cut) has no
    replica of its own, so its spread is fitted on the other substations' replica residuals.

    Args:
        cache: The feature cache.
        config: The model configuration.
        sites: The substations, each fitted alone.
        months: The training window in months before the cut, or `None` for all years.
        cut: The cut.
        through: The last scored instant.
        all_present_rows: Also score the rows inside switching masks.
        override_for: Given a substation, the cut of a fit, and its training batch, the capacities
            (points, shape (1, season-years)) to hold fixed in that fit, or `None` to learn them.

    Returns:
        The forecasts, and the summed fit record.
    """
    cut_r = common.months_before(moment=cut, months=common.REPLICA_MONTHS)
    fitted: dict[str, tuple[PooledDemandModel, PooledBatch]] = {}
    residuals: dict[str, object] = {}
    record = {"seconds": 0.0, "rows": 0.0, "peak_gpu_gb": 0.0}
    for site in sites:
        train = common.pooled_batch_for(cache=cache, sites=(site,), cut=cut, months=months)
        pinned = override_for(site, cut, train) if override_for else None
        model, seconds, peak = fit(
            batch=train, config=config, copy=FitCopy(name=site, capacity_override=pinned)
        )
        score = common.scoring_batch_for(
            cache=cache,
            sites=(site,),
            after=cut,
            through=through,
            all_present_rows=all_present_rows,
        )
        fitted[site] = (model, score)
        record["seconds"] += seconds
        record["rows"] += float(train.n_rows)
        record["peak_gpu_gb"] = max(record["peak_gpu_gb"], peak)
        train_r = common.pooled_batch_for(cache=cache, sites=(site,), cut=cut_r, months=months)
        replica_year = common.scoring_batch_for(
            cache=cache, sites=(site,), after=cut_r, through=cut
        )
        if train_r.n_rows >= MIN_SPREAD_GROUP_ROWS and replica_year.n_rows:
            pinned_r = override_for(site, cut_r, train_r) if override_for else None
            model_r, seconds_r, _ = fit(
                batch=train_r, config=config, copy=FitCopy(name=site, capacity_override=pinned_r)
            )
            residuals[site] = pooled_residuals(model=model_r, batch=replica_year)
            record["seconds"] += seconds_r
    forecasts: dict[str, SiteForecast] = {}
    for site, (model, score) in fitted.items():
        own = residuals.get(site)
        pool = [r for other, r in residuals.items() if other != site] if own is None else [own]
        spread = fit_centred_spread(
            residual=np.concatenate([r.residual for r in pool]),
            band=np.concatenate([r.band for r in pool]),
            quarter=np.concatenate([r.quarter for r in pool]),
        )
        forecasts |= forecasts_by_site(
            model=model, score=score, replica=None, spread_for={site: spread}
        )
    return forecasts, record
