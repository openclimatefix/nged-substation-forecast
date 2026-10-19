"""The arms of experiment 109, built on the joint fit of `studies.dp_gam_pooled`.

An arm is a model (partial pooling at a given `lambda`, complete pooling, or per-site at
`lambda = 0`) plus, for a short-history arm, how much history one held-out site keeps. This module
builds each arm's substations at a cut, fits the joint model, builds the spread, and returns the
quantile predictions of the scored sites.

**Capacity.** Each substation's capacity is pinned to the real-time vintage
(`studies.dp_pinned_capacity`), except where the rule falls back to a vintage fitted after the cut.
There the substation's capacity is fitted from its own rows before the cut, in the pooled and the
per-site arm alike, so the late vintage cannot reach another substation's forecast through the
fleet mean. The held-out site of a short-history arm fits its capacity from its own short window,
because its vintage was fitted on its whole history.

**Scale.** A held-out site's points of effective capacity are rebuilt from the rows the arm sees
(the 99th percentile of the absolute power in its window), in the pooled and the per-site arm
alike. The forecast is converted back to the site's whole-history points before it is scored.

**Spread.** A full-history arm takes the median-centred spread of its own replica fit 12 months
before the cut. A short-history arm for site `s` takes the spread of each other decision site `t`
in the replica fit in which `t` held the same history and every other site its full window.
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import e00109_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache, load_feature_cache
from studies.dp_gam import (
    GamSpec,
    ReplicaResiduals,
    ResidualSpread,
    TrainingWindow,
    derated_unit_shape,
    predict_gam,
    replica_cut,
    training_rows,
)
from studies.dp_gam_pooled import (
    PooledSite,
    PoolingStrength,
    ShortHistoryReplica,
    fit_pooled,
    pooled_replica_residuals,
    short_history_spread,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

ModelKindType = Literal["partial", "tied"]
"""Partial pooling at the arm's strength (zero is the per-site model), or complete pooling."""

REPLICA_DIR: Final[Path] = common.WORK_DIR / "replicas"
"""Replica residuals, stored once per (cut, model, history, site) so workers share them."""


@dataclass(frozen=True)
class ArmConfig:
    """One arm of the experiment.

    Attributes:
        name: The prediction set's base name.
        kind: Partial or complete pooling.
        strength: The deviation penalty's weight (ignored for complete pooling).
        history: The held-out site's history, or `None` for an all-rows arm.
        short_ridge: The base ridge of the held-out site, for the per-site arm on a short history.
        alone: True to fit the held-out site on its own (the per-site arm); the other sites cannot
            affect it at `lambda = 0`, so this only saves time.
        oracle_pin: True to give the held-out site the vintage capacity, an oracle leak.
        capacity_prior: True to shrink the held-out site's capacity towards the median of the other
            sites' pinned capacities.
        rebuild_scale: True to rebuild the held-out site's points of effective capacity from its
            window. The known-answer gate turns it off to isolate shrinkage from the scale.
        tag: Names the data the arm runs on (`real`, or a synthetic gate variant), so stored
            replicas never mix.
        pool: The substations of the joint fit.
    """

    name: str
    kind: ModelKindType = "partial"
    strength: PoolingStrength = field(default_factory=lambda: PoolingStrength.uniform(value=0.0))
    history: common.History | None = None
    short_ridge: float | None = None
    alone: bool = False
    oracle_pin: bool = False
    capacity_prior: bool = False
    rebuild_scale: bool = True
    tag: str = "real"
    pool: tuple[str, ...] = common.POOL_SITES

    @property
    def family(self) -> str:
        """A key that names the model, so replicas are shared between arms that fit alike."""
        return (
            f"{self.tag}_{self.kind}_{self.strength.heating:g}_{self.strength.other:g}"
            f"_{self.short_ridge}_{self.alone}_{self.oracle_pin}_{self.capacity_prior}"
            f"_{self.rebuild_scale}"
        )


@dataclass(frozen=True)
class Transform:
    """A held-out site's change of target: a new scale and a capacity prior shifted into it.

    Attributes:
        site: The substation's label.
        scale: Original points divided by rebuilt points (the factor the target is multiplied by).
        prior: The capacity prior in rebuilt points, or 0 for none.
    """

    site: str
    scale: float = 1.0
    prior: float = 0.0


def transformed_cache(
    *, cache: FeatureCache, transform: Transform | None, shape_rows: np.ndarray | None = None
) -> FeatureCache:
    """Return the cache with one site's target rescaled and its capacity prior shifted in.

    Args:
        cache: The cache in original points.
        transform: The change, or `None` for the cache as it is.
        shape_rows: Unused; present so callers may pass precomputed rows.

    Returns:
        A cache whose site has `y * scale + prior * derated_shape` and `effective_capacity_mw`
        divided by the scale, so pinned capacities in megawatts convert to the rebuilt points.
    """
    if transform is None:
        return cache
    features = cache.sites[transform.site]
    index = np.arange(len(cache.end_times))
    shape = np.nan_to_num(derated_unit_shape(features=features, index=index), nan=0.0)
    table = replace(
        features.table, effective_capacity_mw=features.table.effective_capacity_mw / transform.scale
    )
    changed = replace(
        features, table=table, y=features.y * transform.scale + transform.prior * shape
    )
    return replace(cache, sites={**cache.sites, transform.site: changed})


def history_scale(
    *, cache: FeatureCache, site: str, cut: datetime, window: TrainingWindow
) -> float:
    """Return original points divided by points rebuilt from the rows a short window holds.

    The effective capacity of a series is the 99th percentile of its absolute power over its
    history. For the short window it is rebuilt from the rows the window trains on.

    Args:
        cache: The cache in original points.
        site: The held-out substation's label.
        cut: The cut.
        window: The held-out site's window.

    Returns:
        `capacity_original / capacity_rebuilt`, so a rebuilt target is the original times this.

    Raises:
        ValueError: If the window holds no training row.
    """
    features = cache.sites[site]
    rows = training_rows(features=features, cache=cache, cut=cut, window=window)
    if len(rows) == 0:
        msg = f"{site} has no training row in its window before {cut}"
        raise ValueError(msg)
    power_mw = np.abs(features.y[rows]) * features.table.effective_capacity_mw / 100.0
    rebuilt = float(np.quantile(power_mw, 0.99))
    return features.table.effective_capacity_mw / rebuilt


def trainable(*, cache: FeatureCache, site: str, cut: datetime, window: TrainingWindow) -> bool:
    """Return True if the site holds at least `MIN_TRAINING_ROWS` usable rows in its window."""
    rows = training_rows(features=cache.sites[site], cache=cache, cut=cut, window=window)
    return len(rows) >= common.MIN_TRAINING_ROWS


@dataclass(frozen=True)
class Built:
    """The substations and cache of one arm at one cut.

    Attributes:
        sites: The substations of the joint fit.
        cache: The cache with the held-out site's target changed, if it is.
        transform: The held-out site's change of target, or `None`.
        capacity_sources: How each site's capacity was set: `pinned`, `fitted_late_vintage`, or
            `fitted_short_window`.
    """

    sites: list[PooledSite]
    cache: FeatureCache
    transform: Transform | None
    capacity_sources: dict[str, str]


def build_arm(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    config: ArmConfig,
    held_out: str | None,
    history: common.History | None,
) -> Built:
    """Build the substations of an arm at a cut, with `held_out` holding only `history`.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut.
        config: The arm.
        held_out: The short-history site, or `None`.
        history: Its history, or `None`.

    Returns:
        The substations, the cache, and how each capacity was set.

    Raises:
        ValueError: If the held-out site has too few rows in its window.
    """
    transform = None
    held_window = None
    if held_out is not None and history is not None:
        held_window = history.window(cut=cut)
        if not trainable(cache=cache, site=held_out, cut=cut, window=held_window):
            msg = f"{held_out} has too few rows in a {history.label} window before {cut}"
            raise ValueError(msg)
        prior = 0.0
        scale = (
            history_scale(cache=cache, site=held_out, cut=cut, window=held_window)
            if config.rebuild_scale
            else 1.0
        )
        if config.capacity_prior:
            prior = _capacity_prior(
                cache=cache, vintages=vintages, cut=cut, config=config, excluded=held_out
            )
        transform = Transform(site=held_out, scale=scale, prior=prior)
    used = transformed_cache(cache=cache, transform=transform)
    labels = (
        [held_out]
        if config.alone and held_out is not None
        else [
            site
            for site in config.pool
            if trainable(
                cache=used,
                site=site,
                cut=cut,
                window=held_window if site == held_out and held_window else common.FULL_WINDOW,
            )
        ]
    )
    sites: list[PooledSite] = []
    sources: dict[str, str] = {}
    for label in labels:
        short = label == held_out and held_window is not None
        window = held_window if short and held_window is not None else common.FULL_WINDOW
        ridge = (config.short_ridge or common.BASE_RIDGE) if short else common.BASE_RIDGE
        pinned = pinned_capacity(vintages=vintages, table=used.sites[label].table, cut=cut)
        use_pinned = pinned.source != "growth_prior" and not (short and not config.oracle_pin)
        if short and config.oracle_pin:
            use_pinned = True
        sources[label] = (
            "pinned" if use_pinned else "fitted_short_window" if short else "fitted_late_vintage"
        )
        sites.append(
            PooledSite(
                site=label,
                spec=GamSpec(ridge=ridge, capacity="pinned" if use_pinned else "fitted"),
                window=window,
                season_years=common.SEASON_YEARS,
                pinned=pinned if use_pinned else None,
            )
        )
    return Built(sites=sites, cache=used, transform=transform, capacity_sources=sources)


def _capacity_prior(
    *, cache: FeatureCache, vintages: pl.DataFrame, cut: datetime, config: ArmConfig, excluded: str
) -> float:
    """Return the median of the other sites' held pinned capacities at the cut, in points."""
    held = []
    for site in config.pool:
        if site == excluded or not trainable(
            cache=cache, site=site, cut=cut, window=common.FULL_WINDOW
        ):
            continue
        pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
        if pinned.source != "growth_prior":
            held.append(pinned.held_points)
    if not held:
        msg = f"no other site has a pinned capacity at {cut} to make a capacity prior"
        raise ValueError(msg)
    return float(np.median(held))


def _centre_spread(*, replica: ReplicaResiduals) -> ResidualSpread:
    """Return the median-centred spread of one site's replica residuals."""
    return ResidualSpread.fit(
        residuals=replica.residuals, band=replica.band, quarter=replica.quarter, centre=True
    )


def _replica_path(*, cut: datetime, config: ArmConfig, history: str, site: str) -> Path:
    return REPLICA_DIR / f"{cut:%Y%m%d}_{config.family}_{history}_{site}.npz"


def _save_replica(*, path: Path, replica: ReplicaResiduals) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        residuals=replica.residuals,
        band=replica.band,
        quarter=replica.quarter,
        index=replica.index,
        switching=replica.switching,
        fit_cut=np.array(replica.fit_cut.isoformat()),
    )
    temporary.replace(path)


def _load_replica(*, path: Path) -> ReplicaResiduals:
    stored = np.load(path)
    return ReplicaResiduals(
        residuals=stored["residuals"],
        band=stored["band"],
        quarter=stored["quarter"],
        index=stored["index"],
        fit_cut=datetime.fromisoformat(str(stored["fit_cut"])),
        switching=stored["switching"],
    )


def replica_of(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    config: ArmConfig,
    held_out: str | None,
    history: common.History | None,
) -> dict[str, ReplicaResiduals]:
    """Return the replica residuals of every site of an arm, stored once on disk.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut of the final fit; the replica is fitted 12 months earlier.
        config: The arm.
        held_out: The short-history site in the replica fit, or `None`.
        history: Its history.

    Returns:
        Each site's residuals, in the transformed points the replica was fitted in.
    """
    label = history.label if history else "full"
    sites_key = held_out or "all"
    fit_cut = replica_cut(cut=cut)
    paths = {
        site: _replica_path(cut=cut, config=config, history=f"{label}_{sites_key}", site=site)
        for site in config.pool
    }
    present = {site: path for site, path in paths.items() if path.exists()}
    built = build_arm(
        cache=cache,
        vintages=vintages,
        cut=fit_cut,
        config=config,
        held_out=held_out,
        history=history,
    )
    wanted = [site.site for site in built.sites]
    if all(site in present for site in wanted):
        return {site: _load_replica(path=present[site]) for site in wanted}
    # The replica is predicted up to the final cut, so the replica's rows are the final year's.
    residuals = pooled_replica_residuals(
        cache=built.cache,
        sites=built.sites,
        cut=cut,
        strength=config.strength,
        tied=config.kind == "tied",
    )
    for site, replica in residuals.items():
        _save_replica(path=paths[site], replica=replica)
    return residuals


def fit_and_score_site(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    config: ArmConfig,
    target: str,
    decision_pool: tuple[str, ...] = common.DECISION_SITES,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Fit an arm at a cut and return one scored site's predictions.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut.
        config: The arm.
        target: The scored site; also the held-out site for a short-history arm.
        decision_pool: The other sites whose short-history replicas make a short arm's spread.

    Returns:
        The predictions in original points, and how each site's capacity was set.
    """
    history = config.history
    held_out = target if history is not None else None
    built = build_arm(
        cache=cache, vintages=vintages, cut=cut, config=config, held_out=held_out, history=history
    )
    joint = fit_pooled(
        cache=built.cache,
        sites=built.sites,
        cut=cut,
        strengths=[config.strength],
        tied=config.kind == "tied",
    )
    index = common.scoring_index(
        cache=cache, site=target, after=cut, through=common.year_end(cut=cut)
    )
    mean = predict_gam(cache=built.cache, fit=joint.fits[target], index=index)
    if history is None:
        replicas = replica_of(
            cache=cache, vintages=vintages, cut=cut, config=config, held_out=None, history=None
        )
        spread = _centre_spread(replica=replicas[target])
    else:
        short_replicas = []
        for other in decision_pool:
            if other == target:
                continue
            others = replica_of(
                cache=cache,
                vintages=vintages,
                cut=cut,
                config=config,
                held_out=other,
                history=history,
            )
            short_replicas.append(
                ShortHistoryReplica(
                    short_site=other,
                    weeks_of_history=history.weeks_of_history,
                    residuals=_to_scale(
                        cache=cache,
                        replica=others[other],
                        other=other,
                        config=config,
                        vintages=vintages,
                        cut=cut,
                        history=history,
                    ),
                )
            )
        spread = short_history_spread(
            replicas=short_replicas, held_out=target, weeks_of_history=history.weeks_of_history
        )
    transform = built.transform
    if transform is not None:
        shape = np.nan_to_num(
            derated_unit_shape(features=cache.sites[target], index=index), nan=0.0
        )
        mean = (mean - transform.prior * shape) / transform.scale
    quantiles = spread.quantiles(
        mean=mean, band=cache.calendar.band[index], quarter=cache.calendar.quarter[index]
    )
    frame = common.prediction_frame(
        cache=cache,
        site=target,
        index=index,
        quantiles=quantiles,
        mean=mean,
        arm=config.name,
        held_out=held_out or "",
        weeks_of_history=history.weeks_of_history if history else 0,
    )
    return frame, built.capacity_sources


def _to_scale(
    *,
    cache: FeatureCache,
    replica: ReplicaResiduals,
    other: str,
    config: ArmConfig,
    vintages: pl.DataFrame,
    cut: datetime,
    history: common.History,
) -> ReplicaResiduals:
    """Return a short site's replica residuals in original points.

    The replica was fitted in the other site's rebuilt points at the replica's cut. Residuals
    are divided by that scale, so they are comparable across sites and with the target's original
    scale.
    """
    fit_cut = replica_cut(cut=cut)
    window = history.window(cut=fit_cut)
    scale = (
        history_scale(cache=cache, site=other, cut=fit_cut, window=window)
        if config.rebuild_scale
        else 1.0
    )
    return replace(replica, residuals=replica.residuals / scale)


def centred_spread_of(*, replica: ReplicaResiduals) -> ResidualSpread:
    """Return the median-centred spread of one site's replica residuals (public alias)."""
    return _centre_spread(replica=replica)


@dataclass(frozen=True)
class Source:
    """Where a worker process reads its data: the real cache and vintages, or a synthetic pair.

    Attributes:
        cache_path: A saved feature cache, or `None` for the real one.
        vintages_path: A parquet of capacity vintages, or `None` for the real ones.
    """

    cache_path: Path | None = None
    vintages_path: Path | None = None


def load_source(*, source: Source) -> tuple[FeatureCache, pl.DataFrame]:
    """Return the cache and the vintages a source names.

    Args:
        source: The source.

    Returns:
        The feature cache and the capacity vintages.
    """
    cache = (
        load_feature_cache(path=source.cache_path)
        if source.cache_path is not None
        else common.load_cache()
    )
    vintages = (
        pl.read_parquet(source.vintages_path)
        if source.vintages_path is not None
        else read_vintages()
    )
    return cache, vintages
