"""The arms of experiment 109 (copied for experiment 150), on `studies.dp_gam_pooled`'s joint fit.

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

import hashlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import e00150_common as common
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
    PooledReplica,
    PooledSite,
    PoolingStrength,
    fit_pooled,
    pooled_replica_residuals,
    rebuilt_effective_capacity_mw,
    short_history_replica,
    short_history_spread,
    with_effective_capacity,
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
    """A held-out site's change of target: a rebuilt scale and a capacity prior shifted into it.

    Attributes:
        site: The substation's label.
        scale: Original points divided by rebuilt points (the factor the target is multiplied by).
        capacity_mw: The rebuilt effective capacity in MW, or 0 to keep the whole-history scale.
        prior: The capacity prior in rebuilt points, or 0 for none.
    """

    site: str
    scale: float = 1.0
    capacity_mw: float = 0.0
    prior: float = 0.0


def transformed_cache(*, cache: FeatureCache, transform: Transform | None) -> FeatureCache:
    """Return the cache with one site's scale rebuilt and its capacity prior shifted in.

    Args:
        cache: The cache in original points.
        transform: The change, or `None` for the cache as it is.

    Returns:
        A cache whose site has `y * scale + prior * derated_shape` and the rebuilt effective
        capacity in its table, so pinned capacities in megawatts convert to the rebuilt points.
    """
    if transform is None:
        return cache
    changed = (
        with_effective_capacity(cache=cache, site=transform.site, capacity_mw=transform.capacity_mw)
        if transform.capacity_mw > 0.0
        else cache
    )
    if transform.prior == 0.0:
        return changed
    features = changed.sites[transform.site]
    index = np.arange(len(cache.end_times))
    shape = np.nan_to_num(derated_unit_shape(features=features, index=index), nan=0.0)
    shifted = replace(features, y=features.y + transform.prior * shape)
    return replace(changed, sites={**changed.sites, transform.site: shifted})


def history_transform(
    *, cache: FeatureCache, site: str, cut: datetime, window: TrainingWindow, prior: float = 0.0
) -> Transform:
    """Return the held-out site's change of target for a window, with the scale rebuilt.

    Args:
        cache: The cache in original points.
        site: The held-out substation's label.
        cut: The cut.
        window: The held-out site's window.
        prior: The capacity prior in rebuilt points.

    Returns:
        The transform; `scale` is `capacity_original / capacity_rebuilt`.
    """
    rebuilt = rebuilt_effective_capacity_mw(cache=cache, site=site, cut=cut, window=window)
    return Transform(
        site=site,
        scale=cache.sites[site].table.effective_capacity_mw / rebuilt,
        capacity_mw=rebuilt,
        prior=prior,
    )


def trainable(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    window: TrainingWindow,
    minimum: int = common.MIN_TRAINING_ROWS,
) -> bool:
    """Return True if the site holds at least `minimum` usable rows in its window."""
    rows = training_rows(features=cache.sites[site], cache=cache, cut=cut, window=window)
    return len(rows) >= minimum


@dataclass(frozen=True)
class Built:
    """The substations and cache of one arm at one cut.

    Attributes:
        sites: The substations of the joint fit.
        cache: The cache with the held-out site's target changed, if it is.
        transform: The held-out site's change of target, or `None`.
        partial_transforms: The rebuilt scale of each pool site that holds only part of the window
            (D18 at the scored cut, D9 at the replicate and winter cuts).
        capacity_sources: How each site's capacity was set: `pinned`, `fitted_late_vintage`, or
            `fitted_short_window`.
    """

    sites: list[PooledSite]
    cache: FeatureCache
    transform: Transform | None
    partial_transforms: dict[str, Transform]
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
        if not trainable(
            cache=cache,
            site=held_out,
            cut=cut,
            window=held_window,
            minimum=history.minimum_rows,
        ):
            msg = f"{held_out} has too few rows in a {history.label} window before {cut}"
            raise ValueError(msg)
        prior = 0.0
        if config.capacity_prior:
            prior = _capacity_prior(
                cache=cache, vintages=vintages, cut=cut, config=config, excluded=held_out
            )
        transform = (
            history_transform(cache=cache, site=held_out, cut=cut, window=held_window, prior=prior)
            if config.rebuild_scale
            else Transform(site=held_out, prior=prior)
        )
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
                minimum=(
                    history.minimum_rows
                    if site == held_out and history is not None
                    else common.MIN_TRAINING_ROWS
                ),
            )
        ]
    )
    partial_transforms: dict[str, Transform] = {}
    if config.rebuild_scale:
        for label in labels:
            if label == held_out:
                continue
            rows = training_rows(
                features=used.sites[label], cache=used, cut=cut, window=common.FULL_WINDOW
            )
            if len(rows) < common.PARTIAL_HISTORY_ROWS:
                partial_transforms[label] = history_transform(
                    cache=used, site=label, cut=cut, window=common.FULL_WINDOW
                )
                used = transformed_cache(cache=used, transform=partial_transforms[label])
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
                allow_late_vintage=short and config.oracle_pin,
            )
        )
    return Built(
        sites=sites,
        cache=used,
        transform=transform,
        partial_transforms=partial_transforms,
        capacity_sources=sources,
    )


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


REPLICA_VERSION: Final[str] = "2"
"""Bump when the arms or the replica logic change, so stored replicas are never reused."""


def _replica_path(
    *, cut: datetime, config: ArmConfig, history: str, site: str, cache: FeatureCache
) -> Path:
    """Return where a replica is stored; the name hashes everything the residuals depend on."""
    key = f"{config.family}|{config.pool}|{cache.key}|{REPLICA_VERSION}"
    digest = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:10]
    return REPLICA_DIR / f"{cut:%Y%m%d}_{digest}_{history}_{site}.npz"


def _save_replica(*, path: Path, replica: PooledReplica) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        residuals=replica.residuals.residuals,
        band=replica.residuals.band,
        quarter=replica.residuals.quarter,
        index=replica.residuals.index,
        switching=replica.residuals.switching,
        fit_cut=np.array(replica.residuals.fit_cut.isoformat()),
        n_training_rows=np.array(replica.n_training_rows),
    )
    temporary.replace(path)


def _load_replica(*, path: Path) -> PooledReplica:
    stored = np.load(path)
    return PooledReplica(
        residuals=ReplicaResiduals(
            residuals=stored["residuals"],
            band=stored["band"],
            quarter=stored["quarter"],
            index=stored["index"],
            fit_cut=datetime.fromisoformat(str(stored["fit_cut"])),
            switching=stored["switching"],
        ),
        n_training_rows=int(stored["n_training_rows"]),
    )


def replica_of(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    config: ArmConfig,
    held_out: str | None,
    history: common.History | None,
) -> dict[str, PooledReplica]:
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
        site: _replica_path(
            cut=cut, config=config, history=f"{label}_{sites_key}", site=site, cache=cache
        )
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
        strengths=() if config.kind == "tied" else [config.strength],
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
        spread = (
            _centre_spread(replica=replicas[target].residuals)
            if target in replicas
            else _pooled_spread(replicas=replicas)
        )
    else:
        short_replicas = []
        for other in decision_pool:
            if other == target:
                continue
            try:
                others = replica_of(
                    cache=cache,
                    vintages=vintages,
                    cut=cut,
                    config=config,
                    held_out=other,
                    history=history,
                )
            except ValueError:
                continue  # `other` held too few rows in this window at the replica cut
            entry = short_history_replica(
                replicas=others, short_site=other, weeks_of_history=history.weeks_of_history
            )
            short_replicas.append(
                replace(
                    entry,
                    residuals=_to_scale(
                        cache=cache,
                        replica=entry.residuals,
                        other=other,
                        config=config,
                        cut=cut,
                        history=history,
                    ),
                )
            )
        spread = short_history_spread(
            replicas=short_replicas, held_out=target, weeks_of_history=history.weeks_of_history
        )
    transform = built.transform if target == held_out else built.partial_transforms.get(target)
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
    cut: datetime,
    history: common.History,
) -> ReplicaResiduals:
    """Return a short site's replica residuals in original points.

    The replica was fitted in the other site's rebuilt points at the replica's cut. Residuals
    are divided by that scale, so they are comparable across sites and with the target's original
    scale.
    """
    if not config.rebuild_scale:
        return replica
    fit_cut = replica_cut(cut=cut)
    window = history.window(cut=fit_cut)
    rebuilt = rebuilt_effective_capacity_mw(cache=cache, site=other, cut=fit_cut, window=window)
    scale = cache.sites[other].table.effective_capacity_mw / rebuilt
    return replace(replica, residuals=replica.residuals / scale)


def _pooled_spread(*, replicas: dict[str, PooledReplica]) -> ResidualSpread:
    """Return the median-centred spread of every site's replica residuals together.

    A site with no rows at the replica cut (D9 at the replicate cut) has no replica of its own,
    so it takes the spread of the sites that do.
    """
    parts = [replica.residuals for replica in replicas.values()]
    return ResidualSpread.fit(
        residuals=np.concatenate([part.residuals for part in parts]),
        band=np.concatenate([part.band for part in parts]),
        quarter=np.concatenate([part.quarter for part in parts]),
        centre=True,
    )


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
