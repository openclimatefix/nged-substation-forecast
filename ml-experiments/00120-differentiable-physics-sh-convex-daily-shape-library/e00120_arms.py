"""The arms of experiment 120: joint fits of `dp_gam_pooled` and the `dp_shape_library` library.

An arm is a model: the per-site convex model G (`lambda = 0`), the partially pooled model N
(`lambda = 3,000`), complete pooling N-full, or the shape library SH with its settings. This module
builds the substations of an arm at a cut, fits the joint model, builds the spread, and returns the
quantile predictions of the scored substations.

**Capacity.** Each substation's capacity is pinned to the real-time vintage
(`studies.dp_pinned_capacity`), except where the rule falls back to a vintage fitted after the cut.
There the substation's capacity is fitted from its own rows before the cut, in every arm alike, so
the late vintage cannot reach another substation's forecast through the fleet mean.

**Scale.** A pool site that holds only part of its 2-year window (D18 at the scored cut, D9 at the
replicate cut) has its points of effective capacity rebuilt from the rows it holds. The forecast is
converted back to the site's whole-history points before it is scored.

**Spread.** Each substation takes the median-centred spread of its own replica fit 12 months before
the cut, fitted with the arm's own recipe. A substation with no replica rows (D9 at the replicate
cut) takes the spread of the substations that have them.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, Literal

import e00120_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    GamFit,
    GamSpec,
    ReplicaResiduals,
    ResidualSpread,
    TrainingWindow,
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
    with_effective_capacity,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages
from studies.dp_shape_library import (
    InitialShapesType,
    LibraryConfig,
    ShapeLibraryFit,
    fit_library,
    library_replica_residuals,
)

ArmKindType = Literal["g", "n", "n_full", "sh"]
"""The per-site model, the partially pooled model, complete pooling, or the shape library."""

REPLICA_DIR: Final[Path] = common.WORK_DIR / "replicas"
"""Replica residuals, stored once per (cut, arm recipe, site) so workers share them."""
FIT_DIR: Final[Path] = common.WORK_DIR / "fits"
"""The shapes, weights, and deviations of every library fit that was scored."""
REPLICA_VERSION: Final[str] = "1"
"""Bump when the arms or the replica logic change, so stored replicas are never reused."""


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the experiment.

    Attributes:
        name: The prediction set's base name.
        kind: The model.
        library: The library settings, required exactly for `sh`.
        init_method: How the library's starting shapes are built.
        seed: The seed of the starting shapes; also written to the `seed` column.
        pool: The substations of the joint fit.
        tag: Names the data the arm runs on, so stored replicas never mix.
        all_years: True to train on every year before the cut instead of the 2 years before it.
    """

    name: str
    kind: ArmKindType = "n"
    library: LibraryConfig | None = None
    init_method: InitialShapesType = "svd"
    seed: int = 0
    pool: tuple[str, ...] = common.POOL_SITES
    tag: str = "real"
    all_years: bool = False

    @property
    def family(self) -> str:
        """A key that names the recipe, so replicas are shared between arms that fit alike."""
        return (
            f"{self.tag}|{self.kind}|{self.library}|{self.init_method}|{self.seed}|{self.all_years}"
        )


@dataclass(frozen=True)
class Built:
    """The substations and cache of one arm at one cut.

    Attributes:
        sites: The substations of the joint fit.
        cache: The cache with the partial-history sites' scale rebuilt.
        scales: The factor each rebuilt site's target was multiplied by (original over rebuilt
            points); sites with the whole-history scale are absent.
        capacity_sources: How each site's capacity was set: `pinned` or `fitted_late_vintage`.
    """

    sites: list[PooledSite]
    cache: FeatureCache
    scales: dict[str, float]
    capacity_sources: dict[str, str]


def trainable(*, cache: FeatureCache, site: str, cut: datetime) -> bool:
    """Return True if the site holds at least `MIN_TRAINING_ROWS` usable rows in its window."""
    rows = training_rows(
        features=cache.sites[site], cache=cache, cut=cut, window=common.FULL_WINDOW
    )
    return len(rows) >= common.MIN_TRAINING_ROWS


def build_pool(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    pool: tuple[str, ...],
    all_years: bool = False,
) -> Built:
    """Build the substations of a joint fit at a cut.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut.
        pool: The candidate substations; each enters only if it holds enough rows.
        all_years: True to train on every year before the cut.

    Returns:
        The substations, the cache, the rebuilt scales, and how each capacity was set.
    """
    labels = [site for site in pool if trainable(cache=cache, site=site, cut=cut)]
    used = cache
    scales: dict[str, float] = {}
    for label in labels:
        rows = training_rows(
            features=used.sites[label], cache=used, cut=cut, window=common.FULL_WINDOW
        )
        if len(rows) < common.PARTIAL_HISTORY_ROWS:
            rebuilt = rebuilt_effective_capacity_mw(
                cache=used, site=label, cut=cut, window=common.FULL_WINDOW
            )
            scales[label] = used.sites[label].table.effective_capacity_mw / rebuilt
            used = with_effective_capacity(cache=used, site=label, capacity_mw=rebuilt)
    sites: list[PooledSite] = []
    sources: dict[str, str] = {}
    for label in labels:
        pinned = pinned_capacity(vintages=vintages, table=used.sites[label].table, cut=cut)
        use_pinned = pinned.source != "growth_prior"
        sources[label] = "pinned" if use_pinned else "fitted_late_vintage"
        sites.append(
            PooledSite(
                site=label,
                spec=GamSpec(
                    ridge=common.BASE_RIDGE, capacity="pinned" if use_pinned else "fitted"
                ),
                window=TrainingWindow() if all_years else common.FULL_WINDOW,
                season_years=common.SEASON_YEARS,
                pinned=pinned if use_pinned else None,
            )
        )
    return Built(sites=sites, cache=used, scales=scales, capacity_sources=sources)


def fit_arm(
    *, built: Built, cut: datetime, spec: ArmSpec
) -> tuple[dict[str, GamFit], ShapeLibraryFit | None]:
    """Fit an arm's joint model at a cut.

    Args:
        built: The arm's substations and cache.
        cut: The cut.
        spec: The arm.

    Returns:
        Each substation's fit, and the library fit for `sh` (else `None`).

    Raises:
        ValueError: If a shape-library arm has no settings.
    """
    if spec.kind == "sh":
        if spec.library is None:
            msg = f"arm {spec.name} is a shape library and needs settings"
            raise ValueError(msg)
        fit = fit_library(
            cache=built.cache,
            sites=built.sites,
            cut=cut,
            config=spec.library,
            init_method=spec.init_method,
            seed=spec.seed,
        )
        return fit.fits, fit
    if spec.kind == "n_full":
        joint = fit_pooled(cache=built.cache, sites=built.sites, cut=cut, tied=True)
    else:
        value = common.PRODUCTION_LAMBDA if spec.kind == "n" else 0.0
        joint = fit_pooled(
            cache=built.cache,
            sites=built.sites,
            cut=cut,
            strengths=[PoolingStrength.uniform(value=value)],
        )
    return joint.fits, None


def _centred_spread(*, replica: ReplicaResiduals) -> ResidualSpread:
    """Return the median-centred spread of one site's replica residuals."""
    return ResidualSpread.fit(
        residuals=replica.residuals, band=replica.band, quarter=replica.quarter, centre=True
    )


def _pooled_spread(*, replicas: dict[str, PooledReplica]) -> ResidualSpread:
    """Return the median-centred spread of every site's replica residuals together."""
    parts = [replica.residuals for replica in replicas.values()]
    return ResidualSpread.fit(
        residuals=np.concatenate([part.residuals for part in parts]),
        band=np.concatenate([part.band for part in parts]),
        quarter=np.concatenate([part.quarter for part in parts]),
        centre=True,
    )


def _replica_path(*, cut: datetime, spec: ArmSpec, site: str, cache: FeatureCache) -> Path:
    """Return where a replica is stored; the name hashes everything the residuals depend on."""
    key = f"{spec.family}|{spec.pool}|{cache.key}|{REPLICA_VERSION}"
    digest = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:10]
    return REPLICA_DIR / f"{cut:%Y%m%d}_{digest}_{site}.npz"


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
    *, cache: FeatureCache, vintages: pl.DataFrame, cut: datetime, spec: ArmSpec
) -> dict[str, PooledReplica]:
    """Return the replica residuals of every site of an arm, stored once on disk.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut of the final fit; the replica is fitted 12 months earlier.
        spec: The arm.

    Returns:
        Each site's residuals, in the rebuilt points the replica was fitted in.
    """
    fit_cut = replica_cut(cut=cut)
    built = build_pool(
        cache=cache, vintages=vintages, cut=fit_cut, pool=spec.pool, all_years=spec.all_years
    )
    wanted = [site.site for site in built.sites]
    paths = {site: _replica_path(cut=cut, spec=spec, site=site, cache=cache) for site in wanted}
    if all(path.exists() for path in paths.values()):
        return {site: _load_replica(path=path) for site, path in paths.items()}
    if spec.kind == "sh":
        if spec.library is None:
            msg = f"arm {spec.name} is a shape library and needs settings"
            raise ValueError(msg)
        residuals = library_replica_residuals(
            cache=built.cache,
            sites=built.sites,
            cut=cut,
            config=spec.library,
            init_method=spec.init_method,
            seed=spec.seed,
        )
    else:
        value = common.PRODUCTION_LAMBDA if spec.kind == "n" else 0.0
        residuals = pooled_replica_residuals(
            cache=built.cache,
            sites=built.sites,
            cut=cut,
            strength=PoolingStrength.uniform(value=value),
            tied=spec.kind == "n_full",
        )
    for site, replica in residuals.items():
        _save_replica(path=paths[site], replica=replica)
    return residuals


@dataclass
class ArmResult:
    """What fitting and scoring an arm at a cut produced.

    Attributes:
        frame: The predictions of every scored site, in original points.
        library: The library fit for `sh`, else `None`.
        capacity_sources: How each site's capacity was set.
        pool: The substations in the joint fit.
        stored: The files the fit's readouts were saved to.
    """

    frame: pl.DataFrame
    library: ShapeLibraryFit | None
    capacity_sources: dict[str, str]
    pool: list[str]
    stored: list[Path] = field(default_factory=list)


def fit_and_score(
    *,
    cache: FeatureCache,
    vintages: pl.DataFrame,
    cut: datetime,
    spec: ArmSpec,
    score_sites: tuple[str, ...] | None = None,
) -> ArmResult:
    """Fit an arm at a cut and return the predictions of the scored sites over the following year.

    Args:
        cache: The cache in original points.
        vintages: The real-time capacity vintages.
        cut: The cut.
        spec: The arm.
        score_sites: The sites to score; every pool site at the cut if `None`.

    Returns:
        The predictions in original points, and the readouts of the fit.
    """
    built = build_pool(
        cache=cache, vintages=vintages, cut=cut, pool=spec.pool, all_years=spec.all_years
    )
    fits, library = fit_arm(built=built, cut=cut, spec=spec)
    replicas = replica_of(cache=cache, vintages=vintages, cut=cut, spec=spec)
    targets = score_sites or tuple(site.site for site in built.sites)
    frames = []
    for target in targets:
        if target not in fits:
            continue
        index = common.scoring_index(
            cache=cache, site=target, after=cut, through=common.year_end(cut=cut)
        )
        mean = predict_gam(cache=built.cache, fit=fits[target], index=index)
        mean = mean / built.scales.get(target, 1.0)
        spread = (
            _centred_spread(replica=replicas[target].residuals)
            if target in replicas
            else _pooled_spread(replicas=replicas)
        )
        quantiles = spread.quantiles(
            mean=mean, band=cache.calendar.band[index], quarter=cache.calendar.quarter[index]
        )
        frames.append(
            common.prediction_frame(
                cache=cache,
                site=target,
                index=index,
                quantiles=quantiles,
                mean=mean,
                arm=spec.name,
                seed=spec.seed,
                cut=cut,
            )
        )
    return ArmResult(
        frame=pl.concat(frames),
        library=library,
        capacity_sources=built.capacity_sources,
        pool=[site.site for site in built.sites],
    )


def save_library_fit(*, name: str, library: ShapeLibraryFit, extra: dict[str, object]) -> Path:
    """Save a library fit's shapes, weights, deviations, and trace, once.

    Args:
        name: The file's base name.
        library: The fit.
        extra: Small JSON-able facts to store beside the arrays (cut, pool, capacity sources).

    Returns:
        The `.npz` path; a `.json` with the scalar facts sits beside it.
    """
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    path = FIT_DIR / f"{name}.npz"
    labels = sorted(library.weights)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        shapes=library.shapes,
        labels=np.array(labels),
        weights=np.stack([library.weights[label] for label in labels]),
        deviations=np.stack([library.deviations[label] for label in labels]),
        band_means=np.stack([library.band_means[label] for label in labels]),
        objective_trace=np.array(library.objective_trace),
    )
    temporary.replace(path)
    facts = {
        "objective": library.objective_trace[-1],
        "n_sweeps": library.n_sweeps,
        "converged": library.converged,
        "kappa": library.kappa,
        "n_library_parameters": library.n_library_parameters,
        **extra,
    }
    path.with_suffix(".json").write_text(json.dumps(facts, indent=2) + "\n")
    return path


def load_source() -> tuple[FeatureCache, pl.DataFrame]:
    """Return the real cache and the real-time capacity vintages."""
    return common.load_cache(), read_vintages()


__all__ = [
    "ArmResult",
    "ArmSpec",
    "Built",
    "build_pool",
    "fit_and_score",
    "fit_arm",
    "load_source",
    "replica_of",
    "save_library_fit",
]
