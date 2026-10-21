"""The known-answer gate: could the shape library recover a library that exists?

**Blocking.** No replicate-year or scored-year fit runs until every blocking line passes. A failed
line caps the verdict at `inconclusive`. The gate is synthetic and contains no generator data.

**The synthetic fleet.** One synthetic substation per Primary in the real pool at the tuning cut
(five), on that Primary's real calendar and weather, with the real pinned capacity of the cut and
the real G fit's coefficients for every family other than the profile. The profile is the real
Primary's band means plus a library of K* = 3 shapes (random sums of Gaussian bumps, in half the
fleets at pairwise cosines of 0.3 to 0.6) with sparse non-negative weights (Dirichlet 0.3, with
one site near each vertex), plus a deviation holding 10% of the library's energy. The noise, for
training and held-out rows alike, is whole weeks block-resampled from G's residuals on rows before
the cut. No tuning-year residual enters it.

**Lines** (a line passes at 15 or more of the 20 fleets of its kind. The plan asked for 30 of 40
fleets, two sets pooled; one fleet took 25 minutes to fit, so the plan's cut order applies, with the
same 75% threshold. Per-line rates and Wilson 90% intervals are printed).

1. Shapes: with K = K*, every truth shape has a matched fitted shape (Hungarian, signed cosine)
   with cosine at least 0.9.
2. Weights: with K = K*, the median of |fitted - true| over the matched weights, relative to the
   substation's total true weight, is at most 0.15.
3. No duplicates: with K = K* + 1 and the tuned `gamma`, the largest pairwise |cosine| is under
   0.5. With `gamma` = 0 it is reported only.
4. Deviation: one site-specific evening bump absent from every shape leaves every shape cosine at
   least 0.9, and the fitted deviation holds at least 60% of the bump (projection coefficient).
5. Null fleet (20 other fleets): independent random profiles with the same energy. SH minus G on
   held-out rows is at least 0, and the median signed cosine between the shapes of two starts is
   under 0.6.
6. K choice: the tuning rule over K in {2, 3, 4, 6} (the K = 3 and 4 tie goes to the smaller K,
   margin 0.023 points) picks K = 3 or 4.
7. Local optima: of 5 starts (one singular value decomposition, four random Dirichlet), at least 4
   reach a training sum of squares within 1% of the best.
8. Module tests pass.
9. Score power (20 other fleets): the library's energy is scaled until the oracle's held-out score
   beats N's by 0.10 points; SH minus N on held-out rows must then be at most -0.05.
10. Economy 7-like step (the sparse fleets): a step of 4 points of capacity from 01:00
    to 07:00, absent from every shape, is injected into 2 of the 5 substations. The library with
    K = K* + 1, told nothing, must recover its time (largest rise within 1 half-hour) and size
    (within 25%) at both substations, and a placebo (same truth, new noise) must show no step of
    25% of the size or more.

The dense prior (Dirichlet 1) is reported with lines 1 and 2 as identification-limited and is not
blocking. Lines 1 and 2 and 4 use the start with the lowest training objective among the 5 starts,
as the headline fit does.

    uv run python ml-experiments/00120-*/e00120_gate.py run [--workers 6]
    uv run python ml-experiments/00120-*/e00120_gate.py report
"""

import argparse
import json
import math
import subprocess
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Final

import e00120_arms as arms
import e00120_common as common
import numpy as np
import polars as pl
import torch
from scipy.stats import norm
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    QUANTILE_LEVELS,
    GamFit,
    derated_unit_shape,
    design_matrix,
    predict_gam,
    training_rows,
)
from studies.dp_gam_pooled import (
    PoolingStrength,
    centring_matrix,
    fit_pooled,
)
from studies.dp_pinned_capacity import pinned_capacity
from studies.dp_shape_library import (
    PROFILE_COLUMNS,
    InitialShapesType,
    LibraryConfig,
    ShapeLibraryFit,
    fit_library,
    largest_pair_cosine,
    match_shapes,
)

GATE_DIR: Final[Path] = common.WORK_DIR / "gate"
N_SETS: Final[int] = 1
"""One set of 20 fleets, the plan's cut order for a gate that takes more than 8 hours."""
FLEETS_PER_SET: Final[int] = 20
SET_SEEDS: Final[tuple[int, ...]] = (20261031, 20261101)
TRUE_SHAPES: Final[int] = 3
"""K*: the number of shapes in the truth."""
DEVIATION_ENERGY: Final[float] = 0.10
"""The deviation's energy as a share of the library's."""
BUMP_ENERGY: Final[float] = 0.25
"""The site-specific bump's energy as a share of the library's, for line 4."""
POWER_GAIN: Final[float] = 0.10
"""The oracle gain over N, in points, that line 9 sizes the truth to."""
POWER_REQUIRED: Final[float] = -0.05
"""SH minus N must be at most this on a power fleet."""
STEP_POINTS: Final[float] = 4.0
"""The injected Economy 7-like step, in points of capacity."""
STEP_FIRST_COLUMN: Final[int] = 2
STEP_LAST_COLUMN: Final[int] = 13
"""The step covers profile columns 2 to 13 (01:00 to 07:00) of every day type."""
STEP_SIZE_TOLERANCE: Final[float] = 0.25
STEP_SITES: Final[tuple[int, ...]] = (0, 2)
K_TIE_MARGIN: Final[float] = 0.023
"""Scores within this many points tie in the tuning rule, and a tie goes to the smaller K."""
LINE_PASS_COUNT: Final[int] = 15
SHAPE_COSINE: Final[float] = 0.9
WEIGHT_ERROR: Final[float] = 0.15
DUPLICATE_COSINE: Final[float] = 0.5
DEVIATION_SHARE: Final[float] = 0.6
NULL_SEED_COSINE: Final[float] = 0.6
LOCAL_OPTIMUM_SHARE: Final[float] = 0.01
K_CHOICES: Final[tuple[int, ...]] = (2, 3, 4, 6)
WEEK_ROWS: Final[int] = 7 * common.HALF_HOURS_PER_DAY
START_METHODS: Final[tuple[tuple[int, InitialShapesType], ...]] = (
    (1, "svd"),
    (2, "dirichlet"),
    (3, "dirichlet"),
    (4, "dirichlet"),
    (5, "dirichlet"),
)
"""The five starts of a fit: seed 1 from the singular value decomposition, seeds 2 to 5 random."""


@dataclass(frozen=True)
class Template:
    """What the synthetic fleets take from the real data at the tuning cut.

    Attributes:
        real_pool: The real pool sites at the tuning cut, in order.
        profiles: Each real site's centred profile from G, shape (sites, 192).
        band_means: Each real site's profile band means as a 192 vector (a step per band).
        points: Each real site's pinned capacity in points at the cut.
        noise_bank: Each real site's whole weeks of G training-window residuals.
        base_y: Each real site's signal from every family but the profile, minus the pinned solar,
            on the grid.
        profile_column: Each real site's active profile column per grid row (-1 where unusable).
    """

    real_pool: tuple[str, ...]
    profiles: np.ndarray
    band_means: np.ndarray
    points: tuple[float, ...]
    noise_bank: dict[str, list[np.ndarray]]
    base_y: dict[str, np.ndarray]
    profile_column: dict[str, np.ndarray]


def _profile_column(*, x: np.ndarray, layout_profile: slice) -> np.ndarray:
    """Return the index of the active profile column in each design row."""
    return np.argmax(x[:, layout_profile], axis=1).astype(np.int16)


def build_template(*, cache: FeatureCache, vintages: pl.DataFrame) -> Template:
    """Fit G at the tuning cut and build what every synthetic fleet needs.

    Args:
        cache: The real cache.
        vintages: The real capacity vintages.

    Returns:
        The template.
    """
    built = arms.build_pool(
        cache=cache, vintages=vintages, cut=common.TUNING_CUT, pool=common.POOL_SITES
    )
    g_fit = fit_pooled(
        cache=built.cache,
        sites=built.sites,
        cut=common.TUNING_CUT,
        strengths=[PoolingStrength.uniform(value=0.0)],
    )
    labels = tuple(site.site for site in built.sites)
    layout = g_fit.fits[labels[0]].layout
    centring = centring_matrix(family="profile", size=PROFILE_COLUMNS)
    profiles, band_means, points = [], [], []
    bank: dict[str, list[np.ndarray]] = {}
    base_y: dict[str, np.ndarray] = {}
    columns: dict[str, np.ndarray] = {}
    n_rows = len(cache.end_times)
    for site in built.sites:
        fit = g_fit.fits[site.site]
        theta = fit.theta[0].copy()
        profile = theta[layout.slices["profile"]].copy()
        profiles.append(centring @ profile)
        band_means.append(profile - centring @ profile)
        pinned = pinned_capacity(
            vintages=vintages, table=built.cache.sites[site.site].table, cut=common.TUNING_CUT
        )
        points.append(pinned.held_points)
        features = built.cache.sites[site.site]
        index = training_rows(
            features=features, cache=built.cache, cut=common.TUNING_CUT, window=common.FULL_WINDOW
        )
        predicted = predict_gam(cache=built.cache, fit=fit, index=index)
        series = np.full(n_rows, np.nan)
        series[index] = features.y[index] - predicted
        weeks = []
        for start in range(0, n_rows - WEEK_ROWS, WEEK_ROWS):
            block = series[start : start + WEEK_ROWS]
            if np.isfinite(block).mean() >= 0.9:
                weeks.append(np.nan_to_num(block, nan=0.0))
        bank[site.site] = weeks
        other = theta.copy()
        other[layout.slices["profile"]] = 0.0
        usable = np.flatnonzero(
            np.isfinite(features.pv_unit_shape)
            & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        )
        y = np.full(n_rows, np.nan)
        column = np.full(n_rows, -1, dtype=np.int16)
        for start in range(0, len(usable), 20000):
            rows = usable[start : start + 20000]
            x = design_matrix(cache=built.cache, features=features, layout=layout, index=rows)
            y[rows] = x @ other - pinned.held_points * derated_unit_shape(
                features=features, index=rows
            )
            column[rows] = _profile_column(x=x, layout_profile=layout.slices["profile"])
        base_y[site.site] = y
        columns[site.site] = column
    return Template(
        real_pool=labels,
        profiles=np.stack(profiles),
        band_means=np.stack(band_means),
        points=tuple(points),
        noise_bank=bank,
        base_y=base_y,
        profile_column=columns,
    )


_LOADED: dict[str, Any] = {}


def source() -> tuple[FeatureCache, pl.DataFrame]:
    """Return the real cache and vintages, loaded once per process."""
    if "source" not in _LOADED:
        _LOADED["source"] = arms.load_source()
    return _LOADED["source"]


def template() -> Template:
    """Return the template, built once per process."""
    if "template" not in _LOADED:
        cache, vintages = source()
        _LOADED["template"] = build_template(cache=cache, vintages=vintages)
    return _LOADED["template"]


def _bump_shape(*, rng: np.random.Generator) -> np.ndarray:
    """Return a centred unit shape made of a few Gaussian bumps, with a gain per day type."""
    half_hours = np.arange(common.HALF_HOURS_PER_DAY)
    day = np.zeros(common.HALF_HOURS_PER_DAY)
    for _ in range(3):
        centre = rng.uniform(4, 44)
        width = rng.uniform(2.0, 6.0)
        day += rng.uniform(-1.0, 1.0) * np.exp(-0.5 * ((half_hours - centre) / width) ** 2)
    gains = np.array([1.0, *rng.uniform(0.5, 1.0, size=3)])
    shape = np.concatenate([gain * day for gain in gains])
    shape = centring_matrix(family="profile", size=PROFILE_COLUMNS) @ shape
    return shape / np.linalg.norm(shape)


def truth_shapes(*, rng: np.random.Generator, overlapping: bool) -> np.ndarray:
    """Return K* truth shapes, shape (K*, 192), centred and unit norm.

    Args:
        rng: The random generator.
        overlapping: True for pairwise cosines of about 0.3 to 0.6: orthogonalised bumps plus a
            common direction.

    Returns:
        The shapes.
    """
    raw = np.stack([_bump_shape(rng=rng) for _ in range(TRUE_SHAPES + 1)])
    if not overlapping:
        return raw[:TRUE_SHAPES]
    basis, _ = np.linalg.qr(raw.T)
    own, common_direction = basis[:, :TRUE_SHAPES].T, basis[:, TRUE_SHAPES]
    target = rng.uniform(0.3, 0.6)
    mix = math.sqrt(target / (1.0 - target))
    shapes = own + mix * common_direction[None, :]
    return shapes / np.linalg.norm(shapes, axis=1, keepdims=True)


def truth_weights(*, rng: np.random.Generator, n_sites: int, sparse: bool) -> np.ndarray:
    """Return non-negative weights, shape (sites, K*), each row summing to one.

    Args:
        rng: The random generator.
        n_sites: The number of synthetic substations.
        sparse: True for Dirichlet 0.3 with one site near each vertex; false for Dirichlet 1.

    Returns:
        The weights.
    """
    alpha = 0.3 if sparse else 1.0
    weights = rng.dirichlet(np.full(TRUE_SHAPES, alpha), size=n_sites)
    if sparse:
        for site in range(min(n_sites, TRUE_SHAPES)):
            vertex = np.zeros(TRUE_SHAPES)
            vertex[site] = 1.0
            weights[site] = 0.85 * vertex + 0.15 * weights[site]
    return weights


def _smooth_noise_profile(*, rng: np.random.Generator) -> np.ndarray:
    """Return a centred smooth random profile of unit norm (a deviation or a null profile)."""
    raw = rng.normal(size=PROFILE_COLUMNS)
    kernel = np.ones(5) / 5.0
    smooth = np.concatenate(
        [
            np.convolve(raw[d * 48 : (d + 1) * 48], kernel, mode="same")
            for d in range(PROFILE_COLUMNS // 48)
        ]
    )
    smooth = centring_matrix(family="profile", size=PROFILE_COLUMNS) @ smooth
    return smooth / np.linalg.norm(smooth)


@dataclass(frozen=True)
class Fleet:
    """A synthetic fleet: its truth and its data.

    Attributes:
        cache: The synthetic cache (labels S1 to S5).
        vintages: Constant pinned-capacity vintages for the synthetic ids.
        shapes: The truth shapes, or `None` for a null fleet.
        weights: The truth weights in points, shape (sites, K*), or `None` for a null fleet.
        library: Each site's library part, shape (sites, 192).
        deviation: Each site's deviation part, shape (sites, 192).
        sites: The synthetic labels.
    """

    cache: FeatureCache
    vintages: pl.DataFrame
    shapes: np.ndarray | None
    weights: np.ndarray | None
    library: np.ndarray
    deviation: np.ndarray
    sites: tuple[str, ...]


def build_fleet(
    *,
    rng: np.random.Generator,
    kind: str,
    overlapping: bool,
    strength: float = 1.0,
    bump_site: int | None = None,
    sparse: bool = True,
    step_sites: tuple[int, ...] = (),
    noise_seed: int | None = None,
) -> Fleet:
    """Build one synthetic fleet on the template's real calendar, weather, and noise.

    Args:
        rng: The random generator; the draws are in a fixed order so a fleet is reproducible.
        kind: `library` for a fleet with a library, or `null` for independent random profiles.
        overlapping: True for truth shapes at pairwise cosines of 0.3 to 0.6.
        strength: A factor on the library and the deviation (the power fleets rescale it).
        bump_site: The index of the site that gets a site-specific bump, or `None`.
        sparse: True for the sparse weight prior.
        step_sites: The indices of sites that get the injected Economy 7-like step.
        noise_seed: A seed for the noise alone, so two fleets share a truth and differ in noise;
            `None` draws the noise from `rng`.

    Returns:
        The fleet.
    """
    cache, _ = source()
    real_data = template()
    n_sites = len(real_data.real_pool)
    labels = tuple(f"S{i + 1}" for i in range(n_sites))
    sizes = np.linalg.norm(real_data.profiles, axis=1)
    shapes = weights = None
    library = np.zeros((n_sites, PROFILE_COLUMNS))
    if kind == "library":
        shapes = truth_shapes(rng=rng, overlapping=overlapping)
        weights = truth_weights(rng=rng, n_sites=n_sites, sparse=sparse) * sizes[:, None]
        library = weights @ shapes
    else:
        for site in range(n_sites):
            library[site] = sizes[site] * _smooth_noise_profile(rng=rng)
    deviation = np.stack(
        [
            DEVIATION_ENERGY**0.5 * np.linalg.norm(library[site]) * _smooth_noise_profile(rng=rng)
            for site in range(n_sites)
        ]
    )
    if bump_site is not None:
        half_hours = np.arange(common.HALF_HOURS_PER_DAY)
        evening = np.exp(-0.5 * ((half_hours - 40) / 2.5) ** 2)
        bump = np.concatenate([evening] * (PROFILE_COLUMNS // common.HALF_HOURS_PER_DAY))
        bump = centring_matrix(family="profile", size=PROFILE_COLUMNS) @ bump
        bump = bump / np.linalg.norm(bump) * (BUMP_ENERGY**0.5) * np.linalg.norm(library[bump_site])
        deviation[bump_site] = bump
    step = np.zeros((n_sites, PROFILE_COLUMNS))
    for site in step_sites:
        for day_type in range(PROFILE_COLUMNS // common.HALF_HOURS_PER_DAY):
            first = day_type * common.HALF_HOURS_PER_DAY + STEP_FIRST_COLUMN
            step[site, first : first + STEP_LAST_COLUMN - STEP_FIRST_COLUMN + 1] = STEP_POINTS
    noise_rng = rng if noise_seed is None else np.random.default_rng(noise_seed)
    n_rows = len(cache.end_times)
    donors_all = list(real_data.noise_bank)
    sites, vintage_rows = {}, []
    weeks = pl.date_range(date(2019, 9, 2), date(2026, 6, 29), "1w", eager=True)
    for number, label in enumerate(labels):
        real = real_data.real_pool[number]
        base = cache.sites[real]
        points = real_data.points[number]
        table = replace(base.table, site=label, time_series_id=9000 + number)
        profile = (
            real_data.band_means[number]
            + strength * (library[number] + deviation[number])
            + step[number]
        )
        column = real_data.profile_column[real]
        y = real_data.base_y[real] + np.where(column >= 0, profile[np.maximum(column, 0)], np.nan)
        donors = [d for d in donors_all if d != real]
        for start in range(0, n_rows, WEEK_ROWS):
            donor = real_data.noise_bank[donors[int(noise_rng.integers(len(donors)))]]
            block = donor[int(noise_rng.integers(len(donor)))]
            stop = min(start + WEEK_ROWS, n_rows)
            y[start:stop] = y[start:stop] + block[: stop - start]
        sites[label] = replace(
            base,
            table=table,
            y=y,
            present=np.ones(n_rows, dtype=bool),
            switching_mask=np.zeros(n_rows, dtype=bool),
            in_pool=np.ones(n_rows, dtype=bool),
            default_rows=cache.sites["D11"].default_rows.copy(),
        )
        vintage_rows.append(
            pl.DataFrame(
                {
                    "time_series_id": 9000 + number,
                    "variant": "frozen",
                    "origin_monday": date(2019, 9, 2),
                    "week_start": weeks,
                    "capacity_mw": points * table.effective_capacity_mw / 100.0,
                    "unobserved": False,
                }
            ).with_columns(pl.col("time_series_id").cast(pl.Int32))
        )
    synthetic = replace(
        cache, key="gate", sites=sites, node_ids={label: (label,) for label in labels}
    )
    return Fleet(
        cache=synthetic,
        vintages=pl.concat(vintage_rows),
        shapes=shapes,
        weights=weights,
        library=strength * library,
        deviation=strength * deviation,
        sites=labels,
    )


def held_out_score(*, fleet: Fleet, fits: dict[str, GamFit], built: arms.Built) -> float:
    """Return the mean held-out quantile score of a fit on the year after the tuning cut.

    The spread is Gaussian with the fit's own training-residual standard deviation, the same recipe
    for every model, because a gate fleet has no replica year.

    Args:
        fleet: The fleet.
        fits: Each site's fit.
        built: The fit's substations and cache.

    Returns:
        The mean quantile score over sites, in points.
    """
    z = norm.ppf(np.array(QUANTILE_LEVELS))
    scores = []
    for label in fleet.sites:
        features = fleet.cache.sites[label]
        train = training_rows(
            features=features, cache=built.cache, cut=common.TUNING_CUT, window=common.FULL_WINDOW
        )
        sigma = float(
            np.std(features.y[train] - predict_gam(cache=built.cache, fit=fits[label], index=train))
        )
        index = common.scoring_index(
            cache=fleet.cache,
            site=label,
            after=common.TUNING_CUT,
            through=common.year_end(cut=common.TUNING_CUT),
        )
        mean = predict_gam(cache=built.cache, fit=fits[label], index=index)
        quantiles = mean[:, None] + sigma * z[None, :]
        scores.append(
            float(common.quantile_score(actual=features.y[index], quantiles=quantiles).mean())
        )
    return float(np.mean(scores))


def training_sse(*, fleet: Fleet, fits: dict[str, GamFit], built: arms.Built) -> float:
    """Return the training sum of squared errors over the fleet's sites."""
    total = 0.0
    for label in fleet.sites:
        features = fleet.cache.sites[label]
        train = training_rows(
            features=features, cache=built.cache, cut=common.TUNING_CUT, window=common.FULL_WINDOW
        )
        residual = features.y[train] - predict_gam(cache=built.cache, fit=fits[label], index=train)
        total += float(residual @ residual)
    return total


def oracle_score(*, fleet: Fleet) -> float:
    """Return the held-out score of the true profile (noise-only residual, Gaussian spread)."""
    real_data = template()
    z = norm.ppf(np.array(QUANTILE_LEVELS))
    scores = []
    for number, label in enumerate(fleet.sites):
        features = fleet.cache.sites[label]
        real = real_data.real_pool[number]
        profile = real_data.band_means[number] + fleet.library[number] + fleet.deviation[number]
        column = real_data.profile_column[real]
        truth = real_data.base_y[real] + np.where(
            column >= 0, profile[np.maximum(column, 0)], np.nan
        )
        index = common.scoring_index(
            cache=fleet.cache,
            site=label,
            after=common.TUNING_CUT,
            through=common.year_end(cut=common.TUNING_CUT),
        )
        sigma = float(np.std(features.y[index] - truth[index]))
        scores.append(
            float(
                common.quantile_score(
                    actual=features.y[index], quantiles=truth[index, None] + sigma * z[None, :]
                ).mean()
            )
        )
    return float(np.mean(scores))


@dataclass(frozen=True)
class FitOutcome:
    """A library fit with its held-out score and training sum of squares."""

    library: ShapeLibraryFit
    score: float
    sse: float


def fit_sh(
    *,
    fleet: Fleet,
    k: int,
    tuned: dict[str, Any],
    gamma: float | None = None,
    init: tuple[int, InitialShapesType] = START_METHODS[0],
) -> FitOutcome:
    """Fit the library on a fleet at the tuning cut and score it on the year after.

    Args:
        fleet: The fleet.
        k: The number of shapes.
        tuned: The tuned settings from `work/tuned.json`.
        gamma: The orthogonality weight; the tuned one if `None`.
        init: The start's seed and method.

    Returns:
        The fit, its held-out score, and its training sum of squares.
    """
    built = arms.build_pool(
        cache=fleet.cache, vintages=fleet.vintages, cut=common.TUNING_CUT, pool=fleet.sites
    )
    config = LibraryConfig(
        n_shapes=k,
        deviation_weight=tuned["lambda_dev"],
        orthogonality=tuned["gamma"] if gamma is None else gamma,
        other_strength=common.PRODUCTION_LAMBDA,
    )
    library = fit_library(
        cache=built.cache,
        sites=built.sites,
        cut=common.TUNING_CUT,
        config=config,
        init_method=init[1],
        seed=init[0],
    )
    return FitOutcome(
        library=library,
        score=held_out_score(fleet=fleet, fits=library.fits, built=built),
        sse=training_sse(fleet=fleet, fits=library.fits, built=built),
    )


def fit_reference(*, fleet: Fleet, strength: float) -> tuple[float, dict[str, GamFit], arms.Built]:
    """Fit G (`strength` 0) or N (`strength` 3,000 rows) on a fleet; return its held-out score."""
    built = arms.build_pool(
        cache=fleet.cache, vintages=fleet.vintages, cut=common.TUNING_CUT, pool=fleet.sites
    )
    joint = fit_pooled(
        cache=built.cache,
        sites=built.sites,
        cut=common.TUNING_CUT,
        strengths=[PoolingStrength.uniform(value=strength)],
    )
    return held_out_score(fleet=fleet, fits=joint.fits, built=built), joint.fits, built


def matched(*, outcome: FitOutcome, fleet: Fleet) -> tuple[np.ndarray, np.ndarray]:
    """Return the matched cosines and the matched weights of a K* fit.

    Args:
        outcome: A fit with K = K*.
        fleet: The fleet, whose truth shapes and weights are not `None`.

    Returns:
        The signed cosine of each truth shape to its matched fitted shape, and the fitted weights
        of the substations in the truth's shape order, shape (sites, K*).
    """
    assert fleet.shapes is not None
    order, cosines = match_shapes(first=fleet.shapes, second=outcome.library.shapes)
    fitted = np.stack([outcome.library.weights[label] for label in fleet.sites])[:, order]
    return cosines, fitted


def chosen_k(*, scores: dict[int, float]) -> int:
    """Return the tuning rule's K: the smallest K within the tie margin of the best score."""
    best = min(scores.values())
    return min(k for k, score in scores.items() if score <= best + K_TIE_MARGIN)


def sparse_fleet_job(*, rng: np.random.Generator, overlapping: bool, tuned: dict[str, Any]) -> dict:
    """Run lines 1, 2, 3, 4, 6, and 7 on one sparse-prior fleet."""
    fleet_seed = int(rng.integers(2**31))
    fleet = build_fleet(
        rng=np.random.default_rng(fleet_seed), kind="library", overlapping=overlapping
    )
    starts = [
        fit_sh(fleet=fleet, k=TRUE_SHAPES, tuned=tuned, init=start) for start in START_METHODS
    ]
    best_sse = min(outcome.sse for outcome in starts)
    near_best = sum(
        1 for outcome in starts if outcome.sse <= best_sse * (1.0 + LOCAL_OPTIMUM_SHARE)
    )
    headline = min(starts, key=lambda outcome: outcome.library.objective_trace[-1])
    cosines, fitted = matched(outcome=headline, fleet=fleet)
    assert fleet.weights is not None
    totals = fleet.weights.sum(axis=1, keepdims=True)
    weight_error = float(np.median(np.abs(fitted - fleet.weights) / totals))
    k4 = fit_sh(fleet=fleet, k=TRUE_SHAPES + 1, tuned=tuned)
    k4_g0 = fit_sh(fleet=fleet, k=TRUE_SHAPES + 1, tuned=tuned, gamma=0.0)
    scores = {TRUE_SHAPES: headline.score, TRUE_SHAPES + 1: k4.score}
    for k in (2, 6):
        try:
            scores[k] = fit_sh(fleet=fleet, k=k, tuned=tuned).score
        except torch.linalg.LinAlgError:
            # More shapes than the pool can use leaves a dead shape and a singular weights step;
            # a library that cannot be fitted is never the tuning rule's choice.
            scores[k] = float("inf")
    bump_site = int(rng.integers(len(fleet.sites)))
    bump_fleet = build_fleet(
        rng=np.random.default_rng(fleet_seed),
        kind="library",
        overlapping=overlapping,
        bump_site=bump_site,
    )
    bump_fit = fit_sh(fleet=bump_fleet, k=TRUE_SHAPES, tuned=tuned)
    bump_cosines, _ = matched(outcome=bump_fit, fleet=bump_fleet)
    bump_vector = bump_fleet.deviation[bump_site]
    deviation = bump_fit.library.deviations[bump_fleet.sites[bump_site]]
    return {
        "line1_min_cosine": float(cosines.min()),
        "line2_weight_error": weight_error,
        "line3_max_pair_cosine": largest_pair_cosine(shapes=k4.library.shapes),
        "line3_gamma0_max_pair_cosine": largest_pair_cosine(shapes=k4_g0.library.shapes),
        "line4_min_cosine": float(bump_cosines.min()),
        "line4_deviation_share": float(deviation @ bump_vector / (bump_vector @ bump_vector)),
        "line6_scores": {str(k): v for k, v in scores.items()},
        "line6_chosen_k": chosen_k(scores=scores),
        "line7_near_best": int(near_best),
        "converged": [outcome.library.converged for outcome in starts],
        **step_job(fleet_seed=fleet_seed, overlapping=overlapping, tuned=tuned),
    }


def fitted_profile(*, outcome: FitOutcome, label: str) -> np.ndarray:
    """Return a site's fitted daily profile (band means included), shape (192,)."""
    return outcome.library.fits[label].theta[0, :PROFILE_COLUMNS]


def step_job(*, fleet_seed: int, overlapping: bool, tuned: dict[str, Any]) -> dict:
    """Run line 10: can the library find an injected Economy 7-like step without being told?

    Three fleets share one truth: with the step (noise A), without it (noise A), and without it
    (noise B, the placebo). The step estimate at a site is the difference between the fitted
    profiles of the stepped and the unstepped fleets, so the fits differ only by the step; the
    placebo estimate is the difference between the two unstepped fits, which differ only by noise.

    Args:
        fleet_seed: The seed of the shared truth.
        overlapping: True for truth shapes at cosines of 0.3 to 0.6.
        tuned: The tuned settings.

    Returns:
        The injected sites' time error and size error, and the largest placebo step.
    """
    noise_a, noise_b = fleet_seed + 1, fleet_seed + 2
    stepped = build_fleet(
        rng=np.random.default_rng(fleet_seed),
        kind="library",
        overlapping=overlapping,
        step_sites=STEP_SITES,
        noise_seed=noise_a,
    )
    plain = build_fleet(
        rng=np.random.default_rng(fleet_seed),
        kind="library",
        overlapping=overlapping,
        noise_seed=noise_a,
    )
    placebo = build_fleet(
        rng=np.random.default_rng(fleet_seed),
        kind="library",
        overlapping=overlapping,
        noise_seed=noise_b,
    )
    k = TRUE_SHAPES + 1
    fit_stepped = fit_sh(fleet=stepped, k=k, tuned=tuned)
    fit_plain = fit_sh(fleet=plain, k=k, tuned=tuned)
    fit_placebo = fit_sh(fleet=placebo, k=k, tuned=tuned)
    window = slice(STEP_FIRST_COLUMN, STEP_LAST_COLUMN + 1)
    time_errors, size_errors, placebo_steps = [], [], []
    for position, label in enumerate(stepped.sites):
        found = fitted_profile(outcome=fit_stepped, label=label) - fitted_profile(
            outcome=fit_plain, label=label
        )
        false_step = fitted_profile(outcome=fit_placebo, label=label) - fitted_profile(
            outcome=fit_plain, label=label
        )
        placebo_steps.append(float(abs(false_step[window].mean())))
        if position in STEP_SITES:
            weekday = found[: common.HALF_HOURS_PER_DAY]
            rise_column = int(np.argmax(np.diff(weekday))) + 1
            time_errors.append(abs(rise_column - STEP_FIRST_COLUMN))
            size_errors.append(abs(float(found[window].mean()) - STEP_POINTS) / STEP_POINTS)
    return {
        "line10_max_time_error": int(max(time_errors)),
        "line10_max_size_error": float(max(size_errors)),
        "line10_placebo_step": float(max(placebo_steps)),
    }


def null_fleet_job(*, rng: np.random.Generator, tuned: dict[str, Any]) -> dict:
    """Run line 5 on one null fleet (independent random profiles)."""
    fleet = build_fleet(rng=rng, kind="null", overlapping=False)
    first = fit_sh(fleet=fleet, k=TRUE_SHAPES, tuned=tuned, init=START_METHODS[0])
    second = fit_sh(fleet=fleet, k=TRUE_SHAPES, tuned=tuned, init=START_METHODS[1])
    g_score, _, _ = fit_reference(fleet=fleet, strength=0.0)
    _, cosines = match_shapes(first=first.library.shapes, second=second.library.shapes)
    best = min(first.score, second.score)
    return {
        "line5_sh_minus_g": best - g_score,
        "line5_median_seed_cosine": float(np.median(cosines)),
    }


def power_fleet_job(*, rng: np.random.Generator, overlapping: bool, tuned: dict[str, Any]) -> dict:
    """Run line 9 on one fleet whose library is scaled to an oracle gain of 0.10 over N."""
    seed_state = int(rng.integers(2**31))
    strength = 1.0
    for attempt in range(4):
        fleet = build_fleet(
            rng=np.random.default_rng(seed_state),
            kind="library",
            overlapping=overlapping,
            strength=strength,
        )
        n_score, _, _ = fit_reference(fleet=fleet, strength=common.PRODUCTION_LAMBDA)
        gain = n_score - oracle_score(fleet=fleet)
        if abs(gain - POWER_GAIN) < 0.01 or attempt == 3:
            break
        strength *= math.sqrt(POWER_GAIN / max(gain, 1e-3))
    starts = [
        fit_sh(fleet=fleet, k=TRUE_SHAPES, tuned=tuned, init=START_METHODS[i]) for i in (0, 1)
    ]
    best = min(starts, key=lambda outcome: outcome.library.objective_trace[-1])
    return {
        "line9_oracle_gain": gain,
        "line9_strength": strength,
        "line9_sh_minus_n": best.score - n_score,
    }


def dense_fleet_job(*, rng: np.random.Generator, tuned: dict[str, Any]) -> dict:
    """Report lines 1 and 2 on one dense-prior fleet (identification-limited, not blocking)."""
    fleet = build_fleet(rng=rng, kind="library", overlapping=False, sparse=False)
    outcome = fit_sh(fleet=fleet, k=TRUE_SHAPES, tuned=tuned)
    cosines, fitted = matched(outcome=outcome, fleet=fleet)
    assert fleet.weights is not None
    totals = fleet.weights.sum(axis=1, keepdims=True)
    return {
        "line1_min_cosine": float(cosines.min()),
        "line2_weight_error": float(np.median(np.abs(fitted - fleet.weights) / totals)),
    }


@dataclass(frozen=True)
class GateJob:
    """One fleet of one set."""

    kind: str
    set_index: int
    fleet_index: int


def job_path(*, job: GateJob) -> Path:
    """Return where a job's result is stored."""
    return GATE_DIR / f"{job.kind}_{job.set_index}_{job.fleet_index:02d}.json"


def run_job(job: GateJob) -> str:
    """Run one fleet's fits and store its result atomically; skip a stored one."""
    torch.set_num_threads(1)
    path = job_path(job=job)
    if path.exists():
        return f"{path.name}: stored"
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    rng = np.random.default_rng([SET_SEEDS[job.set_index], job.fleet_index, hash_kind(job.kind)])
    overlapping = job.fleet_index % 2 == 1
    try:
        if job.kind == "sparse":
            result = sparse_fleet_job(rng=rng, overlapping=overlapping, tuned=tuned)
        elif job.kind == "null":
            result = null_fleet_job(rng=rng, tuned=tuned)
        elif job.kind == "power":
            result = power_fleet_job(rng=rng, overlapping=overlapping, tuned=tuned)
        else:
            result = dense_fleet_job(rng=rng, tuned=tuned)
    except torch.linalg.LinAlgError as error:
        # A fleet whose fit is singular is reported as missing, so the line's denominator shrinks
        # visibly instead of the whole gate stopping.
        return f"{path.name}: failed, {error}"
    result |= {
        "kind": job.kind,
        "set": job.set_index,
        "fleet": job.fleet_index,
        "overlapping": overlapping,
    }
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)
    return f"{path.name}: done"


def hash_kind(kind: str) -> int:
    """Return a small stable integer for a job kind, so each kind draws its own fleets."""
    return {"sparse": 1, "null": 2, "power": 3, "dense": 4}[kind]


def all_jobs() -> list[GateJob]:
    """Return every job: sparse, null, and power fleets for both sets, then the dense fleets."""
    kinds = ("sparse", "null", "power", "dense")
    return [
        GateJob(kind, s, f) for kind in kinds for s in range(N_SETS) for f in range(FLEETS_PER_SET)
    ]


def wilson(*, passed: int, n: int, z: float = 1.645) -> tuple[float, float]:
    """Return the Wilson score interval (90% by default) for a rate."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def module_tests_pass() -> bool:
    """Run the module's unit tests from the repository root and return True if they pass."""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "pytest",
            "packages/studies/tests/test_dp_shape_library.py",
            "--run-studies",
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=root,
    )
    return result.returncode == 0


def load_results() -> pl.DataFrame:
    """Return every stored job result as a frame."""
    rows = [json.loads(path.read_text()) for path in sorted(GATE_DIR.glob("*_*_*.json"))]
    return pl.DataFrame(rows, infer_schema_length=None)


def report() -> dict[str, Any]:
    """Print every line's pass count and Wilson interval, and return the lines' outcomes."""
    table = load_results()
    sparse = table.filter(pl.col("kind") == "sparse")
    null = table.filter(pl.col("kind") == "null")
    power = table.filter(pl.col("kind") == "power")
    dense = table.filter(pl.col("kind") == "dense")
    lines: dict[str, tuple[int, int]] = {}

    def count(name: str, frame: pl.DataFrame, condition: pl.Expr) -> None:
        lines[name] = (int(frame.filter(condition).height), frame.height)

    count("1 shapes (min cosine >= 0.9)", sparse, pl.col("line1_min_cosine") >= SHAPE_COSINE)
    count("2 weights (error <= 0.15)", sparse, pl.col("line2_weight_error") <= WEIGHT_ERROR)
    count(
        "3 no duplicates (max |cos| < 0.5)",
        sparse,
        pl.col("line3_max_pair_cosine") < DUPLICATE_COSINE,
    )
    count(
        "4 deviation (cosines >= 0.9 and share >= 0.6)",
        sparse,
        (pl.col("line4_min_cosine") >= SHAPE_COSINE)
        & (pl.col("line4_deviation_share") >= DEVIATION_SHARE),
    )
    count(
        "10 step recovery (time within 1 half-hour, size within 25%, placebo under 25%)",
        sparse,
        (pl.col("line10_max_time_error") <= 1)
        & (pl.col("line10_max_size_error") <= STEP_SIZE_TOLERANCE)
        & (pl.col("line10_placebo_step") < STEP_SIZE_TOLERANCE * STEP_POINTS),
    )
    count(
        "5 null fleet (SH - G >= 0, seed cosine < 0.6)",
        null,
        (pl.col("line5_sh_minus_g") >= 0.0)
        & (pl.col("line5_median_seed_cosine") < NULL_SEED_COSINE),
    )
    count("6 K choice (3 or 4)", sparse, pl.col("line6_chosen_k").is_in([3, 4]))
    count("7 local optima (4 of 5 within 1%)", sparse, pl.col("line7_near_best") >= 4)
    count("9 score power (SH - N <= -0.05)", power, pl.col("line9_sh_minus_n") <= POWER_REQUIRED)
    tests = module_tests_pass()
    lines["8 module tests"] = (int(tests), 1)
    outcome = {}
    print("line | passed / fleets | Wilson 90% | gate")
    for name, (passed, n) in lines.items():
        low, high = wilson(passed=passed, n=n)
        ok = tests if name.startswith("8") else passed >= LINE_PASS_COUNT
        outcome[name] = ok
        print(f"{name} | {passed} / {n} | {low:.2f} to {high:.2f} | {'PASS' if ok else 'FAIL'}")
    print("\nReported, not blocking:")
    print(
        "  line 3 with gamma = 0: max |cos| median",
        float(np.median(sparse["line3_gamma0_max_pair_cosine"].to_numpy())),
        "; tuned gamma median",
        float(np.median(sparse["line3_max_pair_cosine"].to_numpy())),
    )
    print(
        "  dense prior: min cosine median",
        float(np.median(dense["line1_min_cosine"].to_numpy())),
        "weight error median",
        float(np.median(dense["line2_weight_error"].to_numpy())),
    )
    print(
        "  power fleets: achieved oracle gain median",
        float(np.median(power["line9_oracle_gain"].to_numpy())),
        "SH - N median",
        float(np.median(power["line9_sh_minus_n"].to_numpy())),
    )
    print("  null fleets: SH - G median", float(np.median(null["line5_sh_minus_g"].to_numpy())))
    gate_passed = all(outcome.values())
    print("\nGATE:", "PASS" if gate_passed else "FAIL (caps the verdict at inconclusive)")
    result = {"lines": {k: list(v) for k, v in lines.items()}, "passed": gate_passed}
    (common.WORK_DIR / "gate_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Run the gate's fleets in worker processes, or print the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "report"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", nargs="*", default=None, help="job kinds to run")
    arguments = parser.parse_args()
    if arguments.command == "report":
        report()
        return
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    jobs: Sequence[GateJob] = [
        job for job in all_jobs() if arguments.only is None or job.kind in arguments.only
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(run_job, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
