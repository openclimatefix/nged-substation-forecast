"""The arms of experiment 129, and the fits and bootstrap weights they share.

Arm 2 is the monotone season-year capacity (experiment 110's `k_base`). Arm 3 adds experiment
110's sun-blind terms (`k_ii`). Arm 4 is arm 2 with the penalty on the capacity replaced by a
penalty on the capacity's increments, except the first, so the level is not shrunk while noise
increments are.
"""

from dataclasses import replace
from datetime import datetime
from typing import Final, Literal

import e00129_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    DEFAULT_MAX_ITERATIONS,
    GamFit,
    GamSpec,
    accumulate_gram,
    fit_gam,
    last_signal_season_year,
    penalty_matrix,
    prepare_fit,
    solve_bounded_quadratic,
    training_rows,
)

ArmType = Literal["arm2", "arm3", "arm4"]
"""Arm 2: monotone capacity. Arm 3: arm 2 with sun-blind demand. Arm 4: arm 2 with a ridge on the
increments."""
ARMS: Final[tuple[ArmType, ...]] = ("arm2", "arm3", "arm4")
FIRST_INCREMENT_RIDGE_SHARE: Final[float] = 1e-6
"""The first increment's ridge, as a share of the other increments', only to keep it well posed."""


def arm_spec(*, arm: ArmType, ridge: float = common.RIDGE) -> GamSpec:
    """Return the penalties and terms of an arm's fit."""
    fitted = GamSpec(ridge=ridge, capacity="fitted")
    if arm == "arm3":
        return replace(fitted, shared_harmonics=True, slow_heating_only=True)
    return fitted


def training_index(
    *, cache: FeatureCache, site: str, cut: datetime = common.DATA_CUT
) -> np.ndarray:
    """Return the grid rows every fit of the substation trains on, before `cut`."""
    return training_rows(
        features=cache.sites[site],
        cache=cache,
        cut=cut,
        window=common.TRAINING_WINDOW,
    )


def stratified_month_weights(
    *, cache: FeatureCache, index: np.ndarray, n_resamples: int, seed: int
) -> np.ndarray:
    """Return month-block bootstrap weights that keep each calendar quarter's months.

    Each resample draws, within each (season-year, quarter of the year) stratum, as many months as
    the stratum has in the window, with replacement, so that no season-year loses its summer.

    Args:
        cache: The cache.
        index: The training rows.
        n_resamples: The number of resamples.
        seed: Seeds the draws.

    Returns:
        Weights of shape (n_resamples, len(index)): how many times each row's month was drawn.
    """
    month = cache.end_times[index].astype("datetime64[M]")
    months, inverse = np.unique(month, return_inverse=True)
    month_number = months.astype(np.int64)
    quarter = (month_number % 12) // 3
    season_year = (month_number + 4) // 12  # September starts a season-year
    stratum = season_year * 4 + quarter
    generator = np.random.default_rng(seed)
    counts = np.zeros((n_resamples, len(months)))
    for resample in range(n_resamples):
        for value in np.unique(stratum):
            members = np.flatnonzero(stratum == value)
            drawn = generator.choice(members, size=len(members), replace=True)
            counts[resample] += np.bincount(drawn, minlength=len(months))
    return counts[:, inverse]


def fit_arm(
    *,
    arm: ArmType,
    cache: FeatureCache,
    site: str,
    device: str,
    increment_share: float = 0.0,
    weights: np.ndarray | None = None,
    cut: datetime = common.DATA_CUT,
) -> GamFit:
    """Fit an arm at `DATA_CUT` on every usable row, once or under row weightings.

    Args:
        arm: The arm.
        cache: The cache.
        site: The substation's label.
        device: Where the arithmetic runs.
        increment_share: For arm 4, the increment penalty as a share of the mean diagonal of the
            Gram matrix over the capacity columns; ignored by the other arms.
        weights: Optional row weights of shape (batch, n_training_rows), for the bootstrap.
        cut: The fit uses rows before this instant.

    Returns:
        The fitted batch.
    """
    spec = arm_spec(arm=arm)
    if arm != "arm4":
        return fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=common.TRAINING_WINDOW,
            season_years=common.SEASON_YEARS,
            weights=weights,
            device=device,
            dtype=torch.float64,
        )
    return _fit_with_increment_penalty(
        cache=cache,
        site=site,
        spec=spec,
        increment_share=increment_share,
        weights=weights,
        device=device,
        cut=cut,
    )


def _fit_with_increment_penalty(
    *,
    cache: FeatureCache,
    site: str,
    spec: GamSpec,
    increment_share: float,
    weights: np.ndarray | None,
    device: str,
    cut: datetime,
) -> GamFit:
    """Fit the convex model with a ridge on the capacity increments instead of the capacities."""
    layout, index = prepare_fit(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=common.TRAINING_WINDOW,
        season_years=common.SEASON_YEARS,
    )
    gram, rhs = accumulate_gram(
        cache=cache,
        features=cache.sites[site],
        layout=layout,
        index=index,
        weights=weights,
        device=device,
        dtype=torch.float64,
        pinned=None,
    )
    matrix, cost = penalty_matrix(layout=layout, n_rows=len(index))
    capacity = layout.slices["capacity"]
    diagonal = torch.diagonal(gram[0])[capacity].mean().item()
    ridge = np.full(capacity.stop - capacity.start, increment_share * diagonal)
    ridge[0] *= FIRST_INCREMENT_RIDGE_SHARE
    matrix[capacity, capacity] = np.diag(ridge)
    batch = gram.shape[0]
    penalty = torch.as_tensor(matrix, dtype=torch.float64, device=device)[None].expand(
        batch, -1, -1
    )
    linear = torch.as_tensor(cost, dtype=torch.float64, device=device)[None].expand(batch, -1)
    theta, violation = solve_bounded_quadratic(
        matrix=gram + penalty,
        rhs=rhs - 0.5 * linear,
        constrained=torch.as_tensor(layout.constrained),
        max_iterations=DEFAULT_MAX_ITERATIONS,
    )
    return GamFit(
        layout=layout,
        theta=theta.cpu().numpy(),
        cut=cut,
        n_rows=len(index),
        site=site,
        last_season_year=last_signal_season_year(
            cache=cache, site=site, index=index, cut=cut, pinned=None
        ),
        violation=violation.cpu().numpy(),
        pinned=None,
    )


def capacities(*, fit: GamFit) -> list[dict[int, float]]:
    """Return the fitted capacity of each season-year for every member of a batch."""
    return [fit.capacity_points(member=member) for member in range(fit.theta.shape[0])]


def fitted_capacity(
    *,
    arm: ArmType,
    cache: FeatureCache,
    site: str,
    device: str,
    increment_share: float = 0.0,
    cut: datetime = common.DATA_CUT,
) -> dict[int, float]:
    """Return an arm's fitted capacity of each season-year, in points of effective capacity."""
    return fit_arm(
        arm=arm, cache=cache, site=site, device=device, increment_share=increment_share, cut=cut
    ).capacity_points()
