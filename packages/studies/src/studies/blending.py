"""Combine several weather products: a climatology control, and a cross-fitted linear stack.

**A model shown more columns can win without being shown more information**, so every blend of
weather products is compared against a control arm whose extra columns keep each product's
climatology and lose its weather. `climatology_permutation` builds those columns by permuting each
product's values among the rows that share a site, a month, and an hour of day.

**A linear stack combines the single-product models' out-of-fold predictions with non-negative
weights that sum to 1.** With weights summing to 1, the stacked prediction's error is the same
weighted sum of the single-product models' errors, so `stacked_errors` fits and scores the stack
from the saved signed errors with no refit. Each generator, seed and scored fold gets its own
weights, fitted on the same generator's other folds, so neither the scored fold's target nor a
neighbouring generator's target reaches them.
"""

from collections.abc import Sequence
from typing import Final, NamedTuple

import numpy as np
import polars as pl
from scipy.optimize import nnls

PERMUTED_SUFFIX: Final[str] = "_shuffled"
"""Appended to a column's name to name its permuted copy."""

SUM_TO_ONE_WEIGHT: Final[float] = 1.0e4
"""How heavily the sum-to-one row weighs in the least-squares fit, relative to the errors' norm.

`nnls` takes no equality constraint, so the constraint enters as one extra row whose residual is
this factor times the errors' norm times the weights' departure from summing to 1. At this weight
the fitted sum misses 1 by about one part in 10⁸, and the weights are renormalised afterwards.
"""


def climatology_permutation(
    *,
    frame: pl.DataFrame,
    column_groups: Sequence[Sequence[str]],
    by: Sequence[str],
    seed: int,
    suffix: str = PERMUTED_SUFFIX,
) -> pl.DataFrame:
    """Add a copy of each column with its values permuted among the rows sharing the `by` columns.

    **Each group's columns move together**, under one row permutation per group: a product's
    direction sine and cosine stay on the unit circle, and its hub-height and 10 m speeds stay
    paired. Group `i` is permuted under seed `seed + i`, so two products never share a permutation,
    and a single group of one column is permuted exactly as `pl.col(column).shuffle(seed)` over the
    `by` columns permutes it.

    Grouping by site, month and hour of day keeps each month's mean at each hour of day and removes
    the hour-to-hour weather. Where the folds are whole months, no permuted value crosses a fold.

    Args:
        frame: The rows, carrying every column named in `column_groups` and `by`.
        column_groups: The columns to permute, one group per permutation.
        by: The columns whose shared values define the rows a value may move between.
        seed: The first group's permutation seed.
        suffix: Appended to each column's name to name its permuted copy, so two sets of groups
            sharing a column can each be permuted without one overwriting the other.

    Returns:
        `frame`, in its own row order, with `<column><suffix>` for every column in `column_groups`.
    """
    row = "__climatology_permutation_row"
    indexed = frame.with_columns(pl.int_range(pl.len(), dtype=pl.UInt32).alias(row))
    permuted = indexed.with_columns(
        pl.col(row).shuffle(seed=seed + index).over(list(by)).alias(f"{row}_{index}")
        for index in range(len(column_groups))
    )
    return permuted.with_columns(
        pl.col(column).gather(pl.col(f"{row}_{index}")).alias(f"{column}{suffix}")
        for index, group in enumerate(column_groups)
        for column in group
    ).drop(row, *(f"{row}_{index}" for index in range(len(column_groups))))


def simplex_weights(*, errors: np.ndarray) -> np.ndarray:
    """Return the non-negative weights summing to 1 that minimise the squared weighted-sum error.

    Args:
        errors: One row per observation and one column per model, each entry that model's signed
            error.

    Returns:
        One weight per column, each at least 0, summing to exactly 1 up to floating-point rounding.
    """
    n_models = errors.shape[1]
    scale = float(np.sqrt(np.square(errors).sum()))
    penalty = SUM_TO_ONE_WEIGHT * (scale if scale > 0.0 else 1.0)
    design = np.vstack([errors, np.full((1, n_models), penalty)])
    target = np.concatenate([np.zeros(errors.shape[0]), [penalty]])
    weights, _ = nnls(design, target)
    return weights / weights.sum()


class StackedErrors(NamedTuple):
    """A stack's error on every row, and the weights each row was scored with."""

    errors: np.ndarray
    """The weighted sum of each row's errors, shape (n_rows,)."""

    weights: np.ndarray
    """The weights applied to each row, shape (n_rows, n_models)."""


def stacked_errors(
    *,
    errors: np.ndarray,
    sites: np.ndarray,
    folds: np.ndarray,
    seeds: np.ndarray,
    fit_rows: np.ndarray,
    cross_fitted: bool = True,
) -> StackedErrors:
    """Stack several models per generator, seed and fold, fitting on the generator's other folds.

    For each (site, seed, fold), the weights come from `simplex_weights` on that site's and seed's
    `fit_rows` in every other fold, and are applied to every row of the fold, `fit_rows` or not. The
    fit rows are the hours the single-product models trained on; an hour the network operator
    curtailed is scored but not fitted, as in `studies.cross_validation`.

    Args:
        errors: One row per (site, time, seed) and one column per model, each entry that model's
            signed error.
        sites: Each row's site.
        folds: Each row's fold.
        seeds: Each row's seed. Each seed is stacked from its own models' errors only.
        fit_rows: Whether each row may be used to fit the weights.
        cross_fitted: Whether to leave the scored fold out of its own fit. Fitting on every fold,
            the scored one included, measures how optimistic the in-sample weights are.

    Returns:
        The stacked error on every row, and the weights it was scored with.

    Raises:
        ValueError: If a fold has no fit rows to fit its weights on.
    """
    stacked = np.empty(errors.shape[0])
    weights = np.empty(errors.shape)
    for site, seed in {(site, seed) for site, seed in zip(sites, seeds, strict=True)}:
        group = (sites == site) & (seeds == seed)
        for fold in np.unique(folds[group]):
            scored = group & (folds == fold)
            fitted = group & fit_rows
            if cross_fitted:
                fitted &= folds != fold
            if not fitted.any():
                msg = f"site {site}, seed {seed}, fold {fold} has no rows to fit its weights on"
                raise ValueError(msg)
            fold_weights = simplex_weights(errors=errors[fitted])
            weights[scored] = fold_weights
            stacked[scored] = errors[scored] @ fold_weights
    return StackedErrors(errors=stacked, weights=weights)
