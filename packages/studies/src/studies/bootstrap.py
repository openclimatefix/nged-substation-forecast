"""Interval a paired difference between two arms by resampling whole months and a seed.

Six meters inside a box roughly 25 km by 23 km share their weather, and in fact resolve to only two
ERA5 grid cells, so the effective sample size is the number of independent weather episodes rather
than the number of site-hours. Resampling whole calendar months keeps each episode's rows together,
and resampling the *same* months for both arms keeps the comparison paired.
"""

from typing import Final, TypedDict

import numpy as np
import polars as pl
from scipy import stats

N_BOOTSTRAP_RESAMPLES: Final[int] = 2000
"""How many resamples each interval is read from."""

BOOTSTRAP_SEED: Final[int] = 20260920
"""Seeds a fresh random stream for every interval, so no interval depends on those before it."""


class BootstrapInterval(TypedDict):
    """A paired difference, its interval, and what the interval rests on."""

    difference: float
    lower_95: float
    upper_95: float
    seed_spread: float
    n_rows: int
    n_months: int


def paired_differences(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-seed, per-row difference in one metric, and each row's month.

    Args:
        losses: Per-row losses holding both arms, restricted to the scope wanted, carrying `arm`,
            `site`, `time`, `seed` and `month`.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
        metric: The loss column to difference.

    Returns:
        An array of shape (n_seeds, n_rows) of treatment-minus-reference differences, in seed order,
        and the month label of each row.
    """
    paired = (
        losses.filter(pl.col("arm") == reference)
        .select("site", "time", "seed", "month", reference=pl.col(metric))
        .join(
            losses.filter(pl.col("arm") == treatment).select(
                "site", "time", "seed", treatment=pl.col(metric)
            ),
            on=["site", "time", "seed"],
            how="inner",
        )
        .sort("seed", "site", "time")
    )
    by_seed = [
        paired.filter(pl.col("seed") == seed).select(
            "month", difference=pl.col("treatment") - pl.col("reference")
        )
        for seed in sorted(paired["seed"].unique().to_list())
    ]
    differences = np.stack([frame["difference"].to_numpy() for frame in by_seed])
    return differences, by_seed[0]["month"].to_numpy()


def _resampled_means(*, values: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Resample whole months and one seed `N_BOOTSTRAP_RESAMPLES` times, and return each mean.

    Shared by `bootstrap_difference` and `bootstrap_level`: both interval a mean over an array of
    shape (n_seeds, n_rows), one a paired difference and the other one arm's own metric, and the
    resampling unit is the same for both.

    Args:
        values: Shape (n_seeds, n_rows), in seed order.
        months: Each row's month label, in the same row order every seed shares.

    Returns:
        `N_BOOTSTRAP_RESAMPLES` resampled means.
    """
    unique_months, month_index = np.unique(months, return_inverse=True)
    rows_by_month = [np.flatnonzero(month_index == index) for index in range(len(unique_months))]

    # The seed draw comes before the month draw in every resample. Swapping them, or vectorising
    # the loop, would change every published interval's digits without changing its definition.
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    resampled = np.empty(N_BOOTSTRAP_RESAMPLES)
    for resample in range(N_BOOTSTRAP_RESAMPLES):
        seed_index = generator.integers(0, values.shape[0])
        drawn = generator.integers(0, len(unique_months), size=len(unique_months))
        rows = np.concatenate([rows_by_month[index] for index in drawn])
        resampled[resample] = values[seed_index, rows].mean()
    return resampled


def bootstrap_difference(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    metric: str,
) -> BootstrapInterval:
    """Bootstrap the paired arm-to-arm difference, resampling whole months and a seed.

    Each resample also draws one of the seeds, for both arms alike. Without that, the interval would
    treat the seed-averaged loss as a fixed quantity and exclude a source of variation measured at
    the same order as the effect itself.

    Args:
        losses: Per-row losses for both arms, already restricted to the scope wanted.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
        metric: The loss column to difference.

    Returns:
        The point estimate, the 2.5th and 97.5th percentiles, and what the estimate rests on.
    """
    differences, months = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=metric
    )
    resampled = _resampled_means(values=differences, months=months)
    return {
        "difference": float(differences.mean()),
        "lower_95": float(np.percentile(resampled, 2.5)),
        "upper_95": float(np.percentile(resampled, 97.5)),
        "seed_spread": float(differences.mean(axis=1).std()),
        "n_rows": differences.shape[1],
        "n_months": len(np.unique(months)),
    }


def arm_values(*, losses: pl.DataFrame, arm: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one arm's per-seed, per-row metric value, and each row's month.

    Args:
        losses: Per-row losses holding the arm, restricted to the scope wanted, carrying `arm`,
            `site`, `time`, `seed` and `month`.
        arm: The arm to read.
        metric: The loss column to read.

    Returns:
        An array of shape (n_seeds, n_rows), in seed order, and the month label of each row.
    """
    rows = (
        losses.filter(pl.col("arm") == arm)
        .select("site", "time", "seed", "month", value=pl.col(metric))
        .sort("seed", "site", "time")
    )
    by_seed = [
        rows.filter(pl.col("seed") == seed).select("month", "value")
        for seed in sorted(rows["seed"].unique().to_list())
    ]
    values = np.stack([frame["value"].to_numpy() for frame in by_seed])
    return values, by_seed[0]["month"].to_numpy()


class LevelBootstrapInterval(TypedDict):
    """One arm's absolute level, its interval, and what the interval rests on."""

    level: float
    lower_95: float
    upper_95: float
    seed_spread: float
    n_rows: int
    n_months: int


def bootstrap_level(*, losses: pl.DataFrame, arm: str, metric: str) -> LevelBootstrapInterval:
    """Bootstrap one arm's absolute mean, resampling whole months and a seed.

    A level interval has no pairing to hold fixed, so it carries the full month-to-month weather
    noise that neighbouring generators share. A paired difference between two arms cancels that
    shared noise, which is why two arms' level intervals can overlap while their paired difference
    is still statistically significant at the 5% level.

    Args:
        losses: Per-row losses for one arm, already restricted to the scope wanted.
        arm: The arm to interval.
        metric: The loss column to average.

    Returns:
        The point estimate, the 2.5th and 97.5th percentiles, and what the estimate rests on.
    """
    values, months = arm_values(losses=losses, arm=arm, metric=metric)
    resampled = _resampled_means(values=values, months=months)
    return {
        "level": float(values.mean()),
        "lower_95": float(np.percentile(resampled, 2.5)),
        "upper_95": float(np.percentile(resampled, 97.5)),
        "seed_spread": float(values.mean(axis=1).std()),
        "n_rows": values.shape[1],
        "n_months": len(np.unique(months)),
    }


def per_fold_differences(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str
) -> list[float]:
    """Return the arm-to-arm difference within each fold separately.

    Five folds agreeing in sign is the cheapest robustness statistic available here, and it is one
    the block bootstrap cannot give: the bootstrap treats months as the unit of independence, while
    each site rests on only five trained models per arm.

    Args:
        losses: Per-row losses for both arms, carrying `fold`.
        treatment: The arm whose metric is being compared.
        reference: The arm it is compared against.
        metric: The loss column to difference.

    Returns:
        One difference per fold that holds rows, in fold order.
    """
    differences: list[float] = []
    for fold in sorted(losses["fold"].unique().to_list()):
        paired, _ = paired_differences(
            losses=losses.filter(pl.col("fold") == fold),
            treatment=treatment,
            reference=reference,
            metric=metric,
        )
        differences.append(float(paired.mean()))
    return differences


def fold_t_interval(*, fold_differences: list[float]) -> tuple[float, float]:
    """Return a 95% t-interval for the mean of the per-fold differences.

    The month bootstrap treats months as independent and holds each fold's fitted models fixed, so
    it covers month-to-month weather and the fitting seed only. The folds' spread also carries what
    one set of trained models, rather than another, contributes. With five folds the interval rests
    on four degrees of freedom, so it is wide; it is a second view of the spread, not a replacement
    for the bootstrap.

    Args:
        fold_differences: One arm-to-arm difference per fold, from `per_fold_differences`.

    Returns:
        The lower and upper ends of the interval.

    Raises:
        ValueError: If fewer than two folds are given, which leaves no spread to measure.
    """
    if len(fold_differences) < 2:
        msg = f"a t-interval needs at least two folds, got {len(fold_differences)}"
        raise ValueError(msg)
    values = np.asarray(fold_differences, dtype=np.float64)
    half_width = float(
        stats.t.ppf(0.975, df=len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    )
    return float(values.mean()) - half_width, float(values.mean()) + half_width
