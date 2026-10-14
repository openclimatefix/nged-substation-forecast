"""Month-resampled intervals for a mean error, a paired difference, and a regression slope.

Every interval resamples whole validation months with replacement and one of the fitting seeds,
the same draw for every arm, so a contrast stays paired and a month's weather moves every arm
together. A mean error is the mean over series of each series' mean, so a series with more rows
does not count more, as the quick-wins report computes it.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

BOOTSTRAP_SEED: Final[int] = 20261013
"""Seeds the month and fitting-seed draws of every interval; every call restarts the stream."""

N_RESAMPLES: Final[int] = 2000
"""How many resamples each interval is read from."""

MATERIAL_DIFFERENCE: Final[float] = -0.15
"""A planned contrast is material only if its mean paired difference is at most this, in points."""

MATERIAL_MONTH_SHARE: Final[float] = 0.75
"""A planned contrast is material only if this share of months (9 of 12) have a negative mean."""


@dataclass(frozen=True)
class Monthly:
    """One arm's totals and row counts by fitting seed, series, and month."""

    totals: np.ndarray
    """Shape (seeds, series, months): sum of the value over the rows."""
    counts: np.ndarray
    """Shape (series, months): number of rows, the same for every seed."""
    months: tuple[str, ...]
    series: tuple[int, ...]


@dataclass(frozen=True)
class Interval:
    """A value with its 95% interval."""

    value: float
    lower: float
    upper: float
    seed_spread: float
    n_months: int


@dataclass(frozen=True)
class Contrast:
    """A paired difference with its interval and the months that agree."""

    difference: float
    lower: float
    upper: float
    months_better: int
    n_months: int
    material: bool


def monthly(*, rows: pl.DataFrame, value: str) -> Monthly:
    """Sum a column by seed, series, and valid-time month.

    Args:
        rows: One arm's rows within one scope, with `seed`, `time_series_id`, `valid_time`.
        value: The column to sum.

    Returns:
        The totals by seed, series, and month, the row counts, and the labels.

    Raises:
        ValueError: If the seeds do not hold the same rows.
    """
    grouped = (
        rows.with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("seed", "time_series_id", "month")
        .agg(total=pl.col(value).sum(), count=pl.len())
    )
    months = tuple(sorted(grouped["month"].unique().to_list()))
    series = tuple(sorted(grouped["time_series_id"].unique().to_list()))
    seeds = sorted(grouped["seed"].unique().to_list())
    totals = np.zeros((len(seeds), len(series), len(months)))
    counts = np.zeros((len(seeds), len(series), len(months)))
    for row in grouped.iter_rows(named=True):
        k = seeds.index(row["seed"])
        s = series.index(row["time_series_id"])
        m = months.index(row["month"])
        totals[k, s, m] = row["total"]
        counts[k, s, m] = row["count"]
    if not all(np.array_equal(counts[0], counts[k]) for k in range(1, len(seeds))):
        msg = "the fitting seeds do not hold the same rows"
        raise ValueError(msg)
    return Monthly(totals=totals, counts=counts[0], months=months, series=series)


def _draws(*, n_seeds: int, n_months: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    seed_index = np.empty(N_RESAMPLES, dtype=int)
    month_index = np.empty((N_RESAMPLES, n_months), dtype=int)
    for draw in range(N_RESAMPLES):
        seed_index[draw] = generator.integers(0, n_seeds)
        month_index[draw] = generator.integers(0, n_months, size=n_months)
    return seed_index, month_index


def _series_mean(*, totals: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Mean over series of totals/counts along the last axis; series with no rows are skipped."""
    with np.errstate(invalid="ignore", divide="ignore"):
        per_series = totals / counts
    return np.nanmean(np.where(counts > 0, per_series, np.nan), axis=-1)


def _resampled(*, table: Monthly) -> np.ndarray:
    n_seeds, _, n_months = table.totals.shape
    seed_index, month_index = _draws(n_seeds=n_seeds, n_months=n_months)
    chosen_totals = np.take_along_axis(
        table.totals[seed_index], month_index[:, None, :], axis=2
    ).sum(axis=2)
    chosen_counts = np.take_along_axis(
        np.broadcast_to(table.counts, (N_RESAMPLES, *table.counts.shape)),
        month_index[:, None, :],
        axis=2,
    ).sum(axis=2)
    return _series_mean(totals=chosen_totals, counts=chosen_counts)


def level(*, table: Monthly) -> Interval:
    """Return an arm's mean error with its month-resampled interval."""
    seed_means = _series_mean(
        totals=table.totals.sum(axis=2), counts=table.counts.sum(axis=1)[None]
    )
    draws = _resampled(table=table)
    return Interval(
        value=float(seed_means.mean()),
        lower=float(np.percentile(draws, 2.5)),
        upper=float(np.percentile(draws, 97.5)),
        seed_spread=float(seed_means.std()),
        n_months=len(table.months),
    )


def contrast(*, treatment: Monthly, reference: Monthly) -> Contrast:
    """Return the paired difference (treatment minus reference) and whether it is material."""
    if not (
        np.array_equal(treatment.counts, reference.counts)
        and treatment.months == reference.months
        and treatment.series == reference.series
    ):
        msg = "the two arms do not hold the same rows"
        raise ValueError(msg)
    difference = Monthly(
        totals=treatment.totals - reference.totals,
        counts=treatment.counts,
        months=treatment.months,
        series=treatment.series,
    )
    interval = level(table=difference)
    per_month = np.nanmean(
        np.where(
            difference.counts > 0,
            difference.totals.mean(axis=0) / np.maximum(difference.counts, 1),
            np.nan,
        ),
        axis=0,
    )
    better = int((per_month < 0).sum())
    needed = int(np.ceil(MATERIAL_MONTH_SHARE * len(difference.months)))
    material = bool(
        interval.value <= MATERIAL_DIFFERENCE and interval.upper < 0 and better >= needed
    )
    return Contrast(
        difference=interval.value,
        lower=interval.lower,
        upper=interval.upper,
        months_better=better,
        n_months=len(difference.months),
        material=material,
    )


def slope_interval(*, rows: pl.DataFrame) -> Interval:
    """Regress observed on predicted power (fractions of capacity); interval by month resampling.

    The slope is pooled over every row of the scope, per fitting seed, then averaged over seeds.
    The interval resamples whole months and a seed using each month's sums.

    Args:
        rows: One arm's rows within one scope with `seed`, `valid_time`, `point_norm`,
            `actual_norm`.

    Returns:
        The pooled slope with its 95% month-resampled interval.
    """
    grouped = (
        rows.with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("seed", "month")
        .agg(
            n=pl.len().cast(pl.Float64),
            sx=pl.col("point_norm").sum(),
            sy=pl.col("actual_norm").sum(),
            sxx=(pl.col("point_norm") ** 2).sum(),
            sxy=(pl.col("point_norm") * pl.col("actual_norm")).sum(),
        )
    )
    months = sorted(grouped["month"].unique().to_list())
    seeds = sorted(grouped["seed"].unique().to_list())
    stats = np.zeros((len(seeds), len(months), 5))
    for row in grouped.iter_rows(named=True):
        stats[seeds.index(row["seed"]), months.index(row["month"])] = [
            row["n"],
            row["sx"],
            row["sy"],
            row["sxx"],
            row["sxy"],
        ]

    def slope(sums: np.ndarray) -> np.ndarray:
        n, sx, sy, sxx, sxy = (sums[..., i] for i in range(5))
        return (n * sxy - sx * sy) / (n * sxx - sx**2)

    per_seed = slope(stats.sum(axis=1))
    seed_index, month_index = _draws(n_seeds=len(seeds), n_months=len(months))
    chosen = np.take_along_axis(stats[seed_index], month_index[:, :, None], axis=1).sum(axis=1)
    draws = slope(chosen)
    return Interval(
        value=float(per_seed.mean()),
        lower=float(np.percentile(draws, 2.5)),
        upper=float(np.percentile(draws, 97.5)),
        seed_spread=float(per_seed.std()),
        n_months=len(months),
    )
