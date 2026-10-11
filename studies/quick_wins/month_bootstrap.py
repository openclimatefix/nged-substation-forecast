"""Intervals for an arm's error, and for the paired difference between two arms, by month.

The 31 series share their weather, so the independent units are weather episodes, not forecast
rows. Each resample draws whole validation months with replacement and applies the same draw to
every arm, which keeps contrasts paired. A metric is computed as the leaderboard computes it: the
mean over a series' rows, then the mean over series, so a series with more rows does not count more.

A percentile interval from 12 months is too narrow, so every contrast also carries a t-interval over
the monthly differences, and the count of months in which the arm beat the reference.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy import stats

N_RESAMPLES: Final[int] = 2000
"""How many month resamples each interval is read from."""

SEED: Final[int] = 20261011
"""Seeds the month draws, so a re-run reproduces every interval."""


@dataclass(frozen=True)
class MonthlyTable:
    """One arm's metric totals and row counts by series and month.

    Attributes:
        series_ids: The series, in the table's row order.
        months: The months, in the table's column order.
        totals: Sum of the metric over the rows of each series and month, shape (series, months).
        counts: Number of rows of each series and month, the same shape.
    """

    series_ids: tuple[int, ...]
    months: tuple[str, ...]
    totals: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class Interval:
    """A value or a paired difference with its intervals.

    Attributes:
        estimate: The value on all months.
        lower_95: Lower end of the 95% month-resampled percentile interval.
        upper_95: Upper end of that interval.
        t_lower_95: Lower end of the 95% t-interval over the monthly values.
        t_upper_95: Upper end of that interval.
        months_better: For a difference, the number of months in which the arm had the lower value
            than the reference; for a level, 0.
        n_months: The number of months resampled.
    """

    estimate: float
    lower_95: float
    upper_95: float
    t_lower_95: float
    t_upper_95: float
    months_better: int
    n_months: int


def monthly_table(*, scores: pl.DataFrame, metric: str) -> MonthlyTable:
    """Total a per-row metric by series and by month of the valid time.

    Args:
        scores: Per-row scores with `time_series_id`, `valid_time`, and the metric column.
        metric: The metric column to total, such as `crps_norm`.

    Returns:
        The table, sorted by series and month.
    """
    grouped = (
        scores.with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))
        .group_by("time_series_id", "month")
        .agg(total=pl.col(metric).sum(), count=pl.len())
    )
    series_ids = tuple(sorted(grouped["time_series_id"].unique().to_list()))
    months = tuple(sorted(grouped["month"].unique().to_list()))
    totals = np.zeros((len(series_ids), len(months)))
    counts = np.zeros((len(series_ids), len(months)))
    series_index = {series_id: i for i, series_id in enumerate(series_ids)}
    month_index = {month: j for j, month in enumerate(months)}
    for series_id, month, total, count in grouped.iter_rows():
        totals[series_index[series_id], month_index[month]] = total
        counts[series_index[series_id], month_index[month]] = count
    return MonthlyTable(series_ids=series_ids, months=months, totals=totals, counts=counts)


def _check_same_rows(*, tables: tuple[MonthlyTable, ...]) -> None:
    first = tables[0]
    for other in tables[1:]:
        if (
            other.series_ids != first.series_ids
            or other.months != first.months
            or not np.array_equal(other.counts, first.counts)
        ):
            msg = "arms must be scored on exactly the same series, months, and row counts"
            raise ValueError(msg)


def _level(*, totals: np.ndarray, counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the metric under month weights: the mean over series of each series' mean."""
    per_series = (totals @ weights.T) / (counts @ weights.T)
    return per_series.mean(axis=0)


def _month_draws(*, n_months: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, n_months, size=(N_RESAMPLES, n_months))
    weights = np.zeros((N_RESAMPLES, n_months))
    for row, drawn in enumerate(picks):
        np.add.at(weights[row], drawn, 1.0)
    return weights


def _t_interval(*, monthly: np.ndarray) -> tuple[float, float]:
    mean = float(monthly.mean())
    half = float(
        stats.t.ppf(0.975, df=len(monthly) - 1) * monthly.std(ddof=1) / np.sqrt(len(monthly))
    )
    return mean - half, mean + half


def level_interval(*, table: MonthlyTable) -> Interval:
    """Interval an arm's own metric, resampling whole months."""
    weights = _month_draws(n_months=len(table.months))
    resampled = _level(totals=table.totals, counts=table.counts, weights=weights)
    estimate = float(
        _level(totals=table.totals, counts=table.counts, weights=np.ones((1, len(table.months))))[0]
    )
    monthly = (table.totals / table.counts).mean(axis=0)
    t_lower, t_upper = _t_interval(monthly=monthly)
    return Interval(
        estimate=estimate,
        lower_95=float(np.percentile(resampled, 2.5)),
        upper_95=float(np.percentile(resampled, 97.5)),
        t_lower_95=t_lower,
        t_upper_95=t_upper,
        months_better=0,
        n_months=len(table.months),
    )


def difference_interval(*, arm: MonthlyTable, reference: MonthlyTable) -> Interval:
    """Interval `arm - reference` with one shared month draw, so a negative value favours the arm."""
    _check_same_rows(tables=(arm, reference))
    weights = _month_draws(n_months=len(arm.months))
    resampled = _level(totals=arm.totals, counts=arm.counts, weights=weights) - _level(
        totals=reference.totals, counts=reference.counts, weights=weights
    )
    ones = np.ones((1, len(arm.months)))
    estimate = float(
        _level(totals=arm.totals, counts=arm.counts, weights=ones)[0]
        - _level(totals=reference.totals, counts=reference.counts, weights=ones)[0]
    )
    monthly = (arm.totals / arm.counts).mean(axis=0) - (reference.totals / reference.counts).mean(
        axis=0
    )
    t_lower, t_upper = _t_interval(monthly=monthly)
    return Interval(
        estimate=estimate,
        lower_95=float(np.percentile(resampled, 2.5)),
        upper_95=float(np.percentile(resampled, 97.5)),
        t_lower_95=t_lower,
        t_upper_95=t_upper,
        months_better=int((monthly < 0).sum()),
        n_months=len(arm.months),
    )
