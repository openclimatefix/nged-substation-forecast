"""Helpers the report scripts share: tables, intervals, thresholds, flags, errors, and coverage."""

from typing import Final

import numpy as np
import polars as pl
from scipy.stats import beta

N_RESAMPLES: Final[int] = 2000
SEED: Final[int] = 20261013
FALSE_ALARM_RATE: Final[float] = 0.05
BLOCK_NAMES_BY_INDEX: Final[dict[int, str]] = {
    0: "Sep-Nov",
    1: "Dec-Feb",
    2: "Mar-May",
    3: "Jun-Aug",
}
SUMMER_BLOCK: Final[int] = 3
"""The index of the Jun-Aug block, where the nulls raise their false alarms."""


def table(frame: pl.DataFrame) -> str:
    """Return a frame as a pipe-separated table with rounded floats."""
    return frame.with_columns(pl.col(pl.Float64).round(4)).write_csv(separator="|")


def clopper_pearson(*, count: int, total: int, level: float = 0.95) -> tuple[float, float]:
    """Return the Clopper-Pearson interval of a binomial proportion."""
    alpha = 1 - level
    low = 0.0 if count == 0 else float(beta.ppf(alpha / 2, count, total - count + 1))
    high = 1.0 if count == total else float(beta.ppf(1 - alpha / 2, count + 1, total - count))
    return low, high


def cluster_bootstrap(
    *, frame: pl.DataFrame, value: str, outer: str, inner: str, seed: int = SEED
) -> tuple[float, float, float]:
    """Return the mean of a column and its 95% interval, resampling outer then inner clusters.

    Whole demand series (outer) are resampled with replacement, then whole batteries (inner) within
    each resampled series, as the prior study resampled whole aggregates.

    Args:
        frame: The rows.
        value: The column to average.
        outer: The outer cluster column (the demand series).
        inner: The inner cluster column (the battery).
        seed: Seeds the resampling.

    Returns:
        The mean, and the 2.5% and 97.5% quantiles of the resampled means.
    """
    rows = (
        frame.filter(pl.col(value).is_finite())
        .group_by(outer, inner)
        .agg(total=pl.col(value).sum(), count=pl.len())
    )
    by_outer: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, part in rows.group_by(outer):
        by_outer[str(key[0])] = (part["total"].to_numpy(), part["count"].to_numpy())
    names = list(by_outer)
    rng = np.random.default_rng(seed)
    means = np.empty(N_RESAMPLES)
    for r in range(N_RESAMPLES):
        total = count = 0.0
        for name in rng.choice(names, size=len(names), replace=True):
            sums, counts = by_outer[str(name)]
            pick = rng.integers(0, len(sums), size=len(sums))
            total += sums[pick].sum()
            count += counts[pick].sum()
        means[r] = total / count
    overall = float(rows["total"].sum()) / float(rows["count"].sum())
    low, high = np.quantile(means, [0.025, 0.975])
    return overall, float(low), float(high)


def thresholds(*, nulls: pl.DataFrame) -> dict[str, float]:
    """Return each series' detection threshold: the 95th percentile of the other series' nulls.

    Args:
        nulls: Rung 1's rows with no added battery; columns `series` and `log_bayes_factor`.

    Returns:
        By series label, the threshold on the log Bayes factor.
    """
    finite = nulls.filter(pl.col("log_bayes_factor").is_finite())
    out = {}
    for label in nulls["series"].unique().to_list():
        others = finite.filter(pl.col("series") != label)["log_bayes_factor"].to_numpy()
        out[label] = float(np.quantile(others, 1 - FALSE_ALARM_RATE))
    return out


def flag(*, frame: pl.DataFrame, threshold: dict[str, float], default: float) -> pl.DataFrame:
    """Add a boolean `flagged` column: the log Bayes factor is finite and above its threshold."""
    return frame.with_columns(
        threshold=pl.col("series").replace_strict(
            threshold, default=default, return_dtype=pl.Float64
        )
    ).with_columns(
        flagged=pl.col("log_bayes_factor").is_finite()
        & (pl.col("log_bayes_factor") > pl.col("threshold"))
    )


def rate_table(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the share of flagged sums per group with a Clopper-Pearson interval."""
    grouped = frame.group_by(by).agg(flagged=pl.col("flagged").sum(), total=pl.len()).sort(by)
    lows, highs = zip(
        *[
            clopper_pearson(count=int(f), total=int(t))
            for f, t in zip(grouped["flagged"], grouped["total"], strict=True)
        ],
        strict=True,
    )
    return grouped.with_columns(
        rate=pl.col("flagged") / pl.col("total"), ci_low=pl.Series(lows), ci_high=pl.Series(highs)
    )


def with_errors(*, frame: pl.DataFrame, power: str, energy: str) -> pl.DataFrame:
    """Add relative errors, interval membership, and a finite-interval flag for one class."""
    return frame.with_columns(
        power_error=(pl.col(f"{power}_median") - pl.col("true_power_mw")).abs()
        / pl.col("true_power_mw"),
        energy_error=(pl.col(f"{energy}_median") - pl.col("true_energy_mwh")).abs()
        / pl.col("true_energy_mwh"),
        power_in_90=(pl.col(f"{power}_q05") <= pl.col("true_power_mw"))
        & (pl.col("true_power_mw") <= pl.col(f"{power}_q95")),
        power_in_50=(pl.col(f"{power}_q25") <= pl.col("true_power_mw"))
        & (pl.col("true_power_mw") <= pl.col(f"{power}_q75")),
        energy_in_90=(pl.col(f"{energy}_q05") <= pl.col("true_energy_mwh"))
        & (pl.col("true_energy_mwh") <= pl.col(f"{energy}_q95")),
        energy_in_50=(pl.col(f"{energy}_q25") <= pl.col("true_energy_mwh"))
        & (pl.col("true_energy_mwh") <= pl.col(f"{energy}_q75")),
    )


def coverage(*, frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the coverage of the 50% and 90% intervals of power and energy per group.

    Sums without a covariance (no interval) count as not covering.
    """
    return (
        frame.group_by(by)
        .agg(
            n=pl.len(),
            with_interval=pl.col("has_interval").sum(),
            power_90=pl.col("power_in_90").fill_null(False).mean(),
            power_50=pl.col("power_in_50").fill_null(False).mean(),
            energy_90=pl.col("energy_in_90").fill_null(False).mean(),
            energy_50=pl.col("energy_in_50").fill_null(False).mean(),
        )
        .sort(by)
    )
