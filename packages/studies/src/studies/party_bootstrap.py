"""Interval a paired difference by resampling lead parties as well as calendar months.

`studies.bootstrap` resamples whole calendar months and a fitting seed while holding the set of
batteries fixed, so its interval speaks about those batteries in that year. Where one lead party
runs many of the batteries, the party is the unit of independence, not the battery. This module
resamples parties with replacement and months with replacement together. A resample draws a count
for each party and for each calendar month, and totals the per-party, per-month loss cells with
those counts as weights. The fitting seed is averaged out beforehand, so the interval leaves out
the seed-to-seed spread that `studies.bootstrap` includes.
"""

from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import polars as pl

from studies.bootstrap import BOOTSTRAP_SEED, N_BOOTSTRAP_RESAMPLES


class PartyMonthCells(NamedTuple):
    """One arm's loss, summed within each lead party and calendar month.

    Attributes:
        sums: The summed loss, shape (n_parties, n_months), averaged over seeds first.
        counts: The number of scored half-hours in each cell, the same shape.
        parties: The lead party of each row of `sums`, sorted.
        months: The month label of each column of `sums`, sorted.
    """

    sums: np.ndarray
    counts: np.ndarray
    parties: list[str]
    months: list[str]


def arm_cells(
    *, losses: pl.DataFrame, arm: str, party_of: Mapping[str, str], metric: str = "crps_pct"
) -> PartyMonthCells:
    """Sum one arm's loss within each lead party and month.

    Args:
        losses: Rows with `site`, `arm`, `time`, `seed`, `month`, and the metric column.
        arm: The arm to sum.
        party_of: The lead party of each `site`.
        metric: The loss column.

    Returns:
        The cells. A cell that no scored half-hour falls in holds zero.
    """
    per_row = (
        losses.filter(pl.col("arm") == arm)
        .group_by("site", "time", "month")
        .agg(loss=pl.col(metric).mean())
        .with_columns(party=pl.col("site").replace_strict(dict(party_of), return_dtype=pl.String))
    )
    cells = per_row.group_by("party", "month").agg(
        total=pl.col("loss").sum(), count=pl.len().cast(pl.Float64)
    )
    parties = sorted(cells["party"].unique().to_list())
    months = sorted(cells["month"].unique().to_list())
    sums = np.zeros((len(parties), len(months)))
    counts = np.zeros((len(parties), len(months)))
    for party, month, total, count in cells.iter_rows():
        sums[parties.index(party), months.index(month)] = total
        counts[parties.index(party), months.index(month)] = count
    return PartyMonthCells(sums=sums, counts=counts, parties=parties, months=months)


def resample_totals(
    *, stacked: np.ndarray, n_resamples: int = N_BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Return, for each resample, the cell totals under a party draw and a month draw.

    Args:
        stacked: One (n_parties, n_months) array of cell sums per quantity, shape
            (n_quantities, n_parties, n_months).
        n_resamples: How many resamples.
        seed: The random seed.

    Returns:
        Shape (n_resamples, n_quantities): each quantity's total over the drawn parties and months.
    """
    generator = np.random.default_rng(seed)
    n_parties, n_months = stacked.shape[1:]
    party_draws = generator.integers(0, n_parties, size=(n_resamples, n_parties))
    month_draws = generator.integers(0, n_months, size=(n_resamples, n_months))
    party_weights = np.stack([np.bincount(row, minlength=n_parties) for row in party_draws])
    month_weights = np.stack([np.bincount(row, minlength=n_months) for row in month_draws])
    return np.einsum("rp,kpm,rm->rk", party_weights, stacked, month_weights)


def party_month_difference(
    *, treatment: PartyMonthCells, reference: PartyMonthCells
) -> dict[str, float]:
    """Interval the mean per-half-hour difference between two arms, resampling parties and months.

    Args:
        treatment: The treatment arm's cells.
        reference: The reference arm's cells, over the same half-hours.

    Returns:
        `difference` (the pooled mean, treatment minus reference), `lower_95` and `upper_95` (the
        2.5th and 97.5th percentiles over the resamples), `party_weighted` (the mean of each
        party's own mean difference, so every party counts once), `n_parties`, and `n_months`.

    Raises:
        ValueError: If the two arms' parties, months, or cell counts differ.
    """
    if (
        treatment.parties != reference.parties
        or treatment.months != reference.months
        or not np.array_equal(treatment.counts, reference.counts)
    ):
        msg = "The two arms must cover the same parties, months, and half-hours."
        raise ValueError(msg)
    gap = treatment.sums - reference.sums
    totals = resample_totals(stacked=np.stack([gap, treatment.counts]))
    resampled = totals[:, 0] / totals[:, 1]
    per_party = gap.sum(axis=1) / treatment.counts.sum(axis=1)
    return {
        "difference": float(gap.sum() / treatment.counts.sum()),
        "lower_95": float(np.percentile(resampled, 2.5)),
        "upper_95": float(np.percentile(resampled, 97.5)),
        "party_weighted": float(per_party.mean()),
        "n_parties": float(len(treatment.parties)),
        "n_months": float(len(treatment.months)),
    }


def party_month_share(
    *, baseline: PartyMonthCells, full: PartyMonthCells, partial: PartyMonthCells
) -> dict[str, float]:
    """Interval the share of a gain that a partial treatment recovers, by party and month draws.

    The share is `(baseline - partial) / (baseline - full)` in total loss.

    Args:
        baseline: The arm with neither treatment.
        full: The arm with the full treatment.
        partial: The arm with the partial treatment.

    Returns:
        `share`, `lower_95`, and `upper_95`.
    """
    totals = resample_totals(stacked=np.stack([baseline.sums, full.sums, partial.sums]))
    resampled = (totals[:, 0] - totals[:, 2]) / (totals[:, 0] - totals[:, 1])
    point = (baseline.sums.sum() - partial.sums.sum()) / (baseline.sums.sum() - full.sums.sum())
    return {
        "share": float(point),
        "lower_95": float(np.percentile(resampled, 2.5)),
        "upper_95": float(np.percentile(resampled, 97.5)),
    }
