"""Scores, paired intervals, and subset rules for experiment 119.

The score is the quantile score: twice the mean pinball loss over the 11 levels, in points of
effective capacity. Intervals resample whole calendar months with `studies.bootstrap`; the hard-day
subsets, which hold too few months, resample 7-day blocks of run dates instead.
"""

import json
from datetime import timedelta
from typing import Final

import e00119_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level

N_RESAMPLES: Final[int] = 4000
"""Enough draws for each tail of a 99% interval to hold 20 draws."""
BLOCK_DAYS: Final[int] = 7
N_CONTRASTS: Final[int] = 5
LEVEL_99: Final[float] = 99.0
LEVEL_BONFERRONI: Final[float] = 100.0 - (100.0 - 95.0) / N_CONTRASTS
"""The coverage of one of five planned contrasts under a Bonferroni correction of 95%, so 99%."""


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the quantile score of each row (twice the mean pinball loss)."""
    levels = np.array(common.QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)


def month_interval(
    *,
    treatment: np.ndarray,
    reference: np.ndarray,
    site: np.ndarray,
    run: np.ndarray,
    month: np.ndarray,
    level: float = LEVEL_BONFERRONI,
) -> dict[str, float]:
    """Return the paired difference of two per-row scores and its month-resampled interval.

    Args:
        treatment: Per-row scores of the arm, shape (n_seeds, n_rows) or (n_rows,).
        reference: Per-row scores of the reference, the same shape or (n_rows,).
        site: Each row's site index.
        run: Each row's run index (with the row's order, the row's identity within a site).
        month: Each row's `year * 12 + month` label of its run date.
        level: The interval's coverage in percent.

    Returns:
        `difference` (mean of the seed-averaged paired differences), `lower`, `upper`, and the
        number of rows and months.
    """
    treatment = np.atleast_2d(treatment)
    reference = np.broadcast_to(np.atleast_2d(reference), treatment.shape)
    n_seeds, n_rows = treatment.shape
    key = [f"{s}|{r}|{i}" for i, (s, r) in enumerate(zip(site.tolist(), run.tolist(), strict=True))]
    label = np.array([f"{m // 12}-{m % 12 + 1:02d}" for m in month.tolist()])
    frames = [
        pl.DataFrame(
            {
                "arm": arm,
                "site": site.astype(np.int64),
                "time": key,
                "seed": seed,
                "month": label,
                "score": values[seed],
            }
        )
        for arm, values in (("treatment", treatment), ("reference", reference))
        for seed in range(n_seeds)
    ]
    losses = pl.concat(frames)
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment="treatment",
        reference="reference",
        metric="score",
        level=level,
        n_resamples=N_RESAMPLES,
    )
    return {
        "difference": float((treatment - reference).mean()),
        "lower": lower,
        "upper": upper,
        "n_rows": int(n_rows),
        "n_months": len(np.unique(month)),
    }


def block_interval(
    *,
    treatment: np.ndarray,
    reference: np.ndarray,
    run: np.ndarray,
    level: float = LEVEL_99,
    seed: int = 20261021,
) -> dict[str, float]:
    """Return a paired difference and a 7-day block bootstrap interval, for hard-day subsets.

    Whole 7-day blocks of run dates are resampled, paired across the two arms, so a subset with
    days in few months still has many resampling units.
    """
    difference = np.atleast_1d(treatment - reference)
    block = run // BLOCK_DAYS
    unique, inverse = np.unique(block, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse).astype(np.float64)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(unique), size=(N_RESAMPLES, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (100.0 - level) / 2.0
    return {
        "difference": float(difference.mean()),
        "lower": float(np.percentile(means, tail)),
        "upper": float(np.percentile(means, 100.0 - tail)),
        "n_rows": len(difference),
        "n_blocks": len(unique),
        "n_days": len(np.unique(run)),
    }


def hard_day_masks(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return a boolean mask over the rows for each hard-day subset, from `hard_days.json`."""
    spec = json.loads((common.WORK_DIR / "hard_days.json").read_text())
    day = rows["run_date"].dt.strftime("%Y-%m-%d").to_numpy()
    site = rows["site_index"].to_numpy()
    cold = np.isin(day, spec["coldest"])
    showery = np.zeros(len(day), dtype=bool)
    largest = np.zeros(len(day), dtype=bool)
    for index, name in enumerate(common.SCORED_SITES):
        at_site = site == index
        showery |= at_site & np.isin(day, spec["showery"][name])
        largest |= at_site & np.isin(day, spec["largest_error"][name])
    return {"coldest": cold, "showery": showery, "largest_error": largest}


def month_of_run(*, run_date: pl.Series) -> np.ndarray:
    """Return `year * 12 + month - 1` of each run date."""
    return (run_date.dt.year() * 12 + run_date.dt.month() - 1).to_numpy()


def is_winter(*, run_date: pl.Series) -> np.ndarray:
    """Return True for run dates in November to March."""
    month = run_date.dt.month().to_numpy()
    return (month >= 11) | (month <= 3)


def weekend_or_holiday(*, valid_time: pl.Series) -> np.ndarray:
    """Return True for Saturdays and Sundays (bank holidays are handled by the report)."""
    return (valid_time.dt.weekday().to_numpy() >= 6).astype(bool)


def purge_window() -> timedelta:
    """Return the purge on each side of a tuning block."""
    return timedelta(days=7)
