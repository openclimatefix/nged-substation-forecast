"""Row metrics and the paired month-block bootstrap, copied from experiment 187's scorer."""

from typing import Final

import e00192_data as data
import numpy as np
import polars as pl

N_DRAWS: Final[int] = 20_000


def row_metrics(*, frame: pl.DataFrame, quantiles: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row score columns for one arm."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    error = actual[:, None] - quantiles
    pinball = np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error)
    q = {
        name: quantiles[:, data.QUANTILES.index(name)]
        for name in ("q01", "q05", "q50", "q95", "q99")
    }
    return {
        "qs": 2.0 * pinball.mean(axis=1),
        "pin01": pinball[:, 0],
        "pin05": pinball[:, 1],
        "pin95": pinball[:, 9],
        "pin99": pinball[:, 10],
        "below05": (actual <= q["q05"]).astype(np.float64),
        "below95": (actual <= q["q95"]).astype(np.float64),
        "inside90": ((actual >= q["q05"]) & (actual <= q["q95"])).astype(np.float64),
        "width": q["q95"] - q["q05"],
        "abs_median": np.abs(actual - q["q50"]),
    }


def month_blocks(
    *, month: np.ndarray, values: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-month sums of `values`, of `base`, and the per-month row counts."""
    labels, inverse = np.unique(month, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(labels))
    base_sums = np.bincount(inverse, weights=base, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(np.float64)
    return sums, base_sums, counts


def percent_change(
    *, month: np.ndarray, arm: np.ndarray, reference: np.ndarray, level: float, seed: int
) -> dict:
    """Return the arm's percentage change from the reference with a paired month-block interval."""
    sums, base_sums, _ = month_blocks(month=month, values=arm - reference, base=reference)
    point = 100.0 * sums.sum() / base_sums.sum()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(N_DRAWS, len(sums)))
    draws = 100.0 * sums[picks].sum(axis=1) / base_sums[picks].sum(axis=1)
    tail = (100.0 - level) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return {"point": float(point), "lower": float(low), "upper": float(high)}
