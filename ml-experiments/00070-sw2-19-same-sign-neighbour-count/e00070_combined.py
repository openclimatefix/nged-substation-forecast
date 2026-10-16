"""The combined score: a series' own 24-hour step plus its neighbours' opposite-sign steps.

A candidate is a local peak of one series' 24-hour step statistic at or above `CANDIDATE_Z` noise
units. Each candidate has two statistics, `x1` (its own step) and `x2` (the largest opposite-sign
neighbour step within one half-hour), and topology numbers. Two scores turn them into one number:
LR, a sum of binned log-likelihood ratios, and LOGIT, a logistic regression. Nothing here reads a
switching log except the explained-alarm mask that the negatives and the alarm budget use.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np
import polars as pl
from e00070_common import ADJACENCY, ML_EXPERIMENTS_DATA_DIR
from e00070_detectors import DetectorInputs, alarms
from e00070_inject import Event
from e00070_score import ONSET_TOLERANCE

CANDIDATE_Z: Final[float] = 2.5
SAME_SIGN_Z: Final[float] = 2.0
"""A same-sign neighbour step above this many noise units counts as common mode (post hoc arm)."""
X1_EDGES: Final[tuple[float, ...]] = (2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0)
X2_EDGES: Final[tuple[float, ...]] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
PSEUDO_COUNT: Final[float] = 0.5
RIDGE: Final[float] = 1.0
TIE_BREAK: Final[float] = 1e-3
"""Added to a score as `TIE_BREAK * (x1 + x2)`, so that binned scores rarely tie."""
METADATA_PATH: Final = ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "metadata.parquet"
FEATURES: Final[tuple[str, ...]] = ("x1", "x2", "n_neighbours", "mva_neighbour")
FEATURES_SAME: Final[tuple[str, ...]] = (*FEATURES, "n_same")
SCORES: Final[tuple[str, ...]] = ("LR", "LOGIT", "LOGIT_SAME")


@cache
def mva_series() -> frozenset[int]:
    """The series ids metered only in MVA: a magnitude with no direction."""
    metadata = pl.read_parquet(METADATA_PATH, columns=["time_series_id", "units"])
    return frozenset(
        metadata.filter(pl.col("units").cast(pl.String) == "MVA")["time_series_id"].to_list()
    )


def candidates(*, inputs: DetectorInputs) -> pl.DataFrame:
    """Every candidate with its statistics: series, stamp, step_mw, x1, x2, n_same, topology."""
    table = alarms(inputs=inputs, detector="D0", threshold=CANDIDATE_Z)
    mva = mva_series()
    rows = []
    for series, stamp, z in table.select("series", "stamp", "z").iter_rows():
        sign = np.sign(z)
        neighbours = [n for n in ADJACENCY.get(series, ()) if n in inputs.shift]
        best, same = 0.0, 0
        for neighbour in neighbours:
            segment = inputs.shift[neighbour][max(stamp - 1, 0) : stamp + 2]
            if not np.isfinite(segment).any():
                continue
            scale = inputs.scale[neighbour]
            best = max(best, float(-sign * segment[np.nanargmax(-sign * segment)]) / scale)
            same += int(float(sign * segment[np.nanargmax(sign * segment)]) / scale > SAME_SIGN_Z)
        rows.append(
            {
                "x1": abs(float(z)),
                "x2": best,
                "n_same": same,
                "n_neighbours": len(neighbours),
                "mva_neighbour": float(any(n in mva for n in neighbours)),
            }
        )
    extra = pl.DataFrame(
        rows,
        schema={
            "x1": pl.Float64,
            "x2": pl.Float64,
            "n_same": pl.Int64,
            "n_neighbours": pl.Int64,
            "mva_neighbour": pl.Float64,
        },
    )
    return pl.concat([table.select("series", "stamp", "step_mw"), extra], how="horizontal")


def positive_flags(*, table: pl.DataFrame, events: list[Event]) -> np.ndarray:
    """True for candidates on a series of an event, within `ONSET_TOLERANCE` of its onset."""
    flags = np.zeros(table.height, dtype=bool)
    series, stamps = table["series"].to_numpy(), table["stamp"].to_numpy()
    for event in events:
        flags |= np.isin(series, event.series) & (np.abs(stamps - event.onset) <= ONSET_TOLERANCE)
    return flags


@dataclass(frozen=True)
class LikelihoodRatio:
    """Binned log-likelihood ratios of x1 and x2, each non-decreasing in its statistic."""

    x1: np.ndarray
    x2: np.ndarray

    def score(self, *, table: pl.DataFrame) -> np.ndarray:
        """The sum of the two ratios plus the tie-break."""
        a = self.x1[np.digitize(table["x1"].to_numpy(), X1_EDGES[1:])]
        b = self.x2[np.digitize(table["x2"].to_numpy(), X2_EDGES[1:])]
        return a + b + TIE_BREAK * (table["x1"].to_numpy() + table["x2"].to_numpy())


def _binned_llr(
    *, positive: np.ndarray, negative: np.ndarray, edges: tuple[float, ...]
) -> np.ndarray:
    """Log-likelihood ratio per bin, with a pseudo-count, made non-decreasing by a running max."""
    n_bins = len(edges)
    pos = np.bincount(np.digitize(positive, edges[1:]), minlength=n_bins) + PSEUDO_COUNT
    neg = np.bincount(np.digitize(negative, edges[1:]), minlength=n_bins) + PSEUDO_COUNT
    return np.maximum.accumulate(np.log(pos / pos.sum()) - np.log(neg / neg.sum()))


def fit_likelihood_ratio(*, positive: pl.DataFrame, negative: pl.DataFrame) -> LikelihoodRatio:
    """Fit LR from positive and negative candidates."""
    return LikelihoodRatio(
        x1=_binned_llr(
            positive=positive["x1"].to_numpy(), negative=negative["x1"].to_numpy(), edges=X1_EDGES
        ),
        x2=_binned_llr(
            positive=positive["x2"].to_numpy(), negative=negative["x2"].to_numpy(), edges=X2_EDGES
        ),
    )


@dataclass(frozen=True)
class Logistic:
    """A fitted logistic regression on standardised features."""

    features: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float

    def score(self, *, table: pl.DataFrame) -> np.ndarray:
        """The log-odds plus the tie-break."""
        x = (table.select(self.features).to_numpy().astype(float) - self.mean) / self.std
        return x @ self.coef + self.intercept + TIE_BREAK * (table["x1"] + table["x2"]).to_numpy()


def fit_logistic(
    *, positive: pl.DataFrame, negative: pl.DataFrame, features: tuple[str, ...]
) -> Logistic:
    """Fit an L2-penalised logistic regression with equal total weight on the two classes."""
    x = np.vstack([positive.select(features).to_numpy(), negative.select(features).to_numpy()])
    x = x.astype(float)
    y = np.concatenate([np.ones(positive.height), np.zeros(negative.height)])
    weight = np.concatenate(
        [
            np.full(positive.height, 0.5 / positive.height),
            np.full(negative.height, 0.5 / negative.height),
        ]
    ) * len(y)
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-9)
    design = np.hstack([(x - mean) / std, np.ones((len(y), 1))])
    beta = np.zeros(design.shape[1])
    penalty = np.diag([RIDGE] * (design.shape[1] - 1) + [0.0])
    for _ in range(100):
        p = 1 / (1 + np.exp(-design @ beta))
        gradient = design.T @ (weight * (p - y)) + penalty @ beta
        hessian = (design * (weight * p * (1 - p))[:, None]).T @ design + penalty
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return Logistic(features, mean, std, beta[:-1], float(beta[-1]))


def threshold_for_count(*, scores: np.ndarray, target: int) -> float:
    """The lowest threshold such that at most `target` of `scores` are above it."""
    if len(scores) <= target:
        return -np.inf
    return float(np.sort(scores)[::-1][target])
