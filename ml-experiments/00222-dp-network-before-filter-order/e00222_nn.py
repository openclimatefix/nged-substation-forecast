"""Quantile-score helpers shared by the scoring scripts of experiment 222."""

from typing import Final

import e00222_common as common
import numpy as np
import polars as pl

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
MEDIAN_INDEX: Final[int] = 5


def quantile_matrix(*, frame: pl.DataFrame) -> np.ndarray:
    """Return a frame's 11 quantiles, shape (rows, 11), float64."""
    return frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)


def row_scores(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return the quantile score of each row (twice the mean pinball loss), in points."""
    error = actual[:, None] - quantiles
    return 2.0 * np.maximum(LEVELS[None] * error, (LEVELS[None] - 1.0) * error).mean(axis=1)
