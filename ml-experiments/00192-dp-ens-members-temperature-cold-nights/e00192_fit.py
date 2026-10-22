# ruff: noqa: E501
"""Helpers shared by experiment 192's arm builder: the member mixture and its pinball loss."""

from typing import Final

import e00192_data as data
import numpy as np
from studies.ens_members import mixture_quantiles

ARMS: Final[tuple[str, ...]] = ("a", "m", "mc", "f", "s", "sp", "mr", "mp")
FACTOR_GRID: Final[np.ndarray] = np.round(np.arange(0.5, 1.3001, 0.05), 2)
FACTOR_FIT_ROWS: Final[int] = 20_000
SCALE_FIT_ROWS: Final[int] = 150_000
FIT_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q10", "q25", "q75", "q90", "q95", "q99")
KNOT_PERCENTILES: Final[tuple[float, ...]] = (25.0, 50.0, 75.0, 90.0)
MINIMUM_OFFSET: Final[float] = 1e-3
INTERCEPT_RANGE: Final[tuple[float, float]] = (0.3, 3.0)
COEFFICIENT_MAX: Final[float] = 5.0
SEED: Final[int] = 184
CHUNK: Final[int] = 4000


def fill_missing(*, delta: np.ndarray) -> np.ndarray:
    """Return the member deltas with an unusable member's entry set to the row's mean of the rest."""
    mean = np.nanmean(delta, axis=1, keepdims=True)
    return np.where(np.isfinite(delta), delta, mean)


def mixture(
    *, mean: np.ndarray, delta: np.ndarray, offsets: np.ndarray, factor: np.ndarray
) -> np.ndarray:
    """Return the 11 mixture quantiles of the members' forecasts, each with scaled offsets around it.

    Args:
        mean: Arm A's mean forecast, shape (rows,).
        delta: Each member's response, shape (rows, members), with no missing value.
        offsets: Arm A's 11 quantiles minus its mean, shape (rows, 11).
        factor: The offsets' factor per row, shape (rows,).

    Returns:
        The mixture's 11 quantiles, shape (rows, 11).
    """
    out = np.empty((len(mean), len(data.QUANTILES)))
    for start in range(0, len(mean), CHUNK):
        part = slice(start, start + CHUNK)
        centres = (mean[part, None] + delta[part])[:, :, None]
        members = centres + factor[part, None, None] * offsets[part, None, :]
        out[part] = mixture_quantiles(member_quantiles=members)
    return out


def pinball_rows(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return each row's pinball loss averaged over the 11 quantiles."""
    error = actual[:, None] - quantiles
    return np.mean(np.maximum(data.LEVELS * error, (data.LEVELS - 1.0) * error), axis=1)
