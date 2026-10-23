"""Quantiles of a sum of forecasts from each part's quantiles, under four assumptions of dependence.

The caller holds, for each part (for example a solar farm), `n` rows of quantile forecasts at the
same ascending levels, as an `(n, n_levels)` array in the same unit. Each function returns the
quantiles of the parts' sum at `levels`, as an `(n, len(levels))` array. The four forms differ only
in how the parts' errors are assumed to move together:

- `sum_of_quantiles`: perfect dependence (every part sits at the same quantile level).
- `independent_sum_quantiles`: independence (each part draws its own level).
- `member_wise_sum_quantiles`: the dependence of an ensemble of weather forecasts. Each part's
  ensemble members are ranked within the part, the rank becomes a level, and the part's quantile
  function maps the level to a value. The values are summed member by member.
- `copula_sum_quantiles`: a Gaussian copula with a given correlation between parts.

`pit_normal_scores` turns an observation into the normal score of its level in the forecast's
quantile function, from which a copula correlation is fitted.
"""

from collections.abc import Sequence
from typing import Final

import numpy as np
from scipy.stats import norm, rankdata

DEFAULT_LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
"""The eleven levels the metered-farm experiments store."""

DEFAULT_DRAWS_PER_MEMBER: Final[int] = 40
"""Draws inside each ensemble member's rank interval, so that 51 members give 2,040 samples."""

_CHUNK_ROWS: Final[int] = 500
"""Rows processed at once, which bounds the memory of the sample arrays."""

_PIT_LIMIT: Final[float] = 0.001


def _check_parts(*, parts: Sequence[np.ndarray], n_levels: int) -> None:
    """Raise unless every part is an `(n, n_levels)` array with the same number of rows."""
    if len(parts) < 1:
        msg = "at least one part is needed"
        raise ValueError(msg)
    for part in parts:
        if part.ndim != 2 or part.shape != (parts[0].shape[0], n_levels):
            msg = f"every part needs shape (n, {n_levels}), got {part.shape}"
            raise ValueError(msg)


def inverse_cdf(
    *,
    quantiles: np.ndarray,
    levels: Sequence[float],
    uniforms: np.ndarray,
    lower: float = 0.0,
    upper: float | None = None,
) -> np.ndarray:
    """Evaluate each row's piecewise-linear quantile function at `uniforms`.

    Between the stored levels the function is linear. Beyond the first and last level it continues
    along the slope of the nearest stored interval, clipped to `[lower, upper]`.

    Args:
        quantiles: `(n, n_levels)` quantiles per row; sorted ascending along the last axis here.
        levels: The `n_levels` ascending levels strictly inside (0, 1).
        uniforms: `(n, k)` levels in (0, 1) to evaluate per row.
        lower: Smallest value the function returns.
        upper: Largest value the function returns, or None for no limit.

    Returns:
        An `(n, k)` array of values.
    """
    values = np.sort(quantiles, axis=1)
    grid = np.asarray(levels, dtype=np.float64)
    out = np.empty_like(uniforms, dtype=np.float64)
    first = (values[:, 1] - values[:, 0]) / (grid[1] - grid[0])
    last = (values[:, -1] - values[:, -2]) / (grid[-1] - grid[-2])
    below = uniforms < grid[0]
    above = uniforms >= grid[-1]
    out[below] = (values[:, :1] + (uniforms - grid[0]) * first[:, None])[below]
    out[above] = (values[:, -1:] + (uniforms - grid[-1]) * last[:, None])[above]
    for j in range(len(grid) - 1):
        inside = (uniforms >= grid[j]) & (uniforms < grid[j + 1])
        step = (uniforms - grid[j]) / (grid[j + 1] - grid[j])
        segment = values[:, j : j + 1] + step * (values[:, j + 1 : j + 2] - values[:, j : j + 1])
        out[inside] = segment[inside]
    return np.clip(out, lower, upper)


def pit_normal_scores(
    *, quantiles: np.ndarray, levels: Sequence[float], actual: np.ndarray, seed: int
) -> np.ndarray:
    """Return the standard-normal score of the level at which each observation sits in its forecast.

    The level is found by interpolating between the stored quantiles. Where several stored
    quantiles tie at the observation (a plateau at zero or at a cap), the level is drawn uniformly
    across the plateau, so that a calibrated forecast gives uniform levels. Levels beyond the
    outermost stored quantiles are set to 0.001 or 0.999 of the way to the end.

    Args:
        quantiles: `(n, n_levels)` quantiles per row.
        levels: The `n_levels` ascending levels.
        actual: `(n,)` observations.
        seed: Seed of the draw across plateaus.

    Returns:
        An `(n,)` array of normal scores.
    """
    rng = np.random.default_rng(seed)
    values = np.sort(quantiles, axis=1)
    grid = np.asarray(levels, dtype=np.float64)
    n_rows, n_levels = values.shape
    obs = actual[:, None]
    n_below = (values < obs).sum(axis=1)
    n_at_most = (values <= obs).sum(axis=1)
    pit = np.empty(n_rows, dtype=np.float64)
    rows = np.arange(n_rows)
    tie = n_at_most > n_below
    low_tie = np.where(n_below > 0, grid[np.maximum(n_below - 1, 0)], 0.0)
    pit[tie] = (
        low_tie[tie]
        + rng.uniform(size=n_rows)[tie] * (grid[np.maximum(n_at_most - 1, 0)] - low_tie)[tie]
    )
    outside_low = (n_below == 0) & ~tie
    outside_high = (n_below == n_levels) & ~tie
    inner = ~tie & ~outside_low & ~outside_high
    idx = np.clip(n_below - 1, 0, n_levels - 2)
    left = values[rows, idx]
    right = values[rows, idx + 1]
    fraction = (actual - left) / np.where(right > left, right - left, 1.0)
    pit[inner] = (grid[idx] + fraction * (grid[idx + 1] - grid[idx]))[inner]
    pit[outside_low] = grid[0] * 0.5
    pit[outside_high] = 1.0 - (1.0 - grid[-1]) * 0.5
    return norm.ppf(np.clip(pit, _PIT_LIMIT, 1.0 - _PIT_LIMIT))


def sample_quantiles(*, samples: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Return each row's empirical quantiles at `levels`, as an `(n, len(levels))` array."""
    out = np.empty((samples.shape[0], len(levels)), dtype=np.float64)
    for start in range(0, samples.shape[0], _CHUNK_ROWS):
        block = slice(start, start + _CHUNK_ROWS)
        out[block] = np.quantile(samples[block], levels, axis=1).T
    return out


def sum_of_quantiles(
    *, parts: Sequence[np.ndarray], part_levels: Sequence[float], levels: Sequence[float]
) -> np.ndarray:
    """Return the sum of the parts' quantiles at `levels`, which assumes perfect dependence.

    Args:
        parts: One `(n, len(part_levels))` quantile array per part.
        part_levels: The levels of the stored quantiles.
        levels: The levels to return, each of which must be in `part_levels`.

    Returns:
        An `(n, len(levels))` array.
    """
    _check_parts(parts=parts, n_levels=len(part_levels))
    columns = [list(part_levels).index(level) for level in levels]
    return np.sum([np.sort(part, axis=1)[:, columns] for part in parts], axis=0)


def independent_sum_quantiles(
    *,
    parts: Sequence[np.ndarray],
    part_levels: Sequence[float],
    levels: Sequence[float],
    n_draws: int,
    seed: int,
    upper: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Return the quantiles of the sum when each part draws its own independent level.

    Args:
        parts: One `(n, len(part_levels))` quantile array per part.
        part_levels: The levels of the stored quantiles.
        levels: The levels to return.
        n_draws: Draws per row.
        seed: Seed of the draws.
        upper: Largest value of each part, or None for no limit.

    Returns:
        An `(n, len(levels))` array.
    """
    _check_parts(parts=parts, n_levels=len(part_levels))
    rng = np.random.default_rng(seed)
    n_rows = parts[0].shape[0]
    caps = upper if upper is not None else [None] * len(parts)
    out = np.empty((n_rows, len(levels)), dtype=np.float64)
    for start in range(0, n_rows, _CHUNK_ROWS):
        block = slice(start, start + _CHUNK_ROWS)
        total = 0.0
        for part, cap in zip(parts, caps, strict=True):
            uniforms = rng.uniform(size=(part[block].shape[0], n_draws))
            total = total + inverse_cdf(
                quantiles=part[block], levels=part_levels, uniforms=uniforms, upper=cap
            )
        out[block] = np.quantile(total, levels, axis=1).T
    return out


def member_rank_levels(
    *, member_values: np.ndarray, rng: np.random.Generator, n_draws: int
) -> np.ndarray:
    """Turn each row's ensemble members into levels, `n_draws` per member.

    A member's level is drawn uniformly inside the interval `(rank - 1, rank) / n_members` of its
    rank among the row's members (ties share the average rank), so that the levels of one row
    cover (0, 1) evenly and keep the members' order.

    Args:
        member_values: `(n, n_members)` ensemble members' values.
        rng: Source of the draws inside each rank interval.
        n_draws: Draws per member.

    Returns:
        An `(n, n_members * n_draws)` array of levels in (0, 1).
    """
    n_members = member_values.shape[1]
    ranks = rankdata(member_values, axis=1, method="average")
    centre = np.repeat(ranks, n_draws, axis=1)
    jitter = rng.uniform(-0.5, 0.5, size=centre.shape)
    return np.clip((centre + jitter) / n_members, 1e-6, 1.0 - 1e-6)


def member_wise_sum_quantiles(
    *,
    parts: Sequence[np.ndarray],
    part_levels: Sequence[float],
    members: Sequence[np.ndarray],
    levels: Sequence[float],
    seed: int,
    n_draws_per_member: int = DEFAULT_DRAWS_PER_MEMBER,
    upper: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Return the quantiles of the sum when the parts share the ensemble's dependence.

    Draw `j` of member `m` uses the level of member `m`'s rank in every part, and the parts' draws
    inside a rank interval are independent. Each row's sample of the sum holds
    `n_members * n_draws_per_member` values.

    Args:
        parts: One `(n, len(part_levels))` quantile array per part.
        part_levels: The levels of the stored quantiles.
        members: One `(n, n_members)` array of ensemble members per part, row-aligned with `parts`.
        levels: The levels to return.
        seed: Seed of the draws inside each rank interval.
        n_draws_per_member: Draws per member.
        upper: Largest value of each part, or None for no limit.

    Returns:
        An `(n, len(levels))` array.
    """
    _check_parts(parts=parts, n_levels=len(part_levels))
    rng = np.random.default_rng(seed)
    n_rows = parts[0].shape[0]
    caps = upper if upper is not None else [None] * len(parts)
    out = np.empty((n_rows, len(levels)), dtype=np.float64)
    for start in range(0, n_rows, _CHUNK_ROWS):
        block = slice(start, start + _CHUNK_ROWS)
        total = 0.0
        for part, member, cap in zip(parts, members, caps, strict=True):
            uniforms = member_rank_levels(
                member_values=member[block], rng=rng, n_draws=n_draws_per_member
            )
            total = total + inverse_cdf(
                quantiles=part[block], levels=part_levels, uniforms=uniforms, upper=cap
            )
        out[block] = np.quantile(total, levels, axis=1).T
    return out


def copula_sum_quantiles(
    *,
    parts: Sequence[np.ndarray],
    part_levels: Sequence[float],
    correlation: np.ndarray,
    levels: Sequence[float],
    n_draws: int,
    seed: int,
    upper: Sequence[float | None] | None = None,
) -> np.ndarray:
    """Return the quantiles of the sum when the parts' levels follow a Gaussian copula.

    Args:
        parts: One `(n, len(part_levels))` quantile array per part.
        part_levels: The levels of the stored quantiles.
        correlation: `(n_parts, n_parts)` correlation matrix of the parts' normal scores.
        levels: The levels to return.
        n_draws: Draws per row.
        seed: Seed of the draws.
        upper: Largest value of each part, or None for no limit.

    Returns:
        An `(n, len(levels))` array.
    """
    _check_parts(parts=parts, n_levels=len(part_levels))
    rng = np.random.default_rng(seed)
    n_rows = parts[0].shape[0]
    factor = np.linalg.cholesky(correlation)
    caps = upper if upper is not None else [None] * len(parts)
    out = np.empty((n_rows, len(levels)), dtype=np.float64)
    for start in range(0, n_rows, _CHUNK_ROWS):
        block = slice(start, start + _CHUNK_ROWS)
        rows = parts[0][block].shape[0]
        scores = rng.standard_normal(size=(rows, n_draws, len(parts))) @ factor.T
        total = 0.0
        for index, (part, cap) in enumerate(zip(parts, caps, strict=True)):
            total = total + inverse_cdf(
                quantiles=part[block],
                levels=part_levels,
                uniforms=norm.cdf(scores[:, :, index]),
                upper=cap,
            )
        out[block] = np.quantile(total, levels, axis=1).T
    return out
