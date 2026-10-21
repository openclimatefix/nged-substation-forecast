"""Scores for a forecast given as a set of weighted values: a rank histogram and two CRPS variants.

An ensemble forecast, or a mixture of quantiles, is a finite set of values `x_1 … x_m` with
weights that sum to 1. Three scores read such a set directly:

- `rank_histogram` counts where the measured value falls among the sorted values. A flat histogram
  means the measured value is exchangeable with the members; a U shape means the spread is too
  small.
- `fair_crps` is the continuous ranked probability score of an equally weighted ensemble, corrected
  so that its expected value does not change with the number of members
  ([Ferro (2014)](https://doi.org/10.1002/qj.2270)).
- `tail_weighted_crps` is the threshold-weighted CRPS of
  [Gneiting and Ranjan (2011)](https://doi.org/10.1198/jbes.2010.08110) with an indicator weight on
  one tail. The threshold-weighted CRPS equals the CRPS of the values after clamping them at a
  threshold, so a forecast is scored only on the tail it was asked about.

All three take one row per case, so a `(n_cases, m)` array of values scored against `n_cases`
measured values. A score is lower when it is better, and every score here is proper
([Gneiting and Raftery (2007)](https://doi.org/10.1198/016214506000001437)).
"""

from typing import Final, Literal

import numpy as np

type Tail = Literal["upper", "lower"]
"""Which tail a threshold-weighted score looks at: `upper` is values above the threshold."""

WEIGHT_SUM_TOLERANCE: Final[float] = 1e-9
"""How far the atom weights may sum from 1 before `weighted_crps` raises."""


def rank_histogram(
    *, members: np.ndarray, actual: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Count how often the measured value has each rank among the members.

    The rank of a case is the number of members strictly below the measured value, plus a uniformly
    random share of the members equal to it, so ties do not bias the histogram.

    Args:
        members: Array of shape `(n_cases, m)`, the member values of each case.
        actual: Array of shape `(n_cases,)`, the measured value of each case.
        rng: The random number generator that breaks ties.

    Returns:
        Integer array of length `m + 1`; entry `r` counts the cases with rank `r`.

    Raises:
        ValueError: If `members` or `actual` holds a NaN.
    """
    if np.isnan(members).any() or np.isnan(actual).any():
        raise ValueError("rank_histogram does not accept NaN; drop those cases first")
    below = (members < actual[:, None]).sum(axis=1)
    equal = (members == actual[:, None]).sum(axis=1)
    rank = below + np.floor(rng.random(size=len(actual)) * (equal + 1)).astype(np.int64)
    return np.bincount(rank, minlength=members.shape[1] + 1)


def weighted_crps(*, values: np.ndarray, weights: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the CRPS of weighted atoms, `E|X − y| − ½ E|X − X'|`, for each case.

    Args:
        values: Array of shape `(n_cases, m)`, the atom values of each case.
        weights: Array of shape `(m,)` of non-negative weights that sum to 1, shared by all cases.
        actual: Array of shape `(n_cases,)`, the measured value of each case.

    Returns:
        Array of shape `(n_cases,)`, the score of each case in the unit of `values`.

    Raises:
        ValueError: If the weights are negative or do not sum to 1.
    """
    if (weights < 0).any() or abs(weights.sum() - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError("weights must be non-negative and sum to 1")
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = weights[order]
    first_term = (sorted_weights * np.abs(sorted_values - actual[:, None])).sum(axis=1)
    # For sorted atoms, ½ Σ_i Σ_j w_i w_j |x_i − x_j| = Σ_i w_i x_i (2 W_i − w_i − 1), where W_i is
    # the cumulative weight up to and including atom i.
    cumulative = np.cumsum(sorted_weights, axis=1)
    half_spread = (sorted_weights * sorted_values * (2.0 * cumulative - sorted_weights - 1.0)).sum(
        axis=1
    )
    return first_term - half_spread


def fair_crps(*, members: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return the fair CRPS of an equally weighted ensemble for each case.

    The fair score is `mean|x_i − y| − Σ_i Σ_j |x_i − x_j| / (2 m (m − 1))`. It divides by
    `m (m − 1)` and not `m²`, so a member drawn from the same distribution as the measured value
    scores no better than the true distribution does, whatever the number of members.

    Args:
        members: Array of shape `(n_cases, m)` with `m >= 2`.
        actual: Array of shape `(n_cases,)`, the measured value of each case.

    Returns:
        Array of shape `(n_cases,)`, the score of each case in the unit of `members`.

    Raises:
        ValueError: If there are fewer than two members.
    """
    m = members.shape[1]
    if m < 2:
        raise ValueError("the fair CRPS needs at least two members")
    plain = weighted_crps(values=members, weights=np.full(m, 1.0 / m), actual=actual)
    # `plain` subtracts ½ Σ Σ |x_i − x_j| / m². Replace m² by m (m − 1) in that term.
    mean_abs = np.abs(members - actual[:, None]).mean(axis=1)
    half_spread_plain = mean_abs - plain
    return mean_abs - half_spread_plain * m / (m - 1)


def tail_weighted_crps(
    *,
    values: np.ndarray,
    weights: np.ndarray,
    actual: np.ndarray,
    threshold: np.ndarray | float,
    tail: Tail,
) -> np.ndarray:
    """Return the CRPS of weighted atoms after clamping at a threshold, for each case.

    With the indicator weight `w(z) = 1{z ≥ t}` the threshold-weighted CRPS is the CRPS of the
    values `max(z, t)`; with `w(z) = 1{z ≤ t}` it is the CRPS of `min(z, t)`. Values on the other
    side of the threshold all become the threshold, so they cannot change the score.

    Args:
        values: Array of shape `(n_cases, m)`, the atom values of each case.
        weights: Array of shape `(m,)` of non-negative weights that sum to 1.
        actual: Array of shape `(n_cases,)`, the measured value of each case.
        threshold: The clamping value `t`, one number or one per case.
        tail: `upper` to score values above the threshold, `lower` to score values below it.

    Returns:
        Array of shape `(n_cases,)`, the score of each case in the unit of `values`.

    Raises:
        ValueError: If `tail` is neither `upper` nor `lower`.
    """
    if tail not in ("upper", "lower"):
        raise ValueError(f"tail must be 'upper' or 'lower', not {tail!r}")
    t = np.asarray(threshold, dtype=float)
    t_col = t[:, None] if t.ndim == 1 else t
    clamp = np.maximum if tail == "upper" else np.minimum
    return weighted_crps(
        values=clamp(values, t_col),
        weights=weights,
        actual=clamp(actual, t),
    )
