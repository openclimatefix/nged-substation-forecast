"""Cluster bootstrap of a mean over units, at the experiment's Bonferroni level."""

import e00073_common as common
import numpy as np


def cluster_interval(*, values: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """Return the mean of `values` and its interval from resampling whole clusters.

    Args:
        values: One value per unit.
        clusters: The cluster label of each unit.

    Returns:
        The mean, and the lower and upper limits at `common.LEVEL`.
    """
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    labels = np.unique(clusters)
    sums = np.array([values[clusters == c].sum() for c in labels])
    counts = np.array([(clusters == c).sum() for c in labels])
    picks = rng.integers(0, len(labels), size=(common.BOOTSTRAP_DRAWS, len(labels)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    tail = (1.0 - common.LEVEL) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return float(values.mean()), float(low), float(high)


def as_float(value: object) -> float:
    """Return a Polars scalar (typed as a wide union) as a float."""
    return float(value)  # ty: ignore[invalid-argument-type]
