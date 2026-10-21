"""Constants shared by the scripts of experiment 152 (the fold-aware GAM-K at the MVA Primaries).

Substations appear by D-label only. Experiment 110's cache and stored predictions are read from
their private folders and never rewritten.
"""

from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, default_input_paths, load_or_build_feature_cache
from studies.dp_gam import QUANTILE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 152
SLUG: Final[str] = "dp-mva-fold-aware-gam-k-at-mva-only-primaries"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
EXPERIMENT_110_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00110-differentiable-physics-5-embedded-generation-and-capacity"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
"""Part 4's feature cache, shared by experiments 110, 124, and 136 and read only here."""
QUANTILE_LEVELS_ARRAY: Final[tuple[float, ...]] = QUANTILE_LEVELS

MVA_ONLY_PRIMARIES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""The 8 Primaries whose series is metered only in MVA."""
SUNNY_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""April to September, as experiment 56."""
MIDDAY_FIRST_HALF_HOUR_UTC: Final[float] = 10.0
MIDDAY_LAST_HALF_HOUR_UTC: Final[float] = 13.5
"""The half-hours that start between 10:00 and 13:30 UTC, as experiment 56."""
CLEAR_SKY_INDEX_MINIMUM: Final[float] = 0.7
"""A sunny half-hour has CAMS irradiance at least this share of the Haurwitz clear-sky value."""


def load_cache() -> FeatureCache:
    """Return the shared feature cache, building and saving it on the first call."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def quantile_score(*, actual: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row, in points.

    Args:
        actual: Measured values, shape (n,).
        quantiles: Predicted quantiles, shape (n, 11).

    Returns:
        The quantile score of each row.
    """
    levels = np.array(QUANTILE_LEVELS)[None, :]
    error = actual[:, None] - quantiles
    return 2.0 * np.mean(np.maximum(levels * error, (levels - 1.0) * error), axis=1)
