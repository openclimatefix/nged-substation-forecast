"""Paths and the shift-fitting functions shared by the scripts of experiment 00340.

The scripts import experiment 00336's modules (which import experiments 00326, 00303, 00285,
00259, and 00230), so run them with
`PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00326-wind-research-default-1-change-arms:ml-experiments/00336-wind-research-default-1-below-cut-in-zeroing`.
Everything read from other experiments' folders is read only.
"""

from pathlib import Path
from typing import Final

import e00336_common as c336  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c303 = c336.c303
EXPERIMENT_ID: Final[int] = 340
SLUG: Final[str] = "wind-research-default-1-constant-shift-then-low-wind-drop"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
FINAL_DIR: Final[Path] = WORK_DIR / "final"
FARMS: Final[tuple[str, ...]] = c336.FARMS
SEEDS: Final[tuple[int, ...]] = c336.SEEDS
REPLICATE_START = c336.REPLICATE_START
GRID_STEP: Final[float] = 0.0001
"""The shift grid: 0.01 points of capacity."""
COARSE_STEP: Final[float] = 0.0005
SEARCH_RANGE: Final[tuple[float, float]] = (-0.05, 0.08)
"""Search range of the shift as a fraction of capacity (-5 to +8 points)."""


def shifted(*, quantiles: np.ndarray, shift: float | np.ndarray) -> np.ndarray:
    """Return `quantiles` plus `shift` (a constant or one value per level), clipped and sorted."""
    return np.sort(np.clip(quantiles + shift, 0.0, 1.0), axis=1)


def _score(*, quantiles: np.ndarray, actual: np.ndarray, shift: float) -> float:
    return float(
        c303.quantile_score(
            quantiles=shifted(quantiles=quantiles, shift=shift), actual=actual
        ).mean()
    )


def fit_constant_shift(*, quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Return the one constant (fraction of capacity) that minimises the headline quantile score.

    A coarse grid over `SEARCH_RANGE`, then a fine grid of `GRID_STEP` around the coarse minimum.
    The score is the nine-level mass-weighted quantile score of experiment 33.

    Args:
        quantiles: Forecast quantiles (rows, 11) after the spread factor and the offset table.
        actual: Observed power as a fraction of capacity (rows,).

    Returns:
        The fitted constant.
    """
    quantiles = quantiles.astype(np.float64)
    coarse = np.arange(SEARCH_RANGE[0], SEARCH_RANGE[1] + COARSE_STEP, COARSE_STEP)
    best = min(coarse, key=lambda s: _score(quantiles=quantiles, actual=actual, shift=float(s)))
    fine = np.arange(best - COARSE_STEP, best + COARSE_STEP + GRID_STEP, GRID_STEP)
    best_fine = min(fine, key=lambda s: _score(quantiles=quantiles, actual=actual, shift=float(s)))
    return round(float(best_fine), 4)  # on the grid, so a zero shift is exactly zero


def fit_level_shifts(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return one constant per quantile level: the level's quantile of (observed minus quantile k).

    Each constant is the pinball-optimal offset of its level over all rows, the pooled version of
    the (farm, lead day) offset table of experiment 259.
    """
    residual = actual[:, None] - quantiles.astype(np.float64)
    return np.array([np.quantile(residual[:, k], level) for k, level in enumerate(c303.LEVELS)])


def known_answer_test() -> dict[str, float]:
    """Check that the shift fit recovers a known shift from synthetic quantiles.

    Observed power is drawn from a Beta distribution; the "forecast" is its true quantiles minus a
    known 0.012 (1.2 points). Both fits must return +0.012 within 0.001.
    """
    rng = np.random.default_rng(340)
    actual = rng.beta(2.0, 5.0, size=60_000)
    true_levels = np.quantile(actual, c303.LEVELS)
    quantiles = np.tile(true_levels - 0.012, (actual.size, 1))
    constant = fit_constant_shift(quantiles=quantiles, actual=actual)
    per_level = fit_level_shifts(quantiles=quantiles, actual=actual)
    return {
        "injected": 0.012,
        "constant_fit": constant,
        "per_level_fit_mean": float(per_level.mean()),
        "per_level_fit_max_abs_error": float(np.abs(per_level - 0.012).max()),
    }
