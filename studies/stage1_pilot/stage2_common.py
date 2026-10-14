"""Shared constants and the mixture, blend, and score arithmetic of the stage 2 pilot (study 2b).

Every distribution in this study is 9 quantile levels of power over effective capacity. A pooled
mixture treats each member's 9 quantiles as atoms (see `ATOM_MASSES`) and reads the mixture's
quantile function on a 199-level grid; values beyond the lowest or highest atom take that atom's
value, so the tails never extrapolate. The CRPS estimate is twice the mass-weighted mean pinball
loss over the 9 levels.
"""

from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import stage1_inputs as inputs
from scipy.stats import beta as beta_distribution

STAGE2_ROOT: Final[Path] = inputs.OUTPUT_ROOT.parent / "stage2_pilot"


def stage2_subfolder(*, name: str) -> Path:
    """Return one of study 2b's output folders, such as `scores` or `results`.

    Args:
        name: The folder's name under study 2b's output root.

    Returns:
        The folder's path.
    """
    return STAGE2_ROOT / name


STAGE1_SCORE_DIR: Final[Path] = inputs.study_subfolder(name="scores")
ENS_FEATURES_DIR: Final[Path] = inputs.INPUTS_DIR / "ens_features"
ANALYSIS_PATH: Final[Path] = inputs.INPUTS_DIR / "analysis_features.parquet"

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
MEDIAN_INDEX: Final[int] = 4
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass each level carries: the interval between the mid-points to its neighbours."""
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""
MEMBERS: Final[int] = 51
MIN_SCALE: Final[float] = 0.01
"""The calibrator's scale floor, as a fraction of capacity."""
BLEND_GRID: Final[tuple[float, ...]] = (0.7, 0.85, 1.0, 1.2, 1.5)
"""Candidate values of each beta-transform shape parameter."""


def band_name(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead band name of each lead in hours."""
    names = np.full(lead_hours.shape, "none", dtype=object)
    for name, low, high in inputs.LEAD_BANDS:
        names[(lead_hours >= low) & (lead_hours < high)] = name
    return names


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's mixture quantile function on `FINE_LEVELS`.

    Args:
        member_quantiles: Shape (cases, members, 9) member quantiles, each member's levels sorted.
        chunk: Cases sorted at a time.

    Returns:
        Shape (cases, 199) float32 quantiles.
    """
    cases, members, levels = member_quantiles.shape
    weights = np.tile(ATOM_MASSES / members, members)
    out = np.empty((cases, FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            out[start + row] = np.interp(FINE_LEVELS, midpoint[row], values[row])
    return out


def read_levels(*, fine: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Return the quantiles of each row of `fine` at arbitrary levels, by linear interpolation.

    Levels outside the grid take the end values (the tail rule).
    """
    position = np.clip(levels * 200 - 1, 0, FINE_LEVELS.size - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def blended_levels(*, fine: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Return the 9 output quantiles after the beta transform with shape parameters alpha, beta.

    The blended distribution is `B(F)`, so its quantile at level `t` is the mixture quantile at
    level `B^-1(t)`. Both parameters equal to 1 returns the mixture itself.
    """
    levels = beta_distribution.ppf(LEVELS, alpha, beta)
    return read_levels(fine=fine, levels=levels)


def crps_estimate(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case.

    Args:
        quantiles: Shape (cases, 9).
        actual: Shape (cases,).

    Returns:
        The estimate per case, shape (cases,).
    """
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def scoring_cases(*, seed: int) -> pl.DataFrame:
    """Return study 2a's scored cases (keys, lead, observed fraction of capacity) for one seed."""
    return (
        pl.read_parquet(STAGE1_SCORE_DIR / f"rows_seed{seed}.parquet")
        .filter(pl.col("arm") == "ens_trained")
        .select("time_series_id", "power_fcst_init_time", "valid_time", "lead_hours", "actual_norm")
    )
