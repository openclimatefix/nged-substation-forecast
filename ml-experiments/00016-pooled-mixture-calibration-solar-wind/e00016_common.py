"""Shared constants and arithmetic of experiment 00016: the pooled mixture and its corrections.

The mixture arithmetic copies `studies/stage1_pilot/stage2_common.py` (a study script, which an
experiment may not import). A distribution here is 9 quantile levels of power over effective
capacity, and the pooled mixture treats each ensemble member's 9 quantiles as atoms of probability.
"""

from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 16
SLUG: Final[str] = "pooled-mixture-calibration-solar-wind"
ROW_SET_ID: Final[str] = "fold_a_tuesdays_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MEMBERS: Final[int] = 51
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
CASE_ORDER: Final[tuple[str, ...]] = ("power_fcst_init_time", "time_series_id", "valid_time")

STAGE2_DIR: Final[Path] = PER_STUDY_DIR / "stage2_pilot"
MEMBER_QUANTILES_DIR: Final[Path] = STAGE2_DIR / "stage1a_q" / "full" / "val"
SCORES_DIR: Final[Path] = STAGE2_DIR / "scores"
ENS_VALIDATION_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot" / "inputs" / "ens_features" / "val"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"

TechnologyType = Literal["solar", "wind"]
TECHNOLOGIES: Final[tuple[TechnologyType, ...]] = ("solar", "wind")
ArmType = Literal["none", "spread", "spread_band", "locscale", "free_quantile", "spread_two_groups"]
ARMS: Final[tuple[ArmType, ...]] = ("none", "spread", "spread_band", "locscale", "free_quantile")
POST_HOC_ARM: Final[ArmType] = "spread_two_groups"
"""The post hoc arm of `e00016_posthoc.py`, outside the planned contrasts."""

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
LOWER_INDEX: Final[int] = 0
UPPER_INDEX: Final[int] = 8
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass each level carries: the interval between the mid-points to its neighbours."""
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""

LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0 to 24 h", 0.0, 24.0),
    ("1 to 2 days", 24.0, 72.0),
    ("3 to 6 days", 72.0, 168.0),
    ("7 to 10 days", 168.0, 264.0),
    ("11 to 15 days", 264.0, 10_000.0),
)
MIN_FIT_CASES: Final[int] = 5_000
MAX_FIT_CASES: Final[int] = 30_000
"""A fit needs at least this many cases; a locscale fit subsamples to at most this many."""
TRIVIAL_LEVEL: Final[float] = 0.01
"""A case is trivial when the uncorrected 95% quantile is at most this fraction of capacity."""
TARGET_COVERAGE: Final[float] = 0.90
SPREAD_GRID: Final[np.ndarray] = np.round(np.arange(0.60, 1.601, 0.05), 2)
SHIFT_GRID: Final[np.ndarray] = np.round(np.arange(-0.05, 0.0501, 0.01), 2)
SUBSAMPLE_SEED: Final[int] = 20261014


def band_names(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the lead band name of each lead in hours."""
    names = np.full(lead_hours.shape, "none", dtype=object)
    for name, low, high in LEAD_BANDS:
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


def read_levels(*, fine: np.ndarray) -> np.ndarray:
    """Return the 9 output quantiles of each row of `fine`, by linear interpolation."""
    position = np.clip(LEVELS * 200 - 1, 0, FINE_LEVELS.size - 1)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def crps_estimate(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case (a fraction)."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observed power lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])


def scale_about_median(*, quantiles: np.ndarray, shift: float, factor: float) -> np.ndarray:
    """Return `median + shift + factor * (q - median)`, clipped to 0 to 1 and sorted."""
    median = quantiles[:, MEDIAN_INDEX : MEDIAN_INDEX + 1]
    scaled = median + shift + factor * (quantiles - median)
    return np.sort(np.clip(scaled, 0.0, 1.0), axis=1)


def add_level_offsets(*, quantiles: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return the quantiles plus one offset per level, clipped to 0 to 1 and sorted."""
    return np.sort(np.clip(quantiles + offsets, 0.0, 1.0), axis=1)


def fit_spread_factor(*, quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Return the grid factor whose 5% to 95% coverage is nearest `TARGET_COVERAGE`."""
    gaps = [
        abs(
            covered(
                quantiles=scale_about_median(quantiles=quantiles, shift=0.0, factor=float(k)),
                actual=actual,
            ).mean()
            - TARGET_COVERAGE
        )
        for k in SPREAD_GRID
    ]
    return float(SPREAD_GRID[int(np.argmin(gaps))])


def fit_location_and_scale(*, quantiles: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    """Return the grid `(shift, factor)` with the lowest mean CRPS estimate."""
    best, choice = np.inf, (0.0, 1.0)
    for shift in SHIFT_GRID:
        for factor in SPREAD_GRID:
            score = crps_estimate(
                quantiles=scale_about_median(
                    quantiles=quantiles, shift=float(shift), factor=float(factor)
                ),
                actual=actual,
            ).mean()
            if score < best:
                best, choice = score, (float(shift), float(factor))
    return choice


def fit_level_offsets(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return, per level, the empirical quantile of the observed power minus that level's value."""
    residual = actual[:, None] - quantiles
    return np.array([np.quantile(residual[:, i], LEVELS[i]) for i in range(LEVELS.size)])


def technology_table() -> pl.DataFrame:
    """Return `time_series_id` and `technology` for the 6 solar farms and 3 wind farms."""
    return pl.concat(
        [
            pv_sites().select("time_series_id", technology=pl.lit("solar")),
            wind_sites().select("time_series_id", technology=pl.lit("wind")),
        ]
    )


def validation_cases() -> pl.DataFrame:
    """Return experiment 00002's scored cases with technology and lead band, in mixture order.

    Columns: the key columns, `lead_hours`, `actual_norm` (a fraction of capacity), `technology`,
    and `band`. Sorted by `CASE_ORDER`, the order in which the mixture is built.
    """
    cases = (
        pl.read_parquet(SCORES_DIR / "rows_seed0.parquet")
        .filter(pl.col("arm") == "b_mixture_full")
        .select(*KEY_COLUMNS, "lead_hours", "actual_norm")
        .join(technology_table(), on="time_series_id", how="left", validate="m:1")
        .sort(*CASE_ORDER)
    )
    return cases.with_columns(band=pl.Series(band_names(lead_hours=cases["lead_hours"].to_numpy())))
