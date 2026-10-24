"""Paths and the fitting functions shared by the scripts of experiment 00342.

The scripts import experiment 00340's and 00336's modules (which import experiments 00326, 00303,
00285, 00259, and 00230), so run them with
`PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00326-wind-research-default-1-change-arms:ml-experiments/00336-wind-research-default-1-below-cut-in-zeroing:ml-experiments/00340-wind-research-default-1-constant-shift-then-low-wind-drop`.
Everything read from other experiments' folders is read only.
"""

from pathlib import Path
from typing import Final

import e00340_common as c340  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c336 = c340.c336
c303 = c340.c303
EXPERIMENT_ID: Final[int] = 342
SLUG: Final[str] = "wind-research-default-1-upper-tail-offsets-and-head"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
FINAL_DIR: Final[Path] = WORK_DIR / "final"
STATS_DIR: Final[Path] = WORK_DIR / "stats"
FARMS: Final[tuple[str, ...]] = c340.FARMS
SEEDS: Final[tuple[int, ...]] = c340.SEEDS
HEAD_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
REPLICATE_START = c340.REPLICATE_START
UPPER_INDEX: Final[tuple[int, ...]] = (8, 9, 10)
"""Positions of q90, q95, and q99 among the 11 levels."""
SPEED_EDGES: Final[tuple[float, ...]] = (5.0, 8.0, 11.0, 14.0, 17.0)
"""Bin edges (m/s) of the ensemble-mean 100 m speed that key the x1 offsets: six bins."""
MIN_CELL_ROWS: Final[int] = 200
CAP_PERCENTILE: Final[float] = 99.9
LAMBDA_GRID: Final[np.ndarray] = np.arange(0.0, 2.0001, 0.05)
MAX_MASS: Final[float] = 0.9
KNOT_LEVELS: Final[np.ndarray] = np.append(c303.LEVELS, 1.0)


def stats_path(*, set_name: str, farm: str) -> Path:
    """Return the path of a per-case statistics file: mean, std, and share above rated speed."""
    return STATS_DIR / f"member_{set_name}_{farm}.npy"


def farm_caps(*, actual: np.ndarray, sites: np.ndarray) -> dict[str, float]:
    """Return each farm's scale: the 99.9th percentile of its observed hindcast power.

    Args:
        actual: Hindcast observed power as a fraction of capacity.
        sites: Farm label of each case.

    Returns:
        Fraction of capacity per farm.
    """
    return {f: float(np.percentile(actual[sites == f], CAP_PERCENTILE)) for f in FARMS}


def farm_caps_strong(
    *, actual: np.ndarray, era5: np.ndarray, sites: np.ndarray, rated: dict[str, float]
) -> dict[str, float]:
    """Return the cap as the plan first defined it: the 99.9th percentile at ERA5 speed >= rated.

    The plan's "strong set" was the hindcast rows whose ERA5 100 m speed is at or above the farm's
    rated speed. The rated speeds are 16 to 22 m/s, so the set is small and its cap came out at
    53% of capacity at W3, below the forecasts; the arm that uses this cap is descriptive only.
    """
    out = {}
    for farm in FARMS:
        strong = (sites == farm) & (np.nan_to_num(era5, nan=-1.0) >= rated[farm])
        out[farm] = float(np.percentile(actual[strong], CAP_PERCENTILE))
    return out


def cap_quantiles(
    *, quantiles: np.ndarray, sites: np.ndarray, caps: dict[str, float]
) -> np.ndarray:
    """Return `quantiles` with every value limited to the case's farm scale."""
    limit = np.array([caps[s] for s in sites])[:, None]
    return np.minimum(quantiles, limit).astype(np.float32)


def speed_bin(*, mean_speed: np.ndarray) -> np.ndarray:
    """Return the x1 speed bin (0 to 5) of each case's ensemble-mean speed."""
    return np.searchsorted(np.array(SPEED_EDGES), mean_speed, side="right")


def cell_ids(*, sites: np.ndarray, lead_day: np.ndarray, mean_speed: np.ndarray) -> np.ndarray:
    """Return an integer id of each case's (farm, lead day, speed bin) cell."""
    farm = np.array([FARMS.index(s) for s in sites])
    return farm * 1000 + lead_day.astype(int) * 10 + speed_bin(mean_speed=mean_speed)


def fit_x1_table(
    *, quantiles: np.ndarray, actual: np.ndarray, cells: np.ndarray
) -> dict[int, np.ndarray]:
    """Return the offsets (3 values: q90, q95, q99) of every cell with enough rows.

    Each offset is the pinball-optimal one: the level's quantile of (observed minus quantile).
    """
    table = {}
    levels = c303.LEVELS[list(UPPER_INDEX)]
    for cell in np.unique(cells):
        rows = cells == cell
        if rows.sum() < MIN_CELL_ROWS:
            continue
        residual = actual[rows, None] - quantiles[rows][:, list(UPPER_INDEX)].astype(np.float64)
        table[int(cell)] = np.array(
            [np.quantile(residual[:, k], level) for k, level in enumerate(levels)]
        )
    return table


def apply_x1_table(
    *, quantiles: np.ndarray, cells: np.ndarray, table: dict[int, np.ndarray]
) -> np.ndarray:
    """Return `quantiles` plus the cell offsets at q90, q95, and q99, sorted along the levels."""
    out = quantiles.astype(np.float64).copy()
    for cell, offset in table.items():
        rows = cells == cell
        out[np.ix_(rows, list(UPPER_INDEX))] += offset
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def saturation_mass(*, quantiles: np.ndarray, weight: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Return the upper quantiles of (1 - w) F + w delta(top), with F interpolated from `quantiles`.

    Args:
        quantiles: Forecast quantiles (rows, 11).
        weight: Probability mass w at the top, per row.
        top: The point mass's location (the farm scale), per row.

    Returns:
        `quantiles` (rows, 11) with q90, q95, and q99 replaced by the mixture's quantiles; a
        mixture quantile at level t is F's quantile at t / (1 - w), or `top` if that exceeds 1.
    """
    out = quantiles.astype(np.float64).copy()
    weight = np.clip(weight, 0.0, MAX_MASS)
    for k in UPPER_INDEX:
        u = np.minimum(c303.LEVELS[k] / (1.0 - weight), 1.0)
        knots = np.concatenate(
            [quantiles.astype(np.float64), np.maximum(top, quantiles[:, 10])[:, None]], axis=1
        )
        position = np.clip(np.searchsorted(KNOT_LEVELS, u, side="right") - 1, 0, 10)
        rows = np.arange(u.size)
        low_level, high_level = KNOT_LEVELS[position], KNOT_LEVELS[position + 1]
        fraction = np.clip((u - low_level) / (high_level - low_level), 0.0, 1.0)
        value = knots[rows, position] * (1 - fraction) + knots[rows, position + 1] * fraction
        out[:, k] = np.where(u < KNOT_LEVELS[0], quantiles[:, 0], value)
    return np.sort(np.clip(out, 0.0, 1.0), axis=1)


def upper_pinball(*, quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Return the mean over rows of the summed pinball losses at q90, q95, and q99."""
    total = 0.0
    for k in UPPER_INDEX:
        difference = actual - quantiles[:, k]
        level = c303.LEVELS[k]
        total += float(
            np.where(difference >= 0, level * difference, (level - 1) * difference).mean()
        )
    return total


def fit_lambda(
    *, quantiles: np.ndarray, actual: np.ndarray, share: np.ndarray, top: np.ndarray
) -> float:
    """Return the x3 constant of one farm: the grid value that minimises the upper pinball loss."""
    scores = [
        upper_pinball(
            quantiles=saturation_mass(quantiles=quantiles, weight=lam * share, top=top),
            actual=actual,
        )
        for lam in LAMBDA_GRID
    ]
    return float(LAMBDA_GRID[int(np.argmin(scores))])


def known_answer_test() -> dict[str, float]:
    """Check that the speed-keyed fit recovers a known offset at the right wind speeds.

    Synthetic cases: ensemble-mean speed uniform on 0 to 20 m/s; observed power Beta(2, 5) in the
    bins below 8 m/s and Beta(5, 2) above; the "forecast" is the true quantile of each bin minus
    0.03 only at speeds of 8 m/s and above. The fit must return about +0.03 at q90, q95, and q99
    above 8 m/s and about 0 below.
    """
    rng = np.random.default_rng(342)
    size = 200_000
    speed = rng.uniform(0.0, 20.0, size=size)
    high = speed >= 8.0
    actual = np.where(high, rng.beta(5.0, 2.0, size), rng.beta(2.0, 5.0, size))
    low_levels = np.quantile(actual[~high], c303.LEVELS)
    high_levels = np.quantile(actual[high], c303.LEVELS)
    quantiles = np.where(high[:, None], high_levels - 0.03, low_levels)
    cells = cell_ids(
        sites=np.full(size, FARMS[0]), lead_day=np.ones(size, dtype=int), mean_speed=speed
    )
    table = fit_x1_table(quantiles=quantiles, actual=actual, cells=cells)
    bins = {cell % 10: offset for cell, offset in table.items()}
    return {
        "injected_high": 0.03,
        "fit_high_mean": float(np.mean([bins[b] for b in (2, 3, 4, 5)])),
        "fit_low_mean": float(np.mean([bins[b] for b in (0, 1)])),
    }
