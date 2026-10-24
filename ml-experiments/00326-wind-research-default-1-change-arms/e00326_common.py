"""Paths, constants, and small functions shared by the scripts of experiment 00326.

The scripts import experiment 00303's modules (which copy experiments 00285, 00259, and 00230), so
run them with `PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting`. Everything read
from other experiments' folders is read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00303_common as c303  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 326
SLUG: Final[str] = "wind-research-default-1-change-arms"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
MIX_DIR: Final[Path] = WORK_DIR / "mix"
FINAL_DIR: Final[Path] = WORK_DIR / "final"
STORED_285_MIX_DIR: Final[Path] = c303.ENS_MIX_285_DIR
STORED_303_MIX_DIR: Final[Path] = c303.MIX_DIR
STORED_303_FINAL_DIR: Final[Path] = c303.WORK_DIR / "final"
IMPLIED_CASES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00307-wind-farm-power-as-wind-sensor"
    / "work"
    / "cases_hindcast.parquet"
)
"""Experiment 307's hindcast cases with the implied 100 m wind and the ensemble mean 100 m wind."""

SEEDS: Final[tuple[int, ...]] = c303.SEEDS
FARMS: Final[tuple[str, ...]] = ("W1", "W2", "W3")
MEMBERS: Final[int] = c303.MEMBERS
REPLICATE_START: Final[datetime] = datetime(2025, 1, 1, tzinfo=UTC)
"""Replicate cases start here; the replicate's table and factor use valid times before this."""
KNOWN_ANSWER_SCALE: Final[float] = 0.85
SHUFFLE_SEED: Final[int] = 326

STORED_ARMS: Final[dict[str, str]] = {"stored_default": "xgb_B0", "c": "P8"}
"""Arms read from storage: name -> the file prefix (experiment 285 xgb_B0, experiment 303 P8)."""
FIT_ARMS: Final[tuple[str, ...]] = ("default", "a", "b", "d", "d_rep", "a_pl", "b_pl", "c_pl")
"""Arms fitted here. `d_rep` is the hindcast set only, with the replicate's factor."""
EXTRA_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "a": ("temperature_c", "density"),
    "b": ("hour", "doy"),
}


def air_density(*, temperature_c: np.ndarray) -> np.ndarray:
    """Return dry-air density in kg/m3 at the fixed surface pressure, as experiment 256 did."""
    return (
        c303.FIXED_SURFACE_PRESSURE_PA / (c303.DRY_AIR_GAS_CONSTANT * (temperature_c + 273.15))
    ).astype(np.float32)


def month_shuffle_index(*, valid: np.ndarray, seed: int) -> np.ndarray:
    """Return, for each row, the row whose column values it takes in the month-shuffled placebo.

    Each calendar month is paired with a different calendar month (a random cyclic order of the
    months present, where a month is a year and a month), and each row takes the row at the same
    rank by valid time within the paired month, modulo the paired month's row count. A single month
    is left unshuffled.

    Args:
        valid: The rows' valid times, any order.
        seed: Seeds the random order of the months.

    Returns:
        Integer indices, one per row.
    """
    months = valid.astype("datetime64[M]")
    labels = np.unique(months)
    by_time = np.argsort(valid, kind="stable")
    rows_of = {m: by_time[months[by_time] == m] for m in labels}
    order = np.random.default_rng(seed).permutation(labels.size)
    source = np.arange(valid.size)
    for position, i in enumerate(order):
        own = rows_of[labels[i]]
        paired = rows_of[labels[order[(position + 1) % labels.size]]]
        source[own] = paired[np.arange(own.size) % paired.size]
    return source


def speed_factor(*, valid_before: datetime | None, scale_members: float = 1.0) -> float:
    """Return the fitted wind-speed factor: mean implied 100 m wind over mean ensemble-mean wind.

    Uses experiment 00307's unmasked hindcast cases with an invertible implied wind, pooled over
    the three farms, as experiment 00307 fitted its factor of 1.13.

    Args:
        valid_before: Use only cases with a valid time before this (the replicate period's fit).
        scale_members: Multiplies the ensemble-mean wind first (the known-answer test).

    Returns:
        The factor, as a ratio of mean wind speeds.
    """
    cases = pl.read_parquet(IMPLIED_CASES_PATH).filter(
        pl.col("implied_100m").is_not_null() & ~pl.col("masked")
    )
    if valid_before is not None:
        cases = cases.filter(pl.col("valid_time") < valid_before)
    return float(
        cases["implied_100m"].to_numpy().mean()
        / (cases["ens_speed_100m"].to_numpy().mean() * scale_members)
    )


def known_answer_test() -> dict[str, float]:
    """Return the factor recovered after the members are scaled by `KNOWN_ANSWER_SCALE`.

    Scaling every member's speed by 0.85 must make the fitted factor 1/0.85 times the unscaled
    factor, so the product of the refit factor and 0.85 must equal the original factor.
    """
    original = speed_factor(valid_before=None)
    refit = speed_factor(valid_before=None, scale_members=KNOWN_ANSWER_SCALE)
    return {
        "original": original,
        "refit_on_scaled_members": refit,
        "recovered_difference": abs(refit * KNOWN_ANSWER_SCALE - original),
    }
