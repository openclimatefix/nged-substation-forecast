# ruff: noqa: E501
"""Shared constants and the row weights of experiment 331 (recency weights on the GAM)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import TrainingWindow, training_rows
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 331
SLUG: Final[str] = "primary-demand-recency-weights-on-gam-and-kalman"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
STREAM_DIR: Final[Path] = WORK_DIR / "streams"
SCORE_DIR: Final[Path] = WORK_DIR / "scores"
EXP328_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00328-lsc-regime-changes-detect-then-model-part-1"
)
FIRST_CUT: Final[datetime] = datetime(2020, 9, 1, tzinfo=UTC)
LAST_CUT: Final[datetime] = datetime(2026, 6, 1, tzinfo=UTC)
SCORE_FROM: Final[datetime] = datetime(2021, 9, 1, tzinfo=UTC)
RIDGES: Final[tuple[float, ...]] = (0.03, 0.1, 0.3)
GAINS: Final[tuple[float, ...]] = (0.05, 0.10, 0.20)
HALF_LIVES: Final[tuple[int, ...]] = (3, 6, 12, 24)
PLACEBO_HALF_LIFE_MONTHS: Final[int] = 12_000
SEASON_WINDOW_DAYS: Final[int] = 30
SEASON_BONUS: Final[float] = 1.0
MIN_AGE_FOR_SEASON_DAYS: Final[int] = 335
MONTHS_AFTER: Final[int] = 10
BAND_DAYS: Final[int] = 365
DAYS_PER_MONTH: Final[float] = 30.4375
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
WINDOW_MONTHS: Final[int] = 24
REAL_ARMS: Final[tuple[str, ...]] = (
    "R0", "W24", *(f"H{h}" for h in HALF_LIVES), "P", *(f"SH{h}" for h in HALF_LIVES),
)  # fmt: skip
INJECT_ARMS: Final[tuple[str, ...]] = ("R0", "W24", *(f"H{h}" for h in HALF_LIVES), "P")
WEIGHTED_ARMS: Final[tuple[str, ...]] = (
    *(f"H{h}" for h in HALF_LIVES), "P", *(f"SH{h}" for h in HALF_LIVES),
)  # fmt: skip


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after the first-of-month `moment`."""
    index = moment.year * 12 + moment.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def monthly_cuts() -> list[datetime]:
    """Return every first-of-month forecast origin from `FIRST_CUT` to `LAST_CUT`."""
    cuts, cut = [], FIRST_CUT
    while cut <= LAST_CUT:
        cuts.append(cut)
        cut = add_months(moment=cut, months=1)
    return cuts


def row_weights(*, end_times: np.ndarray, cut: datetime, arms: tuple[str, ...]) -> np.ndarray:
    """Return the weights of each weighted arm for training rows ending at `end_times`.

    The weight of a row depends only on its end time and the cut, so it uses no information from
    after the origin. Each row of the result is normalised to mean 1, so the ridge penalty keeps
    the scale it has with equal weights.

    Args:
        end_times: The training rows' end times (naive UTC), all no later than the cut.
        cut: The forecast origin.
        arms: Weighted arm names: `H<months>`, `P`, or `SH<months>`.

    Returns:
        Array of shape (len(arms), len(end_times)).
    """
    cut64 = naive_utc(moment=cut)
    age_days = (cut64 - end_times) / np.timedelta64(1, "D")
    assert (age_days >= 0).all(), "a training row ends after the cut"
    mid = np.datetime64(naive_utc(moment=cut), "D") + np.timedelta64(15, "D")
    mid_doy = (mid - mid.astype("datetime64[Y]")).astype(int)
    row_doy = ((end_times - np.timedelta64(1, "s")).astype("datetime64[D]")
               - (end_times - np.timedelta64(1, "s")).astype("datetime64[Y]").astype("datetime64[D]")
               ).astype(int)  # fmt: skip
    gap = np.abs(row_doy - mid_doy)
    gap = np.minimum(gap, 365 - gap)
    seasonal = (gap <= SEASON_WINDOW_DAYS) & (age_days >= MIN_AGE_FOR_SEASON_DAYS)
    out = np.empty((len(arms), len(end_times)))
    for k, arm in enumerate(arms):
        if arm == "P":
            months = PLACEBO_HALF_LIFE_MONTHS
        else:
            months = int(arm.removeprefix("SH").removeprefix("H"))
        w = 0.5 ** (age_days / (months * DAYS_PER_MONTH))
        if arm.startswith("SH"):
            w = w * (1.0 + SEASON_BONUS * seasonal)
        out[k] = w / w.mean()
    return out


def training_end_times(
    *, cache: FeatureCache, site: str, cut: datetime
) -> tuple[np.ndarray, np.ndarray]:
    """Return the full-history training rows of a fit and their end times."""
    index = training_rows(
        features=cache.sites[site],
        cache=cache,
        cut=cut,
        window=TrainingWindow(),
        keep_lockdown=False,
    )
    return index, cache.end_times[index]
