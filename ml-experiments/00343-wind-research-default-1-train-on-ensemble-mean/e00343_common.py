"""Paths and constants shared by the scripts of experiment 00343.

The scripts import experiment 00340's modules (which import experiments 00336, 00326, 00303, 00285,
00259, and 00230), so run them with `PYTHONPATH` set to the 00303, 00326, 00336, and 00340 folders,
colon separated. Everything read from other experiments' folders is read only.
"""

from pathlib import Path
from typing import Final

import e00340_common as c340  # isort: skip  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

c303 = c340.c303
c326 = c340.c336.c326
EXPERIMENT_ID: Final[int] = 343
SLUG: Final[str] = "wind-research-default-1-train-on-ensemble-mean"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
MIX_DIR: Final[Path] = WORK_DIR / "mix"
FINAL_DIR: Final[Path] = WORK_DIR / "final"
DEFAULT_FINAL_DIR: Final[Path] = c340.FINAL_DIR
"""Experiment 340's final folder: its `default*.npy` is the default arm (the reproduction gate)."""
FARMS: Final[tuple[str, ...]] = c340.FARMS
SEEDS: Final[tuple[int, ...]] = c340.SEEDS
MEMBERS: Final[int] = c303.MEMBERS
FOLDS: Final[int] = 3
FIT_ARMS: Final[tuple[str, ...]] = ("M", "MS", "Mpl")
"""M: ensemble-mean columns. MS: M plus member spread of the 100 m and 10 m speeds. Mpl: M with
the training target permuted across cases within the calendar month (the placebo)."""
MEAN_COLUMNS: Final[tuple[str, ...]] = (
    "speed_10m",
    "speed_100m",
    "dir_sin_10m",
    "dir_cos_10m",
    "direction_sin",
    "direction_cos",
)
SPREAD_COLUMNS: Final[tuple[str, ...]] = ("speed_100m", "speed_10m")
PLACEBO_SEED: Final[int] = 343


def case_features(*, columns: dict[str, np.ndarray], with_spread: bool) -> np.ndarray:
    """Return one row per case: the member mean of each weather column, then optional spreads.

    Args:
        columns: Member rows' columns, 51 consecutive rows per case, as `member_columns` returns.
        with_spread: Append the member standard deviation of the 100 m and 10 m speeds.

    Returns:
        Float32 array of shape (cases, 6) or (cases, 8).
    """
    parts = [columns[name].reshape(-1, MEMBERS).mean(axis=1) for name in MEAN_COLUMNS]
    if with_spread:
        parts += [columns[name].reshape(-1, MEMBERS).std(axis=1) for name in SPREAD_COLUMNS]
    return np.column_stack(parts).astype(np.float32)


def month_permutation(*, valid: np.ndarray, seed: int) -> np.ndarray:
    """Return, for each case, the case whose target it takes: a random draw within its month."""
    months = valid.astype("datetime64[M]")
    rng = np.random.default_rng(seed)
    out = np.arange(valid.size)
    for month in np.unique(months):
        rows = np.flatnonzero(months == month)
        out[rows] = rng.permutation(rows)
    return out


def known_answer_test() -> dict[str, float]:
    """Check that the case-mean fit recovers a planted power curve from synthetic members.

    Each synthetic case has 51 members with speeds around a case speed; the observed power is a
    known function of the case speed. The XGBoost median must come within 0.03 of the function on
    held-out cases, and the permuted-target placebo must not.
    """
    import xgboost as xgb

    rng = np.random.default_rng(PLACEBO_SEED)
    cases = 6000
    speed = rng.uniform(0, 20, cases)
    members = speed[:, None] + rng.normal(0, 1.0, (cases, MEMBERS))
    mean_speed = members.mean(axis=1)
    truth = np.clip((speed - 3.0) / 9.0, 0.0, 1.0)
    train, test = slice(0, 4500), slice(4500, None)
    result = {}
    for name, target in (("true", truth), ("placebo", truth[rng.permutation(cases)])):
        booster = xgb.train(
            {"objective": "reg:quantileerror", "quantile_alpha": 0.5, "max_depth": 4, "eta": 0.1},
            xgb.DMatrix(mean_speed[train, None], label=target[train]),
            num_boost_round=100,
        )
        predicted = booster.predict(xgb.DMatrix(mean_speed[test, None]))
        result[name] = float(np.abs(predicted - truth[test]).mean())
    return result
