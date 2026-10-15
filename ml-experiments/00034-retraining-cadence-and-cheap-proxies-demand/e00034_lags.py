"""The weekly-analogue lag summaries of experiment 00030's arm A7, copied for experiment 00034."""

import warnings
from typing import Final

import e00034_common as common
import numpy as np

WEEK: Final[int] = 7 * common.STEPS_PER_DAY
WEEKLY_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
A7_COLUMNS: Final[tuple[str, ...]] = ("weekly_mean", "weekly_median")


def a7_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the mean and median of the six weekly analogues usable at the origin.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 2): `weekly_mean` and `weekly_median`. A weekly analogue is
        power 1 to 6 whole weeks before the valid time, and NaN when that half-hour is later than
        the newest usable observation.

    Raises:
        ValueError: If a case's valid time is not after its origin.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - common.NEWEST_AGE
    analogues = []
    for weeks in WEEKLY_WEEKS:
        index = target - weeks * WEEK
        analogues.append(
            np.where(index <= newest, common.take(values=power, index=index), np.nan).astype(
                np.float32
            )
        )
    weekly = np.stack(analogues, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.stack([np.nanmean(weekly, axis=1), np.nanmedian(weekly, axis=1)], axis=1).astype(
            np.float32
        )
