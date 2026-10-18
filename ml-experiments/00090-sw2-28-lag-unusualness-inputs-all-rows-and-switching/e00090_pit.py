"""Probability integral transform (PIT) of each power value against a probabilistic climatology.

The climatology of one series at one refit is the empirical distribution of its non-switching
power over the refit's 365-day training window, in cells of local month, day type (weekday or
weekend), and half-hour of day, each pooled with the two half-hours either side. The PIT of a value
is the share of its cell's reference values below it, counting ties as half: near 0 or 1 when the
value is unusual for that time of day, month, and day type. A lag's PIT is the PIT of the power at
the lag's own time.
"""

from datetime import datetime
from typing import Final

import e00090_common as common
import numpy as np
import polars as pl
from e00090_regimes import REFIT_CUTOFF_AGE, TRAIN_WINDOW

POOL_HALF_HOURS: Final[int] = 2
"""Each cell pools the half-hours this far either side of its own."""
MIN_REFERENCE: Final[int] = 10
"""A cell with fewer reference values gives a missing PIT."""
PIT_COLUMNS: Final[tuple[str, ...]] = (
    "pit_origin",
    "pit_same_time_latest_day",
    "pit_same_time_latest_week",
)


def cell_keys() -> tuple[np.ndarray, np.ndarray]:
    """Return, for each grid half-hour, its (month and day type) group and its local half-hour.

    Returns:
        `group` in 0..23 (local month 0..11 times two, plus one on a weekend), and `slot`
        in 0..47 (the local half-hour of day of the period-ending stamp).
    """
    local = common.grid_times().dt.convert_time_zone("Europe/London")
    slot = (local.dt.hour() * 2 + local.dt.minute() // 30).to_numpy()
    group = (local.dt.month() - 1).to_numpy() * 2 + (local.dt.weekday() >= 6).to_numpy()
    return group.astype(np.int16), slot.astype(np.int16)


def pit_grid(
    *,
    power: np.ndarray,
    ara: np.ndarray,
    refit: datetime,
    keys: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return the PIT of every grid half-hour's power against the refit's climatology.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        ara: True where the half-hour lies inside a switching event or closure.
        refit: The refit time; the climatology uses valid times from 365 days before it to 3.5
            hours before it.
        keys: `cell_keys()`.

    Returns:
        A float32 array on the grid: the PIT where the power is observed and the cell has at
        least `MIN_REFERENCE` reference values, NaN elsewhere.
    """
    group, slot = keys
    first = max(int(common.grid_index(times=refit - TRAIN_WINDOW)), 0)
    last = int(common.grid_index(times=refit - REFIT_CUTOFF_AGE))
    window = np.zeros(power.size, dtype=bool)
    window[first : last + 1] = True
    usable = window & np.isfinite(power) & ~ara
    out = np.full(power.size, np.nan, dtype=np.float32)
    for g in range(24):
        in_group = group == g
        reference_by_slot = [np.sort(power[usable & in_group & (slot == s)]) for s in range(48)]
        for s in range(48):
            pooled = np.sort(
                np.concatenate(
                    [
                        reference_by_slot[(s + d) % 48]
                        for d in range(-POOL_HALF_HOURS, POOL_HALF_HOURS + 1)
                    ]
                )
            )
            if pooled.size < MIN_REFERENCE:
                continue
            cell = in_group & (slot == s) & np.isfinite(power)
            values = power[cell]
            below = np.searchsorted(pooled, values, side="left")
            at_or_below = np.searchsorted(pooled, values, side="right")
            out[cell] = ((below + at_or_below) / 2.0 / pooled.size).astype(np.float32)
    return out


def lag_pits(*, grid: np.ndarray, index: np.ndarray) -> pl.DataFrame:
    """Return the three lag PITs for lag grid indices of shape (cases, 3)."""
    return pl.DataFrame(
        np.stack([common.take(values=grid, index=index[:, i]) for i in range(3)], axis=1),
        schema=list(PIT_COLUMNS),
        orient="row",
    ).cast(pl.Float32)
