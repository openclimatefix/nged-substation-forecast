"""Same-half-hour analogue columns for experiment 00074.

An analogue of a forecast case (a series, a forecast origin, and a valid time) is a past day, at the
same UTC half-hour of the day as the valid time, with the same day type as the valid time's day and
the CAMS unit PV shape closest to the shape forecast for the valid time. The candidates are the 14
days from `FIRST_GAP_DAYS` to `FIRST_GAP_DAYS + CANDIDATE_DAYS - 1` before the UTC date of the
origin. The newest candidate is then at least 57 hours before the earliest origin of a day (09:00
UTC, the 00:00 UTC run plus its 9-hour publication delay), which is longer than the 48 hours
that CAMS is assumed to need.

Day types are working day (0), weekend (1), and bank holiday in England (2), taken from the
Europe/London date of the candidate's or the target's period-ending stamp.

The placebo `A3S` draws the candidates at random from the same 14 days (those with an observed power
and a CAMS shape), ignoring day type and shape. The draw is a hash of the case, so the same case
gets the same candidates in every call.
"""

import warnings
from typing import Final

import e00074_common as common
import holidays
import numpy as np
import polars as pl

FIRST_GAP_DAYS: Final[int] = 3
CANDIDATE_DAYS: Final[int] = 14
WORKING_DAY: Final[int] = 0
WEEKEND: Final[int] = 1
BANK_HOLIDAY: Final[int] = 2
COLUMN_NAMES: Final[tuple[str, ...]] = ("an_power_mean", "an_power_sd", "an_shape_gap", "an_count")
"""The four columns; the arm `A1` drops the standard deviation, which one analogue cannot give."""
HASH_MULTIPLIERS: Final[tuple[int, int, int]] = (
    0x9E3779B97F4A7C15,
    0xBF58476D1CE4E5B9,
    0x94D049BB133111EB,
)
"""SplitMix64 constants for the hash that draws the placebo's candidates."""


def day_class_grid() -> np.ndarray:
    """Return the day type of every half-hour of the dense grid.

    Returns:
        An int8 array on the grid: 0 for a working day, 1 for a Saturday or Sunday, 2 for a bank
        holiday in England, by the Europe/London date of the period-ending stamp.
    """
    local = common.grid_times().dt.convert_time_zone("Europe/London")
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=common.HOLIDAY_YEARS)
    holiday_dates = np.array(sorted(calendar), dtype="datetime64[D]")
    dates = local.dt.date().to_numpy().astype("datetime64[D]")
    weekend = local.dt.weekday().to_numpy() >= 6
    out = np.where(weekend, WEEKEND, WORKING_DAY).astype(np.int8)
    out[np.isin(dates, holiday_dates)] = BANK_HOLIDAY
    return out


def cams_grid(*, site: str) -> np.ndarray:
    """Return a site's CAMS unit PV shape on the dense grid, NaN where CAMS is missing."""
    table = pl.read_parquet(common.E46_SOLAR_DIR / "cams_u.parquet").filter(pl.col("site") == site)
    index = np.asarray(common.grid_index(times=table["time"]))
    out = np.full(common.grid_times().len(), np.nan, dtype=np.float32)
    inside = (index >= 0) & (index < out.size)
    out[index[inside]] = table["u"].to_numpy().astype(np.float32)[inside]
    return out


def _hash_unit(*, target: np.ndarray, origin: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Return a deterministic number in [0, 1) for each (target, origin, candidate offset)."""
    with np.errstate(over="ignore"):
        x = (
            target.astype(np.uint64)[:, None] * np.uint64(HASH_MULTIPLIERS[0])
            + origin.astype(np.uint64)[:, None] * np.uint64(HASH_MULTIPLIERS[1])
            + offset.astype(np.uint64)[None, :] * np.uint64(HASH_MULTIPLIERS[2])
        )
        x ^= x >> np.uint64(30)
        x *= np.uint64(HASH_MULTIPLIERS[1])
        x ^= x >> np.uint64(27)
        x *= np.uint64(HASH_MULTIPLIERS[2])
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def analogue_columns(
    *,
    power: np.ndarray,
    cams: np.ndarray,
    day_class: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
    shape: np.ndarray,
    k: int,
    matched: bool,
) -> np.ndarray:
    """Return the analogue columns of a batch of cases.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        cams: The CAMS unit PV shape on the grid, NaN where missing.
        day_class: `day_class_grid()`.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's forecast origin.
        shape: The unit PV shape forecast for each case's valid time (on the CAMS scale).
        k: The number of analogues wanted.
        matched: Pick the `k` candidates of the target's day type with the smallest gap in shape
            (ties go to the most recent day) when true; pick `k` candidates at random, ignoring day
            type and shape, when false.

    Returns:
        A float32 array of shape (cases, 4) in `COLUMN_NAMES` order: the mean power of the chosen
        candidates, their standard deviation (NaN below 2 candidates), the mean absolute gap
        between their CAMS shape and `shape`, and the number chosen (0 to `k`). A case with no
        candidate has NaN in the first three columns and 0 in the last.
    """
    steps = common.STEPS_PER_DAY
    offsets = np.arange(FIRST_GAP_DAYS, FIRST_GAP_DAYS + CANDIDATE_DAYS)
    day_start = (origin // steps) * steps
    candidate = day_start[:, None] - offsets[None, :] * steps + (target % steps)[:, None]
    inside = candidate >= 0
    safe = np.clip(candidate, 0, power.size - 1)
    candidate_power = power[safe].astype(np.float64)
    candidate_shape = cams[safe].astype(np.float64)
    available = inside & np.isfinite(candidate_power) & np.isfinite(candidate_shape)
    gap = np.abs(candidate_shape - shape[:, None])
    if matched:
        same_type = day_class[safe] == day_class[target][:, None]
        score = np.where(available & same_type, gap + offsets[None, :] * 1e-9, np.inf)
    else:
        score = np.where(
            available, _hash_unit(target=target, origin=origin, offset=offsets), np.inf
        )
    order = np.argsort(score, axis=1, kind="stable")[:, :k]
    chosen = np.isfinite(np.take_along_axis(score, order, axis=1))
    chosen_power = np.where(chosen, np.take_along_axis(candidate_power, order, axis=1), np.nan)
    chosen_gap = np.where(chosen, np.take_along_axis(gap, order, axis=1), np.nan)
    count = chosen.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(chosen_power, axis=1)
        spread = np.nanstd(chosen_power, axis=1)
        mean_gap = np.nanmean(chosen_gap, axis=1)
    spread = np.where(count >= 2, spread, np.nan)
    return np.stack([mean, spread, mean_gap, count.astype(np.float64)], axis=1).astype(np.float32)


def arm_columns(*, columns: np.ndarray, k: int) -> np.ndarray:
    """Return the columns an arm with `k` analogues receives: all four, or three when `k` is 1."""
    if k == 1:
        return columns[:, [0, 2, 3]]
    return columns
