"""Recipe R of experiment 155, copied for experiment 173 without the night offset or analog days.

`add_calendar` adds the two columns recipe R reads. `recipe_r_shift` is experiment 115's recipe R
rebuilt on the rows of a frame: minus 0.75 times the mean D+1 error profile of the 14 days that end
two days before the target day. No shift reads anything dated on or after the origin of the run it
corrects.
"""

from dataclasses import dataclass
from typing import Final

import e00173_common as common
import numpy as np
import polars as pl

LONDON: Final[str] = "Europe/London"
N_HALF_HOURS: Final[int] = 48


def add_calendar(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `origin_day` (days since 1970-01-01 of the origin's UTC date) and `local_hh`.

    Args:
        frame: A forecast frame with `valid_time` and `power_fcst_init_time`.

    Returns:
        The frame with `origin_day` and `local_hh` (the local half-hour of day, 0 to 47, from the
        valid time in Europe/London).
    """
    local = pl.col("valid_time").dt.convert_time_zone(LONDON)
    return frame.with_columns(
        origin_day=pl.col("power_fcst_init_time").dt.date().cast(pl.Int32),
        local_hh=(local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30),
    )


@dataclass(frozen=True)
class DayBlocks:
    """One block of centred D+1 errors per run.

    Attributes:
        runs: The run indices, ascending by origin.
        targets: Each run's target day (days since 1970-01-01): the origin's UTC day plus one.
        blocks: Shape (runs, 2, 48): the sum and the count of each local half-hour's centred errors.
        usable: True where the block has at least `MIN_DAY_READINGS` readings.
        row_run: For each row of the frame, the index into `runs`.
    """

    runs: np.ndarray
    targets: np.ndarray
    blocks: np.ndarray
    usable: np.ndarray
    row_run: np.ndarray


def day_blocks(*, frame: pl.DataFrame, forecast: np.ndarray) -> DayBlocks:
    """Return each run's centred D+1 error block (forecast minus actual, by local half-hour).

    A run's block covers its rows at leads of 15.5 to 39 hours (the UTC day after the origin's). A
    clock-change day holds one local half-hour twice, so the block keeps a sum and a count.

    Args:
        frame: The forecast frame, with the columns of `add_calendar`.
        forecast: The forecast to take errors of, at every row (A's mean plus any earlier shift).

    Returns:
        The blocks.
    """
    run_ids = frame["run"].to_numpy()
    unique, row_run = np.unique(run_ids, return_inverse=True)
    lead = frame["lead_hours"].to_numpy()
    actual = frame["actual"].to_numpy().astype(np.float64)
    local_hh = frame["local_hh"].to_numpy()
    origin_day = frame["origin_day"].to_numpy()
    error = forecast.astype(np.float64) - actual
    in_d1 = (lead >= common.D1_FIRST_LEAD) & (lead <= common.D1_LAST_LEAD) & np.isfinite(error)
    blocks = np.zeros((len(unique), 2, N_HALF_HOURS))
    for run in range(len(unique)):
        rows = np.flatnonzero((row_run == run) & in_d1)
        if len(rows) < common.MIN_DAY_READINGS:
            continue
        centred = error[rows] - error[rows].mean()
        np.add.at(blocks[run, 0], local_hh[rows], centred)
        np.add.at(blocks[run, 1], local_hh[rows], 1.0)
    first_row = np.array([np.flatnonzero(row_run == run)[0] for run in range(len(unique))])
    targets = origin_day[first_row].astype(np.int64) + 1
    usable = blocks[:, 1].sum(axis=1) >= common.MIN_DAY_READINGS
    return DayBlocks(
        runs=unique, targets=targets, blocks=blocks, usable=usable, row_run=row_run.astype(np.int64)
    )


def _mean_by_half_hour(*, blocks: np.ndarray) -> np.ndarray:
    """Return the mean over blocks of shape (n, 2, 48), by half-hour; zero where none counts."""
    total = blocks[:, 0].sum(axis=0)
    count = blocks[:, 1].sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def fourteen_day_profiles(*, days: DayBlocks) -> tuple[np.ndarray, np.ndarray]:
    """Return recipe R's 14-day profile at each run, and whether it has enough source days.

    A source day is `T - back` for `back` in 2 to 15, where `T` is the run's target day.
    """
    index_of = {int(day): i for i, day in enumerate(days.targets)}
    out = np.zeros((len(days.targets), N_HALF_HOURS))
    ok = np.zeros(len(days.targets), dtype=bool)
    first = common.R_FIRST_SOURCE_BACK
    for run, target in enumerate(days.targets.tolist()):
        chosen = [
            i
            for back in range(first, common.R_PROFILE_DAYS + first)
            if (i := index_of.get(target - back)) is not None and days.usable[i]
        ]
        if len(chosen) >= common.R_MIN_DAYS:
            out[run] = _mean_by_half_hour(blocks=days.blocks[chosen])
            ok[run] = True
    return out, ok


def shift_at_rows(
    *, days: DayBlocks, frame: pl.DataFrame, by_run: np.ndarray, short_only: bool
) -> np.ndarray:
    """Spread each run's shift by local half-hour to its rows (zero beyond 48 h if `short_only`)."""
    shift = by_run[days.row_run, frame["local_hh"].to_numpy()]
    if short_only:
        shift = np.where(
            frame["lead_hours"].to_numpy() <= common.SHORT_LEAD_LIMIT_HOURS, shift, 0.0
        )
    return shift


def recipe_r_shift(
    *, frame: pl.DataFrame, forecast: np.ndarray, short_only: bool = False
) -> np.ndarray:
    """Return recipe R's shift at every row: minus 0.75 times the 14-day profile.

    A run with fewer than 3 usable source days gets a shift of zero.

    Args:
        frame: The forecast frame, with the columns of `add_calendar`.
        forecast: The forecast whose past errors form the profile.
        short_only: True to zero the shift on rows with a lead above 48 hours.

    Returns:
        The shift per row, in points.
    """
    days = day_blocks(frame=frame, forecast=forecast)
    profile, ok = fourteen_day_profiles(days=days)
    by_run = np.where(ok[:, None], -common.R_KAPPA * profile, 0.0)
    return shift_at_rows(days=days, frame=frame, by_run=by_run, short_only=short_only)
