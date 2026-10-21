"""Turn detector alarms into switching intervals by tracking the level of the relative residual.

Nothing here reads a switching log. The residual of each series is divided by the series' expected
power, so a transfer that follows the moved load's daily shape shows as a constant relative step.
An interval's end is only known 48 half-hours after it happens, and `known_at` hides what an
origin could not have known.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from e00156_d_common import GRID_LENGTH

BLOCK: Final[int] = 48
"""Half-hours in a tracking block (24 hours)."""
PRE_STAMPS: Final[int] = 7 * BLOCK
CAP_STAMPS: Final[int] = 60 * BLOCK
MIN_BLOCK_PRESENT: Final[float] = 0.5
EXPECTED_FLOOR_SHARE: Final[float] = 0.1
"""Expected power is floored at this share of the series' median, so q cannot blow up."""


@dataclass(frozen=True)
class Interval:
    """One detected switching interval on one series, in half-hour stamps.

    `end` is the first stamp after the event. `known_from` is the first stamp at which an origin can
    know `end`: the end of the 24-hour block that failed (or `end` when the 60-day cap ended it).
    """

    series: int
    onset: int
    end: int
    known_from: int
    alarm_series: int
    detector: str
    role: str
    relative_step: float


def relative_residual(*, residual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return residual divided by expected power, with expected floored at 10% of its median."""
    floor = EXPECTED_FLOOR_SHARE * np.nanmedian(expected)
    return residual / np.maximum(expected, floor)


def _block_mean(*, q: np.ndarray, start: int) -> float:
    """Mean of `q` over a 24-hour block, NaN if under half the block is present."""
    block = q[start : start + BLOCK]
    present = np.isfinite(block)
    return float(block[present].mean()) if present.mean() >= MIN_BLOCK_PRESENT else float("nan")


def level_track(
    *,
    q: np.ndarray,
    onset: int,
    window: int,
    excluded: np.ndarray | None = None,
    step_from_block: bool = False,
    patience: int = 1,
) -> tuple[float, int, int]:
    """Return the onset step `rho`, the end stamp, and the stamp from which that end is known.

    The pre-level is the median of `q` over the 7 days before `onset`, leaving out `excluded`
    stamps. The event continues in 24-hour blocks while the block mean minus the pre-level keeps
    the sign of `rho` and at least half its size. A block with under half its stamps present is
    skipped, since it says nothing. If the first block fails, the event ends at `onset + window`.
    With `step_from_block`, `rho` is the mean over the first 24-hour block (not the first `window`
    half-hours), which averages out the daily shape of the moved load. The event ends after
    `patience` consecutive failing blocks.
    """
    lo = max(onset - PRE_STAMPS, 0)
    before = q[lo:onset].copy()
    if excluded is not None:
        before[excluded[lo:onset]] = np.nan
    pre = float(np.nanmedian(before)) if np.isfinite(before).any() else 0.0
    after = q[onset : onset + window]
    if not np.isfinite(after).any():
        return float("nan"), onset + window, onset + window
    first_block = q[onset : onset + BLOCK]
    if step_from_block and np.isfinite(first_block).any():
        after = first_block
    rho = float(np.nanmean(after)) - pre
    last_good = onset
    failures = 0
    failed_end = onset + BLOCK
    start = onset
    while start - onset < CAP_STAMPS and start < GRID_LENGTH:
        mean = _block_mean(q=q, start=start)
        if np.isfinite(mean):
            step = mean - pre
            if np.sign(step) == np.sign(rho) and abs(step) >= 0.5 * abs(rho):
                last_good, failures = start + BLOCK, 0
            else:
                failures += 1
                if failures >= patience:
                    failed_end = start + BLOCK
                    break
        start += BLOCK
    if last_good == onset:
        return rho, onset + window, onset + BLOCK
    if start - onset >= CAP_STAMPS:  # ended by the cap, or by data still to come
        return rho, last_good, start
    return rho, last_good, failed_end


def build_intervals(
    *,
    alarm_table: pl.DataFrame,
    relative: dict[int, np.ndarray],
    window: int,
    detector: str,
    fixed: bool = False,
) -> list[Interval]:
    """Build one interval per alarm series, and per qualifying partner for D1 alarms.

    A partner is masked only when its own relative step has the sign opposite to the alarm series'
    step and its own first block keeps at least half that step, so masking the wrong side of a
    transfer is avoided. Alarms are processed in time order, and stamps inside an earlier
    interval of the series are left out of its pre-level.

    With `fixed` (the post hoc variant), `rho` comes from the first 24-hour block, the event ends
    only after two consecutive failing blocks (the daily level is noisy), and an alarm that
    falls inside an earlier interval of its series, or within one block after its end, is skipped
    because it is the return step of that event and not a new event.
    """
    excluded = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in relative}
    out: list[Interval] = []
    for row in alarm_table.sort("stamp").iter_rows(named=True):
        series, onset = row["series"], row["stamp"]
        if fixed and any(iv.series == series and iv.onset < onset <= iv.end + BLOCK for iv in out):
            continue
        rho, end, known = level_track(
            q=relative[series],
            onset=onset,
            window=window,
            excluded=excluded[series],
            step_from_block=fixed,
            patience=2 if fixed else 1,
        )
        if not np.isfinite(rho):
            continue
        out.append(Interval(series, onset, end, known, series, detector, "alarm", rho))
        excluded[series][onset:end] = True
        partners = [int(p.split(":")[0]) for p in (row["partners"] or "").split(";") if p]
        for partner in partners:
            if partner not in relative:
                continue
            p_rho, p_end, p_known = level_track(
                q=relative[partner],
                onset=onset,
                window=window,
                excluded=excluded[partner],
                step_from_block=fixed,
                patience=2 if fixed else 1,
            )
            if not np.isfinite(p_rho) or np.sign(p_rho) == np.sign(rho):
                continue
            if p_end - onset < BLOCK:  # the partner's own first block failed
                continue
            out.append(Interval(partner, onset, p_end, p_known, series, detector, "partner", p_rho))
            excluded[partner][onset:p_end] = True
    return out
