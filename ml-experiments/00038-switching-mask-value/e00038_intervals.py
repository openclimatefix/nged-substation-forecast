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
from e00038_base import ADJACENCY, GRID_LENGTH

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


def ready_after(*, window: int) -> int:
    """Return the half-hours after onset before an alarm is final.

    The peak test looks `window` stamps ahead of a shift that itself needs `window` more, and the
    partner search reads 2 stamps beyond the onset.
    """
    return 2 * window + 2


def known_at(*, intervals: list[Interval], newest: int, window: int) -> list[tuple[int, int, int]]:
    """Return the `(series, start, stop_exclusive)` masks an origin can know of.

    An alarm interval counts once the peak test and the partner search that follow its onset are
    complete (`onset + ready_after(window) <= newest`). A partner interval waits for its own first
    24-hour block (`onset + 48 <= newest`), since that block chooses the partner. If the end is not
    yet known (`known_from > newest`), the mask runs to the newest observation.
    """
    out = []
    for iv in intervals:
        ready = iv.onset + (BLOCK if iv.role == "partner" else ready_after(window=window))
        if ready > newest:
            continue
        stop = iv.end if iv.known_from <= newest else newest + 1
        out.append((iv.series, iv.onset, min(stop, newest + 1)))
    return out


def intervals_frame(*, intervals: list[Interval]) -> pl.DataFrame:
    """Return the intervals as a table."""
    return pl.DataFrame(
        [iv.__dict__ for iv in intervals],
        schema={
            "series": pl.Int64,
            "onset": pl.Int64,
            "end": pl.Int64,
            "known_from": pl.Int64,
            "alarm_series": pl.Int64,
            "detector": pl.String,
            "role": pl.String,
            "relative_step": pl.Float64,
        },
    )


def _clock_block_step(
    *, power: np.ndarray, during: np.ndarray, before: np.ndarray, block: int
) -> float:
    """Mean power over `during` minus mean power over `before`, both in 6-hour clock `block`."""
    in_event = power[during[(during % BLOCK) // 12 == block]]
    reference = power[before[(before % BLOCK) // 12 == block]]
    return float(np.nanmean(in_event) - np.nanmean(reference))


def split_fractions(*, power: dict[int, np.ndarray], interval_set: list[Interval]) -> pl.DataFrame:
    """Return, per partner and 6-hour clock block, the share of the alarm series' MW step it takes.

    For each alarm with at least one partner, the MW step of a series in a clock block is its mean
    power over the event minus its mean power in the same clock block over the 7 days before onset.
    The fraction is the partner's step over minus the alarm series' step.
    """
    rows: list[dict] = []
    by_alarm: dict[tuple[int, int], list[Interval]] = {}
    for iv in interval_set:
        by_alarm.setdefault((iv.alarm_series, iv.onset), []).append(iv)
    for (alarm_series, onset), members in by_alarm.items():
        main = next((m for m in members if m.role == "alarm"), None)
        partners = [m for m in members if m.role == "partner"]
        if main is None or not partners or main.end - onset < 2 * BLOCK:
            continue
        during = np.arange(onset, min(main.end, GRID_LENGTH))
        before = np.arange(max(onset - PRE_STAMPS, 0), onset)
        for block in range(4):
            base = _clock_block_step(
                power=power[alarm_series], during=during, before=before, block=block
            )
            rows.extend(
                {
                    "alarm_series": alarm_series,
                    "onset": onset,
                    "partner": p.series,
                    "block": block,
                    "alarm_step_mw": base,
                    "fraction": -_clock_block_step(
                        power=power[p.series], during=during, before=before, block=block
                    )
                    / base
                    if base
                    else np.nan,
                }
                for p in partners
            )
    return pl.DataFrame(
        rows,
        schema={
            "alarm_series": pl.Int64,
            "onset": pl.Int64,
            "partner": pl.Int64,
            "block": pl.Int64,
            "alarm_step_mw": pl.Float64,
            "fraction": pl.Float64,
        },
    )


def adjacency_check() -> None:
    """Raise if the topology is not symmetric."""
    for a, neighbours in ADJACENCY.items():
        for b in neighbours:
            if a not in ADJACENCY[b]:
                msg = f"{a} lists {b} but not the reverse"
                raise ValueError(msg)
