"""The paired low-threshold onset (PP): weak onset alarms accepted only when a step back follows.

PP raises every D0 alarm at a strong threshold `T`. It also raises a weak alarm at a local peak of
the step statistic between `weak_share * T` and `T` when a matching step back follows within the
cap. A matching step back is a later local peak of at least `weak_share * T` noise units, of the
opposite sign, between `MIN_RATIO` and `MAX_RATIO` times the size of the onset step, from 24 hours
to the cap after the onset. The rule uses up to the cap's worth of hindsight, so it serves
training-time masks and the event table, not a live forecast input. Nothing here reads a switching
log except `tune_pp`, which counts the alarms that a logged closure explains.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
from e00069_detect import odd_unexplained
from e00069_detectors import DetectorInputs, alarms
from e00069_endrule import MIN_DURATION
from e00069_intervals import BLOCK

MIN_RATIO: Final[float] = 0.5
"""The step back must be at least this multiple of the onset step's size."""
MAX_RATIO: Final[float] = 2.0
"""The step back must be at most this multiple of the onset step's size."""
CAP_DAYS: Final[int] = 14
"""Longest gap between the weak onset and its step back."""
WEAK_SHARES: Final[tuple[float, ...]] = (0.5, 0.75)
"""The primary weak threshold is half of `T`; the descriptive variant is three quarters."""
TARGET_ALARMS: Final[int] = 80
"""Odd-month unexplained alarms of D0M at 4.0 noise units, which PP is matched to."""
SCAN_START: Final[float] = 4.0
SCAN_STEP: Final[float] = 0.05
SCAN_CEILING: Final[float] = 9.0


def paired_alarms(*, inputs: DetectorInputs, threshold: float, weak_share: float) -> pl.DataFrame:
    """D0's alarms at `threshold` plus the weak alarms that a later opposite step confirms.

    The result has the columns of `alarms`, and `weak`, which is True for an accepted weak alarm.
    """
    strong = alarms(inputs=inputs, detector="D0", threshold=threshold).with_columns(
        weak=pl.lit(False)
    )
    low = weak_share * threshold
    everything = alarms(inputs=inputs, detector="D0", threshold=low)
    if everything.is_empty():
        return strong
    keys = set(strong.select("series", "stamp").iter_rows())
    candidates = everything.filter(
        pl.struct("series", "stamp").map_elements(
            lambda row: (row["series"], row["stamp"]) not in keys, return_dtype=pl.Boolean
        )
    )
    peaks = {
        series: (group["stamp"].to_numpy(), group["z"].to_numpy())
        for (series,), group in everything.group_by("series")
    }
    accepted = [
        row
        for row in candidates.iter_rows(named=True)
        if _has_step_back(row=row, peaks=peaks[row["series"]])
    ]
    if not accepted:
        return strong
    weak = pl.DataFrame(accepted, schema=everything.schema).with_columns(weak=pl.lit(True))
    return pl.concat([strong, weak]).sort("series", "stamp")


def _has_step_back(*, row: dict, peaks: tuple[np.ndarray, np.ndarray]) -> bool:
    """Whether a local peak of the opposite sign and matching size follows within the cap."""
    stamps, z = peaks
    onset, size = row["stamp"], abs(row["z"])
    ahead = (stamps >= onset + MIN_DURATION) & (stamps <= onset + CAP_DAYS * BLOCK)
    opposite = np.sign(z) == -np.sign(row["z"])
    ratio = np.abs(z) / size
    return bool((ahead & opposite & (ratio >= MIN_RATIO) & (ratio <= MAX_RATIO)).any())


def tune_pp(
    *, inputs: DetectorInputs, weak_share: float, logged: dict[int, np.ndarray]
) -> tuple[float, list[tuple[float, int, int]]]:
    """The lowest strong threshold on a grid from 4.0 to 9.0 with at most 80 odd-month alarms.

    The whole grid is scanned, because the count need not fall steadily as the threshold rises (a
    weak alarm is accepted only when a step back follows). The count is of unexplained odd-month
    alarms, accepted weak alarms included. The second return value holds every (threshold, count,
    weak count) triple evaluated, the weak count being the unexplained odd-month weak alarms.
    """
    curve: list[tuple[float, int, int]] = []
    for step in range(round((SCAN_CEILING - SCAN_START) / SCAN_STEP) + 1):
        threshold = round(SCAN_START + step * SCAN_STEP, 2)
        table = paired_alarms(inputs=inputs, threshold=threshold, weak_share=weak_share)
        curve.append(
            (
                threshold,
                odd_unexplained(table=table, logged=logged),
                odd_unexplained(table=table.filter(pl.col("weak")), logged=logged),
            )
        )
    fits = [t for t, count, _ in curve if count <= TARGET_ALARMS]
    if not fits:
        raise RuntimeError(f"no threshold up to {SCAN_CEILING} gives {TARGET_ALARMS} alarms")
    return min(fits), curve
