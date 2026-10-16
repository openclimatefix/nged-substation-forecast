"""Single-window and combined multi-window detectors, and the odd-month calibration of each.

Every detector turns the residual panel into a score per series and stamp, and raises an alarm at
each local peak of the score above a threshold. The threshold of each detector is the lowest on a
grid at which its odd-month alarms that no logged closure explains number at most `TARGET_COUNT`.
Nothing here reads a switching log except that count.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, Literal

import numpy as np
import polars as pl
from e00059_common import TUNING_MONTHS, stamp_months
from e00059_detectors import (
    ALARM_SCHEMA,
    REFIT_PERIOD_WINDOW,
    DetectorInputs,
    alarms,
    local_peaks,
    prepare,
    refit_stamps,
)

WINDOWS: Final[tuple[int, ...]] = (8, 24, 48, 96)
"""Window lengths in half-hours: 4, 12, 24, and 48 hours."""
DetectorNameType = Literal["W8", "W24", "W48", "W96", "MAX", "SUM"]
SINGLE_NAMES: Final[dict[int, DetectorNameType]] = {8: "W8", 24: "W24", 48: "W48", 96: "W96"}
CombinedNameType = Literal["MAX", "SUM"]
COMBINED_NAMES: Final[tuple[CombinedNameType, ...]] = ("MAX", "SUM")
DETECTOR_NAMES: Final[tuple[DetectorNameType, ...]] = (*SINGLE_NAMES.values(), *COMBINED_NAMES)
TARGET_COUNT: Final[int] = 80
"""Odd-month unexplained alarms allowed: experiment 43's 24-hour D0 count."""
SUPPRESSION_WINDOW: Final[int] = 48
"""A combined score alarms only where it is the largest within this many half-hours."""
SINGLE_SCAN: Final[tuple[float, float, float, float]] = (12.0, 0.25, 0.05, 1.0)
COMBINED_SCAN: Final[tuple[float, float, float, float]] = (8.0, 0.25, 0.01, 0.5)
"""Scan start, coarse step, fine step, and floor of a threshold search."""

Arrays = dict[int, np.ndarray]


def match_threshold(
    *, count: Callable[[float], int], scan: tuple[float, float, float, float], target: int
) -> float:
    """Return the lowest threshold on the fine grid whose alarm count is at most `target`.

    The scan moves down from its start in coarse steps to the first count above the target, then
    up in fine steps until the count is at most the target, then down while the count stays there.
    """
    start, coarse, fine, floor = scan
    threshold = start
    while threshold > floor and count(round(threshold, 4)) <= target:
        threshold = round(threshold - coarse, 4)
    threshold = max(threshold, floor)
    while count(round(threshold, 4)) > target:
        threshold = round(threshold + fine, 4)
    while threshold - fine >= floor and count(round(threshold - fine, 4)) <= target:
        threshold = round(threshold - fine, 4)
    return round(threshold, 4)


def odd_unexplained(*, table: pl.DataFrame, logged: Arrays) -> int:
    """Count the alarms in odd months that no logged closure within 24 hours explains."""
    if table.is_empty():
        return 0
    odd = np.isin(stamp_months()[table["stamp"].to_numpy()], list(TUNING_MONTHS))
    explained = np.array(
        [bool(logged[s][t]) for s, t in table.select("series", "stamp").iter_rows()]
    )
    return int((odd & ~explained).sum())


def single_inputs(
    *, residuals: Arrays, scales: dict[int, dict[int, float]]
) -> dict[int, DetectorInputs]:
    """Level shifts of every window, in the noise units of the uninjected residual."""
    return {w: prepare(residuals=residuals, window=w, scale=scales.get(w)) for w in WINDOWS}


def calibrate_singles(*, inputs: dict[int, DetectorInputs], logged: Arrays) -> dict[int, float]:
    """The matched threshold `T_w` of each single window, in noise units."""
    return {
        w: match_threshold(
            count=lambda t, w=w: odd_unexplained(
                table=alarms(inputs=inputs[w], threshold=t), logged=logged
            ),
            scan=SINGLE_SCAN,
            target=TARGET_COUNT,
        )
        for w in WINDOWS
    }


def calibrated_steps(
    *, inputs: dict[int, DetectorInputs], thresholds: dict[int, float]
) -> dict[int, np.ndarray]:
    """Per series, an array of shape (windows, stamps): `z_w / T_w`, with NaN set to zero."""
    series = next(iter(inputs.values())).shift
    return {
        s: np.stack(
            [
                np.nan_to_num(inputs[w].shift[s] / inputs[w].scale[s] / thresholds[w])
                for w in WINDOWS
            ]
        )
        for s in series
    }


def combined_scores(*, steps: dict[int, np.ndarray], name: Literal["MAX", "SUM"]) -> Arrays:
    """The combined score per series: the maximum of `|z_w / T_w|`, or the absolute signed sum."""
    if name == "MAX":
        return {s: np.abs(a).max(axis=0) for s, a in steps.items()}
    return {s: np.abs(a.sum(axis=0)) for s, a in steps.items()}


def single_scores(*, inputs: dict[int, DetectorInputs], window: int) -> Arrays:
    """The single-window score `|z_w|` per series, in noise units."""
    return {
        s: np.nan_to_num(np.abs(shift / inputs[window].scale[s]))
        for s, shift in inputs[window].shift.items()
    }


def combined_alarms(
    *, scores: Arrays, steps: dict[int, np.ndarray], threshold: float, name: CombinedNameType
) -> pl.DataFrame:
    """Alarms at the local peaks of a combined score above `threshold`.

    The window of an alarm is the one with the largest `|z_w / T_w|` at the alarm stamp among the
    windows whose step has the sign of the summed step. A window of 24 hours (48 half-hours) or
    longer drops its alarm within that window of a monthly refit, as for single windows. Peaks are
    chosen first, then the window, then the refit drop, and the calibration counts the alarms that
    remain.
    """
    refits = refit_stamps()
    rows: list[dict] = []
    for series, score in scores.items():
        for stamp in local_peaks(z=score, threshold=threshold, window=SUPPRESSION_WINDOW):
            column = steps[series][:, stamp]
            sign = (
                np.sign(column[np.abs(column).argmax()] if name == "MAX" else column.sum()) or 1.0
            )
            index = int(np.argmax(np.where(np.sign(column) == sign, np.abs(column), -1.0)))
            window = WINDOWS[index]
            if window >= REFIT_PERIOD_WINDOW and (np.abs(refits - stamp) < window).any():
                continue
            rows.append(
                {
                    "series": series,
                    "stamp": int(stamp),
                    "step_mw": float("nan"),
                    "z": float(score[stamp]),
                    "window": window,
                    "partners": "",
                }
            )
    return pl.DataFrame(rows, schema=ALARM_SCHEMA)


def calibrate_combined(
    *, steps: dict[int, np.ndarray], logged: Arrays
) -> dict[Literal["MAX", "SUM"], float]:
    """The matched threshold of the maximum and of the sum detector."""
    out: dict[Literal["MAX", "SUM"], float] = {}
    for name in ("MAX", "SUM"):
        scores = combined_scores(steps=steps, name=name)
        out[name] = match_threshold(
            count=lambda t, scores=scores, name=name: odd_unexplained(
                table=combined_alarms(scores=scores, steps=steps, threshold=t, name=name),
                logged=logged,
            ),
            scan=COMBINED_SCAN,
            target=TARGET_COUNT,
        )
    return out
