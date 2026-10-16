"""Run all six detectors on a residual, build their intervals, and score injected transfers.

The six detectors are the four single windows and the maximum and sum of the calibrated windows.
A transfer is found when an interval on one of its series matches its onset and its end. A transfer
that is not found is given one reason, from the alarms and scores near its onset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import polars as pl
from e00059_common import BLOCK, load_panel
from e00059_detectors import alarms, prepare
from e00059_events import Arrays
from e00059_inject import Event, Injection
from e00059_intervals import Interval, build_intervals, relative_residual
from e00059_score import ONSET_TOLERANCE, by_series, match_event
from e00059_windows import (
    COMBINED_NAMES,
    SINGLE_NAMES,
    WINDOWS,
    DetectorNameType,
    calibrated_steps,
    combined_alarms,
    combined_scores,
    single_inputs,
    single_scores,
)

ReasonType = Literal[
    "found", "end_outside_tolerance", "absorbed", "onset_outside_tolerance", "no_alarm"
]
LARGE_STEP_Z: Final[float] = 4.0
"""A donor step of this many 24-hour noise units counts as large, as in experiment 58."""
NEAR_ONSET: Final[int] = ONSET_TOLERANCE
WIDE_ONSET: Final[int] = BLOCK
"""An alarm this close to the onset but outside the 2-hour tolerance has the wrong onset."""
WINDOW_24_HOURS: Final[int] = 48


@dataclass(frozen=True)
class Thresholds:
    """The matched thresholds: `T_w` per single window, and one for each combined detector."""

    single: dict[int, float]
    combined: dict[Literal["MAX", "SUM"], float]


@dataclass(frozen=True)
class Detection:
    """One detector's alarms, intervals, and the score and threshold its alarms came from."""

    table: pl.DataFrame
    intervals: dict[int, list[Interval]]
    scores: Arrays
    threshold: float


def load() -> tuple[dict, dict[int, dict[int, float]]]:
    """Return the stored panel and its odd-month noise scales for every window."""
    panel = load_panel()
    scales = {w: prepare(residuals=panel["residual"], window=w).scale for w in WINDOWS}
    return panel, scales


def detect_all(
    *,
    residual: Arrays,
    expected: Arrays,
    scales: dict[int, dict[int, float]],
    thresholds: Thresholds,
) -> dict[DetectorNameType, Detection]:
    """Alarms, intervals, and scores of all six detectors on one residual."""
    relative = {s: relative_residual(residual=residual[s], expected=expected[s]) for s in residual}
    inputs = single_inputs(residuals=residual, scales=scales)
    steps = calibrated_steps(inputs=inputs, thresholds=thresholds.single)
    out: dict[DetectorNameType, Detection] = {}
    for window, name in SINGLE_NAMES.items():
        table = alarms(inputs=inputs[window], threshold=thresholds.single[window])
        out[name] = Detection(
            table=table,
            intervals=by_series(intervals=build_intervals(alarm_table=table, relative=relative)),
            scores=single_scores(inputs=inputs, window=window),
            threshold=thresholds.single[window],
        )
    for name in COMBINED_NAMES:
        scores = combined_scores(steps=steps, name=name)
        table = combined_alarms(scores=scores, steps=steps, threshold=thresholds.combined[name])
        out[name] = Detection(
            table=table,
            intervals=by_series(intervals=build_intervals(alarm_table=table, relative=relative)),
            scores=scores,
            threshold=thresholds.combined[name],
        )
    return out


def miss_reason(
    *, event: Event, detection: Detection, found: bool, onset_found: bool
) -> tuple[ReasonType, float]:
    """Return why a transfer was missed, and its best score near the onset as a share of threshold.

    The reasons are tried in this order: the end is outside tolerance (an alarm has a matching
    onset); the alarm was absorbed (the score exceeds the threshold within 2 hours of the onset on
    a series of the transfer but no alarm stands there, because a larger peak within the
    suppression window took it or a monthly refit dropped it); the onset is outside tolerance (an
    alarm on a series of the transfer lies within 24 hours of the onset); else no alarm.
    """
    near = slice(max(event.onset - NEAR_ONSET, 0), event.onset + NEAR_ONSET + 1)
    share = max(float(detection.scores[s][near].max()) for s in event.series) / detection.threshold
    if found:
        return "found", share
    if onset_found:
        return "end_outside_tolerance", share
    if share > 1.0:
        return "absorbed", share
    table = detection.table.filter(pl.col("series").is_in(list(event.series)))
    if (table["stamp"] - event.onset).abs().le(WIDE_ONSET).any():
        return "onset_outside_tolerance", share
    return "no_alarm", share


def donor_step(*, injection: Injection, event: Event, scale_24h: dict[int, float]) -> dict:
    """The MW removed from the donor in the first 24 hours, in MW and in 24-hour noise units."""
    span = slice(event.onset, min(event.end, event.onset + BLOCK))
    step_mw = float(-np.mean(injection.delta[event.donor][span]))
    largest = max(
        abs(float(np.mean(injection.delta[s][span]))) / scale_24h[s] for s in event.series
    )
    return {
        "donor_step_mw": step_mw,
        "donor_step_z": step_mw / scale_24h[event.donor],
        "largest_step_z": largest,
    }


def rows_for_events(
    *,
    batch: list[Event],
    injection: Injection,
    detections: dict[DetectorNameType, Detection],
    scale_24h: dict[int, float],
    shape: str,
) -> list[dict]:
    """One row per transfer and detector: found, reason, and the transfer's step."""
    rows: list[dict] = []
    for event in batch:
        step = donor_step(injection=injection, event=event, scale_24h=scale_24h)
        for name, detection in detections.items():
            match = match_event(
                series=event.series, onset=event.onset, end=event.end, intervals=detection.intervals
            )
            reason, share = miss_reason(
                event=event, detection=detection, found=match.found, onset_found=match.onset_found
            )
            rows.append(
                {
                    "event_id": event.event_id,
                    "shape": shape,
                    "detector": name,
                    "onset_found": match.onset_found,
                    "found": match.found,
                    "end_error": match.end_error,
                    "reason": reason,
                    "score_share": share,
                    **step,
                }
            )
    return rows
