"""Sort the labelled real transfers with no D0M alarm at their start by the reason for the miss.

The classes are tested in order and the first that applies is the class of the transfer. Every
test reads the 24-hour step statistic `z` (the step in noise units) around the labelled start and
D0M's alarms at 4.0 noise units. The labelled start is the only thing taken from the log.
"""

from __future__ import annotations

import zlib
from typing import Final

import numpy as np
import polars as pl
from e00069_common import GRID_LENGTH, TUNING_MONTHS, stamp_months
from e00069_detectors import DetectorInputs
from e00069_endrule import near_refit
from e00069_score import ONSET_TOLERANCE

THRESHOLD: Final[float] = 4.0
WEAK_FLOOR: Final[float] = 2.0
NEAR: Final[int] = 96
"""Two days in half-hours: the search distance for an alarm or a step away from the start."""
SOON: Final[int] = 24
"""Twelve hours in half-hours: splits the alarms outside the tolerance into early and later ones."""
CLASSES: Final[tuple[str, ...]] = (
    "data gap",
    "monthly refit",
    "alarm outside the tolerance",
    "peak suppressed",
    "sharp step under the threshold",
    "spread or late step under the threshold",
    "no step visible",
)
RECOVERABLE: Final[tuple[str, ...]] = CLASSES[4:6]
"""The classes a weaker onset threshold can reach."""


def classify(
    *, row: dict, inputs: DetectorInputs, alarm_table: pl.DataFrame, logged: dict[int, np.ndarray]
) -> dict[str, object]:
    """The class of one unalarmed transfer, with the numbers behind it.

    `row` holds the transfer's `onset` stamp and `series` (comma-separated). `alarm_table` is
    D0M's alarm table at `THRESHOLD`. The returned `near_above_control` and `wide_above_control`
    say whether the largest step exceeds the 95th percentile of the control windows' peaks.
    """
    series = [int(s) for s in row["series"].split(",")]
    onset = row["onset"]
    refit = near_refit()
    near, wide = (
        slice(max(onset - ONSET_TOLERANCE, 0), onset + ONSET_TOLERANCE + 1),
        slice(max(onset - NEAR, 0), onset + NEAR + 1),
    )
    z_near = np.concatenate([np.abs(inputs.shift[s][near] / inputs.scale[s]) for s in series])
    z_wide = np.concatenate([np.abs(inputs.shift[s][wide] / inputs.scale[s]) for s in series])
    peak_near = float(np.nanmax(z_near)) if np.isfinite(z_near).any() else float("nan")
    peak_wide = float(np.nanmax(z_wide)) if np.isfinite(z_wide).any() else float("nan")
    nearby = alarm_table.filter(
        pl.col("series").is_in(series), (pl.col("stamp") - onset).abs() <= NEAR
    )
    distance = (
        int((nearby["stamp"] - onset).abs().min()) if nearby.height else None
    )  # None: no alarm within two days
    if not np.isfinite(z_near).any():
        label = CLASSES[0]
    elif (z_near >= THRESHOLD)[np.tile(refit[near], len(series))].any():
        label = CLASSES[1]
    elif distance is not None:
        label = CLASSES[2]
    elif peak_wide >= THRESHOLD:
        label = CLASSES[3]
    elif peak_near >= WEAK_FLOOR:
        label = CLASSES[4]
    elif peak_wide >= WEAK_FLOOR:
        label = CLASSES[5]
    else:
        label = CLASSES[6]
    control_near, control_wide = control_peaks(
        series=series, inputs=inputs, logged=logged, seed=zlib.crc32(str(row["card_id"]).encode())
    )
    return {
        "card_id": row["card_id"],
        "near_above_control": peak_near > np.nanquantile(control_near, CONTROL_PERCENTILE),
        "wide_above_control": peak_wide > np.nanquantile(control_wide, CONTROL_PERCENTILE),
        "control_median_wide": float(np.median(control_wide)),
        "class": label,
        "peak_near_z": peak_near,
        "peak_wide_z": peak_wide,
        "alarm_distance": distance,
        "alarm_within_12h": distance is not None and distance <= SOON,
    }


CONTROLS: Final[int] = 2000
"""Control windows drawn per transfer."""
CONTROL_PERCENTILE: Final[float] = 0.95
"""A step counts as above noise when it exceeds this share of the control windows' peaks."""


def control_peaks(
    *, series: list[int], inputs: DetectorInputs, logged: dict[int, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Largest |z| of control windows with the transfer's series and no transfer in them.

    Each control draws one stamp per series from odd months, at least 7 days from any logged
    closure on that series, and takes the largest |z| over the series within 4 and within 96
    half-hours of its stamp. Returns the two arrays of `CONTROLS` peaks (near, wide).
    """
    rng = np.random.default_rng(seed)
    odd = np.isin(stamp_months(), list(TUNING_MONTHS))
    near = np.full((CONTROLS, len(series)), np.nan)
    wide = np.full((CONTROLS, len(series)), np.nan)
    for k, s in enumerate(series):
        z = np.abs(inputs.shift[s] / inputs.scale[s])
        ok = np.flatnonzero(odd & ~logged[s])
        ok = ok[(ok > NEAR) & (ok < GRID_LENGTH - NEAR - 1)]
        for i, t in enumerate(rng.choice(ok, size=CONTROLS)):
            for out, half in ((near, ONSET_TOLERANCE), (wide, NEAR)):
                window = z[t - half : t + half + 1]
                out[i, k] = np.nanmax(window) if np.isfinite(window).any() else np.nan
    return np.nanmax(near, axis=1), np.nanmax(wide, axis=1)
