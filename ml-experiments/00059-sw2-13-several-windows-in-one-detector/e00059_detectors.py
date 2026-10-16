"""Detector D0 of load-transfer onsets in the residual: a step statistic on one series.

Nothing in this module reads a switching log. It sees the residual panel only.
"""

from dataclasses import dataclass
from functools import cache
from typing import Final

import numpy as np
import polars as pl
from e00059_common import GRID_LENGTH, GRID_START, TUNING_MONTHS, stamp_months

MIN_PRESENT: Final[float] = 0.75
REFIT_PERIOD_WINDOW: Final[int] = 48
"""Windows of at least this many stamps skip alarms that straddle a monthly refit."""


ALARM_SCHEMA: Final[dict[str, type[pl.DataType]]] = {
    "series": pl.Int64,
    "stamp": pl.Int64,
    "step_mw": pl.Float64,
    "z": pl.Float64,
    "window": pl.Int64,
    "partners": pl.String,
}
"""Columns of an alarm table; `partners` stays empty because D0 looks at one series."""


def level_shift(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return, at each stamp t, mean(values[t:t+window]) minus mean(values[t-window:t]).

    The result is NaN where fewer than 75% of either window is present. Stamp t is the first
    stamp of the "after" window, so a step that starts at stamp s peaks at t = s.
    """
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values))])
    count = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    t = np.arange(GRID_LENGTH)
    lo, hi = np.clip(t - window, 0, GRID_LENGTH), np.clip(t + window, 0, GRID_LENGTH)
    n_before, n_after = count[t] - count[lo], count[hi] - count[t]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = (total[hi] - total[t]) / n_after - (total[t] - total[lo]) / n_before
    shift[(n_before < MIN_PRESENT * window) | (n_after < MIN_PRESENT * window)] = np.nan
    return shift


def robust_scale(*, shift: np.ndarray) -> float:
    """Return 1.4826 times the MAD of `shift`, using stamps in tuning (odd) months only."""
    months = np.array(stamp_months())
    selected = shift[np.isin(months, list(TUNING_MONTHS)) & np.isfinite(shift)]
    return float(1.4826 * np.median(np.abs(selected - np.median(selected))))


def local_peaks(*, z: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """Return stamps where |z| exceeds `threshold` and is the largest within +/- `window`."""
    magnitude = np.nan_to_num(np.abs(z))
    peaks = []
    for t in np.flatnonzero(magnitude > threshold):
        low, high = max(t - window, 0), min(t + window + 1, GRID_LENGTH)
        if magnitude[t] == magnitude[low:high].max():
            peaks.append(t)
    return np.array(peaks, dtype=int)


@cache
def refit_stamps() -> np.ndarray:
    """Return the stamps at 00:00 UTC on the 1st of each month, when the residual model refits."""
    stamps = [
        i
        for i in range(GRID_LENGTH)
        if (t := GRID_START.timestamp() + 1800 * i) is not None
        and np.datetime64(int(t), "s").astype("datetime64[D]").astype(object).day == 1
        and (int(t) % 86400) == 0
    ]
    return np.array(stamps, dtype=int)


@dataclass(frozen=True)
class DetectorInputs:
    """Level shifts, scales, and topology for one window length."""

    window: int
    shift: dict[int, np.ndarray]
    scale: dict[int, float]


def prepare(
    *, residuals: dict[int, np.ndarray], window: int, scale: dict[int, float] | None = None
) -> DetectorInputs:
    """Compute the level shift for every series, and its tuning-month scale unless one is given.

    A given `scale` keeps the noise scale of the uninjected residual when a batch of transfers is
    added, so that the injected events do not change the threshold they are scored against.
    """
    shift = {sid: level_shift(values=values, window=window) for sid, values in residuals.items()}
    return DetectorInputs(
        window=window,
        shift=shift,
        scale=scale or {s: robust_scale(shift=d) for s, d in shift.items()},
    )


def alarms(*, inputs: DetectorInputs, threshold: float) -> pl.DataFrame:
    """Return one row per D0 alarm: series, stamp, step_mw, z, and the window that alarmed."""
    rows: list[dict] = []
    refits = refit_stamps() if inputs.window >= REFIT_PERIOD_WINDOW else np.array([], dtype=int)
    for series, shift in inputs.shift.items():
        z = shift / inputs.scale[series]
        for stamp in local_peaks(z=z, threshold=threshold, window=inputs.window):
            if len(refits) and (np.abs(refits - stamp) < inputs.window).any():
                continue
            rows.append(
                {
                    "series": series,
                    "stamp": int(stamp),
                    "step_mw": float(shift[stamp]),
                    "z": float(z[stamp]),
                    "window": inputs.window,
                    "partners": "",
                }
            )
    return pl.DataFrame(rows, schema=ALARM_SCHEMA)
