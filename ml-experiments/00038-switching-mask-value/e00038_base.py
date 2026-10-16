"""Constants, the grid, the topology, the detectors, and the scorers, copied from experiment 00037.

An experiment may not import another experiment, so the pieces of `e00037_grid.py`,
`e00037_common.py`, `e00037_detect.py`, and `e00037_tail.py` that experiment 00038 needs sit here.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = int((GRID_END - GRID_START).total_seconds() // 1800)
"""Number of half-hour stamps from `GRID_START` to `GRID_END`."""
LOG_COVERAGE_END: Final[datetime] = datetime(2026, 3, 26, tzinfo=UTC)
"""The switching logs run to 2026-03-25; later months are reported apart."""


def stamp_index(*, utc_time: datetime) -> int:
    """Return the index of the half-hour containing `utc_time` (floor), from `GRID_START`."""
    return int((utc_time - GRID_START).total_seconds() // 1800)


COVERAGE_END_STAMP: Final[int] = stamp_index(utc_time=LOG_COVERAGE_END)


def stamp_months() -> list[int]:
    """Return the UTC calendar month (1 to 12) of each grid stamp."""
    return [(GRID_START + timedelta(minutes=30 * i)).month for i in range(GRID_LENGTH)]


SOURCE_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look"
"""Experiment 00037's private tables (panel, logs-derived oracle)."""
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00038-switching-mask-value"
"""Private tables; never committed."""
FITS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00017-training-window-and-retraining-rule-drift-demand" / "fits"
)
FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
INTERCONNECTIONS_PATH: Final[Path] = (
    Path("/mnt/data/nged-substation-forecast/data/NGED/substation_hierarchy_2026-10-15")
    / "Primary Substation Interconnections.csv"
)
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
MVA_SERIES: Final[frozenset[int]] = frozenset({3, 9, 11, 14, 15, 24, 25, 26})
"""Non-directional apparent-power series; conservation holds only approximately for them."""
TUNING_MONTHS: Final[frozenset[int]] = frozenset({1, 3, 5, 7, 9, 11})
"""Odd calendar months set thresholds; even months validate."""
H3_RESIDUAL_ARM: Final[str] = "H0.5"
# Static topology between the 16 Primaries, from the interconnection list mapped to series
# (substation 900019 is series 12 and 13, two transformers of one Primary, adjacent to each other).
ADJACENCY: Final[dict[int, tuple[int, ...]]] = {
    3: (15, 16, 26),
    7: (24,),
    8: (10, 14, 24),
    9: (11, 24),
    10: (8, 12, 13, 14, 24, 25),
    11: (9, 25),
    12: (10, 13, 14, 16, 25),
    13: (10, 12, 14, 16, 25),
    14: (8, 10, 12, 13, 15, 16),
    15: (3, 14, 17, 18),
    16: (3, 12, 13, 14, 26),
    17: (15, 18, 26),
    18: (15, 17),
    24: (7, 8, 9, 10, 25),
    25: (10, 11, 12, 13, 24),
    26: (3, 16, 17),
}


DetectorType = Literal["D0", "D1"]
NEIGHBOUR_Z: Final[float] = 2.0
BALANCE_RANGE: Final[tuple[float, float]] = (0.4, 2.5)
MIN_PRESENT: Final[float] = 0.75
REFIT_PERIOD_WINDOW: Final[int] = 48
"""Windows of at least this many stamps skip alarms that straddle a monthly refit."""


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


def prepare(*, residuals: dict[int, np.ndarray], window: int) -> DetectorInputs:
    """Compute the level shift and its tuning-month scale for every series."""
    shift = {sid: level_shift(values=values, window=window) for sid, values in residuals.items()}
    return DetectorInputs(
        window=window, shift=shift, scale={s: robust_scale(shift=d) for s, d in shift.items()}
    )


def _balance(*, inputs: DetectorInputs, series: int, stamp: int) -> tuple[float, dict[int, float]]:
    """Return the summed opposite-sign neighbour steps and each neighbour's step, at `stamp`."""
    own = inputs.shift[series][stamp]
    partners: dict[int, float] = {}
    for neighbour in ADJACENCY.get(series, ()):
        if neighbour not in inputs.shift:
            continue
        segment = inputs.shift[neighbour][max(stamp - 1, 0) : stamp + 2]
        if not np.isfinite(segment).any():
            continue
        step = float(segment[np.nanargmax(-np.sign(own) * segment)])
        if -np.sign(own) * step / inputs.scale[neighbour] > NEIGHBOUR_Z:
            partners[neighbour] = step
    return float(sum(partners.values())), partners


def alarms(*, inputs: DetectorInputs, detector: DetectorType, threshold: float) -> pl.DataFrame:
    """Return one row per alarm: series, stamp, step_mw, z, and for D1 the partner split."""
    rows: list[dict] = []
    refits = refit_stamps() if inputs.window >= REFIT_PERIOD_WINDOW else np.array([], dtype=int)
    for series, shift in inputs.shift.items():
        z = shift / inputs.scale[series]
        for stamp in local_peaks(z=z, threshold=threshold, window=inputs.window):
            if len(refits) and (np.abs(refits - stamp) < inputs.window).any():
                continue
            row = {
                "series": series,
                "stamp": int(stamp),
                "step_mw": float(shift[stamp]),
                "z": float(z[stamp]),
                "detector": detector,
                "window": inputs.window,
                "partners": "",
                "partner_sum_mw": None,
                "balance_ratio": None,
            }
            if detector == "D1":
                total, partners = _balance(inputs=inputs, series=series, stamp=int(stamp))
                ratio = -total / shift[stamp]
                if not partners or not BALANCE_RANGE[0] <= ratio <= BALANCE_RANGE[1]:
                    continue
                row["partners"] = ";".join(f"{k}:{v / total:.2f}" for k, v in partners.items())
                row["partner_sum_mw"], row["balance_ratio"] = total, float(ratio)
            rows.append(row)
    schema = {
        "series": pl.Int64,
        "stamp": pl.Int64,
        "step_mw": pl.Float64,
        "z": pl.Float64,
        "detector": pl.String,
        "window": pl.Int64,
        "partners": pl.String,
        "partner_sum_mw": pl.Float64,
        "balance_ratio": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass of each level: the interval between the mid-points to its neighbours."""
STACK_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00019-primary-demand-stack-vs-manual-heuristic"
    / "predictions"
    / "stack"
    / "predictions.parquet"
)
THRESHOLDS_PATH: Final = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "thresholds.parquet"
)


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Twice the mass-weighted mean pinball loss of the 9 levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def threshold_weighted_score(
    *, quantiles: np.ndarray, actual: np.ndarray, threshold: np.ndarray
) -> np.ndarray:
    """The quantile score after the forecast and the observation are floored at the threshold."""
    return quantile_score(
        quantiles=np.maximum(quantiles, threshold[:, None]), actual=np.maximum(actual, threshold)
    )
