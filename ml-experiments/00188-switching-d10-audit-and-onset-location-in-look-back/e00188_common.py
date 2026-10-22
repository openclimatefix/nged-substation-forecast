"""Paths, the half-hour grid, the residual panel, and the logged and labelled events.

A copy of experiment 43's harness module, with this experiment's private folder.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00188-switching-d10-audit-and-onset-location-in-look-back"
)
"""Private tables; never committed."""
E171_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00171-switching-gated-mask-no-harm-pooled-months"
"""Experiment 171's stored per-row scores, flags and masked spans, read here and never written."""
E156_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00156-switching-faster-onset-alarm-from-opposite-steps"
)
"""Experiment 156's stored alarm tables, read here and never written."""
GAM_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00130-ro-nc-detector-on-the-gam-k-residual" / "gam"
)
"""Experiment 130's stored frozen convex-model panels, read here and never written."""
EXPERIMENT_138_RUN_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00138-weaker-onset-detector-on-the-ro-nc-convex-residual" / "run"
)
"""Experiment 138's stored scores, thresholds, and arm B outcomes, which this experiment re-uses."""
EXPERIMENT_43_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00043-sw2-1-detector-scoreboard"
"""Experiment 43's stored events, alarm tables, and event results, which this experiment re-uses."""
PANEL_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "panel.parquet"
LOGGED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00037-switching-first-look" / "intervals.parquet"
)
LABELLED_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)
LABELLED_META_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "meta.json"
)
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
GRID_LENGTH: Final[int] = int((GRID_END - GRID_START).total_seconds() // 1800)
BLOCK: Final[int] = 48
"""Half-hours in a day."""
SERIES_IDS: Final[tuple[int, ...]] = (3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26)
TUNING_MONTHS: Final[frozenset[int]] = frozenset({1, 3, 5, 7, 9, 11})
"""The detectors' noise scale is estimated on odd months, as in experiment 37."""
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
OPERATING_POINTS: Final[dict[str, tuple[Literal["D0", "D1"], int, float]]] = {
    "alarms_D0_8_6.0": ("D0", 8, 6.0),
    "alarms_D1_8_3.75": ("D1", 8, 3.75),
    "alarms_D0_48_4.0": ("D0", 48, 4.0),
    "alarms_D1_48_2.5": ("D1", 48, 2.5),
}
"""The four thresholds that drew SW2-0b's alarm sample, fixed in experiment 37."""


def stamp_of(*, when: datetime) -> int:
    """Index of the half-hour containing `when`, from the grid start."""
    return int((when - GRID_START).total_seconds() // 1800)


def stamp_months() -> np.ndarray:
    """The UTC calendar month (1 to 12) of every grid stamp."""
    index = pl.datetime_range(
        GRID_START, GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )
    return index.dt.month().to_numpy()


def stamp_year_months() -> np.ndarray:
    """A year-and-month label (such as 202203) for every grid stamp, for resampling whole months."""
    index = pl.datetime_range(
        GRID_START, GRID_END, interval="30m", closed="left", time_zone="UTC", eager=True
    )
    return (index.dt.year() * 100 + index.dt.month()).to_numpy()


def load_panel() -> dict[str, dict[int, np.ndarray]]:
    """Return `power`, `expected`, and `residual` arrays per series, NaN where missing."""
    frame = pl.read_parquet(PANEL_PATH)
    out: dict[str, dict[int, np.ndarray]] = {"power": {}, "expected": {}, "residual": {}}
    for (series,), group in frame.group_by(["series"]):
        for name, column in (
            ("power", "obs_mw"),
            ("expected", "expected_mw"),
            ("residual", "res_mw"),
        ):
            array = np.full(GRID_LENGTH, np.nan)
            array[group["stamp"].to_numpy()] = group[column].to_numpy()
            out[name][int(series)] = array
    return out


def near_log_masks(*, pad_stamps: int) -> dict[int, np.ndarray]:
    """True within `pad_stamps` of a logged closure that touches the series."""
    logged = pl.read_parquet(LOGGED_PATH)
    out = {s: np.zeros(GRID_LENGTH, dtype=bool) for s in SERIES_IDS}
    for row in logged.iter_rows(named=True):
        lo = max(row["first_stamp"] - pad_stamps, 0)
        hi = min(row["last_stamp_exclusive"] + pad_stamps, GRID_LENGTH)
        for side in (row["side_a"], row["side_b"]):
            if side is not None and side.isdigit() and int(side) in out:
                out[int(side)][lo:hi] = True
    return out
