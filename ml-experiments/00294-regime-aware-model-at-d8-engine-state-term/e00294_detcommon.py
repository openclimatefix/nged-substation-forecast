"""Paths, constants and loaders shared by the scripts of experiment 196.

G2 is the metered generator that the design brief calls G2; its series identifier stays in this
file and is never printed. D8 is the panel's site labelled D8.
"""

import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "regime-aware-model-at-d8-engine-state-term"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00294-{SLUG}"
ARRAYS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)
G2_SERIES: Final[int] = 33
UNIT_MW: Final[float] = 1.97
"""One engine unit of G2, from the four output levels 0, 1.96, 3.93 and 5.93 MW."""
N_UNITS: Final[int] = 3
CLEAN_END: Final[datetime] = datetime(2023, 6, 13, tzinfo=UTC)
G2_FIRST: Final[datetime] = datetime(2023, 7, 3, tzinfo=UTC)
G2_LAST: Final[datetime] = datetime(2023, 8, 16, tzinfo=UTC)
"""G2's non-zero readings lie in [G2_FIRST, G2_LAST)."""
FIT_START: Final[datetime] = datetime(2023, 9, 1, tzinfo=UTC)
FIT_END: Final[datetime] = datetime(2026, 1, 1, tzinfo=UTC)
FALL_START: Final[datetime] = datetime(2026, 2, 1, tzinfo=UTC)
POST_START: Final[datetime] = datetime(2026, 4, 1, tzinfo=UTC)
DECISION_LABELS: Final[tuple[str, ...]] = ("D3", "D5", "D10", "D11", "D14", "D16")
Q_FLOOR: Final[float] = 0.14
"""Reactive-power floor in MVA (experiment 152's lowest readings)."""


def stamp(when: datetime) -> int:
    """Index of the half-hour starting at `when`."""
    return int((when - GRID_START).total_seconds() // 1800)


def load_panel() -> dict[int, dict]:
    """Return the stored residual panel."""
    return pickle.loads(ARRAYS_PATH.read_bytes())


def key_of(*, panel: dict[int, dict], label: str) -> int:
    """Panel key of a D-label."""
    return next(k for k, v in panel.items() if v["site"] == label)


def times(*, n: int) -> pl.Series:
    """UTC start of every half-hour of the grid."""
    return pl.datetime_range(
        GRID_START, GRID_START + HALF_HOUR * (n - 1), interval="30m", time_zone="UTC", eager=True
    )


def g2_output(*, n: int) -> np.ndarray:
    """G2's recorded output (MW) on the grid; NaN where there is no reading."""
    frame = (
        scan_power().filter(pl.col("time_series_id") == G2_SERIES).select("time", "power").collect()
    )
    out = np.full(n, np.nan)
    idx = ((frame["time"].dt.epoch("s").to_numpy() - int(GRID_START.timestamp())) // 1800).astype(
        int
    )
    keep = (idx >= 0) & (idx < n)
    out[idx[keep]] = frame["power"].to_numpy()[keep]
    return out
