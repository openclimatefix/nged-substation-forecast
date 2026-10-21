"""Paths, the half-hour grid, the residual panel, and the logged and labelled events.

A copy of experiment 43's harness module, with this experiment's private folder.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00132-ro-nc-rescored-with-p12-and-p13-residuals-in-2025"
)
"""Private tables; never committed."""
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


FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
RESTORE_PAD: Final[timedelta] = timedelta(days=2)
"""Half-hours either side of a labelled transfer in which flagged zeros are restored."""
VARIANTS: Final[tuple[str, ...]] = ("half_life", "equal", "masked")
"""Blocked-fit variants: experiment 17's half-life weights, equal weights, and half-life weights
with the six transfer windows left out of the training rows. The first is the headline."""
VARIANT: Final[str] = os.environ.get("E132_VARIANT", "half_life")
"""The blocked-fit variant the scoring scripts read, set by `E132_VARIANT`."""
NEW_PANEL: Final[bool] = os.environ.get("E132_PANEL", "new") != "old"
"""False (`E132_PANEL=old`) scores experiment 101's panel, without the P12 and P13 additions."""
BLOCKED_SERIES: Final[tuple[int, ...]] = (12, 13)


def fits_folder(*, name: str, variant: str) -> str:
    """Folder of one variant's blocked fits (`name` is `main` or `offset`)."""
    return f"fits_{name}" if variant == "half_life" else f"fits_{variant}_{name}"


"""The two series whose residual before 2026 comes from this experiment's blocked fits."""


def blocked_arrays(*, name: str) -> dict[int, dict[str, np.ndarray]]:
    """`power`, `expected`, and `residual` of this experiment's blocked fits, NaN elsewhere.

    `name` is `main` (calendar-month blocks) or `offset` (blocks from the 16th). Scaling by the
    series' median effective capacity is the one of experiment 37's panel.
    """
    capacity = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id").is_in(BLOCKED_SERIES))
        .group_by("time_series_id")
        .agg(pl.col("effective_capacity_mw").median())
        .collect()
    )
    out: dict[int, dict[str, np.ndarray]] = {}
    for series in BLOCKED_SERIES:
        cap = capacity.filter(pl.col("time_series_id") == series)["effective_capacity_mw"].item()
        frame = pl.read_parquet(
            WORK_DIR / fits_folder(name=name, variant=VARIANT) / f"{series}.parquet"
        ).with_columns(
            stamp=(pl.col("valid_time").dt.epoch("s") - int(GRID_START.timestamp())) // 1800
        )
        stamps = frame["stamp"].to_numpy()
        arrays = {k: np.full(GRID_LENGTH, np.nan) for k in ("power", "expected", "residual")}
        arrays["power"][stamps] = frame["target"].to_numpy().astype(float) * cap
        arrays["expected"][stamps] = frame["H0_5"].to_numpy() * cap
        restored = read_restored().filter(pl.col("time_series_id") == series)
        arrays["power"][restored["stamp"].to_numpy()] = restored["power"].to_numpy()
        arrays["residual"][:] = arrays["power"] - arrays["expected"]
        out[series] = arrays
    return out


def read_restored() -> pl.DataFrame:
    """The flagged half-hours re-read from the raw power table by `e00132_restore.py`."""
    return pl.read_parquet(WORK_DIR / "restored.parquet")


def fill_missing(*, stored: np.ndarray, added: np.ndarray) -> np.ndarray:
    """`stored` with its NaN stamps filled from `added`; the two must not overlap."""
    if (np.isfinite(stored) & np.isfinite(added)).any():
        raise ValueError("blocked fits overlap the stored panel")
    return np.where(np.isfinite(stored), stored, added)


def load_panel(*, blocked: bool = NEW_PANEL) -> dict[str, dict[int, np.ndarray]]:
    """Return `power`, `expected`, and `residual` arrays per series, NaN where missing.

    With `blocked`, P12 and P13 also hold this experiment's blocked-fit values before their stored
    panel begins; with `blocked=False` the panel is experiment 101's.
    """
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
    if blocked:
        for series, added in blocked_arrays(name="main").items():
            for name, arrays in out.items():
                arrays[series] = fill_missing(stored=arrays[series], added=added[name])
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
