"""Paths, the half-hour grid, the residual panel, and the logged and labelled events.

A copy of experiment 43's harness module, with this experiment's private folder.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

BASE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00098-det-bg-solar-and-heating-offsets-in-switching-detector-residual"
)
"""Private tables; never committed."""
ARM: Final[str] = os.environ.get("E98_ARM", "ctrl")
"""The arm whose residual panels the detectors read (`h05_stored`, `ctrl`, `era5_solar`, ...)."""
WORK_DIR: Final[Path] = BASE_DIR / "detector" / ARM
"""This arm's thresholds, scores, and event results."""
FITS_ROOT: Final[Path] = BASE_DIR / "fits"
"""One folder per arm, with `first/` (refit on the 1st) and `second/` (the 16th) per series."""
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
FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
"""Experiment 6's ERA5 feature table."""
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


def arm_fit_path(*, arm: str, which: str, series: int) -> Path:
    """The fit file of `series` under `arm`; an arm without that series falls back to `ctrl`."""
    own = FITS_ROOT / arm / which / f"{series}.parquet"
    return own if own.exists() else FITS_ROOT / "ctrl" / which / f"{series}.parquet"


def load_panel(*, arm: str | None = None) -> dict[str, dict[int, np.ndarray]]:
    """Return `power`, `expected`, and `residual` arrays per series, NaN where missing.

    The arm `h05_stored` reads experiment 37's panel. Any other arm replaces the expected power of
    that panel with its own refits on the 1st, where the arm has a fit, and leaves NaN elsewhere.
    `arm` defaults to the arm named by `E98_ARM`.
    """
    arm = ARM if arm is None else arm
    frame = pl.read_parquet(PANEL_PATH)
    if arm != "h05_stored":
        capacity = (
            pl.scan_parquet(FEATURES_PATH)
            .group_by("time_series_id")
            .agg(pl.col("effective_capacity_mw").median())
            .collect()
        )
        parts = []
        for series in SERIES_IDS:
            path = arm_fit_path(arm=arm, which="first", series=series)
            cap = capacity.filter(pl.col("time_series_id") == series)[
                "effective_capacity_mw"
            ].item()
            fit = pl.read_parquet(path).select("valid_time", "pred", "target")
            joined = frame.filter(pl.col("series") == series).join(
                fit, on="valid_time", how="inner", validate="1:1"
            )
            mismatch = (joined["target"] * cap - joined["obs_mw"]).abs().max()
            if mismatch is None or float(mismatch) > 1e-3:
                raise ValueError(f"series {series}: fit target differs from observed power")
            parts.append(
                joined.with_columns(expected_mw=pl.col("pred") * cap)
                .with_columns(res_mw=pl.col("obs_mw") - pl.col("expected_mw"))
                .drop("pred", "target")
            )
        frame = pl.concat(parts)
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
