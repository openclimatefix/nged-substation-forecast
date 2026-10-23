# ruff: noqa: E501
"""Constants and loaders shared by the scripts of experiment 265 (prior and visibility switch)."""

import datetime as dt
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 265
SLUG: Final[str] = "dp-gas-engine-bands-prior-and-visibility-switch"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREDS_DIR: Final[Path] = WORK_DIR / "preds"
E191_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00191-dp-price-sensitive-assets-in-the-history" / "work" / "preds" / "gated"
)  # fmt: skip

SITE_IDS: Final[dict[str, int]] = {"D5": 17, "D8": 26, "D10": 10, "D11": 8, "D14": 16, "D16": 18}
"""`time_series_id` of each Primary in experiment 191's stored forecasts."""
SITES: Final[tuple[str, ...]] = ("D5", "D8", "D10", "D11", "D14", "D16")
ENGINE_SITE: Final[str] = "D8"
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
LOWER_TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q10", "q25")
UPPER_TAILS: Final[tuple[str, ...]] = ("q75", "q90", "q95", "q99")

DROP_POINTS: Final[float] = -25.0
"""A residual below this (-2.5 MW at D8) counts as a drop."""
SWITCH_ON_SHARE: Final[float] = 0.07
SWITCH_OFF_SHARE: Final[float] = 0.04
SENSITIVITY_ON_SHARE: Final[float] = 0.05
SENSITIVITY_OFF_SHARE: Final[float] = 0.03
LATENCY_HOURS: Final[int] = 6
HALF_WIDTH_FLOOR: Final[float] = 1.0
SWITCH_WINDOW_DAYS: Final[int] = 30
FIT_WINDOW_DAYS: Final[int] = 365
REFIT_EVERY_ORIGINS: Final[int] = 7
MIN_FIT_ROWS: Final[int] = 1300
"""28 days of half-hours: the control and the as-of pool need this many resolved rows."""
MIN_CELL_ROWS: Final[int] = 300
SHRINKAGE_ROWS: Final[float] = 2000.0
"""`K` in `w = n / (n + K)`: about six weeks of half-hours in one of the 16 cells' parent block."""
BURN_IN_END: Final[dt.datetime] = dt.datetime(2025, 8, 1, tzinfo=dt.UTC)

ARMS: Final[tuple[str, ...]] = (
    "arm1",
    "control",
    "arm2",
    "arm2c",
    "arm3",
    "arm3_sens",
    "arm2p",
    "arm3p",
)
KEYS: Final[list[str]] = ["run", "valid_time"]


def load_reference(*, site: str) -> pl.DataFrame:
    """Return experiment 191's reference forecast (arm a0) for one site, scored rows only.

    Columns: `run`, `power_fcst_init_time`, `valid_time`, `lead_hours`, `q01` to `q99`, `mean`,
    `actual`. Rows are sorted by `power_fcst_init_time` then `valid_time`; leads are 0.5 to 168
    hours, and rows in a hand-filtered event or without a reading are dropped.
    """
    frame = pl.read_parquet(E191_PREDS / f"scored_{site}.parquet")
    return (
        frame.filter(
            (pl.col("arm") == "a0")
            & (pl.col("lead_hours") > 0.0)
            & (pl.col("lead_hours") <= 168.0)
            & ~pl.col("switching_mask")
            & pl.col("actual").is_not_null()
        )
        .select("time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", *QUANTILES, "mean", "actual")
        .sort("power_fcst_init_time", "valid_time")
    )  # fmt: skip


def write_json_atomic(*, path: Path, content: dict) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def write_parquet_atomic(*, path: Path, frame: pl.DataFrame) -> None:
    """Write a parquet file through a temporary name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)
