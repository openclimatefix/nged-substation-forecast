# ruff: noqa: E501
"""Constants and loaders shared by the scripts of experiment 264 (honest bands at D8)."""

import datetime as dt
import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 264
SLUG: Final[str] = "dp-gas-engine-asymmetric-bands-at-d8"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
PREDS_DIR: Final[Path] = WORK_DIR / "preds"
E191_PREDS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00191-dp-price-sensitive-assets-in-the-history" / "work" / "preds" / "gated"
)  # fmt: skip

SITE_IDS: Final[dict[str, int]] = {"D8": 26, "D10": 10, "D16": 18}
"""`time_series_id` of each site in experiment 191's stored forecasts."""
ENGINE_SITE: Final[str] = "D8"
PLACEBO_SITES: Final[tuple[str, ...]] = ("D10", "D16")
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
LOWER_TAILS: Final[tuple[str, ...]] = ("q01", "q05", "q10", "q25")
UPPER_TAILS: Final[tuple[str, ...]] = ("q75", "q90", "q95", "q99")

UNIT_POINTS: Final[float] = 19.7
"""One running engine unit (1.97 MW), in points of effective capacity (0.1 MW at D8)."""
DROP_POINTS: Final[float] = -25.0
"""A residual below this (-2.5 MW) counts as a drop."""
SWITCH_ON_SHARE: Final[float] = 0.05
"""The engine becomes visible when this share of the last 30 days' resolved half-hours dropped."""
SWITCH_OFF_SHARE: Final[float] = 0.03
"""A visible engine becomes invisible when the share falls below this (hysteresis)."""
LATENCY_HOURS: Final[int] = 6
"""A reading becomes known this long after its valid time (NGED's real delivery lag is unknown)."""
HALF_WIDTH_FLOOR: Final[float] = 1.0
"""Floor, in points, on q50 - q05 and q95 - q50 when a ratio is formed."""
SWITCH_WINDOW_DAYS: Final[int] = 30
SHARE_WINDOW_DAYS: Final[int] = 14
FIT_WINDOW_DAYS: Final[int] = 365
REFIT_EVERY_ORIGINS: Final[int] = 7
MIN_FIT_ROWS: Final[int] = 1300
"""28 days of half-hours: a fit needs this many resolved valid times, else the arm equals arm 1."""
MIN_CELL_ROWS: Final[int] = 300
NOISE_SCALES: Final[tuple[float, ...]] = (1.0, 1.5, 2.0)
RIDGE: Final[float] = 1.0
EM_ITERATIONS: Final[int] = 30
N_UNITS: Final[int] = 4
"""Number of mixture components: 0, 1, 2, or 3 running units."""
BURN_IN_END: Final[dt.datetime] = dt.datetime(2025, 8, 1, tzinfo=dt.UTC)
SCORED_START: Final[dt.datetime] = dt.datetime(2025, 7, 2, tzinfo=dt.UTC)

ARMS: Final[tuple[str, ...]] = ("arm1", "arm2", "arm2b", "arm3", "arm4")
KEYS: Final[list[str]] = ["run", "valid_time"]


def load_reference(*, site: str) -> pl.DataFrame:
    """Return experiment 191's reference forecast (arm a0) for one site, scored rows only.

    Columns: `run`, `power_fcst_init_time`, `valid_time` (Datetimes, UTC), `lead_hours` (Float64),
    `q01` to `q99`, `mean`, `actual` (Float32), `spread_positive` and `price_p75` (Boolean). The
    rows are sorted by `power_fcst_init_time` then `valid_time`; leads are 0.5 to 168 hours, rows in
    a hand-filtered switching event and rows with no reading are dropped.
    """
    frame = pl.read_parquet(E191_PREDS / f"scored_{site}.parquet")
    return (
        frame.filter(
            (pl.col("arm") == "a0")
            & (pl.col("lead_hours") > 0.0)
            & (pl.col("lead_hours") <= 168.0)
            & ~pl.col("switching_mask")
            & pl.col("actual").is_not_null()
            & (pl.col("time_series_id") == SITE_IDS[site])
        )
        .select(
            "run",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            *QUANTILES,
            "mean",
            "actual",
            "spread_positive",
            "price_p75",
        )
        .sort("power_fcst_init_time", "valid_time")
    )


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
