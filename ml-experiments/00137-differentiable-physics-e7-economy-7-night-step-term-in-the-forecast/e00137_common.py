"""Constants and loaders shared by the scripts of experiment 137 (the Economy 7 night-step term).

The base forecasts are experiment 133's stored `pool` arm (GAM-K on the ENS ensemble mean with one
shared noise setting). The ENS-mean temperature comes from experiment 111's stored ensemble arrays.
Both are read from those experiments' private folders; this experiment writes only under `WORK_DIR`.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 137
SLUG: Final[str] = "differentiable-physics-e7-economy-7-night-step-term-in-the-forecast"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
BASE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00133-differentiable-physics-vk-learned-noise-and-observation-variance-in-gam-k"
    / "work"
    / "dp_preds"
)
ENS_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00111-differentiable-physics-6-ens-forecast-engine-against-bg2"
    / "work"
    / "ens_store"
)
FIT_HEADLINE: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00123-differentiable-physics-e7-storage-heating-night-step-estimation"
    / "work"
    / "fit_headline.parquet"
)

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
SCORED_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
FIRST_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
HEATING_THRESHOLD_C: Final[float] = 15.5
COLD_DAY_C: Final[float] = 3.0
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
N_NIGHT_SLOTS: Final[int] = 24
NIGHT_START_MINUTE: Final[int] = 21 * 60 + 30
"""The first night slot is the half-hour that ends at 21:30 UTC; the last ends at 09:00 UTC."""
WINDOW_ENDS_MINUTES: Final[tuple[int, ...]] = (60, 90, 120, 150, 180)
"""The E7 window rows: half-hours ending 01:00, 01:30, 02:00, 02:30, and 03:00 UTC."""
QUANTILES: Final[tuple[str, ...]] = (
    "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99",
)  # fmt: skip
LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
KEEP: Final[tuple[str, ...]] = (
    "time_series_id", "run", "power_fcst_init_time", "valid_time", "lead_hours", "seed",
    *QUANTILES, "actual", "mean",
)  # fmt: skip


def write_json_atomic(*, path: Path, content: object) -> None:
    """Write a JSON file through a temporary name so an interrupted write leaves no file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True, default=float))
    temporary.rename(path)


def load_base(*, site: str) -> pl.DataFrame:
    """Return experiment 133's `pool` rows at a site, over every stored period."""
    pieces = [pl.read_parquet(BASE_DIR / f"pool_{period}_{site}.parquet") for period in PERIODS]
    frame = pl.concat(pieces).filter(
        ~pl.col("switching_mask") & pl.col("actual").is_finite() & (pl.col("lead_hours") <= 168.0)
    )
    return (
        frame.select([*KEEP, "hour_band"])
        .with_columns(site=pl.lit(site))
        .sort(["power_fcst_init_time", "valid_time"])
    )


def ens_mean_temperature(*, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the run origins (`datetime64[us]`) and the ENS-member mean temperature per run.

    Args:
        site: The substation's label.

    Returns:
        The origins, shape (n_runs,), and the temperature in degrees Celsius, shape (n_runs,
        n_targets); target `k` of a run ends `k + 1` half-hours after its origin.
    """
    with np.load(ENS_DIR / f"{site}.npz") as stored:
        origins = stored["origins"]
        temperature = np.nanmean(stored["temperature"].astype(np.float64), axis=1)
    return origins, temperature
