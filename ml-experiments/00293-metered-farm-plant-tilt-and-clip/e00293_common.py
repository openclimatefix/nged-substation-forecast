# ruff: noqa: E501
"""Shared constants, farm loading, and the plant variants for experiment 293.

Each farm's rows come from experiment 224's band-0 tables (ECMWF ensemble 00 UTC weather at 12 to 36
hour leads) joined to the clear-sky irradiance and the farm's own power one or two days earlier,
exactly as in experiments 270 and 287. A plant variant is a unit-response column (fleet shape at a
tilt, or the pooled leave-one-farm-out table), a clip form (hard cap or soft cap), and a cap level.
Farms appear only as labels A to F.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from scipy.optimize import minimize_scalar
from studies.ens_members import MIXTURE_LEVELS
from studies.pv_plant_fit import soft_minimum
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00293-metered-farm-plant-tilt-and-clip"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
SHAPES_DIR: Final[Path] = DATA_DIR / "shapes"
E224_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
E287_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00287-metered-farm-spread-binned-plant-heads"
E282_FRAME: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00282-pv-plant-curve-audit" / "curve_frame.parquet"
)
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
SCORED_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
SCORED_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
N_MEMBERS: Final[int] = 51
MIN_BIN_ROWS: Final[int] = 50
MIN_FORECAST_POWER: Final[float] = 0.02
N_BINS: Final[int] = 4
N_MU_BINS: Final[int] = 6
N_SPREAD_BINS: Final[int] = 3
HARD_CAP_QUANTILE: Final[float] = 0.995
SOFT_CAP_QUANTILE: Final[float] = 0.999
FLEET_AZIMUTH_DEG: Final[float] = 180.0
HEAT_PER_KELVIN: Final[float] = 0.004
CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
SETTINGS: Final[tuple[str, ...]] = ("own_history", "zero_history")


@dataclass(frozen=True)
class Variant:
    """One arm's plant: which unit response, which clip, and which settings it is scored in."""

    name: str
    shape: str
    """`tilt35`, `tilt20`, or `pooled`."""
    soft: bool
    settings: tuple[str, ...]


VARIANTS: Final[tuple[Variant, ...]] = (
    Variant("p2", "tilt35", soft=False, settings=SETTINGS),
    Variant("a_tilt20", "tilt20", soft=False, settings=SETTINGS),
    Variant("b_soft_clip", "tilt35", soft=True, settings=SETTINGS),
    Variant("c_tilt20_soft_clip", "tilt20", soft=True, settings=SETTINGS),
    Variant("d_pooled_shape", "pooled", soft=False, settings=("own_history",)),
)


@dataclass
class Farm:
    """One farm's band-0 rows and the tuning and scored masks."""

    site: str
    frame: pl.DataFrame
    tuning: np.ndarray
    scored: np.ndarray


def load_farm(site: str) -> Farm:
    """Return a farm's rows with the lagged power, clear-sky index, and tuning and scored masks."""
    table = pl.read_parquet(E224_DIR / "members" / f"{site}_b0.parquet")
    sky = pl.read_parquet(FARM_ROWS).filter(pl.col("site") == site).select("time", "clear_sky")
    table = table.join(sky, on="time", how="left").sort("time")
    table = table.with_columns(
        elevation_deg=90.0 - pl.col("zenith_deg"),
        csi=(pl.col("ghi_mean") / pl.col("clear_sky").clip(lower_bound=1.0))
        .clip(0.0, 1.5)
        .fill_null(0.0),
    )
    time = table["time"]
    return Farm(
        site,
        table,
        (time < SCORED_START).to_numpy(),
        ((time >= SCORED_START) & (time < SCORED_END)).to_numpy(),
    )


def clip(*, power: np.ndarray, cap: float, soft: bool) -> np.ndarray:
    """Return the power capped at `cap`, with a hard minimum or a p-norm (exponent 6) minimum."""
    return soft_minimum(power=power, limit=cap) if soft else np.minimum(power, cap)


def fit_scale(*, unit: np.ndarray, output: np.ndarray, cap: float, soft: bool) -> float:
    """Return the scale of the unit response: closed-form for a hard cap, fitted through a soft one."""
    closed = float(np.sum(output * unit) / np.sum(unit**2))
    if not soft:
        return closed

    def error(scale: float) -> float:
        return float(np.sum((output - clip(power=scale * unit, cap=cap, soft=True)) ** 2))

    return float(
        minimize_scalar(error, bounds=(0.3 * closed, 3.0 * closed), method="bounded").x
    )


def bin_offsets(*, mu: np.ndarray, output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the four forecast-bin edges and the (4, 11) residual quantiles per bin."""
    usable = mu > MIN_FORECAST_POWER
    edges = np.quantile(mu[usable], np.linspace(0, 1, N_BINS + 1)[1:-1])
    bins = np.digitize(mu, edges)
    residual = output - mu
    overall = np.quantile(residual, LEVELS)
    table = np.empty((N_BINS, len(LEVELS)))
    for index in range(N_BINS):
        rows = bins == index
        table[index] = (
            np.quantile(residual[rows], LEVELS) if rows.sum() >= MIN_BIN_ROWS else overall
        )
    return edges, table
