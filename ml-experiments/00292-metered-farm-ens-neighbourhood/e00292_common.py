# ruff: noqa: E501
"""Shared constants and data preparation for experiment 292 (ensemble neighbourhood of a metered farm).

Each farm's rows come from experiment 224's band-0 tables (ECMWF ensemble mean weather at 12 to 36 h
leads, the plant's unit response, and the observed output over capacity), joined to the clear-sky
irradiance and to the farm's own observed power one day earlier (two days earlier when the lead is
over 24 hours). Farms appear only as labels A to F.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.ens_members import MIXTURE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "00292-metered-farm-ens-neighbourhood"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
PRED_DIR: Final[Path] = DATA_DIR / "predictions"
E224_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
E243_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00243-metered-farm-architecture-comparison"
NEIGHBOUR_DIR: Final[Path] = DATA_DIR / "neighbourhood"
FARM_ROWS: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00218-metered-solar-farm-plant-fit" / "farm_rows_cams.parquet"
)
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
SCORED_START: Final[datetime] = datetime(2025, 9, 1, tzinfo=UTC)
SCORED_END: Final[datetime] = datetime(2026, 9, 1, tzinfo=UTC)
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
"""The last two tuning months select the training epoch."""
CEILING_QUANTILE: Final[float] = 0.995
N_BINS: Final[int] = 4
MIN_BIN_ROWS: Final[int] = 50
MIN_FORECAST_POWER: Final[float] = 0.02
SETTINGS: Final[tuple[str, ...]] = ("own_history",)
NEURAL_ARMS: Final[tuple[str, ...]] = ("n0", "n1", "n2", "placebo")
XGB_ARMS: Final[tuple[str, ...]] = ("x0", "x1")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
RAW_FEATURES: Final[tuple[str, ...]] = (
    "ghi_mean",
    "ghi_std",
    "dni_mean",
    "dhi_mean",
    "temperature_mean",
    "elevation_deg",
    "azimuth_sin",
    "azimuth_cos",
    "lead_hours",
    "csi",
    "lag_power",
    "lag_missing",
)
PLANT_FEATURES: Final[tuple[str, ...]] = ("mu", "relative_spread", "lag_residual")
XGB_FEATURES: Final[tuple[str, ...]] = (*RAW_FEATURES, "unit_mean")
SUMMARY_FEATURES: Final[tuple[str, ...]] = (
    "k1_mean",
    "k1_std",
    "k1_min",
    "k1_max",
    "k2_mean",
    "k2_std",
    "k2_min",
    "k2_max",
    "k1_clear_share",
    "k2_clear_share",
    "upwind_ghi",
)
"""The 11 neighbourhood summaries; irradiance in W/m2, shares in [0, 1]."""
MIN_RING1_CELLS: Final[int] = 5
MIN_DISK_CELLS: Final[int] = 15


@dataclass
class Farm:
    """One farm's band-0 rows with the raw features, and the tuning and scored masks."""

    site: str
    frame: pl.DataFrame
    tuning: np.ndarray
    validation: np.ndarray
    scored: np.ndarray
    own_scale: float
    own_ceiling: float


def load_farm(site: str) -> Farm:
    """Return a farm's rows, features, and the plant scale and ceiling fitted on its tuning months."""
    table = pl.read_parquet(E224_DIR / "members" / f"{site}_b0.parquet")
    summaries = NEIGHBOUR_DIR / f"{site}.parquet"
    if summaries.exists():
        table = table.join(pl.read_parquet(summaries), on="time", how="left")
    sky = pl.read_parquet(FARM_ROWS).filter(pl.col("site") == site).select("time", "clear_sky")
    table = table.join(sky, on="time", how="left").sort("time")
    # Observed power one day earlier (two when the lead is over 24 h), before the forecast was made.
    power = table.select("time", "output")
    lag_frames = [
        power.select(
            time=pl.col("time") + pl.duration(days=days), **{f"lag{days}": pl.col("output")}
        )
        for days in (1, 2)
    ]
    table = table.join(lag_frames[0], on="time", how="left").join(
        lag_frames[1], on="time", how="left"
    )
    table = table.with_columns(
        lag_power=pl.when(pl.col("lead_hours") <= 24).then(pl.col("lag1")).otherwise(pl.col("lag2"))
    ).with_columns(
        lag_missing=pl.col("lag_power").is_null().cast(pl.Float64),
        lag_power=pl.col("lag_power").fill_null(0.0),
        elevation_deg=90.0 - pl.col("zenith_deg"),
        azimuth_sin=(pl.col("sun_azimuth_deg") * np.pi / 180).sin(),
        azimuth_cos=(pl.col("sun_azimuth_deg") * np.pi / 180).cos(),
        csi=(pl.col("ghi_mean") / pl.col("clear_sky").clip(lower_bound=1.0))
        .clip(0.0, 1.5)
        .fill_null(0.0),
    )
    time = table["time"]
    tuning = (time < SCORED_START).to_numpy()
    validation = ((time >= VALIDATION_START) & (time < SCORED_START)).to_numpy()
    scored = ((time >= SCORED_START) & (time < SCORED_END)).to_numpy()
    output = table["output"].to_numpy().astype(np.float64)
    x = table["unit_mean"].to_numpy()
    scale = float(np.sum(output[tuning] * x[tuning]) / np.sum(x[tuning] ** 2))
    ceiling = float(np.quantile(output[tuning], CEILING_QUANTILE))
    return Farm(site, table, tuning, validation, scored, scale, ceiling)


def plant_forecast(*, farm: Farm, scale: float, ceiling: float) -> np.ndarray:
    """Return the capped plant forecast: scale times the unit response, at most the ceiling."""
    return np.minimum(scale * farm.frame["unit_mean"].to_numpy(), ceiling)


def bin_offsets(*, mu: np.ndarray, output: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the bin edges of the forecast and the (bins, 11) residual quantiles per bin."""
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


def offsets_for(*, mu: np.ndarray, edges: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Return each row's 11 offsets from its bin of the forecast."""
    return table[np.digitize(mu, edges)]
