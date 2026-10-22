# ruff: noqa: E501
"""Paths, constants and loaders shared by the scripts of experiment 194.

The generator that the design brief calls G2 is series 33 in the power table. The identifier stays in
this file and is never printed. D8 is key 26 of the stored residual panel.
"""

import pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "ps1-ps5-generator-visibility-d8-analogue-break-and-price-following-census"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00194-{SLUG}"
ARRAYS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "briefs" / "sw_false_alarms" / "work" / "arrays.pkl"
)
PRICE_DIRS: Final[tuple[Path, ...]] = (
    ML_EXPERIMENTS_DATA_DIR / "00186-switching-d8-gas-peaker-spark-spread-and-triads" / "public",
    ML_EXPERIMENTS_DATA_DIR / "00185-dp-price-dfs-national-signals-past-demand" / "public",
    ML_EXPERIMENTS_DATA_DIR / "00177-dp-worst-days-other-data-sources" / "public",
)
GRID_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
D8_KEY: Final[int] = 26
G2_SERIES: Final[int] = 33
CONTROL_LABELS: Final[tuple[str, ...]] = ("D3", "D5", "D14")
LEFT_TAIL_MW: Final[float] = -2.5
"""A half-hour is in the left tail when the metered minus expected power is below this."""
G2_UNIT_MW: Final[float] = 1.97
"""Output of one engine unit, from the four levels 0, 1.96, 3.93 and 5.93 MW."""
G2_FIRST: Final[datetime] = datetime(2023, 7, 3, tzinfo=UTC)
G2_LAST: Final[datetime] = datetime(2023, 8, 16, tzinfo=UTC)
"""The weeks in which G2's readings arrive with non-zero values (end exclusive)."""
HALF_HOUR: Final[timedelta] = timedelta(minutes=30)


def load_panel() -> dict[int, dict]:
    """Return the stored residual panel: key to a dict of per-half-hour arrays."""
    return pickle.loads(ARRAYS_PATH.read_bytes())


def panel_stamps(*, n: int) -> pl.Series:
    """Return the UTC start of every half-hour of the panel."""
    return pl.datetime_range(
        GRID_START, GRID_START + HALF_HOUR * (n - 1), interval="30m", time_zone="UTC", eager=True
    )


def key_of(*, panel: dict[int, dict], label: str) -> int:
    """Return the panel key of a D-label."""
    return next(k for k, v in panel.items() if v["site"] == label)


def residual_frame(*, panel: dict[int, dict], label: str) -> pl.DataFrame:
    """Return time, metered power, expected power and residual (metered minus expected) of one site."""
    arrays = panel[key_of(panel=panel, label=label)]
    n = len(arrays["power"])
    return pl.DataFrame(
        {
            "time": panel_stamps(n=n),
            "power": arrays["power"],
            "expected": arrays["expected"],
            "resid": arrays["resid"],
        }
    )


def g2_readings() -> pl.DataFrame:
    """Return G2's raw readings: time (UTC half-hour start) and power in MW."""
    return (
        scan_power()
        .filter(pl.col("time_series_id") == G2_SERIES)
        .select("time", "power")
        .sort("time")
        .collect()
    )


def system_prices() -> pl.DataFrame:
    """Return one row per half-hour: start (UTC) and the system sell price in pounds per MWh."""
    files = [f for d in PRICE_DIRS for f in sorted((d / "system_prices").glob("*.parquet"))]
    return (
        pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        .with_columns(
            start=pl.col("startTime").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC")
        )
        .select("start", price=pl.col("systemSellPrice"))
        .unique(subset="start", keep="first", maintain_order=True)
        .sort("start")
    )


def wilson(*, k: int, n: int) -> tuple[float, float]:
    """Return the 95% Wilson interval for a share."""
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)
