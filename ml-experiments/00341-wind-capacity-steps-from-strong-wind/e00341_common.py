"""Paths, constants, and readers shared by the scripts of experiment 00341.

The half-hour table is built from experiment 230's exported cores and expected power (read only)
and the observed power of W1 to W3, as experiment 233 did. Farms are labelled W1 to W3, and no
farm's coordinates, identifier, or series leaves this process.
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 341
SLUG: Final[str] = "wind-capacity-steps-from-strong-wind"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
E230_WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work"
E210_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00210-wind-idle-noise-mva-artefact"
CAPACITY_PATH: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
) / "capacity.parquet"
SITES: Final[tuple[str, ...]] = ("W1", "W2", "W3")
N_TURBINES: Final[dict[str, int]] = {"W1": 13, "W2": 8, "W3": 2}
"""Turbine counts from the Renewable Energy Planning Database rows (2009 to 2010 vintage)."""

DAY0: Final[date] = date(2019, 9, 1)
ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
DATA_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
N_DAYS: Final[int] = (date(2026, 6, 30) - DAY0).days + 1
CAL_START_DAY: Final[int] = (date(2021, 9, 1) - DAY0).days
CAL_END_DAY: Final[int] = (date(2024, 6, 30) - DAY0).days
START_FIRST_DAY: Final[int] = (date(2024, 7, 1) - DAY0).days
START_LAST_DAY: Final[int] = (date(2025, 12, 31) - DAY0).days
FIT_END_DAY: Final[int] = (date(2025, 6, 17) - DAY0).days

STRONG_MARGIN_M_S: Final[float] = 2.0
"""The strong set starts this far above the saturation speed (1 and 3 are sensitivities)."""
SATURATION_FRACTION: Final[float] = 0.95
MIN_HALF_HOURS: Final[int] = 20
MIN_HALF_HOURS_DAILY: Final[int] = 4
MID_LOW: Final[float] = 0.3
MID_HIGH: Final[float] = 0.8
WEIGHTED_LOW: Final[float] = 0.25
BUSY_LOW: Final[float] = 0.2
EPS_RATIO: Final[float] = 0.02
LOG_RATIO_CLIP: Final[tuple[float, float]] = (-2.0, 1.0)
SPEED_BIAS_PRIOR_SD: Final[float] = 0.05


@dataclass(frozen=True)
class Farm:
    """One farm's half-hours as arrays sorted by time, all as shares of the fitted scale."""

    site: str
    day: np.ndarray
    obs: np.ndarray
    exp: np.ndarray
    speed: np.ndarray
    v_sat: float
    cut_out: float
    scale: float
    n_turbines: int


def site_table() -> pl.DataFrame:
    """Return the 3 wind farms with anonymous label and capacity (coordinates stay private)."""
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if LABEL_PERMUTATION_SEED != 784 or not SITE_LABELS:
        msg = "the anonymisation helper has changed"
        raise ValueError(msg)
    capacity = pl.read_parquet(CAPACITY_PATH).select("time_series_id", "effective_capacity_mw")
    return wind.join(capacity, on="time_series_id", how="left", validate="1:1").sort("site")


def observed_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return power over capacity from the ERA5 era to `DATA_END`, as experiment 230 did.

    The rows of a farm's first weeks after a new meter (the first run of 48 half-hours above 5% of
    capacity, plus 6 weeks) are dropped, and so are non-finite values.
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= ERA5_START,
            pl.col("time") < DATA_END,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    run = 48
    live = (pl.col("power").abs() > 0.05 * pl.col("effective_capacity_mw")).fill_null(value=False)
    run_end = live.cast(pl.Int32).rolling_sum(window_size=run).over("time_series_id") == run
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time").min().dt.offset_by(f"-{run * 30 - 30}m").dt.offset_by("6w")
        )
    )
    return (
        power.join(cutoffs, on="time_series_id", how="left")
        .filter(pl.col("cutoff").is_null() | (pl.col("valid_time") >= pl.col("cutoff")))
        .filter(pl.col("power").is_finite())
        .select(
            "time_series_id", "valid_time", target=pl.col("power") / pl.col("effective_capacity_mw")
        )
    )


def build_half_hours() -> pl.DataFrame:
    """Join observed power to experiment 230's expected power and speed; write the work table."""
    sites = site_table()
    expected = (
        pl.read_parquet(E230_WORK / "b1_expected_power.parquet")
        .filter(pl.col("valid_time") < DATA_END)
        .select(
            "site", "valid_time", pl.col("expected_fraction").cast(pl.Float64), "wind_speed_100m"
        )
    )
    observed = observed_power(sites=sites).join(
        sites.select("time_series_id", "site"), on="time_series_id"
    )
    rows = observed.join(expected, on=["site", "valid_time"], how="inner", validate="1:1").select(
        "site", "valid_time", "target", "expected_fraction", "wind_speed_100m"
    )
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows.sort("site", "valid_time").write_parquet(WORK_DIR / "half_hours.parquet")
    return rows


def power_curve_lookup(
    *, speed: np.ndarray, exp: np.ndarray, cut_out: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (speed grid, expected share of scale) from the core's deterministic output.

    The expected power is a function of ERA5 speed alone (the core has no other input), so its
    median in 0.1 m/s bins below cut-out is the curve.
    """
    grid = np.arange(0.0, cut_out - 0.5, 0.1)
    index = np.clip(np.digitize(speed, grid) - 1, 0, len(grid) - 1)
    curve = np.full(len(grid), np.nan)
    for i in range(len(grid)):
        values = exp[index == i]
        if len(values):
            curve[i] = float(np.median(values))
    ok = np.isfinite(curve)
    return grid[ok], curve[ok]


def load_farms() -> dict[str, Farm]:
    """Load the work table as `Farm` arrays, normalised by the fitted scale."""
    cores = json.loads((E230_WORK / "b1_cores.json").read_text())
    table = pl.read_parquet(WORK_DIR / "half_hours.parquet")
    farms: dict[str, Farm] = {}
    for site in SITES:
        part = table.filter(pl.col("site") == site).sort("valid_time")
        scale = float(cores[site]["scale_fraction_of_capacity"])
        speed = part["wind_speed_100m"].to_numpy().astype(np.float64)
        exp = part["expected_fraction"].to_numpy() / scale
        obs = part["target"].to_numpy() / scale
        hours = (
            part.select((pl.col("valid_time") - pl.lit(ERA5_START)).dt.total_minutes() // 30)
            .to_series()
            .to_numpy()
        )
        cut_out = float(cores[site]["cut_out_m_s"])
        grid, curve = power_curve_lookup(speed=speed, exp=exp, cut_out=cut_out)
        v_sat = float(grid[np.argmax(curve >= SATURATION_FRACTION)])
        farms[site] = Farm(
            site=site,
            day=(hours // 48).astype(np.int32),
            obs=obs,
            exp=exp,
            speed=speed,
            v_sat=v_sat,
            cut_out=cut_out,
            scale=scale,
            n_turbines=N_TURBINES[site],
        )
    return farms
