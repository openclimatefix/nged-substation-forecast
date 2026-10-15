"""Paths, series, and the PV_Live inputs of experiment 00028.

The experiment asks whether a forecast of Sheffield Solar's PV_Live estimate helps the stage 2
calibrator forecast the demand of the two grid supply points
(`docs/ml-experiments/00028-pvlive-forecast-in-stage2-gsp-demand/result.md`). It reads the stage 1a
statistics and out-of-fold ERA5 predictions that experiment 00005 stored, and copies that
experiment's grid, lag, and calibrator helpers. Everything runs on one dense half-hourly grid per
series, indexed by period-ending UTC stamps, so a lag is an index offset.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 28
SLUG: Final[str] = "pvlive-forecast-in-stage2-gsp-demand"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
CONFIG: Final[str] = "from2023"
"""Experiment 00005's primary training window, from 2023-09."""

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
WORK_DIR_00005: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00005-lag-placement-stage1b-vs-stage2-demand" / "work"
)
"""Experiment 00005's stage 1a statistics, ERA5 predictions, and arms, read only."""
STATS_DIR: Final[Path] = WORK_DIR_00005 / "stats"
PV_LIVE_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent
    / "studies"
    / "downloads"
    / "market"
    / "pv_live_gsp"
    / "pv_live_gsp.parquet"
)
DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
"""Experiment 00003's inputs, read only."""
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
FEATURE_SET_PATH: Final[Path] = WORK_DIR_00005 / "feature_set.json"

GSP_PV_LIVE_NAMES: Final[dict[str, str]] = {"D1": "BICF_1", "D6": "WALP_B"}
"""Bicker Fen and Walpole, the two GSPs, and the PV_Live name of each."""

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
FINAL_TRAIN_CUTOFF: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
TRAIN_START: Final[datetime] = GRID_START
"""The solar model trains from the start of the grid; a demand model would start in 2023-09."""
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
PURGE: Final[timedelta] = timedelta(days=15)
CHAIN_STARTS: Final[tuple[datetime, ...]] = tuple(
    datetime(year, month, 1, tzinfo=UTC)
    for year, month in ((2024, 4), (2024, 7), (2024, 10), (2025, 1), (2025, 4))
)
"""Quarter starts of the forward chain; each quarter's models end `PURGE` before it starts."""
FIRST_CALIBRATION_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
CALIBRATION_STRIDE_DAYS: Final[int] = 3
VALIDATION_DIR: Final[Path] = DEMAND_DIR / "ens_every5d" / "val"
PV_FEATURES: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
    "solar_zenith",
    "solar_azimuth",
)
"""The solar model's inputs: weather at the valid time, and the sun's position."""

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95",
)  # fmt: skip
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
LAG_STEPS: Final[tuple[int, ...]] = (48, 96, 336, 672)
LAG_NAMES: Final[tuple[str, ...]] = ("lag_24h", "lag_48h", "lag_168h", "lag_336h")
RECENT_COLUMNS: Final[tuple[str, ...]] = (
    "residual_origin",
    "residual_origin_1h",
    "residual_origin_3h",
    "residual_same_time_of_day",
    "residual_same_time_of_week",
    *(f"power_{name}" for name in LAG_NAMES),
    *(f"anomaly_{name}" for name in LAG_NAMES),
)
"""The recent-origin inputs of experiment 00005's calibrator."""
PV_POSITIONS: Final[tuple[str, ...]] = (*LAG_NAMES, "same_time_of_day", "same_time_of_week")
STAT_COLUMNS: Final[tuple[str, ...]] = ("mean", "std", "q10", "q50", "q90")

XGBOOST_PARAMETERS: Final[dict[str, float | int | str]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 2,
}
N_ROUNDS: Final[int] = 500
CALIBRATOR_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 2,
}
CALIBRATOR_ROUNDS: Final[int] = 300


def series_table() -> pl.DataFrame:
    """Return the two GSP series with label, capacity, coordinates, and PV_Live name.

    The coordinates never leave this process.

    Returns:
        One row per GSP with `time_series_id`, `site`, `latitude`, `longitude`,
        `effective_capacity_mw`, `h3_res_5`, and `pv_live_name`.
    """
    table = pl.read_parquet(SERIES_PATH).filter(pl.col("site").is_in(list(GSP_PV_LIVE_NAMES)))
    metadata = pl.read_parquet(
        study_dir_for(study="quick_wins_fixed_pipeline_1156")
        / "frames"
        / "combined"
        / "fold_a"
        / "metadata.parquet",
        columns=["time_series_id", "h3_res_5"],
    )
    if table.height != len(GSP_PV_LIVE_NAMES):
        msg = "expected the two GSP series"
        raise ValueError(msg)
    return (
        table.join(metadata, on="time_series_id", how="left", validate="1:1")
        .with_columns(pv_live_name=pl.col("site").replace(GSP_PV_LIVE_NAMES))
        .sort("site")
    )


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


def grid_index(*, times: datetime | pl.Series) -> int | np.ndarray:
    """Return the grid index of one timestamp, or of a series of timestamps."""
    if isinstance(times, datetime):
        return int((times - GRID_START) / STEP)
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid."""
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def load_power(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each series' observed power over effective capacity on the grid, NaN if unobserved."""
    grid = pl.DataFrame({"valid_time": grid_times()})
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            observed.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", pl.col("power") / row["effective_capacity_mw"]
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        out[row["site"]] = joined["power"].to_numpy().astype(np.float32)
    return out


def load_pv_live(*, series: pl.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Return each series' PV_Live level and share on the grid.

    PV_Live labels each half-hour by its start (the download shifted PV_Live's own end label), and
    the grid is labelled by period end, so grid stamp `t` takes the estimate that starts at
    `t - 30 min`.

    Args:
        series: `series_table()`.

    Returns:
        For each site label, `level` (generation over the series' effective capacity) and `share`
        (PV_Live's percent of its own capacity, divided by 100), float32, NaN where missing.
    """
    grid = pl.DataFrame({"valid_time": grid_times()}).with_columns(
        start=pl.col("valid_time") - STEP
    )
    pv = pl.read_parquet(PV_LIVE_PATH)
    out = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            pv.filter(pl.col("gsp_name") == row["pv_live_name"]).select(
                start="time",
                generation_mw="generation_mw",
                percent="generation_percent_of_capacity",
            ),
            on="start",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        out[row["site"]] = {
            "level": (joined["generation_mw"] / row["effective_capacity_mw"])
            .to_numpy()
            .astype(np.float32),
            "share": (joined["percent"] / 100.0).to_numpy().astype(np.float32),
        }
    return out


def read_era5_pred(*, seed: int, sites: list[str]) -> dict[str, np.ndarray]:
    """Read experiment 00005's stage 1a predictions at ERA5 weather, keyed by site label.

    Args:
        seed: The fitting seed.
        sites: The 20 site labels in experiment 00005's series-table order (D1 to D22 without the
            dropped D7 and D20).

    Returns:
        Each site label's array on the grid: out-of-fold before 2025-07-01, final model after.
    """
    path = WORK_DIR_00005 / "era5_pred" / f"{CONFIG}_seed{seed}.npy"
    return dict(zip(sites, np.load(path), strict=True))


def all_sites() -> list[str]:
    """Return the 20 site labels in the order experiment 00005 stacked its ERA5 predictions."""
    table = pl.read_parquet(SERIES_PATH).filter(~pl.col("site").is_in(["D7", "D20"]))
    return table.sort(pl.col("site").str.slice(1).cast(pl.Int32))["site"].to_list()


def recent_origin_inputs(
    *, power: np.ndarray, residual: np.ndarray, target: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Return experiment 00005's recent-origin inputs, using only observations before the origin.

    Args:
        power: Observed power over effective capacity on the grid.
        residual: Observed power minus stage 1a at the observed ERA5 weather, on the grid.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        Shape (cases, len(RECENT_COLUMNS)).

    Raises:
        ValueError: If a case's valid time is not after its origin.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    last = origin - 1
    gap = target - last
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(int)
    lead = target - origin
    columns = [
        take(values=residual, index=last),
        take(values=residual, index=last - 2),
        take(values=residual, index=last - 6),
        take(values=residual, index=day),
        take(values=residual, index=week),
    ]
    usable = [lag > lead for lag in LAG_STEPS]
    columns += [
        np.where(ok, take(values=power, index=target - lag), np.nan)
        for lag, ok in zip(LAG_STEPS, usable, strict=True)
    ]
    columns += [
        np.where(ok, take(values=residual, index=target - lag), np.nan)
        for lag, ok in zip(LAG_STEPS, usable, strict=True)
    ]
    return np.stack(columns, axis=1).astype(np.float32)


def pinball_quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)
