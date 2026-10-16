"""Generator inputs of experiment 00036: sites, ERA5 training rows, and the power curve.

Copied from experiment 00033's `e00033_gen_common.py` (itself copied from experiment 00020's
`e00020_common.py`); an experiment may not import another experiment. The inputs it reads were built
by the stage 1 pilot and are read only.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from e00036_common import LEVELS
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import pv_sites, wind_sites
from studies.sources import PER_STUDY_DIR

PILOT_DIR: Final[Path] = PER_STUDY_DIR / "stage1_pilot"
ANALYSIS_PATH: Final[Path] = PILOT_DIR / "inputs" / "analysis_features.parquet"
ENS_TRAIN_PATH: Final[Path] = PILOT_DIR / "inputs" / "ens_features" / "train.parquet"
ENS_VALIDATION_DIR: Final[Path] = PILOT_DIR / "inputs" / "ens_features" / "val"
FRAMES_DIR: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
)
ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Training uses valid times before this: 14 days before the first validation run."""
SETTLING_LEVEL_FRACTION: Final[float] = 0.05
SETTLING_RUN_HALF_HOURS: Final[int] = 48
SETTLING_WEEKS: Final[int] = 6
"""The `new_meter_settling` rule, applied as the stage 1 pilot applied it."""
SPEED_BIN_WIDTH_M_S: Final[float] = 0.5

SOLAR_FEATURES: Final[tuple[str, ...]] = (
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

WIND_FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
)

XGBOOST_PARAMETERS: Final[dict[str, object]] = {
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "nthread": 8,
}
"""Experiment 00002's multi-quantile settings with the thread count of this experiment's slot."""
N_ROUNDS: Final[int] = 500


def site_table() -> pl.DataFrame:
    """Return the nine generator series with anonymous label, technology, and capacity.

    Returns:
        One row per series with `time_series_id`, `site` (A to F, W1 to W3), `technology`,
        `latitude`, `longitude`, and `effective_capacity_mw`. The coordinates never leave this
        process.

    Raises:
        ValueError: If the anonymisation helper's labels have changed.
    """
    capacity = pl.read_parquet(FRAMES_DIR / "capacity.parquet").select(
        "time_series_id", "effective_capacity_mw"
    )
    solar = pv_sites().select("time_series_id", "site", "latitude", "longitude")
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if set(solar["site"].to_list()) != set(SITE_LABELS) or LABEL_PERMUTATION_SEED != 784:
        msg = "the anonymisation helper's solar labels have changed"
        raise ValueError(msg)
    return (
        pl.concat(
            [
                solar.with_columns(technology=pl.lit("solar")),
                wind.with_columns(technology=pl.lit("wind")),
            ]
        )
        .join(capacity, on="time_series_id", how="left", validate="1:1")
        .sort("technology", "site")
    )


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return cleaned power over capacity from the ERA5 era to the end of training.

    Rows before each series' settling cutoff are dropped, as the stage 1 pilot did.

    Args:
        sites: `site_table()`.

    Returns:
        Columns `time_series_id`, `valid_time`, and `target` (a fraction of capacity).
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= ERA5_START,
            pl.col("time") < TRAIN_END,
        )
        .select("time_series_id", pl.col("time").alias("valid_time"), "power")
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .sort("time_series_id", "valid_time")
    )
    live = (
        pl.col("power").abs() > SETTLING_LEVEL_FRACTION * pl.col("effective_capacity_mw")
    ).fill_null(value=False)
    run_end = (
        live.cast(pl.Int32).rolling_sum(window_size=SETTLING_RUN_HALF_HOURS).over("time_series_id")
        == SETTLING_RUN_HALF_HOURS
    )
    cutoffs = (
        power.filter(run_end)
        .group_by("time_series_id")
        .agg(
            cutoff=pl.col("valid_time")
            .min()
            .dt.offset_by(f"-{SETTLING_RUN_HALF_HOURS * 30 - 30}m")
            .dt.offset_by(f"{SETTLING_WEEKS}w")
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


def power_curve(*, speed: np.ndarray, power_fraction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a non-decreasing curve of power fraction on 100 m wind speed.

    Args:
        speed: 100 m wind speeds of the training rows, m/s.
        power_fraction: Observed power over capacity of the same rows.

    Returns:
        The bin centres and the running maximum of each bin's mean power fraction, for
        `np.interp`.
    """
    bins = np.floor(speed / SPEED_BIN_WIDTH_M_S).astype(np.int64)
    frame = (
        pl.DataFrame({"bin": bins, "power": power_fraction})
        .group_by("bin")
        .agg(pl.col("power").mean())
    )
    ordered = frame.sort("bin")
    centres = (ordered["bin"].to_numpy() + 0.5) * SPEED_BIN_WIDTH_M_S
    return centres, np.maximum.accumulate(ordered["power"].to_numpy())
