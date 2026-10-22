"""Paths, constants, and readers shared by the scripts of experiment 00233.

Copied from experiment 230's `e00230_common.py` (an experiment may not import another). The inputs
are read only, and include the fitted cores and expected power that experiment 230 exported.
Farms are labelled W1 to W3, and no farm's coordinates, identifier, or series leaves this process.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR

EXPERIMENT_ID: Final[int] = 233
SLUG: Final[str] = "wind-physics-core-part-2-capacity-availability"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
EXPERIMENT_230_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00230-wind-physics-core-part-1" / "work"
)
CORES_PATH: Final[Path] = EXPERIMENT_230_WORK / "b1_cores.json"
"""Experiment 230's fitted B1 core parameters per farm (read only)."""
EXPECTED_PATH: Final[Path] = EXPERIMENT_230_WORK / "b1_expected_power.parquet"
"""Experiment 230's B1 expected power on ERA5 at every half-hour (read only)."""
CAPACITY_PATH: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
) / "capacity.parquet"

ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""The core is fitted on valid times before this."""
DATA_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No valid time on or after this is used (kept for a finalist)."""
ABOVE_RATED_FRACTION_OF_SCALE: Final[float] = 0.95
"""A half-hour is above rated when the core expects at least this share of the fitted scale."""

BUSY_FRACTION_OF_SCALE: Final[float] = 0.2
"""The comparison row set: half-hours whose expected power is at least this share of the scale."""


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

    The rows of a farm's first weeks after a new meter, defined as experiment 33 defined them (the
    first run of 48 half-hours above 5% of capacity, plus 6 weeks), are dropped.

    Returns:
        `time_series_id`, `valid_time`, and `target`, the observed power as a fraction of the
        farm's effective capacity.
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


def daily_table(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return, per farm and UTC day, the sums that the capacity statistics are built from.

    Joins experiment 230's B1 expected power to the observed power, both as fractions of the farm's
    effective capacity, and drops half-hours with no observation.

    Returns:
        `site`, `day`, and for each of the two row sets (`above` = expected power at least
        `ABOVE_RATED_FRACTION_OF_SCALE` of the fitted scale; `busy` = expected power at least
        `BUSY_FRACTION_OF_SCALE` of it) the half-hour count `n_*`, the observed sum `obs_*`, and the
        expected sum `exp_*`.
    """
    cores = json.loads(CORES_PATH.read_text())
    scale = pl.DataFrame(
        {
            "site": list(cores),
            "scale": [cores[k]["scale_fraction_of_capacity"] for k in cores],
        }
    )
    expected = (
        pl.read_parquet(EXPECTED_PATH)
        .select("site", "valid_time", pl.col("expected_fraction").cast(pl.Float64))
        .filter(pl.col("valid_time") < DATA_END)
    )
    observed = observed_power(sites=sites).join(
        sites.select("time_series_id", "site"), on="time_series_id"
    )
    rows = (
        observed.join(expected, on=["site", "valid_time"], how="inner", validate="1:1")
        .join(scale, on="site")
        .with_columns(
            above=pl.col("expected_fraction") >= ABOVE_RATED_FRACTION_OF_SCALE * pl.col("scale"),
            busy=pl.col("expected_fraction") >= BUSY_FRACTION_OF_SCALE * pl.col("scale"),
            day=pl.col("valid_time").dt.date(),
        )
    )
    return (
        rows.group_by("site", "day")
        .agg(
            n_above=pl.col("above").sum(),
            obs_above=(pl.col("target") * pl.col("above")).sum(),
            exp_above=(pl.col("expected_fraction") * pl.col("above")).sum(),
            n_busy=pl.col("busy").sum(),
            obs_busy=(pl.col("target") * pl.col("busy")).sum(),
            exp_busy=(pl.col("expected_fraction") * pl.col("busy")).sum(),
        )
        .sort("site", "day")
    )
