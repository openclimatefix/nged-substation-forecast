"""The readers of experiment 00288 that serve the three metered wind farms (W1 to W3).

Copied from experiment 00259's `e00259_common.py`. The inputs are read only, the farms are labelled
W1 to W3, and no farm's coordinates, identifier, or series leaves this process.
"""

from pathlib import Path
from typing import Final

import e00288_common as common
import numpy as np
import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR, STUDIES_DATA_DIR

BEST_GUESS_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "generator_tables"
    / "bg_full.parquet"
)
MASK_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00210-wind-idle-noise-mva-artefact" / "mask.parquet"
)
PILOT_INPUTS: Final[Path] = PER_STUDY_DIR / "stage1_pilot" / "inputs"
ANALYSIS_PATH: Final[Path] = PILOT_INPUTS / "analysis_features.parquet"
ENS_VALIDATION_DIR: Final[Path] = PILOT_INPUTS / "ens_features" / "val"
ENS_TRAIN_PATH: Final[Path] = PILOT_INPUTS / "ens_features" / "train.parquet"
ERA5_TEMPERATURE_PATH: Final[Path] = (
    STUDIES_DATA_DIR
    / "downloads"
    / "reanalysis"
    / "ERA5"
    / "site_points"
    / "beam_diffuse_open_meteo.parquet"
)
CAPACITY_PATH: Final[Path] = (
    PER_STUDY_DIR / "quick_wins_fixed_pipeline_1156" / "frames" / "combined" / "fold_a"
) / "capacity.parquet"


def site_table() -> pl.DataFrame:
    """Return the 3 wind farms with anonymous label and capacity (coordinates stay private)."""
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if LABEL_PERMUTATION_SEED != 784 or not SITE_LABELS:
        msg = "the anonymisation helper has changed"
        raise ValueError(msg)
    capacity = pl.read_parquet(CAPACITY_PATH).select("time_series_id", "effective_capacity_mw")
    return wind.join(capacity, on="time_series_id", how="left", validate="1:1").sort("site")


def era5_temperature(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return ERA5 2 m temperature in degrees Celsius at each farm's nearest grid cell, half-hourly.

    Args:
        sites: The `site_table` rows, whose coordinates choose each farm's nearest 0.25 degree cell.

    Returns:
        `time_series_id`, `valid_time`, and `temperature_c`, with the half-hour's value the mean of
        the hourly instants either side of a `:30` time.
    """
    cells = pl.read_parquet(ERA5_TEMPERATURE_PATH).select("time", "latitude", "longitude", "temp_c")
    centres = cells.select("latitude", "longitude").unique()
    pieces = []
    for row in sites.iter_rows(named=True):
        distance = (centres["latitude"] - row["latitude"]) ** 2 + (
            (centres["longitude"] - row["longitude"]) * np.cos(np.radians(row["latitude"]))
        ) ** 2
        nearest = centres.filter(distance == distance.min())
        hourly = (
            cells.join(nearest, on=["latitude", "longitude"]).select("time", "temp_c").sort("time")
        )
        shifted = hourly.select(time=pl.col("time") - pl.duration(hours=1), later=pl.col("temp_c"))
        half = hourly.join(shifted, on="time", how="left").select(
            valid_time=pl.col("time") + pl.duration(minutes=30),
            temperature_c=(pl.col("temp_c") + pl.col("later")) / 2,
        )
        whole = hourly.select(valid_time=pl.col("time"), temperature_c=pl.col("temp_c"))
        pieces.append(
            pl.concat([whole, half.drop_nulls()])
            .with_columns(time_series_id=pl.lit(row["time_series_id"], dtype=pl.Int32))
            .select("time_series_id", "valid_time", "temperature_c")
        )
    return pl.concat(pieces).sort("time_series_id", "valid_time")


def era5_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the ERA5 weather at every half-hour of each farm, with `WEATHER_COLUMNS`."""
    analysis = (
        pl.scan_parquet(ANALYSIS_PATH)
        .filter(pl.col("time_series_id").is_in(sites["time_series_id"].to_list()))
        .select("time_series_id", "valid_time", *common.WEATHER_COLUMNS[:-1])
        .collect()
    )
    return analysis.join(
        era5_temperature(sites=sites), on=["time_series_id", "valid_time"], how="inner"
    ).with_columns(pl.col(common.WEATHER_COLUMNS).cast(pl.Float32))


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return power over capacity from the ERA5 era to the end of training, as experiment 52 did.

    The rows of a farm's first weeks after a new meter, defined as experiment 33 defined them (the
    first run of 48 half-hours above 5% of capacity, plus 6 weeks), are dropped.
    """
    power = (
        scan_power()
        .filter(
            pl.col("time_series_id").is_in(sites["time_series_id"].to_list()),
            pl.col("time") >= common.ERA5_START,
            pl.col("time") < common.TRAIN_END,
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


def training_table(*, sites: pl.DataFrame, era5: pl.DataFrame) -> pl.DataFrame:
    """Return ERA5 weather joined to the observed target, before the end of training."""
    return era5.join(
        observed_training_power(sites=sites),
        on=["time_series_id", "valid_time"],
        how="inner",
        validate="1:1",
    ).sort("time_series_id", "valid_time")


def idle_mask() -> pl.DataFrame:
    """Return experiment 210's rows to drop: `time_series_id` and `valid_time` (kept private)."""
    return pl.read_parquet(MASK_PATH).select("time_series_id", "valid_time")


def cases_table(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the best guess's wind cases (seed 0) with the observed fraction, lead, and mask flag.

    Columns: the key columns, `site`, `lead_hours`, `actual_norm` (a fraction of capacity),
    `common.lead_day`, and `masked` (true where experiment 210's mask drops the row).
    """
    cases = (
        pl.read_parquet(BEST_GUESS_PATH)
        .filter((pl.col("technology") == "wind") & (pl.col("seed") == 0))
        .select(*common.KEY_COLUMNS, "lead_hours", "actual_norm")
        .join(sites.select("time_series_id", "site"), on="time_series_id", validate="m:1")
    )
    mask = idle_mask().with_columns(masked=pl.lit(value=True))
    return (
        cases.with_columns(
            lead_day=common.lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"), valid_time=pl.col("valid_time")
            )
        )
        .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(masked=pl.col("masked").fill_null(value=False))
        .sort(*common.KEY_COLUMNS)
    )


def ens_member_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return every ENS member's weather at every validation case, sorted by case then member.

    Columns: the key columns, `ensemble_member`, and `WEATHER_COLUMNS`, with `temperature_c` the
    ENS 2 m temperature (already in degrees Celsius on disk).
    """
    ids = sites["time_series_id"].to_list()
    frames = [
        pl.scan_parquet(path)
        .filter(pl.col("time_series_id").is_in(ids))
        .select(
            *common.KEY_COLUMNS,
            "ensemble_member",
            *common.WEATHER_COLUMNS[:-1],
            temperature_c=pl.col("temperature_2m"),
        )
        .collect()
        for path in sorted(ENS_VALIDATION_DIR.glob("init=*.parquet"))
    ]
    return (
        pl.concat(frames)
        .with_columns(pl.col(common.WEATHER_COLUMNS).cast(pl.Float32))
        .sort(*common.KEY_COLUMNS, "ensemble_member")
    )


def ens_control_training_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the ENS control member's weather joined to observed power, before `TRAIN_END`."""
    ids = sites["time_series_id"].to_list()
    return (
        pl.scan_parquet(ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("valid_time") < common.TRAIN_END))
        .select(
            "time_series_id",
            "valid_time",
            "power",
            *common.WEATHER_COLUMNS[:-1],
            temperature_c=pl.col("temperature_2m"),
        )
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .select(
            "time_series_id",
            "valid_time",
            *[pl.col(c).cast(pl.Float32) for c in common.WEATHER_COLUMNS],
            target=pl.col("power") / pl.col("effective_capacity_mw"),
        )
        .filter(pl.col("target").is_finite())
    )
