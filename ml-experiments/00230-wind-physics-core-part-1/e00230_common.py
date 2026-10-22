"""Paths, constants, readers, and scoring shared by the scripts of experiment 00230.

Copied from experiments 00033 and 00052 (`e00033_gen_common.py`, `e00052_common.py`); an experiment
may not import another experiment. The inputs it reads are read only. Farms are labelled W1 to W3,
and no farm's coordinates, identifier, or series leaves this process.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.anonymise import LABEL_PERMUTATION_SEED, SITE_LABELS
from studies.power import scan_power
from studies.pv_dataset import wind_sites
from studies.sources import ML_EXPERIMENTS_DATA_DIR, PER_STUDY_DIR, STUDIES_DATA_DIR

EXPERIMENT_ID: Final[int] = 230
SLUG: Final[str] = "wind-physics-core-part-1"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
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

SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
MEMBERS: Final[int] = 51
ERA5_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
TRAIN_END: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Training uses valid times before this: 14 days before the first validation run."""
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
POINTS_PER_FRACTION: Final[float] = 100.0
"""Scores are in points of capacity: a fraction of capacity times this."""

LEVELS: Final[np.ndarray] = np.array([0.01, 0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95, 0.99])
"""The 11 levels every arm forecasts: experiment 33's nine, with the 0.01 and 0.99 tails."""
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
NINE_INDEX: Final[np.ndarray] = np.arange(1, 10)
"""Positions in `LEVELS` of experiment 33's nine levels."""
MEDIAN_INDEX: Final[int] = 5
LOWER_INDEX: Final[int] = 1
UPPER_INDEX: Final[int] = 9
NINE_LEVELS: Final[np.ndarray] = LEVELS[NINE_INDEX]
NINE_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (NINE_LEVELS[:-1] + NINE_LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass each of the nine levels carries (experiment 33's quantile score weights)."""
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Mass each of the 11 levels carries in a member mixture."""
FINE_LEVELS: Final[np.ndarray] = np.arange(1, 200) / 200
"""The 199 levels on which each mixture's quantile function is stored."""
NAMED_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q50", "q95", "q99")

TARGET_COVERAGE: Final[float] = 0.90
SPREAD_GRID: Final[np.ndarray] = np.round(np.arange(0.60, 3.001, 0.05), 2)
MIN_FIT_CASES: Final[int] = 5_000
TRIVIAL_LEVEL: Final[float] = 0.01
TELEMETRY_DELAY: Final = np.timedelta64(210, "m")
"""A case is known to the spread-factor fit 3.5 hours after its valid time."""
SHORT_BAND_LAST_DAY: Final[int] = 2
"""The spread factor is fitted on lead days 0 to 2 and, separately, on later lead days."""

LEAD_DAY_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("days 0-2", 0, 2),
    ("days 1-2", 1, 2),
    ("days 3-6", 3, 6),
    ("days 7-10", 7, 10_000),
    ("all leads", 0, 10_000),
)
"""Lead-day bands, by calendar days from the ENS run's date to the valid time (inclusive)."""
STANDARD_AIR_DENSITY_KG_M3: Final[float] = 1.225
DRY_AIR_GAS_CONSTANT: Final[float] = 287.05
FIXED_SURFACE_PRESSURE_PA: Final[float] = 100_000.0
"""ERA5 surface pressure at the farms is not on disk, so every arm uses this fixed value."""
WEATHER_COLUMNS: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_dir_sin_10m",
    "wind_dir_cos_10m",
    "wind_dir_sin_100m",
    "wind_dir_cos_100m",
    "temperature_c",
)
XGBOOST_FEATURES: Final[tuple[str, ...]] = WEATHER_COLUMNS[:-1]
"""Experiment 52's six wind features, which carry no temperature."""
IDLE_WIND_M_S: Final[float] = 2.0
"""Experiment 210's idle floor: the power distribution at ERA5 100 m wind below this speed."""


def site_table() -> pl.DataFrame:
    """Return the 3 wind farms with anonymous label and capacity (coordinates stay private)."""
    wind = wind_sites().select("time_series_id", "site", "latitude", "longitude")
    if LABEL_PERMUTATION_SEED != 784 or not SITE_LABELS:
        msg = "the anonymisation helper has changed"
        raise ValueError(msg)
    capacity = pl.read_parquet(CAPACITY_PATH).select("time_series_id", "effective_capacity_mw")
    return wind.join(capacity, on="time_series_id", how="left", validate="1:1").sort("site")


def lead_day(*, power_fcst_init_time: pl.Expr, valid_time: pl.Expr) -> pl.Expr:
    """Return calendar days from the ENS run's date to the valid time's date."""
    return (valid_time.dt.date() - power_fcst_init_time.dt.date()).dt.total_days().cast(pl.Int32)


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
        .select("time_series_id", "valid_time", *WEATHER_COLUMNS[:-1])
        .collect()
    )
    return analysis.join(
        era5_temperature(sites=sites), on=["time_series_id", "valid_time"], how="inner"
    ).with_columns(pl.col(WEATHER_COLUMNS).cast(pl.Float32))


def observed_training_power(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return power over capacity from the ERA5 era to the end of training, as experiment 52 did.

    The rows of a farm's first weeks after a new meter, defined as experiment 33 defined them (the
    first run of 48 half-hours above 5% of capacity, plus 6 weeks), are dropped.
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
    `lead_day`, and `masked` (true where experiment 210's mask drops the row).
    """
    cases = (
        pl.read_parquet(BEST_GUESS_PATH)
        .filter((pl.col("technology") == "wind") & (pl.col("seed") == 0))
        .select(*KEY_COLUMNS, "lead_hours", "actual_norm")
        .join(sites.select("time_series_id", "site"), on="time_series_id", validate="m:1")
    )
    mask = idle_mask().with_columns(masked=pl.lit(value=True))
    return (
        cases.with_columns(
            lead_day=lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"), valid_time=pl.col("valid_time")
            )
        )
        .join(mask, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .with_columns(masked=pl.col("masked").fill_null(value=False))
        .sort(*KEY_COLUMNS)
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
            *KEY_COLUMNS,
            "ensemble_member",
            *WEATHER_COLUMNS[:-1],
            temperature_c=pl.col("temperature_2m"),
        )
        .collect()
        for path in sorted(ENS_VALIDATION_DIR.glob("init=*.parquet"))
    ]
    return (
        pl.concat(frames)
        .with_columns(pl.col(WEATHER_COLUMNS).cast(pl.Float32))
        .sort(*KEY_COLUMNS, "ensemble_member")
    )


def ens_control_training_rows(*, sites: pl.DataFrame) -> pl.DataFrame:
    """Return the ENS control member's weather joined to observed power, before `TRAIN_END`."""
    ids = sites["time_series_id"].to_list()
    return (
        pl.scan_parquet(ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id").is_in(ids) & (pl.col("valid_time") < TRAIN_END))
        .select(
            "time_series_id",
            "valid_time",
            "power",
            *WEATHER_COLUMNS[:-1],
            temperature_c=pl.col("temperature_2m"),
        )
        .collect()
        .join(sites.select("time_series_id", "effective_capacity_mw"), on="time_series_id")
        .select(
            "time_series_id",
            "valid_time",
            *[pl.col(c).cast(pl.Float32) for c in WEATHER_COLUMNS],
            target=pl.col("power") / pl.col("effective_capacity_mw"),
        )
        .filter(pl.col("target").is_finite())
    )


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return experiment 33's score per case: twice the mass-weighted pinball loss of nine levels.

    Args:
        quantiles: Shape (cases, 11) quantiles at `LEVELS`; only the nine levels of experiment 33
            are used.
        actual: The observed fractions of capacity.

    Returns:
        Shape (cases,) scores in fractions of capacity.
    """
    difference = actual[:, None] - quantiles[:, NINE_INDEX]
    pinball = np.where(difference >= 0, NINE_LEVELS * difference, (NINE_LEVELS - 1) * difference)
    return 2.0 * (pinball * NINE_MASSES).sum(axis=1)


def pinball_at(*, quantiles: np.ndarray, actual: np.ndarray, column: str) -> np.ndarray:
    """Return the pinball loss of one named level per case."""
    index = LEVEL_COLUMNS.index(column)
    level = LEVELS[index]
    difference = actual - quantiles[:, index]
    return np.where(difference >= 0, level * difference, (level - 1) * difference)


def covered(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return whether the observation lies in the 5% to 95% band, per case."""
    return (actual >= quantiles[:, LOWER_INDEX]) & (actual <= quantiles[:, UPPER_INDEX])


def scale_about_median(*, quantiles: np.ndarray, factor: float) -> np.ndarray:
    """Return `median + factor * (q - median)`, clipped to 0 to 1 and sorted."""
    median = quantiles[:, MEDIAN_INDEX : MEDIAN_INDEX + 1]
    return np.sort(np.clip(median + factor * (quantiles - median), 0.0, 1.0), axis=1)


def fit_spread_factor(*, quantiles: np.ndarray, actual: np.ndarray) -> float:
    """Return the grid factor whose 5% to 95% coverage is nearest `TARGET_COVERAGE`."""
    gaps = [
        abs(
            covered(
                quantiles=scale_about_median(quantiles=quantiles, factor=float(k)), actual=actual
            ).mean()
            - TARGET_COVERAGE
        )
        for k in SPREAD_GRID
    ]
    return float(SPREAD_GRID[int(np.argmin(gaps))])


def mixture_quantiles(*, member_quantiles: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Return each case's equal-weight member mixture, read at `LEVELS`.

    Args:
        member_quantiles: Shape (cases, members, 11) member quantiles, each member's levels sorted.
        chunk: Cases sorted at a time.

    Returns:
        Shape (cases, 11) float32 quantiles of the mixture, by linear interpolation of the mixture's
        quantile function on `FINE_LEVELS`.
    """
    cases, members, levels = member_quantiles.shape
    weights = np.tile(ATOM_MASSES / members, members)
    fine = np.empty((cases, FINE_LEVELS.size), dtype=np.float32)
    for start in range(0, cases, chunk):
        flat = member_quantiles[start : start + chunk].reshape(-1, members * levels)
        order = np.argsort(flat, axis=1, kind="stable")
        values = np.take_along_axis(flat, order, axis=1)
        mass = weights[order]
        midpoint = np.cumsum(mass, axis=1) - mass / 2
        for row in range(flat.shape[0]):
            fine[start + row] = np.interp(FINE_LEVELS, midpoint[row], values[row])
    position = LEVELS * 200 - 1
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, FINE_LEVELS.size - 1)
    fraction = (position - low).astype(np.float32)
    return fine[:, low] * (1 - fraction) + fine[:, high] * fraction


def apply_spread(*, cases: pl.DataFrame, mixture: np.ndarray) -> np.ndarray:
    """Return the quantiles after the spread factor, fitted month by month on earlier cases.

    Follows experiment 33's wind rule: one factor for lead days 0 to 2 and one for the later days,
    each fitted on the non-trivial cases whose valid time precedes the month's first run by the
    telemetry delay, and applied from the month's first run.

    Args:
        cases: The `cases_table` rows in the order of `mixture`.
        mixture: Shape (cases, 11) mixture quantiles.

    Returns:
        Shape (cases, 11) quantiles; months with too few earlier cases keep the mixture.
    """
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    valid = cases["valid_time"].to_numpy()
    init = cases["power_fcst_init_time"].to_numpy()
    month = cases["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    short = cases["lead_day"].to_numpy() <= SHORT_BAND_LAST_DAY
    scorable = mixture[:, UPPER_INDEX] > TRIVIAL_LEVEL
    out = mixture.copy()
    for month_label in sorted(set(month)):
        in_month = month == month_label
        cutoff = init[in_month].min()
        for group in (short, ~short):
            target = in_month & group & scorable
            fit_index = np.flatnonzero((valid <= cutoff - TELEMETRY_DELAY) & group & scorable)
            if not target.any() or fit_index.size < MIN_FIT_CASES:
                continue
            factor = fit_spread_factor(quantiles=mixture[fit_index], actual=actual[fit_index])
            out[target] = scale_about_median(quantiles=mixture[target], factor=factor)
    return out
