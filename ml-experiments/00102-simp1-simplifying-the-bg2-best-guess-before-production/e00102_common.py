"""Paths, series, grids, lags, and the 11-level quantile arithmetic of experiment 00102.

Experiment 00102 (SIMP1) removes parts of the BG2 best guess one at a time and scores the result
against BG2 (`docs/ml-experiments/00102-*/result.md`). The helpers are copied from experiment
00087's `e00087_common.py`, with a work folder of their own and a read-only path to experiment
00087's work folder. An experiment may not import another experiment, so the copies are the only
link. The 14-day correction lives in `e00102_stack.py`.
"""

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 102
SLUG: Final[str] = "simp1-simplifying-the-bg2-best-guess-before-production"
BG2_SLUG: Final[str] = "bg2-best-guess-assembled-with-drop-one-ablations"
ROW_SET_ID: Final[str] = "fold_a_daily_gap_v1"
"""The daily 00:00 UTC runs of the validation year that `fold_a_every5d_v1` does not hold."""
ROW_SET_ID_EVERY5D: Final[str] = "fold_a_every5d_v1"
SEED: Final[int] = 0

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and boosters; private and never committed."""
BG2_WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00087-{BG2_SLUG}" / "work"
"""Experiment 00087's work folder, read only: its member predictions, arms, and blend weights."""
EXPERIMENT_00048_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00048-weekly-retraining-confirmation-and-extension-demand-generators"
    / "work"
)
"""Experiment 00048's work folder, read only: its frames of the 292 gap runs."""
FRAMES_DEMAND_DIR: Final[Path] = EXPERIMENT_00048_WORK_DIR / "frames_demand"
"""The 292 gap runs' demand frames, one `init=<date>.parquet` file per run; read only."""
FRAMES_GENERATORS_DIR: Final[Path] = EXPERIMENT_00048_WORK_DIR / "frames_generators"
"""All 365 daily runs' generator frames, one `init=<date>.parquet` file per run; read only."""
ANCHOR_PREDS_DIR: Final[Path] = WORK_DIR / "anchor_preds"
"""Demand refit predictions, one folder per member and one file per series and refit."""
GENERATOR_PREDS_DIR: Final[Path] = WORK_DIR / "generator_preds"
GPU_HOURS_LOG: Final[Path] = WORK_DIR / "gpu_hours.jsonl"
"""One JSON line per fit job: its name, wall-clock seconds, and the workers running alongside."""
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
METRICS_DIR: Final[Path] = WORK_DIR / "metrics"
RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"

EXPERIMENT_00034_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00034-retraining-cadence-and-cheap-proxies-demand" / "work"
)
"""Experiment 00034's intermediate tables, read only."""
AFTER_DIR: Final[Path] = EXPERIMENT_00034_WORK_DIR / "ens_after_training"
"""The control member's daily runs from 2025-07-01 to 2026-06-29, read only."""
EXPERIMENT_00034_ARMS_DIR: Final[Path] = EXPERIMENT_00034_WORK_DIR / "arms"

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
"""The stage 1 pilot's demand inputs, read only."""
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
VALIDATION_DIR: Final[Path] = DEMAND_DIR / "ens_every5d" / "val"
"""The 73 demand frames of `fold_a_every5d_v1`, read only."""
ENS_TRAIN_PATH: Final[Path] = DEMAND_DIR / "ens_every5d" / "train.parquet"
FEATURE_SET_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00005-lag-placement-stage1b-vs-stage2-demand"
    / "work"
    / "feature_set.json"
)
"""Experiment 00006's final Primary column list, as experiment 00005 froze it; read only."""
FOLD_A_FRAMES_DIR: Final[Path] = (
    study_dir_for(study="quick_wins_fixed_pipeline_1156") / "frames" / "combined" / "fold_a"
)
NGED_METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
"""The metadata table, read only for each series' metered units."""
ARCHIVE_DIR: Final[Path] = REPO_DATA_DIR / "NWP" / "nwp_model_id=ECMWF_ENS_0_25_degree"
"""The ENS archive's Delta table directory, one partition folder per initialisation time."""
POWER_FORECASTS_DIR: Final[Path] = REPO_DATA_DIR / "power_forecasts"

DROPPED_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
"""The battery and the biofuel generator, whose substation type is EHV Customer."""
SUBSTATION_TYPES: Final[frozenset[str]] = frozenset({"Primary", "BSP", "GSP"})
GROUPS: Final[tuple[str, ...]] = ("Primary", "BSP", "GSP")

FIRST_RUN: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_RUN: Final[datetime] = datetime(2026, 6, 30, tzinfo=UTC)
EVERY5D_STRIDE_DAYS: Final[int] = 5
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
"""`power_fcst_init_time` is the run's initialisation time plus this."""
GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
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
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass of each of the 11 levels: the interval between the mid-points to its neighbours.

The lowest level covers [0, mid-point) and the highest covers (mid-point, 1].
"""
MEDIAN: Final[int] = 5
"""Position of the median (q50) among the 11 levels."""
INNER_LEVEL_COLUMNS: Final[tuple[str, ...]] = LEVEL_COLUMNS[1:-1]
"""The nine levels (q05 to q95) of experiments 00034 and 00048, for the reproduction check."""
INNER_LEVELS: Final[np.ndarray] = LEVELS[1:-1]
INNER_ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (INNER_LEVELS[:-1] + INNER_LEVELS[1:]) / 2, [1.0]])
)
"""Experiment 00048's nine-level masses, which the reproduction check scores with."""
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("1-2 d", 24.0, 48.0),
    ("2-6 d", 48.0, 144.0),
    ("6-10 d", 144.0, 240.0),
    ("10+ d", 240.0, 10_000.0),
)
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")

DERIVED: Final[tuple[str, ...]] = (
    "solar_zenith",
    "solar_azimuth",
    "hour_local",
    "day_of_week_local",
    "covid_indicator",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
"""Stage 1a columns computed from the timestamp and the location, not read from a weather file."""
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
"""The 0/1 lockdown indicator is 1 from the first to the last of these days, inclusive."""
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_CLIP_DAYS: Final[int] = 14
QUANTILE_PARAMETERS: Final[dict[str, object]] = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "colsample_bynode": 1.0,
    "tree_method": "hist",
    "device": "cuda",
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
}
"""N1's parameters (experiment 00003's, with the 11-level quantile loss); callers set `nthread`."""
N_ROUNDS: Final[int] = 500
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
SHORT_HALF_LIFE_DAYS: Final[float] = 0.5 * 365.25
SPREAD_FACTOR: Final[float] = 1.15
"""The calibrator spread factor of experiment 00018, applied to the demand arms."""
CORRECTION_DAYS: Final[tuple[int, ...]] = (7, 14)
MIN_CORRECTION_DAYS: Final[int] = 3
TELEMETRY_DELAY: Final[timedelta] = timedelta(hours=3)
"""A value is usable once it is this old, plus the half-hour it covers."""
NEWEST_AGE: Final[int] = int(TELEMETRY_DELAY / STEP) + 1
"""Half-hours between the origin and the newest usable observation (3.5 hours)."""
LAG_COLUMNS: Final[tuple[str, ...]] = (
    "lag_origin",
    "lag_same_time_latest_day",
    "lag_same_time_latest_week",
)
N1_EXTRA_COLUMNS: Final[tuple[str, ...]] = ("lead_hours",)


@dataclass(frozen=True)
class SeriesArrays:
    """One series' arrays on the dense half-hourly grid.

    Attributes:
        features: Stage 1a's input columns, float32, one row per grid half-hour.
        power: Observed power over effective capacity, NaN where unobserved.
        valid_time: The grid's UTC timestamps.
    """

    features: np.ndarray
    power: np.ndarray
    valid_time: pl.Series


def run_dates(*, subset: str) -> list[datetime]:
    """Return the 00:00 UTC initialisation times of a set of daily ENS runs, sorted.

    Args:
        subset: `all` for the 365 daily runs of the validation year, `every5d` for the 73 of
            `fold_a_every5d_v1`, or `gap` for the 292 of `fold_a_daily_gap_v1`.

    Returns:
        The runs' initialisation times.

    Raises:
        ValueError: If `subset` is not one of the three names.
    """
    days = (LAST_RUN - FIRST_RUN).days + 1
    every_day = [FIRST_RUN + timedelta(days=i) for i in range(days)]
    if subset == "all":
        return every_day
    on_stride = [d for i, d in enumerate(every_day) if i % EVERY5D_STRIDE_DAYS == 0]
    if subset == "every5d":
        return on_stride
    if subset == "gap":
        return [d for d in every_day if d not in set(on_stride)]
    msg = f"unknown run subset {subset!r}"
    raise ValueError(msg)


def origin_of(*, run: datetime) -> datetime:
    """Return a run's `power_fcst_init_time`: its initialisation time plus the publication delay."""
    return run + PUBLICATION_DELAY


def all_origins() -> list[datetime]:
    """Return the `power_fcst_init_time` of the 365 daily runs, sorted."""
    return [origin_of(run=run) for run in run_dates(subset="all")]


def origins_of_subset(*, subset: str) -> list[datetime]:
    """Return the `power_fcst_init_time` of one run subset (`all`, `every5d`, or `gap`)."""
    return [origin_of(run=run) for run in run_dates(subset=subset)]


def demand_frame_path(*, run: datetime) -> Path:
    """Return the demand frame of a run: the stored one for a run of `fold_a_every5d_v1`.

    Args:
        run: The run's 00:00 UTC initialisation time.

    Returns:
        The file holding the run's 51-member rows for the 22 demand series.
    """
    name = f"init={run:%Y-%m-%d}.parquet"
    if run in set(run_dates(subset="every5d")):
        return VALIDATION_DIR / name
    return FRAMES_DEMAND_DIR / name


def feature_columns() -> list[str]:
    """Return stage 1a's input columns from the frozen sweep set, calendar included."""
    columns: list[str] = json.loads(FEATURE_SET_PATH.read_text())["columns"]
    return columns


def series_table() -> pl.DataFrame:
    """Return the 20 substation series with label, group, capacity, coordinates, and H3 cell.

    The coordinates never leave this process.

    Returns:
        One row per series with `time_series_id`, `site` (D1 to D22), `group`,
        `substation_type` (Primary, BSP, or GSP), `units` (MW or MVA), `latitude`,
        `longitude`, `h3_res_5`, and `effective_capacity_mw`.

    Raises:
        ValueError: If the table does not hold 20 series or a series is not a substation.
    """
    table = pl.read_parquet(SERIES_PATH)
    metadata = pl.read_parquet(
        FOLD_A_FRAMES_DIR / "metadata.parquet",
        columns=["time_series_id", "substation_type", "h3_res_5"],
    )
    units = pl.read_parquet(NGED_METADATA_PATH, columns=["time_series_id", "units"])
    kept = (
        table.filter(~pl.col("site").is_in(DROPPED_SITES))
        .join(metadata, on="time_series_id", how="left", validate="1:1")
        .join(units, on="time_series_id", how="left", validate="1:1")
        .with_columns(pl.col("units").cast(pl.String))
        .with_columns(pl.col("substation_type").cast(pl.String))
    )
    if kept.height != 20 or not set(kept["substation_type"].to_list()) <= SUBSTATION_TYPES:
        msg = "expected the 20 Primary, BSP, and GSP series"
        raise ValueError(msg)
    return kept.sort(pl.col("site").str.slice(1).cast(pl.Int32))


def grid_times() -> pl.Series:
    """Return the dense half-hourly grid every per-series array is indexed on."""
    return pl.datetime_range(
        GRID_START, GRID_END, interval="30m", time_unit="us", time_zone="UTC", eager=True
    )


def grid_index(*, times: datetime | pl.Series) -> int | np.ndarray:
    """Return the grid index of one timestamp, or of a series of timestamps.

    Args:
        times: A UTC timestamp, or a Polars series of them.

    Returns:
        The index, or an integer array of indices.
    """
    if isinstance(times, datetime):
        return int((times - GRID_START) / STEP)
    return ((times.dt.epoch("us") - int(GRID_START.timestamp() * 1e6)) // 1_800_000_000).to_numpy()


def take(*, values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Return `values[index]`, NaN where the index falls off either end of the grid.

    Args:
        values: A float array on the grid.
        index: Integer indices.

    Returns:
        A float32 array the shape of `index`.
    """
    inside = (index >= 0) & (index < values.size)
    out = np.full(index.shape, np.nan, dtype=np.float32)
    out[inside] = values[index[inside]]
    return out


def holiday_gaps() -> pl.DataFrame:
    """Return, for each calendar date, the signed days to the nearest holiday and to Christmas.

    Returns:
        A frame with `date`, `days_to_nearest_holiday` (positive before the holiday, negative after
        it, clipped at 7 days either way, from the `holidays` package's England calendar), and
        `days_to_christmas` (the same sign, to 25 December, missing beyond 14 days either way).
    """
    calendar = holidays.country_holidays("GB", subdiv="ENG", years=HOLIDAY_YEARS)
    holiday_dates = np.array(sorted(calendar), dtype="datetime64[D]")
    days = pl.date_range(
        date(HOLIDAY_YEARS.start, 1, 1),
        date(HOLIDAY_YEARS.stop - 1, 12, 31),
        eager=True,
    )
    gaps = (holiday_dates[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    nearest = gaps[np.arange(days.len()), np.abs(gaps).argmin(axis=1)]
    christmas = np.array(
        [np.datetime64(f"{year}-12-25") for year in HOLIDAY_YEARS], dtype="datetime64[D]"
    )
    to_christmas = (
        (christmas[None, :] - days.to_numpy()[:, None]).astype("timedelta64[D]").astype(int)
    )
    nearest_christmas = to_christmas[np.arange(days.len()), np.abs(to_christmas).argmin(axis=1)]
    return pl.DataFrame(
        {
            "date": days,
            "days_to_nearest_holiday": np.clip(nearest, -HOLIDAY_CLIP_DAYS, HOLIDAY_CLIP_DAYS),
            "days_to_christmas": np.where(
                np.abs(nearest_christmas) <= CHRISTMAS_CLIP_DAYS, nearest_christmas, np.nan
            ),
        }
    )


def derived_columns(*, valid_time: pl.Series, latitude: float, longitude: float) -> pl.DataFrame:
    """Return the timestamp-and-location columns of stage 1a for period-ending UTC stamps.

    The sun position is at the midpoint of the half-hour ending at each stamp. The local hour,
    weekday, and holiday distance use the stamp in Europe/London time.

    Args:
        valid_time: Period-ending UTC stamps.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        One row per stamp with the `DERIVED` columns, float32 (the holiday distance in whole days).
    """
    midpoint = valid_time.dt.offset_by("-15m")
    local = valid_time.dt.convert_time_zone("Europe/London")
    frame = pl.DataFrame(
        {
            "solar_zenith": zenith(stamps=midpoint, latitude=latitude, longitude=longitude),
            "solar_azimuth": azimuth(stamps=midpoint, latitude=latitude, longitude=longitude),
            "hour_local": local.dt.hour().cast(pl.Float64)
            + local.dt.minute().cast(pl.Float64) / 60.0,
            "day_of_week_local": local.dt.weekday().cast(pl.Float64),
            "covid_indicator": (
                (valid_time >= LOCKDOWN_START) & (valid_time < LOCKDOWN_END + timedelta(days=1))
            ).cast(pl.Float64),
            "date": local.dt.date(),
        }
    )
    return (
        frame.join(holiday_gaps(), on="date", how="left", validate="m:1", maintain_order="left")
        .select(DERIVED)
        .cast(dict.fromkeys(DERIVED, pl.Float32))
    )


def with_derived(*, frame: pl.DataFrame, latitude: float, longitude: float) -> pl.DataFrame:
    """Add the `DERIVED` columns to a frame with a `valid_time` column, replacing any it has.

    Args:
        frame: Rows with a UTC `valid_time`.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        The frame with the `DERIVED` columns, computed once per distinct valid time.
    """
    times = frame["valid_time"].unique().sort()
    table = derived_columns(valid_time=times, latitude=latitude, longitude=longitude).with_columns(
        valid_time=times
    )
    return frame.drop(DERIVED, strict=False).join(
        table, on="valid_time", how="left", validate="m:1", maintain_order="left"
    )


def load_series_arrays(*, series: pl.DataFrame) -> dict[str, SeriesArrays]:
    """Return each series' ERA5 features, calendar, and observed power on the dense grid.

    Args:
        series: `series_table()`.

    Returns:
        Each site label's `SeriesArrays`.
    """
    columns = feature_columns()
    weather = [c for c in columns if c not in DERIVED]
    grid = pl.DataFrame({"valid_time": grid_times()})
    era5 = pl.read_parquet(ERA5_PATH)
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    out: dict[str, SeriesArrays] = {}
    for row in series.iter_rows(named=True):
        joined = (
            grid.join(
                era5.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                    "valid_time", *weather
                ),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .join(
                observed.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                    "valid_time", pl.col("power") / row["effective_capacity_mw"]
                ),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .sort("valid_time")
        )
        derived = with_derived(
            frame=joined.select("valid_time"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        matrix = (
            joined.select(weather)
            .with_columns(derived.select(DERIVED))
            .select(columns)
            .to_numpy()
            .astype(np.float32)
        )
        out[row["site"]] = SeriesArrays(
            features=matrix,
            power=joined["power"].to_numpy().astype(np.float32),
            valid_time=joined["valid_time"],
        )
    return out


def pinball_quantile_score(
    *,
    quantiles: np.ndarray,
    actual: np.ndarray,
    levels: np.ndarray = LEVELS,
    masses: np.ndarray = ATOM_MASSES,
) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the quantile levels, per case.

    Args:
        quantiles: Shape (cases, len(levels)).
        actual: Shape (cases,).
        levels: The quantile levels; the 11 levels unless the reproduction check passes the nine.
        masses: The probability mass of each level, matching `levels`.

    Returns:
        The quantile score per case, shape (cases,).
    """
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, levels * difference, (levels - 1) * difference)
    return 2.0 * (pinball * masses).sum(axis=1)


def origin_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the three lags measured from the forecast origin, using only usable observations.

    The newest usable observation is at grid index `origin - NEWEST_AGE`, where the age covers
    the telemetry delay and the half-hour a value spans. No lag index is later than that
    observation, which the function asserts for every case (the lag-leakage check of the brief).

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 3) in `LAG_COLUMNS` order: the newest usable value, the
        value at the latest earlier time with the valid time's half-hour of day, and the value at
        the latest earlier time with its weekday and half-hour.

    Raises:
        ValueError: If a case's valid time is not after its origin, or a lag is younger than the
            newest usable observation.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - NEWEST_AGE
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(int)
    if (day > newest).any() or (week > newest).any():
        msg = "a lag index is younger than the newest usable observation"
        raise ValueError(msg)
    return np.stack(
        [
            take(values=power, index=newest),
            take(values=power, index=day),
            take(values=power, index=week),
        ],
        axis=1,
    )


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool the ensemble members' quantiles into the mixture's 11 quantiles (experiment 00019).

    Each member contributes its 11 quantiles as atoms of mass `ATOM_MASSES / members`; the
    mixture's level-`p` quantile is the smallest atom value whose cumulative mass reaches `p`.

    Args:
        member_quantiles: Shape (cases, members, 11).

    Returns:
        Shape (cases, 11), float32, sorted along the levels.
    """
    cases, members, _levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * len(LEVELS))
    weights = np.tile(ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def historical_thresholds(*, power: np.ndarray) -> dict[str, float]:
    """Return a series' `historical_p99` and `historical_p95`, as fractions of capacity.

    Args:
        power: Observed power over capacity on the dense grid, NaN where unobserved.

    Returns:
        The two quantiles of the finite values before the holdout.
    """
    values = power[: int(grid_index(times=LAST_TRAINING_VALID)) + 1]
    values = values[np.isfinite(values)]
    return {"p99": float(np.quantile(values, 0.99)), "p95": float(np.quantile(values, 0.95))}


def atomic_write(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file through a temporary name so an interrupted write leaves no file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)


def log_gpu_seconds(*, job: str, seconds: float, workers: int) -> None:
    """Append one fit job's wall-clock seconds to the GPU-hour log.

    The line is written with one `write` call on a file opened for appending, so concurrent
    workers do not interleave their lines.

    Args:
        job: A name for the job: the member, the refit, and the series or pool.
        seconds: Wall-clock seconds of the job's fits.
        workers: Number of worker processes sharing the GPU while the job ran.
    """
    GPU_HOURS_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "job": job,
            "seconds": round(seconds, 2),
            "workers": workers,
            "logged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
    )
    with GPU_HOURS_LOG.open("a") as handle:
        handle.write(line + "\n")


def gpu_hours_total(*, log: Path = GPU_HOURS_LOG) -> float:
    """Return the running total of GPU hours: the hours during which at least one job was running.

    Each log line holds a job's end time (`logged_at`) and its wall-clock seconds, so its interval
    is known. Jobs that ran side by side on the GPU are counted once, as the design's table counts
    experiment 48's weekly fits.

    Args:
        log: The log file; a missing file counts as zero hours.

    Returns:
        Hours in the union of the jobs' intervals.
    """
    if not log.exists():
        return 0.0
    intervals = []
    for line in log.read_text().splitlines():
        if not line:
            continue
        entry = json.loads(line)
        end = datetime.fromisoformat(entry["logged_at"]).timestamp()
        intervals.append((end - entry["seconds"], end))
    total = 0.0
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    total = sum(end - start for start, end in merged)
    return total / 3600.0


def load_power_arrays(*, series: pl.DataFrame) -> dict[str, SeriesArrays]:
    """Return each series' observed power over capacity on the dense grid, with no weather columns.

    The 14-day correction and the scoring read only `power`, so this skips the ERA5 features that
    `load_series_arrays` joins. `features` is an empty array.

    Args:
        series: `series_table()`.

    Returns:
        Each site label's `SeriesArrays`, NaN where unobserved.
    """
    grid = grid_times()
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    out: dict[str, SeriesArrays] = {}
    for row in series.iter_rows(named=True):
        own = observed.filter(pl.col("time_series_id") == row["time_series_id"])
        power = np.full(grid.len(), np.nan, dtype=np.float32)
        index = np.asarray(grid_index(times=own["valid_time"]))
        inside = (index >= 0) & (index < power.size)
        power[index[inside]] = (
            own["power"].to_numpy()[inside] / row["effective_capacity_mw"]
        ).astype(np.float32)
        out[row["site"]] = SeriesArrays(
            features=np.empty((0, 0), dtype=np.float32), power=power, valid_time=grid
        )
    return out
