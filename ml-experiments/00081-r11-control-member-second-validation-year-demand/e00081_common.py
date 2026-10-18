"""Paths, folds, series, lags, corrections, and metrics shared by the scripts of experiment 00081.

The experiment repeats experiment 00048's retraining comparison and experiment 00049's
rolling-window drop-one contrasts on the ENS control member alone, for forecast runs from 2022-07 to
2024-03 (`docs/ml-experiments/00081-r11-control-member-second-validation-year-demand/result.md`).
The helpers are copied from experiment 00048's `e00048_common.py`; an experiment may not import
another experiment.
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import holidays
import numpy as np
import polars as pl
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 81
SLUG: Final[str] = "r11-control-member-second-validation-year-demand"
SEED: Final[int] = 0

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
FRAMES_DIR: Final[Path] = WORK_DIR / "frames"
"""One `init=<date>.parquet` demand frame per daily run, control member only."""
OBSERVED_PATH: Final[Path] = WORK_DIR / "observed.parquet"
PREDS_DIR: Final[Path] = WORK_DIR / "anchor_preds"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
METRICS_DIR: Final[Path] = WORK_DIR / "metrics"
RESULTS_DIR: Final[Path] = WORK_DIR / "results"
CHART_DIR: Final[Path] = WORK_DIR / "charts"

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
"""The stage 1 pilot's demand inputs, read only."""
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
FOLD_A_METADATA_PATH: Final[Path] = (
    study_dir_for(study="quick_wins_fixed_pipeline_1156")
    / "frames"
    / "combined"
    / "fold_a"
    / "metadata.parquet"
)
"""The metadata table that holds each series' H3 cell; read only."""
FEATURE_SET_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00005-lag-placement-stage1b-vs-stage2-demand"
    / "work"
    / "feature_set.json"
)
"""Experiment 00006's final Primary column list, as experiment 00005 froze it; read only."""
CLEANED_POWER_DELTA: Final[Path] = REPO_DATA_DIR / "NGED" / "cleaned_power_time_series.delta"

PRIMARY_GROUP: Final[str] = "primary demand"
SERIES_WITHOUT_BACKFILL_POWER: Final[frozenset[int]] = frozenset({7, 13, 12})
"""`time_series_id`s whose first valid reading falls after the backfill period starts."""
EXPECTED_SERIES: Final[int] = 13

PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
"""`power_fcst_init_time` is the run's initialisation time plus this."""
BACKFILL_FIRST_RUN: Final[date] = date(2021, 3, 21)
BACKFILL_LAST_RUN: Final[date] = date(2024, 3, 31)
LAG_WARMUP: Final[timedelta] = timedelta(hours=336)
"""Every series needs power from before the first run plus this, so the longest lag is defined."""
GRID_START: Final[datetime] = datetime(2021, 3, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2024, 4, 30, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
HORIZON_HOURS: Final[float] = 351.0
"""The last lead, in hours after `power_fcst_init_time`, that a frame holds."""

LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = (
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
"""Probability mass of each level: the interval between the mid-points to its neighbours."""
MEDIAN: Final[int] = 4
"""Position of the median (q50) among the nine levels."""
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
"""Columns computed from the timestamp and the location, not read from a weather file."""
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
"""N1's parameters (experiment 00003's, with the 9-level quantile loss); callers set `nthread`."""
N_ROUNDS: Final[int] = 500
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
SHORT_HALF_LIFE_DAYS: Final[float] = 0.5 * 365.25
SPREAD_FACTOR: Final[float] = 1.15
"""The calibrator spread factor of experiment 00018, applied to every model arm."""
CORRECTION_DAYS: Final[int] = 14
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
LAG_HOURS: Final[dict[str, float]] = {
    "power_lag_24h": 24.0,
    "power_lag_48h": 48.0,
    "power_lag_168h": 168.0,
    "power_lag_336h": 336.0,
}
"""Each stored lag column and its lag; a lag is null where it is no longer than the lead."""

FoldNameType = Literal["b", "c", "d"]


@dataclass(frozen=True)
class Fold:
    """A set of daily forecast runs scored together.

    Attributes:
        name: `b` (2023-07 to 2024-03), `c` (2022-07 to 2023-03), or `d` (the descriptive 9 km
            check, April to September 2023).
        first_run: The first 00:00 UTC run date.
        last_run: The last run date.
        row_set_id: The prediction row set the fold's runs form.
        expected_runs: How many runs exist between the two dates; asserted when the runs are listed.
    """

    name: FoldNameType
    first_run: date
    last_run: date
    row_set_id: str
    expected_runs: int

    @property
    def anchor(self) -> datetime:
        """The frozen model's anchor: the last half-hour before the fold's first run."""
        start = datetime(self.first_run.year, self.first_run.month, self.first_run.day, tzinfo=UTC)
        return start - STEP


FOLD_NAMES: Final[tuple[FoldNameType, ...]] = ("b", "c", "d")


def fold_names(*, selected: list[str]) -> list[FoldNameType]:
    """Return the fold names among `selected`, in the order `b`, `c`, `d`."""
    return [name for name in FOLD_NAMES if name in selected]


FOLDS: Final[dict[FoldNameType, Fold]] = {
    "b": Fold("b", date(2023, 7, 1), date(2024, 3, 31), "control_fold_b_daily_v1", 274),
    "c": Fold("c", date(2022, 7, 1), date(2023, 3, 31), "control_fold_c_daily_v1", 273),
    "d": Fold("d", date(2023, 4, 1), date(2023, 9, 30), "control_nine_km_check_daily_v1", 183),
}
"""Fold `d` is the descriptive 9 km check: a frozen model anchored at 2023-03-31 scored on April to
September 2023, and it decides nothing."""


def _midnight(*, day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def runs_between(*, first: date, last: date, folder: Path) -> list[datetime]:
    """Return the 00:00 UTC runs between two dates that have a backfill file, sorted.

    Args:
        first: The first run date.
        last: The last run date.
        folder: The folder of the backfill's `<date>.npz` files.

    Returns:
        Each run's initialisation time; a date with no file is skipped.
    """
    days = (last - first).days + 1
    return [
        _midnight(day=first + timedelta(days=i))
        for i in range(days)
        if (folder / f"{first + timedelta(days=i):%Y-%m-%d}.npz").exists()
    ]


def fold_runs(*, fold: Fold, folder: Path) -> list[datetime]:
    """Return a fold's runs, asserting the count the plan states."""
    runs = runs_between(first=fold.first_run, last=fold.last_run, folder=folder)
    if len(runs) != fold.expected_runs:
        msg = f"fold {fold.name}: expected {fold.expected_runs} runs, found {len(runs)}"
        raise ValueError(msg)
    return runs


def origin_of(*, run: datetime) -> datetime:
    """Return a run's `power_fcst_init_time`: its initialisation time plus the publication delay."""
    return run + PUBLICATION_DELAY


def frame_path(*, run: datetime, folder: Path = FRAMES_DIR) -> Path:
    """Return the demand frame of a run."""
    return folder / f"init={run:%Y-%m-%d}.parquet"


def feature_columns() -> list[str]:
    """Return the input columns from the frozen sweep set, calendar included."""
    columns: list[str] = json.loads(FEATURE_SET_PATH.read_text())["columns"]
    return columns


def series_table() -> pl.DataFrame:
    """Return the 13 Primary substations that have cleaned power throughout the backfill period.

    The coordinates never leave this process.

    Returns:
        One row per series with `time_series_id`, `site` (D1 to D22), `latitude`, `longitude`,
        `effective_capacity_mw`, and `h3_res_5`.

    Raises:
        ValueError: If the table does not hold 13 series.
    """
    cells = pl.read_parquet(FOLD_A_METADATA_PATH, columns=["time_series_id", "h3_res_5"])
    table = (
        pl.read_parquet(SERIES_PATH)
        .filter(
            (pl.col("group") == PRIMARY_GROUP)
            & ~pl.col("time_series_id").is_in(sorted(SERIES_WITHOUT_BACKFILL_POWER))
        )
        .join(cells, on="time_series_id", how="left", validate="1:1")
        .drop("group")
    )
    if table.height != EXPECTED_SERIES or table["h3_res_5"].null_count():
        msg = f"expected {EXPECTED_SERIES} Primary series with an H3 cell, got {table.height}"
        raise ValueError(msg)
    return table.sort(pl.col("site").str.slice(1).cast(pl.Int32))


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


def load_power_arrays(*, series: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return each series' observed power over effective capacity on the dense grid.

    Args:
        series: `series_table()`.

    Returns:
        Each site label's float32 array, NaN where unobserved.
    """
    grid = pl.DataFrame({"valid_time": grid_times()})
    observed = pl.read_parquet(OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"])
    out: dict[str, np.ndarray] = {}
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
    """Return the timestamp-and-location columns for period-ending UTC stamps.

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


def pinball_quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the 9 levels, per case.

    Args:
        quantiles: Shape (cases, 9).
        actual: Shape (cases,).

    Returns:
        The quantile score per case, shape (cases,).
    """
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def origin_lags(*, power: np.ndarray, target: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the three lags measured from the forecast origin, using only usable observations.

    The newest usable observation is at grid index `origin - NEWEST_AGE`, where the age covers
    the telemetry delay and the half-hour a value spans.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 3) in `LAG_COLUMNS` order: the newest usable value, the
        value at the latest earlier time with the valid time's half-hour of day, and the value at
        the latest earlier time with its weekday and half-hour.

    Raises:
        ValueError: If a case's valid time is not after its origin.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - NEWEST_AGE
    gap = target - newest
    day = target - STEPS_PER_DAY * np.ceil(gap / STEPS_PER_DAY).astype(int)
    week = target - 7 * STEPS_PER_DAY * np.ceil(gap / (7 * STEPS_PER_DAY)).astype(int)
    return np.stack(
        [
            take(values=power, index=newest),
            take(values=power, index=day),
            take(values=power, index=week),
        ],
        axis=1,
    )


def corrections_for(*, frame: pl.DataFrame, power: np.ndarray, window_days: int) -> pl.DataFrame:
    """Return each run's error correction for one series, with its history count.

    The correction is the mean of (observed minus median forecast) over the `window_days` days
    ending at the newest usable observation. Each past half-hour takes the forecast as issued:
    the median forecast of the latest run that began at or before it, from the refit that served
    that run. A run is read only if the frame holds it, so with a frame that starts at the fold's
    first run no correction reads a forecast older than that run.

    Args:
        frame: One series' quantiles, one row per run and valid time, with every daily run that the
            correction may read.
        power: The series' observed power over capacity on the dense grid.
        window_days: Length of the correction window in days.

    Returns:
        One row per run: `power_fcst_init_time`, `correction`, and `history_count`.

    Raises:
        ValueError: If a correction reads a forecast or observation that is not older than the
            forecast origin.
    """
    window = window_days * STEPS_PER_DAY
    minimum = MIN_CORRECTION_DAYS * STEPS_PER_DAY
    runs = (
        frame.select("power_fcst_init_time")
        .unique()
        .sort("power_fcst_init_time")["power_fcst_init_time"]
    )
    origins = np.asarray(grid_index(times=runs))
    valid_index = np.asarray(grid_index(times=frame["valid_time"]))
    run_index = np.asarray(grid_index(times=frame["power_fcst_init_time"]))
    next_origin = np.append(origins[1:], np.iinfo(np.int64).max)[
        np.searchsorted(origins, run_index)
    ]
    keep = (valid_index >= run_index) & (valid_index < next_origin)
    median = np.full(power.shape, np.nan, dtype=np.float64)
    source = np.full(power.shape, -1, dtype=np.int64)
    median[valid_index[keep]] = frame[LEVEL_COLUMNS[MEDIAN]].to_numpy()[keep]
    source[valid_index[keep]] = run_index[keep]
    rows = []
    for origin_time, origin in zip(runs, origins, strict=True):
        newest = origin - NEWEST_AGE
        span = np.arange(newest - window + 1, newest + 1)
        used = ~np.isnan(median[span]) & ~np.isnan(power[span])
        if span.max() >= origin or (source[span][used] >= origin).any():
            msg = "a past forecast or observation is not older than the forecast origin"
            raise ValueError(msg)
        count = int(used.sum())
        residual = power[span][used].astype(np.float64) - median[span][used]
        rows.append(
            {
                "power_fcst_init_time": origin_time,
                "correction": float(residual.mean()) if count >= minimum else 0.0,
                "history_count": count,
            }
        )
    return pl.DataFrame(rows)


def with_corrections(
    *, base: pl.DataFrame, series: pl.DataFrame, power_of: dict[str, np.ndarray], window_days: int
) -> pl.DataFrame:
    """Add a `correction` column to a fit's quantiles, one correction per series and run."""
    sites = dict(zip(series["time_series_id"], series["site"], strict=True))
    pieces = []
    for (series_id,), part in base.group_by("time_series_id", maintain_order=True):
        table = corrections_for(
            frame=part, power=power_of[sites[series_id]], window_days=window_days
        )
        pieces.append(
            part.join(
                table.select("power_fcst_init_time", "correction"),
                on="power_fcst_init_time",
                how="left",
                validate="m:1",
            )
        )
    return pl.concat(pieces)


def stack(*, frame: pl.DataFrame, correct: bool, spread: bool) -> pl.DataFrame:
    """Apply the correction and the spread factor to quantiles that carry a `correction` column."""
    quantiles = frame.select(LEVEL_COLUMNS).to_numpy().astype(np.float64)
    if correct:
        quantiles = quantiles + frame["correction"].to_numpy()[:, None]
    if spread:
        median = quantiles[:, [MEDIAN]]
        quantiles = median + SPREAD_FACTOR * (quantiles - median)
    return frame.select(*KEYS, "seed").with_columns(
        pl.DataFrame(
            np.sort(quantiles, axis=1).astype(np.float32),
            schema=list(LEVEL_COLUMNS),
            orient="row",
        )
    )


def case_metrics(*, quantiles: np.ndarray, actual: np.ndarray) -> dict[str, np.ndarray]:
    """Return every per-case metric, in points of capacity where a score and 0/1 otherwise.

    Args:
        quantiles: Shape (cases, 9), sorted.
        actual: Observed power over effective capacity, shape (cases,).

    Returns:
        `qs` (the quantile score), `mae` (absolute error of the median), `ex95` (the observation
        exceeds the 0.95 quantile), and `cov90` (the observation lies inside `q05` to `q95`).
    """
    return {
        "qs": pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0,
        "mae": np.abs(quantiles[:, MEDIAN] - actual) * 100.0,
        "ex95": (actual > quantiles[:, 8]).astype(np.float64),
        "cov90": ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 8])).astype(np.float64),
    }


def atomic_write(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file through a temporary name so an interrupted write leaves no file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
