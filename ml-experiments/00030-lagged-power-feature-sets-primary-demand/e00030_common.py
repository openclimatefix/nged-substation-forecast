"""Paths, series, grids, and the lag inputs of experiment 00030.

The experiment compares sets of lagged power columns in N1 for Primary substation demand
(`docs/ml-experiments/00030-lagged-power-feature-sets-primary-demand/result.md`). Everything runs on
one dense half-hourly grid per series, so a lag is an index offset and a forecast origin is an
index. The helpers are copied from experiment 00019's `e00019_common.py`.
"""

import json
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 30
SLUG: Final[str] = "lagged-power-feature-sets-primary-demand"
EXPERIMENT_ID_00013: Final[int] = 13
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and boosters; private and never committed."""
ARMS_DIR_00019: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00019-primary-demand-stack-vs-manual-heuristic" / "work" / "arms"
)
"""Experiment 00019's N1H fits, read only."""
ARMS_DIR_00013: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00013-primary-demand-against-stored-baselines" / "work" / "arms"
)
"""Experiment 00013's arm tables (N1 and the stored baselines), read only."""

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
"""Experiment 00003's inputs, read only."""
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
VALIDATION_DIR: Final[Path] = DEMAND_DIR / "ens_every5d" / "val"
ENS_TRAIN_PATH: Final[Path] = DEMAND_DIR / "ens_every5d" / "train.parquet"
FEATURE_SET_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00005-lag-placement-stage1b-vs-stage2-demand"
    / "work"
    / "feature_set.json"
)
"""Experiment 00006's final Primary column list, as experiment 00005 froze it; read only."""
PREDICTIONS_00005_ID: Final[int] = 5

DROPPED_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
"""The battery and the biofuel generator, whose substation type is EHV Customer."""
SUBSTATION_TYPES: Final[frozenset[str]] = frozenset({"Primary", "BSP", "GSP"})

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
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
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("1-2 d", 24.0, 48.0),
    ("3-6 d", 48.0, 144.0),
    ("7-10 d", 144.0, 240.0),
    ("11-15 d", 240.0, 10_000.0),
)

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
NO_CALENDAR_DROPS: Final[tuple[str, ...]] = (
    "hour_local",
    "day_of_week_local",
    "days_to_nearest_holiday",
    "days_to_christmas",
)
"""The calendar-position columns the no-calendar control removes."""
LOCKDOWN_START: Final[datetime] = datetime(2020, 3, 23, tzinfo=UTC)
LOCKDOWN_END: Final[datetime] = datetime(2021, 7, 19, tzinfo=UTC)
"""The 0/1 lockdown indicator is 1 from the first to the last of these days, inclusive."""
HOLIDAY_YEARS: Final[range] = range(2018, 2029)
HOLIDAY_CLIP_DAYS: Final[int] = 7
CHRISTMAS_CLIP_DAYS: Final[int] = 14
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
    "nthread": 4,
}
N_ROUNDS: Final[int] = 500
QUANTILE_PARAMETERS: Final[dict[str, object]] = {
    **XGBOOST_PARAMETERS,
    "objective": "reg:quantileerror",
    "quantile_alpha": LEVELS,
    "nthread": 4,
}
"""N1's parameters: 00003's, with the 9-level quantile loss and 4 cores a worker."""
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
SHORT_HALF_LIFE_DAYS: Final[float] = 0.5 * 365.25
"""N1H's half-life, from experiment 00017."""
SPREAD_FACTOR: Final[float] = 1.15
"""The calibrator spread factor of experiment 00018."""
CORRECTION_DAYS: Final[int] = 14
MIN_CORRECTION_DAYS: Final[int] = 3
TELEMETRY_DELAY: Final[timedelta] = timedelta(hours=3)
"""A value is usable once it is this old, plus the half-hour it covers (see the plan)."""
PUBLICATION_DELAY: Final[timedelta] = timedelta(hours=9)
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


def feature_columns(*, calendar: bool) -> list[str]:
    """Return stage 1a's input columns from the frozen sweep set, with or without the calendar.

    Args:
        calendar: Keep hour, weekday, and days to Christmas when true.

    Returns:
        The column names.
    """
    columns: list[str] = json.loads(FEATURE_SET_PATH.read_text())["columns"]
    return columns if calendar else [c for c in columns if c not in NO_CALENDAR_DROPS]


def series_table() -> pl.DataFrame:
    """Return the 20 substation series with label, group, capacity, coordinates, and H3 cell.

    The coordinates never leave this process.

    Returns:
        One row per series with `time_series_id`, `site` (D1 to D22), `group`,
        `substation_type` (Primary, BSP, or GSP), `latitude`,
        `longitude`, `h3_res_5`, and `effective_capacity_mw`.

    Raises:
        ValueError: If the table does not hold 20 series or a series is not a substation.
    """
    table = pl.read_parquet(SERIES_PATH)
    metadata = pl.read_parquet(
        study_dir_for(study="quick_wins_fixed_pipeline_1156")
        / "frames"
        / "combined"
        / "fold_a"
        / "metadata.parquet",
        columns=["time_series_id", "substation_type", "h3_res_5"],
    )
    kept = (
        table.filter(~pl.col("site").is_in(DROPPED_SITES))
        .join(metadata, on="time_series_id", how="left", validate="1:1")
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
        frame.join(holiday_gaps(), on="date", how="left", validate="m:1")
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
        table, on="valid_time", how="left", validate="m:1"
    )


def fit_booster(
    *, features: np.ndarray, target: np.ndarray, seed: int, weights: np.ndarray | None = None
) -> xgb.Booster:
    """Fit a squared-error XGBoost model with 00003's hyperparameters.

    Args:
        features: Shape (rows, columns).
        target: Shape (rows,).
        seed: The fitting seed.
        weights: Optional row weights.

    Returns:
        The fitted booster.
    """
    return xgb.train(
        {**XGBOOST_PARAMETERS, "seed": seed},
        xgb.DMatrix(features, label=target, weight=weights),
        num_boost_round=N_ROUNDS,
    )


def predict(*, model: xgb.Booster, features: np.ndarray) -> np.ndarray:
    """Return a booster's predictions for a feature matrix.

    Args:
        model: A fitted booster.
        features: Shape (rows, columns).

    Returns:
        Predictions, shape (rows,) or (rows, levels) for the multi-quantile calibrator.
    """
    return model.predict(xgb.DMatrix(features))


def load_series_arrays(*, series: pl.DataFrame) -> dict[str, SeriesArrays]:
    """Return each series' ERA5 features, calendar, and observed power on the dense grid.

    Args:
        series: `series_table()`.

    Returns:
        Each site label's `SeriesArrays`.
    """
    columns = feature_columns(calendar=True)
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

    The newest usable observation is at grid index `origin - newest_age`, where the age covers
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
    newest = origin - int(TELEMETRY_DELAY / STEP) - 1
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


NEWEST_AGE: Final[int] = int(TELEMETRY_DELAY / STEP) + 1
"""Half-hours between the origin and the newest usable observation (3.5 hours)."""
WEEK: Final[int] = 7 * STEPS_PER_DAY
WEEKLY_WEEKS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
ANNUAL_WEEKS: Final[tuple[int, ...]] = (49, 50, 51, 52, 53, 54, 55)
"""The weeks back of `manual_heuristic`'s 13 lags."""
WEEKLY_COLUMNS: Final[tuple[str, ...]] = tuple(f"lag_{w}w" for w in WEEKLY_WEEKS)
ANNUAL_COLUMNS: Final[tuple[str, ...]] = tuple(f"lag_{w}w" for w in ANNUAL_WEEKS)
SUMMARY_COLUMNS: Final[tuple[str, ...]] = ("weekly_mean", "weekly_median", "annual_mean")
RECENT_COLUMNS: Final[tuple[str, ...]] = ("recent_mean_6h", "recent_minus_1h", "recent_minus_2h")
DAY_TYPE_COUNT: Final[int] = 3
DAY_TYPE_COLUMNS: Final[tuple[str, ...]] = tuple(f"same_day_type_{i}" for i in range(1, 4))
DAY_TYPE_SEARCH_DAYS: Final[int] = 28
ARM_LAG_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "a1": (*LAG_COLUMNS, *WEEKLY_COLUMNS),
    "a2": (*LAG_COLUMNS, *WEEKLY_COLUMNS, *ANNUAL_COLUMNS),
    "a3": (*LAG_COLUMNS, *SUMMARY_COLUMNS),
    "a4": (*LAG_COLUMNS, *WEEKLY_COLUMNS, *ANNUAL_COLUMNS, *RECENT_COLUMNS),
    "a5": (*LAG_COLUMNS, *WEEKLY_COLUMNS, *ANNUAL_COLUMNS, *DAY_TYPE_COLUMNS),
    "a7": (*LAG_COLUMNS, "weekly_mean", "weekly_median"),
}
"""Each arm's lag columns, in matrix order; A0 is 00013's stored N1 fit, A7 is post hoc."""


def day_type_grid() -> np.ndarray:
    """Return 1 on weekend days and England bank holidays, 0 otherwise, for each grid half-hour.

    Returns:
        An int8 array on the dense grid; the day is the Europe/London date of the stamp.
    """
    local = grid_times().dt.convert_time_zone("Europe/London")
    frame = pl.DataFrame({"date": local.dt.date(), "weekday": local.dt.weekday()}).join(
        holiday_gaps().select("date", "days_to_nearest_holiday"),
        on="date",
        how="left",
        maintain_order="left",
    )
    return (
        (frame["weekday"].to_numpy() >= 6) | (frame["days_to_nearest_holiday"].to_numpy() == 0)
    ).astype(np.int8)


def _analogue(
    *, power: np.ndarray, target: np.ndarray, newest: np.ndarray, weeks: int
) -> np.ndarray:
    """Return power `weeks` weeks before each target, NaN when that half-hour is after `newest`."""
    index = target - weeks * WEEK
    return np.where(index <= newest, take(values=power, index=index), np.nan).astype(np.float32)


def lag_block(
    *,
    columns: tuple[str, ...],
    power: np.ndarray,
    day_type: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Return an arm's lag columns, using only observations at or before the newest usable one.

    Args:
        columns: One of `ARM_LAG_COLUMNS`' values.
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        day_type: `day_type_grid()`.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, len(columns)).

    Raises:
        ValueError: If a case's valid time is not after its origin, or a column is unknown.
    """
    if (target <= origin).any():
        msg = "every valid time must be after its initialisation time"
        raise ValueError(msg)
    newest = origin - NEWEST_AGE
    base = origin_lags(power=power, target=target, origin=origin)
    weekly = np.stack(
        [_analogue(power=power, target=target, newest=newest, weeks=w) for w in WEEKLY_WEEKS],
        axis=1,
    )
    annual = np.stack(
        [_analogue(power=power, target=target, newest=newest, weeks=w) for w in ANNUAL_WEEKS],
        axis=1,
    )
    blocks: dict[str, np.ndarray] = {
        **dict(zip(LAG_COLUMNS, base.T, strict=True)),
        **dict(zip(WEEKLY_COLUMNS, weekly.T, strict=True)),
        **dict(zip(ANNUAL_COLUMNS, annual.T, strict=True)),
    }
    if any(c in SUMMARY_COLUMNS for c in columns):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blocks["weekly_mean"] = np.nanmean(weekly, axis=1).astype(np.float32)
            blocks["weekly_median"] = np.nanmedian(weekly, axis=1).astype(np.float32)
            blocks["annual_mean"] = np.nanmean(annual, axis=1).astype(np.float32)
    if any(c in RECENT_COLUMNS for c in columns):
        window = np.stack([take(values=power, index=newest - i) for i in range(12)], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blocks["recent_mean_6h"] = np.nanmean(window, axis=1).astype(np.float32)
        blocks["recent_minus_1h"] = window[:, 2]
        blocks["recent_minus_2h"] = window[:, 4]
    if any(c in DAY_TYPE_COLUMNS for c in columns):
        same = np.full((target.size, DAY_TYPE_COUNT), np.nan, dtype=np.float32)
        found = np.zeros(target.size, dtype=int)
        wanted = day_type[target]
        for days_back in range(1, DAY_TYPE_SEARCH_DAYS + 1):
            candidate = target - days_back * STEPS_PER_DAY
            usable = (candidate <= newest) & (candidate >= 0) & (found < DAY_TYPE_COUNT)
            usable &= day_type[np.clip(candidate, 0, day_type.size - 1)] == wanted
            rows = np.flatnonzero(usable)
            same[rows, found[rows]] = take(values=power, index=candidate[rows])
            found[rows] += 1
        blocks.update(zip(DAY_TYPE_COLUMNS, same.T, strict=True))
    unknown = [c for c in columns if c not in blocks]
    if unknown:
        msg = f"unknown lag columns: {unknown}"
        raise ValueError(msg)
    return np.stack([blocks[c] for c in columns], axis=1).astype(np.float32)
