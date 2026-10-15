"""Paths, series, grids, and the lag inputs of experiment 00024.

The experiment compares groupings of substations that share one XGBoost model
(`docs/ml-experiments/00024-pooling-groupings-demand-xgboost/result.md`). Everything runs on
one dense half-hourly grid per series, so a lag is an index offset and a forecast origin is an
index. The helpers are copied from experiment 00019's `e00019_common.py`.
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 32
SLUG: Final[str] = "substation-fingerprints-pooled-demand"
EXPERIMENT_ID_00013: Final[int] = 13
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and boosters; private and never committed."""
ARMS_DIR_00024: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00024-pooling-groupings-demand-xgboost" / "work" / "arms"
)
"""Experiment 00024's arm tables (the by-type pools with and without the identifier), read only."""
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
NGED_METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
"""The metadata table, read only for each series' metered units."""
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
    "nthread": 10,
}
"""N1's parameters: 00003's, with the 9-level quantile loss and slot B's 10 cores."""
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
        `substation_type` (Primary, BSP, or GSP), `units` (MW or MVA), `latitude`,
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


TRAIN_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""First valid time of the training window the fingerprints and the models share."""
RECENT_DAYS: Final[int] = 14
RECENT_GAP: Final[int] = 2 * STEPS_PER_DAY
"""The recent window ends this many half-hours before the forecast origin (origin minus 2 days)."""
RECENT_MIN_COUNT: Final[int] = 3 * STEPS_PER_DAY
DAY_SLOTS: Final[slice] = slice(20, 32)
NIGHT_SLOTS: Final[slice] = slice(0, 10)
"""Half-hour-of-day slots (period-ending UTC stamps): 10:00 to 15:30 and 00:00 to 04:30."""
GRID_START_WEEKDAY: Final[int] = 6
"""2019-09-01, the grid start, is a Sunday (Monday is 0)."""
STATIC_COLUMNS: Final[tuple[str, ...]] = ("latitude", "longitude")
TRAIN_STAT_COLUMNS: Final[tuple[str, ...]] = (
    "fp_mean",
    "fp_std",
    "fp_p05",
    "fp_p95",
    "fp_day_minus_night",
    "fp_weekend_minus_weekday",
    "fp_peak_slot",
)
RECENT_COLUMNS: Final[tuple[str, ...]] = (
    "fp_recent_mean",
    "fp_recent_std",
    "fp_recent_day_minus_night",
)
KIND_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "S": STATIC_COLUMNS,
    "T": TRAIN_STAT_COLUMNS,
    "R": RECENT_COLUMNS,
}
"""The fingerprint blocks: static, training window, and recent window."""
WARM_ARMS: Final[dict[str, tuple[bool, str]]] = {
    "type_noid": (False, ""),
    "type_id": (True, ""),
    "id_static": (True, "S"),
    "id_train": (True, "T"),
    "id_recent": (True, "R"),
    "id_train_recent": (True, "TR"),
    "train_noid": (False, "T"),
}
"""Each warm arm's use of the series identifier and its fingerprint blocks."""
NEW_WARM_ARMS: Final[tuple[str, ...]] = tuple(
    arm for arm in WARM_ARMS if arm not in ("type_noid", "type_id")
)
"""Arms this experiment fits; `type_noid` and `type_id` are copied from experiment 00024."""
COLD_ARMS: Final[dict[str, str]] = {
    "cold_none": "",
    "cold_static": "S",
    "cold_train": "T",
    "cold_train_recent": "TR",
}
"""Each cold-start arm's fingerprint blocks; none uses the identifier."""
COLD_FINGERPRINT_DAYS: Final[int] = 28
"""A held-out series' training-window statistics use only this many days of telemetry."""
HELD_OUT: Final[tuple[str, ...]] = ("D5", "D10", "D14", "D16", "D2", "D4", "D12", "D21")
"""Every second Primary-MW and Primary-MVA series in site order, fixed before any fit."""


def training_fingerprint(*, power: np.ndarray, days: int | None = None) -> np.ndarray:
    """Return the training-window statistics of one series, in `TRAIN_STAT_COLUMNS` order.

    Uses only the grid indices from `TRAIN_START` (or, with `days`, the last `days` days) to
    `LAST_TRAINING_VALID`.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        days: Use only this many days ending at `LAST_TRAINING_VALID`; the whole window if None.

    Returns:
        A float32 vector of the 7 statistics.
    """
    start = int(grid_index(times=TRAIN_START))
    stop = int(grid_index(times=LAST_TRAINING_VALID)) + 1
    if days is not None:
        start = stop - days * STEPS_PER_DAY
    values = power[start:stop].astype(np.float64)
    index = np.arange(start, stop)
    slot = index % STEPS_PER_DAY
    weekend = (GRID_START_WEEKDAY + index // STEPS_PER_DAY) % 7 >= 5
    good = np.isfinite(values)
    day = good & (slot >= DAY_SLOTS.start) & (slot < DAY_SLOTS.stop)
    night = good & (slot >= NIGHT_SLOTS.start) & (slot < NIGHT_SLOTS.stop)
    profile = np.array(
        [
            np.nanmean(values[good & (slot == k)]) if (good & (slot == k)).any() else np.nan
            for k in range(STEPS_PER_DAY)
        ]
    )
    return np.array(
        [
            values[good].mean(),
            values[good].std(),
            np.percentile(values[good], 5),
            np.percentile(values[good], 95),
            values[day].mean() - values[night].mean(),
            values[good & weekend].mean() - values[good & ~weekend].mean(),
            float(np.nanargmax(profile)),
        ],
        dtype=np.float32,
    )


def recent_fingerprint(*, power: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the recent-window statistics before each forecast origin.

    The window covers the 14 days ending `RECENT_GAP` half-hours (2 days) before the origin, so no
    observation within 2 days of the origin is used.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, 3) in `RECENT_COLUMNS` order, NaN where the window holds
        fewer than `RECENT_MIN_COUNT` observations.
    """
    values = np.nan_to_num(power.astype(np.float64))
    good = np.isfinite(power).astype(np.float64)
    slot = np.arange(power.size) % STEPS_PER_DAY
    day = ((slot >= DAY_SLOTS.start) & (slot < DAY_SLOTS.stop)).astype(np.float64)
    night = ((slot >= NIGHT_SLOTS.start) & (slot < NIGHT_SLOTS.stop)).astype(np.float64)

    def prefix(x: np.ndarray) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(x)])

    total, count, squares = prefix(values), prefix(good), prefix(values**2)
    day_total, day_count = prefix(values * day), prefix(good * day)
    night_total, night_count = prefix(values * night), prefix(good * night)
    high = np.clip(origin - RECENT_GAP, 0, power.size)
    low = np.clip(high - RECENT_DAYS * STEPS_PER_DAY, 0, power.size)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (total[high] - total[low]) / n
        variance = (squares[high] - squares[low]) / n - mean**2
        day_mean = (day_total[high] - day_total[low]) / (day_count[high] - day_count[low])
        night_mean = (night_total[high] - night_total[low]) / (night_count[high] - night_count[low])
    out = np.stack([mean, np.sqrt(np.maximum(variance, 0.0)), day_mean - night_mean], axis=1)
    out[n < RECENT_MIN_COUNT] = np.nan
    return out.astype(np.float32)


def fingerprint_matrix(
    *, kinds: str, latitude: float, longitude: float, training: np.ndarray, recent: np.ndarray
) -> np.ndarray:
    """Return the fingerprint columns of the requested blocks for the cases of one series.

    Args:
        kinds: Blocks to include, any of `S`, `T`, and `R`, in that order.
        latitude: Degrees north.
        longitude: Degrees east.
        training: The series' `training_fingerprint`.
        recent: The cases' `recent_fingerprint`, shape (cases, 3).

    Returns:
        A float32 array of shape (cases, columns); an empty width when `kinds` is empty.
    """
    cases = recent.shape[0]
    parts = []
    if "S" in kinds:
        parts.append(np.tile(np.array([latitude, longitude], dtype=np.float32), (cases, 1)))
    if "T" in kinds:
        parts.append(np.tile(training, (cases, 1)))
    if "R" in kinds:
        parts.append(recent)
    return np.hstack(parts).astype(np.float32) if parts else np.empty((cases, 0), np.float32)


def group_labels(*, series: pl.DataFrame, grouping: str) -> dict[str, str]:
    """Return the by-type model group of each site: BSP, GSP, Primary-MW, or Primary-MVA.

    Args:
        series: `series_table()`.
        grouping: Must be `"type"`.

    Returns:
        Each site label's group name; sites with the same name share one model.

    Raises:
        ValueError: If the grouping is not `"type"`.
    """
    if grouping != "type":
        msg = f"unknown grouping {grouping!r}"
        raise ValueError(msg)
    return {
        row["site"]: row["substation_type"]
        if row["substation_type"] != "Primary"
        else f"Primary-{row['units']}"
        for row in series.iter_rows(named=True)
    }
