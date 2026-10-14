"""Paths, series, arms, grids, and the recent-origin inputs of experiment 00005.

The experiment asks where recent observed power should enter a two-stage forecast of substation
demand (`docs/ml-experiments/00005-lag-placement-stage1b-vs-stage2-demand/result.md`). Everything
runs on one dense half-hourly grid per series, so a lag is an index offset and a forecast origin is
an index. The newest observation a forecast initialised at `T` may use is the half-hour before `T`.
"""

import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays  # ty: ignore[unresolved-import]  # `uv run --with holidays`
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 5
SLUG: Final[str] = "lag-placement-stage1b-vs-stage2-demand"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

SMOKE: Final[bool] = os.environ.get("E00005_SMOKE") == "1"
"""A smoke test uses two series and a few runs, and writes under `E00005_WORK_DIR`."""

WORK_DIR: Final[Path] = (
    Path(os.environ["E00005_WORK_DIR"])
    if SMOKE
    else ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
)
"""Intermediate tables and boosters; private and never committed."""

DEMAND_DIR: Final[Path] = study_dir_for(study="stage1_pilot") / "demand"
"""Experiment 00003's inputs, read only."""
SERIES_PATH: Final[Path] = DEMAND_DIR / "inputs" / "series.parquet"
ERA5_PATH: Final[Path] = DEMAND_DIR / "inputs" / "era5_features.parquet"
OBSERVED_PATH: Final[Path] = DEMAND_DIR / "inputs" / "observed.parquet"
VALIDATION_DIR: Final[Path] = DEMAND_DIR / "ens_every5d" / "val"
ENS_TRAIN_PATH: Final[Path] = DEMAND_DIR / "ens_every5d" / "train.parquet"
FEATURE_SET_PATH: Final[Path] = WORK_DIR / "feature_set.json"

DROPPED_SITES: Final[frozenset[str]] = frozenset({"D7", "D20"})
"""The battery and the biofuel generator, whose substation type is EHV Customer."""
SUBSTATION_TYPES: Final[frozenset[str]] = frozenset({"Primary", "BSP", "GSP"})

GRID_START: Final[datetime] = datetime(2019, 9, 1, tzinfo=UTC)
GRID_END: Final[datetime] = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
STEP: Final[timedelta] = timedelta(minutes=30)
STEPS_PER_DAY: Final[int] = 48
VALIDATION_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
FINAL_TRAIN_CUTOFF: Final[datetime] = datetime(2025, 6, 17, tzinfo=UTC)
"""Stage 1a's final fit uses valid times before this: 14 days before the first validation run."""
LAST_TRAINING_VALID: Final[datetime] = datetime(2025, 6, 30, 23, 30, tzinfo=UTC)
PURGE: Final[timedelta] = timedelta(days=15)
BLOCKS: Final[int] = 6
CHAIN_STARTS: Final[tuple[datetime, ...]] = tuple(
    datetime(year, month, 1, tzinfo=UTC)
    for year, month in ((2024, 4), (2024, 7), (2024, 10), (2025, 1), (2025, 4))
)
"""Quarter starts of the forward chain; each quarter's models end `PURGE` before it starts."""
CHAIN_END: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
CALIBRATION_STRIDE_DAYS: Final[int] = 3
FIRST_CALIBRATION_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
LEADS_PER_TARGET: Final[int] = 4
MAX_LEAD_STEPS: Final[int] = 702
"""351 hours: the longest lead of a run."""

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
"""The recent-origin inputs shared by stage 1b and the stage 2 calibrator."""

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
    "nthread": 4,
}
CALIBRATOR_ROUNDS: Final[int] = 300


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


@dataclass(frozen=True)
class Config:
    """One training configuration of the ERA5-trained stage 1a.

    Attributes:
        start: First ERA5 valid time the models train on.
        drop_calendar: Remove the calendar columns from the feature set.
        stage1b: Whether stage 1b is built on top of stage 1a.
        calibrators: Whether the stage 2 calibrators and their forward chain are built.
    """

    start: datetime
    drop_calendar: bool
    stage1b: bool
    calibrators: bool


CONFIGS: Final[dict[str, Config]] = {
    "from2023": Config(
        start=datetime(2023, 9, 1, tzinfo=UTC), drop_calendar=False, stage1b=True, calibrators=True
    ),
    "full": Config(start=GRID_START, drop_calendar=False, stage1b=True, calibrators=True),
    "from2024": Config(
        start=datetime(2024, 4, 1, tzinfo=UTC),
        drop_calendar=False,
        stage1b=False,
        calibrators=False,
    ),
    "from2023_nocal": Config(
        start=datetime(2023, 9, 1, tzinfo=UTC), drop_calendar=True, stage1b=True, calibrators=False
    ),
}
"""`from2023` is the primary window; `from2023_nocal` is the no-calendar control (R0 and L2)."""


def feature_columns(*, config: str) -> list[str]:
    """Return stage 1a's input columns for a configuration, from the sweep's frozen set.

    Args:
        config: A key of `CONFIGS`.

    Returns:
        The column names.

    Raises:
        FileNotFoundError: If the sweep's feature set has not been written to `FEATURE_SET_PATH`.
    """
    if not FEATURE_SET_PATH.exists():
        msg = f"{FEATURE_SET_PATH} is missing: the input-feature sweep has not chosen the columns"
        raise FileNotFoundError(msg)
    columns: list[str] = json.loads(FEATURE_SET_PATH.read_text())["columns"]
    if CONFIGS[config].drop_calendar:
        columns = [c for c in columns if c not in NO_CALENDAR_DROPS]
    return columns


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
    if SMOKE:
        return kept.head(2).sort(pl.col("site").str.slice(1).cast(pl.Int32))
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


def recent_origin_inputs(
    *, power: np.ndarray, residual: np.ndarray, target: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    """Return the recent-origin inputs for each case, using only observations before the origin.

    The newest observation a forecast initialised at grid index `origin` may use is at
    `origin - 1`. A lag is null where it is no longer than the lead, which is `target - origin`.

    Args:
        power: Observed power over effective capacity on the grid, NaN where unobserved.
        residual: Observed power minus stage 1a at the observed ERA5 weather, on the grid.
        target: Grid index of each case's valid time.
        origin: Grid index of each case's initialisation time.

    Returns:
        A float32 array of shape (cases, len(RECENT_COLUMNS)), columns in `RECENT_COLUMNS` order.

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


def holiday_gaps() -> pl.DataFrame:
    """Return, for each calendar date, the signed days to the nearest holiday and to Christmas.

    Returns:
        A frame with `date`, `days_to_nearest_holiday` (positive before the holiday, negative after
        it, clipped at 7 days either way, from the `holidays` package's England calendar), and
        `days_to_christmas` (the same sign, to 25 December, clipped at 14 days either way).
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
            "days_to_christmas": np.clip(
                nearest_christmas, -CHRISTMAS_CLIP_DAYS, CHRISTMAS_CLIP_DAYS
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


def model_path(*, config: str, seed: int, kind: str, site: str, tag: str) -> Path:
    """Return where one fitted booster is stored.

    Args:
        config: A key of `CONFIGS`.
        seed: The fitting seed.
        kind: `stage1a` or `stage1b`.
        site: The series' label, such as `D3`.
        tag: `final`, or `q0` to `q4` for the forward-chain quarter.

    Returns:
        The path of the `.ubj` file.
    """
    return WORK_DIR / "models" / config / f"seed{seed}" / f"{kind}_{site}_{tag}.ubj"


def load_series_arrays(*, series: pl.DataFrame, config: str) -> dict[str, SeriesArrays]:
    """Return each series' ERA5 features, calendar, and observed power on the dense grid.

    Args:
        series: `series_table()`.
        config: A key of `CONFIGS`, which fixes the feature columns.

    Returns:
        Each site label's `SeriesArrays`.
    """
    columns = feature_columns(config=config)
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


def era5_pred_path(*, config: str, seed: int) -> Path:
    """Return the file of one configuration and seed's stage 1a predictions at ERA5 weather."""
    return WORK_DIR / "era5_pred" / f"{config}_seed{seed}.npy"


def write_era5_pred(*, config: str, seed: int, predictions: dict[str, np.ndarray]) -> None:
    """Write each series' stage 1a ERA5 predictions on the grid, in series-table order.

    Args:
        config: A key of `CONFIGS`.
        seed: The fitting seed.
        predictions: Each site label's array on the grid, keyed in series-table order.
    """
    path = era5_pred_path(config=config, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix(".tmp.npy"), np.stack(list(predictions.values())))
    path.with_suffix(".tmp.npy").rename(path)


def read_era5_pred(*, config: str, seed: int, sites: list[str]) -> dict[str, np.ndarray]:
    """Read what `write_era5_pred` wrote.

    Args:
        config: A key of `CONFIGS`.
        seed: The fitting seed.
        sites: The site labels, in the order they were written.

    Returns:
        Each site label's array on the grid.
    """
    return dict(zip(sites, np.load(era5_pred_path(config=config, seed=seed)), strict=True))
