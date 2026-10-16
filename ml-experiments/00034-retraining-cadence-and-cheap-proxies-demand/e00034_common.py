"""Paths, series, grids, lags, and metrics shared by the scripts of experiment 00034.

The experiment compares retraining cadences and cheap proxies for monthly retraining
(`docs/ml-experiments/00034-retraining-cadence-and-cheap-proxies-demand/result.md`). Demand
helpers are copied from experiment 00019's `e00019_common.py`; an experiment may not import
another experiment.
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

EXPERIMENT_ID: Final[int] = 34
SLUG: Final[str] = "retraining-cadence-and-cheap-proxies-demand"
EXPERIMENT_ID_00013: Final[int] = 13
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and boosters; private and never committed."""
ARMS_DIR_00013: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00013-primary-demand-against-stored-baselines" / "work" / "arms"
)
"""Experiment 00013's arm tables (N1 and the stored baselines), read only."""
GENERATOR_ROW_SET_ID: Final[str] = "fold_a_tuesdays_v1"


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

NGED_METADATA_PATH: Final[Path] = REPO_DATA_DIR / "NGED" / "metadata.parquet"
"""The metadata table, read only for each series' metered units."""
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
    ("2-6 d", 48.0, 144.0),
    ("6-10 d", 144.0, 240.0),
    ("10+ d", 240.0, 10_000.0),
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
    "nthread": 12,
}
"""N1's parameters: 00003's, with the 9-level quantile loss and slot A's 12 cores."""
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


KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
MEDIAN: Final[int] = 4
"""Position of the median (q50) among the nine levels."""
NEWEST_AGE: Final[int] = int(TELEMETRY_DELAY / STEP) + 1
"""Half-hours between the origin and the newest usable observation (00013's 3.5 hours)."""
WINDOW: Final[int] = CORRECTION_DAYS * STEPS_PER_DAY
MIN_COUNT: Final[int] = MIN_CORRECTION_DAYS * STEPS_PER_DAY
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
RESULTS_DIR: Final[Path] = WORK_DIR / "results"
GENERATOR_ARMS_DIR: Final[Path] = WORK_DIR / "generator_arms"


def corrections_for(*, frame: pl.DataFrame, power: np.ndarray) -> pl.DataFrame:
    """Return each run's 14-day correction for one series and seed, with its history count.

    The correction is the mean of (observed minus median forecast) over the 14 days ending at the
    newest usable observation, using for each past half-hour the forecast of the latest earlier
    run, as experiment 00019 computed it.

    Args:
        frame: One series' quantiles for one seed, keyed by run and valid time.
        power: The series' observed power over capacity on the dense grid.

    Returns:
        One row per run: `power_fcst_init_time`, `correction`, and `history_count`.

    Raises:
        ValueError: If a past forecast comes from a run that began after its half-hour.
    """
    runs = (
        frame.select("power_fcst_init_time")
        .unique()
        .sort("power_fcst_init_time")["power_fcst_init_time"]
    )
    origins = np.asarray(grid_index(times=runs))
    median = np.full(power.shape, np.nan, dtype=np.float64)
    source = np.full(power.shape, -1, dtype=np.int64)
    for position, origin in enumerate(runs):
        part = frame.filter(pl.col("power_fcst_init_time") == origin)
        index = np.asarray(grid_index(times=part["valid_time"]))
        end = origins[position + 1] if position + 1 < len(origins) else np.inf
        keep = (index >= origins[position]) & (index < end)
        median[index[keep]] = part[LEVEL_COLUMNS[MEDIAN]].to_numpy()[keep]
        source[index[keep]] = origins[position]
    rows = []
    for origin_time, origin in zip(runs, origins, strict=True):
        newest = origin - NEWEST_AGE
        window = np.arange(newest - WINDOW + 1, newest + 1)
        used = ~np.isnan(median[window]) & ~np.isnan(power[window])
        if (
            window.max() >= origin
            or (source[window][used] > window[used]).any()
            or (source[window][used] >= origin).any()
        ):
            msg = "a past forecast or observation is not older than the forecast origin"
            raise ValueError(msg)
        count = int(used.sum())
        residual = power[window][used].astype(np.float64) - median[window][used]
        rows.append(
            {
                "power_fcst_init_time": origin_time,
                "correction": float(residual.mean()) if count >= MIN_COUNT else 0.0,
                "history_count": count,
            }
        )
    return pl.DataFrame(rows)


def with_corrections(
    *, base: pl.DataFrame, series: pl.DataFrame, arrays: dict[str, SeriesArrays]
) -> pl.DataFrame:
    """Add a `correction` column to a fit's quantiles, one correction per series, seed, and run."""
    sites = series.select("time_series_id", "site")
    pieces = []
    for (series_id, _seed), part in base.group_by("time_series_id", "seed", maintain_order=True):
        site = sites.filter(pl.col("time_series_id") == series_id)["site"][0]
        table = corrections_for(frame=part, power=arrays[site].power)
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


def actual_table(*, series: pl.DataFrame, arrays: dict[str, SeriesArrays]) -> pl.DataFrame:
    """Return every demand validation case with the observed power over capacity."""
    pieces = []
    for row in series.iter_rows(named=True):
        keys = pl.concat(
            [
                pl.read_parquet(path, columns=list(KEYS))
                .filter(pl.col("time_series_id") == row["time_series_id"])
                .unique()
                for path in sorted(VALIDATION_DIR.glob("init=*.parquet"))
            ]
        )
        index = np.asarray(grid_index(times=keys["valid_time"]))
        pieces.append(
            keys.with_columns(actual=pl.Series(take(values=arrays[row["site"]].power, index=index)))
        )
    return pl.concat(pieces)


def repeat_seeds(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Repeat a seedless stored arm once per seed."""
    return pl.concat([frame.with_columns(seed=pl.lit(s).cast(pl.Int8)) for s in SEEDS])


def read_seeded(*, folder: Path, name: str) -> pl.DataFrame:
    """Read an arm's three seed files, keyed by series, run, valid time, and seed."""
    paths = [folder / f"{name}_seed{seed}.parquet" for seed in SEEDS]
    return pl.concat(
        [pl.read_parquet(p).select(*KEYS, "seed", *LEVEL_COLUMNS) for p in paths]
    ).with_columns(pl.col("seed").cast(pl.Int8))


HISTORY_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""The threshold's history ends here, before the held-out period."""
THRESHOLD_LEVELS: Final[dict[str, float]] = {"p99": 0.99, "p95": 0.95}


def historical_thresholds(*, power: np.ndarray) -> dict[str, float]:
    """Return a series' `historical_p99` and `historical_p95`, as fractions of capacity.

    Args:
        power: Observed power over capacity on the dense grid, NaN where unobserved.

    Returns:
        The two quantiles of the finite values before `HISTORY_END`.
    """
    end = int(grid_index(times=HISTORY_END))
    values = power[:end]
    values = values[np.isfinite(values)]
    return {name: float(np.quantile(values, level)) for name, level in THRESHOLD_LEVELS.items()}


def exceedance_probability(*, quantiles: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Return P(power > threshold) from nine sorted quantiles, with exponential tails.

    Between `q05` and `q95` the cumulative distribution is the linear interpolation of the level
    against the quantile. Above `q95` the exceedance probability halves for every width
    `q95 - q90` beyond `q95`, from 0.05 (zero if `q95 == q90`). Below `q05` the cumulative
    probability falls to zero by halving for every width `q10 - q05` below `q05`, from 0.05.

    Args:
        quantiles: Shape (cases, 9), sorted along the levels.
        threshold: Shape (cases,).

    Returns:
        Probabilities, shape (cases,).
    """
    cases = quantiles.shape[0]
    count = (quantiles <= threshold[:, None]).sum(axis=1)
    out = np.empty(cases)
    rows = np.arange(cases)
    low = count == 0
    width = quantiles[:, 1] - quantiles[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        below = np.where(
            width > 0,
            0.05 * 2.0 ** (-(quantiles[:, 0] - threshold) / np.where(width > 0, width, 1)),
            0.0,
        )
        out[low] = 1.0 - below[low]
        high = count == len(LEVELS)
        gap = quantiles[:, 8] - quantiles[:, 7]
        above = np.where(
            gap > 0, 0.05 * 0.5 ** ((threshold - quantiles[:, 8]) / np.where(gap > 0, gap, 1)), 0.0
        )
        out[high] = above[high]
        middle = ~low & ~high
        k = np.clip(count - 1, 0, len(LEVELS) - 2)
        lo, hi = quantiles[rows, k], quantiles[rows, k + 1]
        fraction = (threshold - lo) / np.where(hi > lo, hi - lo, 1)
        cdf = LEVELS[k] + (LEVELS[k + 1] - LEVELS[k]) * fraction
        out[middle] = 1.0 - cdf[middle]
    return np.clip(out, 0.0, 1.0)


def case_metrics(
    *, quantiles: np.ndarray, actual: np.ndarray, thresholds: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return every per-case metric, in points of capacity where a score and 0/1 otherwise.

    Args:
        quantiles: Shape (cases, 9), sorted.
        actual: Observed power over capacity, shape (cases,).
        thresholds: `{"p99": ..., "p95": ...}`, each of shape (cases,), the series' thresholds.

    Returns:
        `qs`, `mae` (absolute error of the median), `tw_p99`, `tw_p95`, `brier_p99`, `brier_p95`
        (the squared difference between the exceedance probability and the 0/1 outcome),
        `ex90`, `ex95` (the observation exceeds that quantile), and `cov90` (inside `q05` to
        `q95`).
    """
    out = {
        "qs": pinball_quantile_score(quantiles=quantiles, actual=actual) * 100.0,
        "mae": np.abs(quantiles[:, MEDIAN] - actual) * 100.0,
        "ex90": (actual > quantiles[:, 7]).astype(np.float64),
        "ex95": (actual > quantiles[:, 8]).astype(np.float64),
        "cov90": ((actual >= quantiles[:, 0]) & (actual <= quantiles[:, 8])).astype(np.float64),
    }
    for name, threshold in thresholds.items():
        out[f"tw_{name}"] = (
            pinball_quantile_score(
                quantiles=np.maximum(quantiles, threshold[:, None]),
                actual=np.maximum(actual, threshold),
            )
            * 100.0
        )
        probability = exceedance_probability(quantiles=quantiles, threshold=threshold)
        out[f"brier_{name}"] = (probability - (actual > threshold)) ** 2
        out[f"hit_{name}"] = ((probability >= 0.5) & (actual > threshold)).astype(np.float64)
        out[f"alarm_{name}"] = (probability >= 0.5).astype(np.float64)
        out[f"false_alarm_{name}"] = ((probability >= 0.5) & ~(actual > threshold)).astype(
            np.float64
        )
        out[f"event_{name}"] = (actual > threshold).astype(np.float64)
    return out
