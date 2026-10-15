"""Paths, series, grids, lags, and the hierarchy of experiment 00036.

The experiment adds forecasts of the series beneath each bulk supply point (BSP) and grid supply
point (GSP) to N1 (`docs/ml-experiments/00036-hierarchy-aware-bsp-gsp-forecasts/result.md`).
Everything runs on one dense half-hourly grid per series, so a lag is an index offset and a forecast
origin is an index. The grid, lag, and N1 helpers are copied from experiment 00013's
`e00013_common.py`.
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
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

EXPERIMENT_ID: Final[int] = 36
SLUG: Final[str] = "hierarchy-aware-bsp-gsp-forecasts"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables and boosters; private and never committed."""

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
    "nthread": 8,
}
"""N1's parameters: 00003's, with the 9-level quantile loss and this experiment's 8 cores."""
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
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


HIERARCHY_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR.parent / "NGED" / "substation_hierarchy_2026-10-15"
)
"""NGED's private hierarchy files; never committed."""
BLOCKS: Final[tuple[tuple[datetime, datetime], ...]] = tuple(
    (
        datetime(2024 + (3 + 3 * i) // 12, (3 + 3 * i) % 12 + 1, 1, tzinfo=UTC),
        datetime(2024 + (6 + 3 * i) // 12, (6 + 3 * i) % 12 + 1, 1, tzinfo=UTC),
    )
    for i in range(5)
)
"""The five cross-fitting blocks by valid time: 2024-04 to 2024-06, then each following quarter."""
PURGE: Final[timedelta] = timedelta(days=15)
"""Training rows within this long of a held-out block are dropped from a cross-fit."""
WEATHER: Final[tuple[str, ...]] = (
    "shortwave",
    "longwave",
    "temperature_2m",
    "dew_point_temperature_2m",
    "pressure_surface",
    "precipitation",
    "wind_speed_10m",
)
"""The forecast weather columns an ERA5-trained N1 reads from ERA5 at training times."""
MEMBERS: Final[int] = 51
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
MEDIAN_INDEX: Final[int] = 4


@dataclass(frozen=True)
class Hierarchy:
    """The forecast children of each BSP, and the BSP each GSP sits above.

    Attributes:
        primaries: Each BSP's `time_series_id` mapped to its Primary substations' ids.
        generators: Each BSP's `time_series_id` mapped to its modelled metered generators' ids.
        gsp_to_bsp: Each GSP's `time_series_id` mapped to the BSP beneath it.
    """

    primaries: dict[int, list[int]]
    generators: dict[int, list[int]]
    gsp_to_bsp: dict[int, int]


def load_hierarchy(*, series: pl.DataFrame, generator_ids: set[int]) -> Hierarchy:
    """Read NGED's hierarchy and keep the children that have a forecast.

    Args:
        series: `series_table()`.
        generator_ids: The ids of the 9 solar and wind series that have a best-guess forecast.

    Returns:
        The hierarchy.

    Raises:
        ValueError: If the Primary substations kept are not the 16 of `series`, or a BSP or GSP is
            not found.
    """
    monitor = pl.read_csv(HIERARCHY_DIR / "Monitor Direction.csv").select(
        "TimeSeriesID", "SubstationNumber", "SubstationType"
    )
    names = pl.read_csv(HIERARCHY_DIR / "Substations.csv").select(
        "substation_number", "bsp_substation_number", "gsp_name"
    )
    joined = monitor.join(
        names, left_on="SubstationNumber", right_on="substation_number", how="inner"
    )
    forecast_primaries = set(
        series.filter(pl.col("substation_type") == "Primary")["time_series_id"].to_list()
    )
    bsp_ids = series.filter(pl.col("substation_type") == "BSP")["time_series_id"].to_list()
    gsp_ids = series.filter(pl.col("substation_type") == "GSP")["time_series_id"].to_list()
    bsp_number = {
        int(r["TimeSeriesID"]): r["SubstationNumber"]
        for r in joined.filter(pl.col("SubstationType") == "BSP").iter_rows(named=True)
    }
    primaries, generators = {}, {}
    for bsp in bsp_ids:
        beneath = joined.filter(pl.col("bsp_substation_number") == bsp_number[bsp])
        primaries[bsp] = sorted(
            int(i)
            for i in beneath.filter(pl.col("SubstationType") == "Primary")["TimeSeriesID"]
            if int(i) in forecast_primaries
        )
        generators[bsp] = sorted(
            int(i) for i in beneath["TimeSeriesID"].to_list() if int(i) in generator_ids
        )
    if sorted(i for ids in primaries.values() for i in ids) != sorted(forecast_primaries):
        msg = "the Primary substations beneath the BSPs are not the 16 of the series table"
        raise ValueError(msg)
    gsp_name = {
        int(r["TimeSeriesID"]): r["gsp_name"]
        for r in joined.filter(pl.col("SubstationType") == "GSP").iter_rows(named=True)
    }
    bsp_gsp = {
        int(r["TimeSeriesID"]): r["gsp_name"]
        for r in joined.filter(pl.col("SubstationType") == "BSP").iter_rows(named=True)
    }
    gsp_to_bsp = {}
    for gsp in gsp_ids:
        match = [b for b in bsp_ids if bsp_gsp[b] == gsp_name[gsp]]
        if len(match) != 1:
            msg = f"GSP {gsp} does not have exactly one BSP beneath it"
            raise ValueError(msg)
        gsp_to_bsp[gsp] = match[0]
    return Hierarchy(primaries=primaries, generators=generators, gsp_to_bsp=gsp_to_bsp)


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's 9 quantiles, as experiment 00013 does.

    Args:
        member_quantiles: Shape (cases, members, 9), each member's quantiles at `LEVELS`.

    Returns:
        Shape (cases, 9): the weighted quantiles of the pooled values, each member weighted equally
        and each level by its atom mass, sorted along the levels so none cross.
    """
    cases, members, levels = member_quantiles.shape
    values = member_quantiles.reshape(cases, members * levels)
    weights = np.tile(ATOM_MASSES, members) / members
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    cumulative = np.cumsum(weights[order], axis=1)
    picks = (cumulative[:, :, None] >= (LEVELS - 1e-9)).argmax(axis=1)
    pooled = np.take_along_axis(sorted_values, picks, axis=1)
    return np.sort(pooled, axis=1).astype(np.float32)


def block_of(*, valid_time: pl.Series) -> np.ndarray:
    """Return each valid time's cross-fitting block index (0 to 4), or -1 outside all blocks."""
    out = np.full(valid_time.len(), -1, dtype=np.int8)
    for index, (start, end) in enumerate(BLOCKS):
        out[((valid_time >= start) & (valid_time < end)).to_numpy()] = index
    return out


def outside_block(*, valid_time: pl.Series, block: int) -> np.ndarray:
    """Return which rows lie outside the held-out block and its purge margin (training rows)."""
    start, end = BLOCKS[block]
    return ((valid_time < start - PURGE) | (valid_time >= end + PURGE)).to_numpy()
