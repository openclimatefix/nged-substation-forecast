"""Paths, series, grids, lags, corrections, and scores shared by the scripts of experiment 00050.

The experiment asks whether a second weather forecast, NOAA GEFS, forecasts the solar anomaly better
than ECMWF ENS (`docs/ml-experiments/00050-second-weather-forecast-for-solar-anomaly/result.md`).
The helpers are copied from experiment 00046's `e00046_common.py` at commit f96f6486 of branch
`pvdis-3-embedded-solar-error`; an experiment may not import another experiment. The scripts read
experiment 00046's files under its `work/` folder and write only under their own.
"""

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final

import holidays
import numpy as np
import polars as pl
import pvlib
from studies.pv_physics import PlantParameters, SunAndSky, plant_power_mw, solar_position
from studies.solar import azimuth, zenith
from studies.sources import ML_EXPERIMENTS_DATA_DIR, product_dir_for, study_dir_for

EXPERIMENT_ID: Final[int] = 50
SLUG: Final[str] = "second-weather-forecast-for-solar-anomaly"
SOURCE_EXPERIMENT_ID: Final[int] = 46
SOURCE_SLUG: Final[str] = "embedded-solar-share-of-primary-forecast-error"
ROW_SET_ID: Final[str] = "fold_a_every5d_v1"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)

WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
"""Intermediate tables; private and never committed."""
SOURCE_WORK_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / f"{SOURCE_EXPERIMENT_ID:05d}-{SOURCE_SLUG}" / "work"
)
"""Experiment 00046's `work/` folder, read only."""
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
    "nthread": 1,
}
"""N1's parameters: 00003's, with the 9-level quantile loss and one thread per worker."""
RECENCY_HALF_LIFE_DAYS: Final[float] = 2 * 365.25
SHORT_HALF_LIFE_DAYS: Final[float] = 0.5 * 365.25
"""N1H's half-life, from experiment 00017."""
SPREAD_FACTOR: Final[float] = 1.15
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


SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six Primaries metered in MW that have an experiment 00040 capacity for 2023/24."""
VERDICT_SITES: Final[tuple[str, ...]] = ("D9", "D10", "D11", "D14", "D16")
"""The series whose rows decide the verdict; D5 (Kirton) is exploratory only."""
CAPACITY_SEASON_YEAR: Final[int] = 2023
"""Experiment 00040's `year` column is the start of a season-year, so 2023 means 2023/24."""
CAPACITY_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00040-pv-capacity-error-bars" / "work" / "real_scores.parquet"
)
FRAME_00039_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00039-pv-disaggregation-cams-injection" / "work" / "frame.parquet"
)
SOLAR_DIR: Final[Path] = WORK_DIR / "solar"
SOURCE_SOLAR_DIR: Final[Path] = SOURCE_WORK_DIR / "solar"
"""Experiment 00046's CAMS and ENS solar tables, read only."""
MEMBER_DIR: Final[Path] = WORK_DIR / "member_quantiles"
RAW_ARMS_DIR: Final[Path] = WORK_DIR / "arms_raw"
ARMS_DIR: Final[Path] = WORK_DIR / "arms"
SOURCE_ARMS_DIR: Final[Path] = SOURCE_WORK_DIR / "arms"
BOOSTER_DIR: Final[Path] = SOURCE_WORK_DIR / "boosters"
"""Experiment 00046's fitted `d1l` boosters, read only."""
BAND_COEFFICIENTS_PATH: Final[Path] = SOURCE_WORK_DIR / "results" / "band_coefficients.parquet"
RESULTS_DIR: Final[Path] = WORK_DIR / "results"
SCORED_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
"""Local (Europe/London) calendar months of the scored rows."""
DAYLIGHT_THRESHOLD: Final[float] = 0.02
"""A month-by-half-hour cell is daylight when the training-window mean of `u` exceeds this."""
MIN_CLIMATOLOGY_DAYS: Final[int] = 30
MIN_RELIABILITY_U: Final[float] = 0.9
MEMBERS: Final[int] = 51


def capacities() -> pl.DataFrame:
    """Return experiment 00040's 2023/24 capacity estimates for the six sites.

    Returns:
        One row per site in `SITES` with `site`, `C` (MW, the no-confound capacity used by the
        verdict arms), `headline` (MW, the sensitivity arms only), `range_low`, and `range_high`
        (the Kirton library range, MW).

    Raises:
        ValueError: If a site has no 2023/24 row.
    """
    table = pl.read_parquet(CAPACITY_PATH).filter(
        (pl.col("year") == CAPACITY_SEASON_YEAR) & pl.col("site").is_in(SITES)
    )
    if set(table["site"].to_list()) != set(SITES):
        msg = "experiment 00040 has no 2023/24 capacity row for every site"
        raise ValueError(msg)
    return table.select("site", "C", "headline", "range_low", "range_high").sort("site")


def lead_band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return the index into `LEAD_BANDS` of each lead time.

    Args:
        lead_hours: Hours from the forecast origin to the valid time.

    Returns:
        An integer array of band indices.
    """
    edges = np.array([low for _, low, _ in LEAD_BANDS[1:]])
    return np.searchsorted(edges, lead_hours, side="right")


def mixture_quantiles(*, member_quantiles: np.ndarray) -> np.ndarray:
    """Pool members' quantiles into the mixture's nine quantiles, as experiment 00019 does.

    Args:
        member_quantiles: Shape (cases, members, 9).

    Returns:
        Shape (cases, 9), float32, sorted along the levels.
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


def write_atomically(*, frame: pl.DataFrame, target: Path) -> None:
    """Write a parquet file through a temporary name so a crash leaves no half-written file.

    Args:
        frame: The table.
        target: The final path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)


FLEET: Final[PlantParameters] = PlantParameters(
    tilt_deg=35.0, azimuth_deg=180.0, dc_capacity_mw=1.15, ac_capacity_mw=1.0
)
"""The generic fleet whose output per MW of AC capacity is the unit PV shape `u`.

Copied from experiment 00039's `e00039_common.py`; the CAMS `u` of its frame uses the same fleet.
"""


def latest_time(*, times: pl.Series) -> datetime:
    """Return the latest timestamp of a datetime series.

    Args:
        times: A non-empty datetime series.

    Returns:
        Its maximum.
    """
    latest = times.max()
    assert isinstance(latest, datetime)
    return latest


def with_cell(*, frame: pl.DataFrame, time_column: str) -> pl.DataFrame:
    """Add the climatology cell of each row: the local calendar month and the UTC half-hour of day.

    Args:
        frame: Rows with a UTC datetime column.
        time_column: The name of that column (a period-ending stamp).

    Returns:
        The frame with `month` (Int8, Europe/London calendar month) and `hh` (Int8, 0 to 47, the
        UTC half-hour of day of the stamp).
    """
    stamp = pl.col(time_column)
    return frame.with_columns(
        month=stamp.dt.convert_time_zone("Europe/London").dt.month().cast(pl.Int8),
        hh=(stamp.dt.hour() * 2 + stamp.dt.minute() // 30).cast(pl.Int8),
    )


SUNNY_INDEX: Final[float] = 0.7
CLOUDY_INDEX: Final[float] = 0.3
"""Clear-sky index (CAMS over clear-sky irradiance) bounds of sunny and cloudy rows."""


def scored_rows(*, series: pl.DataFrame, arrays: dict[str, SeriesArrays]) -> pl.DataFrame:
    """Return every validation case of the six sites with the facts that decide if it is scored.

    A case is scored when its local month is April to September, its month-by-half-hour cell is
    daylight (CAMS climatology above `DAYLIGHT_THRESHOLD`, training window only), CAMS `u` is
    present, and the observed power is finite.

    Args:
        series: `series_table()` filtered to `SITES`.
        arrays: `load_series_arrays` for those series.

    Returns:
        One row per run, valid time, and series with `KEYS`, `site`, `actual`, `u_cams`,
        `anomaly_cams`, `clear_sky_index` (null where the clear-sky irradiance is zero), `month`,
        `hh`, `lead_hours`, `band` (index into `LEAD_BANDS`), the boolean `scored`, and the boolean
        `scored_all_months` (the same without the April-to-September restriction).
    """
    keys = pl.concat(
        [
            pl.read_parquet(path, columns=list(KEYS))
            .filter(pl.col("time_series_id").is_in(series["time_series_id"].implode()))
            .unique()
            for path in sorted(VALIDATION_DIR.glob("init=*.parquet"))
        ]
    ).join(series.select("time_series_id", "site"), on="time_series_id", validate="m:1")
    actual = pl.concat(
        [
            keys.filter(pl.col("site") == site).with_columns(
                actual=pl.Series(
                    take(
                        values=arrays[site].power,
                        index=np.asarray(
                            grid_index(times=keys.filter(pl.col("site") == site)["valid_time"])
                        ),
                    )
                )
            )
            for site in series["site"]
        ]
    )
    cams = (
        pl.read_parquet(FRAME_00039_PATH)
        .filter((pl.col("kind") == "substation") & pl.col("site").is_in(list(SITES)))
        .select(
            "site",
            valid_time=pl.col("time"),
            clear_sky_index=pl.when(pl.col("ghi_u_clear") > 0)
            .then(pl.col("ghi_u") / pl.col("ghi_u_clear"))
            .otherwise(None),
        )
    )
    anomaly = pl.read_parquet(SOURCE_SOLAR_DIR / "cams_anomaly.parquet")
    u_cams = pl.read_parquet(SOURCE_SOLAR_DIR / "cams_u.parquet").rename(
        {"time": "valid_time", "u": "u_cams"}
    )
    daylight = pl.read_parquet(SOURCE_SOLAR_DIR / "climatology.parquet").select(
        "site", "month", "hh", "daylight"
    )
    lead = (
        (actual["valid_time"] - actual["power_fcst_init_time"]).dt.total_minutes() / 60.0
    ).to_numpy()
    return (
        with_cell(frame=actual, time_column="valid_time")
        .with_columns(lead_hours=pl.Series(lead), band=pl.Series(lead_band(lead_hours=lead)))
        .join(u_cams, on=["site", "valid_time"], how="left", validate="m:1")
        .join(anomaly, on=["site", "valid_time"], how="left", validate="m:1")
        .join(cams, on=["site", "valid_time"], how="left", validate="m:1")
        .join(daylight, on=["site", "month", "hh"], how="left", validate="m:1")
        .with_columns(
            scored_all_months=pl.col("daylight").fill_null(False)
            & pl.col("u_cams").is_not_null()
            & pl.col("actual").is_finite()
            & (pl.col("valid_time") < HOLDOUT)
        )
        .with_columns(
            scored=pl.col("scored_all_months") & pl.col("month").is_in(list(SCORED_MONTHS))
        )
        .drop("daylight")
    )


GEFS_DIR: Final[Path] = product_dir_for(product="GEFS")
GEFS_PATH: Final[Path] = GEFS_DIR / "GEFS.parquet"
GEFS_GRID_PATH: Final[Path] = GEFS_DIR / "_grid_cells.parquet"
"""Maps the archive's `lat_index` and `lon_index` to coordinates; private, never published."""
GEFS_SHORTWAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
GEFS_MEMBERS: Final[int] = 31
GEFS_GRID_DEGREES: Final[float] = 0.25
GEFS_STORE: Final[str] = "noaa-gefs-forecast-35-day"
FIRST_TRAINING_RUN: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
HORIZON_HOURS: Final[int] = 351
"""Hours from the forecast origin to the last valid time kept, as experiment 00046 kept for ENS."""
ENS_NATIVE_STEP_CHANGE_HOURS: Final[int] = 144
"""ENS radiation windows are 3 hours to this lead and 6 hours after it."""
ENS_LAST_LEAD_HOURS: Final[int] = 360
"""The last lead of ENS's radiation schedule; `_upsample_nwp_to_half_hourly` rejects later ones."""
GEFS_STEP_CHANGE_HOURS: Final[int] = 240
"""GEFS lead times are 3-hourly to this lead and 6-hourly after it."""
MIN_PLAUSIBLE_MEAN_WM2: Final[float] = -1.0
"""A de-accumulated 3-hour mean below this is a fault; smaller negatives are quantisation."""
SOLAR_DIR_TRAIN: Final[Path] = SOLAR_DIR / "gefs_train"
SOLAR_DIR_VAL: Final[Path] = SOLAR_DIR / "gefs_val"
LATENCY_PATH: Final[Path] = RESULTS_DIR / "latency_decision.json"
SELECTION_PATH: Final[Path] = RESULTS_DIR / "selection.json"
CV_FIRST_MONTH: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
CV_FOLDS: Final[int] = 15
"""One fold per training-window valid month, April 2024 to June 2025."""
SELECTION_MARGIN: Final[float] = 0.01
"""The blend or GEFS replaces ENS only if its cross-validated error is this share lower."""
SELECTION_MAX_LEAD_HOURS: Final[float] = 240.0
MAX_ZENITH_DEG: Final[float] = 75.0
ARMS_ANOMALY: Final[tuple[str, ...]] = ("ens", "gefs", "blend")


def gefs_run_offset_days() -> int:
    """Return how many days before the ENS run the GEFS run used for it was initialised.

    Returns:
        0 when the same-day 00 UTC GEFS run is reliably present by the forecast origin (09:00 UTC),
        1 when the previous day's run is used.

    Raises:
        FileNotFoundError: If `e00050_latency.py` has not recorded the decision yet (check 5).
    """
    if not LATENCY_PATH.exists():
        msg = f"{LATENCY_PATH} is missing; run e00050_latency.py before building GEFS inputs"
        raise FileNotFoundError(msg)
    offset = int(json.loads(LATENCY_PATH.read_text())["run_offset_days"])
    if offset not in (0, 1):
        msg = "run_offset_days must be 0 or 1"
        raise ValueError(msg)
    return offset


def u_from_half_hour_ghi(
    *, stamps: pl.Series, ghi: np.ndarray, latitude: float, longitude: float
) -> np.ndarray:
    """Return the unit PV shape of half-hour mean irradiance, the chain CAMS goes through.

    The sun is at the midpoint of the half-hour ending at each stamp, the Erbs correlation splits
    the irradiance, and the generic fleet turns it into power per MW of AC capacity. Copied from
    experiment 00046's `e00046_solar_inputs.py`.

    Args:
        stamps: Distinct period-ending UTC stamps, shape (n,).
        ghi: Half-hour mean global horizontal irradiance in W/m2, shape (k, n); NaN where unknown.
        latitude: Degrees north.
        longitude: Degrees east.

    Returns:
        `u`, shape (k, n), NaN where `ghi` is NaN.
    """
    copies = ghi.shape[0]
    midpoints = stamps.dt.offset_by("-15m")
    sun_zenith, sun_azimuth = solar_position(
        stamps=midpoints, latitude=latitude, longitude=longitude
    )
    day = midpoints.dt.ordinal_day().to_numpy()
    unknown = ~np.isfinite(ghi)
    flat = np.clip(np.where(unknown, 0.0, ghi), 0.0, None).ravel().astype(np.float64)
    split = pvlib.irradiance.erbs(
        ghi=flat, zenith=np.tile(sun_zenith, copies), datetime_or_doy=np.tile(day, copies)
    )
    sky = SunAndSky(
        half_hour_end_time=np.tile(stamps.to_numpy(), copies),
        ghi_w_m2=flat,
        dni_w_m2=np.nan_to_num(np.asarray(split["dni"]).astype(np.float64)),
        dhi_w_m2=np.nan_to_num(np.asarray(split["dhi"]).astype(np.float64)),
        zenith_deg=np.tile(sun_zenith, copies),
        sun_azimuth_deg=np.tile(sun_azimuth, copies),
    )
    shape = plant_power_mw(sky=sky, parameters=FLEET).reshape(copies, -1)
    return np.where(unknown, np.nan, shape)


def climatology(*, frame: pl.DataFrame, value: str, time_column: str, pool: bool) -> pl.DataFrame:
    """Return the month-by-half-hour mean of a column over training-window valid times.

    Copied from experiment 00046's `e00046_solar_inputs.py`.

    Args:
        frame: Columns `site`, a UTC time column, and the value column.
        value: The value column.
        time_column: The time column.
        pool: Pool each local month with its two neighbours when true.

    Returns:
        Columns `site`, `month`, `hh`, `clim` (the mean), and `days` (distinct UTC dates behind it).

    Raises:
        ValueError: If a row is later than the last training valid time.
    """
    training = frame.filter(pl.col(value).is_not_null() & pl.col(value).is_not_nan())
    if latest_time(times=training[time_column]) > LAST_TRAINING_VALID:
        msg = "climatology rows must be training-window valid times only"
        raise ValueError(msg)
    cells = with_cell(frame=training, time_column=time_column).with_columns(
        date=pl.col(time_column).dt.date()
    )
    if pool:
        cells = pl.concat(
            [
                cells.with_columns(month=((pl.col("month") - 1 + shift) % 12 + 1).cast(pl.Int8))
                for shift in (-1, 0, 1)
            ]
        )
    return cells.group_by("site", "month", "hh").agg(
        clim=pl.col(value).mean(), days=pl.col("date").n_unique()
    )


def scalar(*, value: object) -> float:
    """Return a Polars aggregate (a number, or None for an empty column) as a float.

    Args:
        value: The result of `Series.max()`, `Series.min()`, or a similar aggregate.

    Returns:
        The number as a float, 0.0 for None.

    Raises:
        TypeError: If the value is neither None nor a real number.
    """
    if value is None:
        return 0.0
    if not isinstance(value, int | float):
        msg = f"expected a number, found {type(value).__name__}"
        raise TypeError(msg)
    return float(value)


def lead_hours_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the hours from each row's `power_fcst_init_time` to its `valid_time`."""
    return (
        (frame["valid_time"] - frame["power_fcst_init_time"]).dt.total_minutes() / 60.0
    ).to_numpy()
