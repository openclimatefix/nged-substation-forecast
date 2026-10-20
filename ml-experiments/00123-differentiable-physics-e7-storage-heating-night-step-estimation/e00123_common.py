"""Constants and the night table shared by the scripts of experiment 123 (DP E7).

The experiment estimates, at Primary substations, how large the step in demand is between four
baseline half-hours (22:00 to 23:30) and five window half-hours (00:30 to 02:30) of the same winter
night, and how the step grows with cold. Every number uses ERA5 temperature, an estimate of past
weather.

**Timestamp convention.** `PowerTimeSeries.time` is the end of the half-hour (checked in test 1),
so a half-hour *starting* at `s` carries the stamp `s + 30 min`, as experiments 114 and 121 assume.
"""

import os
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Final, Literal, cast

import polars as pl
from studies.dp_calendar import LOCAL_TIME_ZONE, bank_holiday_names
from studies.power import scan_power
from studies.sources import ML_EXPERIMENTS_DATA_DIR, REPO_DATA_DIR

EXPERIMENT_ID: Final[int] = 123
SLUG: Final[str] = "differentiable-physics-e7-storage-heating-night-step-estimation"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}"
WORK_DIR: Final[Path] = DATA_DIR / "work"
"""Small result files (night tables, fits, gates, reports); private and never committed."""
INPUT_DIR: Final[Path] = REPO_DATA_DIR / "studies/per_study/stage1_pilot/demand/inputs"
SERIES_PATH: Final[Path] = INPUT_DIR / "series.parquet"
ERA5_PATH: Final[Path] = INPUT_DIR / "era5_features.parquet"
SERIES_TABLE_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00103-differentiable-physics-1-data-look" / "series_table.parquet"
)
EXPERIMENT_114_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00114-differentiable-physics-n3-n4-interactions-and-heating-threshold"
    / "predictions"
)
DECISIONS_PATH: Final[Path] = WORK_DIR / "decisions.json"
"""Holds the keys that unlock steps: `pr1397_merge_sha` unlocks the scored winter."""

HOLDOUT_START: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""No row on or after this instant is read."""

DECISION_SERIES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
"""The six DP Primaries: the headline pool."""
SELECTED_SERIES: Final[tuple[str, ...]] = ("D5", "D10", "D16")
"""Chosen because winter 2025-26 showed the step; reported as a labelled subset."""
UNSELECTED_DECISION_SERIES: Final[tuple[str, ...]] = ("D9", "D11", "D14")
EXTENSION_SERIES: Final[tuple[str, ...]] = ("D2", "D3", "D4", "D8", "D12", "D19", "D21", "D22")
"""E7-X: Primaries not chosen for any step."""
FITTED_SERIES: Final[tuple[str, ...]] = (*DECISION_SERIES, *EXTENSION_SERIES)
GENERATOR_LINKED_SERIES: Final[tuple[str, ...]] = ("D15", "D18")
"""Never fitted; the unit-level time series of these is not published."""

WINTERS: Final[tuple[int, ...]] = (2021, 2022, 2023, 2024, 2025)
"""Winter `w` runs from the October clock change of year `w` to the March change of `w + 1`."""
TUNING_WINTER: Final[int] = 2023
REPLICATE_WINTER: Final[int] = 2024
SELECTION_WINTER: Final[int] = 2025
DESIGN_WINTERS: Final[tuple[int, ...]] = (TUNING_WINTER, REPLICATE_WINTER)
HEADLINE_WINTERS: Final[tuple[int, ...]] = (2021, 2022, 2023, 2024)
SUMMER_YEARS: Final[tuple[int, ...]] = (2022, 2023, 2024, 2025)
"""Summers whose June to August nights make `d_summer`; summer 2021 is too close to lockdown."""

BASELINE_START_MINUTES: Final[tuple[int, ...]] = (22 * 60, 22 * 60 + 30, 23 * 60, 23 * 60 + 30)
"""Start minutes (from local midnight of the evening date) of the four baseline half-hours."""
WINDOW_START_MINUTES: Final[tuple[int, ...]] = (30, 60, 90, 120, 150)
"""Start minutes (from midnight of the morning date) of the five window half-hours."""
SHORT_WINDOW_ROWS: Final[int] = 3
"""The descriptive three-row window uses the first three window half-hours."""
HEATING_THRESHOLD_C: Final[float] = 15.5
REFERENCE_COLD_C: Final[float] = 12.5
"""A daily mean of 3 C."""
COLD_BST_NIGHT_C: Final[float] = 8.0
BLOCK_DAYS: Final[int] = 14
N_BOOTSTRAP: Final[int] = 1000
INTERVAL_LEVEL: Final[float] = 0.90
MIN_PRESENT_SHARE: Final[float] = 0.90

ClockType = Literal["local", "utc"]
DayType = Literal["weekday", "saturday", "sunday"]
SeasonType = Literal["gmt", "summer", "bst_other", "other"]

SATURDAY: Final[int] = 5
SUNDAY: Final[int] = 6
THREADS_ENVIRONMENT: Final[tuple[str, ...]] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def cap_threads(*, threads: int = 1) -> None:
    """Cap the BLAS thread pools of this process; call before numpy does any work in a worker.

    Args:
        threads: The number of threads per worker.
    """
    for name in THREADS_ENVIRONMENT:
        os.environ[name] = str(threads)


@dataclass(frozen=True)
class SeriesInfo:
    """One fitted series' identity.

    Attributes:
        site: The anonymised label, such as `D10`.
        time_series_id: NGED's series identifier.
        units: `MW` or `MVA`.
        effective_capacity_mw: The series' effective capacity.
    """

    site: str
    time_series_id: int
    units: str
    effective_capacity_mw: float


def series_infos() -> dict[str, SeriesInfo]:
    """Return every Primary series' identity, keyed by label."""
    table = pl.read_parquet(SERIES_TABLE_PATH).select(
        "site", "time_series_id", "units", "effective_capacity_mw"
    )
    return {
        row["site"]: SeriesInfo(
            site=row["site"],
            time_series_id=int(row["time_series_id"]),
            units=str(row["units"]),
            effective_capacity_mw=float(row["effective_capacity_mw"]),
        )
        for row in table.iter_rows(named=True)
    }


def winter_of(*, evening: date) -> int | None:
    """Return the winter year (the year of its October) of an evening in October to March."""
    if evening.month >= 10:
        return evening.year
    if evening.month <= 3:
        return evening.year - 1
    return None


def evening_dates(*, first: date, last: date) -> list[date]:
    """Return every local calendar date from `first` to `last` inclusive."""
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def slot_table(*, clock: ClockType, evenings: list[date]) -> pl.DataFrame:
    """Return the nine half-hour slots of every night as UTC end stamps.

    Args:
        clock: `local` takes the slots as Europe/London wall times; `utc` as UTC times.
        evenings: The evening dates.

    Returns:
        Columns `evening` (date), `role` (`baseline` or `window`), `slot` (0 to 3 or 0 to 4),
        and `end_utc` (the UTC end of the half-hour; null where the local time does not exist).
    """
    rows: list[dict[str, object]] = []
    for evening in evenings:
        morning = evening + timedelta(days=1)
        for slot, minute in enumerate(BASELINE_START_MINUTES):
            start = datetime.combine(evening, time.min) + timedelta(minutes=minute)
            rows.append({"evening": evening, "role": "baseline", "slot": slot, "start": start})
        for slot, minute in enumerate(WINDOW_START_MINUTES):
            start = datetime.combine(morning, time.min) + timedelta(minutes=minute)
            rows.append({"evening": evening, "role": "window", "slot": slot, "start": start})
    frame = pl.DataFrame(rows)
    if clock == "local":
        start_utc = pl.col("start").dt.replace_time_zone(
            LOCAL_TIME_ZONE, ambiguous="earliest", non_existent="null"
        )
    else:
        start_utc = pl.col("start").dt.replace_time_zone("UTC")
    return frame.with_columns(
        end_utc=start_utc.dt.convert_time_zone("UTC").dt.offset_by("30m")
    ).drop("start")


def load_power(*, time_series_ids: list[int]) -> pl.DataFrame:
    """Return the cleaned half-hourly power of the series, never at or after 2026-07-01.

    Args:
        time_series_ids: The series to read.

    Returns:
        Columns `time_series_id`, `time` (UTC end of the half-hour), and `power` (Float64).
    """
    return (
        scan_power()
        .filter(pl.col("time_series_id").is_in(time_series_ids) & (pl.col("time") < HOLDOUT_START))
        .select("time_series_id", "time", power=pl.col("power").cast(pl.Float64))
        .collect()
    )


def load_hourly_temperature(*, time_series_id: int) -> pl.DataFrame:
    """Return ERA5 hourly 2 m temperature of the series' cell with its local date and hour.

    Args:
        time_series_id: The series.

    Returns:
        Columns `valid_time` (UTC), `local_date`, `local_hour`, and `temperature_c` (Float64).
    """
    return (
        pl.scan_parquet(ERA5_PATH)
        .filter(pl.col("time_series_id") == time_series_id)
        .select("valid_time", temperature_c=pl.col("temperature_2m").cast(pl.Float64))
        .filter((pl.col("valid_time").dt.minute() == 0) & (pl.col("valid_time") < HOLDOUT_START))
        .with_columns(local=pl.col("valid_time").dt.convert_time_zone(LOCAL_TIME_ZONE))
        .with_columns(local_date=pl.col("local").dt.date(), local_hour=pl.col("local").dt.hour())
        .drop("local")
        .collect()
    )


def temperature_by_night(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Return each evening's daily mean and night mean temperature.

    The daily mean `daily_mean_c` covers the local calendar day of the morning (the day that
    contains the window rows). The night mean `night_mean_c` covers 18:00 of the evening to 07:00
    of the morning.

    Args:
        hourly: The output of `load_hourly_temperature`.

    Returns:
        Columns `evening`, `daily_mean_c`, `night_mean_c`.
    """
    morning = hourly.group_by("local_date").agg(daily_mean_c=pl.col("temperature_c").mean())
    daily = morning.select(
        evening=pl.col("local_date").dt.offset_by("-1d"), daily_mean_c=pl.col("daily_mean_c")
    )
    evening_part = (
        hourly.filter(pl.col("local_hour") >= 18)
        .group_by("local_date")
        .agg(evening_sum=pl.col("temperature_c").sum(), evening_n=pl.len())
        .rename({"local_date": "evening"})
    )
    morning_part = (
        hourly.filter(pl.col("local_hour") <= 7)
        .group_by("local_date")
        .agg(morning_sum=pl.col("temperature_c").sum(), morning_n=pl.len())
        .select(
            evening=pl.col("local_date").dt.offset_by("-1d"),
            morning_sum="morning_sum",
            morning_n="morning_n",
        )
    )
    return (
        daily.join(evening_part, on="evening", how="left", validate="1:1")
        .join(morning_part, on="evening", how="left", validate="1:1")
        .with_columns(
            night_mean_c=(pl.col("evening_sum") + pl.col("morning_sum"))
            / (pl.col("evening_n") + pl.col("morning_n"))
        )
        .select("evening", "daily_mean_c", "night_mean_c")
    )


def night_differences(*, power: pl.DataFrame, slots: pl.DataFrame, rows: int) -> pl.DataFrame:
    """Return the night difference `d` of one series for one clock.

    Args:
        power: One series' `time` and `power`.
        slots: The output of `slot_table`.
        rows: How many window rows enter the mean (5, or 3 for the descriptive sensitivity).

    Returns:
        Columns `evening`, `d` (window mean minus baseline mean, null unless every row is present),
        `present_rows` (how many of the nine slots have a reading), `window_mean`, `baseline_mean`.
    """
    joined = slots.filter((pl.col("role") == "baseline") | (pl.col("slot") < rows)).join(
        power.select(end_utc=pl.col("time"), power=pl.col("power")),
        on="end_utc",
        how="left",
        validate="m:1",
    )
    expected_rows = rows + len(BASELINE_START_MINUTES)
    summary = joined.group_by("evening").agg(
        present_rows=pl.col("power").is_not_null().sum(),
        window_mean=pl.col("power").filter(pl.col("role") == "window").mean(),
        baseline_mean=pl.col("power").filter(pl.col("role") == "baseline").mean(),
    )
    return (
        summary.with_columns(
            d=pl.when(pl.col("present_rows") == expected_rows)
            .then(pl.col("window_mean") - pl.col("baseline_mean"))
            .otherwise(None)
        )
        .sort("evening")
        .select("evening", "d", "present_rows", "window_mean", "baseline_mean")
    )


def clock_change_dates(*, years: range) -> set[date]:
    """Return the dates on which the UK clocks change (the last Sundays of March and October)."""
    found: set[date] = set()
    for year in years:
        for month in (3, 10):
            day = date(year, month, 31)
            while day.weekday() != SUNDAY:
                day -= timedelta(days=1)
            found.add(day)
    return found


def night_labels(*, evenings: list[date]) -> pl.DataFrame:
    """Return each evening's winter, season, day type, and exclusion flags.

    A night is `gmt` when its evening and morning both fall between the October and March clock
    changes (so the local and UTC clocks agree), `summer` in June to August, and the rest are
    `bst_other` (BST nights outside the summer months) or `other`. Nights are excluded when the
    evening or the morning is a clock-change date or an English bank holiday, or when the
    evening falls from 23 December to 1 January.

    Args:
        evenings: The evening dates.

    Returns:
        Columns `evening`, `winter`, `season`, `day_type`, `excluded`.
    """
    holidays = set(bank_holiday_names())
    changes = clock_change_dates(years=range(evenings[0].year - 1, evenings[-1].year + 2))
    rows = []
    for evening in evenings:
        morning = evening + timedelta(days=1)
        season = _season(evening=evening, morning=morning, changes=changes)
        weekday = evening.weekday()
        day_type: DayType = (
            "saturday" if weekday == SATURDAY else "sunday" if weekday == SUNDAY else "weekday"
        )
        christmas_run = (evening.month == 12 and evening.day >= 23) or (
            evening.month == 1 and evening.day == 1
        )
        excluded = (
            evening in holidays
            or morning in holidays
            or evening in changes
            or morning in changes
            or christmas_run
        )
        rows.append(
            {
                "evening": evening,
                "winter": winter_of(evening=evening) if season == "gmt" else None,
                "season": season,
                "day_type": day_type,
                "excluded": excluded,
            }
        )
    return pl.DataFrame(rows, schema_overrides={"winter": pl.Int64})


def _season(*, evening: date, morning: date, changes: set[date]) -> SeasonType:
    """Classify a night by the clocks: GMT, summer, other BST, or neither."""
    winter = winter_of(evening=evening)
    if winter is not None:
        autumn = _change_date(changes=changes, year=winter, month=10)
        spring = _change_date(changes=changes, year=winter + 1, month=3)
        if autumn < evening and morning < spring:
            return "gmt"
    if evening.month in (6, 7, 8):
        return "summer"
    march = _change_date(changes=changes, year=evening.year, month=3)
    october = _change_date(changes=changes, year=evening.year, month=10)
    return "bst_other" if march < evening < october else "other"


def _change_date(*, changes: set[date], year: int, month: int) -> date:
    """Return the clock-change date of the given month and year."""
    return next(day for day in changes if day.year == year and day.month == month)


def build_night_table(*, info: SeriesInfo, power: pl.DataFrame) -> pl.DataFrame:
    """Return one series' table with one row per evening, on both clocks.

    Args:
        info: The series.
        power: The series' `time` and `power` (from `load_power`).

    Returns:
        Columns `site`, `evening`, `winter`, `season`, `day_type`, `excluded`, `daily_mean_c`,
        `night_mean_c`, `cold_c` (degrees below 15.5 C of the daily mean, floored at 0),
        `cold_night_c` (the same from the night mean), `d` and `present_rows` on the local clock,
        `d_utc` and `present_rows_utc` on the UTC clock, `d_three_row` (local clock, three window
        rows), `window_mean`, and `baseline_mean` (local clock). `d` is null unless every one of the
        nine slots has a reading.
    """
    first_reading = cast("datetime | None", power["time"].min())
    last_reading = cast("datetime | None", power["time"].max())
    if first_reading is None or last_reading is None:
        msg = f"{info.site} has no readings"
        raise ValueError(msg)
    evenings = evening_dates(first=first_reading.date(), last=last_reading.date())
    evenings = [day for day in evenings if day <= date(2026, 6, 28)]
    labels = night_labels(evenings=evenings)
    temperature = temperature_by_night(
        hourly=load_hourly_temperature(time_series_id=info.time_series_id)
    )
    local = night_differences(
        power=power, slots=slot_table(clock="local", evenings=evenings), rows=5
    )
    utc = night_differences(power=power, slots=slot_table(clock="utc", evenings=evenings), rows=5)
    three = night_differences(
        power=power, slots=slot_table(clock="local", evenings=evenings), rows=3
    )
    return (
        labels.join(temperature, on="evening", how="left", validate="1:1")
        .join(local, on="evening", how="left", validate="1:1")
        .join(
            utc.select("evening", d_utc="d", present_rows_utc="present_rows"),
            on="evening",
            how="left",
            validate="1:1",
        )
        .join(three.select("evening", d_three_row="d"), on="evening", how="left", validate="1:1")
        .with_columns(
            site=pl.lit(info.site),
            cold_c=(HEATING_THRESHOLD_C - pl.col("daily_mean_c")).clip(lower_bound=0.0),
            cold_night_c=(HEATING_THRESHOLD_C - pl.col("night_mean_c")).clip(lower_bound=0.0),
        )
        .select(
            "site",
            "evening",
            "winter",
            "season",
            "day_type",
            "excluded",
            "daily_mean_c",
            "night_mean_c",
            "cold_c",
            "cold_night_c",
            "d",
            "present_rows",
            "d_utc",
            "present_rows_utc",
            "d_three_row",
            "window_mean",
            "baseline_mean",
        )
    )


def scored_winter_unlocked() -> bool:
    """Return True once `decisions.json` pins experiment 114's merge commit.

    The maintainer ruled that winter 2025-26 (and the scored-year residuals of experiment 114) are
    read only after PR 1397 has merged. The run writes the merge commit's SHA under the key
    `pr1397_merge_sha`, and nothing reads the scored winter before that key exists.
    """
    if not DECISIONS_PATH.exists():
        return False
    import json

    sha = json.loads(DECISIONS_PATH.read_text()).get("pr1397_merge_sha")
    return re.fullmatch(r"[0-9a-f]{40}", str(sha)) is not None


def allowed_winters(*, winters: tuple[int, ...]) -> tuple[int, ...]:
    """Return `winters` without the selection winter unless the scored winter is unlocked."""
    if scored_winter_unlocked():
        return winters
    return tuple(w for w in winters if w != SELECTION_WINTER)


def b0_residual_mw(*, info: SeriesInfo, tags: tuple[str, ...]) -> pl.DataFrame:
    """Return B0's day-ahead median residual in MW for a series, from experiment 114's predictions.

    Args:
        info: The series.
        tags: Prediction-set prefixes (`tuning`, `replicate`, `scored`); `scored` needs the unlock.

    Returns:
        Columns `time` (UTC end of the half-hour) and `power` (actual minus the median forecast, in
        MW), the same columns as `load_power`.

    Raises:
        PermissionError: If `scored` is requested before the scored winter is unlocked.
    """
    if "scored" in tags and not scored_winter_unlocked():
        msg = "experiment 114's scored-year residuals are locked until PR 1397 has merged"
        raise PermissionError(msg)
    frames = [
        pl.read_parquet(EXPERIMENT_114_DIR / f"{tag}_b0" / "predictions.parquet").filter(
            pl.col("time_series_id") == info.time_series_id
        )
        for tag in tags
    ]
    scale = info.effective_capacity_mw / 100.0
    return (
        pl.concat(frames)
        .filter(pl.col("valid_time") < HOLDOUT_START)
        .select(
            time=pl.col("valid_time"),
            power=(pl.col("actual").cast(pl.Float64) - pl.col("q50").cast(pl.Float64)) * scale,
        )
        .unique(subset="time", keep="first")
        .sort("time")
    )
