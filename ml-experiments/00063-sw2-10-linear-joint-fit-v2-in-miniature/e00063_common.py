"""Paths, series, row filters, and the loader of experiment 00063 (SW2-10).

The experiment fits one linear joint model per Primary substation on top of a weather baseline
(`docs/ml-experiments/00063-sw2-10-linear-joint-fit-v2-in-miniature/result.md`). Every script reads
the private tables of experiments 00041 (hindsight baseline, labelled switching events), 00045
(weekly capacity, CAMS unit PV shape, solar-farm output), and 00006 (features), and writes under
`work/` of this experiment's private folder. Run each script on at most four cores:

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python ...

Two environment variables select sensitivity runs; the defaults are the plan's choices, and any
other value writes its results to a separate folder (`results_<tags>/`):

- `E63_STATUSES` (comma separated, default `clean,low-power`) sets which capacity statuses of
  experiment 00045 are kept.
- `E63_EVENT_ROWS` (default `plan`; `any_week` is the sensitivity) sets whether the half-hours
  inside labelled events must lie in an observed week. A week is `unobserved` when closure padding
  leaves fewer than 3 usable days, so the plan's filter removes most event rows.
"""

import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_FOLDER: Final[str] = "00063-sw2-10-linear-joint-fit-v2-in-miniature"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / EXPERIMENT_FOLDER / "work"
"""Private tables of this experiment; never committed."""
BASELINE_DIR: Final[Path] = WORK_DIR / "baseline_no_radiation"
"""Leave-one-month-out XGBoost medians without shortwave and longwave, one file per series."""

FEATURES_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00006-stage1a-demand-input-feature-sweep"
    / "inputs"
    / "features.parquet"
)
LOGGED_INTERVALS_PATH: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "sw1b_scratch" / "intervals.parquet"
"""Every closure reconstructed from the confirmed logs, local times, any duration."""
FROZEN_BASELINE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "hindsight"
)
"""Experiment 00041's frozen v1 baseline (13 columns, radiation included), one file per series."""
LABELLED_SET_PATH: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00041-sw2-0a-labeller-gate" / "real" / "labelled_set.parquet"
)
PVDIS_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00045-pvdis-4-weekly-capacity-asymmetric-l1"
CAPACITY_PATH: Final[Path] = PVDIS_DIR / "capacity_weekly.parquet"
HALF_HOURLY_PATH: Final[Path] = PVDIS_DIR / "inputs_half_hourly.parquet"
FARM_HALF_HOURLY_PATH: Final[Path] = PVDIS_DIR / "farm_half_hourly.parquet"

STATUSES_ENV: Final[str] = os.environ.get("E63_STATUSES", "clean,low-power")
ALLOWED_STATUSES: Final[tuple[str, ...]] = tuple(STATUSES_ENV.split(","))
"""Capacity statuses (experiment 00045) whose series are scored; the plan: clean or low-power."""
EVENT_ROWS_ENV: Final[str] = os.environ.get("E63_EVENT_ROWS", "plan")
EVENT_ROWS_ANY_WEEK: Final[bool] = EVENT_ROWS_ENV == "any_week"
"""Whether labelled-event rows may lie in `unobserved` weeks (the capacity is still defined)."""
_TAGS: Final[list[str]] = [
    *([] if STATUSES_ENV == "clean,low-power" else [STATUSES_ENV.replace(",", "_")]),
    *(["events_any_week"] if EVENT_ROWS_ANY_WEEK else []),
]
RESULTS_DIR: Final[Path] = WORK_DIR / ("results" + "".join(f"_{tag}" for tag in _TAGS))
"""Tables written by the fit, injection, real-event, and report scripts."""

SERIES_IDS: Final[tuple[int, ...]] = (7, 8, 10, 12, 13, 16, 17, 18)
"""The 8 Primaries metered in MW that have a weekly capacity."""
SERIES_NAMES: Final[dict[int, str]] = {
    7: "Trusthorpe",
    8: "Horncastle",
    10: "Spilsby",
    12: "Wrangle T1",
    13: "Wrangle T2",
    16: "Mount Bridge Boston",
    17: "Kirton",
    18: "Donington",
}
"""Substation names; substations may be named on a published page."""
SERIES_SITE_CODES: Final[dict[int, str]] = {
    7: "D9",
    8: "D11",
    10: "D10",
    12: "D15",
    13: "D18",
    16: "D14",
    17: "D5",
    18: "D16",
}
"""The frame site code of each series in experiment 00045's half-hourly inputs."""

WINDOW_START: Final[datetime] = datetime(2023, 9, 4, tzinfo=UTC)
R1_WINDOW_START: Final[datetime] = datetime(2022, 1, 1, tzinfo=UTC)
"""R1 reads events from the first month of the baseline files, because the plan sets no window."""
WINDOW_END: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
"""Scored and fitted rows lie in [WINDOW_START, WINDOW_END)."""
SKIPPED_WINDOWS: Final[dict[int, tuple[datetime, datetime]]] = {
    16: (datetime(2023, 9, 4, tzinfo=UTC), datetime(2025, 8, 26, tzinfo=UTC))
}
"""Mount Bridge Boston's constant capacity is about 0 MW from 2023-09-04 to 2025-08-25 inclusive."""
FIRST_WEEK_START: Final[date] = date(2019, 9, 2)
"""Monday of week 0 of experiment 00045's week index."""
OK_ROW_STATUSES: Final[tuple[str, ...]] = ("clean", "low-power")
CAPACITY_VARIANT: Final[str] = "season_year_constant"
WEEKLY_CAPACITY_VARIANT: Final[str] = "alt_post_hoc"
"""The recommended weekly capacity series (post hoc pair a=2, r=10), the sensitivity C3."""
SUMMER_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9)
PAD: Final[timedelta] = timedelta(hours=24)
VISIBLE: Final[tuple[str, ...]] = ("conserved", "one_sided")
"""Visibilities of a labelled event whose onset and end are given."""
EARLIEST_LOG: Final[str] = "2019-01-01"
"""Logged times before this date are typing errors (the sheet holds a 2008 row)."""

SeasonType = Literal["all", "summer", "winter"]


def logged_intervals() -> pl.DataFrame:
    """Return every closed logged interval with UTC `start_utc`, `end_utc`, and sides `a`, `b`."""
    return (
        pl.read_parquet(LOGGED_INTERVALS_PATH)
        .filter(
            pl.col("end").is_not_null(), pl.col("start") >= pl.lit(EARLIEST_LOG).str.to_datetime()
        )
        .with_columns(
            start_utc=pl.col("start")
            .dt.replace_time_zone("Europe/London", ambiguous="earliest", non_existent="null")
            .dt.convert_time_zone("UTC"),
            end_utc=pl.col("end")
            .dt.replace_time_zone("Europe/London", ambiguous="earliest", non_existent="null")
            .dt.convert_time_zone("UTC"),
        )
        .drop_nulls(["start_utc", "end_utc"])
        .filter(pl.col("end_utc") > pl.col("start_utc"))
    )


def effective_capacity_mw() -> dict[int, float]:
    """Return each series' `effective_capacity_mw` (the 99th percentile of its power magnitude)."""
    table = (
        pl.scan_parquet(FEATURES_PATH)
        .filter(pl.col("time_series_id").is_in(list(SERIES_IDS)))
        .group_by("time_series_id")
        .agg(capacity=pl.col("effective_capacity_mw").max())
        .collect()
    )
    return {
        int(s): float(c) for s, c in zip(table["time_series_id"], table["capacity"], strict=True)
    }


@dataclass(frozen=True)
class Event:
    """One labelled visible switching event of experiment 00041, between two Primaries.

    Attributes:
        card_id: The labelled set's card id.
        kind: The card kind (`real_operation`, `real_short`, `real_edge_close`, `real_edge_open`).
        side_a: `time_series_id` of the first side of the closure.
        side_b: `time_series_id` of the second side, or `None` for a one-sided closure.
        onset: First stamp of the event, UTC.
        end: Last stamp of the event, UTC.
        donor: `time_series_id` of the donor, or `None` when the labeller named none.
        visibility: `conserved` or `one_sided`.
    """

    card_id: str
    kind: str
    side_a: int
    side_b: int | None
    onset: datetime
    end: datetime
    donor: int | None
    visibility: str

    def touches(self, *, series: int) -> bool:
        """Return whether `series` is one of the event's sides."""
        return series in (self.side_a, self.side_b)

    def counterparty(self, *, series: int) -> int | None:
        """Return the other side of the event, or `None` when `series` is not a side."""
        if series == self.side_a:
            return self.side_b
        if series == self.side_b:
            return self.side_a
        return None


def real_visible_events() -> list[Event]:
    """Return experiment 00041's real visible events (`kind` starting `real_`) with a time span.

    Cards of kind `alarm`, `alarm_placebo`, and `sentinel_*` are excluded: the sentinel cards hold
    injected or jittered transfers and the alarm cards hold detector alarms, not labelled events.
    """
    labelled = (
        pl.read_parquet(LABELLED_SET_PATH)
        .filter(
            pl.col("kind").str.starts_with("real_"),
            pl.col("visibility").is_in(list(VISIBLE)),
            pl.col("onset_utc").is_not_null(),
            pl.col("end_utc").is_not_null(),
        )
        .sort("onset_utc", "card_id")
    )
    return [
        Event(
            card_id=row["card_id"],
            kind=row["kind"],
            side_a=int(row["side_a"]),
            side_b=None if row["side_b"] is None else int(row["side_b"]),
            onset=row["onset_utc"],
            end=row["end_utc"],
            donor=None if row["donor"] is None else int(row["donor"]),
            visibility=row["visibility"],
        )
        for row in labelled.iter_rows(named=True)
    ]


def _load_frozen(*, series: int, directory: Path, suffix: str) -> pl.DataFrame:
    """Read one baseline file and rename its quantile columns with `suffix`."""
    return pl.read_parquet(directory / f"{series}.parquet").select(
        "valid_time",
        "obs_mw",
        "in_logged",
        **{f"q{q}_{suffix}": pl.col(f"q{q}_mw") for q in (10, 50, 90)},
    )


def power_wide() -> pl.DataFrame:
    """Return the metered power of every series in MW, one column `p<series>` per series.

    The power comes from the frozen baseline files' `obs_mw`, so it holds the same rows as the
    baseline (every half-hour from 2022-01 with a measurement).
    """
    frame: pl.DataFrame | None = None
    for series in SERIES_IDS:
        part = pl.read_parquet(FROZEN_BASELINE_DIR / f"{series}.parquet").select(
            "valid_time", **{f"p{series}": pl.col("obs_mw")}
        )
        frame = (
            part if frame is None else frame.join(part, on="valid_time", how="full", coalesce=True)
        )
    assert frame is not None
    return frame.sort("valid_time")


@dataclass
class SeriesRows:
    """One Primary's half-hours in the scoring window, with every column the fits need.

    Columns of `frame` (one row per half-hour of the frozen baseline file inside the window that
    also has a row in experiment 00045's half-hourly inputs):

    `valid_time` (UTC), `obs_mw`, `v1_q10/q50/q90` (frozen baseline with radiation),
    `nr_q10/q50/q90` (baseline without radiation), `in_logged` (a logged closure padded by 24
    hours, from the baseline file), `masked` (the same, from experiment 00045), `u` (CAMS unit PV
    shape), `daylight`, `month_id` (year * 12 + month - 1), `cal_month` (1 to 12), `half_hour` (0 to
    47, UTC), `capacity_const`, `capacity_weekly`, `capacity_previous` (previous season-year's
    constant, null when none), `unobserved` (the week's flag), `eligible` (see `load_series`).

    Attributes:
        series: The `time_series_id`.
        status: The capacity status of experiment 00045.
        capacity_norm_mw: The `effective_capacity_mw` that errors are normalised by.
        frame: The table described above.
    """

    series: int
    status: str
    capacity_norm_mw: float
    frame: pl.DataFrame


def _capacity_columns(*, series: int) -> tuple[pl.DataFrame, str]:
    """Return the series' weekly capacity table (constant, weekly, previous) and its status."""
    table = pl.read_parquet(CAPACITY_PATH).filter(pl.col("time_series_id") == series)
    status = table["status"][0]
    constant = table.filter(pl.col("variant") == CAPACITY_VARIANT).select(
        "week_start", capacity_const="capacity_mw", unobserved="unobserved"
    )
    weekly = table.filter(pl.col("variant") == WEEKLY_CAPACITY_VARIANT).select(
        "week_start", capacity_weekly="capacity_mw"
    )
    by_season_year = (
        constant.with_columns(
            season_year=pl.col("week_start").dt.year()
            - (pl.col("week_start").dt.month() < 9).cast(pl.Int32)
        )
        .group_by("season_year")
        .agg(previous_value=pl.col("capacity_const").median())
        .with_columns(season_year=pl.col("season_year") + 1)
    )
    joined = (
        constant.join(weekly, on="week_start", how="left")
        .with_columns(
            season_year=pl.col("week_start").dt.year()
            - (pl.col("week_start").dt.month() < 9).cast(pl.Int32)
        )
        .join(by_season_year, on="season_year", how="left")
        .rename({"previous_value": "capacity_previous"})
        .drop("season_year")
    )
    return joined, status


def load_series(
    *, series: int, with_no_radiation: bool = True, window_start: datetime = WINDOW_START
) -> SeriesRows:
    """Build one Primary's table of half-hours and mark which rows are `eligible`.

    A row is `eligible` when the series' capacity status is in `ALLOWED_STATUSES`, the row lies in
    the window and outside `SKIPPED_WINDOWS`, its week is not `unobserved`, and the metered power,
    the CAMS shape, both baselines, and the season-year constant capacity are present.
    `eligible_any_week` is the same without the `unobserved` condition. Row blocking
    by logged closures and labelled events is left to the caller, because the fits treat labelled
    rows differently from the other blocked rows.

    Args:
        series: A `time_series_id` in `SERIES_IDS`.
        with_no_radiation: Whether to read the baseline without radiation (false skips it, for the
            baseline script's own check).
        window_start: The first stamp kept (the scoring window starts at `WINDOW_START`).

    Returns:
        The series' rows.
    """
    capacity_table, status = _capacity_columns(series=series)
    frozen = _load_frozen(series=series, directory=FROZEN_BASELINE_DIR, suffix="v1")
    frame = frozen.select(
        "valid_time",
        "obs_mw",
        "in_logged",
        v1_q10="q10_v1",
        v1_q50="q50_v1",
        v1_q90="q90_v1",
    )
    if with_no_radiation:
        no_radiation = _load_frozen(series=series, directory=BASELINE_DIR, suffix="nr").select(
            "valid_time", nr_q10="q10_nr", nr_q50="q50_nr", nr_q90="q90_nr"
        )
        frame = frame.join(no_radiation, on="valid_time", how="inner")
    inputs = (
        pl.scan_parquet(HALF_HOURLY_PATH)
        .filter(pl.col("series_id") == series)
        .select("time", "u", "daylight", "masked", "week")
        .collect()
    )
    frame = (
        frame.join(inputs, left_on="valid_time", right_on="time", how="inner")
        .filter(pl.col("valid_time") >= window_start, pl.col("valid_time") < WINDOW_END)
        .with_columns(
            week_start=pl.lit(FIRST_WEEK_START) + pl.duration(weeks=pl.col("week")),
            month_id=pl.col("valid_time").dt.year().cast(pl.Int32) * 12
            + pl.col("valid_time").dt.month().cast(pl.Int32)
            - 1,
            cal_month=pl.col("valid_time").dt.month(),
            half_hour=(pl.col("valid_time").dt.hour() * 2 + pl.col("valid_time").dt.minute() // 30),
        )
        .join(capacity_table, on="week_start", how="left")
        .sort("valid_time")
    )
    skipped = SKIPPED_WINDOWS.get(series)
    in_skip = (
        (pl.col("valid_time") >= skipped[0]) & (pl.col("valid_time") < skipped[1])
        if skipped is not None
        else pl.lit(False)
    )
    any_week = (
        pl.col("obs_mw").is_not_null()
        & pl.col("u").is_not_null()
        & pl.col("v1_q50").is_not_null()
        & pl.col("capacity_const").is_not_null()
        & ~in_skip
        & pl.lit(status in ALLOWED_STATUSES)
    )
    if with_no_radiation:
        any_week = any_week & pl.col("nr_q50").is_not_null()
    frame = frame.with_columns(
        eligible=any_week & ~pl.col("unobserved").fill_null(True),
        eligible_any_week=any_week,
        blocked=pl.col("in_logged") | pl.col("masked"),
    )
    return SeriesRows(
        series=series,
        status=status,
        capacity_norm_mw=effective_capacity_mw()[series],
        frame=frame,
    )


def event_mask(*, times: np.ndarray, event: Event) -> np.ndarray:
    """Return True for each stamp (datetime64[us]) with `onset <= stamp <= end`."""
    onset = np.datetime64(event.onset.astimezone(UTC).replace(tzinfo=None), "us")
    end = np.datetime64(event.end.astimezone(UTC).replace(tzinfo=None), "us")
    return (times >= onset) & (times <= end)


def times_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the frame's `valid_time` as a numpy datetime64[us] array of UTC stamps."""
    return frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
