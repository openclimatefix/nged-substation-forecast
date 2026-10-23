"""The corrections of experiment 155: the winter night offset, recipe R, and the analog profile.

**Every correction is a shift added to arm A's mean and quantiles, then the spread is refitted.**
`add_calendar` adds the columns that the corrections read. `winter_offset_shift` is the night term
of experiment 137 with a winter indicator in place of the cold depth (the audit of that experiment
found the two do equally well). `recipe_r_shift` and `analog_shift` are experiment 115's recipe R
and experiment 145's analog-day profile, rebuilt on the rows of a frame. No correction reads
anything dated on or after the origin of the run it corrects (see the leakage control).
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import e00278_d_common as common
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names
from studies.dp_ens import LeadHourSpread

EPOCH: Final[np.datetime64] = np.datetime64("1970-01-01", "D")
EPOCH_DATE: Final[date] = date(1970, 1, 1)
LONDON: Final[str] = "Europe/London"
N_HALF_HOURS: Final[int] = 48
DAY_TYPES: Final[tuple[int, ...]] = (0, 1, 1, 1, 2, 3, 4)
"""The day type of Monday (0) to Sunday (6)."""
DECEMBER_RUN_DAYS: Final[tuple[int, ...]] = (24, 27, 28, 29, 30, 31)
MIN_ANALOGS: Final[int] = 3
MIN_HOLIDAY_FALLBACK: Final[int] = 2
N_COLUMNS: Final[int] = common.N_NIGHT_SLOTS * 3
VARIANTS: Final[tuple[str, ...]] = ("winter", "const", "placebo")
"""`winter`: constant, winter indicator, weekend. `const`: no winter indicator. `placebo`: the
winter indicator moved six months (May to September)."""


def add_calendar(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the columns the corrections read.

    Args:
        frame: A forecast frame with `valid_time` and `power_fcst_init_time`.

    Returns:
        The frame with `origin_day` (days since 1970-01-01 of the origin's UTC date), `local_hh`
        (the local half-hour of day, 0 to 47, from the valid time in Europe/London), `slot` (0 to
        23 for the night half-hours that end between 21:30 and 09:00 UTC, else -1), `winter`,
        `placebo_winter`, and `weekend` (flags of the evening's date).
    """
    minute = pl.col("valid_time").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "valid_time"
    ).dt.minute().cast(pl.Int32)
    slot = ((minute - common.NIGHT_START_MINUTE) % 1440) // 30
    evening = (pl.col("valid_time") - pl.duration(hours=21, minutes=30)).dt.date()
    local = pl.col("valid_time").dt.convert_time_zone(LONDON)
    return frame.with_columns(
        origin_day=pl.col("power_fcst_init_time").dt.date().cast(pl.Int32),
        local_hh=(local.dt.hour().cast(pl.Int32) * 2 + local.dt.minute().cast(pl.Int32) // 30),
        slot=pl.when(slot < common.N_NIGHT_SLOTS).then(slot).otherwise(-1).cast(pl.Int32),
        evening=evening,
    ).with_columns(
        winter=evening.dt.month().is_in(list(common.WINTER_MONTHS)).cast(pl.Float64),
        placebo_winter=((evening.dt.month() + 5) % 12 + 1)
        .is_in(list(common.WINTER_MONTHS))
        .cast(pl.Float64),
        weekend=(evening.dt.weekday() >= 6).cast(pl.Float64),
    )


def night_design(*, frame: pl.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the offset's design matrix on the night rows, and the mask of those rows.

    Each night slot has three columns: a constant, a second flag (the winter indicator, zero for
    `const`, or the placebo indicator), and a weekend flag.
    """
    slot = frame["slot"].to_numpy()
    keep = slot >= 0
    second = {
        "winter": frame["winter"].to_numpy(),
        "const": np.zeros(frame.height),
        "placebo": frame["placebo_winter"].to_numpy(),
    }[variant]
    weekend = frame["weekend"].to_numpy()
    rows = np.flatnonzero(keep)
    x = np.zeros((len(rows), N_COLUMNS))
    s = slot[rows]
    x[np.arange(len(rows)), 3 * s] = 1.0
    x[np.arange(len(rows)), 3 * s + 1] = second[rows]
    x[np.arange(len(rows)), 3 * s + 2] = weekend[rows]
    return x, keep


def winter_offset_shift(
    *, frame: pl.DataFrame, variant: str = "winter", ridge: float = common.OFFSET_RIDGE
) -> tuple[np.ndarray, dict[object, np.ndarray]]:
    """Return the offset's shift at every row, and its coefficients at each monthly cut.

    The coefficients of the cut at the first of a month are a ridge regression of the forecast's
    own errors (actual minus mean) on the night rows whose run origin and valid time are both
    before that cut (an expanding window). The shift applies to the runs whose origin is in the
    month. A month with no training rows gets a shift of zero.

    Args:
        frame: The forecast frame, with the columns of `add_calendar`.
        variant: `winter`, `const`, or `placebo`.
        ridge: The ridge penalty.

    Returns:
        The shift per row, and the coefficients per cut.
    """
    x, keep = night_design(frame=frame, variant=variant)
    night = frame.filter(pl.Series(keep))
    error = (night["actual"] - night["mean"]).to_numpy().astype(np.float64)
    finite = np.isfinite(error)
    origin = night["power_fcst_init_time"].to_numpy()
    valid = night["valid_time"].to_numpy()
    month = night["power_fcst_init_time"].dt.truncate("1mo").to_numpy()
    shift_night = np.zeros(night.height)
    betas: dict[object, np.ndarray] = {}
    for cut in np.unique(month):
        train = finite & (origin < cut) & (valid < cut)
        if not train.any():
            continue
        beta = np.linalg.solve(
            x[train].T @ x[train] + ridge * np.eye(N_COLUMNS), x[train].T @ error[train]
        )
        betas[cut] = beta
        this = month == cut
        shift_night[this] = x[this] @ beta
    shift = np.zeros(frame.height)
    shift[np.flatnonzero(keep)] = shift_night
    return shift, betas


@dataclass(frozen=True)
class DayBlocks:
    """One block of centred D+1 errors per run.

    Attributes:
        runs: The run indices, ascending by origin.
        targets: Each run's target day (days since 1970-01-01): the origin's UTC day plus one.
        blocks: Shape (runs, 2, 48): the sum and the count of each local half-hour's centred errors.
        usable: True where the block has at least `MIN_DAY_READINGS` readings.
        row_run: For each row of the frame, the index into `runs`.
    """

    runs: np.ndarray
    targets: np.ndarray
    blocks: np.ndarray
    usable: np.ndarray
    row_run: np.ndarray


def day_blocks(*, frame: pl.DataFrame, forecast: np.ndarray) -> DayBlocks:
    """Return each run's centred D+1 error block (forecast minus actual, by local half-hour).

    A run's block covers its rows at leads of 15.5 to 39 hours (the UTC day after the origin's). A
    clock-change day holds one local half-hour twice, so the block keeps a sum and a count.

    Args:
        frame: The forecast frame, with the columns of `add_calendar`.
        forecast: The forecast to take errors of, at every row (A's mean plus any earlier shift).

    Returns:
        The blocks.
    """
    run_ids = frame["run"].to_numpy()
    unique, row_run = np.unique(run_ids, return_inverse=True)
    lead = frame["lead_hours"].to_numpy()
    actual = frame["actual"].to_numpy().astype(np.float64)
    local_hh = frame["local_hh"].to_numpy()
    origin_day = frame["origin_day"].to_numpy()
    error = forecast.astype(np.float64) - actual
    in_d1 = (lead >= common.D1_FIRST_LEAD) & (lead <= common.D1_LAST_LEAD) & np.isfinite(error)
    blocks = np.zeros((len(unique), 2, N_HALF_HOURS))
    for run in range(len(unique)):
        rows = np.flatnonzero((row_run == run) & in_d1)
        if len(rows) < common.MIN_DAY_READINGS:
            continue
        centred = error[rows] - error[rows].mean()
        np.add.at(blocks[run, 0], local_hh[rows], centred)
        np.add.at(blocks[run, 1], local_hh[rows], 1.0)
    first_row = np.array([np.flatnonzero(row_run == run)[0] for run in range(len(unique))])
    targets = origin_day[first_row].astype(np.int64) + 1
    usable = blocks[:, 1].sum(axis=1) >= common.MIN_DAY_READINGS
    return DayBlocks(
        runs=unique, targets=targets, blocks=blocks, usable=usable, row_run=row_run.astype(np.int64)
    )


def _mean_by_half_hour(*, blocks: np.ndarray) -> np.ndarray:
    """Return the mean over blocks of shape (n, 2, 48), by half-hour; zero where none counts."""
    total = blocks[:, 0].sum(axis=0)
    count = blocks[:, 1].sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def fourteen_day_profiles(*, days: DayBlocks) -> tuple[np.ndarray, np.ndarray]:
    """Return recipe R's 14-day profile at each run, and whether it has enough source days.

    A source day is `T - back` for `back` in 2 to 15, where `T` is the run's target day.
    """
    index_of = {int(day): i for i, day in enumerate(days.targets)}
    out = np.zeros((len(days.targets), N_HALF_HOURS))
    ok = np.zeros(len(days.targets), dtype=bool)
    first = common.R_FIRST_SOURCE_BACK
    for run, target in enumerate(days.targets.tolist()):
        chosen = [
            i
            for back in range(first, common.R_PROFILE_DAYS + first)
            if (i := index_of.get(target - back)) is not None and days.usable[i]
        ]
        if len(chosen) >= common.R_MIN_DAYS:
            out[run] = _mean_by_half_hour(blocks=days.blocks[chosen])
            ok[run] = True
    return out, ok


def shift_at_rows(
    *, days: DayBlocks, frame: pl.DataFrame, by_run: np.ndarray, short_only: bool
) -> np.ndarray:
    """Spread each run's shift by local half-hour to its rows (zero beyond 48 h if `short_only`)."""
    shift = by_run[days.row_run, frame["local_hh"].to_numpy()]
    if short_only:
        shift = np.where(
            frame["lead_hours"].to_numpy() <= common.SHORT_LEAD_LIMIT_HOURS, shift, 0.0
        )
    return shift


def recipe_r_shift(
    *, frame: pl.DataFrame, forecast: np.ndarray, short_only: bool = False
) -> np.ndarray:
    """Return recipe R's shift at every row: minus 0.75 times the 14-day profile.

    A run with fewer than 3 usable source days gets a shift of zero.

    Args:
        frame: The forecast frame, with the columns of `add_calendar`.
        forecast: The forecast whose past errors form the profile.
        short_only: True to zero the shift on rows with a lead above 48 hours.

    Returns:
        The shift per row, in points.
    """
    days = day_blocks(frame=frame, forecast=forecast)
    profile, ok = fourteen_day_profiles(days=days)
    by_run = np.where(ok[:, None], -common.R_KAPPA * profile, 0.0)
    return shift_at_rows(days=days, frame=frame, by_run=by_run, short_only=short_only)


@dataclass(frozen=True)
class AnalogSpec:
    """How the analog profile shift is built (experiment 145's tuned ANALOG+R values).

    Attributes:
        kappa: The shrink applied to the profile.
        k: The number of analog days averaged.
        weeks: The season window, in weeks either side of the target's day of the year.
        weight: The weight of the analog profile against the 14-day profile.
    """

    kappa: float = 1.0
    k: int = 20
    weeks: int = 13
    weight: float = 0.5


def holiday_identities(*, first: date, last: date) -> dict[date, str]:
    """Return the holiday-like days between two dates, each with its identity."""
    out = {day: name for day, name in bank_holiday_names().items() if first <= day <= last}
    for day, name in bank_holiday_names().items():
        if name == "Good Friday":
            out[day + timedelta(days=1)] = "Easter Saturday"
            out[day + timedelta(days=2)] = "Easter Sunday"
    day = first
    while day <= last:
        if day.month == 12 and day.day in DECEMBER_RUN_DAYS:
            out.setdefault(day, f"December {day.day}")
        day += timedelta(days=1)
    return {day: name for day, name in out.items() if first <= day <= last}


def forecast_temperature(*, site: str, days: DayBlocks) -> np.ndarray:
    """Return each run's D+1 daily mean ENS ensemble-mean temperature, `NaN` where it has none.

    The value is the raw ENS value of experiment 111's store, the weather the run saw, so it exists
    at forecast time. Row `i` of a run's window is the half-hour `i + 1` after the origin's, so the
    D+1 block (leads 15.5 to 39 hours) is positions 30 to 77.
    """
    with np.load(common.E111_STORE_DIR / f"{site}.npz") as stored:
        temperature = stored["temperature"]
    out = np.full(len(days.runs), np.nan)
    for position, run in enumerate(days.runs.tolist()):
        values = temperature[run][:, 30:78]
        if values.shape[1] == N_HALF_HOURS and np.isfinite(values).any():
            out[position] = np.nanmean(values.mean(axis=0))
    return out


@dataclass(frozen=True)
class AnalogDays:
    """The calendar facts of each run's target day, for choosing analog source days."""

    days: DayBlocks
    temperature_c: np.ndarray
    day_type: np.ndarray
    day_of_year: np.ndarray
    identity: np.ndarray


def build_analog_days(*, site: str, days: DayBlocks) -> AnalogDays:
    """Return the target day's type, day of year, holiday identity, and forecast temperature."""
    weekday = (days.targets + 3) % 7  # 1970-01-01 was a Thursday; Monday is 0
    dates = [EPOCH_DATE + timedelta(days=int(t)) for t in days.targets]
    names = holiday_identities(first=dates[0], last=dates[-1])
    return AnalogDays(
        days=days,
        temperature_c=forecast_temperature(site=site, days=days),
        day_type=np.array([DAY_TYPES[w] for w in weekday]),
        day_of_year=np.array([d.timetuple().tm_yday for d in dates]),
        identity=np.array([names.get(d, "") for d in dates]),
    )


def analog_days_of_run(*, analog: AnalogDays, run: int, spec: AnalogSpec) -> np.ndarray | None:
    """Return the source days nearest in forecast temperature to a run's target, or `None`.

    A non-holiday target takes non-holiday source days of the same day type within `spec.weeks`
    weeks of its day of the year. A holiday-like target takes source days with the same holiday
    identity, else any holiday-like days when at least 2 exist. The nearest `spec.k` by forecast
    daily mean temperature are kept, ties going to the more recent day.
    """
    days = analog.days
    target = int(days.targets[run])
    newest = int(np.searchsorted(days.targets, target - common.R_FIRST_SOURCE_BACK, side="right"))
    past = np.arange(newest)
    identity = str(analog.identity[run])
    candidate = days.usable[past] & np.isfinite(analog.temperature_c[past])
    if not np.isfinite(analog.temperature_c[run]):
        return None
    if identity:
        same = candidate & (analog.identity[past] == identity)
        if same.any():
            candidate = same
        else:
            candidate = candidate & (analog.identity[past] != "")
            if candidate.sum() < MIN_HOLIDAY_FALLBACK:
                return None
    else:
        gap = np.abs(analog.day_of_year[past] - analog.day_of_year[run])
        gap = np.minimum(gap, 365 - gap)
        candidate = (
            candidate
            & (analog.identity[past] == "")
            & (analog.day_type[past] == analog.day_type[run])
            & (gap <= 7 * spec.weeks)
        )
        if candidate.sum() < MIN_ANALOGS:
            return None
    chosen = past[candidate]
    distance = np.abs(analog.temperature_c[chosen] - analog.temperature_c[run])
    return chosen[np.lexsort((-days.targets[chosen], distance))[: spec.k]]


def analog_shift(
    *, frame: pl.DataFrame, forecast: np.ndarray, site: str, spec: AnalogSpec
) -> np.ndarray:
    """Return the ANALOG+R shift at every row: minus kappa times (A + weight * (W - A)).

    `A` is the 14-day profile and `W` the analog profile; a run with no analog set gets `-kappa * A`
    and a run whose 14-day profile has too few source days gets zero.
    """
    days = day_blocks(frame=frame, forecast=forecast)
    profile_a, ok = fourteen_day_profiles(days=days)
    analog = build_analog_days(site=site, days=days)
    by_run = np.zeros((len(days.targets), N_HALF_HOURS))
    for run in range(len(days.targets)):
        if not ok[run]:
            continue
        profile = profile_a[run]
        chosen = analog_days_of_run(analog=analog, run=run, spec=spec)
        if chosen is not None:
            profile = profile + spec.weight * (
                _mean_by_half_hour(blocks=days.blocks[chosen]) - profile
            )
        by_run[run] = -spec.kappa * profile
    return shift_at_rows(days=days, frame=frame, by_run=by_run, short_only=False)


def apply_shift_and_refit_spread(*, frame: pl.DataFrame, shift: np.ndarray) -> pl.DataFrame:
    """Return the frame with `shift` added to the mean and the spread refitted on the tuning runs.

    The quantiles are the corrected mean plus the centred residual table (hour band, calendar
    quarter, and lead band) of the corrected forecasts of the tuning runs outside the hand-filtered
    switching events, every spread factor 1, as experiment 136 builds them.

    Args:
        frame: The base forecast frame, with the `period` column.
        shift: The shift per row, in points.

    Returns:
        The corrected frame, with the same columns as `frame`.
    """
    mean = frame["mean"].to_numpy().astype(np.float64) + shift
    tuning = (
        (frame["period"] == "tuning").to_numpy()
        & ~frame["switching_mask"].to_numpy()
        & np.isfinite(frame["actual"].to_numpy())
    )
    band = frame["hour_band"].to_numpy().astype(np.int64)
    quarter = frame["quarter"].to_numpy().astype(np.int64)
    lead = frame["lead_hours"].to_numpy()
    actual = frame["actual"].to_numpy().astype(np.float64)
    spread = LeadHourSpread.fit(
        residuals=(actual - mean)[tuning],
        band=band[tuning],
        quarter=quarter[tuning],
        lead_hours=lead[tuning],
    )
    quantiles = spread.quantiles(mean=mean, band=band, quarter=quarter, lead_hours=lead)
    return frame.with_columns(
        pl.Series("mean", mean.astype(np.float32)),
        *[
            pl.Series(name, quantiles[:, position].astype(np.float32))
            for position, name in enumerate(common.QUANTILES)
        ],
    )
