"""Analog-day error profiles on top of GAM-K, for experiment 145.

**Every arm is K0's level and transient plus a profile shift.** For a run with origin on UTC day
`O`, the target is the UTC day `T = O + 1`. A source day is a target day `S <= T - 2`: its D+1 block
came from the run of day `S - 1`, and the day `S` ended before the origin, so it is fully observed.
Each source day's block is K0's median forecast minus the reading, by local half-hour, with the
day's mean removed (as in experiment 115's recipe R).

Recipe R's profile `A` averages the usable source days of the last 14 (`T - 15` to `T - 2`). The
analog profile `W` averages the `k` source days that look most like the target:

- **Class.** For a non-holiday target, a non-holiday source day of the same day type. For a
  holiday-like target, a source day with the same holiday identity, else any holiday-like source
  day when at least 2 exist.
- **Season.** For a non-holiday target, a source day within `weeks` weeks of the target's day of the
  year (circular). A holiday-like target has no season window.
- **Nearest.** The source days whose forecast daily mean temperature (the ENS ensemble-mean
  temperature the source day's own run saw, averaged over the 48 half-hours of its D+1 block) is
  closest to the target's, ties going to the more recent day.

The shift is `-kappa * (A + weight * (W - A))`, with `W = A` where no analog set exists.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final

import e00145_common as common
import e00145_site as site_module
import numpy as np
from studies.dp_calendar import bank_holiday_names

PROFILE_DAYS: Final[int] = 14
MIN_PROFILE_DAYS: Final[int] = 3
MIN_DAY_READINGS: Final[int] = 40
FIRST_SOURCE_BACK: Final[int] = 2
"""The newest source day is `T - 2`: the target day `T - 1` is the origin's own day."""
MIN_ANALOGS: Final[int] = 3
"""A non-holiday target needs at least this many candidate days, otherwise it gets `A`."""
MIN_HOLIDAY_FALLBACK: Final[int] = 2
"""A holiday-like target with no earlier occurrence of its own identity needs this many earlier
holiday-like days of any identity, otherwise it gets `A`."""
DAY_TYPES: Final[tuple[int, ...]] = (0, 1, 1, 1, 2, 3, 4)
"""The day type of Monday (0) to Sunday (6): Monday, Tuesday to Thursday, Friday, Saturday, or
Sunday."""
DECEMBER_RUN_DAYS: Final[tuple[int, ...]] = (24, 27, 28, 29, 30, 31)
EPOCH: Final[np.datetime64] = np.datetime64("1970-01-01", "D")
EPOCH_DATE: Final[date] = date(1970, 1, 1)
D1_HALF_HOURS: Final[int] = 48


def holiday_identities(*, first: date, last: date) -> dict[date, str]:
    """Return the holiday-like days between two dates, each with its identity.

    A holiday-like day is an English bank holiday (identity: its name), Easter Saturday or Sunday,
    or 24 or 27 to 31 December (identity: `December <day>`, except that a bank holiday keeps its
    name).
    """
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


@dataclass(frozen=True)
class AnalogSpec:
    """How an analog profile shift is built.

    Attributes:
        kappa: The shrink applied to the profile.
        k: The number of analog days averaged.
        weeks: The season window, in weeks either side of the target's day of the year.
        weight: The weight of the analog profile against `A` (1 for the pure analog arm).
        temperature: True to rank candidates by forecast temperature, False to take the most recent.
        holiday_only: True to use the analog profile on holiday-like targets only and `A` elsewhere.
        placebo_shift: Days added to the target's date before reading its day type (the gate's
            placebo uses 3, so a Saturday target reads Tuesday-to-Thursday days).
    """

    kappa: float
    k: int
    weeks: int
    weight: float = 1.0
    temperature: bool = True
    holiday_only: bool = False
    placebo_shift: int = 0


@dataclass(frozen=True)
class AnalogDays:
    """One site's source days, one per run (a run's D+1 block is one target day).

    Attributes:
        targets: Each run's target day number (days since 1970-01-01), ascending.
        blocks: Each run's centred D+1 error block, shape (runs, 2, 48): the sum and count of each
            local half-hour's centred errors, zero for an unusable run.
        usable: True where the run's block has at least `MIN_DAY_READINGS` readings.
        temperature_c: The ENS ensemble-mean temperature of the run's D+1 block, daily mean, `NaN`
            where the run has none.
        day_type: The day type of each target day.
        day_of_year: The day of the year of each target day.
        identity: The holiday identity of each target day, empty for a day that is not holiday-like.
        profile_a: Recipe R's 14-day profile at each run, shape (runs, 48).
        profile_a_ok: True where `profile_a` has at least `MIN_PROFILE_DAYS` source days.
    """

    targets: np.ndarray
    blocks: np.ndarray
    usable: np.ndarray
    temperature_c: np.ndarray
    day_type: np.ndarray
    day_of_year: np.ndarray
    identity: np.ndarray
    profile_a: np.ndarray
    profile_a_ok: np.ndarray


def day_blocks(
    *, data: site_module.SiteData, k0_q50: np.ndarray, actual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return each run's centred D+1 error block by local half-hour, and its target day number.

    Args:
        data: The site data.
        k0_q50: K0's median forecast at every row.
        actual: The reading at every row (`NaN` where there is none); may be an injected copy.

    Returns:
        Blocks of shape (runs, 2, 48): the sum and the count of each local half-hour's centred
        errors (a clock-change day holds one half-hour twice), zero for an unusable run, and the UTC
        day number (days since 1970-01-01) of each run's target day.
    """
    d1 = data.d1_rows()
    error = k0_q50 - actual
    blocks = np.zeros((len(data.runs), 2, 48))
    origin_days = data.run_origins.astype("datetime64[D]")
    for run in range(len(data.runs)):
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        rows = rows[d1[rows]]
        if len(rows) != D1_HALF_HOURS:
            continue
        block = error[rows]
        if int(np.isfinite(block).sum()) < MIN_DAY_READINGS:
            continue
        centred = block - np.nanmean(block)
        finite = np.isfinite(centred)
        np.add.at(blocks[run, 0], data.row_half_hour[rows][finite], centred[finite])
        np.add.at(blocks[run, 1], data.row_half_hour[rows][finite], 1.0)
    targets = (origin_days + np.timedelta64(1, "D") - EPOCH).astype(np.int64)
    return blocks, targets


def forecast_temperature(*, data: site_module.SiteData) -> np.ndarray:
    """Return each run's D+1 daily mean ENS ensemble-mean temperature, `NaN` where it has none.

    The temperature is the raw ENS value of experiment 111's store, the weather the run saw, so it
    exists at forecast time. Row `i` of a run's window is the half-hour `i + 1` after the origin's.
    """
    with np.load(common.STORE_DIR / f"{data.site}.npz") as stored:
        temperature = stored["temperature"]
        origins = stored["origins"]
    first_position = np.searchsorted(data.cache.end_times, origins) + 1
    d1 = data.d1_rows()
    out = np.full(len(data.runs), np.nan)
    for run, index in enumerate(data.runs):
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        if data.position[rows[0]] != first_position[index]:
            msg = f"run {index} of {data.site} starts at an unexpected grid position"
            raise ValueError(msg)
        local = np.flatnonzero(d1[rows])
        if len(local) != D1_HALF_HOURS or local.max() >= temperature.shape[2]:
            continue
        values = temperature[index][:, local]
        out[run] = np.nanmean(values.mean(axis=0)) if np.isfinite(values).any() else np.nan
    return out


def _mean_by_half_hour(*, blocks: np.ndarray) -> np.ndarray:
    """Return the mean over blocks of shape (n, 2, 48), by half-hour; zero where none counts."""
    total = blocks[:, 0].sum(axis=0)
    count = blocks[:, 1].sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def all_days_profiles(*, blocks: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return recipe R's 14-day profile at each run, and whether it has enough source days.

    A source day is `T - back` for `back` in 2 to 15, counted whether or not it is a holiday.
    """
    index_of = {int(day): i for i, day in enumerate(targets)}
    usable = blocks[:, 1].sum(axis=1) > 0
    out = np.zeros((len(targets), 48))
    ok = np.zeros(len(targets), dtype=bool)
    for run, target in enumerate(targets.tolist()):
        chosen = [
            i
            for back in range(FIRST_SOURCE_BACK, PROFILE_DAYS + FIRST_SOURCE_BACK)
            if (i := index_of.get(target - back)) is not None and usable[i]
        ]
        if len(chosen) >= MIN_PROFILE_DAYS:
            out[run] = _mean_by_half_hour(blocks=blocks[chosen])
            ok[run] = True
    return out, ok


def build_days(
    *, data: site_module.SiteData, k0_q50: np.ndarray, actual: np.ndarray, temperature_c: np.ndarray
) -> AnalogDays:
    """Return a site's source days for the given readings (real, or an injected copy)."""
    blocks, targets = day_blocks(data=data, k0_q50=k0_q50, actual=actual)
    weekday = (targets + 3) % 7  # 1970-01-01 was a Thursday; Monday is 0
    dates = [EPOCH_DATE + timedelta(days=int(t)) for t in targets]
    names = holiday_identities(first=dates[0], last=dates[-1])
    profile_a, profile_a_ok = all_days_profiles(blocks=blocks, targets=targets)
    return AnalogDays(
        targets=targets,
        blocks=blocks,
        usable=blocks[:, 1].sum(axis=1) > 0,
        temperature_c=temperature_c,
        day_type=np.array([DAY_TYPES[w] for w in weekday]),
        day_of_year=np.array([d.timetuple().tm_yday for d in dates]),
        identity=np.array([names.get(d, "") for d in dates]),
        profile_a=profile_a,
        profile_a_ok=profile_a_ok,
    )


def analog_profile(*, days: AnalogDays, run: int, spec: AnalogSpec) -> np.ndarray | None:
    """Return the mean block of the analog days of one run's target, or `None` if there is no set.

    Args:
        days: The site's source days.
        run: The index of the run (its target is `days.targets[run]`).
        spec: The setting; `weight` and `kappa` are not used here.

    Returns:
        The mean centred error by local half-hour of the `spec.k` closest candidates, or `None`.
    """
    target = int(days.targets[run])
    newest = int(np.searchsorted(days.targets, target - FIRST_SOURCE_BACK, side="right"))
    past = np.arange(newest)
    identity = str(days.identity[run])
    if spec.holiday_only and not identity:
        return None
    candidate = days.usable[past]
    if spec.temperature:
        candidate = candidate & np.isfinite(days.temperature_c[past])
        if not np.isfinite(days.temperature_c[run]):
            return None
    if identity:
        same = candidate & (days.identity[past] == identity)
        if same.any():
            candidate = same
        else:
            candidate = candidate & (days.identity[past] != "")
            if candidate.sum() < MIN_HOLIDAY_FALLBACK:
                return None
    else:
        wanted = DAY_TYPES[(target + spec.placebo_shift + 3) % 7]
        gap = np.abs(days.day_of_year[past] - days.day_of_year[run])
        gap = np.minimum(gap, 365 - gap)
        candidate = (
            candidate
            & (days.identity[past] == "")
            & (days.day_type[past] == wanted)
            & (gap <= 7 * spec.weeks)
        )
        if candidate.sum() < MIN_ANALOGS:
            return None
    chosen = past[candidate]
    distance = (
        np.abs(days.temperature_c[chosen] - days.temperature_c[run])
        if spec.temperature
        else np.zeros(len(chosen))
    )
    nearest = chosen[np.lexsort((-days.targets[chosen], distance))[: spec.k]]
    return _mean_by_half_hour(blocks=days.blocks[nearest])


def analog_shift_by_run(*, days: AnalogDays, spec: AnalogSpec) -> np.ndarray:
    """Return each run's shift by local half-hour, shape (runs, 48): `-kappa` times the profile.

    A run whose 14-day profile has fewer than `MIN_PROFILE_DAYS` source days gets zeros, as recipe
    R does. A run with no analog set gets `-kappa * A`.
    """
    out = np.zeros((len(days.targets), 48))
    for run in range(len(days.targets)):
        if not days.profile_a_ok[run]:
            continue
        profile = days.profile_a[run]
        analog = analog_profile(days=days, run=run, spec=spec)
        if analog is not None:
            profile = profile + spec.weight * (analog - profile)
        out[run] = -spec.kappa * profile
    return out


def shift_at_rows(*, data: site_module.SiteData, by_run: np.ndarray) -> np.ndarray:
    """Spread each run's shift per half-hour to the run's rows (every lead)."""
    return by_run[data.run_of_row, data.row_half_hour]
