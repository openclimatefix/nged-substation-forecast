"""Six deterministic evidence tools that read a solar plant's record, for the study in issue #1147.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.
`build_evidence` turns one plant's record, up to a cut-off day, into the tables and series a
labeller reads (a language model or the tools-only rule in `studies.capacity_labeller`).

**The builder is blind by construction: it takes a record, a plant, a cut-off day, the plant's
fitted quantities, and an optional export-cap overlay, and nothing else.** It has no argument that
names a planted event, and it reads the record only up to `last_day`, so a case file for the causal
labeller cannot hold a number from a later day. The record's `static_p99_mw` is the one quantity
computed over the whole record, which the first capacity study treats as the plant's fixed power.

**The six tools:**

1. The clear-sky ratio: a day's power over modelled output, on hours whose satellite irradiance is
   at least 0.8 of the clear-sky irradiance, leaving out gap days and days with a stuck run of 6 or
   more equal readings. The commissioning ramp is removed when the record is built (see
   `studies.commissioning`), so it shows here as a gap.
2. The seasonal correction (`studies.capacity_seasonal`), as the shared and own terms beside the
   corrected ratio, and the 7-day neighbour ratio.
3. The hour-of-day shape, by month, with the monthly power envelope and clipping share.
4. The export-cap overlay, for a plant that has one.
5. Changepoint candidates on the corrected clear-sky ratio.
6. The tools-only labeller, in `studies.capacity_labeller`.
"""

import hashlib
import warnings
from dataclasses import dataclass, replace
from datetime import date
from typing import Final, Literal

import numpy as np

from studies.capacity_estimators import (
    HOURS_PER_DAY,
    LOW_SUN_CLEAR_SKY_W_M2,
    MIN_SHAPE_SUM,
    REFERENCE_END,
    REFERENCE_START,
    CapacityRecord,
    neighbour_ratio,
    valid_days,
)
from studies.capacity_seasonal import seasonal_reference

CLEAR_SKY_INDEX_MINIMUM: Final[float] = 0.8
"""An hour is clear when the satellite irradiance is at least this share of the clear-sky value."""

MIN_CLEAR_HOURS: Final[int] = 3
"""A day needs this many clear hours to give a clear-sky ratio."""

STUCK_RUN_HOURS: Final[int] = 6
"""A run of this many equal positive readings in a row marks a stuck day."""

NORMALISER_FALLBACK_DAYS: Final[int] = 60
"""A plant with no clear day in the reference year is normalised on its first clear days instead."""

IRRADIANCE_BIN_EDGES: Final[tuple[float, ...]] = (
    100.0,
    200.0,
    300.0,
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
    1000.0,
)
"""The satellite-irradiance bins (watts per square metre) of the envelope's 95th percentiles."""

ENVELOPE_MIN_HOURS: Final[int] = 5
"""An envelope bin needs this many hours in a month."""

CLIP_TOLERANCE: Final[float] = 0.02
"""An hour is at the plateau when its power is within this share of the month's 99th percentile."""

HOUR_TABLE_MIN_DAYS: Final[int] = 3
"""A month's hour-of-day value needs this many days."""

CHANGEPOINT_MIN_SEGMENT: Final[int] = 7
"""The fewest clear-sky observations in a segment either side of a changepoint."""

CHANGEPOINT_PERMUTATIONS: Final[int] = 300
"""How many shuffles give a changepoint's p-value."""

CHANGEPOINT_MAX_P: Final[float] = 0.05
"""A candidate with a larger permutation p-value is not listed."""

CHANGEPOINT_MAX_CANDIDATES: Final[int] = 8
"""The most candidates listed per case file."""

UNCONFIRMED_DAYS: Final[int] = 14
"""A candidate this close to the cut-off day is flagged unconfirmed."""

DirectionType = Literal["down", "up"]
"""The sign of a changepoint's size."""


@dataclass(frozen=True)
class PlantFit:
    """A plant's fitted quantities, from the first study's `orientation.json`.

    Attributes:
        ac_capacity_mw: The fitted alternating-current capacity.
        dc_capacity_mw: The fitted direct-current capacity.
        tilt_deg: The fitted panel tilt.
        azimuth_deg: The fitted panel azimuth.
    """

    ac_capacity_mw: float
    dc_capacity_mw: float
    tilt_deg: float
    azimuth_deg: float


@dataclass(frozen=True)
class CapOverlay:
    """An export cap on the record's hourly grid.

    Attributes:
        cap_mw: The cap in megawatts, shape (days, 24), NaN where no cap was recorded.
        binding: Whether the cap was below the connection limit, shape (days, 24).
    """

    cap_mw: np.ndarray
    binding: np.ndarray


@dataclass(frozen=True)
class Changepoint:
    """One changepoint candidate on the corrected clear-sky ratio.

    Attributes:
        day_index: The index of the first clear-sky observation after the change.
        day: That observation's date.
        direction: `down` or `up`.
        size: The fractional change of the segment mean, for example -0.12.
        p_value: The permutation p-value of the split.
        unconfirmed: Whether the candidate lies within `UNCONFIRMED_DAYS` of the cut-off day.
    """

    day_index: int
    day: date
    direction: DirectionType
    size: float
    p_value: float
    unconfirmed: bool


@dataclass(frozen=True)
class Evidence:
    """Everything the tools measure for one plant up to a cut-off day.

    Daily arrays have one entry per day of the cut record. The monthly arrays have one row per
    calendar month the cut record touches, in `months` order.

    Attributes:
        plant: The plant's label.
        dates: The days, as `datetime64[D]`.
        static_p99_mw: The plant's fixed 99th-percentile power.
        fit: The plant's fitted quantities.
        gap: Whether the day is a gap (too little of its clear-sky energy observed).
        stuck: Whether the day holds a stuck run.
        clear_raw: The daily clear-sky ratio over its normaliser, NaN where unusable.
        clear_corrected: The same after the seasonal correction.
        shared: The shared term as a factor, NaN where undefined.
        own: The own-profile term as a factor, NaN where undefined.
        neighbour: The 7-day neighbour ratio, NaN where null.
        energy_mwh: The day's measured energy, NaN where the day is a gap.
        modelled_mwh: The day's modelled energy at the fitted direct-current capacity, NaN likewise.
        cap_binding_share: The share of daylight hours the export cap binds, or `None` without
            an overlay.
        cap_mean_mw: The mean cap over daylight hours with a recorded cap, or `None`.
        months: The `YYYY-MM` labels.
        hour_table: The corrected ratio by month and hour, shape (months, 24), NaN where thin.
        envelope: The 95th percentile of power over static power per month and irradiance bin.
        plateau: The 99th percentile of power over static power per month.
        clip_share: The share of daylight hours within `CLIP_TOLERANCE` of that plateau per month.
        changepoints: The candidates, newest last.
    """

    plant: str
    dates: np.ndarray
    static_p99_mw: float
    fit: PlantFit
    gap: np.ndarray
    stuck: np.ndarray
    clear_raw: np.ndarray
    clear_corrected: np.ndarray
    shared: np.ndarray
    own: np.ndarray
    neighbour: np.ndarray
    energy_mwh: np.ndarray
    modelled_mwh: np.ndarray
    cap_binding_share: np.ndarray | None
    cap_mean_mw: np.ndarray | None
    months: tuple[str, ...]
    hour_table: np.ndarray
    envelope: np.ndarray
    plateau: np.ndarray
    clip_share: np.ndarray
    changepoints: tuple[Changepoint, ...]


def truncate(*, record: CapacityRecord, last_day: int) -> CapacityRecord:
    """Return the record up to and including day index `last_day`.

    Args:
        record: The record.
        last_day: The latest day index kept.

    Returns:
        The shorter record.
    """
    end = last_day + 1
    return replace(
        record,
        dates=record.dates[:end],
        power_mw=record.power_mw[:, :end],
        clear_sky_w_m2=record.clear_sky_w_m2[:, :end],
        cams_ghi_w_m2=record.cams_ghi_w_m2[:, :end],
        cams_shape=record.cams_shape[:, :end],
        clear_shape=record.clear_shape[:, :end],
    )


def stuck_days(*, power: np.ndarray) -> np.ndarray:
    """Return which days hold a run of equal positive readings that is long enough to be stuck.

    Args:
        power: One plant's hourly power, shape (days, 24), NaN where missing.

    Returns:
        A boolean array of shape (days,). A run may cross midnight and marks every day it touches.
    """
    flat = power.reshape(-1)
    days = len(power)
    marked = np.zeros(days, dtype=bool)
    run_start = 0
    for index in range(1, len(flat) + 1):
        same = (
            index < len(flat)
            and np.isfinite(flat[index])
            and flat[index] > 0
            and flat[index] == flat[run_start]
        )
        if same:
            continue
        if (
            index - run_start >= STUCK_RUN_HOURS
            and np.isfinite(flat[run_start])
            and flat[run_start] > 0
        ):
            marked[run_start // HOURS_PER_DAY : (index - 1) // HOURS_PER_DAY + 1] = True
        run_start = index
    return marked


def _usable_hours(*, record: CapacityRecord, plant: int, bad_day: np.ndarray) -> np.ndarray:
    """Return the hours with sun, a measured power, a modelled output, and a good day."""
    return (
        (record.clear_sky_w_m2[plant] >= LOW_SUN_CLEAR_SKY_W_M2)
        & np.isfinite(record.power_mw[plant])
        & np.isfinite(record.cams_shape[plant])
        & ~bad_day[:, None]
    )


def clear_sky_ratio(
    *, record: CapacityRecord, plant: int, bad_day: np.ndarray, log_reference: np.ndarray
) -> np.ndarray:
    """Return each day's power over modelled output on clear hours, divided by a reference.

    Args:
        record: The record.
        plant: The plant's index.
        bad_day: Days to leave out (gaps and stuck days), shape (days,).
        log_reference: The logarithm of the factor to divide by, shape (days,); 0 for none.

    Returns:
        One value per day, NaN where the day has fewer than `MIN_CLEAR_HOURS` clear hours or too
        little modelled output.
    """
    usable = _usable_hours(record=record, plant=plant, bad_day=bad_day)
    with np.errstate(invalid="ignore", divide="ignore"):
        index = record.cams_ghi_w_m2[plant] / record.clear_sky_w_m2[plant]
    clear = usable & (index >= CLEAR_SKY_INDEX_MINIMUM)
    top = np.where(clear, record.power_mw[plant], 0.0).sum(axis=1)
    bottom = np.where(clear, record.cams_shape[plant], 0.0).sum(axis=1)
    enough = (clear.sum(axis=1) >= MIN_CLEAR_HOURS) & (bottom >= MIN_SHAPE_SUM)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(enough, top / np.where(bottom > 0, bottom, np.nan), np.nan)
        return ratio / np.exp(log_reference)


def _normaliser_days(*, record: CapacityRecord, values: np.ndarray) -> np.ndarray:
    """Return the day indices whose finite `values` define the plant's normal level.

    Args:
        record: The record.
        values: One plant's daily values, shape (days,).

    Returns:
        The reference year's days that have a value, else the first `NORMALISER_FALLBACK_DAYS`
        days that do.
    """
    days = np.arange(len(values))
    in_year = (record.dates >= np.datetime64(REFERENCE_START)) & (
        record.dates < np.datetime64(REFERENCE_END)
    )
    chosen = days[in_year & np.isfinite(values)]
    if len(chosen):
        return chosen
    return days[np.isfinite(values)][:NORMALISER_FALLBACK_DAYS]


def normalise(*, record: CapacityRecord, values: np.ndarray) -> np.ndarray:
    """Divide a daily series by its median over the plant's normaliser days.

    Args:
        record: The record.
        values: One plant's daily values.

    Returns:
        The series, 1 at the plant's normal level; unchanged where no day defines the level.
    """
    days = _normaliser_days(record=record, values=values)
    if len(days) == 0:
        return values
    return values / float(np.median(values[days]))


def hour_of_day_table(
    *,
    record: CapacityRecord,
    plant: int,
    bad_day: np.ndarray,
    log_reference: np.ndarray,
    months: np.ndarray,
) -> np.ndarray:
    """Return the monthly median of the corrected hourly ratio, per hour of the day.

    Args:
        record: The record.
        plant: The plant's index.
        bad_day: Days to leave out.
        log_reference: The log of the seasonal factor per day.
        months: The `datetime64[M]` month of each day.

    Returns:
        An array of shape (months, 24) normalised per hour on the plant's normaliser days, NaN
        where a month has fewer than `HOUR_TABLE_MIN_DAYS` days at that hour.
    """
    usable = _usable_hours(record=record, plant=plant, bad_day=bad_day) & (
        record.cams_shape[plant] >= 0.02
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (
            np.where(
                usable,
                record.power_mw[plant] / np.where(usable, record.cams_shape[plant], 1.0),
                np.nan,
            )
            / np.exp(log_reference)[:, None]
        )
    days = _normaliser_days(record=record, values=np.where(usable.any(axis=1), 1.0, np.nan))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        normal = np.nanmedian(ratio[days], axis=0) if len(days) else np.full(HOURS_PER_DAY, np.nan)
        unique_months = np.unique(months)
        table = np.full((len(unique_months), HOURS_PER_DAY), np.nan)
        for row, month in enumerate(unique_months):
            block = ratio[months == month]
            enough = np.isfinite(block).sum(axis=0) >= HOUR_TABLE_MIN_DAYS
            table[row] = np.where(enough, np.nanmedian(block, axis=0) / normal, np.nan)
    return table


def power_envelope(
    *, record: CapacityRecord, plant: int, bad_day: np.ndarray, months: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the monthly power envelope, plateau, and share of hours on the plateau.

    Args:
        record: The record.
        plant: The plant's index.
        bad_day: Days to leave out.
        months: The `datetime64[M]` month of each day.

    Returns:
        The 95th percentile of power over static power per month and irradiance bin, shape
        (months, bins); the 99th percentile per month; and the share of daylight hours within
        `CLIP_TOLERANCE` of that plateau per month. NaN where a month has too few hours.
    """
    usable = _usable_hours(record=record, plant=plant, bad_day=bad_day)
    static = record.static_p99_mw[plant]
    unique_months = np.unique(months)
    edges = np.asarray(IRRADIANCE_BIN_EDGES)
    envelope = np.full((len(unique_months), len(edges) - 1), np.nan)
    plateau = np.full(len(unique_months), np.nan)
    clip = np.full(len(unique_months), np.nan)
    for row, month in enumerate(unique_months):
        selected = usable & (months == month)[:, None]
        power = record.power_mw[plant][selected] / static
        ghi = record.cams_ghi_w_m2[plant][selected]
        if len(power) < ENVELOPE_MIN_HOURS:
            continue
        plateau[row] = np.quantile(power, 0.99)
        clip[row] = float(np.mean(power >= plateau[row] * (1.0 - CLIP_TOLERANCE)))
        for column in range(len(edges) - 1):
            in_bin = (ghi >= edges[column]) & (ghi < edges[column + 1])
            if in_bin.sum() >= ENVELOPE_MIN_HOURS:
                envelope[row, column] = np.quantile(power[in_bin], 0.95)
    return envelope, plateau, clip


def cap_daily(
    *, record: CapacityRecord, plant: int, cap: CapOverlay
) -> tuple[np.ndarray, np.ndarray]:
    """Return the share of daylight hours the cap binds and the mean cap, per day.

    Args:
        record: The record.
        plant: The plant's index.
        cap: The plant's overlay, cut to the same days.

    Returns:
        The binding share and the mean cap in megawatts over daylight hours with a recorded cap,
        both shape (days,), NaN where undefined.
    """
    daylight = record.clear_sky_w_m2[plant] >= LOW_SUN_CLEAR_SKY_W_M2
    with np.errstate(all="ignore"):
        share = np.where(
            daylight.any(axis=1),
            (cap.binding & daylight).sum(axis=1) / daylight.sum(axis=1),
            np.nan,
        )
        recorded = daylight & np.isfinite(cap.cap_mw)
        mean_cap = np.where(
            recorded.any(axis=1),
            np.where(recorded, cap.cap_mw, 0.0).sum(axis=1) / recorded.sum(axis=1),
            np.nan,
        )
    return share, mean_cap


def _best_split(*, values: np.ndarray) -> tuple[int, float]:
    """Return the split index that most reduces the squared error, and the reduction."""
    count = len(values)
    cumulative = np.cumsum(values)
    firsts = np.arange(CHANGEPOINT_MIN_SEGMENT, count - CHANGEPOINT_MIN_SEGMENT + 1)
    mean_before = cumulative[firsts - 1] / firsts
    mean_after = (cumulative[-1] - cumulative[firsts - 1]) / (count - firsts)
    gain = firsts * (count - firsts) / count * (mean_before - mean_after) ** 2
    best = int(np.argmax(gain))
    return int(firsts[best]), float(gain[best])


def _permutation_p(*, values: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """Return the share of shuffles of `values` whose best split is at least `observed`."""
    count = len(values)
    firsts = np.arange(CHANGEPOINT_MIN_SEGMENT, count - CHANGEPOINT_MIN_SEGMENT + 1)
    shuffled = np.stack([rng.permutation(values) for _ in range(CHANGEPOINT_PERMUTATIONS)])
    cumulative = np.cumsum(shuffled, axis=1)
    mean_before = cumulative[:, firsts - 1] / firsts
    mean_after = (cumulative[:, -1:] - cumulative[:, firsts - 1]) / (count - firsts)
    gain = (firsts * (count - firsts) / count * (mean_before - mean_after) ** 2).max(axis=1)
    return float((1 + np.sum(gain >= observed)) / (1 + CHANGEPOINT_PERMUTATIONS))


def find_changepoints(*, dates: np.ndarray, values: np.ndarray) -> tuple[Changepoint, ...]:
    """List changepoint candidates in a daily series by binary segmentation.

    Each segment is split at the point that most reduces the squared error of two flat levels, and
    the split is kept when its permutation p-value is at most `CHANGEPOINT_MAX_P`. The scan is a
    CUSUM of the demeaned series and the two-segment fit is the same statistic.

    Args:
        dates: One date per entry of `values`.
        values: The corrected clear-sky ratio, NaN where unusable.

    Returns:
        The candidates in date order, at most `CHANGEPOINT_MAX_CANDIDATES`, the most significant
        ones when there are more. A candidate within `UNCONFIRMED_DAYS` of the last date is marked
        unconfirmed.
    """
    index = np.flatnonzero(np.isfinite(values))
    series = values[index]
    seed = int.from_bytes(hashlib.sha256(series.tobytes()).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    found: list[tuple[float, Changepoint]] = []
    pending = [(0, len(series))]
    while pending and len(found) < CHANGEPOINT_MAX_CANDIDATES:
        start, stop = pending.pop()
        segment = series[start:stop]
        if len(segment) < 2 * CHANGEPOINT_MIN_SEGMENT:
            continue
        split, gain = _best_split(values=segment)
        p_value = _permutation_p(values=segment, observed=gain, rng=rng)
        if p_value > CHANGEPOINT_MAX_P:
            continue
        before, after = float(segment[:split].mean()), float(segment[split:].mean())
        size = after / before - 1.0
        day_index = int(index[start + split])
        found.append(
            (
                p_value,
                Changepoint(
                    day_index=day_index,
                    day=dates[day_index].astype("datetime64[D]").item(),
                    direction="down" if size < 0 else "up",
                    size=float(size),
                    p_value=p_value,
                    unconfirmed=bool((dates[-1] - dates[day_index]).astype(int) < UNCONFIRMED_DAYS),
                ),
            )
        )
        pending += [(start, start + split), (start + split, stop)]
    return tuple(sorted((change for _, change in found), key=lambda change: change.day_index))


def build_evidence(
    *,
    record: CapacityRecord,
    plant: int,
    last_day: int,
    fit: PlantFit,
    cap: CapOverlay | None,
) -> Evidence:
    """Run the tools on one plant, reading the record up to and including `last_day`.

    Args:
        record: The record (planted or original; the builder cannot tell).
        plant: The plant's index.
        last_day: The latest day index the tools may read. For the causal labeller this is the issue
            day minus 2; for the hindsight labeller it is the record's last day.
        fit: The plant's fitted quantities.
        cap: The plant's export-cap overlay on the full record's grid, or `None` for a plant with
            no recorded cap.

    Returns:
        The evidence.
    """
    cut = truncate(record=record, last_day=last_day)
    days = len(cut.dates)
    reference = seasonal_reference(record=cut)
    log_reference = reference.log_reference[plant]
    gap = ~valid_days(record=cut)[plant]
    stuck = stuck_days(power=cut.power_mw[plant])
    bad_day = gap | stuck
    clear_raw = normalise(
        record=cut,
        values=clear_sky_ratio(
            record=cut, plant=plant, bad_day=bad_day, log_reference=np.zeros(days)
        ),
    )
    clear_corrected = normalise(
        record=cut,
        values=clear_sky_ratio(
            record=cut, plant=plant, bad_day=bad_day, log_reference=log_reference
        ),
    )
    months = cut.dates.astype("datetime64[M]")
    envelope, plateau, clip = power_envelope(
        record=cut, plant=plant, bad_day=bad_day, months=months
    )
    cap_share, cap_mean = (None, None)
    if cap is not None:
        cut_cap = CapOverlay(cap_mw=cap.cap_mw[:days], binding=cap.binding[:days])
        cap_share, cap_mean = cap_daily(record=cut, plant=plant, cap=cut_cap)
    return Evidence(
        plant=cut.sites[plant],
        dates=cut.dates,
        static_p99_mw=float(cut.static_p99_mw[plant]),
        fit=fit,
        gap=gap,
        stuck=stuck,
        clear_raw=clear_raw,
        clear_corrected=clear_corrected,
        shared=np.exp(reference.shared[plant]),
        own=np.exp(reference.own[plant]),
        neighbour=neighbour_ratio(record=cut)[plant],
        energy_mwh=np.where(gap, np.nan, np.nansum(cut.power_mw[plant], axis=1)),
        modelled_mwh=np.where(
            gap, np.nan, np.nansum(cut.cams_shape[plant], axis=1) * fit.dc_capacity_mw
        ),
        cap_binding_share=cap_share,
        cap_mean_mw=cap_mean,
        months=tuple(str(month) for month in np.unique(months)),
        hour_table=hour_of_day_table(
            record=cut, plant=plant, bad_day=bad_day, log_reference=log_reference, months=months
        ),
        envelope=envelope,
        plateau=plateau,
        clip_share=clip,
        changepoints=find_changepoints(dates=cut.dates, values=clear_corrected),
    )
