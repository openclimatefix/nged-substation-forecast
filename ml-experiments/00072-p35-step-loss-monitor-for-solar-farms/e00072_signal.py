"""The daily signal of experiment 72: a farm's power over modelled power, divided by its neighbours.

**The signal is the logarithm of each farm's daily ratio of power to modelled output on clear,
unclipped hours, centred, minus the median of the other pool farms' centred ratios on the same
day.** A lost inverter scales a farm's output by a fixed share at every irradiance, so the ratio
steps down by that share. The farms share one 25 km satellite box, so subtracting the neighbours'
median cancels most of the satellite and physics-model error.

**The centring constants and the neighbours' median are computed once from the unmodified record and
never again.** An injected loss changes only the injected farm's own ratio, so a loss cannot move
any other farm's signal or its own neighbour median. The pool is farms A, B, C, D, and F. Farm E
is monitored but never in a pool.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from studies.capacity_estimators import (
    LOW_SUN_CLEAR_SKY_W_M2,
    CapacityRecord,
    valid_days,
)
from studies.sources import PER_STUDY_DIR

RECORD_PATH: Final[Path] = PER_STUDY_DIR / "effective_capacity" / "built" / "record.npz"
"""The capacity record the two capacity studies built (satellite and clear-sky hourly grid)."""

POOL_SITES: Final[tuple[str, ...]] = ("A", "B", "C", "D", "F")
"""The farms whose ratios form the neighbour median. Farm E is never in the pool."""

FARM_A_LOW_MONTHS: Final[tuple[str, ...]] = ("2024-12", "2025-03", "2026-04")
"""The calendar months in which farm A ran at 0.67 to 0.71 of its neighbours."""

CLEAR_SKY_INDEX_MINIMUM: Final[float] = 0.8
"""An hour is clear when satellite irradiance is at least this share of clear-sky irradiance."""

MODELLED_SHARE_RANGE: Final[tuple[float, float]] = (0.3, 0.85)
"""Hours keep a modelled output between these shares of the farm's 99th percentile of modelled
output, which leaves out the hours where an inverter clips."""

STUCK_RUN_HOURS: Final[int] = 6
"""A run of this many equal hourly readings is a stuck meter and is left out."""

MIN_SELECTED_HOURS: Final[int] = 3
"""A day needs this many selected hours to have a ratio."""

MIN_NEIGHBOURS: Final[int] = 2
"""A day has a signal only when at least this many other pool farms have a ratio."""


@dataclass(frozen=True)
class CleanBaseline:
    """What is computed once from the unmodified record.

    Attributes:
        centre: Each farm's median logarithmic ratio over the whole record, with farm A's low
            months masked, shape (farms,).
        neighbour: For each farm and day, the median centred ratio of the other pool farms (with
            farm A's low months masked), NaN where fewer than `MIN_NEIGHBOURS` have one; shape
            (farms, days).
        signal: The unmodified record's signal, shape (farms, days), NaN where undefined.
        low_months: Boolean, shape (days,); True on the days of farm A's three low months.
    """

    centre: np.ndarray
    neighbour: np.ndarray
    signal: np.ndarray
    low_months: np.ndarray


def load_record() -> CapacityRecord:
    """Read the capacity record, with the export-capped hours of farm E already masked.

    Returns:
        The record.
    """
    with np.load(RECORD_PATH) as data:
        return CapacityRecord(
            sites=tuple(str(site) for site in data["sites"]),
            dates=data["dates"].astype("datetime64[D]"),
            power_mw=data["power_mw"],
            clear_sky_w_m2=data["clear_sky_w_m2"],
            cams_ghi_w_m2=data["cams_ghi_w_m2"],
            cams_shape=data["cams_shape"],
            clear_shape=data["clear_shape"],
            static_p99_mw=data["static_p99_mw"],
        )


def _stuck_hours(*, power: np.ndarray) -> np.ndarray:
    """Return which hourly readings sit in a run of `STUCK_RUN_HOURS` or more equal positive values.

    Args:
        power: One farm's hourly power, shape (days, 24); runs are found along the flattened time.

    Returns:
        A boolean array of the same shape.
    """
    flat = power.reshape(-1)
    stuck = np.zeros(len(flat), dtype=bool)
    start = 0
    for index in range(1, len(flat) + 1):
        if index == len(flat) or not (np.isfinite(flat[index]) and flat[index] == flat[start]):
            if index - start >= STUCK_RUN_HOURS and np.isfinite(flat[start]) and flat[start] > 0:
                stuck[start:index] = True
            start = index
    return stuck.reshape(power.shape)


def farm_log_ratio(*, record: CapacityRecord, farm: int, power_mw: np.ndarray) -> np.ndarray:
    """Return one farm's logarithmic daily ratio of power to modelled output, uncentred.

    Args:
        record: The record, for the satellite irradiance and the modelled output.
        farm: The farm's index in the record.
        power_mw: The farm's hourly power to use, shape (days, 24); an injected loss changes this.

    Returns:
        An array of length days, NaN on an invalid day, a day with under `MIN_SELECTED_HOURS`
        selected hours, or a non-positive ratio.
    """
    modelled = record.cams_shape[farm]
    sunny = record.clear_sky_w_m2[farm] >= LOW_SUN_CLEAR_SKY_W_M2
    clear_index = np.where(
        sunny, record.cams_ghi_w_m2[farm] / np.where(sunny, record.clear_sky_w_m2[farm], 1.0), 0.0
    )
    ceiling = np.nanpercentile(modelled[np.isfinite(modelled) & (modelled > 0)], 99)
    low, high = MODELLED_SHARE_RANGE
    keep = (
        valid_days(record=record)[farm][:, None]
        & (clear_index >= CLEAR_SKY_INDEX_MINIMUM)
        & np.isfinite(modelled)
        & (modelled >= low * ceiling)
        & (modelled <= high * ceiling)
        & np.isfinite(power_mw)
        & ~_stuck_hours(power=power_mw)
    )
    top = np.where(keep, power_mw, 0.0).sum(axis=1)
    bottom = np.where(keep, modelled, 0.0).sum(axis=1)
    enough = keep.sum(axis=1) >= MIN_SELECTED_HOURS
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(enough & (bottom > 0), top / np.where(bottom > 0, bottom, 1.0), np.nan)
        return np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), np.nan)


def low_month_days(*, record: CapacityRecord) -> np.ndarray:
    """Return which days fall in farm A's three low months.

    Args:
        record: The record.

    Returns:
        A boolean array of length days.
    """
    months = record.dates.astype("datetime64[M]")
    return np.isin(months, np.array(FARM_A_LOW_MONTHS, dtype="datetime64[M]"))


def clean_baseline(*, record: CapacityRecord) -> CleanBaseline:
    """Compute the centring constants, the neighbour medians, and the unmodified signal.

    Args:
        record: The record with its unmodified power.

    Returns:
        The baseline.
    """
    farms = len(record.sites)
    low = low_month_days(record=record)
    farm_a = record.sites.index("A")
    uncentred = np.stack(
        [
            farm_log_ratio(record=record, farm=farm, power_mw=record.power_mw[farm])
            for farm in range(farms)
        ]
    )
    masked = uncentred.copy()
    masked[farm_a, low] = np.nan
    centre = np.array([np.nanmedian(row) if np.isfinite(row).any() else 0.0 for row in masked])
    centred_masked = masked - centre[:, None]
    pool = [record.sites.index(site) for site in POOL_SITES]
    neighbour = np.full(uncentred.shape, np.nan)
    for farm in range(farms):
        others = centred_masked[[index for index in pool if index != farm]]
        count = np.isfinite(others).sum(axis=0)
        with np.errstate(all="ignore"):
            median = np.nanmedian(np.where(np.isfinite(others), others, np.nan), axis=0)
        neighbour[farm] = np.where(count >= MIN_NEIGHBOURS, median, np.nan)
    signal = uncentred - centre[:, None] - neighbour
    return CleanBaseline(centre=centre, neighbour=neighbour, signal=signal, low_months=low)


def farm_signal(
    *, record: CapacityRecord, baseline: CleanBaseline, farm: int, power_mw: np.ndarray
) -> np.ndarray:
    """Return one farm's signal for a power series, using the unmodified neighbours.

    Args:
        record: The record.
        baseline: `clean_baseline`'s result.
        farm: The farm's index.
        power_mw: The farm's hourly power, shape (days, 24), possibly with an injected loss.

    Returns:
        An array of length days, NaN where the farm has no ratio or fewer than `MIN_NEIGHBOURS`
        pool neighbours have one.
    """
    ratio = farm_log_ratio(record=record, farm=farm, power_mw=power_mw)
    return ratio - baseline.centre[farm] - baseline.neighbour[farm]
