"""The daily signal of experiment 72: a farm's power over modelled power, divided by its neighbours.

**The signal is the logarithm of each farm's daily ratio of power to modelled output on clear,
unclipped hours, minus the median of the other farms' same-day ratios.** A lost inverter scales a
farm's output by a fixed share at every irradiance, so the ratio steps down by that share. The six
farms share one 25 km satellite box, so subtracting the neighbours' median cancels most of the
satellite and physics-model error. The signal is centred on each farm's median over the reference
year (the capacity constant within the season-year that experiment 45 recommended).
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np

from studies.capacity_estimators import (
    LOW_SUN_CLEAR_SKY_W_M2,
    CapacityRecord,
    reference_day_indices,
    valid_days,
)
from studies.sources import PER_STUDY_DIR

RECORD_PATH: Final[Path] = PER_STUDY_DIR / "effective_capacity" / "built" / "record.npz"
"""The capacity record the two capacity studies built (satellite and clear-sky hourly grid)."""

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
"""A day is divided by the neighbours only when at least this many other farms have a ratio."""


class DailySignal(NamedTuple):
    """The signal for every farm and day.

    Attributes:
        signal: Centred logarithmic double ratio, shape (farms, days), NaN where undefined.
        single_ratio_days: Boolean, shape (farms, days); True where fewer than `MIN_NEIGHBOURS`
            neighbours had a ratio, so `signal` is the single ratio.
    """

    signal: np.ndarray
    single_ratio_days: np.ndarray


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
    """Return which hourly readings sit in a run of `STUCK_RUN_HOURS` or more equal values.

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


def selected_hours(*, record: CapacityRecord, power_mw: np.ndarray) -> np.ndarray:
    """Return the hours each farm's daily ratio is built from.

    Args:
        record: The record, for the satellite and modelled output.
        power_mw: The power to use, shape (farms, days, 24); an injected loss changes this.

    Returns:
        A boolean array of shape (farms, days, 24).
    """
    valid = valid_days(record=record)
    sunny = record.clear_sky_w_m2 >= LOW_SUN_CLEAR_SKY_W_M2
    clear_index = np.where(
        sunny, record.cams_ghi_w_m2 / np.where(sunny, record.clear_sky_w_m2, 1.0), 0.0
    )
    selected = np.zeros(power_mw.shape, dtype=bool)
    for farm in range(power_mw.shape[0]):
        modelled = record.cams_shape[farm]
        ceiling = np.nanpercentile(modelled[np.isfinite(modelled) & (modelled > 0)], 99)
        low, high = MODELLED_SHARE_RANGE
        selected[farm] = (
            valid[farm][:, None]
            & (clear_index[farm] >= CLEAR_SKY_INDEX_MINIMUM)
            & np.isfinite(modelled)
            & (modelled >= low * ceiling)
            & (modelled <= high * ceiling)
            & np.isfinite(power_mw[farm])
            & ~_stuck_hours(power=power_mw[farm])
        )
    return selected


def log_ratio(*, record: CapacityRecord, power_mw: np.ndarray) -> np.ndarray:
    """Return each farm's logarithmic daily ratio of power to modelled output.

    Args:
        record: The record.
        power_mw: The power to use, shape (farms, days, 24).

    Returns:
        An array of shape (farms, days), NaN where a day has under `MIN_SELECTED_HOURS` hours or a
        non-positive ratio.
    """
    keep = selected_hours(record=record, power_mw=power_mw)
    top = np.where(keep, power_mw, 0.0).sum(axis=2)
    bottom = np.where(keep, record.cams_shape, 0.0).sum(axis=2)
    enough = keep.sum(axis=2) >= MIN_SELECTED_HOURS
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(enough & (bottom > 0), top / np.where(bottom > 0, bottom, 1.0), np.nan)
        return np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), np.nan)


def double_ratio(*, record: CapacityRecord, power_mw: np.ndarray) -> DailySignal:
    """Return the neighbour-normalised, reference-year-centred signal.

    Args:
        record: The record.
        power_mw: The power to use, shape (farms, days, 24).

    Returns:
        The signal. Neighbours are read from the same `power_mw`.
    """
    uncentred = log_ratio(record=record, power_mw=power_mw)
    reference = reference_day_indices(record=record)
    centre = np.array(
        [
            np.nanmedian(row[reference]) if np.isfinite(row[reference]).any() else 0.0
            for row in uncentred
        ]
    )
    single = uncentred - centre[:, None]
    neighbour = np.full(single.shape, np.nan)
    few = np.zeros(single.shape, dtype=bool)
    for farm in range(single.shape[0]):
        others = np.delete(single, farm, axis=0)
        count = np.isfinite(others).sum(axis=0)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(others, axis=0)
        enough = count >= MIN_NEIGHBOURS
        neighbour[farm] = np.where(enough, median, 0.0)
        few[farm] = ~enough
    return DailySignal(signal=single - neighbour, single_ratio_days=few & np.isfinite(single))


SignalBuilder = Callable[[np.ndarray], DailySignal]
"""A function from power of shape (farms, days, 24) to a `DailySignal`."""
