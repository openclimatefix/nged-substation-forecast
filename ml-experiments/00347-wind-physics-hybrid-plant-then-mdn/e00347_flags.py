"""The Stage 3b flagging rule: days on which a farm made much less than its plant expects.

The rule is fixed before any pass-2 result is read. For one plant and one farm it forms, for every
half-hour whose expected power lies between 25% and 100% of the plant's flat top (experiment 210's
idle-band rows excluded), the log ratio of observed to expected power (observed power floored at 2%
of the flat top, so a stopped farm gives a finite number). Each half-hour is weighted by the
inverse of its variance `s(v)^2 * sigma_v^2 + sigma_p^2`, where `s` is the local slope of the log
power curve with respect to the log wind speed, `sigma_v` is the relative error of the ERA5 speed
and `sigma_p` is the farm's own power noise; both are estimated per farm from the spread of the
log ratios by slope bin. Half-hours are averaged into one weighted log ratio per calendar day (days
with fewer than `MIN_HALF_HOURS` such half-hours are unscored). A day is flagged when its ratio is
below the median of all scored daily ratios by more than `k` robust standard deviations (1.4826
times the median absolute deviation). A block is a maximal run of consecutive flagged days, so its
length is at least 24 hours. Separately, any run of at least 12 hours at the idle floor while the
ERA5 100 m speed exceeds the plant's rated speed is flagged (the experiment 210 outage rule).
"""

from dataclasses import dataclass
from typing import Final

import e00347_common as common
import e00347_plant as plant
import numpy as np

EXPECTED_RANGE: Final[tuple[float, float]] = (0.25, 1.0)
FLOOR_SHARE: Final[float] = 0.02
MIN_HALF_HOURS: Final[int] = 6
SLOPE_BINS: Final[int] = 12
IDLE_RUN_HALF_HOURS: Final[int] = 24
IDLE_SPEED_M_S: Final[float] = 2.0
ROBUST_SD_FACTOR: Final[float] = 1.4826


@dataclass(frozen=True)
class Evidence:
    """Per-half-hour evidence about a farm's health under one plant.

    Attributes:
        expected: The plant's expected power for every row of the farm (train and hold-out).
        log_ratio: Log of observed over expected power where the row counts, NaN elsewhere.
        weight: Inverse-variance weight where the row counts, 0 elsewhere.
        sigma_speed: Estimated relative ERA5 speed error.
        sigma_power: Estimated power noise of the log ratio on the flat top.
        scale: The plant's flat top.
    """

    expected: np.ndarray
    log_ratio: np.ndarray
    weight: np.ndarray
    sigma_speed: float
    sigma_power: float
    scale: float


def robust_sd(values: np.ndarray) -> float:
    """Return 1.4826 times the median absolute deviation."""
    return float(ROBUST_SD_FACTOR * np.median(np.abs(values - np.median(values))))


def build_evidence(
    *, farm: common.FarmData, parameters: plant.PlantParameters, prior_scale: float
) -> Evidence:
    """Return the evidence rows for every half-hour of `farm` under the plant `parameters`."""
    weather = farm.weather
    expected = plant.expected_power(parameters=parameters, weather=weather, prior_scale=prior_scale)
    counts = (
        ~farm.idle_mask
        & (expected >= EXPECTED_RANGE[0] * parameters.scale)
        & (expected <= EXPECTED_RANGE[1] * parameters.scale)
    )
    floor = FLOOR_SHARE * parameters.scale
    log_ratio = np.where(
        counts, np.log(np.maximum(farm.target, floor) / np.maximum(expected, 1e-6)), np.nan
    )
    slope = plant.log_slope(parameters=parameters, weather=weather, prior_scale=prior_scale)
    train_rows = counts & farm.is_train
    # Estimate sigma_v and sigma_p from the robust variance of the log ratio by slope bin.
    edges = np.quantile(slope[train_rows], np.linspace(0, 1, SLOPE_BINS + 1))
    bin_index = np.clip(np.digitize(slope[train_rows], edges[1:-1]), 0, SLOPE_BINS - 1)
    variance, slope_squared, size = [], [], []
    for b in range(SLOPE_BINS):
        selection = bin_index == b
        if selection.sum() < 50:
            continue
        variance.append(robust_sd(log_ratio[train_rows][selection]) ** 2)
        slope_squared.append(float(np.mean(slope[train_rows][selection] ** 2)))
        size.append(selection.sum())
    design = np.stack([np.ones(len(slope_squared)), np.array(slope_squared)], axis=1)
    root = np.sqrt(np.array(size, dtype=float))
    coefficients, *_ = np.linalg.lstsq(
        design * root[:, None], np.array(variance) * root, rcond=None
    )
    sigma_power_sq = max(float(coefficients[0]), 1e-4)
    sigma_speed_sq = max(float(coefficients[1]), 1e-6)
    variance_row = slope**2 * sigma_speed_sq + sigma_power_sq
    weight = np.where(counts, 1.0 / variance_row, 0.0)
    return Evidence(
        expected=expected,
        log_ratio=log_ratio,
        weight=weight,
        sigma_speed=float(np.sqrt(sigma_speed_sq)),
        sigma_power=float(np.sqrt(sigma_power_sq)),
        scale=parameters.scale,
    )


@dataclass(frozen=True)
class DailySeries:
    """Weighted daily log ratios.

    Attributes:
        day: Day index (days since 1970-01-01) of each scored day, ascending.
        ratio: The weighted mean log ratio of each scored day.
    """

    day: np.ndarray
    ratio: np.ndarray


def daily_series(
    *, farm: common.FarmData, log_ratio: np.ndarray, weight: np.ndarray, rows: np.ndarray
) -> DailySeries:
    """Return the weighted daily log ratio of the selected `rows` (days with enough half-hours)."""
    counted = rows & (weight > 0) & np.isfinite(log_ratio)
    days, inverse = np.unique(farm.day[counted], return_inverse=True)
    n = np.bincount(inverse)
    total_weight = np.bincount(inverse, weights=weight[counted])
    total = np.bincount(inverse, weights=weight[counted] * log_ratio[counted])
    scored = n >= MIN_HALF_HOURS
    return DailySeries(day=days[scored], ratio=(total / total_weight)[scored])


@dataclass(frozen=True)
class Reference:
    """The median and robust standard deviation of the daily log ratios."""

    median: float
    sd: float


def reference_of(series: DailySeries) -> Reference:
    """Return the reference of a daily series."""
    return Reference(median=float(np.median(series.ratio)), sd=robust_sd(series.ratio))


def blocks_of(*, days: np.ndarray) -> list[tuple[int, int]]:
    """Return the (first day, last day) of each maximal run of consecutive days in `days`."""
    if days.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(days) > 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [days.size - 1]])
    return [(int(days[a]), int(days[b])) for a, b in zip(starts, ends, strict=True)]


def flagged_days(*, series: DailySeries, reference: Reference, k: float) -> np.ndarray:
    """Return the flagged day indices: ratio below the median by more than `k` robust sd."""
    return series.day[series.ratio < reference.median - k * reference.sd]


def idle_run_rows(*, farm: common.FarmData, rated: float) -> np.ndarray:
    """Return rows in 12-hour runs at the idle floor while ERA5 speed exceeds `rated`."""
    low = farm.speed_100m < IDLE_SPEED_M_S
    floor = float(np.quantile(farm.target[low & farm.is_train], 0.95))
    condition = (farm.target <= floor) & (farm.speed_100m > rated)
    step_is_half_hour = np.diff(farm.time).astype("timedelta64[m]").astype(int) == 30
    same_run = np.zeros(condition.size, dtype=bool)
    same_run[1:] = condition[1:] & condition[:-1] & step_is_half_hour
    run_id = np.cumsum(~same_run)
    lengths = np.bincount(run_id[condition], minlength=run_id.max() + 1)
    return condition & (lengths[run_id] >= IDLE_RUN_HALF_HOURS)


def rows_in_days(*, farm: common.FarmData, days: np.ndarray) -> np.ndarray:
    """Return the rows of `farm` that fall on any of the day indices `days`."""
    return np.isin(farm.day, days)
