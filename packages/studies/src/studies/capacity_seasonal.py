"""Capacity estimators with the physics model's seasonal bias divided out, for issue #1147.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. The
first capacity study (issue #1143) found that the ratio of a plant's power to the physics model's
output has a winter bias, which the planted-loss ratio cancels and which therefore rewarded agile
estimators on the tuning draw. This module removes the bias before any estimator sees the ratio.

**The seasonal reference of one plant on one day has two parts, and both read only days up to that
day.** The shared part is the median, over the other plants, of that day's log ratio (the six plants
share one 25 km satellite box, so the bias is common). The own part is the plant's own day-of-year
profile of what the shared part leaves, from earlier calendar years only. **The shared part hides a
loss that hits every plant on the same day, which the planted losses (one plant per event) do not
test.** Each plant's log ratio is centred on its median over the reference year first, a constant
that reads the reference year (all scored days fall after it).

`corrected_record` multiplies the modelled output by the reference, so that every estimator in
`studies.capacity_estimators` and `studies.capacity_convex` runs unchanged on the corrected ratio.
`hyst2_scale`, `hm2_scale`, and `cx2s_scale` turn that into the issue-time scales of the three
designs the study compares.
"""

import warnings
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import numpy as np

from studies.capacity_estimators import (
    KF_PARAMS,
    MIN_SHAPE_SUM,
    CapacityRecord,
    IssueScale,
    RawEstimate,
    issue_time_estimates,
    reference_day_indices,
    reference_levels,
    run_cheap_estimator,
    valid_days,
)
from studies.capacity_posthoc import (
    HYSTERESIS_DROP,
    HYSTERESIS_PERSIST_DAYS,
    day_of_year,
    hysteresis_level,
    profile_before,
)


def daily_log_ratio(*, record: CapacityRecord) -> np.ndarray:
    """Return each plant's logarithmic daily ratio of power to modelled power.

    Args:
        record: The record.

    Returns:
        An array of shape (plants, days), NaN on an invalid day or a day with little modelled power.
    """
    valid = valid_days(record=record)
    both = valid[:, :, None] & np.isfinite(record.power_mw) & np.isfinite(record.cams_shape)
    top = np.where(both, record.power_mw, 0.0).sum(axis=2)
    bottom = np.where(both, record.cams_shape, 0.0).sum(axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(
            bottom >= MIN_SHAPE_SUM, top / np.where(bottom > 0, bottom, np.nan), np.nan
        )
        return np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), np.nan)


def centring_constants(*, log_ratio: np.ndarray, record: CapacityRecord) -> np.ndarray:
    """Return each plant's median log ratio over the reference year.

    Args:
        log_ratio: `daily_log_ratio`'s result.
        record: The record, for the reference year.

    Returns:
        One constant per plant, 0 where the plant has no valid day in the reference year.
    """
    days = reference_day_indices(record=record)
    if len(days) == 0:
        return np.zeros(len(record.sites))
    # The stride-7 days are enough to centre; the constant is a median, not a fit.
    values = log_ratio[:, days]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.array([np.nanmedian(row) if np.isfinite(row).any() else 0.0 for row in values])


def shared_term(*, centred: np.ndarray) -> np.ndarray:
    """Return, for each plant and day, the median of the other plants' centred log ratios.

    Args:
        centred: Centred log ratios, shape (plants, days).

    Returns:
        An array of the same shape, NaN where no other plant has a ratio that day.
    """
    plants = centred.shape[0]
    out = np.full(centred.shape, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for plant in range(plants):
            others = np.delete(centred, plant, axis=0)
            has_any = np.isfinite(others).any(axis=0)
            out[plant] = np.where(
                has_any, np.nanmedian(np.where(np.isfinite(others), others, np.nan), axis=0), np.nan
            )
    return out


def own_profile_term(*, residual: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Return each plant's day-of-year profile of `residual` from earlier calendar years only.

    Args:
        residual: Log ratios left after the shared term, shape (plants, days).
        dates: One calendar day per column, as `datetime64[D]`.

    Returns:
        An array of the same shape. A day in the record's first calendar year, or one whose
        profile pooled fewer than the least number of values, is NaN.
    """
    doy = day_of_year(dates=dates)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    out = np.full(residual.shape, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for year in np.unique(years):
            prior = years < year
            if not prior.any():
                continue
            today = years == year
            for plant in range(residual.shape[0]):
                profile = profile_before(
                    log_level=residual[plant : plant + 1], doy=doy, prior=prior
                )
                out[plant, today] = profile[doy[today]]
    return out


class SeasonalReference:
    """The log reference of every plant and day, and the pieces it is made of.

    Attributes:
        centre: The centring constant per plant, shape (plants,).
        shared: The shared term, shape (plants, days), NaN where undefined.
        own: The own-profile term, shape (plants, days), NaN where undefined.
    """

    def __init__(self, *, centre: np.ndarray, shared: np.ndarray, own: np.ndarray) -> None:
        """Hold the pieces.

        Args:
            centre: The centring constants.
            shared: The shared term.
            own: The own-profile term.
        """
        self.centre = centre
        self.shared = shared
        self.own = own

    @property
    def log_reference(self) -> np.ndarray:
        """The plant's log ratio expected on each day, a term counting as 0 where it is NaN."""
        return (
            self.centre[:, None]
            + np.nan_to_num(self.shared, nan=0.0)
            + np.nan_to_num(self.own, nan=0.0)
        )


def seasonal_reference(*, record: CapacityRecord) -> SeasonalReference:
    """Compute the seasonal reference of a record, reading each day's own and earlier days only.

    Args:
        record: The record.

    Returns:
        The reference.
    """
    log_ratio = daily_log_ratio(record=record)
    centre = centring_constants(log_ratio=log_ratio, record=record)
    centred = log_ratio - centre[:, None]
    shared = shared_term(centred=centred)
    own = own_profile_term(residual=centred - np.nan_to_num(shared, nan=0.0), dates=record.dates)
    return SeasonalReference(centre=centre, shared=shared, own=own)


def corrected_record(*, record: CapacityRecord, reference: SeasonalReference) -> CapacityRecord:
    """Return the record whose modelled output carries the seasonal reference.

    Power over the corrected modelled output is the original ratio divided by the reference, so it
    sits near 1 whenever the plant behaves as it did in the reference year.

    Args:
        record: The record.
        reference: `seasonal_reference`'s result on the same record.

    Returns:
        The record with `cams_shape` multiplied by the reference, everything else unchanged.
    """
    factor = np.exp(reference.log_reference)[:, :, None]
    return replace(record, cams_shape=record.cams_shape * factor)


KF_REFERENCE_PARAMS: Final[Mapping[str, float]] = KF_PARAMS
"""The Kalman filter that smooths the corrected ratio before the hysteresis rule reads it."""


def hyst2_ratio(
    *, corrected: CapacityRecord, reference_level: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return the smoothed corrected ratio that the hysteresis rule reads, and its reference level.

    Args:
        corrected: `corrected_record`'s result.
        reference_level: Each plant's reference-year median of the smoothed ratio, from the
            original record, or `None` to take it from this record.

    Returns:
        The smoothed ratio over the reference level, shape (plants, days), and the reference level.
    """
    smoothed = run_cheap_estimator(name="KF", record=corrected, params=KF_REFERENCE_PARAMS)
    level = (
        reference_levels(raw=smoothed, record=corrected)
        if reference_level is None
        else reference_level
    )
    return smoothed.level / level[:, None], level


def hyst2_scale(
    *,
    record: CapacityRecord,
    ratio: np.ndarray,
    drop: float = HYSTERESIS_DROP,
    persist_days: int = HYSTERESIS_PERSIST_DAYS,
) -> IssueScale:
    """Return HYST2's issue-time scale: hold at the static power until a drop persists.

    Args:
        record: The record (for the static power and the history rule).
        ratio: `hyst2_ratio`'s first result.
        drop: The share below 1 that counts as a drop.
        persist_days: How many days a drop, or a recovery, must last.

    Returns:
        The scale for every plant and target day.
    """
    held = np.stack(
        [hysteresis_level(ratio=row, drop=drop, persist_days=persist_days) for row in ratio]
    )
    raw = RawEstimate(level=held * record.static_p99_mw[:, None], sd=None)
    return issue_time_estimates(record=record, raw=raw, reference=None)


def hm2_scale(
    *, record: CapacityRecord, corrected: CapacityRecord, params: Mapping[str, float]
) -> IssueScale:
    """Return HM2's issue-time scale: the hidden Markov model run on the corrected ratio.

    Args:
        record: The record (for the static power and the history rule).
        corrected: `corrected_record`'s result.
        params: `stay_probability` and `observation_sd`.

    Returns:
        The scale for every plant and target day.
    """
    raw = run_cheap_estimator(name="HM", record=corrected, params=params)
    return issue_time_estimates(record=record, raw=raw, reference=None)


def cx2s_scale(*, record: CapacityRecord, raw: RawEstimate, reference: np.ndarray) -> IssueScale:
    """Return CX2S's issue-time scale from the linear program's raw level on the corrected ratio.

    Args:
        record: The record.
        raw: CX2's raw level on the corrected record.
        reference: Each plant's reference-year median raw level on the original corrected record.

    Returns:
        The scale for every plant and target day.
    """
    return issue_time_estimates(record=record, raw=raw, reference=reference)
