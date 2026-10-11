"""Estimate a solar plant's time-varying capacity from its own power, for the study in issue #1143.

The study asks whether dividing a solar forecast's target by a time-varying capacity estimate beats
dividing by one fixed number per plant. `studies/effective_capacity/` runs it and
`docs/studies/forecasts/effective-capacity.md` holds the design and the results.

**Every estimator here is causal: its estimate for day `d` reads only whole days up to and including
`d`, apart from one constant per plant.** That constant is the median of the estimator's level over
the reference year (`REFERENCE_START` to `REFERENCE_END`), which rescales the estimator to the
static power. A day inside or before the reference year therefore sees a constant that read later
days. The scored rows all fall after the reference year, where the constant reads only the past.

**The record is a day-by-hour grid per plant.** A day is labelled by the date of its hours'
midpoints, so a power reading stamped `2025-03-02 00:00` (the hour ending at midnight) belongs to
2025-03-01. A forecast issued at 09:00 UTC on day `T - 1` can read whole days up to `T - 2`, so
**the capacity used for target day `T` is the estimate for day `T - 2`** (`ISSUE_LAG_DAYS`), held
constant across day `T`. `issue_time_estimates` applies that shift, the fallback to the plant's
static 99th-percentile power, and the floor, and the forecast-row join reads its result.

**The hindsight reference reads both sides of a target week and never the week itself.**
`hindsight_reference_estimates` combines the forward estimate from whole days up to the day before
the week with the same estimator run on the time-reversed record from the day after the week.

**This module holds the record, the eight estimators that are neither linear programs nor neural
networks (EM, R30, CSQ, NR, PR, KF, and HM, and the no-skill reference S0), the issue-time driver,
and the hindsight driver.** The
linear programs are in `studies.capacity_convex`, the neural network is in
`studies.capacity_physics_torch`, and the planted losses, the metrics, and the selection rule are in
`studies.capacity_recovery`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Final, Literal, NamedTuple

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from studies.baselines import MIN_OBSERVED_CLEAR_SKY_SHARE

HOURS_PER_DAY: Final[int] = 24
"""The hour axis of a record: hours whose midpoints fall in the day."""

ISSUE_LAG_DAYS: Final[int] = 2
"""How many days before the target day the latest whole day a forecast has seen falls."""

MIN_VALID_DAYS_OF_HISTORY: Final[int] = 30
"""Fewer valid days than this up to the latest whole day, and the plant's static power is used."""

FLOOR_SHARE_OF_STATIC: Final[float] = 0.2
"""The least a scale may be, as a share of the plant's static 99th-percentile power."""

REFERENCE_START: Final[date] = date(2023, 12, 1)
"""The first day of the reference year, which calibrates each estimator against the static power."""

REFERENCE_END: Final[date] = date(2024, 12, 1)
"""The day after the reference year, which is also the first day of the lag study's scored rows."""

REFERENCE_STRIDE_DAYS: Final[int] = 7
"""The reference year's median is taken over every seventh day, to keep slow estimators cheap."""

MIN_SHAPE_SUM: Final[float] = 0.2
"""A day's ratio of power to modelled power needs this many full-capacity hours modelled."""

LOW_SUN_CLEAR_SKY_W_M2: Final[float] = 50.0
"""Hours with less clear-sky irradiance than this are left out of ratios, where they are noise."""

FamilyType = Literal["cheap", "wacky", "convex", "pytorch", "reference"]
"""The four families of capacity estimator the study compares, and the references that are not
candidates (the expanding maximum EM and the static power S0)."""

EstimatorNameType = Literal["S0", "EM", "R30", "CSQ", "NR", "PR", "KF", "HM", "CX1", "CX2", "DP1"]
"""The nine candidate estimators and the two references."""

REFERENCE_ESTIMATORS: Final[frozenset[str]] = frozenset({"S0", "EM"})
"""Estimators that are references for the page and never candidates for selection: S0 never moves
(no skill), and EM is the published running maximum, which cannot follow a loss down."""

ESTIMATOR_FAMILY: Final[Mapping[str, FamilyType]] = {
    "S0": "reference",
    "EM": "reference",
    "R30": "cheap",
    "CSQ": "cheap",
    "NR": "cheap",
    "PR": "cheap",
    "KF": "cheap",
    "HM": "wacky",
    "CX1": "convex",
    "CX2": "convex",
    "DP1": "pytorch",
}
"""Each estimator's family."""

CALIBRATED_ESTIMATORS: Final[frozenset[str]] = frozenset({"CSQ", "PR", "KF", "CX1", "CX2", "DP1"})
"""Estimators whose raw level has arbitrary units, rescaled so that the reference year's median
level equals the plant's static power. R30 is already in megawatts, NR is multiplied by the static
power, and HM returns a share of the reference year's level that is also multiplied by it."""

FALLBACK_NONE: Final[int] = 0
FALLBACK_SHORT_HISTORY: Final[int] = 1
FALLBACK_NULL_ESTIMATE: Final[int] = 2
"""Why a day's scale is the static power: not at all, too little history, or a null estimate."""


@dataclass(frozen=True, eq=False)
class CapacityRecord:
    """Every plant's record on a day-by-hour grid.

    Attributes:
        sites: The anonymous plant labels, in axis-0 order.
        dates: One calendar day per axis-1 entry, consecutive, as `datetime64[D]`.
        power_mw: Measured hourly power, shape (plants, days, 24); NaN where no clean reading
            exists.
        clear_sky_w_m2: The clear-sky irradiance of the same hours, never NaN.
        cams_ghi_w_m2: Satellite-derived global horizontal irradiance of the same hours; NaN where
            missing.
        cams_shape: The modelled power per megawatt of direct-current capacity driven by the
            satellite irradiance at the plant's fitted orientation, without clipping; NaN where
            the irradiance is.
        clear_shape: The same with the clear-sky irradiance as the sky.
        static_p99_mw: Each plant's fixed 99th-percentile power, which the baseline divides by.
    """

    sites: tuple[str, ...]
    dates: np.ndarray
    power_mw: np.ndarray
    clear_sky_w_m2: np.ndarray
    cams_ghi_w_m2: np.ndarray
    cams_shape: np.ndarray
    clear_shape: np.ndarray
    static_p99_mw: np.ndarray

    def __post_init__(self) -> None:
        """Check that every array has the record's shape.

        Raises:
            ValueError: If the arrays' shapes disagree with the sites and the dates.
        """
        grid = (len(self.sites), len(self.dates), HOURS_PER_DAY)
        for name in ("power_mw", "clear_sky_w_m2", "cams_ghi_w_m2", "cams_shape", "clear_shape"):
            if getattr(self, name).shape != grid:
                msg = f"{name} has shape {getattr(self, name).shape}, not {grid}"
                raise ValueError(msg)
        if self.static_p99_mw.shape != (len(self.sites),):
            msg = f"static_p99_mw has shape {self.static_p99_mw.shape}, not ({len(self.sites)},)"
            raise ValueError(msg)

    def reversed(self) -> CapacityRecord:
        """Return the record with its days in reverse order, for the hindsight reference.

        Returns:
            The reversed record. Dates run backwards, so they are no longer a calendar.
        """
        return replace(
            self,
            dates=self.dates[::-1].copy(),
            power_mw=self.power_mw[:, ::-1].copy(),
            clear_sky_w_m2=self.clear_sky_w_m2[:, ::-1].copy(),
            cams_ghi_w_m2=self.cams_ghi_w_m2[:, ::-1].copy(),
            cams_shape=self.cams_shape[:, ::-1].copy(),
            clear_shape=self.clear_shape[:, ::-1].copy(),
        )

    def with_power(self, *, power_mw: np.ndarray) -> CapacityRecord:
        """Return the record with other measured power.

        Args:
            power_mw: The new power, shape (plants, days, 24).

        Returns:
            The record.
        """
        return replace(self, power_mw=power_mw)


class RawEstimate(NamedTuple):
    """An estimator's level for every plant and day, using whole days up to that day."""

    level: np.ndarray
    sd: np.ndarray | None


class IssueScale(NamedTuple):
    """The scale a forecast for each target day divides by, after the fallback and the floor."""

    scale_mw: np.ndarray
    relative_sd: np.ndarray
    fallback_reason: np.ndarray


def valid_days(*, record: CapacityRecord) -> np.ndarray:
    """Return which days are valid: their observed hours hold half their clear-sky energy.

    Args:
        record: The record.

    Returns:
        A boolean array of shape (plants, days).
    """
    clear = record.clear_sky_w_m2
    total = clear.sum(axis=2)
    seen = np.where(np.isfinite(record.power_mw), clear, 0.0).sum(axis=2)
    return (total > 0) & (seen >= MIN_OBSERVED_CLEAR_SKY_SHARE * total)


def reference_day_indices(*, record: CapacityRecord) -> np.ndarray:
    """Return the indices of the reference year's days used to calibrate, every seventh day.

    Args:
        record: The record.

    Returns:
        The day indices, possibly empty.
    """
    in_year = (record.dates >= np.datetime64(REFERENCE_START)) & (
        record.dates < np.datetime64(REFERENCE_END)
    )
    return np.flatnonzero(in_year)[::REFERENCE_STRIDE_DAYS]


def _trailing_windows(*, values: np.ndarray, window: int) -> np.ndarray:
    """Return, for every day, the `window` days ending at it, NaN-padded before the record starts.

    Args:
        values: An array of shape (days, ...).
        window: The window length in days.

    Returns:
        An array of shape (days, window, ...).
    """
    pad = np.full((window - 1, *values.shape[1:]), np.nan)
    windows = sliding_window_view(np.concatenate([pad, values]), window, axis=0)
    return np.moveaxis(windows, -1, 1)


def _rolling_mean(*, values: np.ndarray, window: int, min_count: int) -> np.ndarray:
    """Return the trailing mean of the finite values, NaN where fewer than `min_count` are finite.

    Args:
        values: A one-dimensional array with NaN for null.
        window: The window length.
        min_count: The fewest finite values the mean needs.

    Returns:
        The mean, aligned with `values`.
    """
    windows = _trailing_windows(values=values, window=window)
    finite = np.isfinite(windows)
    count = finite.sum(axis=1)
    total = np.where(finite, windows, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= min_count, total / np.maximum(count, 1), np.nan)


def _daily_ratio(
    *, numerator: np.ndarray, denominator: np.ndarray, usable: np.ndarray, minimum: float
) -> np.ndarray:
    """Return each day's sum of numerators over the sum of denominators, on usable hours only.

    Args:
        numerator: Hourly values of shape (days, 24), NaN for null.
        denominator: Hourly values of the same shape.
        usable: A (days, 24) mask of hours to include.
        minimum: A day with a denominator sum below this is NaN.

    Returns:
        One ratio per day.
    """
    both = usable & np.isfinite(numerator) & np.isfinite(denominator)
    top = np.where(both, numerator, 0.0).sum(axis=1)
    bottom = np.where(both, denominator, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(bottom >= minimum, top / np.where(bottom > 0, bottom, np.nan), np.nan)


def _daily_shape_ratio(*, record: CapacityRecord, plant: int, valid: np.ndarray) -> np.ndarray:
    """Return each valid day's power over its modelled power at unit direct-current capacity.

    Args:
        record: The record.
        plant: The plant's index.
        valid: The record's valid days.

    Returns:
        One value per day, NaN on an invalid day or a day with little modelled power.
    """
    usable = np.broadcast_to(valid[plant][:, None], (valid.shape[1], HOURS_PER_DAY))
    return _daily_ratio(
        numerator=record.power_mw[plant],
        denominator=record.cams_shape[plant],
        usable=usable,
        minimum=MIN_SHAPE_SUM,
    )


def estimate_em(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the running maximum of the plant's hourly power over every valid day so far.

    This is the published reference that the running-maximum normalisation of Dantas and Browell and
    the cumulative-maximum baseline of Viotti et al. both use. It never falls, so it cannot follow a
    loss downwards; it is in the study to put Stage A on the same footing as those papers.

    Args:
        record: The record.
        plant: The plant's index.
        params: Unused; EM has no setting.

    Returns:
        The running maximum in megawatts, null before the first valid day.
    """
    del params
    valid = valid_days(record=record)[plant]
    daily_max = np.where(
        valid,
        np.nanmax(
            np.where(np.isfinite(record.power_mw[plant]), record.power_mw[plant], -np.inf), axis=1
        ),
        np.nan,
    )
    return RawEstimate(level=np.fmax.accumulate(daily_max), sd=None)


def estimate_s0(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the plant's static power on every day: the no-skill reference that never moves.

    Args:
        record: The record.
        plant: The plant's index.
        params: Unused; S0 has no setting.

    Returns:
        The static 99th-percentile power, in megawatts, on every day.
    """
    del params
    return RawEstimate(level=np.full(len(record.dates), record.static_p99_mw[plant]), sd=None)


def estimate_r30(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the rolling 30-day 99th percentile of the plant's hourly power.

    Args:
        record: The record.
        plant: The plant's index.
        params: Unused; R30 has no setting.

    Returns:
        The estimate, null before 15 valid days are in the window.
    """
    del params
    valid = valid_days(record=record)[plant]
    power = np.where(valid[:, None], record.power_mw[plant], np.nan)
    windows = _trailing_windows(values=power, window=30).reshape(len(valid), -1)
    count = _trailing_windows(values=valid.astype(float), window=30)
    enough = np.nansum(count, axis=1) >= 15
    level = np.full(len(valid), np.nan)
    for day in np.flatnonzero(enough):
        level[day] = np.nanquantile(windows[day], 0.99)
    return RawEstimate(level=level, sd=None)


def estimate_csq(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the rolling 30-day 90th percentile of power over modelled clear-sky output.

    The denominator is the physics model's output under a clear sky at the plant's fitted tilt and
    azimuth, not the horizontal irradiance, so the ratio carries no seasonal swing from the tilt.

    Args:
        record: The record.
        plant: The plant's index.
        params: Unused; CSQ has no setting.

    Returns:
        The raw level, null before 15 valid days and 100 sunny hours are in the window.
    """
    del params
    valid = valid_days(record=record)[plant]
    clear = record.clear_shape[plant]
    sunny = valid[:, None] & (record.clear_sky_w_m2[plant] >= LOW_SUN_CLEAR_SKY_W_M2) & (clear > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(sunny, record.power_mw[plant] / np.where(sunny, clear, 1.0), np.nan)
    windows = _trailing_windows(values=ratio, window=30).reshape(len(valid), -1)
    days = _trailing_windows(values=valid.astype(float), window=30)
    level = np.full(len(valid), np.nan)
    for day in np.flatnonzero(np.nansum(days, axis=1) >= 15):
        values = windows[day][np.isfinite(windows[day])]
        if len(values) >= 100:
            level[day] = np.quantile(values, 0.9)
    return RawEstimate(level=level, sd=None)


def relative_energy(*, record: CapacityRecord) -> np.ndarray:
    """Return each plant's valid-day energy over its static power, NaN on an invalid day.

    Args:
        record: The record.

    Returns:
        An array of shape (plants, days).
    """
    energy = np.nansum(record.power_mw, axis=2)
    return np.where(valid_days(record=record), energy / record.static_p99_mw[:, None], np.nan)


def neighbour_ratio(*, record: CapacityRecord) -> np.ndarray:
    """Return each plant's 7-day relative energy over the other plants' mean, for every day.

    The seven-day mean of a plant's daily energy over its static power needs 5 valid days. The
    ratio divides it by the mean of the other plants' same seven-day means, over the plants that
    have one.

    Args:
        record: The record.

    Returns:
        An array of shape (plants, days) with NaN where the ratio is null.
    """
    rolled = np.stack(
        [_rolling_mean(values=row, window=7, min_count=5) for row in relative_energy(record=record)]
    )
    finite = np.isfinite(rolled)
    others_count = finite.sum(axis=0)[None, :] - finite
    others_total = np.where(finite, rolled, 0.0).sum(axis=0)[None, :] - np.where(
        finite, rolled, 0.0
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        others_mean = others_total / np.where(others_count > 0, others_count, np.nan)
        ratio = rolled / others_mean
    return np.where((others_count > 0) & (others_mean > 0), ratio, np.nan)


def estimate_nr(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the plant's 7-day energy ratio to the other plants, times its static power.

    Args:
        record: The record.
        plant: The plant's index.
        params: Unused; NR has no setting.

    Returns:
        The estimate in megawatts.
    """
    del params
    return RawEstimate(
        level=neighbour_ratio(record=record)[plant] * record.static_p99_mw[plant], sd=None
    )


def estimate_pr(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the sum of power over the sum of modelled output across the latest 7 days.

    The denominator is the physics model's output driven by the satellite irradiance at the plant's
    fitted tilt and azimuth, not the horizontal irradiance, so the ratio carries no seasonal swing
    from the tilt.

    Args:
        record: The record.
        plant: The plant's index.
        params: Unused; PR has no setting.

    Returns:
        The raw level, null with fewer than 4 days that hold both.
    """
    del params
    valid = valid_days(record=record)[plant]
    usable = np.broadcast_to(valid[:, None], (len(valid), HOURS_PER_DAY))
    both = usable & np.isfinite(record.power_mw[plant]) & np.isfinite(record.cams_shape[plant])
    top = np.where(both, record.power_mw[plant], 0.0).sum(axis=1)
    bottom = np.where(both, record.cams_shape[plant], 0.0).sum(axis=1)
    has_day = np.where(bottom > 0, 1.0, np.nan)
    window_days = np.nansum(_trailing_windows(values=has_day, window=7), axis=1)
    top_sum = _trailing_windows(values=top, window=7)
    bottom_sum = _trailing_windows(values=bottom, window=7)
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.nansum(top_sum, axis=1) / np.nansum(bottom_sum, axis=1)
    return RawEstimate(level=np.where(window_days >= 4, level, np.nan), sd=None)


KF_PARAMS: Final[Mapping[str, float]] = {"process_sd": 0.01, "observation_sd": 0.1}
"""The Kalman filter's daily random-walk standard deviation and observation standard deviation, both
in the logarithm of the daily ratio."""

KF_INNOVATION_CLIP: Final[float] = 3.0
"""An observation further than this many innovation standard deviations from the prediction is
pulled back to that distance, so that one bad satellite day cannot move the level."""


def estimate_kf(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return a Kalman local-level filter on the daily log ratio of power to modelled power.

    The state is the logarithm of the plant's output per modelled unit of direct-current capacity,
    which follows a random walk. A day without a valid observation only widens the state's variance.

    Args:
        record: The record.
        plant: The plant's index.
        params: `process_sd` and `observation_sd`, in logarithmic units per day.

    Returns:
        The raw level and its standard deviation (by the delta method), null until the first
        valid day.
    """
    process_variance = params["process_sd"] ** 2
    observation_variance = params["observation_sd"] ** 2
    ratio = _daily_shape_ratio(record=record, plant=plant, valid=valid_days(record=record))
    observations = np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), np.nan)
    days = len(observations)
    level = np.full(days, np.nan)
    sd = np.full(days, np.nan)
    state, variance = np.nan, np.nan
    for day in range(days):
        observation = observations[day]
        if np.isnan(state):
            if np.isfinite(observation):
                state, variance = observation, observation_variance
        else:
            variance += process_variance
            if np.isfinite(observation):
                innovation_variance = variance + observation_variance
                limit = KF_INNOVATION_CLIP * np.sqrt(innovation_variance)
                innovation = float(np.clip(observation - state, -limit, limit))
                gain = variance / innovation_variance
                state += gain * innovation
                variance *= 1.0 - gain
        if np.isfinite(state):
            level[day] = np.exp(state)
            sd[day] = level[day] * np.sqrt(variance)
    return RawEstimate(level=level, sd=sd)


HM_LEVELS: Final[tuple[float, ...]] = (1.0, 0.9, 0.75, 0.5)
"""The hidden Markov model's capacity levels, as shares of the reference year's capacity."""

HM_PARAMS: Final[Mapping[str, float]] = {"stay_probability": 0.98, "observation_sd": 0.15}
"""The probability of staying at a level from one day to the next, and the logarithmic standard
deviation of a day's ratio around its level."""

HM_INITIAL_FULL_PROBABILITY: Final[float] = 0.7
"""The prior probability of the full level before the first valid day."""


def estimate_hm(*, record: CapacityRecord, plant: int, params: Mapping[str, float]) -> RawEstimate:
    """Return the posterior mean capacity of a hidden Markov model over discrete capacity levels.

    **The model encodes that strings and inverters fail in whole units.** Each day's ratio of power
    to modelled power, over its median in the reference year, is log-normal around the current
    level. The level stays put with `stay_probability` and otherwise jumps to another level. The
    forward algorithm gives the posterior over levels after each day.

    Args:
        record: The record.
        plant: The plant's index.
        params: `stay_probability` and `observation_sd`.

    Returns:
        The posterior mean level in megawatts (a share times the plant's static power), null where
        the reference year holds no valid day. A day without an observation moves the posterior
        through the transition matrix only.
    """
    ratio = _daily_shape_ratio(record=record, plant=plant, valid=valid_days(record=record))
    reference = ratio[reference_day_indices(record=record)]
    reference = reference[np.isfinite(reference)]
    days = len(ratio)
    level = np.full(days, np.nan)
    if len(reference) == 0:
        return RawEstimate(level=level, sd=None)
    relative = ratio / np.median(reference)
    log_levels = np.log(np.asarray(HM_LEVELS))
    n_levels = len(HM_LEVELS)
    stay = params["stay_probability"]
    transition = np.full((n_levels, n_levels), (1.0 - stay) / (n_levels - 1))
    np.fill_diagonal(transition, stay)
    posterior = np.full(n_levels, (1.0 - HM_INITIAL_FULL_PROBABILITY) / (n_levels - 1))
    posterior[0] = HM_INITIAL_FULL_PROBABILITY
    started = False
    for day in range(days):
        posterior = posterior @ transition
        value = relative[day]
        if np.isfinite(value) and value > 0:
            started = True
            z = (np.log(value) - log_levels) / params["observation_sd"]
            log_likelihood = -0.5 * z**2
            posterior = posterior * np.exp(log_likelihood - log_likelihood.max())
            posterior /= posterior.sum()
        if started:
            level[day] = float(posterior @ np.asarray(HM_LEVELS)) * record.static_p99_mw[plant]
    return RawEstimate(level=level, sd=None)


CheapEstimatorFunction = Callable[..., RawEstimate]

CHEAP_ESTIMATORS: Final[Mapping[str, CheapEstimatorFunction]] = {
    "S0": estimate_s0,
    "EM": estimate_em,
    "R30": estimate_r30,
    "CSQ": estimate_csq,
    "NR": estimate_nr,
    "PR": estimate_pr,
    "KF": estimate_kf,
    "HM": estimate_hm,
}
"""The estimators of this module, by name."""

DEFAULT_PARAMS: Final[Mapping[str, Mapping[str, float]]] = {
    "S0": {},
    "EM": {},
    "R30": {},
    "CSQ": {},
    "NR": {},
    "PR": {},
    "KF": KF_PARAMS,
    "HM": HM_PARAMS,
}
"""The settings the tuning draw starts from."""


def run_cheap_estimator(
    *, name: str, record: CapacityRecord, params: Mapping[str, float] | None = None
) -> RawEstimate:
    """Run one of this module's estimators on every plant.

    Args:
        name: One of `CHEAP_ESTIMATORS`.
        record: The record.
        params: The estimator's settings, or `None` for `DEFAULT_PARAMS`.

    Returns:
        The raw level (and standard deviation, for KF) for every plant and day.
    """
    function = CHEAP_ESTIMATORS[name]
    settings = DEFAULT_PARAMS[name] if params is None else params
    parts = [
        function(record=record, plant=plant, params=settings) for plant in range(len(record.sites))
    ]
    level = np.stack([part.level for part in parts])
    sds = [part.sd for part in parts if part.sd is not None]
    sd = np.stack(sds) if len(sds) == len(parts) else None
    return RawEstimate(level=level, sd=sd)


def reference_levels(*, raw: RawEstimate, record: CapacityRecord) -> np.ndarray:
    """Return each plant's median raw level over the reference year's sampled days.

    Args:
        raw: An estimator's raw estimate on the original record.
        record: The same record.

    Returns:
        One value per plant, NaN where the estimator is null on every reference day.
    """
    days = reference_day_indices(record=record)
    if len(days) == 0:
        return np.full(len(record.sites), np.nan)
    values = raw.level[:, days]
    with np.errstate(all="ignore"):
        return np.array([np.nanmedian(row) if np.isfinite(row).any() else np.nan for row in values])


def calibrate(
    *, raw: RawEstimate, record: CapacityRecord, reference: np.ndarray | None
) -> RawEstimate:
    """Rescale a raw estimate so that its reference-year median equals the static power.

    Args:
        raw: The raw estimate.
        record: The record, for the static power.
        reference: `reference_levels`' result, or `None` for an estimator already in megawatts.

    Returns:
        The estimate in megawatts.
    """
    if reference is None:
        return raw
    factor = (record.static_p99_mw / reference)[:, None]
    return RawEstimate(level=raw.level * factor, sd=None if raw.sd is None else raw.sd * factor)


def _history_counts(*, record: CapacityRecord) -> np.ndarray:
    """Return how many valid days each plant has had up to and including each day."""
    return np.cumsum(valid_days(record=record), axis=1)


def _finalise(
    *,
    level: np.ndarray,
    sd: np.ndarray | None,
    history: np.ndarray,
    static: np.ndarray,
) -> IssueScale:
    """Apply the history rule, the null rule, the floor, and the fallback to a shifted level.

    Args:
        level: The calibrated level that applies to each target day, shape (plants, days).
        sd: Its standard deviation or `None`.
        history: Valid days seen by the latest whole day, same shape.
        static: Each plant's static power, shape (plants,).

    Returns:
        The scale and why each fallback happened.
    """
    anchor = static[:, None]
    unusable = ~np.isfinite(level) | (level <= 0)
    short = history < MIN_VALID_DAYS_OF_HISTORY
    reason = np.where(
        short, FALLBACK_SHORT_HISTORY, np.where(unusable, FALLBACK_NULL_ESTIMATE, FALLBACK_NONE)
    ).astype(np.int8)
    scale = np.where(
        reason == FALLBACK_NONE,
        np.maximum(np.where(unusable, anchor, level), FLOOR_SHARE_OF_STATIC * anchor),
        anchor,
    )
    relative_sd = (
        np.full(level.shape, np.nan)
        if sd is None
        else np.where(reason == FALLBACK_NONE, sd / np.where(unusable, np.nan, level), np.nan)
    )
    return IssueScale(scale_mw=scale, relative_sd=relative_sd, fallback_reason=reason)


def _lagged(*, values: np.ndarray, lag: int) -> np.ndarray:
    """Return `values` moved `lag` days later, NaN-filled at the start, along the last axis."""
    out = np.full(values.shape, np.nan)
    out[..., lag:] = values[..., : values.shape[-1] - lag]
    return out


def issue_time_estimates(
    *,
    record: CapacityRecord,
    raw: RawEstimate,
    reference: np.ndarray | None,
    lag_days: int = ISSUE_LAG_DAYS,
) -> IssueScale:
    """Return the scale a forecast for each target day divides by.

    The scale for target day `T` is the calibrated estimate for day `T - ISSUE_LAG_DAYS`. It falls
    back to the plant's static power where the plant has had fewer than
    `MIN_VALID_DAYS_OF_HISTORY` valid days by then or the estimate is null, and it is never below
    `FLOOR_SHARE_OF_STATIC` of the static power.

    Args:
        record: The record.
        raw: The estimator's raw estimate, whose day `d` read days up to `d` only.
        reference: `reference_levels`' result, or `None` for an estimator already in megawatts.
        lag_days: How many days before the target day the estimate is read; 3 gives the sensitivity
            in which satellite irradiance is read one day less fresh.

    Returns:
        The scale, its relative standard deviation, and the fallback reason, for every plant and
        target day.
    """
    calibrated = calibrate(raw=raw, record=record, reference=reference)
    history = _lagged(values=_history_counts(record=record).astype(float), lag=lag_days)
    return _finalise(
        level=_lagged(values=calibrated.level, lag=lag_days),
        sd=None if calibrated.sd is None else _lagged(values=calibrated.sd, lag=lag_days),
        history=np.nan_to_num(history, nan=0.0),
        static=record.static_p99_mw,
    )


def week_starts(*, days: int) -> np.ndarray:
    """Return the first day index of each 7-day target week of a record.

    Args:
        days: The number of days in the record.

    Returns:
        The indices 0, 7, 14, and so on.
    """
    return np.arange(0, days, 7)


def hindsight_reference_estimates(
    *,
    record: CapacityRecord,
    forward: RawEstimate,
    backward_on_reversed: RawEstimate,
    reference: np.ndarray | None,
) -> IssueScale:
    """Return the hindsight reference: both sides of each target week, never the week itself.

    For the week of days `[a, b]`, the forward estimate is the estimator's value for day `a - 1`
    and the backward estimate is the same estimator run on the time-reversed record, valued at the
    reversed record's equivalent of day `b + 1`. The two are averaged, weighted by inverse variance
    when both have a standard deviation and plainly otherwise. A side with fewer than
    `MIN_VALID_DAYS_OF_HISTORY` valid days behind it, or a null side, is left out. Every day of the
    week takes the week's value.

    Args:
        record: The (forward) record.
        forward: The estimator's raw estimate on `record`.
        backward_on_reversed: The estimator's raw estimate on `record.reversed()`, in reversed day
            order.
        reference: `reference_levels`' result, or `None` for an estimator already in megawatts.

    Returns:
        The scale, its relative standard deviation, and the fallback reason, for every plant and
        day.
    """
    days = len(record.dates)
    forward_calibrated = calibrate(raw=forward, record=record, reference=reference)
    backward_calibrated = calibrate(raw=backward_on_reversed, record=record, reference=reference)
    forward_history = _history_counts(record=record)
    backward_history = _history_counts(record=record.reversed())
    level = np.full((len(record.sites), days), np.nan)
    sd = np.full((len(record.sites), days), np.nan)
    history = np.zeros((len(record.sites), days))
    for first in week_starts(days=days):
        last = min(first + 7, days) - 1
        sides: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
        if first >= 1:
            sides.append(
                (
                    forward_calibrated.level[:, first - 1],
                    None if forward_calibrated.sd is None else forward_calibrated.sd[:, first - 1],
                    forward_history[:, first - 1],
                )
            )
        if last + 1 <= days - 1:
            reversed_day = days - 1 - (last + 1)
            sides.append(
                (
                    backward_calibrated.level[:, reversed_day],
                    None
                    if backward_calibrated.sd is None
                    else backward_calibrated.sd[:, reversed_day],
                    backward_history[:, reversed_day],
                )
            )
        combined, combined_sd, support = _combine_sides(sides=sides, plants=len(record.sites))
        level[:, first : last + 1] = combined[:, None]
        sd[:, first : last + 1] = combined_sd[:, None]
        history[:, first : last + 1] = support[:, None]
    return _finalise(
        level=level,
        sd=sd if forward.sd is not None else None,
        history=history,
        static=record.static_p99_mw,
    )


def _combine_sides(
    *, sides: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]], plants: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine the forward and backward estimates of one week for every plant.

    Args:
        sides: Each side's level, standard deviation (or `None`), and valid-day history.
        plants: The number of plants.

    Returns:
        The combined level, its standard deviation, and the largest history among the sides used.
        A side with too little history or a null level is left out; with no side left the level is
        NaN and the history 0. The sides are weighted by inverse variance when every side has a
        positive standard deviation, and equally otherwise.
    """
    levels = np.full((len(sides), plants), np.nan)
    variances = np.full((len(sides), plants), np.nan)
    supports = np.zeros((len(sides), plants))
    every_side_has_sd = len(sides) > 0
    for index, (level, sd, support) in enumerate(sides):
        usable = (support >= MIN_VALID_DAYS_OF_HISTORY) & np.isfinite(level) & (level > 0)
        levels[index] = np.where(usable, level, np.nan)
        supports[index] = np.where(usable, support, 0.0)
        if sd is None:
            every_side_has_sd = False
        else:
            variances[index] = np.where(usable & np.isfinite(sd) & (sd > 0), sd**2, np.nan)
    used = np.isfinite(levels)
    count = used.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        if every_side_has_sd:
            weights = np.where(used & np.isfinite(variances), 1.0 / variances, 0.0)
            weight_total = weights.sum(axis=0)
            has_weights = weight_total > 0
            combined = np.where(
                has_weights,
                (np.where(used, levels, 0.0) * weights).sum(axis=0) / weight_total,
                np.nan,
            )
            combined_sd = np.where(has_weights, 1.0 / np.sqrt(weight_total), np.nan)
        else:
            combined = np.where(count > 0, np.where(used, levels, 0.0).sum(axis=0) / count, np.nan)
            combined_sd = np.full(plants, np.nan)
    support = supports.max(axis=0) if len(sides) else np.zeros(plants)
    return combined, combined_sd, support


def target_day_indices(
    *, record: CapacityRecord, rows: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's plant index and target-day index in the record.

    A row's target day is the date of its hour's midpoint, which is 30 minutes before its label.

    Args:
        record: The record.
        rows: Rows with `site` and `time` (UTC hour ends).

    Returns:
        The plant indices and the day indices.

    Raises:
        ValueError: If a row's site is not in the record or its day is outside it.
    """
    site_index = {site: index for index, site in enumerate(record.sites)}
    unknown = set(rows["site"].unique().to_list()) - set(site_index)
    if unknown:
        msg = f"rows hold sites {sorted(unknown)} that the record lacks"
        raise ValueError(msg)
    plants = np.array([site_index[site] for site in rows["site"].to_list()], dtype=np.int64)
    midpoint_dates = (
        rows.select((pl.col("time") - pl.duration(minutes=30)).dt.date().alias("date"))["date"]
        .to_numpy()
        .astype("datetime64[D]")
    )
    days = (midpoint_dates - record.dates[0]).astype(np.int64)
    if days.min() < 0 or days.max() >= len(record.dates):
        msg = "a row's target day falls outside the record"
        raise ValueError(msg)
    return plants, days


def scale_for_rows(
    *, record: CapacityRecord, scale_mw: np.ndarray, rows: pl.DataFrame
) -> np.ndarray:
    """Return the scale for each forecast row, read at the row's plant and target day.

    Args:
        record: The record.
        scale_mw: A scale of shape (plants, days) indexed by target day.
        rows: Rows with `site` and `time`.

    Returns:
        One scale per row, in the rows' order.
    """
    plants, days = target_day_indices(record=record, rows=rows)
    return scale_mw[plants, days]


def scramble_from_day(*, record: CapacityRecord, first_day: int, seed: int) -> CapacityRecord:
    """Return the record with every input from `first_day` on replaced by noise.

    The noise covers measured power (including the days it was missing), the satellite irradiance,
    and the power modelled from it. The clear-sky irradiance is astronomy known in advance, so it
    stays.

    Args:
        record: The record.
        first_day: The first day index to replace.
        seed: The noise generator's seed.

    Returns:
        The record.
    """
    rng = np.random.default_rng(seed)
    shape = record.power_mw[:, first_day:].shape
    power = record.power_mw.copy()
    power[:, first_day:] = rng.uniform(0.0, 1.5, shape) * record.static_p99_mw[:, None, None]
    ghi = record.cams_ghi_w_m2.copy()
    ghi[:, first_day:] = rng.uniform(0.0, 1000.0, shape)
    modelled = record.cams_shape.copy()
    modelled[:, first_day:] = rng.uniform(0.0, 1.2, shape)
    return replace(record, power_mw=power, cams_ghi_w_m2=ghi, cams_shape=modelled)


ScaleFunction = Callable[[CapacityRecord, np.ndarray], np.ndarray]
"""Maps a record and target-day indices to the scale of every plant on those days, shape
(plants, len(days))."""


def assert_scale_ignores_the_issue_day_onward(
    *, record: CapacityRecord, scale_function: ScaleFunction, target_days: np.ndarray, seed: int
) -> None:
    """Raise if a target day's scale changes when every input on or after its issue day is noise.

    The issue day of target day `T` is `T - 1`; a causal scale reads days up to `T - 2` only.

    Args:
        record: The record.
        scale_function: Computes the scales of a record at the given target days.
        target_days: The target-day indices to probe.
        seed: The noise generator's seed.

    Raises:
        ValueError: Naming the plant and target day whose scale moved.
    """
    for target in target_days:
        original = scale_function(record, np.array([target]))
        scrambled = scale_function(
            scramble_from_day(record=record, first_day=int(target) - 1, seed=seed + int(target)),
            np.array([target]),
        )
        if not np.array_equal(original, scrambled, equal_nan=True):
            moved = int(
                np.flatnonzero(~np.isclose(original[:, 0], scrambled[:, 0], equal_nan=True))[0]
            )
            msg = (
                f"the scale for plant {record.sites[moved]} on target day index {int(target)} "
                "changed when inputs from its issue day onward were replaced by noise"
            )
            raise ValueError(msg)


def synthetic_record(
    *,
    plants: int = 3,
    first_day: date = date(2023, 6, 1),
    days: int = 680,
    capacity_factor: np.ndarray | None = None,
    seed: int = 0,
) -> CapacityRecord:
    """Build a record with a known capacity, for tests and for the scripts' self-checks.

    The plants share one cloud index per day and have clear-sky irradiance with a seasonal swing.
    A plant's power is its capacity factor times the modelled power from the satellite irradiance,
    with 4% noise. The first plant is a little larger than the others.

    Args:
        plants: How many plants.
        first_day: The first date.
        days: How many days.
        capacity_factor: The capacity as a share of normal, shape (plants, days), or `None` for 1.
        seed: The noise generator's seed.

    Returns:
        The record, with every hour present.
    """
    rng = np.random.default_rng(seed)
    dates = np.datetime64(first_day) + np.arange(days)
    day_of_year = (dates - dates.astype("datetime64[Y]")).astype(int)
    seasonal = 0.55 + 0.45 * (1.0 + np.cos(2 * np.pi * (day_of_year - 172) / 365.0)) / 2.0
    hours = np.arange(HOURS_PER_DAY)
    curve = np.clip(np.sin(np.pi * (hours - 5.5) / 13.0), 0.0, None)
    clear = 900.0 * seasonal[:, None] * curve[None, :]
    cloud = np.clip(rng.beta(2.0, 1.2, days), 0.12, 1.0)
    grid = (plants, days, HOURS_PER_DAY)
    ghi = clear[None] * cloud[None, :, None] * (1.0 + 0.03 * rng.standard_normal(grid))
    ghi = np.clip(ghi, 0.0, None)
    sizes = np.array([1.0 + 0.2 * (p == 0) + 0.1 * p for p in range(plants)])
    factor = np.ones((plants, days)) if capacity_factor is None else capacity_factor
    shape = ghi / 1000.0 * 1.1
    power = (
        sizes[:, None, None] * factor[:, :, None] * shape * (1.0 + 0.04 * rng.standard_normal(grid))
    )
    power = np.clip(power, 0.0, None)
    static = np.array([np.quantile(power[p][power[p] > 0], 0.99) for p in range(plants)])
    clear_all = np.broadcast_to(clear[None], grid).copy()
    return CapacityRecord(
        sites=tuple("ABCDEFGH"[:plants]),
        dates=dates,
        power_mw=power,
        clear_sky_w_m2=clear_all,
        cams_ghi_w_m2=ghi,
        cams_shape=shape,
        clear_shape=clear_all / 1000.0 * 1.1,
        static_p99_mw=static,
    )
