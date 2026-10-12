"""POST HOC capacity estimators built on the Kalman filter's level (DSKF and HYST).

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>.
**Neither estimator was in the pre-registered analysis.** A science review found that the Kalman
filter's raw level has a seasonal shape caused by the physics model, which the planted-loss ratio
cancels, so the tuning draw rewarded agility. DSKF removes that shape causally. HYST then holds the
scale at the static power until a drop persists.

**DSKF** divides each plant's level by a cross-plant day-of-year profile computed only from the
calendar years before the day's own year. **HYST** holds the scale at the static power until the
deseasonalised level has been more than 10% below it for 7 days, follows the level while it stays
down, and releases to the static power once the level has been back within 10% for 7 days.
"""

from typing import Final

import numpy as np

from studies.capacity_estimators import RawEstimate

PROFILE_HALF_WINDOW_DAYS: Final[int] = 7
"""A day-of-year profile value pools the days within this many days either side, around the year."""

MIN_PROFILE_VALUES: Final[int] = 30
"""A profile value needs this many finite pooled levels, else its days have no level."""

DAYS_IN_PROFILE: Final[int] = 365
"""The profile's length; day 366 of a leap year uses day 365's value."""

HYSTERESIS_DROP: Final[float] = 0.10
"""The share below the static power that counts as a drop (and, once following, as recovered)."""

HYSTERESIS_PERSIST_DAYS: Final[int] = 7
"""How many consecutive days a drop, or a recovery, must last before the state changes."""


def day_of_year(*, dates: np.ndarray) -> np.ndarray:
    """Return each date's zero-based day of the year, capped at 364."""
    years = dates.astype("datetime64[Y]")
    return np.minimum(
        (dates.astype("datetime64[D]") - years.astype("datetime64[D]")).astype(int), 364
    )


def profile_before(*, log_level: np.ndarray, doy: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Return the cross-plant logarithmic day-of-year profile of the days flagged `prior`.

    Args:
        log_level: Logarithmic levels, shape (plants, days).
        doy: Zero-based day of the year per day.
        prior: Which days may be read, shape (days,).

    Returns:
        A profile of length `DAYS_IN_PROFILE`, NaN where fewer than `MIN_PROFILE_VALUES` levels
        were pooled. Each plant is first centred on its own median over the prior days.
    """
    values = log_level[:, prior]
    centred = values - np.nanmedian(values, axis=1, keepdims=True)
    days = doy[prior]
    profile = np.full(DAYS_IN_PROFILE, np.nan)
    for k in range(DAYS_IN_PROFILE):
        gap = np.abs(days - k)
        near = np.minimum(gap, DAYS_IN_PROFILE - gap) <= PROFILE_HALF_WINDOW_DAYS
        pooled = centred[:, near]
        pooled = pooled[np.isfinite(pooled)]
        if len(pooled) >= MIN_PROFILE_VALUES:
            profile[k] = np.median(pooled)
    return profile


def deseasonalise_causally(*, raw: RawEstimate, dates: np.ndarray) -> RawEstimate:
    """Divide a level by its cross-plant day-of-year profile from earlier calendar years only.

    **The profile for a day in year Y reads only days before 1 January of year Y,** so nothing from
    the issue day or later enters. **A day in the record's first calendar year has no earlier year,
    so its level is NaN and the issue-time scale falls back to the static power.**

    Args:
        raw: The level (and standard deviation), shape (plants, days).
        dates: One calendar day per column, consecutive, as `datetime64[D]`.

    Returns:
        The deseasonalised level and standard deviation, in the same units as `raw`.
    """
    with np.errstate(all="ignore"):
        log_level = np.where(raw.level > 0, np.log(np.where(raw.level > 0, raw.level, 1.0)), np.nan)
    doy = day_of_year(dates=dates)
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    level = np.full(raw.level.shape, np.nan)
    sd = None if raw.sd is None else np.full(raw.level.shape, np.nan)
    for year in np.unique(years):
        prior = years < year
        if not prior.any():
            continue
        profile = profile_before(log_level=log_level, doy=doy, prior=prior)
        today = years == year
        factor = np.exp(profile[doy[today]])
        level[:, today] = raw.level[:, today] / factor
        if sd is not None and raw.sd is not None:
            sd[:, today] = raw.sd[:, today] / factor
    return RawEstimate(level=level, sd=sd)


def hysteresis_level(
    *,
    ratio: np.ndarray,
    drop: float = HYSTERESIS_DROP,
    persist_days: int = HYSTERESIS_PERSIST_DAYS,
) -> np.ndarray:
    """Hold at 1 until a drop persists, follow the ratio while it stays down, then release.

    A NaN ratio changes nothing: the state and the last level carry over.

    Args:
        ratio: One plant's deseasonalised level over its reference level (1 is normal).
        drop: The share below 1 that counts as a drop.
        persist_days: Consecutive days needed to start following and to release.

    Returns:
        The level over the same days, 1 while held. Day `d` reads days up to `d` only.
    """
    out = np.ones(len(ratio))
    following = False
    run = 0
    for day, value in enumerate(ratio):
        if np.isfinite(value):
            down = value < 1.0 - drop
            run = run + 1 if down != following else 0
            if run >= persist_days:
                following = not following
                run = 0
        carried = out[day - 1] if day else 1.0
        out[day] = (value if np.isfinite(value) else carried) if following else 1.0
    return out
