"""Scores of a recovered weekly capacity path against a known one (experiment 00045).

Every function takes plain arrays of length `common.N_WEEKS` (or any common length), in MW, with NaN
for weeks that hold no data. The functions know nothing about which units are tuning or scoring
units; the caller decides that.
"""

from typing import Final

import e00045_common as common
import numpy as np

STEP_WINDOW_WEEKS: Final[int] = 4
"""A false step is a rise over at most this many consecutive weeks."""
WEEKS_PER_YEAR: Final[float] = 365.25 / 7
YEAR_WEEKS: Final[int] = 52


def path_error(
    *, estimate: np.ndarray, truth: np.ndarray, weeks: np.ndarray, scale: float | None = None
) -> float:
    """Return the mean absolute path difference over `weeks`, as a share of a capacity scale.

    Args:
        estimate: Recovered capacity of every week, MW.
        truth: True capacity of every week, MW.
        weeks: Boolean mask of the weeks that hold data.
        scale: The capacity to divide by, MW; the mean of `truth` over `weeks` when `None`.

    Returns:
        The mean of `|estimate - truth|` over `weeks`, divided by `scale`.
    """
    divisor = float(np.mean(truth[weeks])) if scale is None else scale
    return float(np.mean(np.abs(estimate[weeks] - truth[weeks])) / divisor)


def count_steps(
    *,
    path: np.ndarray,
    threshold: float,
    exempt: np.ndarray | None = None,
    window: int = STEP_WINDOW_WEEKS,
) -> int:
    """Count the rises of at least `threshold` MW within `window` consecutive weeks.

    The scan moves left to right. A rise starting at week `t` is counted when some later week
    `t + k` (k up to `window`) is at least `threshold` above week `t`; the scan then resumes
    `window` weeks later so that one rise is counted once. Weeks without a path value (NaN) are
    skipped, and a rise starting in an exempt week is not counted.

    Args:
        path: Capacity of every week, MW (NaN outside the fitted span).
        threshold: Smallest rise that counts, MW.
        exempt: Boolean mask of weeks in which a rise is not counted, or `None`.
        window: Number of consecutive weeks over which a rise is measured.

    Returns:
        The number of rises.
    """
    count, week = 0, 0
    n = len(path)
    while week < n - 1:
        if np.isnan(path[week]):
            week += 1
            continue
        ahead = path[week + 1 : week + 1 + window]
        rises = np.nan_to_num(ahead - path[week], nan=-np.inf)
        if rises.size and rises.max() >= threshold and not (exempt is not None and exempt[week]):
            count += 1
            week += window
        else:
            week += 1
    return count


def steps_per_year(*, steps: int, n_weeks: int) -> float:
    """Return `steps` as a rate per year of `n_weeks` weeks of data."""
    return steps / (n_weeks / WEEKS_PER_YEAR)


def rise_weeks(
    *, path: np.ndarray, threshold: float, window: int = STEP_WINDOW_WEEKS
) -> np.ndarray:
    """Return a mask of the weeks within `window` weeks (either side) of a rise of `threshold`.

    Args:
        path: Capacity of every week, MW.
        threshold: Smallest rise that counts, MW.
        window: Number of consecutive weeks over which a rise is measured, and the margin added.

    Returns:
        Boolean mask, True from `window` weeks before a rise starts to `window` weeks after it ends.
    """
    mask = np.zeros(len(path), dtype=bool)
    for week in range(len(path) - 1):
        if np.isnan(path[week]):
            continue
        rises = np.nan_to_num(path[week + 1 : week + 1 + window] - path[week], nan=-np.inf)
        if rises.size and rises.max() >= threshold:
            mask[max(0, week - window) : week + 2 * window + 1] = True
    return mask


def fall_recovered(
    *, estimate: np.ndarray, truth: np.ndarray, weeks: np.ndarray, start: int, duration: int
) -> float:
    """Return the share of a temporary fall's depth that the estimate shows.

    The share is the estimate's mean over the 8 observed weeks before the fall minus its mean over
    the observed weeks of the fall, divided by the same difference in the truth.

    Args:
        estimate: Recovered capacity of every week, MW.
        truth: True capacity of every week, MW.
        weeks: Boolean mask of the weeks that hold data.
        start: First week of the fall.
        duration: Number of weeks of the fall.

    Returns:
        The share recovered (1 is the full depth; it can be negative or above 1), or NaN when
        the fall or the weeks before it hold no data.
    """
    inside = weeks.copy()
    inside[:start] = False
    inside[start + duration :] = False
    before = weeks.copy()
    before[start:] = False
    before[: max(0, start - 8)] = False
    if not inside.any() or not before.any():
        return float("nan")
    return float(
        (estimate[before].mean() - estimate[inside].mean())
        / (truth[before].mean() - truth[inside].mean())
    )


def ratchet_drift_per_5_years(*, path: np.ndarray, reference: float) -> float:
    """Return the rise of the path from its first to its last year, per 5 years, in capacity units.

    Args:
        path: Capacity of every week, MW (NaN outside the fitted span).
        reference: The capacity that the drift is divided by, MW.

    Returns:
        The mean of the last 52 fitted weeks minus the mean of the first 52, scaled to 5 years of
        separation and divided by `reference`.
    """
    fitted = path[~np.isnan(path)]
    years_apart = max((len(fitted) - YEAR_WEEKS) / WEEKS_PER_YEAR, 1.0)
    rise = fitted[-YEAR_WEEKS:].mean() - fitted[:YEAR_WEEKS].mean()
    return float(rise / years_apart * 5 / reference)


def seasonal_swing(*, estimate: np.ndarray, truth: np.ndarray, weeks: np.ndarray) -> float:
    """Return the seasonal swing of the error, as a share of the true path's mean.

    The swing is the largest minus the smallest of the four seasonal mean errors (estimate minus
    truth) over the observed weeks, where a week's season is the quarter, counted from September,
    of the month that its Monday falls in.

    Args:
        estimate: Recovered capacity of every week, MW.
        truth: True capacity of every week, MW.
        weeks: Boolean mask of the weeks that hold data.

    Returns:
        The swing divided by the mean of `truth` over the observed weeks.
    """
    index = np.nonzero(weeks)[0]
    quarter = np.array([((common.week_start(week=int(w)).month - 9) % 12) // 3 for w in index])
    error = estimate[index] - truth[index]
    means = [error[quarter == q].mean() for q in range(4) if (quarter == q).any()]
    return float((max(means) - min(means)) / truth[index].mean())
