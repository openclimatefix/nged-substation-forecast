"""The two statistical detectors of experiment 328, both as-of: they read data up to the origin.

`cusum_alarms` is the rolling lagless GAM detector: a two-sided CUSUM on the daily mean of the
out-of-sample residual. `weekly_features` and `histogram_alarms` are the histogram detector: per
week, how much better two Gaussian components describe the half-hourly power (and residual) than
one, plus the 5% and 95% quantiles, segmented by penalised optimal partitioning.
"""

from __future__ import annotations

from dataclasses import dataclass

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00328_common as c
import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class Alarm:
    """One alarm: the day it is raised and the day the detector says the change began."""

    alarm_day: np.datetime64
    change_day: np.datetime64


def daily_means(*, end_times: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the days and the mean residual of each day with at least 40 half-hours."""
    day = c.day_of(end_times=end_times)
    unique, inverse = np.unique(day, return_inverse=True)
    count = np.bincount(inverse)
    mean = np.bincount(inverse, weights=residual) / np.maximum(count, 1)
    keep = count >= 40
    return unique[keep], mean[keep]


def cusum_alarms(*, days: np.ndarray, means: np.ndarray, h: float) -> list[Alarm]:
    """Return the alarms of a two-sided CUSUM on standardised daily mean residuals.

    Each day is standardised by the median and the robust spread of the previous 180 days, so the
    statistic reads nothing from the day itself or from later days. An alarm starts a 120-day
    refractory period, because a persistent step keeps the out-of-sample residual biased for
    months and would otherwise alarm repeatedly.
    """
    alarms: list[Alarm] = []
    up = down = 0.0
    up_start = down_start = 0
    quiet_until = -1
    for t in range(len(days)):
        if t < c.CUSUM_MIN_DAYS:
            continue
        lo = np.searchsorted(days, days[t] - np.timedelta64(c.CUSUM_TRAILING_DAYS, "D"))
        past = means[lo:t]
        if len(past) < c.CUSUM_MIN_DAYS:
            continue
        centre = float(np.median(past))
        spread = 1.4826 * float(np.median(np.abs(past - centre)))
        if spread <= 0 or t <= quiet_until:
            continue
        z = (means[t] - centre) / spread
        if up == 0.0:
            up_start = t
        if down == 0.0:
            down_start = t
        up = max(0.0, up + z - c.CUSUM_K)
        down = max(0.0, down - z - c.CUSUM_K)
        if up > h or down > h:
            start = up_start if up > h else down_start
            alarms.append(Alarm(alarm_day=days[t], change_day=days[start]))
            up = down = 0.0
            quiet_until = int(np.searchsorted(days, days[t] + np.timedelta64(c.REFRACTORY_DAYS, "D")))
    return alarms


def _delta_bic(*, values: np.ndarray) -> float:
    """Return BIC of one component minus BIC of two; positive means two components fit better."""
    x = values.reshape(-1, 1)
    one = GaussianMixture(1, random_state=0).fit(x).bic(x)
    two = GaussianMixture(2, random_state=0, n_init=1).fit(x).bic(x)
    return float(one - two)


def weekly_features(
    *, end_times: np.ndarray, power: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return week-start dates and four features per week (needs at least 200 half-hours).

    The features are the two-component gain in BIC of the week's power and of the week's
    residual, and the 5% and 95% quantiles of the week's power.
    """
    day = c.day_of(end_times=end_times)
    week = (day.astype("datetime64[D]").astype(np.int64) // 7).astype(np.int64)
    out_weeks: list[int] = []
    out: list[list[float]] = []
    for w in np.unique(week):
        pick = week == w
        if pick.sum() < 200:
            continue
        p, r = power[pick], residual[pick]
        out_weeks.append(int(w))
        out.append([_delta_bic(values=p), _delta_bic(values=r), *np.percentile(p, [5, 95])])
    starts = np.array(out_weeks, dtype=np.int64) * 7
    return starts.astype("datetime64[D]"), np.array(out, dtype=float)


def histogram_alarms(
    *, weeks: np.ndarray, features: np.ndarray, beta: float, origins: range | None = None
) -> list[Alarm]:
    """Return the first-appearance alarms of an as-of segmentation of the weekly features.

    At each weekly origin the segmentation runs on the weeks up to the origin. A boundary at least
    4 weeks old that is more than 8 weeks from every boundary already alarmed raises an alarm.
    """
    alarms: list[Alarm] = []
    seen: list[np.datetime64] = []
    n = len(weeks)
    for origin in origins if origins is not None else range(2 * c.HISTOGRAM_MIN_WEEKS, n):
        part = features[: origin + 1]
        sigma = c295.noise_scale(values=part)
        starts = c295.segment(
            values=part, sigma=sigma, beta=beta, min_length=c.HISTOGRAM_MIN_WEEKS
        )
        for s in starts:
            if s > origin - c.HISTOGRAM_MIN_WEEKS:
                continue
            date = weeks[s]
            if any(abs((date - old) / np.timedelta64(1, "D")) <= 56 for old in seen):
                continue
            seen.append(date)
            alarms.append(Alarm(alarm_day=weeks[origin] + np.timedelta64(7, "D"), change_day=date))
    return alarms
