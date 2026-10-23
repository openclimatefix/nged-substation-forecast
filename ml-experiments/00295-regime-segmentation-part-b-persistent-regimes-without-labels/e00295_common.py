"""Residual, daily distribution features, penalised segmentation, and boundary labelling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
from studies.dp_features import (
    FeatureCache,
    default_input_paths,
    load_or_build_feature_cache,
    naive_utc,
)
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00295-regime-segmentation-part-b-persistent-regimes-without-labels"
)
CACHE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00106-differentiable-physics-4-all-years-drift-and-recent-level"
    / "cache"
)
SEASON_YEARS: Final[range] = range(2019, 2027)
RIDGE: Final[float] = 0.1
END_OF_DATA: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
D8_BREAK: Final[np.datetime64] = np.datetime64("2023-08-15", "D")
BETAS: Final[tuple[float, ...]] = (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
MIN_LENGTHS: Final[tuple[int, ...]] = (7, 14, 30)
RETURN_DAYS: Final[int] = 120
MIRROR_DAYS: Final[int] = 3
PERSISTENT_MARGIN_DAYS: Final[int] = 150
MIN_HALF_HOURS_PER_DAY: Final[int] = 40


def load_cache() -> FeatureCache:
    """Return the feature cache of experiment 106."""
    return load_or_build_feature_cache(paths=default_input_paths(), cache_dir=CACHE_DIR)


def all_rows(*, cache: FeatureCache, site: str) -> np.ndarray:
    """Return the grid rows with a reading and finite model inputs."""
    features = cache.sites[site]
    keep = (
        features.present
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
    )
    return np.flatnonzero(keep)


def fit_and_predict(
    *, cache: FeatureCache, site: str, cut: datetime, window: TrainingWindow, index: np.ndarray
) -> np.ndarray:
    """Fit GAM_ONLY on the rows before `cut` and predict the rows `index`."""
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[GamSpec(ridge=RIDGE, drift="none", capacity="fitted", final_knot="free")],
        cut=cut,
        window=window,
        season_years=SEASON_YEARS,
    )
    return predict_gam(cache=cache, fit=fit, index=index)


FEATURES: Final[tuple[str, ...]] = ("dip_rate", "depth_p05", "top_p95", "shape")
CENTRE: Final[int] = 1
"""Index of `depth_p05`; the centre of a boundary's shift is the mean of the 5% and 95% shifts."""
DIP_SIGMAS: Final[float] = 2.5
RETURN_TOLERANCE_Z: Final[float] = 1.0


@dataclass(frozen=True)
class DailySeries:
    """Four daily features of one Primary's residual, in points of effective capacity.

    Attributes:
        days: The UTC date of each day (`datetime64[D]`).
        values: Shape (days, 4): the share of the day's half-hours more than 2.5 half-hour noise
            units below zero (the dip rate), the day's 5% and 95% residual quantiles (the depth of
            the dips and the height of the top), and the 00:00 to 06:00 mean minus the 10:00 to
            16:00 mean (the daily shape).
        half_hour_sigma: The Primary's half-hour residual noise scale, in points.
    """

    days: np.ndarray
    values: np.ndarray
    half_hour_sigma: float


def daily_series(*, end_times: np.ndarray, residual: np.ndarray) -> DailySeries:
    """Aggregate a half-hourly residual to the four daily distribution features."""
    start = end_times - np.timedelta64(1, "s")
    day = start.astype("datetime64[D]")
    hour = (start - day.astype("datetime64[us]")) // np.timedelta64(1, "h")
    unique, inverse = np.unique(day, return_inverse=True)
    n = len(unique)
    count = np.bincount(inverse, minlength=n)
    sigma_h = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    dip = np.bincount(
        inverse, weights=(residual < -DIP_SIGMAS * sigma_h).astype(float), minlength=n
    )
    order = np.argsort(inverse, kind="stable")
    edges = np.concatenate([[0], np.cumsum(count)])
    p05 = np.full(n, np.nan)
    p95 = np.full(n, np.nan)
    for k in range(n):
        if count[k] >= MIN_HALF_HOURS_PER_DAY:
            values = residual[order[edges[k] : edges[k + 1]]]
            p05[k], p95[k] = np.percentile(values, [5, 95])

    def part(mask: np.ndarray) -> np.ndarray:
        weights = np.bincount(inverse[mask], minlength=n)
        sums = np.bincount(inverse[mask], weights=residual[mask], minlength=n)
        return np.where(weights >= 6, sums / np.maximum(weights, 1), np.nan)

    shape = part(hour < 6) - part((hour >= 10) & (hour < 16))
    keep = (count >= MIN_HALF_HOURS_PER_DAY) & np.isfinite(shape)
    values = np.column_stack([dip / np.maximum(count, 1), p05, p95, shape])[keep]
    return DailySeries(days=unique[keep], values=values, half_hour_sigma=sigma_h)


def noise_scale(*, values: np.ndarray) -> np.ndarray:
    """Return a robust noise scale per column, from day-to-day changes.

    The scale is floored at a quarter of the column's standard deviation, because the dip rate is
    zero on most days and its day-to-day changes would otherwise give a scale of zero.
    """
    diff = np.diff(values, axis=0)
    mad = np.median(np.abs(diff - np.median(diff, axis=0)), axis=0)
    return np.maximum(1.4826 * mad / np.sqrt(2.0), 0.25 * values.std(axis=0) + 1e-9)


def segment(*, values: np.ndarray, sigma: np.ndarray, beta: float, min_length: int) -> list[int]:
    """Return the start index of every segment after the first, by penalised optimal partitioning.

    Args:
        values: Shape (days, columns) series.
        sigma: Noise scale per column; the L2 cost is computed on `values / sigma`.
        beta: The penalty per break, in units of the log of the number of days.
        min_length: The shortest allowed segment, in days.

    Returns:
        The sorted start indices of the second and later segments.
    """
    x = values / sigma
    n = len(x)
    if n < 2 * min_length:
        return []
    s1 = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    s2 = np.concatenate([[0.0], np.cumsum((x**2).sum(axis=1))])
    penalty = beta * np.log(n) * x.shape[1] / 2
    best = np.full(n + 1, np.inf)
    previous = np.zeros(n + 1, dtype=np.int64)
    best[0] = -penalty
    for t in range(min_length, n + 1):
        first_starts = np.arange(0, t - min_length + 1)
        starts = first_starts[(first_starts == 0) | (first_starts >= min_length)]
        length = (t - starts).astype(float)
        diff = s1[t] - s1[starts]
        cost = s2[t] - s2[starts] - (diff**2).sum(axis=1) / length
        total = best[starts] + cost + penalty
        k = int(np.argmin(total))
        best[t], previous[t] = total[k], starts[k]
    bounds: list[int] = []
    t = n
    while t > 0:
        s = int(previous[t])
        if s > 0:
            bounds.append(s)
        t = s
    return sorted(bounds)


@dataclass(frozen=True)
class Boundary:
    """One segment boundary.

    Attributes:
        date: The first day of the new segment.
        shift: Mean after minus mean before, on the four features.
        before: Mean features of the segment before.
        after_next: Mean features of the segment after the next boundary, or `nan`.
        days_to_next: Days from this boundary to the next, or -1 when there is none.
        label: `temporary` (a state inside a regime), `persistent` (a regime change), or
            `unconfirmed`.
    """

    date: np.datetime64
    shift: np.ndarray
    before: np.ndarray
    after_next: np.ndarray
    days_to_next: int
    label: str = ""

    @property
    def centre_shift(self) -> float:
        """Return the mean shift of the 5% and 95% quantiles, in points."""
        return float(self.shift[CENTRE : CENTRE + 2].mean())


def boundaries_of(*, series: DailySeries, starts: list[int]) -> list[Boundary]:
    """Describe each boundary of a segmentation, without a label."""
    edges = [0, *starts, len(series.days)]
    out = []
    for k in range(1, len(edges) - 1):
        before = series.values[edges[k - 1] : edges[k]]
        after = series.values[edges[k] : edges[k + 1]]
        after_next = series.values[edges[k + 1] : edges[k + 2]] if k + 2 < len(edges) else None
        out.append(
            Boundary(
                date=series.days[edges[k]],
                shift=after.mean(axis=0) - before.mean(axis=0),
                before=before.mean(axis=0),
                after_next=np.full(before.shape[1], np.nan)
                if after_next is None
                else after_next.mean(axis=0),
                days_to_next=-1
                if k + 2 >= len(edges)
                else int(
                    (series.days[edges[k + 1]] - series.days[edges[k]]) / np.timedelta64(1, "D")
                ),
            )
        )
    return out


def label_boundaries(
    *,
    boundaries: list[Boundary],
    neighbours: list[list[Boundary]],
    sigma: np.ndarray,
    last_day: np.datetime64,
) -> list[Boundary]:
    """Label each boundary temporary (a state), persistent (a regime change), or unconfirmed.

    Args:
        boundaries: The Primary's own boundaries.
        neighbours: The boundaries of the five most correlated Primaries.
        sigma: The Primary's noise scale on each feature.
        last_day: The last day of the Primary's series.

    Returns:
        The same boundaries with `label` set.
    """
    out = []
    for b in boundaries:
        returns = (
            b.days_to_next >= 0
            and b.days_to_next <= RETURN_DAYS
            and bool(np.all(np.abs(b.after_next - b.before) / sigma <= RETURN_TOLERANCE_Z))
        )
        mirrored = any(
            abs((n.date - b.date) / np.timedelta64(1, "D")) <= MIRROR_DAYS
            and n.centre_shift * b.centre_shift < 0
            and abs(n.centre_shift) >= 0.5 * abs(b.centre_shift)
            for others in neighbours
            for n in others
        )
        late = (last_day - b.date) / np.timedelta64(1, "D") < PERSISTENT_MARGIN_DAYS
        if returns or mirrored:
            label = "temporary"
        elif late:
            label = "unconfirmed"
        else:
            label = "persistent"
        out.append(replace(b, label=label))
    return out


def days_between(a: np.datetime64, b: np.datetime64) -> float:
    """Return b minus a in days."""
    return float((b - a) / np.timedelta64(1, "D"))


__all__ = [
    "naive_utc",
]
