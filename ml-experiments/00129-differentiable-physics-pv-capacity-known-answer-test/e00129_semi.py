"""The semi-synthetic target and the placebo sky of experiment 129.

The semi-synthetic target of a Primary is its sun-blind demand prediction plus residuals resampled
by calendar month, so its true solar capacity is zero and its noise is realistic. Adding a known
farm to the target gives a case whose true capacity is known, for coverage, drift, and the
placebo. The placebo sky replaces the fit's PV shape with the previous year's shape on the same
dates, so the day-to-day weather no longer matches the target.
"""

from dataclasses import replace

import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import GamFit, predict_gam, training_rows

import e00129_common as common  # isort: skip


def demand_and_residual(
    *, cache: FeatureCache, site: str, fit: GamFit
) -> tuple[np.ndarray, np.ndarray]:
    """Return the demand terms alone and the residual of the full fit, on the cache's grid.

    The demand terms are the fit's prediction with the capacity columns switched off. The residual
    is the reading minus the full prediction. Both are `NaN` on rows that cannot train.

    Args:
        cache: The cache.
        site: The substation's label.
        fit: A fit of the substation with a fitted capacity.

    Returns:
        The demand terms and the residual, each on the cache's grid.
    """
    features = cache.sites[site]
    index = _training_index(cache=cache, site=site, fit=fit)
    keep = np.zeros(fit.theta.shape[1], dtype=bool)
    capacity = fit.layout.slices["capacity"]
    keep[:] = True
    keep[capacity] = False
    demand = np.full(len(cache.end_times), np.nan)
    residual = np.full(len(cache.end_times), np.nan)
    demand[index] = predict_gam(cache=cache, fit=fit, index=index, columns=keep)
    residual[index] = features.y[index] - predict_gam(cache=cache, fit=fit, index=index)
    return demand, residual


def _training_index(*, cache: FeatureCache, site: str, fit: GamFit) -> np.ndarray:
    """Return the rows the fit trained on."""
    return training_rows(
        features=cache.sites[site],
        cache=cache,
        cut=fit.cut,
        window=common.TRAINING_WINDOW,
        keep_lockdown=fit.layout.spec.covid_flag,
    )


def resample_by_month(*, residual: np.ndarray, end_times: np.ndarray, seed: int) -> np.ndarray:
    """Return residuals in which each calendar month is replaced by the same month of another year.

    A target month takes the residuals of a source month with the same month of the year, drawn at
    random with replacement from the months that have residuals, row by row from the start of the
    month and tiled if the source is shorter. A row whose own residual is `NaN` stays `NaN`.

    Args:
        residual: The residual on the grid, `NaN` where it does not exist.
        end_times: The grid's half-hour ends.
        seed: Seeds the draws.

    Returns:
        The resampled residual on the grid.
    """
    generator = np.random.default_rng(seed)
    months = end_times.astype("datetime64[M]")
    month_number = months.astype(np.int64)
    out = np.full(len(residual), np.nan)
    finite = np.isfinite(residual)
    sources: dict[int, list[np.ndarray]] = {}
    for label in np.unique(month_number[finite]):
        rows = np.flatnonzero((month_number == label) & finite)
        if len(rows) > 0.5 * np.sum(month_number == label):
            sources.setdefault(int(label % 12), []).append(rows)
    for label in np.unique(month_number):
        target = np.flatnonzero((month_number == label) & finite)
        candidates = sources.get(int(label % 12))
        if not candidates or len(target) == 0:
            continue
        source = candidates[generator.integers(len(candidates))]
        out[target] = residual[source[np.arange(len(target)) % len(source)]]
    return out


def semi_synthetic_cache(
    *, cache: FeatureCache, site: str, demand: np.ndarray, resampled: np.ndarray
) -> FeatureCache:
    """Return the cache with the substation's reading replaced by demand plus resampled residual.

    The solar capacity of the result is zero. The row is `NaN` wherever either part is `NaN`.
    """
    return common.with_reading(cache=cache, site=site, y=demand + resampled)


def placebo_sky(*, cache: FeatureCache, site: str) -> FeatureCache:
    """Return the cache with the substation's PV shape taken from the previous year.

    The shape at a half-hour is the shape at the same date and time one year earlier, so the
    seasonal and daily envelope is kept and the day-to-day weather is not. A half-hour in the first
    year, which has no previous year, gets `NaN`, so those rows drop out of the fit.
    """
    features = cache.sites[site]
    ends = cache.end_times
    month = ends.astype("datetime64[M]")
    previous = (month - np.timedelta64(12, "M")).astype("datetime64[us]") + (
        ends - month.astype("datetime64[us]")
    )
    position = np.searchsorted(ends, previous)
    position = np.minimum(position, len(ends) - 1)
    exists = (ends[position] == previous) & (previous >= ends[0])
    shifted = np.where(exists, features.pv_unit_shape[position], np.nan)
    return replace(cache, sites={**cache.sites, site: replace(features, pv_unit_shape=shifted)})
