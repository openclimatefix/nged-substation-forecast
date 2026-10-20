"""Metering-error injections and the frozen-fit score of the pilot (experiment 124).

**The score is the mean absolute error of the frozen fit's forecast over the year after the cut,
in points of effective capacity, on rows the hand-filtered switching events do not cover.** It
replaces the plan's day-D+1 quantile score and Kalman filter, which the pilot does not run, and
the page says so. `nged_data.cleaning` flags only an exact zero, so a stuck non-zero value of any
length is never flagged.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import numpy as np
from e00124_common import HELD_OUT, to_64
from studies.dp_features import FeatureCache
from studies.dp_gam import design_matrix, solar_shape
from studies.dp_hierarchy import JointAssembly, JointFit

ROWS_PER_DAY: Final[int] = 48
KINDS: Final[tuple[tuple[str, str, float, int], ...]] = (
    ("gap 6 h", "gap", 0.0, 12),
    ("gap 24 h", "gap", 0.0, 48),
    ("gap 72 h", "gap", 0.0, 144),
    ("stuck 1 d", "stuck", 0.0, ROWS_PER_DAY),
    ("stuck 3 d", "stuck", 0.0, 3 * ROWS_PER_DAY),
    ("stuck 7 d", "stuck", 0.0, 7 * ROWS_PER_DAY),
    ("scale x0.8 for 30 d", "scale", 0.8, 30 * ROWS_PER_DAY),
    ("scale x1.2 for 30 d", "scale", 1.2, 30 * ROWS_PER_DAY),
    ("scale x0.8 for 90 d", "scale", 0.8, 90 * ROWS_PER_DAY),
    ("scale x1.2 for 90 d", "scale", 1.2, 90 * ROWS_PER_DAY),
)
"""Each injection cell: its label, its kind, the scale factor, and its length in half-hours."""
MIN_PRESENT: Final[float] = 0.8


@dataclass(frozen=True)
class ScoreRows:
    """One Primary's scored year: the held-capacity design, the observations, and the rows."""

    design: np.ndarray
    observed: np.ndarray
    rows: np.ndarray


def score_rows(
    *, cache: FeatureCache, assembly: JointAssembly, fit: JointFit, site: str, cut: datetime
) -> ScoreRows:
    """Build the design of a Primary's scored year with its capacity held after its last signal."""
    features = cache.sites[site]
    layout = assembly.layouts[site]
    end = min(cut.replace(year=cut.year + 1), HELD_OUT)
    keep = (
        (cache.end_times > to_64(moment=cut))
        & (cache.end_times <= to_64(moment=end))
        & features.present
        & np.isfinite(features.y)
        & np.isfinite(features.pv_unit_shape)
        & np.isfinite(features.smoothed_temperature_c).all(axis=1)
        & ~features.switching_mask
    )
    rows = np.flatnonzero(keep)
    x = design_matrix(cache=cache, features=features, layout=layout, index=rows)
    capacity = layout.slices["capacity"]
    held_year = np.minimum(cache.calendar.season_year[rows], fit.last_season_years[site])
    held = held_year[:, None] >= np.array(layout.season_years)[None, :]
    shape = solar_shape(cache=cache, features=features, layout=layout, index=rows)
    x[:, capacity] = -shape[:, None] * held
    return ScoreRows(design=x, observed=features.y[rows], rows=rows)


def mae(*, score: ScoreRows, fit: JointFit, site: str) -> np.ndarray:
    """Return the mean absolute error in points of each batch member of the fit."""
    theta = fit.theta[:, fit.site_slices[site]]
    return np.abs(score.design @ theta.T - score.observed[:, None]).mean(axis=0)


def monthly_mae(
    *, cache: FeatureCache, score: ScoreRows, fit: JointFit, site: str
) -> dict[str, list[float]]:
    """Return each batch member's mean absolute error in each calendar month of the scored year."""
    theta = fit.theta[:, fit.site_slices[site]]
    error = np.abs(score.design @ theta.T - score.observed[:, None])
    months = cache.end_times[score.rows].astype("datetime64[M]")
    return {str(m): error[months == m].mean(axis=0).tolist() for m in np.unique(months)}


@dataclass(frozen=True)
class Injection:
    """One injected error: the replaced readings and the rows to leave out for the oracle arm."""

    label: str
    kind: str
    power: np.ndarray
    rows: np.ndarray


def draw_injection(
    *,
    cache: FeatureCache,
    assembly: JointAssembly,
    site: str,
    cell: tuple[str, str, float, int],
    cut: datetime,
    rng: np.random.Generator,
) -> Injection | None:
    """Draw one injection in the 12 months before the cut, or `None` if no window has the rows."""
    label, kind, factor, length = cell
    index = assembly.indices[site]
    y = cache.sites[site].y
    recent = index[cache.end_times[index] > to_64(moment=cut - timedelta(days=365))]
    if len(recent) <= length:
        return None
    for _ in range(50):
        start = int(rng.choice(recent[: max(len(recent) - length, 1)]))
        window = np.arange(start, start + length)
        if np.isfinite(y[window]).mean() < MIN_PRESENT or not np.isfinite(y[start - 1]):
            continue
        power = y.copy()
        if kind == "stuck":
            power[window] = y[start - 1]
        elif kind == "scale":
            power[window] = factor * y[window]
        return Injection(label=label, kind=kind, power=power, rows=window)
    return None
