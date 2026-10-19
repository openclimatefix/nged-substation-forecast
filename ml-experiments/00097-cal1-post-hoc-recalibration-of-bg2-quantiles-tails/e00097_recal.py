"""The recalibrations of experiment 97, each fitted on calibration rows and applied to test rows.

Every function takes the stored quantiles as a matrix with one column per level of `OUT_LEVELS`
(the 11 delivered levels plus four levels beyond q01 and q99, filled by a normal extrapolation) and
returns the recalibrated matrix of the same shape. Nothing here reads a file or knows about months:
the caller decides which rows are calibration rows, so the leakage rule lives in one place.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import stats

DELIVERED_LEVELS: Final[tuple[float, ...]] = (
    0.01,
    0.05,
    0.10,
    0.25,
    0.40,
    0.50,
    0.60,
    0.75,
    0.90,
    0.95,
    0.99,
)
"""The quantile levels stored in the BG2 prediction sets."""

EXTRA_LEVELS: Final[tuple[float, ...]] = (0.001, 0.005, 0.995, 0.999)
"""Levels beyond q01 and q99, read off an extrapolation or a fitted tail."""

OUT_LEVELS: Final[tuple[float, ...]] = tuple(sorted(DELIVERED_LEVELS + EXTRA_LEVELS))
"""Every level an arm outputs, lowest first."""

DELIVERED_COLUMNS: Final[tuple[int, ...]] = tuple(OUT_LEVELS.index(p) for p in DELIVERED_LEVELS)
"""Position of each delivered level inside `OUT_LEVELS`."""

MEDIAN_COLUMN: Final[int] = OUT_LEVELS.index(0.5)
"""Position of the median inside `OUT_LEVELS`."""

CENTRE_LEVELS: Final[tuple[float, ...]] = (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)
"""The levels whose mean pinball loss is the centre score."""

MINIMUM_SPREAD: Final[float] = 1e-3
"""Floor on a stored spread, as a fraction of capacity, so a collapsed spread cannot divide by 0."""

SCALE_LIMITS: Final[tuple[float, float]] = (0.5, 5.0)
"""The range a fitted scale factor is clipped to."""

GPD_SHAPE_LIMITS: Final[tuple[float, float]] = (-0.25, 0.5)
"""The range a fitted generalised Pareto shape is clipped to."""

MIN_TAIL_ROWS: Final[int] = 200
"""Fewest exceedances for which a Pareto shape is fitted; below it the tail is exponential."""

FIT_SUBSAMPLE: Final[int] = 20_000
"""Most rows handed to a distribution fit, drawn with a fixed seed."""

NORMAL_Z: Final[dict[float, float]] = {p: float(stats.norm.ppf(p)) for p in OUT_LEVELS}
"""The standard normal quantile of each output level."""


@dataclass(frozen=True)
class Rows:
    """The rows of one group: measurements and the stored quantile matrix.

    Attributes:
        actual: Measured value, shape (n,), as a fraction of capacity.
        quantiles: Stored quantiles with the extrapolated extra levels, shape (n, len(OUT_LEVELS)),
            sorted along the second axis.
    """

    actual: np.ndarray
    quantiles: np.ndarray


class ClipCounter:
    """Counts how often a fitted scale factor hit the limit of `SCALE_LIMITS`."""

    def __init__(self) -> None:
        """Start with no fits counted."""
        self.fits = 0
        self.clipped = 0

    def record(self, *, raw: float) -> float:
        """Clip one fitted scale factor, count it, and return the clipped value."""
        self.fits += 1
        low, high = SCALE_LIMITS
        if not low <= raw <= high:
            self.clipped += 1
        return float(np.clip(raw, low, high))


CLIPS: Final[ClipCounter] = ClipCounter()
"""Process-wide count of scale-factor fits and clipped fits, printed by the report."""


def extend_quantiles(*, delivered: np.ndarray) -> np.ndarray:
    """Build the (n, 15) matrix from the (n, 11) delivered quantiles, extrapolating the tails.

    A level beyond q01 (or q99) is `q01 + (q01 - q05) / (z_01 - z_05) * (z_level - z_01)` with `z`
    the standard normal quantile, so the extension has the normal spacing implied by q01 and q05.
    Each row is then sorted so that quantiles never cross.

    Args:
        delivered: Quantiles at `DELIVERED_LEVELS`, shape (n, 11).

    Returns:
        The quantiles at `OUT_LEVELS`, shape (n, 15), sorted along the second axis.
    """
    delivered = np.sort(delivered, axis=1)
    out = np.empty((delivered.shape[0], len(OUT_LEVELS)), dtype=np.float64)
    for j, p in zip(DELIVERED_COLUMNS, DELIVERED_LEVELS, strict=True):
        out[:, j] = delivered[:, DELIVERED_LEVELS.index(p)]
    lo1, lo5 = delivered[:, 0], delivered[:, 1]
    hi99, hi95 = delivered[:, -1], delivered[:, -2]
    for p in EXTRA_LEVELS:
        j = OUT_LEVELS.index(p)
        if p < 0.5:
            slope = (lo1 - lo5) / (NORMAL_Z[0.01] - NORMAL_Z[0.05])
            out[:, j] = lo1 + slope * (NORMAL_Z[p] - NORMAL_Z[0.01])
        else:
            slope = (hi99 - hi95) / (NORMAL_Z[0.99] - NORMAL_Z[0.95])
            out[:, j] = hi99 + slope * (NORMAL_Z[p] - NORMAL_Z[0.99])
    return np.sort(out, axis=1)


def as_stored(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arm 0: the stored quantiles, with the normal extrapolation beyond q01 and q99."""
    del calibration
    return test.quantiles.copy()


def global_spread(*, calibration: Rows, test: Rows, factor: float = 1.15) -> np.ndarray:
    """Arm 1: scale every level about the median by one global factor."""
    del calibration
    median = test.quantiles[:, [MEDIAN_COLUMN]]
    return np.sort(median + factor * (test.quantiles - median), axis=1)


def _scale_factors(*, calibration: Rows) -> np.ndarray:
    """Fit one scale factor per output level; the median's factor is 1.

    Levels beyond q01 and q99 take the factor of q01 and q99, because only delivered levels have a
    stored quantile to compare with.
    """
    median = calibration.quantiles[:, MEDIAN_COLUMN]
    factors = np.ones(len(OUT_LEVELS))
    fitted: dict[int, float] = {}
    for j, p in zip(DELIVERED_COLUMNS, DELIVERED_LEVELS, strict=True):
        if j == MEDIAN_COLUMN:
            continue
        distance = calibration.quantiles[:, j] - median
        usable = np.abs(distance) > MINIMUM_SPREAD / 10
        ratio = (calibration.actual[usable] - median[usable]) / distance[usable]
        level = p if p > 0.5 else 1.0 - p
        fitted[j] = CLIPS.record(raw=float(np.quantile(ratio, level)))
    lowest, highest = DELIVERED_COLUMNS[0], DELIVERED_COLUMNS[-1]
    for j in range(len(OUT_LEVELS)):
        if j == MEDIAN_COLUMN:
            continue
        nearest = lowest if j < lowest else min(j, highest)
        factors[j] = fitted[nearest]
    return factors


def scale_per_level(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arms 2 and 3: a scale factor about the median per level, fitted on the calibration rows."""
    factors = _scale_factors(calibration=calibration)
    median = test.quantiles[:, [MEDIAN_COLUMN]]
    return np.sort(median + factors[None, :] * (test.quantiles - median), axis=1)


def _isotonic_increasing(*, values: np.ndarray) -> np.ndarray:
    """Return the non-decreasing least-squares fit to `values` (pool adjacent violators)."""
    blocks: list[list[float]] = []
    for v in values:
        blocks.append([float(v), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            value, weight = blocks.pop()
            previous = blocks[-1]
            previous[0] = (previous[0] * previous[1] + value * weight) / (previous[1] + weight)
            previous[1] += weight
    return np.concatenate([np.full(int(w), v) for v, w in blocks])


def remap_levels(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arm 4: isotonic remapping of nominal level to empirical coverage, inverted per target.

    The calibration coverage at each delivered level is made non-decreasing by isotonic regression,
    anchored at (0, 0) and (1, 1). For each target level the function finds the lowest stored level
    whose fitted coverage reaches the target, and reads the stored quantile function at that level
    by linear interpolation (the extrapolated levels extend it beyond q01 and q99).
    """
    coverage = np.array(
        [
            float(np.mean(calibration.actual <= calibration.quantiles[:, j]))
            for j in DELIVERED_COLUMNS
        ]
    )
    levels = np.concatenate([[0.0], DELIVERED_LEVELS, [1.0]])
    fitted = _isotonic_increasing(values=np.concatenate([[0.0], coverage, [1.0]]))
    fitted = fitted + np.arange(len(fitted)) * 1e-12
    grid = np.array(OUT_LEVELS)
    out = np.empty_like(test.quantiles)
    for j, target in enumerate(OUT_LEVELS):
        level = float(np.interp(target, fitted, levels))
        level = float(np.clip(level, grid[0], grid[-1]))
        upper = int(np.searchsorted(grid, level))
        if upper == 0 or grid[upper] == level:
            out[:, j] = test.quantiles[:, upper]
            continue
        weight = (level - grid[upper - 1]) / (grid[upper] - grid[upper - 1])
        out[:, j] = (1 - weight) * test.quantiles[:, upper - 1] + weight * test.quantiles[:, upper]
    return np.sort(out, axis=1)


def conformal_shift(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arm 5: add to each level the calibration quantile of `actual - stored quantile`."""
    n = len(calibration.actual)
    out = np.empty_like(test.quantiles)
    for j, p in enumerate(OUT_LEVELS):
        score = calibration.actual - calibration.quantiles[:, j]
        corrected = min(1.0, math.ceil((n + 1) * p) / n)
        out[:, j] = test.quantiles[:, j] + float(np.quantile(score, corrected))
    return np.sort(out, axis=1)


def _subsample(*, values: np.ndarray, seed: int = 0) -> np.ndarray:
    """Return at most `FIT_SUBSAMPLE` of `values`, drawn with a fixed seed."""
    if len(values) <= FIT_SUBSAMPLE:
        return values
    return np.random.default_rng(seed).choice(values, size=FIT_SUBSAMPLE, replace=False)


def _fit_pareto(*, excess: np.ndarray) -> tuple[float, float]:
    """Fit a generalised Pareto distribution to positive excesses; return (shape, scale)."""
    sample = _subsample(values=excess)
    if len(sample) < MIN_TAIL_ROWS:
        return 0.0, float(np.mean(sample)) if len(sample) else MINIMUM_SPREAD
    shape, _, scale = stats.genpareto.fit(sample, floc=0.0)
    return float(np.clip(shape, *GPD_SHAPE_LIMITS)), float(scale)


def pareto_tails(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arm 6: keep q10 to q90 as stored and fit a generalised Pareto beyond q05 and q95.

    The excess of an actual value over the stored q95, divided by the stored spread, is modelled
    with a generalised Pareto distribution. The share of calibration rows beyond the stored q95 is
    `p_u`. A level whose tail probability is below `p_u` is `q95 + spread * GPD quantile at
    1 - tail/p_u`; any other level keeps its stored value. The lower tail mirrors this about q05.
    A tail that is already too wide, so that `p_u` is below the level's tail probability, is left as
    stored: the arm can widen a tail and cannot narrow one.
    """
    out = test.quantiles.copy()
    q05_col, q95_col = OUT_LEVELS.index(0.05), OUT_LEVELS.index(0.95)

    def spread(rows: Rows) -> np.ndarray:
        return np.maximum(rows.quantiles[:, q95_col] - rows.quantiles[:, q05_col], MINIMUM_SPREAD)

    cal_spread, test_spread = spread(calibration) / 3.29, spread(test) / 3.29
    for side in ("upper", "lower"):
        if side == "upper":
            excess = (calibration.actual - calibration.quantiles[:, q95_col]) / cal_spread
            anchor = test.quantiles[:, q95_col]
            levels = [(j, 1.0 - p) for j, p in enumerate(OUT_LEVELS) if p >= 0.95]
        else:
            excess = (calibration.quantiles[:, q05_col] - calibration.actual) / cal_spread
            anchor = test.quantiles[:, q05_col]
            levels = [(j, p) for j, p in enumerate(OUT_LEVELS) if p <= 0.05]
        beyond = excess[excess > 0]
        exceed_probability = len(beyond) / len(excess)
        shape, scale = _fit_pareto(excess=beyond)
        for j, tail in levels:
            if tail >= exceed_probability:
                continue
            ratio = tail / exceed_probability
            quantile = (
                scale * (ratio ** (-shape) - 1.0) / shape
                if abs(shape) > 1e-6
                else -scale * math.log(ratio)
            )
            sign = 1.0 if side == "upper" else -1.0
            out[:, j] = anchor + sign * test_spread * quantile
    return np.sort(out, axis=1)


def _parametric(*, calibration: Rows, test: Rows, family: str) -> np.ndarray:
    """Fit a Student-t or skew-normal to the standardised residual and read all levels from it."""
    q25, q75 = OUT_LEVELS.index(0.25), OUT_LEVELS.index(0.75)

    def sigma(rows: Rows) -> np.ndarray:
        return np.maximum(rows.quantiles[:, q75] - rows.quantiles[:, q25], MINIMUM_SPREAD) / 1.349

    z = (calibration.actual - calibration.quantiles[:, MEDIAN_COLUMN]) / sigma(calibration)
    sample = _subsample(values=z)
    distribution = stats.t if family == "t" else stats.skewnorm
    params = distribution.fit(sample)
    standard = distribution.ppf(np.array(OUT_LEVELS), *params)
    median = test.quantiles[:, [MEDIAN_COLUMN]]
    return np.sort(
        median + sigma(test)[:, None] * (standard[None, :] - standard[MEDIAN_COLUMN]), axis=1
    )


def student_t(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arm 7a: Student-t on the standardised residual."""
    return _parametric(calibration=calibration, test=test, family="t")


def skew_normal(*, calibration: Rows, test: Rows) -> np.ndarray:
    """Arm 7b: skew-normal on the standardised residual."""
    return _parametric(calibration=calibration, test=test, family="skewnorm")


ArmFunction = Callable[..., np.ndarray]

ARMS: Final[dict[str, tuple[ArmFunction, str]]] = {
    "a0_as_stored": (as_stored, "none"),
    "a1_global_1.15": (global_spread, "none"),
    "a2_scale_per_level": (scale_per_level, "class"),
    "a3_scale_per_level_band_class": (scale_per_level, "class_band"),
    "a4_isotonic_remap": (remap_levels, "class_band"),
    "a5_conformal_shift": (conformal_shift, "class_band"),
    "a6_pareto_tails": (pareto_tails, "class_band"),
    "a7a_student_t": (student_t, "class_band"),
    "a7b_skew_normal": (skew_normal, "class_band"),
    "a8_post_hoc_scale_recent_90_days": (scale_per_level, "class_band_recent"),
    "a9_post_hoc_conformal_recent_90_days": (conformal_shift, "class_band_recent"),
}
"""Each arm's name, function, and the grouping its calibration is fitted within.

Arms 8 and 9 are post hoc: arms 3 and 5 refitted on the last 90 days only.
"""

POST_HOC_ARMS: Final[tuple[str, ...]] = (
    "a8_post_hoc_scale_recent_90_days",
    "a9_post_hoc_conformal_recent_90_days",
)
"""The arms added after the first results; the decision rule ignores them."""

RECENT_DAYS: Final[int] = 90
"""Length of the calibration window of the post hoc arms."""
