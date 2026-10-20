"""Estimator D of experiment 123: the per-night regression of the night difference on cold.

For one series, `d = a_w + b*c + day-type terms + noise` on GMT nights, with a separate intercept
`a_w` per winter and a common slope `b`. The step at the reference cold night is
`a_hat + b*c_ref - d_summer`, where `a_hat` is the mean of the winter intercepts and `d_summer` the
mean of `d` over June to August nights. Intervals come from a bootstrap that resamples whole
14-day blocks of nights within each winter (and within each summer), the same count of draws in
the real fits and in the known-answer gate.
"""

from dataclasses import dataclass
from datetime import date
from typing import Final

import e00123_common as common
import numpy as np
import polars as pl

EXTREME_QUANTILES: Final[tuple[float, float]] = (
    (1.0 - common.INTERVAL_LEVEL) / 2.0,
    1.0 - (1.0 - common.INTERVAL_LEVEL) / 2.0,
)
"""The 5% and 95% percentiles of a 90% interval."""


@dataclass(frozen=True)
class Nights:
    """One series' winter nights and summer nights, ready to fit.

    Attributes:
        site: The label.
        winter: Winter year of each winter night.
        cold: Degrees below 15.5 C of the daily mean, per winter night.
        cold_night: The same from the 18:00 to 07:00 mean temperature.
        saturday: 1.0 on a Saturday evening, else 0.0.
        sunday: 1.0 on a Sunday evening, else 0.0.
        d: The night difference in the series' units.
        evening: The evening's date, per winter night.
        block: The block index within the winter (14-day blocks), per winter night.
        summer_d: The night difference of each summer night.
        summer_block: The block index within the summer year, per summer night.
        summer_year: The calendar year of each summer night.
    """

    site: str
    winter: np.ndarray
    cold: np.ndarray
    cold_night: np.ndarray
    saturday: np.ndarray
    sunday: np.ndarray
    d: np.ndarray
    evening: np.ndarray
    block: np.ndarray
    summer_d: np.ndarray
    summer_block: np.ndarray
    summer_year: np.ndarray

    @property
    def winters(self) -> list[int]:
        """The winters present, ascending."""
        return sorted({int(w) for w in self.winter})

    def with_arrays(self, **changes: np.ndarray) -> Nights:
        """Return a copy with some arrays replaced."""
        values = {field: getattr(self, field) for field in self.__dataclass_fields__}
        values.update(changes)
        return Nights(**values)


def _blocks(*, evenings: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Return the 14-day block index of each night within its stratum."""
    out = np.zeros(len(evenings), dtype=np.int64)
    for stratum in np.unique(strata):
        mask = strata == stratum
        start = evenings[mask].min()
        out[mask] = ((evenings[mask] - start) // np.timedelta64(common.BLOCK_DAYS, "D")).astype(
            np.int64
        )
    return out


def load_nights(
    *, table: pl.DataFrame, site: str, winters: tuple[int, ...], summer_years: tuple[int, ...]
) -> Nights:
    """Build one series' `Nights` from the night table.

    Args:
        table: `night_tables.parquet`.
        site: The label.
        winters: The winters to include (the caller decides which are opened).
        summer_years: The calendar years whose June to August nights make `d_summer`.

    Returns:
        The usable GMT winter nights (not excluded, `d` and temperature present) and summer nights.
    """
    own = table.filter(pl.col("site") == site)
    winter = own.filter(
        (pl.col("season") == "gmt")
        & ~pl.col("excluded")
        & pl.col("winter").is_in(list(winters))
        & pl.col("d").is_not_null()
        & pl.col("cold_c").is_not_null()
        & pl.col("cold_night_c").is_not_null()
    ).sort("evening")
    summer = own.filter(
        (pl.col("season") == "summer")
        & ~pl.col("excluded")
        & pl.col("evening").dt.year().is_in(list(summer_years))
        & pl.col("d").is_not_null()
    ).sort("evening")
    evening = winter["evening"].to_numpy().astype("datetime64[D]")
    summer_evening = summer["evening"].to_numpy().astype("datetime64[D]")
    summer_year = summer["evening"].dt.year().to_numpy().astype(np.int64)
    winter_year = winter["winter"].to_numpy().astype(np.int64)
    return Nights(
        site=site,
        winter=winter_year,
        cold=winter["cold_c"].to_numpy().astype(np.float64),
        cold_night=winter["cold_night_c"].to_numpy().astype(np.float64),
        saturday=(winter["day_type"] == "saturday").to_numpy().astype(np.float64),
        sunday=(winter["day_type"] == "sunday").to_numpy().astype(np.float64),
        d=winter["d"].to_numpy().astype(np.float64),
        evening=evening,
        block=_blocks(evenings=evening, strata=winter_year),
        summer_d=summer["d"].to_numpy().astype(np.float64),
        summer_block=_blocks(evenings=summer_evening, strata=summer_year),
        summer_year=summer_year,
    )


def design_matrix(*, nights: Nights, cold: np.ndarray | None = None) -> np.ndarray:
    """Return `[winter dummies, c, saturday - mean, sunday - mean]`.

    The day-type columns are centred on their means over `nights`, so that each winter intercept
    is the average night rather than a weekday night.

    Args:
        nights: The nights.
        cold: The cold column to use (default: the daily-mean cold).

    Returns:
        An array of shape (n, number of winters + 3).
    """
    cold_column = nights.cold if cold is None else cold
    dummies = np.stack([(nights.winter == w).astype(np.float64) for w in nights.winters], axis=1)
    return np.column_stack(
        [
            dummies,
            cold_column,
            nights.saturday - nights.saturday.mean(),
            nights.sunday - nights.sunday.mean(),
        ]
    )


def rank_check(*, nights: Nights) -> dict[str, float]:
    """Return the full-rank flag, smallest singular value, and variance inflation of `c`.

    Args:
        nights: The nights.

    Returns:
        `rank`, `columns`, `smallest_singular_value`, and `vif_cold`.
    """
    x = design_matrix(nights=nights)
    singular = np.linalg.svd(x - 0.0, compute_uv=False)
    others = np.delete(x, len(nights.winters), axis=1)
    cold = x[:, len(nights.winters)]
    coefficients, *_ = np.linalg.lstsq(others, cold, rcond=None)
    residual = cold - others @ coefficients
    explained = 1.0 - float(residual @ residual) / float(((cold - cold.mean()) ** 2).sum())
    return {
        "rank": float(np.linalg.matrix_rank(x)),
        "columns": float(x.shape[1]),
        "smallest_singular_value": float(singular.min()),
        "vif_cold": float(1.0 / max(1.0 - explained, 1e-12)),
    }


def block_counts(
    *, strata: np.ndarray, block: np.ndarray, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """Return per-night resampling weights for `draws` block bootstraps stratified by `strata`.

    Args:
        strata: The stratum (winter or summer year) of each night.
        block: The block index of each night within its stratum.
        draws: The number of bootstrap draws.
        rng: The random generator.

    Returns:
        An array of shape (draws, n): how many times each night's block was drawn.
    """
    weights = np.zeros((draws, len(strata)))
    for stratum in np.unique(strata):
        mask = strata == stratum
        n_blocks = int(block[mask].max()) + 1
        picks = rng.integers(0, n_blocks, size=(draws, n_blocks))
        counts = np.zeros((draws, n_blocks))
        rows = np.repeat(np.arange(draws), n_blocks)
        np.add.at(counts, (rows, picks.ravel()), 1.0)
        weights[:, mask] = counts[:, block[mask]]
    return weights


@dataclass(frozen=True)
class Fit:
    """Estimates of one series from one set of weights.

    Attributes:
        slope: `b`, in the series' units per degree.
        flat: `a_hat - d_summer`.
        step: `flat + b*c_ref`.
        intercepts: The winter intercepts, by winter.
    """

    slope: np.ndarray
    flat: np.ndarray
    step: np.ndarray
    intercepts: np.ndarray


def weighted_fit(
    *,
    nights: Nights,
    winter_weights: np.ndarray,
    summer_weights: np.ndarray,
    cold: np.ndarray | None = None,
) -> Fit:
    """Fit D for every row of weights at once.

    Args:
        nights: The nights.
        winter_weights: Shape (draws, n winter nights).
        summer_weights: Shape (draws, n summer nights).
        cold: The cold column to regress on (default: the daily-mean cold).

    Returns:
        One estimate per row of weights.
    """
    x = design_matrix(nights=nights, cold=cold)
    gram = np.einsum("bn,ni,nj->bij", winter_weights, x, x)
    moment = np.einsum("bn,ni,n->bi", winter_weights, x, nights.d)
    coefficients = np.linalg.solve(gram, moment[..., None])[..., 0]
    k = len(nights.winters)
    intercepts = coefficients[:, :k]
    summer_mean = (summer_weights @ nights.summer_d) / summer_weights.sum(axis=1)
    flat = intercepts.mean(axis=1) - summer_mean
    slope = coefficients[:, k]
    return Fit(
        slope=slope, flat=flat, step=flat + slope * common.REFERENCE_COLD_C, intercepts=intercepts
    )


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a percentile interval.

    Attributes:
        value: The point estimate.
        low: The 5th percentile of the bootstrap draws.
        high: The 95th percentile of the bootstrap draws.
        sd: The standard deviation of the bootstrap draws.
    """

    value: float
    low: float
    high: float
    sd: float


@dataclass(frozen=True)
class SeriesResult:
    """The three estimates of one series.

    Attributes:
        site: The label.
        winters: The winters used.
        n_nights: The number of winter nights.
        slope: The slope.
        flat: The flat part.
        step: The step at the reference cold night.
        step_draws: The bootstrap draws of the step.
    """

    site: str
    winters: list[int]
    n_nights: int
    slope: Estimate
    flat: Estimate
    step: Estimate
    step_draws: np.ndarray


def _estimate(*, value: float, draws: np.ndarray) -> Estimate:
    low, high = np.quantile(draws, EXTREME_QUANTILES)
    return Estimate(value=float(value), low=float(low), high=float(high), sd=float(draws.std()))


def fit_series(
    *,
    nights: Nights,
    rng: np.random.Generator,
    draws: int = common.N_BOOTSTRAP,
    cold: np.ndarray | None = None,
) -> SeriesResult:
    """Fit D with its block bootstrap.

    Args:
        nights: The nights.
        rng: The random generator for the bootstrap.
        draws: The number of bootstrap draws.
        cold: The cold column to regress on (default: the daily-mean cold).

    Returns:
        The estimates with 90% percentile intervals.
    """
    point = weighted_fit(
        nights=nights,
        winter_weights=np.ones((1, len(nights.d))),
        summer_weights=np.ones((1, len(nights.summer_d))),
        cold=cold,
    )
    winter_weights = block_counts(strata=nights.winter, block=nights.block, draws=draws, rng=rng)
    summer_weights = block_counts(
        strata=nights.summer_year, block=nights.summer_block, draws=draws, rng=rng
    )
    boot = weighted_fit(
        nights=nights, winter_weights=winter_weights, summer_weights=summer_weights, cold=cold
    )
    return SeriesResult(
        site=nights.site,
        winters=nights.winters,
        n_nights=len(nights.d),
        slope=_estimate(value=point.slope[0], draws=boot.slope),
        flat=_estimate(value=point.flat[0], draws=boot.flat),
        step=_estimate(value=point.step[0], draws=boot.step),
        step_draws=boot.step,
    )


def raw_cold_contrast(*, nights: Nights, cold_snap: np.ndarray | None = None) -> dict[str, float]:
    """Return the raw cold-night mean of `d` minus `d_summer`, and the mean `c` of those nights.

    Cold nights are those with a daily mean below 8 C (`c` above 7.5), or `cold_snap` when given.

    Args:
        nights: The nights.
        cold_snap: An optional boolean mask of winter nights replacing the 8 C rule.

    Returns:
        `nights`, `mean_cold`, `contrast`.
    """
    mask = (
        nights.cold > common.HEATING_THRESHOLD_C - common.COLD_BST_NIGHT_C
        if cold_snap is None
        else cold_snap
    )
    if not mask.any():
        return {"nights": 0.0, "mean_cold": float("nan"), "contrast": float("nan")}
    return {
        "nights": float(mask.sum()),
        "mean_cold": float(nights.cold[mask].mean()),
        "contrast": float(nights.d[mask].mean() - nights.summer_d.mean()),
    }


def winter_start(*, winter: int) -> date:
    """Return the first day of a winter's November (a stable stratum anchor)."""
    return date(winter, 11, 1)
