"""Fit a transfer's effect on a series' residual as a constant, a daily shape, or both.

A window of the residual (observed power minus the convex model's forecast, in MW) is modelled as

    r(t) = base + I(t) * delta(t) + noise,

where `I` marks the transfer's interval. Three models for `delta`, with `D` the convex model's
demand term and `S` its solar output (positive MW):

- `M0` (constant): `delta = c`.
- `M1` (shape): `delta = alpha * D + beta * S`. A donor that gives away a share `f` of its demand
  and `g` of its solar output has `alpha = -f` and `beta = +g`.
- `M2` (constant plus shape): `delta = c + alpha * D + beta * S`.

The held-out reduction in residual sum of squares fits on every other day of the interval (and the
rows outside it) and scores the days left out, so a model with more parameters is not rewarded for
fitting noise. Intervals for `alpha` and `beta` resample whole UTC days of the window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

BLOCK: Final[int] = 48
N_BOOT: Final[int] = 400
MODELS: Final[tuple[str, ...]] = ("M0", "M1", "M2")
MIN_ROWS: Final[int] = 24
"""Fewest valid rows inside the interval for a fit."""


@dataclass(frozen=True)
class Window:
    """One window of a series: the residual, the two terms, and the interval mask."""

    residual: np.ndarray
    demand: np.ndarray
    solar_output: np.ndarray
    inside: np.ndarray
    day: np.ndarray
    """UTC day number of each row."""


def design(*, window: Window, model: str) -> np.ndarray:
    """The design matrix of a model: base, then the interval columns."""
    inside = window.inside.astype(float)
    ones = np.ones(len(inside))
    cols = {
        "M0": [ones, inside],
        "M1": [ones, inside * window.demand, inside * window.solar_output],
        "M2": [ones, inside, inside * window.demand, inside * window.solar_output],
    }[model]
    return np.stack(cols, axis=1)


def _lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(x, y, rcond=None)[0]


def valid_rows(*, window: Window) -> np.ndarray:
    """Rows with finite residual and terms."""
    return (
        np.isfinite(window.residual) & np.isfinite(window.demand) & np.isfinite(window.solar_output)
    )


def subset(*, window: Window, rows: np.ndarray) -> Window:
    """The window restricted to `rows`."""
    return Window(
        residual=window.residual[rows],
        demand=window.demand[rows],
        solar_output=window.solar_output[rows],
        inside=window.inside[rows],
        day=window.day[rows],
    )


def held_out_rss(*, window: Window) -> dict[str, float]:
    """Residual sum of squares on interval days left out, by two-fold alternate-day splitting."""
    days = np.unique(window.day[window.inside])
    out = dict.fromkeys(MODELS, 0.0)
    for fold in (0, 1):
        test_days = days[fold::2]
        test = window.inside & np.isin(window.day, test_days)
        train = ~test
        for model in MODELS:
            x = design(window=window, model=model)
            beta = _lstsq(x[train], window.residual[train])
            out[model] += float(np.square(window.residual[test] - x[test] @ beta).sum())
    return out


@dataclass(frozen=True)
class Fit:
    """The fit of one window."""

    n_inside: int
    n_days: int
    in_sample_rss: dict[str, float]
    held_out_rss: dict[str, float]
    constant: float
    """`c` of M2."""
    alpha: float
    """Demand coefficient of M1."""
    beta: float
    """Solar coefficient of M1."""
    alpha2: float
    beta2: float
    """The coefficients of M2."""
    alpha_ci: tuple[float, float]
    beta_ci: tuple[float, float]
    alpha2_ci: tuple[float, float]
    beta2_ci: tuple[float, float]
    demand_energy: float
    """Sum of the demand term over the interval, MW x half-hours."""
    solar_energy: float
    """Sum of the solar output over the interval."""
    corr_terms: float
    """Correlation of the demand term and the solar output inside the interval."""
    solar_day_cv: float
    """Coefficient of variation of the daily sum of solar output across the interval's days."""


def fit_window(*, window: Window, rng: np.random.Generator, n_boot: int = N_BOOT) -> Fit | None:
    """Fit the three models, with held-out scores and day-block bootstrap intervals."""
    keep = valid_rows(window=window)
    w = subset(window=window, rows=keep)
    n_inside = int(w.inside.sum())
    if n_inside < MIN_ROWS or int((~w.inside).sum()) < MIN_ROWS:
        return None
    in_days = np.unique(w.day[w.inside])
    if len(in_days) < 2:
        return None
    in_sample = {}
    coefs = {}
    for model in MODELS:
        x = design(window=w, model=model)
        beta = _lstsq(x, w.residual)
        coefs[model] = beta
        in_sample[model] = float(np.square(w.residual[w.inside] - x[w.inside] @ beta).sum())
    held = held_out_rss(window=w)
    boot = _bootstrap(window=w, rng=rng, n_boot=n_boot)
    daily = np.array([w.solar_output[w.inside & (w.day == d)].sum() for d in in_days])
    cv = float(daily.std() / daily.mean()) if daily.mean() > 0 else float("nan")
    d_in, s_in = w.demand[w.inside], w.solar_output[w.inside]
    corr = (
        float(np.corrcoef(d_in, s_in)[0, 1]) if s_in.std() > 0 and d_in.std() > 0 else float("nan")
    )
    return Fit(
        n_inside=n_inside,
        n_days=len(in_days),
        in_sample_rss=in_sample,
        held_out_rss=held,
        constant=float(coefs["M2"][1]),
        alpha=float(coefs["M1"][1]),
        beta=float(coefs["M1"][2]),
        alpha2=float(coefs["M2"][2]),
        beta2=float(coefs["M2"][3]),
        alpha_ci=_ci(boot["M1"][:, 1]),
        beta_ci=_ci(boot["M1"][:, 2]),
        alpha2_ci=_ci(boot["M2"][:, 2]),
        beta2_ci=_ci(boot["M2"][:, 3]),
        demand_energy=float(d_in.sum()),
        solar_energy=float(s_in.sum()),
        corr_terms=corr,
        solar_day_cv=cv,
    )


def _ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap(*, window: Window, rng: np.random.Generator, n_boot: int) -> dict[str, np.ndarray]:
    """Refit M1 and M2 on resampled days of the window."""
    days, inverse = np.unique(window.day, return_inverse=True)
    rows_of_day = [np.flatnonzero(inverse == k) for k in range(len(days))]
    x_all = {m: design(window=window, model=m) for m in ("M1", "M2")}
    out = {m: np.empty((n_boot, x_all[m].shape[1])) for m in x_all}
    for b in range(n_boot):
        pick = rng.integers(0, len(days), len(days))
        rows = np.concatenate([rows_of_day[k] for k in pick])
        for m, x in x_all.items():
            out[m][b] = _lstsq(x[rows], window.residual[rows])
    return out


def held_out_reduction(*, fit: Fit, model: str) -> float:
    """`1 - RSS(model) / RSS(M0)` on interval days left out; positive when the model is better."""
    return 1.0 - fit.held_out_rss[model] / fit.held_out_rss["M0"]
