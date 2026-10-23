"""Series preparation, the noise fit, and the detect and re-centre loop used by every script."""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl
from e00291_common import CLEAN_END, Q_FLOOR, key_of, load_panel, stamp, times
from e00291_hmm import (
    Noise,
    Params,
    Posterior,
    context_ids,
    detect,
    fit_em,
    forward_backward,
    initial_params,
    log_emissions,
    recentre,
    rolling_offset,
)


@dataclass
class Series:
    """One site on the half-hour grid: reading, expected real power, and calendar arrays."""

    label: str
    y: np.ndarray
    load: np.ndarray
    resid: np.ndarray
    ctx: np.ndarray
    month: np.ndarray
    hour: np.ndarray
    weekend: np.ndarray

    def with_reading(self, y: np.ndarray) -> Series:
        """Same site with another reading (an injected copy)."""
        return Series(
            self.label, y, self.load, y - self.load, self.ctx, self.month, self.hour, self.weekend
        )

    def window(self, lo: int, hi: int) -> Series:
        """A slice of the grid."""
        s = slice(lo, hi)
        return Series(
            self.label,
            self.y[s],
            self.load[s],
            self.resid[s],
            self.ctx[s],
            self.month[s],
            self.hour[s],
            self.weekend[s],
        )


def load_series(*, label: str, q: float = Q_FLOOR) -> Series:
    """Load a site from the stored panel; L recovers real power from the expected reading."""
    panel = load_panel()
    arrays = panel[key_of(panel=panel, label=label)]
    y = np.asarray(arrays["power"], dtype=float)
    expected = np.asarray(arrays["expected"], dtype=float)
    load = np.sqrt(np.maximum(expected**2 - q**2, 0.0))
    t = times(n=len(y))
    hour = t.dt.hour().to_numpy()
    weekday = t.dt.weekday().to_numpy() - 1
    month = (t.dt.year().to_numpy() * 12 + t.dt.month().to_numpy()).astype(int)
    return Series(
        label,
        y,
        load,
        y - expected,
        context_ids(hours=hour, weekdays=weekday),
        month,
        hour,
        (weekday >= 5).astype(int),
    )


def clean_mask(*, n: int) -> np.ndarray:
    """True on D8's engine-free history: before 2023-06-13 and July 2023."""
    m = np.zeros(n, dtype=bool)
    m[: stamp(CLEAN_END)] = True
    m[stamp(datetime(2023, 7, 1, tzinfo=UTC)) : stamp(datetime(2023, 8, 1, tzinfo=UTC))] = True
    return m


def fit_noise(*, s: Series, rows: np.ndarray, odd_months_only: bool = True) -> Noise:
    """Fit the autoregressive error on engine-free rows (odd calendar months by default)."""
    month_of_year = ((s.month - 1) % 12) + 1
    use = rows & np.isfinite(s.resid)
    if odd_months_only:
        use &= (month_of_year % 2) == 1
    # remove each month's mean (the offset) from the engine-free residual
    r = np.full(len(s.y), np.nan)
    for m in np.unique(s.month[rows]):
        sel = rows & (s.month == m) & np.isfinite(s.resid)
        if sel.sum() > 100:
            r[sel] = s.resid[sel] - s.resid[sel].mean()
    pair = use & np.isfinite(r) & np.concatenate([[False], np.isfinite(r[:-1])])
    pair[0] = False
    x, z = r[:-1][pair[1:]], r[1:][pair[1:]]
    rho = float(np.clip((x * z).sum() / (x * x).sum(), 0.0, 0.995))
    u = z - rho * x
    lo = s.load[1:][pair[1:]]
    edges = np.quantile(lo, np.linspace(0, 1, 6))
    centres, sds = [], []
    for a, b in itertools.pairwise(edges):
        sel = (lo >= a) & (lo <= b)
        if sel.sum() > 200:
            centres.append(lo[sel].mean())
            sds.append(1.4826 * np.median(np.abs(u[sel] - np.median(u[sel]))))
    slope, intercept = np.polyfit(np.array(centres) - 7.0, np.array(sds), 1)
    return Noise(rho=rho, s0=float(intercept), s1=float(slope))


LOCAL_WINDOW: int = 672
"""A 14-day window, in half-hours, for the local noise multiplier."""


def with_local_noise(*, s: Series, noise: Noise, reference_rows: np.ndarray) -> Noise:
    """Scale the innovation by its 14-day rolling median |u| over its median on `reference_rows`.

    The median of |u| ignores the few big steps of an engine (a spell has two edges), so it follows
    the slow growth of the residual noise rather than the engine.
    """
    r = s.resid
    u = np.abs(r[1:] - noise.rho * r[:-1])
    frame = pl.DataFrame({"u": np.concatenate([[np.nan], u])}).with_columns(
        pl.col("u")
        .rolling_median(window_size=LOCAL_WINDOW, min_samples=200, center=True)
        .alias("m")
    )
    local = frame["m"].to_numpy()
    ok = np.isfinite(local)
    idx = np.arange(len(local))
    local = np.interp(idx, idx[ok], local[ok])
    ref = np.nanmedian(np.concatenate([[np.nan], u])[reference_rows])
    return Noise(noise.rho, noise.s0, noise.s1, np.clip(local / ref, 0.5, 4.0))


def run_loop(
    *,
    s: Series,
    noise: Noise,
    params: Params,
    q: float = Q_FLOOR,
    rounds: int = 2,
    local_reference: np.ndarray | None = None,
    nan_to_null: bool = False,
) -> tuple[np.ndarray, Posterior]:
    """Frozen detector: detect, re-centre from the zero-unit half-hours, detect again.

    Returns the final offset and the final posterior.
    """
    if local_reference is not None:
        noise = with_local_noise(s=s, noise=noise, reference_rows=local_reference)
    offset = rolling_offset(y=s.y, load=s.load, g_unused=params.g, q=q, nan_to_null=nan_to_null)
    for _ in range(rounds):
        post = detect(y=s.y, load=s.load, offset=offset, ctx=s.ctx, noise=noise, q=q, params=params)
        offset = recentre(
            y=s.y,
            load=s.load,
            p_zero=post.smoothed[:, 0],
            month=s.month,
            hour=s.hour,
            weekend=s.weekend,
            q=q,
        )
    post = detect(y=s.y, load=s.load, offset=offset, ctx=s.ctx, noise=noise, q=q, params=params)
    return offset, post


STARTS: tuple[float, ...] = tuple(float(x) for x in np.arange(1.0, 3.51, 0.25))
"""Unit sizes of the profile likelihood, in MW; the transitions are refitted at each."""


def objective(
    *,
    sub: Series,
    offset: np.ndarray,
    noise: Noise,
    params: Params,
    q: float,
    prior_mean: float,
    prior_sd: float,
) -> float:
    """Log-likelihood of the stretch plus the Gaussian log prior on the unit size."""
    log_e = log_emissions(y=sub.y, load=sub.load, offset=offset, g=params.g, q=q, noise=noise)
    post = forward_backward(log_e=log_e, ctx=sub.ctx, params=params, weight=np.ones(len(sub.y)))
    return post.loglik - 0.5 * ((params.g - prior_mean) / prior_sd) ** 2


def best_start(
    *,
    sub: Series,
    offset: np.ndarray,
    noise: Noise,
    q: float,
    prior_mean: float,
    prior_sd: float,
    init: Params | None,
    log: Callable[[str], None],
) -> Params:
    """Profile the unit size: refit the transitions at each size, keep the best objective."""
    best, best_value = None, -np.inf
    starts = STARTS if init is None else (init.g,)
    for g0 in starts:
        start = initial_params(g=g0) if init is None else init
        params = fit_em(
            y=sub.y,
            load=sub.load,
            offset=offset,
            ctx=sub.ctx,
            noise=noise,
            q=q,
            params=start,
            prior_mean=g0,
            prior_sd=1e-3,
            n_iter=8,
        )
        value = objective(
            sub=sub,
            offset=offset,
            noise=noise,
            params=params,
            q=q,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
        )
        log(f"    start g0={g0:.1f} -> g={params.g:.2f} MW, objective {value:.0f}")
        if value > best_value:
            best, best_value = params, value
    if best is None:
        raise ValueError("no start produced a fit")
    return best


def train_rounds(
    *,
    s: Series,
    lo: int,
    hi: int,
    noise: Noise,
    prior_mean: float,
    prior_sd: float,
    q: float = Q_FLOOR,
    retrains: int = 2,
    log: Callable[[str], None] = print,
) -> tuple[Params, list[float]]:
    """Multi-start EM on rows [lo, hi), then `retrains` times: detect, re-centre, refit."""
    sub = s.window(lo, hi)
    noise = noise.window(lo, hi)
    offset = rolling_offset(y=s.y, load=s.load, g_unused=0.0, q=q)[lo:hi]
    params = best_start(
        sub=sub,
        offset=offset,
        noise=noise,
        q=q,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        init=None,
        log=log,
    )
    path = [params.g]
    log(f"  EM round 0: g = {params.g:.3f} MW")
    for r in range(retrains):
        post = detect(
            y=sub.y, load=sub.load, offset=offset, ctx=sub.ctx, noise=noise, q=q, params=params
        )
        offset = recentre(
            y=sub.y,
            load=sub.load,
            p_zero=post.smoothed[:, 0],
            month=sub.month,
            hour=sub.hour,
            weekend=sub.weekend,
            q=q,
        )
        params = best_start(
            sub=sub,
            offset=offset,
            noise=noise,
            q=q,
            prior_mean=prior_mean,
            prior_sd=prior_sd,
            init=None,
            log=log,
        )
        path.append(params.g)
        log(f"  retrain {r + 1}: g = {params.g:.3f} MW")
    return params, path
