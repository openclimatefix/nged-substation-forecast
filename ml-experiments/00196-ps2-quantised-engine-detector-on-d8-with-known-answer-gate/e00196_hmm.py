"""Hidden Markov model of 0 to 3 engine units read through a magnitude meter.

State: the active-period flag a (0 or 1) and the number of units running k (0 to 3), eight states in
all. Reading: y_t = sqrt((L_t - k g)^2 + q^2) + d_t + e_t, where L_t is the expected real power, g is
the unit size, q is a reactive-power floor, d_t is a slowly varying offset, and e_t is a
first-order autoregressive error with Student-t innovations. The innovation is computed from the
previous error under the previous state, so the emission depends on the pair (previous k, new k)
and the forward-backward pass stays exact. A whole-unit step is therefore outside the
autoregression.
"""

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy.special import gammaln

N_K: Final[int] = 4
N_STATES: Final[int] = 8
N_CONTEXTS: Final[int] = 12
"""Six 4-hour blocks of the UTC day times weekday or weekend."""
K_OF_STATE: Final[np.ndarray] = np.tile(np.arange(N_K), 2)
A_OF_STATE: Final[np.ndarray] = np.repeat(np.arange(2), N_K)
NU: Final[float] = 5.0
PRIOR_STRENGTH: Final[float] = 20.0
"""Pseudo-counts that hold a transition row to its starting value where the data are thin."""
G_GRID: Final[np.ndarray] = np.linspace(0.8, 3.6, 57)


@dataclass(frozen=True, eq=False)
class Noise:
    """Autoregressive error: coefficient and innovation scale `s0 + s1 * (L - 7)`, in MW."""

    rho: float
    s0: float
    s1: float
    mult: np.ndarray | None = None
    """Optional local multiplier of the scale at each half-hour (the local-noise variant)."""

    def scale(self, *, load: np.ndarray) -> np.ndarray:
        """Innovation scale at each half-hour, floored at a quarter of `s0`."""
        base = np.maximum(self.s0 + self.s1 * (load - 7.0), 0.25 * self.s0)
        return base if self.mult is None else base * self.mult

    def window(self, lo: int, hi: int) -> "Noise":
        """The noise for rows [lo, hi)."""
        return Noise(self.rho, self.s0, self.s1, None if self.mult is None else self.mult[lo:hi])


@dataclass
class Params:
    """Fitted quantities: unit size, active-flag transitions, and unit transitions per context."""

    g: float
    a_matrix: np.ndarray
    k_matrix: np.ndarray
    """Shape (2, N_CONTEXTS, 4, 4): by the new active flag, context, previous k, new k."""

    def copy(self) -> "Params":
        """Return an independent copy."""
        return Params(g=self.g, a_matrix=self.a_matrix.copy(), k_matrix=self.k_matrix.copy())


def initial_params(*, g: float) -> Params:
    """Return the starting transitions: rare moves to any other count (several units can start together), rarer outside active periods."""
    a_matrix = np.array([[0.998, 0.002], [0.01, 0.99]])
    k_matrix = np.zeros((2, N_CONTEXTS, N_K, N_K))
    for a, move in ((0, 0.01), (1, 0.12)):
        for k in range(N_K):
            row = np.zeros(N_K)
            nbrs = [j for j in range(N_K) if j != k]
            for j in nbrs:
                row[j] = move / len(nbrs)
            row[k] = 1.0 - row.sum()
            k_matrix[a, :, k, :] = row
    return Params(g=g, a_matrix=a_matrix, k_matrix=k_matrix)


def context_ids(*, hours: np.ndarray, weekdays: np.ndarray) -> np.ndarray:
    """Context of each half-hour from its UTC hour (0 to 23) and weekday (0 = Monday)."""
    return (hours // 4) * 2 + (weekdays >= 5).astype(int)


def _t_logpdf(*, z: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Log density of a Student-t with `NU` degrees of freedom and the given scale."""
    const = gammaln((NU + 1) / 2) - gammaln(NU / 2) - 0.5 * np.log(NU * np.pi)
    return const - np.log(scale) - (NU + 1) / 2 * np.log1p((z / scale) ** 2 / NU)


def unit_means(*, load: np.ndarray, g: float, q: float) -> np.ndarray:
    """Expected reading for 0 to 3 units, shape (T, 4), through the magnitude meter."""
    k = np.arange(N_K)[None, :]
    return np.sqrt((load[:, None] - k * g) ** 2 + q**2)


def log_emissions(
    *, y: np.ndarray, load: np.ndarray, offset: np.ndarray, g: float, q: float, noise: Noise
) -> np.ndarray:
    """Return log p(y_t | previous k, new k), shape (T, 4, 4); zero where y_t is missing."""
    mu = unit_means(load=load, g=g, q=q)
    err = y[:, None] - mu - offset[:, None]  # (T, 4) error under each state
    scale = noise.scale(load=load)
    prev_err = np.concatenate([np.full((1, N_K), np.nan), err[:-1]])
    first = ~np.isfinite(prev_err[:, 0])
    innov = err[:, None, :] - noise.rho * np.nan_to_num(prev_err)[:, :, None]  # (T, prev, new)
    stationary = scale / np.sqrt(1 - noise.rho**2)
    out = np.where(
        first[:, None, None],
        _t_logpdf(z=err[:, None, :] + np.zeros_like(innov), scale=stationary[:, None, None]),
        _t_logpdf(z=innov, scale=scale[:, None, None]),
    )
    out[~np.isfinite(y)] = 0.0
    return out


def transition_tensor(*, params: Params) -> np.ndarray:
    """Full 8x8 transition matrix per context, shape (N_CONTEXTS, 8, 8)."""
    full = np.zeros((N_CONTEXTS, N_STATES, N_STATES))
    for c in range(N_CONTEXTS):
        for a in range(2):
            for a2 in range(2):
                full[c, a * N_K : (a + 1) * N_K, a2 * N_K : (a2 + 1) * N_K] = (
                    params.a_matrix[a, a2] * params.k_matrix[a2, c]
                )
    return full


@dataclass
class Posterior:
    """Output of one forward-backward pass."""

    smoothed: np.ndarray  # (T, 4) P(k)
    filtered: np.ndarray  # (T, 4)
    pair: np.ndarray  # (T, 4, 4) P(previous k, new k)
    k_counts: np.ndarray  # (2, N_CONTEXTS, 4, 4)
    a_counts: np.ndarray  # (2, 2)
    loglik: float


def forward_backward(
    *, log_e: np.ndarray, ctx: np.ndarray, params: Params, weight: np.ndarray
) -> Posterior:
    """Exact smoothing over the eight states; `weight` (0 or 1) selects rows that add counts."""
    n = len(ctx)
    tmat = transition_tensor(params=params)[ctx]  # (T, 8, 8)
    shift = log_e.reshape(n, -1).max(axis=1)
    e4 = np.exp(log_e - shift[:, None, None])
    emis = e4[:, K_OF_STATE[:, None], K_OF_STATE[None, :]]  # (T, 8, 8)
    m = tmat * emis
    alpha = np.zeros((n, N_STATES))
    norm = np.zeros(n)
    prev = np.zeros(N_STATES)
    prev[0], prev[1], prev[2], prev[3] = 0.9, 0.05, 0.03, 0.02
    for t in range(n):
        nxt = prev @ m[t]
        norm[t] = nxt.sum()
        prev = nxt / norm[t]
        alpha[t] = prev
    beta = np.ones((n, N_STATES))
    for t in range(n - 1, 0, -1):
        beta[t - 1] = (m[t] @ beta[t]) / norm[t]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    alpha_prev = np.concatenate([np.array([[0.9, 0.05, 0.03, 0.02, 0, 0, 0, 0]]), alpha[:-1]])
    xi = alpha_prev[:, :, None] * m * beta[:, None, :] / norm[:, None, None]
    xi /= xi.sum(axis=(1, 2), keepdims=True)
    x4 = xi.reshape(n, 2, N_K, 2, N_K)
    pair = x4.sum(axis=(1, 3))
    k_counts = np.zeros((2, N_CONTEXTS, N_K, N_K))
    a_counts = np.zeros((2, 2))
    w = weight[:, None, None, None, None] * x4
    for c in range(N_CONTEXTS):
        sel = w[ctx == c]
        k_counts[:, c] = _fold(sel)
    a_counts = w.sum(axis=(0, 2, 4))
    k_marg = gamma.reshape(n, 2, N_K).sum(axis=1)
    f_marg = alpha.reshape(n, 2, N_K).sum(axis=1)
    return Posterior(
        smoothed=k_marg,
        filtered=f_marg,
        pair=pair,
        k_counts=k_counts,
        a_counts=a_counts,
        loglik=float((np.log(norm) + shift)[weight > 0].sum()),
    )


def _fold(sel: np.ndarray) -> np.ndarray:
    """Sum over time and the old active flag: (rows, a, k, a', k') -> (a', k, k')."""
    return sel.sum(axis=(0, 1)).transpose(1, 0, 2)


def m_step_transitions(*, post: Posterior, params: Params, start: Params) -> None:
    """Update the transition tables in place from expected counts, smoothed towards `start`."""
    for a2 in range(2):
        for c in range(N_CONTEXTS):
            for k in range(N_K):
                row = post.k_counts[a2, c, k] + PRIOR_STRENGTH * start.k_matrix[a2, c, k]
                params.k_matrix[a2, c, k] = row / row.sum()
    for a in range(2):
        row = post.a_counts[a] + 200.0 * start.a_matrix[a]
        params.a_matrix[a] = row / row.sum()


def m_step_unit_size(
    *,
    post: Posterior,
    y: np.ndarray,
    load: np.ndarray,
    offset: np.ndarray,
    noise: Noise,
    q: float,
    weight: np.ndarray,
    prior_mean: float,
    prior_sd: float,
) -> float:
    """Grid search for the unit size that maximises the expected log-likelihood plus the prior."""
    best, best_value = prior_mean, -np.inf
    for g in G_GRID:
        log_e = log_emissions(y=y, load=load, offset=offset, g=float(g), q=q, noise=noise)
        value = float((weight[:, None, None] * post.pair * log_e).sum())
        value += -0.5 * ((g - prior_mean) / prior_sd) ** 2
        if value > best_value:
            best, best_value = float(g), value
    return best


def fit_em(
    *,
    y: np.ndarray,
    load: np.ndarray,
    offset: np.ndarray,
    ctx: np.ndarray,
    noise: Noise,
    q: float,
    params: Params,
    prior_mean: float,
    prior_sd: float,
    n_iter: int = 12,
) -> Params:
    """Expectation-maximisation of the unit size and transition tables on one stretch of rows."""
    params = params.copy()
    start = initial_params(g=params.g)
    weight = np.ones(len(y))
    for _ in range(n_iter):
        log_e = log_emissions(y=y, load=load, offset=offset, g=params.g, q=q, noise=noise)
        post = forward_backward(log_e=log_e, ctx=ctx, params=params, weight=weight)
        m_step_transitions(post=post, params=params, start=start)
        params.g = m_step_unit_size(
            post=post, y=y, load=load, offset=offset, noise=noise, q=q, weight=weight,
            prior_mean=prior_mean, prior_sd=prior_sd,
        )
    return params


def detect(
    *, y: np.ndarray, load: np.ndarray, offset: np.ndarray, ctx: np.ndarray, noise: Noise,
    q: float, params: Params,
) -> Posterior:
    """Run the frozen model over all rows."""
    log_e = log_emissions(y=y, load=load, offset=offset, g=params.g, q=q, noise=noise)
    return forward_backward(log_e=log_e, ctx=ctx, params=params, weight=np.zeros(len(y)))


def rolling_offset(*, y: np.ndarray, load: np.ndarray, g_unused: float, q: float) -> np.ndarray:
    """Round-zero offset: the centred 30-day rolling median of y minus the zero-unit mean."""
    res = y - unit_means(load=load, g=0.0, q=q)[:, 0]
    frame = pl.DataFrame({"r": res}).with_columns(
        pl.col("r").rolling_median(window_size=1441, min_samples=300, center=True).alias("m")
    )
    out = frame["m"].to_numpy().copy()
    ok = np.isfinite(out)
    if not ok.any():
        return np.zeros(len(y))
    idx = np.arange(len(out))
    return np.interp(idx, idx[ok], out[ok])


def recentre(
    *, y: np.ndarray, load: np.ndarray, p_zero: np.ndarray, month: np.ndarray, hour: np.ndarray,
    weekend: np.ndarray, q: float, threshold: float = 0.9, min_rows: int = 100,
) -> np.ndarray:
    """Offset from the half-hours labelled zero units: a monthly level plus an hour-by-day-type profile."""
    res = y - unit_means(load=load, g=0.0, q=q)[:, 0]
    use = np.isfinite(res) & (p_zero > threshold)
    levels = {}
    for m in np.unique(month):
        sel = use & (month == m)
        if sel.sum() >= min_rows:
            levels[m] = float(res[sel].mean())
    keys = sorted(levels)
    level = np.zeros(len(y))
    for m in np.unique(month):
        if m in levels:
            level[month == m] = levels[m]
        else:
            nearest = min(keys, key=lambda k: abs(k - m))
            level[month == m] = levels[nearest]
    profile = np.zeros((24, 2))
    dev = res - level
    for h in range(24):
        for w in range(2):
            sel = use & (hour == h) & (weekend == w)
            if sel.sum() >= min_rows:
                profile[h, w] = float(dev[sel].mean())
    profile -= profile.mean()
    return level + profile[hour, weekend]
