"""Statistics, injections, change-point search, CUSUM, and state-space fits for experiment 00341.

A *block* is a run of `B` consecutive UTC days, counted from 2019-09-01. A block statistic is
missing (the block is dropped) when the block holds fewer than the minimum number of half-hours in
the set the statistic uses. Every function takes arrays and returns arrays, so the pool workers
pickle only small objects.
"""

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_name, "1")

from dataclasses import dataclass  # noqa: E402
from typing import Final  # noqa: E402

import e00341_common as common  # noqa: E402
import numpy as np  # noqa: E402

QUANTILES: Final[tuple[float, ...]] = (0.90, 0.99, 0.999, 1.0)
QUANTILE_NAMES: Final[tuple[str, ...]] = ("p90", "p99", "p999", "max")
HORIZON_DAYS: Final[int] = 190
CUSUM_K: Final[float] = 0.5
SWITCH_PROBABILITY: Final[float] = 0.02
MAX_TURBINES_OUT: Final[int] = 4
CLIP_LEVELS: Final[tuple[float, ...]] = (0.5, 0.75, 0.9)


@dataclass(frozen=True)
class Prep:
    """A farm prepared for one strong-set margin and one evidence model."""

    farm: common.Farm
    margin: float
    strong: np.ndarray
    mid: np.ndarray
    weighted: np.ndarray
    busy: np.ndarray
    slope: np.ndarray
    weight: np.ndarray
    curve_speed: np.ndarray
    curve_exp: np.ndarray
    sigma_v: float
    sigma_p: float
    ref_p999: float
    ref_rmid: float
    ref_iqr: float


@dataclass(frozen=True)
class Series:
    """The observed blocks of one statistic: block positions, values, and inverse variances."""

    pos: np.ndarray
    y: np.ndarray
    omega: np.ndarray


def log_slope(*, speed: np.ndarray, curve_speed: np.ndarray, curve_exp: np.ndarray) -> np.ndarray:
    """Return d log(expected power) / d log(speed) at each speed, from the fitted curve."""
    up = np.interp(speed * 1.02, curve_speed, curve_exp)
    down = np.interp(speed * 0.98, curve_speed, curve_exp)
    return (np.log(np.maximum(up, 1e-4)) - np.log(np.maximum(down, 1e-4))) / np.log(1.02 / 0.98)


def prepare(*, farm: common.Farm, margin: float, sigma_v: float, sigma_p: float) -> Prep:
    """Build the half-hour index sets and the evidence weights for one farm."""
    grid, curve = common.power_curve_lookup(speed=farm.speed, exp=farm.exp, cut_out=farm.cut_out)
    strong = np.flatnonzero(
        (farm.speed >= farm.v_sat + margin) & (farm.speed <= farm.cut_out - common.STRONG_MARGIN_M_S)
    )
    mid = np.flatnonzero((farm.exp >= common.MID_LOW) & (farm.exp <= common.MID_HIGH))
    weighted = np.flatnonzero((farm.exp >= common.WEIGHTED_LOW) & (farm.exp <= 1.05))
    busy = np.flatnonzero(farm.exp >= common.BUSY_LOW)
    slope = log_slope(speed=farm.speed[weighted], curve_speed=grid, curve_exp=curve)
    weight = 1.0 / (slope**2 * sigma_v**2 + sigma_p**2)
    cal = (farm.day >= common.CAL_START_DAY) & (farm.day <= common.CAL_END_DAY)
    cal_strong = strong[cal[strong]]
    cal_mid = mid[cal[mid]]
    ref_p999 = float(np.quantile(farm.obs[cal_strong], 0.999)) if len(cal_strong) else np.nan
    ref_iqr = (
        float(np.subtract(*np.quantile(farm.obs[cal_strong], [0.75, 0.25])))
        if len(cal_strong)
        else np.nan
    )
    ref_rmid = float(farm.obs[cal_mid].sum() / farm.exp[cal_mid].sum())
    return Prep(farm, margin, strong, mid, weighted, busy, slope, weight, grid, curve,
                sigma_v, sigma_p, ref_p999, ref_rmid, ref_iqr)  # fmt: skip


def modified_obs(*, prep: Prep, effects: tuple[tuple, ...]) -> np.ndarray:
    """Return the observed share of scale after the injected effects.

    Effects are tuples: ("loss", fraction, first_day, last_day), ("clip", level, first_day,
    last_day) with the level a share of the calibration 99.9th percentile of the strong set,
    ("swing", range), ("drift", per_year), ("bias", relative_speed_error, first_day, last_day).
    Multiplicative effects apply first, then clips.
    """
    farm = prep.farm
    obs = farm.obs.copy()
    day = farm.day
    clips: list[tuple[float, int, int]] = []
    for effect in effects:
        kind = effect[0]
        if kind == "loss":
            _, fraction, first, last = effect
            obs[(day >= first) & (day <= last)] *= 1.0 - fraction
        elif kind == "swing":
            after = day >= common.START_FIRST_DAY
            phase = 2 * np.pi * (day[after] - common.START_FIRST_DAY) / 365.25
            obs[after] *= 1.0 - 0.5 * effect[1] * (1.0 - np.cos(phase))
        elif kind == "drift":
            after = day >= common.START_FIRST_DAY
            obs[after] *= 1.0 - effect[1] * (day[after] - common.START_FIRST_DAY) / 365.25
        elif kind == "bias":
            _, delta, first, last = effect
            sel = (day >= first) & (day <= last)
            base = np.interp(farm.speed[sel], prep.curve_speed, prep.curve_exp)
            biased = np.interp(farm.speed[sel] * (1.0 + delta), prep.curve_speed, prep.curve_exp)
            obs[sel] *= biased / np.maximum(base, 1e-3)
        elif kind == "clip":
            clips.append((effect[1], effect[2], effect[3]))
    for level, first, last in clips:
        sel = (day >= first) & (day <= last)
        obs[sel] = np.minimum(obs[sel], level * prep.ref_p999)
    return obs


def _bounds(*, days: np.ndarray, block: int, n_blocks: int) -> np.ndarray:
    return np.searchsorted(days // block, np.arange(n_blocks + 1))


def block_arms(*, prep: Prep, obs: np.ndarray, block: int) -> tuple[dict[str, Series], dict]:
    """Return the arm series (A quantiles, B, C, D) and the state-space features for one block size.

    Returns:
        A dict of `Series` keyed `A_p90`, `A_p99`, `A_p999`, `A_max`, `B`, `C`, `D`, and a dict of
        full-length feature arrays (NaN where missing) keyed `p999`, `rmid`, `iqr`, `n_strong`.
    """
    farm = prep.farm
    n_blocks = common.N_DAYS // block + 1
    min_n = common.MIN_HALF_HOURS_DAILY if block == 1 else common.MIN_HALF_HOURS
    arms: dict[str, Series] = {}
    # Arm A: quantiles of the strong set
    sdays = farm.day[prep.strong]
    sval = obs[prep.strong]
    bounds = _bounds(days=sdays, block=block, n_blocks=n_blocks)
    counts = np.diff(bounds)
    quant = np.full((n_blocks, len(QUANTILES)), np.nan)
    iqr = np.full(n_blocks, np.nan)
    for b in np.flatnonzero(counts >= min_n):
        v = sval[bounds[b] : bounds[b + 1]]
        quant[b] = np.quantile(v, QUANTILES)
        q75, q25 = np.quantile(v, [0.75, 0.25])
        iqr[b] = q75 - q25
    ok_a = counts >= min_n
    for j, name in enumerate(QUANTILE_NAMES):
        pos = np.flatnonzero(ok_a)
        arms[f"A_{name}"] = Series(pos, quant[pos, j], np.ones(len(pos)))
    # Arms B and C: weighted log ratio over the 25 % to 100 % set
    wb = farm.day[prep.weighted] // block
    y = np.clip(
        np.log(
            (np.maximum(obs[prep.weighted], 0.0) + common.EPS_RATIO)
            / (farm.exp[prep.weighted] + common.EPS_RATIO)
        ),
        *common.LOG_RATIO_CLIP,
    )
    w = prep.weight
    s = prep.slope
    cnt = np.bincount(wb, minlength=n_blocks)
    sw = np.bincount(wb, weights=w, minlength=n_blocks)
    sws = np.bincount(wb, weights=w * s, minlength=n_blocks)
    swss = np.bincount(wb, weights=w * s * s, minlength=n_blocks)
    swy = np.bincount(wb, weights=w * y, minlength=n_blocks)
    swsy = np.bincount(wb, weights=w * s * y, minlength=n_blocks)
    ok_w = cnt >= common.MIN_HALF_HOURS
    pos = np.flatnonzero(ok_w)
    arms["B"] = Series(pos, swy[pos] / sw[pos], sw[pos])
    ridge = 1.0 / common.SPEED_BIAS_PRIOR_SD**2
    a11 = sw[pos]
    a12 = sws[pos]
    a22 = swss[pos] + ridge
    det = a11 * a22 - a12 * a12
    a_hat = (a22 * swy[pos] - a12 * swsy[pos]) / det
    arms["C"] = Series(pos, a_hat, det / a22)
    # Arm D: unweighted ratio on the busy set (experiment 233's rows)
    bb = farm.day[prep.busy] // block
    so = np.bincount(bb, weights=obs[prep.busy], minlength=n_blocks)
    se = np.bincount(bb, weights=farm.exp[prep.busy], minlength=n_blocks)
    cb = np.bincount(bb, minlength=n_blocks)
    pos = np.flatnonzero(cb >= common.MIN_HALF_HOURS)
    arms["D"] = Series(pos, so[pos] / se[pos], se[pos])
    # State-space features
    mb = farm.day[prep.mid] // block
    mo = np.bincount(mb, weights=obs[prep.mid], minlength=n_blocks)
    me = np.bincount(mb, weights=farm.exp[prep.mid], minlength=n_blocks)
    mc = np.bincount(mb, minlength=n_blocks)
    rmid = np.where(mc >= common.MIN_HALF_HOURS, mo / np.maximum(me, 1e-9), np.nan)
    feats = {"p999": quant[:, 2], "rmid": rmid, "iqr": iqr, "n_strong": counts.astype(float)}
    return arms, feats


def kappa(series: Series) -> float:
    """Return the robust noise scale of a series: squared MAD of standardised successive differences."""
    if len(series.y) < 8:
        return np.nan
    d = np.diff(series.y)
    u = d / np.sqrt(1.0 / series.omega[1:] + 1.0 / series.omega[:-1])
    mad = np.median(np.abs(u - np.median(u)))
    return max((1.4826 * mad) ** 2, 1e-12)


def partition(*, series: Series, penalty_c: float) -> tuple[np.ndarray, float]:
    """Exact penalised least-squares partition of the series (optimal partitioning).

    The cost of a segment is its weighted sum of squared deviations from its weighted mean divided
    by the robust noise scale, and each segment adds `penalty_c * ln(n)`.

    Returns:
        `last` (length n+1, the start index of the final segment of the best partition of each
        prefix) and the noise scale.
    """
    n = len(series.y)
    k = kappa(series)
    last = np.zeros(n + 1, dtype=np.int64)
    if n < 8 or not np.isfinite(k):
        return last, k
    om = series.omega / k
    cw = np.concatenate([[0.0], np.cumsum(om)])
    cy = np.concatenate([[0.0], np.cumsum(om * series.y)])
    cyy = np.concatenate([[0.0], np.cumsum(om * series.y**2)])
    beta = penalty_c * np.log(n)
    best = np.zeros(n + 1)
    best[0] = -beta
    for t in range(1, n + 1):
        s = np.arange(t)
        sw = cw[t] - cw[s]
        cost = (cyy[t] - cyy[s]) - (cy[t] - cy[s]) ** 2 / sw
        cand = best[s] + cost + beta
        j = int(np.argmin(cand))
        best[t] = cand[j]
        last[t] = j
    return last, k


def backtrack(*, last: np.ndarray, t: int) -> list[int]:
    """Return the segment start indices of the best partition of the first `t` observed blocks."""
    starts: list[int] = []
    while t > 0:
        t = int(last[t])
        starts.append(t)
    return starts[::-1]


def segment_means(*, series: Series, starts: list[int], t: int) -> np.ndarray:
    """Return the weighted mean of each segment of a partition of the first `t` blocks."""
    ends = [*starts[1:], t]
    return np.array(
        [
            np.average(series.y[a:b], weights=series.omega[a:b]) if b > a else np.nan
            for a, b in zip(starts, ends, strict=True)
        ]
    )


def accepted_by_prefix(
    *, series: Series, last: np.ndarray, start_block: int, block: int, direction: int = -1
) -> dict[int, bool]:
    """Return, per prefix length in the detection window, whether a consistent boundary exists.

    A boundary before observed block `k` is consistent when the start lies between the previous
    observed block and block `k` (with one block of tolerance after) and the level moves in the
    wanted direction. Only prefixes that end within `HORIZON_DAYS` after the start are evaluated.
    """
    pos = series.pos
    n = len(pos)
    cw = np.concatenate([[0.0], np.cumsum(series.omega)])
    cy = np.concatenate([[0.0], np.cumsum(series.omega * series.y)])
    lo = int(np.searchsorted(pos, start_block)) + 1
    hi = int(np.searchsorted(pos, start_block + HORIZON_DAYS // block + 1)) + 1
    out: dict[int, bool] = {}
    for t in range(max(lo, 2), min(hi, n) + 1):
        ok = False
        end = t
        i = t
        while i > 0:
            k = int(last[i])
            if k == 0:
                break
            if pos[k] < start_block - 1:
                break
            if pos[k - 1] <= start_block + 1 and pos[k] >= start_block:
                prev = int(last[k])
                mean_after = (cy[end] - cy[k]) / (cw[end] - cw[k])
                mean_before = (cy[k] - cy[prev]) / (cw[k] - cw[prev])
                if direction * (mean_after - mean_before) > 0:
                    ok = True
                    break
            end = k
            i = k
        out[t] = ok
    return out


def cusum_alarms(*, series: Series, h: float, block: int) -> list[tuple[int, int]]:
    """Return (observed index, direction) of two-sided CUSUM alarms on the standardised series.

    The reference level is the median of the blocks in the calibration window, and the scale is the
    series' robust noise scale. The statistic restarts at zero after each alarm.
    """
    k = kappa(series)
    if not np.isfinite(k):
        return []
    first = common.CAL_START_DAY // block
    last = common.CAL_END_DAY // block
    cal = (series.pos >= first) & (series.pos <= last)
    mu = float(np.median(series.y[cal] if cal.sum() >= 5 else series.y))
    z = (series.y - mu) * np.sqrt(series.omega / k)
    hi = lo = 0.0
    alarms: list[tuple[int, int]] = []
    for i, zi in enumerate(z):
        hi = max(0.0, hi + zi - CUSUM_K)
        lo = max(0.0, lo - zi - CUSUM_K)
        if lo > h:
            alarms.append((i, -1))
            hi = lo = 0.0
        elif hi > h:
            alarms.append((i, 1))
            hi = lo = 0.0
    return alarms


def block_end_day(*, pos: int, block: int) -> int:
    """Return the last day of block `pos`."""
    return (pos + 1) * block - 1


def detection_delay(
    *, series_inj: Series, series_base: Series, last_inj: np.ndarray, last_base: np.ndarray,
    start_day: int, block: int,
) -> float | None:  # fmt: skip
    """Return the days from the injected start to the first consistent boundary, or None.

    NaN is returned when the unmodified series already shows a consistent boundary no later than
    the injected one (the start is excluded). None means no boundary within the horizon.
    """
    start_block = start_day // block
    acc_inj = accepted_by_prefix(series=series_inj, last=last_inj, start_block=start_block, block=block)
    acc_base = accepted_by_prefix(series=series_base, last=last_base, start_block=start_block, block=block)
    first_inj = next((t for t, ok in acc_inj.items() if ok), None)
    first_base = next((t for t, ok in acc_base.items() if ok), None)
    if first_base is not None and (first_inj is None or first_base <= first_inj):
        return float("nan")
    if first_inj is None:
        return None
    return float(max(0, block_end_day(pos=int(series_inj.pos[first_inj - 1]), block=block) - start_day))


def cusum_delay(
    *, series_inj: Series, series_base: Series, start_day: int, block: int, h: float
) -> float | None:
    """Return the days to the first downward CUSUM alarm at or after the start, as in the search."""
    start_block = start_day // block
    inj = [
        i for i, d in cusum_alarms(series=series_inj, h=h, block=block)
        if d < 0 and series_inj.pos[i] >= start_block
    ]
    base = [
        i for i, d in cusum_alarms(series=series_base, h=h, block=block)
        if d < 0 and series_base.pos[i] >= start_block
    ]
    if base and (not inj or base[0] <= inj[0]):
        return float("nan")
    if not inj:
        return None
    delay = block_end_day(pos=int(series_inj.pos[inj[0]]), block=block) - start_day
    return float(delay) if delay <= HORIZON_DAYS else None


def boundary_positions(*, series: Series, last: np.ndarray) -> np.ndarray:
    """Return the block position just after each change of the full-series partition."""
    starts = backtrack(last=last, t=len(series.y))
    return np.array([series.pos[k] for k in starts[1:]], dtype=int)


def new_boundaries(*, series: Series, last: np.ndarray, base: np.ndarray, tolerance: int) -> int:
    """Count boundaries of the full-series partition with no base boundary within `tolerance` blocks."""
    mine = boundary_positions(series=series, last=last)
    return int(sum(1 for p in mine if not len(base) or np.min(np.abs(base - p)) > tolerance))


# ----- state-space (Viterbi) fit -------------------------------------------------------------


def state_table(*, prep: Prep, block: int, turbines_only: bool) -> tuple[list[str], np.ndarray]:
    """Return state names and their mean feature vectors (p999, mid ratio, IQR), all relative.

    Turbine states scale every feature by (N-k)/N. Clip states take the mid ratio and the IQR from
    clipping the calibration half-hours at the clip level.
    """
    farm = prep.farm
    names = ["nameplate"]
    means = [np.array([1.0, 1.0, 1.0])]
    n_turb = farm.n_turbines
    for k in range(1, min(MAX_TURBINES_OUT, n_turb - 1) + 1):
        f = (n_turb - k) / n_turb
        names.append(f"{k}_out")
        means.append(np.array([f, f, f]))
    if not turbines_only:
        cal = (farm.day >= common.CAL_START_DAY) & (farm.day <= common.CAL_END_DAY)
        cal_strong = prep.strong[cal[prep.strong]]
        cal_mid = prep.mid[cal[prep.mid]]
        for c in CLIP_LEVELS:
            level = c * prep.ref_p999
            rm = np.minimum(farm.obs[cal_mid], level).sum() / farm.obs[cal_mid].sum()
            clipped = np.minimum(farm.obs[cal_strong], level)
            iq = float(np.subtract(*np.quantile(clipped, [0.75, 0.25]))) / prep.ref_iqr
            names.append(f"clip_{int(c * 100)}")
            means.append(np.array([c, rm, iq]))
    return names, np.array(means)


def viterbi(
    *, feats: dict, prep: Prep, block: int, turbines_only: bool, noise: np.ndarray,
) -> tuple[np.ndarray, list[str]]:  # fmt: skip
    """Return the best state index per block and the state names.

    `noise` holds the standard deviation of the three relative features (p999, mid ratio, IQR).
    Missing features are skipped, and a block with none keeps the previous state.
    """
    names, means = state_table(prep=prep, block=block, turbines_only=turbines_only)
    rel = np.column_stack(
        [feats["p999"] / prep.ref_p999, feats["rmid"] / prep.ref_rmid, feats["iqr"] / prep.ref_iqr]
    )
    if turbines_only:
        rel[:, 1:] = np.nan
    n_blocks, n_states = rel.shape[0], len(names)
    loglik = np.zeros((n_blocks, n_states))
    for j in range(3):
        ok = np.isfinite(rel[:, j])
        z = (rel[ok, j][:, None] - means[None, :, j]) / noise[j]
        loglik[ok] += -0.5 * z**2
    stay = np.log(1.0 - SWITCH_PROBABILITY)
    move = np.log(SWITCH_PROBABILITY / (n_states - 1))
    trans = np.full((n_states, n_states), move)
    np.fill_diagonal(trans, stay)
    score = np.zeros((n_blocks, n_states))
    back = np.zeros((n_blocks, n_states), dtype=np.int64)
    score[0] = loglik[0]
    score[0, 1:] += move
    for t in range(1, n_blocks):
        cand = score[t - 1][:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        score[t] = cand[back[t], np.arange(n_states)] + loglik[t]
    path = np.zeros(n_blocks, dtype=np.int64)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n_blocks - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path, names


def feature_noise(*, feats: dict, prep: Prep) -> np.ndarray:
    """Return the robust standard deviation of the relative features in the calibration window."""
    rel = np.column_stack(
        [feats["p999"] / prep.ref_p999, feats["rmid"] / prep.ref_rmid, feats["iqr"] / prep.ref_iqr]
    )
    n_blocks = rel.shape[0]
    block_days = common.N_DAYS / (n_blocks - 1)
    first = int(common.CAL_START_DAY / block_days)
    last = int(common.CAL_END_DAY / block_days)
    out = np.zeros(3)
    for j in range(3):
        v = rel[first:last, j]
        v = v[np.isfinite(v)]
        out[j] = max(1.4826 * np.median(np.abs(v - np.median(v))), 0.03) if len(v) > 5 else 0.2
    return out
