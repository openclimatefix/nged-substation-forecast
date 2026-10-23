# ruff: noqa: E501
"""Alarm episodes from the gap scores, the match rule, and the scores of every detector.

Fixed before any score was read (see the plan in `result.md`): a flagged origin has an absolute
score above k times the series' robust scale in odd months of ordinary rows; consecutive flagged
origins within 48 hours form one episode that opens at the first (sustained) flagged origin; k is
the smallest value whose odd-month false alarm rate does not exceed a target; a real shift is
matched when an episode opens from 2 hours before its onset to its end, on either Primary of its
pair.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00309_detect.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import e00309_data as dat
import e00309_harness as h
import numpy as np
import polars as pl

GAP: Final = dat.WORK_DIR / "gap"
RESULTS: Final = dat.WORK_DIR / "results"
SCORES: Final[tuple[str, ...]] = ("d", "r24", "pit24", "ext24")
"""`d24` is post hoc: the gap over leads 0 to 24 hours."""
VARIANTS: Final[tuple[str, ...]] = ("a", "b", "c6", "c24")
MIN_RUN: Final[dict[str, int]] = {"a": 1, "b": 1, "c6": 2, "c24": 5}
"""Consecutive flagged origins (6 hours apart) a variant needs before it raises an alarm."""
MERGE_ORIGINS: Final[int] = 8
"""Flagged origins within 48 hours belong to one episode."""
MATCH_EARLY: Final[int] = 4
"""A shift is matched by an alarm opening up to 2 hours (4 half-hours) before its onset."""
STEP_TOLERANCE: Final[int] = 12
"""The loose rule for the carried detectors: their step stamp within 6 hours of the true onset."""
K_GRID: Final[np.ndarray] = np.round(np.arange(1.0, 20.01, 0.1), 1)
SIZE_RATIO: Final[float] = 0.3
SUM_RATIO: Final[float] = 0.5
ORIGIN_YEARS: Final[float] = h.ORIGIN_STEP / 48.0 / 365.25
"""Years of record one origin stands for."""
D8, D9 = 26, 7


@dataclass(frozen=True)
class Series:
    """One series' scores and masks on the origin grid."""

    series: int
    cap: float
    origin: np.ndarray
    score: dict[str, np.ndarray]
    sigma: dict[str, float]
    ordinary: np.ndarray
    odd: np.ndarray
    ordinary_stamp: np.ndarray


def closure_blocks(*, series: int) -> np.ndarray:
    """True inside a logged closure that touches the series or a connected neighbour, and 72 hours after it."""
    wanted = {str(series)} | {str(s) for s in dat.ADJACENCY[series]}
    out = np.zeros(h.N_GRID + 144, dtype=bool)
    for r in dat.logged_closures().iter_rows(named=True):
        if r["side_a"] in wanted or r["side_b"] in wanted:
            out[
                max(int(r["first_stamp"]), 0) : min(int(r["last_stamp_exclusive"]) + 144, h.N_GRID)
            ] = True
    return out[: h.N_GRID]


def extra_blocks(*, series: int) -> np.ndarray:
    """True on D8's multi-day drops and the series' unlabelled block days."""
    out = np.zeros(h.N_GRID, dtype=bool)
    if series == D8:
        for r in pl.read_parquet(dat.D8_DROPS).iter_rows(named=True):
            out[int(r["first_stamp"]) : int(r["last_stamp"]) + 1] = True
    for lo, hi in dat.block_day_ranges().get(series, []):
        out[max(lo, 0) : hi + 1] = True
    return out


def robust_sigma(*, x: np.ndarray, mask: np.ndarray) -> float:
    """1.4826 times the median absolute deviation of `x` over the masked finite values, about zero."""
    v = x[mask & np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(v)))


def load_real(*, series: int, month: np.ndarray, caps: dict[int, float]) -> Series:
    """One series' gap scores with its ordinary mask (clean set) and odd-month flag at each origin."""
    z = np.load(GAP / f"real_{series}.npz")
    origin = z["origin"]
    ordinary_stamp = ~closure_blocks(series=series) & ~extra_blocks(series=series)
    odd = np.isin(month[origin], list(h.TUNING_MONTHS))
    score = {k: z[k] for k in SCORES}
    ordinary = ordinary_stamp[origin] & z["has_data"]
    sigma = {k: robust_sigma(x=score[k], mask=ordinary & odd) for k in SCORES}
    return Series(series, caps[series], origin, score, sigma, ordinary, odd, ordinary_stamp)


def sustained(*, flag: np.ndarray, run: int) -> np.ndarray:
    """True where the last `run` origins were all flagged."""
    if run == 1:
        return flag
    return np.convolve(flag.astype(int), np.ones(run, dtype=int), mode="full")[: len(flag)] == run


def episode_starts(*, flag: np.ndarray) -> np.ndarray:
    """Indices of the first flagged origin of each episode (gaps over `MERGE_ORIGINS` split episodes)."""
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return idx
    return np.concatenate([idx[:1], idx[1:][np.diff(idx) > MERGE_ORIGINS]])


def neighbour_confirm(*, all_series: dict[int, Series], series: int, score: str) -> np.ndarray:
    """True where a connected metered neighbour has an opposite gap of at least 30% the size.

    The node's gap plus the opposite neighbours' gaps must also sum to at most half the node's (MW).
    """
    me = all_series[series]
    a = me.score[score] * me.cap / 100.0
    opp_sum = np.zeros(len(a))
    found = np.zeros(len(a), dtype=bool)
    for n in dat.ADJACENCY[series]:
        if n not in all_series:
            continue
        b = all_series[n].score[score] * all_series[n].cap / 100.0
        ok = np.isfinite(a) & np.isfinite(b) & (a * b < 0) & (np.abs(b) >= SIZE_RATIO * np.abs(a))
        opp_sum += np.where(ok, b, 0.0)
        found |= ok
    return found & (np.abs(a + opp_sum) <= SUM_RATIO * np.abs(a))


def opens_of(
    *, s: Series, score: str, variant: str, k: float, confirm: np.ndarray | None
) -> np.ndarray:
    """Stamps at which a detector opens an alarm episode in one series."""
    z = np.abs(s.score[score]) / s.sigma[score]
    flag = np.isfinite(z) & (z > k)
    if variant == "b":
        flag &= confirm
    flag = sustained(flag=flag, run=MIN_RUN[variant])
    return s.origin[episode_starts(flag=flag)]


@dataclass(frozen=True)
class Context:
    """Everything the scorers share."""

    real: dict[int, Series]
    month: np.ndarray
    year_month: np.ndarray
    odd_years: float
    shifts: pl.DataFrame
    confirm: dict[tuple[int, str], np.ndarray]


def build_context() -> Context:
    """Load the real record's scores for the 15 series."""
    caps = h.capacities()
    month = h.utc_month_grid()
    real = {s: load_real(series=s, month=month, caps=caps) for s in h.SERIES}
    odd_years = sum(float((v.ordinary & v.odd).sum()) for v in real.values()) * ORIGIN_YEARS
    confirm = {
        (s, sc): neighbour_confirm(all_series=real, series=s, score=sc)
        for s in real
        for sc in SCORES
    }
    shifts = dat.labelled_loads().filter(
        pl.col("series").list.eval(pl.element().is_in(list(h.SERIES))).list.any()
    )
    return Context(real, month, dat.stamp_year_month(), odd_years, shifts, confirm)


def shift_weight(*, ctx: Context, row: dict) -> float:
    """Absolute step in MW of the larger of the pair.

    The step is the median residual over the first 48 half-hours against the 7 days before.
    """
    best = 0.0
    for s in row["series"]:
        if s not in ctx.real:
            continue
        d = h.load_series(series=s)
        r = d.residual
        after = np.nanmedian(r[row["onset"] : row["onset"] + 48])
        before = np.nanmedian(r[max(row["onset"] - 336, 0) : row["onset"]])
        best = max(best, abs(after - before) * d.capacity_mw / 100.0)
    return best


def opens_all(*, ctx: Context, score: str, variant: str, k: float) -> dict[int, np.ndarray]:
    """Detector's opening stamps per series."""
    return {
        s: opens_of(s=v, score=score, variant=variant, k=k, confirm=ctx.confirm[(s, score)])
        for s, v in ctx.real.items()
    }


def fa_counts(*, ctx: Context, opens: dict[int, np.ndarray], split: str) -> tuple[int, float]:
    """False alarms in ordinary clean rows of odd, even, or all months, and the years those rows cover."""
    count, rows = 0, 0.0
    for s, v in ctx.real.items():
        o = opens.get(s, np.array([], dtype=np.int64))
        o = o[(o >= 0) & (o < h.N_GRID)]
        odd_o = np.isin(ctx.month[o], list(h.TUNING_MONTHS))
        want_o = {"odd": odd_o, "even": ~odd_o, "all": np.ones(len(o), dtype=bool)}[split]
        want_r = {"odd": v.odd, "even": ~v.odd, "all": np.ones_like(v.odd)}[split]
        count += int((v.ordinary_stamp[o] & want_o).sum())
        rows += float((v.ordinary & want_r).sum())
    return count, rows * ORIGIN_YEARS


LATE: Final[int] = 192
"""Post hoc rule M2: an alarm may open up to 4 days (the learned score's delay) after the shift ends."""


def matches(
    *, ctx: Context, opens: dict[int, np.ndarray], late: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Per real shift: matched, and the lag in hours (NaN if unmatched)."""
    matched = np.zeros(ctx.shifts.height, dtype=bool)
    lag = np.full(ctx.shifts.height, np.nan)
    for i, r in enumerate(ctx.shifts.iter_rows(named=True)):
        best = np.inf
        for s in r["series"]:
            o = opens.get(s)
            if o is None:
                continue
            hit = o[(o >= r["onset"] - MATCH_EARLY) & (o <= r["end"] + late)]
            if len(hit):
                best = min(best, hit.min() - r["onset"])
        if np.isfinite(best):
            matched[i] = True
            lag[i] = max(best, 0) / 2.0
    return matched, lag


# --- the carried detectors, scored by the same rules ---


def carried_opens(
    *, ctx: Context, source: str
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Opening stamps (alarm known) and step stamps of one carried detector's alarms, per series."""
    opens: dict[int, list[int]] = {s: [] for s in ctx.real}
    steps: dict[int, list[int]] = {s: [] for s in ctx.real}
    if source == "PG_g70":
        table = dat.alarm_tables(source=source)[0]
        for r in table.iter_rows(named=True):
            s = int(r["alarm_series"])
            if s in opens:
                opens[s].append(int(r["onset_known"]))
                steps[s].append(int(r["onset"]))
    else:
        arm = "RO_b80" if source == "RO_b80" else "LR_b20"
        for s in ctx.real:
            spans = pl.read_parquet(h.HARNESS_DIR / f"spans_{s}.parquet")
            spans = spans.filter(pl.col("arm").str.starts_with(arm)).unique(subset=["onset", "end"])
            opens[s] = [int(x) for x in spans["start"].to_list()]
            steps[s] = [int(x) for x in spans["onset"].to_list()]
    return (
        {s: np.array(sorted(v), dtype=np.int64) for s, v in opens.items()},
        {s: np.array(sorted(v), dtype=np.int64) for s, v in steps.items()},
    )


def step_matches(*, ctx: Context, steps: dict[int, np.ndarray]) -> np.ndarray:
    """The loose rule: a carried detector's step stamp within 6 hours of the true onset."""
    out = np.zeros(ctx.shifts.height, dtype=bool)
    for i, r in enumerate(ctx.shifts.iter_rows(named=True)):
        for s in r["series"]:
            o = steps.get(s)
            if o is not None and np.any(np.abs(o - r["onset"]) <= STEP_TOLERANCE):
                out[i] = True
    return out


def interval_hits(*, opens: dict[int, np.ndarray], series: int, lo: int, hi: int) -> bool:
    """True when an episode opens on the series between `lo` (minus the early margin) and `hi`."""
    o = opens.get(series)
    return o is not None and bool(np.any((o >= lo - MATCH_EARLY) & (o <= hi)))


def bootstrap_matches(
    *, matched: np.ndarray, weights: np.ndarray, rng: np.random.Generator, n_draws: int = 20000
) -> dict:
    """Matched count and MW-weighted share with a 95% interval, resampling whole shifts."""
    n = len(matched)
    idx = rng.integers(0, n, size=(n_draws, n))
    count = matched[idx].sum(axis=1)
    share = (matched * weights)[idx].sum(axis=1) / weights[idx].sum(axis=1)
    return {
        "matched": int(matched.sum()),
        "matched_lo": float(np.quantile(count, 0.025)),
        "matched_hi": float(np.quantile(count, 0.975)),
        "mw_share": float((matched * weights).sum() / weights.sum()),
        "mw_lo": float(np.quantile(share, 0.025)),
        "mw_hi": float(np.quantile(share, 0.975)),
    }


def bootstrap_difference(
    *, a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_draws: int = 20000
) -> dict:
    """Paired difference in matched shifts (a minus b) with a 95% interval over whole shifts."""
    n = len(a)
    idx = rng.integers(0, n, size=(n_draws, n))
    diff = (a.astype(int) - b.astype(int))[idx].sum(axis=1)
    return {
        "diff": int(a.sum() - b.sum()),
        "lo": float(np.quantile(diff, 0.025)),
        "hi": float(np.quantile(diff, 0.975)),
    }


def fa_by_month_interval(
    *,
    ctx: Context,
    opens: dict[int, np.ndarray],
    rng: np.random.Generator,
    split: str,
    n_draws: int = 20000,
) -> tuple[float, float, float]:
    """False alarms per Primary-year with a 95% interval, resampling whole calendar months."""
    months = np.unique(ctx.year_month)
    cnt = {int(m): 0 for m in months}
    rows = {int(m): 0.0 for m in months}
    for s, v in ctx.real.items():
        o = opens.get(s, np.array([], dtype=np.int64))
        o = o[(o >= 0) & (o < h.N_GRID)]
        odd_o = np.isin(ctx.month[o], list(h.TUNING_MONTHS))
        want_o = {"odd": odd_o, "even": ~odd_o, "all": np.ones(len(o), dtype=bool)}[split]
        keep = v.ordinary_stamp[o] & want_o
        for m, c in zip(*np.unique(ctx.year_month[o][keep], return_counts=True), strict=True):
            cnt[int(m)] += int(c)
        want_r = {"odd": v.odd, "even": ~v.odd, "all": np.ones_like(v.odd)}[split]
        ok = v.ordinary & want_r
        for m, c in zip(*np.unique(ctx.year_month[v.origin][ok], return_counts=True), strict=True):
            rows[int(m)] += float(c)
    keys = [m for m in cnt if rows[m] > 0]
    c = np.array([cnt[m] for m in keys], dtype=float)
    r = np.array([rows[m] for m in keys]) * ORIGIN_YEARS
    idx = rng.integers(0, len(keys), size=(n_draws, len(keys)))
    rate = c[idx].sum(axis=1) / r[idx].sum(axis=1)
    return (
        float(c.sum() / r.sum()),
        float(np.quantile(rate, 0.025)),
        float(np.quantile(rate, 0.975)),
    )
