"""The abnormal-arrangement state that RULE reads, from experiment 308's stored scores.

A score is either `r24` (the mean of measured power minus the GAM_ONLY forecast over the last 24
hours before the origin) or `d` (the mean of K0 minus GAM_ONLY over leads 0 to 6 hours), in units
of the series' robust scale on odd-month ordinary rows. RULE enters the abnormal state when the
absolute score exceeds `k` at two consecutive origins, and leaves it when the absolute score stays
under `k / 2` at two consecutive origins or after 14 days.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Final

import e00309_data as dat
import e00309_harness as h
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

GAP_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00308-lsc-two-forecast-gap-as-a-detector" / "gap"
THRESHOLDS: Final[dict[str, tuple[float, ...]]] = {"r24": (2.6, 3.5, 4.5), "d": (2.3, 3.2, 4.5)}
"""Per score: the thresholds that match RO-NC's and the learned score's odd-month false alarm rates in
experiment 308, and a stricter one."""
ENTER_RUN: Final[int] = 2
EXIT_RUN: Final[int] = 2
MAX_ORIGINS: Final[int] = 56
"""14 days of 6-hourly origins."""
D8: Final[int] = 26
SITES: Final[dict[str, int]] = {"D5": 17, "D10": 10, "D11": 8, "D14": 16, "D16": 18, "D9": 7}


def closure_blocks(*, series: int) -> np.ndarray:
    """True inside a logged closure that touches the series or a neighbour, and 72 hours after it."""
    wanted = {str(series)} | {str(s) for s in dat.ADJACENCY[series]}
    out = np.zeros(h.N_GRID + 144, dtype=bool)
    for r in dat.logged_closures().iter_rows(named=True):
        if r["side_a"] in wanted or r["side_b"] in wanted:
            lo, hi = int(r["first_stamp"]), int(r["last_stamp_exclusive"]) + 144
            out[max(lo, 0) : min(hi, h.N_GRID)] = True
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


@lru_cache(maxsize=None)
def sigma_of(*, series: int, score: str) -> float:
    """1.4826 times the median absolute score over odd-month ordinary rows with a reading."""
    z = np.load(GAP_DIR / f"real_{series}.npz")
    origin = z["origin"]
    ordinary = ~closure_blocks(series=series)[origin] & ~extra_blocks(series=series)[origin]
    odd = np.isin(h.utc_month_grid()[origin], list(h.TUNING_MONTHS))
    d = z[score]
    keep = ordinary & odd & np.isfinite(d) & z["has_data"]
    return float(1.4826 * np.median(np.abs(d[keep])))


def abnormal_state(*, gap: np.ndarray, sigma: float, k: float) -> np.ndarray:
    """True at origins where RULE is in the abnormal state."""
    z = np.abs(gap) / sigma
    above = np.isfinite(z) & (z > k)
    below = ~np.isfinite(z) | (z < k / 2.0)
    state = np.zeros(len(z), dtype=bool)
    on, age = False, 0
    for i in range(len(z)):
        if not on:
            if i >= ENTER_RUN - 1 and above[i - ENTER_RUN + 1 : i + 1].all():
                on, age = True, 0
        else:
            age += 1
            if age >= MAX_ORIGINS or (i >= EXIT_RUN - 1 and below[i - EXIT_RUN + 1 : i + 1].all()):
                on = False
        state[i] = on
    return state


def state_for_origin(*, state: np.ndarray, stamp: np.ndarray) -> np.ndarray:
    """The state at the latest 6-hourly origin at or before each stamp."""
    index = np.clip((stamp - h.FIRST_ORIGIN) // h.ORIGIN_STEP, 0, len(state) - 1)
    return state[index.astype(np.int64)]
