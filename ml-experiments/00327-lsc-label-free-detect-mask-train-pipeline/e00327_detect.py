# ruff: noqa: E501
"""Label-free load-shift detectors on one demand model's out-of-sample residual.

Reads no switching label, logged closure, or labelled shift: every threshold, quantile function and
neighbour comparison uses the residual stream and the electricity network's connectivity only.

Signals, evaluated at each 6-hourly origin from the readings that ended 3.5 hours before it:
`pit12` is the 12 hour mean of the normal score of each half-hour's residual inside the quantile
function of its series and hour band (fitted on all odd-month half-hours); `r24` is the 24 hour mean
residual in points of effective capacity. An alarm opens at the first origin whose absolute signal
exceeds the series' threshold and passes the neighbour filter of experiment 315, closes when the
signed signal has fallen below half the threshold at two consecutive origins (or after 14 days),
and masks readings from 24 hours before the opening origin's data cutoff to the first low origin's
cutoff. The threshold is the smallest value that gives 5 alarms per series-year on odd months.

    PYTHONPATH=ml-experiments/00309-lsc-two-forecast-gap-choosing-without-truth \
        uv run --no-sync python e00327_detect.py <detector-forecast-arm> <signal> <spans-name>
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist
from typing import Final

import e00309_data as dat
import e00309_harness as h
import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
TARGET_RATE: Final[float] = 5.0
"""Alarms per series-year on odd months, before the neighbour filter."""
CLOSE_SHARE: Final[float] = 0.5
CLOSE_RUN: Final[int] = 2
MAX_ORIGINS_OPEN: Final[int] = 56
"""14 days of 6-hourly origins."""
ONSET_BACK: Final[int] = 47
"""Half-hours between the opening origin's data cutoff and the first masked half-hour."""
WINDOW: Final[dict[str, int]] = {"pit12": 24, "r24": 48}
THETA: Final[float] = 0.25
MAX_CONTRADICT: Final[int] = 2
Z_LIMIT: Final[float] = 3.5
N_PROBS: Final[int] = 199
SERIES: Final[tuple[int, ...]] = h.SERIES


def window_means(*, x: np.ndarray, cutoffs: np.ndarray, width: int) -> np.ndarray:
    """Mean over the `width` half-hours ending at each cutoff; NaN under half valid."""
    valid = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(valid)])
    hi = cutoffs + 1
    lo = np.maximum(hi - width, 0)
    n = cnt[hi] - cnt[lo]
    return np.where(n >= width / 2, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)


def normal_scores(*, residual: np.ndarray, band: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Normal score of each residual inside its hour band's quantile function (`fit` rows only)."""
    probs = np.arange(1, N_PROBS + 1) / (N_PROBS + 1)
    inverse = np.array([NormalDist().inv_cdf(p) for p in probs])
    out = np.full(len(residual), np.nan)
    for b in np.unique(band):
        sample = residual[(band == b) & fit & np.isfinite(residual)]
        quantiles = np.quantile(sample, probs)
        idx = (band == b) & np.isfinite(residual)
        out[idx] = np.interp(residual[idx], quantiles, inverse)
    return np.clip(out, -Z_LIMIT, Z_LIMIT)


def load_residual(
    *, forecast_dir: Path, series: int, power: np.ndarray, caps: dict[int, float]
) -> np.ndarray:
    """Reading minus the forecast, in points of effective capacity."""
    expected = np.load(forecast_dir / f"forecast_{series}.npy")
    return (power - expected) * 100.0 / caps[series]


def signals(
    *, residual: np.ndarray, band: np.ndarray, odd_stamp: np.ndarray, kind: str
) -> np.ndarray:
    """The signal at every origin."""
    _, cutoffs, _ = h.origin_grid()
    base = (
        normal_scores(residual=residual, band=band, fit=odd_stamp) if kind == "pit12" else residual
    )
    return window_means(x=base, cutoffs=cutoffs, width=WINDOW[kind])


def run_machine(
    *, score: np.ndarray, threshold: float, keep: np.ndarray | None
) -> list[tuple[int, int, int]]:
    """Alarm episodes as (opening origin index, first low origin index, second low origin index).

    The closing indices are -1 when the alarm stays open to the cap or the end of the record.
    """
    out = []
    i, n = 0, len(score)
    while i < n:
        s = score[i]
        if np.isfinite(s) and abs(s) > threshold and (keep is None or keep[i]):
            sign = np.sign(s)
            low = 0
            first_low = -1
            j = i + 1
            closed = -1
            while j < n and j - i <= MAX_ORIGINS_OPEN:
                v = score[j]
                if np.isfinite(v) and sign * v < CLOSE_SHARE * threshold:
                    if low == 0:
                        first_low = j
                    low += 1
                    if low >= CLOSE_RUN:
                        closed = j
                        break
                else:
                    low = 0
                j += 1
            out.append((i, first_low if closed >= 0 else min(i + MAX_ORIGINS_OPEN, n - 1), closed))
            i = (closed if closed >= 0 else j) + 1
        else:
            i += 1
    return out


def threshold_for(*, score: np.ndarray, odd_origin: np.ndarray) -> float:
    """Smallest threshold with at most `TARGET_RATE` odd-month alarms per series-year (bisection)."""
    years = float((odd_origin & np.isfinite(score)).sum()) * h.ORIGIN_STEP / 48.0 / 365.25
    lo, hi = 0.0, float(np.nanmax(np.abs(score)))
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        eps = run_machine(score=score, threshold=mid, keep=None)
        rate = sum(1 for e in eps if odd_origin[e[0]]) / years
        if rate > TARGET_RATE:
            lo = mid
        else:
            hi = mid
    return hi


def neighbour_keep(
    *, scores: dict[int, np.ndarray], caps: dict[int, float], kind: str, series: int
) -> np.ndarray:
    """Experiment 315's rule: keep a flagged origin unless two or more neighbours show the same sign."""
    me = scores[series]
    scale = (lambda s: caps[s] / 100.0) if kind == "r24" else (lambda s: 1.0)
    a = me * scale(series)
    confirm = np.zeros(len(a), dtype=bool)
    contradict = np.zeros(len(a), dtype=int)
    finite = np.zeros(len(a), dtype=int)
    for n in dat.ADJACENCY[series]:
        if n not in scores:
            continue
        b = scores[n] * scale(n)
        ok = np.isfinite(a) & np.isfinite(b)
        big = ok & (np.abs(b) >= THETA * np.abs(a))
        confirm |= big & (a * b < 0)
        contradict += (big & (a * b > 0)).astype(int)
        finite += ok.astype(int)
    return confirm | (finite == 0) | (contradict < MAX_CONTRADICT)


def detect(*, forecast_arm: str, kind: str, name: str) -> None:
    """Detect on every series' residual and write the spans and the signals."""
    caps = h.capacities()
    band = h.band_grid()
    month = h.utc_month_grid()
    odd_stamp = np.isin(month, list(h.TUNING_MONTHS))
    origins, cutoffs, _ = h.origin_grid()
    odd_origin = odd_stamp[origins]
    forecast_dir = WORK / "forecast" / forecast_arm
    power = {s: np.load(WORK / "power" / f"power_{s}.npy") for s in SERIES}
    score = {}
    for s in SERIES:
        resid = load_residual(forecast_dir=forecast_dir, series=s, power=power[s], caps=caps)
        score[s] = signals(residual=resid, band=band, odd_stamp=odd_stamp, kind=kind)
    rows, plain_rows, summary = [], [], []
    for s in SERIES:
        thr = threshold_for(score=score[s], odd_origin=odd_origin)
        keep = neighbour_keep(scores=score, caps=caps, kind=kind, series=s)
        for filtered, kp in (("plain", None), ("filtered", keep)):
            eps = run_machine(score=score[s], threshold=thr, keep=kp)
            summary.append({"series": s, "threshold": thr, "variant": filtered, "alarms": len(eps)})
            for o, first_low, closed in eps:
                c_open = int(cutoffs[o])
                onset = max(c_open - ONSET_BACK, 0)
                end = int(cutoffs[first_low])
                known = int(cutoffs[closed]) if closed >= 0 else h.N_GRID
                row = {
                    "series": s, "onset": onset, "end": max(end, onset + 1), "start": c_open,
                    "known": max(known, end + 1), "sign": int(np.sign(score[s][o])),
                    "peak": float(np.nanmax(np.abs(score[s][o : max(first_low, o) + 1]))),
                }  # fmt: skip
                (rows if filtered == "filtered" else plain_rows).append(row)
    (WORK / "spans").mkdir(parents=True, exist_ok=True)
    schema = {
        "series": pl.Int64,
        "onset": pl.Int64,
        "end": pl.Int64,
        "start": pl.Int64,
        "known": pl.Int64,
        "sign": pl.Int64,
        "peak": pl.Float64,
    }
    pl.DataFrame(rows, schema=schema).write_parquet(WORK / "spans" / f"{name}.parquet")
    pl.DataFrame(plain_rows, schema=schema).write_parquet(WORK / "spans" / f"{name}_plain.parquet")
    pl.DataFrame(summary).write_parquet(WORK / "spans" / f"{name}_summary.parquet")
    np.savez(WORK / "spans" / f"{name}_scores.npz", **{str(s): v for s, v in score.items()})  # ty: ignore[invalid-argument-type]
    print(pl.DataFrame(summary).pivot(on="variant", index=["series", "threshold"], values="alarms"))


if __name__ == "__main__":
    detect(forecast_arm=sys.argv[1], kind=sys.argv[2], name=sys.argv[3])
