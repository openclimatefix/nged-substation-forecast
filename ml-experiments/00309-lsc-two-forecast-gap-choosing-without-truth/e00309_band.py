# ruff: noqa: E501
"""The band-position signal on the real record, scored against the 24-hour residual.

GAM_ONLY's predictive bands are its expected value plus a residual quantile function per series
and hour band, fitted on odd-month ordinary half-hours (the GAM of experiment 130 is frozen, so
nothing is refitted). `pit24` is the 24-hour mean of each half-hour's normal score and `ext24` the
24-hour mean of the absolute normal scores. Thresholds are set on odd months to the false alarm
rates of LR_b20 and RO-NC, and the matched real shifts are compared with the 24-hour residual's.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00309_band.py
"""

from __future__ import annotations

import json
from statistics import NormalDist
from typing import Final

import e00309_data as dat
import e00309_detect as det
import e00309_harness as h
import e00309_state as st
import numpy as np

N_PROBS: Final[int] = 199
Z_LIMIT: Final[float] = 3.5
WIDTH: Final[int] = 48
SEED: Final[int] = 309
SRC_GAP: Final = st.GAP_DIR


def normal_scores(*, residual: np.ndarray, band: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Normal score of each half-hour's residual inside the quantile function of its hour band."""
    probs = np.arange(1, N_PROBS + 1) / (N_PROBS + 1)
    inverse = np.vectorize(NormalDist().inv_cdf)(probs)
    out = np.full(len(residual), np.nan)
    for b in np.unique(band):
        sample = residual[(band == b) & fit & np.isfinite(residual)]
        quantiles = np.quantile(sample, probs)
        idx = (band == b) & np.isfinite(residual)
        out[idx] = np.interp(residual[idx], quantiles, inverse)
    return np.clip(out, -Z_LIMIT, Z_LIMIT)


def window_means(*, x: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """Mean over the `WIDTH` half-hours ending at each cutoff; NaN under half valid."""
    valid = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(valid)])
    hi = cutoffs + 1
    lo = np.maximum(hi - WIDTH, 0)
    n = cnt[hi] - cnt[lo]
    return np.where(n >= WIDTH / 2, (csum[hi] - csum[lo]) / np.maximum(n, 1), np.nan)


def write_scores() -> None:
    """Copy experiment 308's real-record scores and add `pit24` and `ext24`."""
    out_dir = dat.WORK_DIR / "gap"
    out_dir.mkdir(parents=True, exist_ok=True)
    band = h.band_grid()
    odd = np.isin(h.utc_month_grid(), list(h.TUNING_MONTHS))
    for series in h.SERIES:
        z = dict(np.load(SRC_GAP / f"real_{series}.npz"))
        data = h.load_series(series=series)
        ordinary = ~st.closure_blocks(series=series) & ~st.extra_blocks(series=series)
        score = normal_scores(residual=data.residual, band=band, fit=odd & ordinary)
        z["pit24"] = window_means(x=score, cutoffs=z["cutoff"])
        z["ext24"] = window_means(x=np.abs(score), cutoffs=z["cutoff"])
        for name in ("pit24", "ext24"):
            z[name] = np.where(z["has_data"], z[name], np.nan)
        z["d24"] = z["d"]
        np.savez(out_dir / f"real_{series}.npz", **z)


def calibrate(*, ctx: det.Context, score: str) -> np.ndarray:
    """Odd-month false alarm rate per Primary-year at every threshold in the grid."""
    rates = []
    for k in det.K_GRID:
        n, years = det.fa_counts(
            ctx=ctx, opens=det.opens_all(ctx=ctx, score=score, variant="a", k=float(k)), split="odd"
        )
        rates.append(n / years)
    return np.array(rates)


def pick_k(*, rates: np.ndarray, target: float) -> float:
    """Smallest threshold whose odd-month false alarm rate does not exceed the target."""
    ok = np.flatnonzero(rates <= target)
    return float(det.K_GRID[ok[0]]) if len(ok) else float(det.K_GRID[-1])


def main() -> None:
    """Write the scores, calibrate each signal at both rates, and print the contrasts."""
    write_scores()
    rng = np.random.default_rng(SEED)
    ctx = det.build_context()
    carried = {}
    for src in ("LR_b20", "RO_b80"):
        opens, _ = det.carried_opens(ctx=ctx, source=src)
        carried[src] = det.fa_counts(ctx=ctx, opens=opens, split="odd")
    result: dict = {}
    for src, (n, years) in carried.items():
        target = n / years
        flags, late_flags = {}, {}
        for score in det.SCORES:
            k = pick_k(rates=calibrate(ctx=ctx, score=score), target=target)
            opens = det.opens_all(ctx=ctx, score=score, variant="a", k=k)
            matched, lag = det.matches(ctx=ctx, opens=opens)
            late, _ = det.matches(ctx=ctx, opens=opens, late=det.LATE)
            fa_odd = det.fa_counts(ctx=ctx, opens=opens, split="odd")
            fa_even = det.fa_by_month_interval(ctx=ctx, opens=opens, rng=rng, split="even")
            flags[score], late_flags[score] = matched, late
            result[f"{score}_{src}"] = {
                "k": k,
                "matched": int(matched.sum()),
                "late": int(late.sum()),
                "fa_odd": fa_odd[0] / fa_odd[1],
                "fa_even": fa_even[0],
                "fa_even_lo": fa_even[1],
                "fa_even_hi": fa_even[2],
                "lag_median_hours": float(np.nanmedian(lag)) if np.isfinite(lag).any() else None,
            }
            print(
                f"@{src} (odd rate {target:.2f}) {score:6s} k={k} matched {int(matched.sum())}/35 late {int(late.sum())} FA odd {fa_odd[0] / fa_odd[1]:.2f} even {fa_even[0]:.2f} ({fa_even[1]:.2f} to {fa_even[2]:.2f})",
                flush=True,
            )
        for score in ("pit24", "ext24", "d"):
            for rule, arrays in (("while on", flags), ("within 4 days", late_flags)):
                diff = det.bootstrap_difference(a=arrays[score], b=arrays["r24"], rng=rng)
                result[f"{score}_minus_r24_{rule}_{src}"] = diff
                print(
                    f"  {score} minus r24, {rule}: {diff['diff']:+d} ({diff['lo']:+.0f} to {diff['hi']:+.0f})"
                )
    (dat.WORK_DIR / "results").mkdir(parents=True, exist_ok=True)
    (dat.WORK_DIR / "results" / "band.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
