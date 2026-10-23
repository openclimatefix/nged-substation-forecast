# ruff: noqa: E501
"""The gap between the forecast with lags (K0) and the forecast without lags (GAM_ONLY), per origin.

For each series, at each of experiment 258's 6-hourly origins, four gap scores in points of
effective capacity (100 x MW / capacity): the mean of K0 minus GAM_ONLY over leads 0 to 6 hours
(`d`), and the mean residual of the readings that ended at least 3.5 hours before the origin, over
the last 6, 24, and 72 hours (`r6`, `r24`, `r72`). The real record reads experiment 258's stored K0
corrections; an injected world adds experiment 141's moved megawatts to the series and reruns the
Kalman filter.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00308_gap.py
"""

from __future__ import annotations

import time
from typing import Final

import e00308_data as dat
import e00308_harness as h
import numpy as np
from studies.dp_assimilation import CorrectionTrace, correction_at

OUT: Final = dat.WORK_DIR / "gap"
WORLDS: Final = dat.DATA / "00141-switching-detector-fed-into-gam-k" / "worlds"
SERIES_OF_SITE: Final[dict[str, int]] = {
    "D5": 17,
    "D10": 10,
    "D11": 8,
    "D14": 16,
    "D16": 18,
    "D9": 7,
}
SEEDS: Final = range(9200, 9244)
GAP_LEADS: Final[int] = 12
"""Leads 0 to 5.5 hours: the first 12 half-hours."""
WIDE_LEADS: Final[int] = 48
"""Post hoc: leads 0 to 23.5 hours, so that all four hour bands of the filter's level are in the mean."""
WINDOWS: Final[dict[str, int]] = {"r6": 12, "r24": 48, "r72": 144}


def window_means(*, residual: np.ndarray, cutoffs: np.ndarray, width: int) -> np.ndarray:
    """Mean residual over the `width` half-hours ending at each cutoff; NaN under half valid."""
    valid = np.isfinite(residual)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, residual, 0.0))])
    cnt = np.concatenate([[0], np.cumsum(valid)])
    hi = cutoffs + 1
    lo = np.maximum(hi - width, 0)
    n = cnt[hi] - cnt[lo]
    total = csum[hi] - csum[lo]
    return np.where(n >= width / 2, total / np.maximum(n, 1), np.nan)


def scores_of(
    *, residual: np.ndarray, gap: np.ndarray, gap_wide: np.ndarray, cutoffs: np.ndarray
) -> dict[str, np.ndarray]:
    """The gap scores at every origin; the gap is undefined before the series' first reading."""
    finite = np.flatnonzero(np.isfinite(residual))
    has_data = cutoffs >= (finite[0] if len(finite) else h.N_GRID)
    out = {
        "d": np.where(has_data, gap, np.nan).astype(np.float64),
        "d24": np.where(has_data, gap_wide, np.nan).astype(np.float64),
        "has_data": has_data,
    }
    for name, width in WINDOWS.items():
        out[name] = window_means(residual=residual, cutoffs=cutoffs, width=width)
    return out


def k0_gap(*, trace: CorrectionTrace, band: np.ndarray, leads: int = GAP_LEADS) -> np.ndarray:
    """Mean K0 correction over the first `leads` leads at every origin."""
    _, cutoffs, targets = h.origin_grid()
    t = np.minimum(targets[:, :leads], h.N_GRID - 1)
    shape = t.shape
    total = correction_at(
        trace=trace,
        cutoff=np.repeat(cutoffs, shape[1]),
        target=t.ravel(),
        band=band[t.ravel()],
    )
    return total.total.reshape(shape).mean(axis=1)


def real_series(*, series: int, band: np.ndarray) -> dict[str, np.ndarray]:
    """Gap scores of one series on the real record, from experiment 258's stored K0 corrections."""
    data = h.load_series(series=series)
    stored = np.load(h.HARNESS_DIR / f"series_{series}.npz")
    gap = stored["K0"][:, :GAP_LEADS].astype(np.float64).mean(axis=1)
    wide = stored["K0"][:, :WIDE_LEADS].astype(np.float64).mean(axis=1)
    origins, cutoffs, _ = h.origin_grid()
    out = scores_of(residual=data.residual, gap=gap, gap_wide=wide, cutoffs=cutoffs)
    out["origin"] = origins
    out["cutoff"] = cutoffs
    return out


def injected_site(*, site: str, seed: int, band: np.ndarray) -> dict[str, np.ndarray] | None:
    """Gap scores of one site in one injected world, or None when the world moves no megawatts there."""
    series = SERIES_OF_SITE[site]
    with np.load(WORLDS / f"world_{seed}.npz") as stored:
        delta = np.nan_to_num(stored[f"delta_{series}"])
    if not np.any(delta):
        return None
    data = h.load_series(series=series)
    stream = data.residual + 100.0 * delta[: h.N_GRID] / data.capacity_mw
    trace = h.run_kalman(residual=stream, band=band, hyper=data.hyper)
    origins, cutoffs, _ = h.origin_grid()
    out = scores_of(
        residual=stream,
        gap=k0_gap(trace=trace, band=band),
        gap_wide=k0_gap(trace=trace, band=band, leads=WIDE_LEADS),
        cutoffs=cutoffs,
    )
    out["origin"] = origins
    out["cutoff"] = cutoffs
    return out


def main() -> None:
    """Write the real record's scores per series, then every injected world's per site."""
    OUT.mkdir(parents=True, exist_ok=True)
    band = h.band_grid()
    started = time.time()
    for series in h.SERIES:
        path = OUT / f"real_{series}.npz"
        if not path.exists():
            np.savez(path, **real_series(series=series, band=band))  # ty: ignore[invalid-argument-type]
    print(f"real done {time.time() - started:.0f} s", flush=True)
    for seed in SEEDS:
        for site in SERIES_OF_SITE:
            path = OUT / f"world_{seed}_{site}.npz"
            if path.exists():
                continue
            scored = injected_site(site=site, seed=seed, band=band)
            if scored is not None:
                np.savez(path, **scored)  # ty: ignore[invalid-argument-type]
        print(f"world {seed} done {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
