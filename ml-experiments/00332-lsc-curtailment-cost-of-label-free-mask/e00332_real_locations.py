# ruff: noqa: E501
"""Re-run experiment 327's per-arm filters and keep each child Primary's 12 to 36 hour locations.

Experiment 327 stores a demand forecast per Primary, not quantiles. This script calls its own
functions (spans, filter hyperparameters, filter run) unchanged and writes, per child Primary of the
bulk supply point D13, the arm's forecast location in megawatts at leads 12 to 36 hours, with the
row-group masks.

    PYTHONPATH=../00327-lsc-label-free-detect-mask-train-pipeline:../00309-lsc-two-forecast-gap-choosing-without-truth:../00130-ro-nc-detector-on-the-gam-k-residual \
        systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00332_real_locations.py
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Final

import e00130_gam_panel as g  # ty: ignore[unresolved-import]
import e00309_harness as h
import e00309_real as real
import e00327_real as r327  # ty: ignore[unresolved-import]
import numpy as np
from studies.sources import ML_EXPERIMENTS_DATA_DIR

OUT: Final = ML_EXPERIMENTS_DATA_DIR / "00332-lsc-curtailment-cost-of-label-free-mask" / "locations"
D13_CHILDREN: Final[tuple[int, ...]] = (7, 8, 9, 10, 11, 24, 25)
"""Child Primary series of the bulk supply point D13 (all 7 have stored forecasts)."""
LEAD_LO: Final = 24
LEAD_HI: Final = 72
"""Lead indices (half-hours) 12 to 36 hours, upper end exclusive."""


def one(series: int) -> int:
    """Write the locations and masks of one child Primary."""
    caps = h.capacities()
    site = g.site_of_series()[series][0]
    band = h.band_grid()
    origins, _, targets = h.origin_grid()
    clipped = np.minimum(targets, h.N_GRID - 1)
    power = np.load(r327.WORK / "power" / f"power_{series}.npy") * 100.0 / caps[series]
    spans = {name: r327.spans_of(source=src, series=series) for name, (_, src) in r327.ARMS.items()}
    masked = np.zeros(h.N_GRID, dtype=bool)
    for name in ("A", "B", "E", "D"):
        for s in spans[name]:
            masked[s.onset : s.end] = True
    masks, _, _ = real.group_masks(series=series, origins=origins, targets=targets, state={})
    short, long, _ = r327.after_masks(series=series, origins=origins, targets=targets)
    arrays: dict[str, np.ndarray] = {
        "ordinary": masks["ordinary"][:, LEAD_LO:LEAD_HI],
        "after48": short[:, LEAD_LO:LEAD_HI],
        "after7d": long[:, LEAD_LO:LEAD_HI],
        "masked_target": masked[clipped][:, LEAD_LO:LEAD_HI],
    }
    for name, (fdir, _) in r327.ARMS.items():
        expected = (
            np.load(r327.WORK / "forecast" / fdir / f"forecast_{series}.npy") * 100.0 / caps[series]
        )
        hyper = r327.fitted_hyper(
            residual_points=power - expected, spans=spans[name], band=band, site=site
        )
        data = h.SeriesData(series, caps[series], power, expected, hyper)
        k0 = h.k0_trace(data=data, band=band)
        corr = h.mask_correction(data=data, spans=spans[name], band=band, k0=k0).astype(np.float64)
        location = data.expected[clipped] + corr
        arrays[f"loc_{name}"] = (location[:, LEAD_LO:LEAD_HI] * caps[series] / 100.0).astype(
            np.float32
        )
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / f"series_{series}.tmp.npz"
    np.savez_compressed(tmp, **arrays)  # ty: ignore[invalid-argument-type]
    tmp.rename(OUT / f"series_{series}.npz")
    print("series", series, "done", flush=True)
    return series


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ctx.Pool(7) as pool:
        pool.map(one, D13_CHILDREN)
