# ruff: noqa: E501
"""Refit experiment 130's convex demand model at every monthly cut under one training mask.

Arms: `D` trains on `default_rows` (the hand-filtered switching rows removed, which is what
experiment 130 stored; this run is its positive control); `C` trains on every row with a reading;
any other arm trains on the rows of `C` minus the half-hours inside the alarm spans of
`spans/<arm>.parquet` that are known at the cut (experiment 212's rule: an alarm counts once its
start is known, its end only once known, otherwise it runs to the cut). Output is one forecast array
per series, on the half-hour grid, in MW, at `forecast/<arm>/forecast_<series>.npy`.

    PYTHONPATH=ml-experiments/00130-ro-nc-detector-on-the-gam-k-residual \
        systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python e00327_fit.py <arm> [--check]
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

import e00130_gam_panel as g  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_gam import TrainingWindow, fit_gam, predict_gam
from studies.sources import ML_EXPERIMENTS_DATA_DIR

WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00327-lsc-label-free-detect-mask-train-pipeline"
N_WORKERS: Final[int] = 4
GRID_START64: Final = np.datetime64("2022-01-01T00:00", "us")


def stamps_of(*, times: np.ndarray) -> np.ndarray:
    """Half-hour grid stamps of datetime64 times."""
    return ((times - GRID_START64) // np.timedelta64(30, "m")).astype(np.int64)


def alarm_mask(*, spans: pl.DataFrame, series: int, row_stamps: np.ndarray, cut: int) -> np.ndarray:
    """True on cache rows inside an alarm span of the series that is known at the cut stamp."""
    out = np.zeros(len(row_stamps), dtype=bool)
    for r in spans.filter(pl.col("series") == series).iter_rows(named=True):
        if r["start"] > cut:
            continue
        stop = r["end"] if r["known"] <= cut else cut + 1
        out |= (row_stamps >= r["onset"]) & (row_stamps < stop)
    return out


def fit_job(job: tuple[str, int]) -> str:
    """Fit every monthly cut of one series for one arm, forecast to the next cut, and store."""
    arm, series = job
    torch.set_num_threads(1)
    site, capacity = g.site_of_series()[series]
    cache = g.load_cache()
    features = cache.sites[site]
    all_rows = features.default_rows | (
        features.switching_mask & features.in_pool & ~features.lockdown
    )
    spans = pl.read_parquet(WORK / "spans" / f"{arm}.parquet") if arm not in ("C", "D") else None
    end = cache.end_times
    row_stamps = stamps_of(times=end)
    cuts = g.cuts_for(offset_day=1)
    naive = np.array([np.datetime64(c.replace(tzinfo=None), "us") for c in cuts])
    spec = g.base_spec()
    forecast = np.full(g.GRID_LENGTH, np.nan)
    fallbacks = 0
    for k, cut in enumerate(cuts):
        rows = features.default_rows if arm == "D" else all_rows
        if spans is not None:
            cut_stamp = int(stamps_of(times=naive[k : k + 1])[0])
            rows = rows & ~alarm_mask(
                spans=spans, series=series, row_stamps=row_stamps, cut=cut_stamp
            )
        usable = rows & np.isfinite(features.y)
        if int((usable & (end <= naive[k])).sum()) < g.MIN_TRAINING_DAYS * 48:
            continue
        fit = None
        for attempt_rows in (rows, all_rows):
            trial = replace(
                cache, sites={**cache.sites, site: replace(features, default_rows=attempt_rows)}
            )
            try:
                fit = fit_gam(
                    cache=trial,
                    site=site,
                    specs=[spec],
                    cut=cut,
                    window=TrainingWindow(months=g.WINDOW_MONTHS),
                    season_years=g.SEASON_YEARS,
                )
            except ValueError:
                fallbacks += 1
                continue
            break
        if fit is None:
            continue
        stop = naive[k + 1] if k + 1 < len(cuts) else np.datetime64("2026-07-01T00:00", "us")
        target = np.flatnonzero((end > naive[k]) & (end <= stop) & np.isfinite(features.y))
        if not len(target):
            continue
        try:
            points = predict_gam(cache=cache, fit=fit, index=target)
        except ValueError:
            continue
        st = row_stamps[target]
        keep = (st >= 0) & (st < g.GRID_LENGTH)
        forecast[st[keep]] = points[keep] * capacity / 100.0
    out = WORK / "forecast" / arm
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"forecast_{series}.npy.tmp.npy"
    np.save(tmp, forecast)
    tmp.rename(out / f"forecast_{series}.npy")
    return f"{arm} series {series}: {int(np.isfinite(forecast).sum())} forecast half-hours, {fallbacks} masked fits refused"


if __name__ == "__main__":
    arm_name = sys.argv[1]
    series_list = [s for s in g.SERIES_IDS if s != 12]
    ctx = mp.get_context("spawn")
    with ctx.Pool(N_WORKERS) as pool:
        for line in pool.imap_unordered(fit_job, [(arm_name, s) for s in series_list]):
            print(line, flush=True)
    if "--check" in sys.argv:
        worst = 0.0
        for s in series_list:
            new = np.load(WORK / "forecast" / arm_name / f"forecast_{s}.npy")
            old = np.load(g.GAM_DIR / f"forecast_{s}_1.npy")
            both = np.isfinite(new) & np.isfinite(old)
            worst = max(worst, float(np.max(np.abs(new[both] - old[both]))) if both.any() else 0.0)
        print(
            f"positive control: max abs difference from experiment 130 stored forecasts {worst:.3e} MW"
        )
