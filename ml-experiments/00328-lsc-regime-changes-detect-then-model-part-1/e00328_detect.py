# ruff: noqa: E501, ANN001, ANN201
r"""Stage 1: run the rolling-GAM CUSUM detector and the weekly-histogram detector.

Stage `base` writes each Primary's out-of-sample residual (a monthly refit of the lagless GAM on
the previous 24 months, scored on the next month) and the real-record alarms. Stage `inject` adds
a permanent step or a 30-day ramp at each onset of experiment 295 (odd months tune, even months
report) and records whether each detector alarms within 60 days.

    PYTHONPATH=ml-experiments/00295-regime-segmentation-part-b-persistent-regimes-without-labels:ml-experiments/00328-lsc-regime-changes-detect-then-model-part-1 \
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= systemd-run --user --scope -p MemoryMax=16G \
        -p OOMPolicy=continue uv run --no-sync python \
        ml-experiments/00328-*/e00328_detect.py --stage base|inject
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from datetime import UTC, datetime

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00295_segment as seg295  # ty: ignore[unresolved-import]
import e00328_common as c
import e00328_detectors as det
import numpy as np
import polars as pl
import torch
from studies.dp_features import naive_utc

BASE_DIR = c.WORK_DIR / "base"
INJECT_DIR = c.WORK_DIR / "inject"
ONSETS = [(*o, "tune") for o in seg295.ODD_ONSETS] + [(*o, "report") for o in seg295.EVEN_ONSETS]
SHAPES = [("step", 0.10), ("step", 0.25), ("ramp", 0.10)]


def _atomic(*, frame: pl.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(path)


def residual_stream(*, cache, site: str, cuts, rows, delta: np.ndarray | None):
    """Return times, power, and residual of the months after `cuts`, optionally with a delta."""
    t_all, y_all, r_all = [], [], []
    for cut in cuts:
        target, y, resid = c.out_of_sample_residual(cache=cache, site=site, cut=cut, rows=rows)
        if len(target) == 0:
            continue
        if delta is not None:
            y, resid = y + delta[target], resid + delta[target]
        t_all.append(cache.end_times[target])
        y_all.append(y)
        r_all.append(resid)
    if not t_all:
        return np.array([], dtype="datetime64[us]"), np.array([]), np.array([])
    return np.concatenate(t_all), np.concatenate(y_all), np.concatenate(r_all)


def alarms_frame(*, site: str, detector: str, param: float, alarms) -> list[dict]:
    """Return one row per alarm."""
    return [
        {
            "site": site,
            "detector": detector,
            "param": param,
            "alarm_day": str(a.alarm_day),
            "change_day": str(a.change_day),
        }
        for a in alarms
    ]


def base_job(site: str) -> str:
    """Write one Primary's residual stream, weekly features, and real-record alarms."""
    torch.set_num_threads(1)
    target = BASE_DIR / f"{site}.npz"
    if target.exists():
        return f"{site}: exists"
    cache = c295.load_cache()
    rows = c295.all_rows(cache=cache, site=site)
    times, power, resid = residual_stream(
        cache=cache, site=site, cuts=c.monthly_cuts(), rows=rows, delta=None
    )
    days, means = det.daily_means(end_times=times, residual=resid)
    weeks, features = det.weekly_features(end_times=times, power=power, residual=resid)
    out: list[dict] = []
    for h in c.CUSUM_H_GRID:
        out += alarms_frame(
            site=site,
            detector="rolling_gam",
            param=h,
            alarms=det.cusum_alarms(days=days, means=means, h=h),
        )
    for beta in c.HISTOGRAM_BETA_GRID:
        out += alarms_frame(
            site=site,
            detector="histogram",
            param=beta,
            alarms=det.histogram_alarms(weeks=weeks, features=features, beta=beta),
        )
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        out,
        schema={
            "site": pl.String,
            "detector": pl.String,
            "param": pl.Float64,
            "alarm_day": pl.String,
            "change_day": pl.String,
        },
    )
    _atomic(frame=frame, path=BASE_DIR / f"{site}_alarms.parquet")
    tmp = BASE_DIR / f"{site}.{os.getpid()}.tmp.npz"
    np.savez(tmp, times=times, power=power, resid=resid, weeks=weeks, features=features)
    tmp.rename(target)
    return f"{site}: {len(times):,} half-hours, {len(out)} alarms"


def inject_job(args: tuple[str, int, int, str]) -> str:
    """Inject each shape at one onset into one Primary and record the detections."""
    site, year, month, parity = args
    torch.set_num_threads(1)
    target = INJECT_DIR / f"{site}_{year}_{month:02d}.parquet"
    if target.exists():
        return f"{site} {year}-{month}: exists"
    cache = c295.load_cache()
    onset = datetime(year, month, 1, tzinfo=UTC)
    before = (cache.end_times <= naive_utc(moment=onset)) & (
        cache.end_times > naive_utc(moment=onset) - np.timedelta64(365, "D")
    )
    y_site = cache.sites[site].y
    if np.isfinite(y_site[before]).sum() < 0.5 * 17520:
        return f"{site} {year}-{month}: under half a year of readings before the onset"
    level = float(np.nanmean(y_site[before]))
    t = cache.end_times
    base = np.load(BASE_DIR / f"{site}.npz")
    rows = c295.all_rows(cache=cache, site=site)
    cuts_after = [c.add_months(moment=onset, months=k) for k in range(6)]
    cuts_after = [x for x in cuts_after if x <= c.LAST_CUT]
    onset64 = np.datetime64(onset.replace(tzinfo=None), "D")
    old = base["times"] < naive_utc(moment=onset)
    results: list[dict] = []
    for shape, size in SHAPES:
        ramp = np.clip((t - naive_utc(moment=onset)) / np.timedelta64(c.RAMP_DAYS, "D"), 0.0, 1.0)
        step = (t > naive_utc(moment=onset)).astype(float)
        delta = size * level * (ramp if shape == "ramp" else step)
        t_new, y_new, r_new = residual_stream(
            cache=cache, site=site, cuts=cuts_after, rows=rows, delta=delta
        )
        times = np.concatenate([base["times"][old], t_new])
        power = np.concatenate([base["power"][old], y_new])
        resid = np.concatenate([base["resid"][old], r_new])
        days, means = det.daily_means(end_times=times, residual=resid)
        for h in c.CUSUM_H_GRID:
            alarms = [
                a for a in det.cusum_alarms(days=days, means=means, h=h) if a.alarm_day >= onset64
            ]
            results.append(
                score(site, "rolling_gam", h, onset, parity, shape, size, alarms, onset64)
            )
        # weekly features: reuse the baseline weeks before the onset week, recompute from there
        onset_week = onset64.astype("datetime64[D]").astype(np.int64) // 7 * 7
        keep = base["weeks"].astype(np.int64) < onset_week
        late = c.day_of(end_times=times).astype(np.int64) >= onset_week
        w_new, f_new = det.weekly_features(
            end_times=times[late], power=power[late], residual=resid[late]
        )
        weeks = np.concatenate([base["weeks"][keep], w_new])
        features = np.vstack([base["features"][keep], f_new])
        first = max(2 * c.HISTOGRAM_MIN_WEEKS, int(keep.sum()) - 2)
        for beta in c.HISTOGRAM_BETA_GRID:
            alarms = det.histogram_alarms(
                weeks=weeks, features=features, beta=beta, origins=range(first, len(weeks))
            )
            near = [
                a
                for a in alarms
                if abs((a.change_day - onset64) / np.timedelta64(1, "D")) <= c.HISTOGRAM_MATCH_DAYS
            ]
            results.append(
                score(site, "histogram", beta, onset, parity, shape, size, near, onset64)
            )
    _atomic(frame=pl.DataFrame(results), path=target)
    return f"{site} {year}-{month}: {len(results)} rows"


def score(site, detector, param, onset, parity, shape, size, alarms, onset64) -> dict:
    """Return one detection row: delay and change-date error of the first matching alarm."""
    hit = None
    for a in alarms:
        delay = (a.alarm_day - onset64) / np.timedelta64(1, "D")
        if 0 <= delay <= c.DETECT_WITHIN_DAYS:
            hit = a
            break
    return {
        "site": site,
        "detector": detector,
        "param": float(param),
        "onset": str(onset.date()),
        "parity": parity,
        "shape": shape,
        "size": size,
        "detected": hit is not None,
        "delay_days": None
        if hit is None
        else float((hit.alarm_day - onset64) / np.timedelta64(1, "D")),
        "date_error_days": None
        if hit is None
        else float((hit.change_day - onset64) / np.timedelta64(1, "D")),
        "n_alarms_after_onset": len(alarms),
    }


def main() -> None:
    """Run one stage in parallel with the spawn start method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "inject"), required=True)
    parser.add_argument("--workers", type=int, default=c.n_workers())
    parser.add_argument("--count", type=int, default=0)
    args = parser.parse_args()
    sites = json.loads((c295.EXPERIMENT_DIR / "usable_sites.json").read_text())
    if args.stage == "base":
        jobs, fn = sites, base_job
    else:
        jobs = [(s, y, m, p) for s in sites if s != "D8" for (y, m, p) in ONSETS]
        fn = inject_job
    if args.count:
        jobs = jobs[: args.count]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(fn, jobs, chunksize=1):  # ty: ignore[invalid-argument-type]
            print(line, flush=True)


if __name__ == "__main__":
    main()
