# ruff: noqa: E501, ANN001, C416, D208
r"""Stage 1: monthly forecasts of every arm, for the real records and the injected-step worlds.

    PYTHONPATH=ml-experiments/00295-*:ml-experiments/00328-* OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00331-*/e00331_fit.py --stage real|inject
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
from datetime import UTC, datetime

import e00295_common as c295  # ty: ignore[unresolved-import]
import e00295_segment as seg295  # ty: ignore[unresolved-import]
import e00331_common as c
import numpy as np
from studies.dp_ens_weather import replace_site
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam


def month_rows(*, cache: FeatureCache, rows: np.ndarray, cut: datetime) -> np.ndarray:
    """Return the grid rows of the month after `cut`."""
    start, stop = naive_utc(moment=cut), naive_utc(moment=c.add_months(moment=cut, months=1))
    return rows[(cache.end_times[rows] > start) & (cache.end_times[rows] <= stop)]


def forecasts_at(
    *, cache: FeatureCache, site: str, cut: datetime, target: np.ndarray, ridges: tuple[float, ...],
    arms: tuple[str, ...],
) -> dict[str, np.ndarray]:  # fmt: skip
    """Return each arm's forecast of the rows `target` for every ridge, keyed `arm|ridge`."""
    out: dict[str, np.ndarray] = {}
    weighted = tuple(a for a in arms if a in c.WEIGHTED_ARMS)
    _, ends = c.training_end_times(cache=cache, site=site, cut=cut)
    weights = c.row_weights(end_times=ends, cut=cut, arms=weighted) if weighted else None
    for ridge in ridges:
        spec = GamSpec(ridge=ridge, drift="none", capacity="fitted", final_knot="free")
        if "R0" in arms:
            fit = fit_gam(
                cache=cache,
                site=site,
                cut=cut,
                specs=[spec],
                season_years=c295.SEASON_YEARS,
                window=TrainingWindow(),
            )
            out[f"R0|{ridge}"] = predict_gam(cache=cache, fit=fit, index=target)
        if "W24" in arms:
            window = TrainingWindow(months=c.WINDOW_MONTHS)
            fit = fit_gam(
                cache=cache,
                site=site,
                cut=cut,
                specs=[spec],
                season_years=c295.SEASON_YEARS,
                window=window,
            )
            out[f"W24|{ridge}"] = predict_gam(cache=cache, fit=fit, index=target)
        if weights is not None:
            fit = fit_gam(
                cache=cache,
                site=site,
                cut=cut,
                specs=[spec],
                season_years=c295.SEASON_YEARS,
                window=TrainingWindow(),
                weights=weights,
            )
            for k, arm in enumerate(weighted):
                out[f"{arm}|{ridge}"] = predict_gam(cache=cache, fit=fit, index=target, member=k)
    return out


def atomic_pickle(*, obj: object, path) -> None:
    """Write a pickle atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(obj))
    tmp.rename(path)


def job(args: tuple[str, str, int, int, float]) -> str:
    """Run one Primary's real record or one injected world, writing a pickle of forecasts."""
    kind, site, year, month, size = args
    cache = c295.load_cache()
    rows = c295.all_rows(cache=cache, site=site)
    if kind == "real":
        path = c.STREAM_DIR / f"real_{site}.pkl"
        cuts, ridges, arms, base, use_cache = c.monthly_cuts(), c.RIDGES, c.REAL_ARMS, {}, cache
    else:
        path = c.STREAM_DIR / f"inj_{site}_{size:.2f}_{year}_{month:02d}.pkl"
        onset = datetime(year, month, 1, tzinfo=UTC)
        base = pickle.loads((c.STREAM_DIR / f"real_{site}.pkl").read_bytes())
        before = (cache.end_times <= naive_utc(moment=onset)) & (
            cache.end_times > naive_utc(moment=onset) - np.timedelta64(365, "D")
        )
        y = cache.sites[site].y
        if np.isfinite(y[before]).sum() < 0.5 * 17520:
            return f"{site} {size} {year}-{month}: too little history"
        delta = np.where(
            cache.end_times > naive_utc(moment=onset), size * float(np.nanmean(y[before])), 0.0
        )
        use_cache = replace_site(cache=cache, site=site, y=y + delta)
        cuts = [
            m
            for m in c.monthly_cuts()
            if onset < m <= c.add_months(moment=onset, months=c.MONTHS_AFTER)
        ]
        ridges, arms = (0.1,), c.INJECT_ARMS
    if path.exists():
        return f"{path.name}: exists"
    streams: dict = {k: v for k, v in base.items()} if kind != "real" else {}
    for cut in cuts:
        target = month_rows(cache=use_cache, rows=rows, cut=cut)
        if len(target) == 0:
            continue
        try:
            pred = forecasts_at(
                cache=use_cache, site=site, cut=cut, target=target, ridges=ridges, arms=arms
            )
        except ValueError:  # no training rows before this cut
            continue
        streams[cut] = {"idx": target, **pred}
    atomic_pickle(obj=streams, path=path)
    return f"{path.name}: {len([k for k in streams if isinstance(k, datetime)])} cuts"


def main() -> None:
    """Run one stage in parallel with the spawn start method."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("real", "inject"), required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--sites", nargs="+", default=None)
    args = parser.parse_args()
    sites = args.sites or json.loads((c295.EXPERIMENT_DIR / "usable_sites.json").read_text())
    if args.stage == "real":
        jobs = [("real", s, 0, 0, 0.0) for s in sites]
    else:
        jobs = [
            ("inj", s, y, m, size)
            for s in sites
            if s != "D8"
            for (y, m) in seg295.EVEN_ONSETS
            for size in (0.10, 0.25)
        ]
    if args.count:
        jobs = jobs[: args.count]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(job, jobs, chunksize=1):
            print(line, flush=True)


if __name__ == "__main__":
    main()
