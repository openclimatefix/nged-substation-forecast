"""Forecast gain from an offset for the latest regime, found by segmenting data before each cut.

Needs `chosen_config.json` (written by `e00295_tune.py`) in the experiment's data folder.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
        uv run --no-sync python e00295_gain.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import e00295_common as c
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00295_segment import STEP_SIZES, delta_of
from studies.dp_ens_weather import replace_site
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam

WINDOW = TrainingWindow(months=24)
MONTHS_AFTER = 6
TRAILING_DAYS = 60
EVEN_ONSETS = [(2021, 10), (2021, 12), (2022, 6), (2022, 12), (2024, 2), (2024, 8), (2025, 4), (2025, 10)]
_STATE: dict = {}


def add_months(*, moment: datetime, months: int) -> datetime:
    """Return the first of the month `months` after the first-of-month `moment`."""
    index = moment.year * 12 + moment.month - 1 + months
    return datetime(index // 12, index % 12 + 1, 1, tzinfo=UTC)


def one_cut(
    *, cache: FeatureCache, site: str, cut: datetime, config: tuple[float, int], rows: np.ndarray
) -> dict:
    """Score the three arms over the month after `cut`."""
    stamps = cache.end_times
    cut64, stop64 = naive_utc(moment=cut), naive_utc(moment=add_months(moment=cut, months=1))
    start64 = naive_utc(moment=add_months(moment=cut, months=-24))
    insample = rows[(stamps[rows] > start64) & (stamps[rows] <= cut64)]
    target = rows[(stamps[rows] > cut64) & (stamps[rows] <= stop64)]
    if len(target) == 0:
        return {}
    fit = fit_gam(
        cache=cache, site=site,
        specs=[GamSpec(ridge=c.RIDGE, drift="none", capacity="fitted", final_knot="free")],
        cut=cut, window=WINDOW, season_years=c.SEASON_YEARS,
    )
    y = cache.sites[site].y
    residual = y[insample] - predict_gam(cache=cache, fit=fit, index=insample)
    pred = predict_gam(cache=cache, fit=fit, index=target)
    series = c.daily_series(end_times=stamps[insample], residual=residual)
    sigma = c.noise_scale(values=series.values)
    starts = c.segment(values=series.values, sigma=sigma, beta=config[0], min_length=config[1])
    boundaries = c.boundaries_of(series=series, starts=starts)
    offset_segment, boundary_date = 0.0, ""
    if boundaries:
        latest = boundaries[-1].date
        since = stamps[insample] >= latest.astype("datetime64[us]")
        offset_segment = float(residual[since].mean()) if since.any() else 0.0
        boundary_date = str(latest)
    recent = stamps[insample] > cut64 - np.timedelta64(TRAILING_DAYS, "D")
    offset_trailing = float(residual[recent].mean()) if recent.any() else 0.0
    truth = y[target]
    return {
        "n_rows": len(target),
        "mae_none": float(np.mean(np.abs(pred - truth))),
        "mae_segment": float(np.mean(np.abs(pred + offset_segment - truth))),
        "mae_trailing": float(np.mean(np.abs(pred + offset_trailing - truth))),
        "bias_none": float(np.mean(pred - truth)),
        "offset_segment": offset_segment,
        "offset_trailing": offset_trailing,
        "latest_boundary": boundary_date,
    }


def scenario_rows(
    *, cache: FeatureCache, site: str, label: str, onset: datetime, config: tuple[float, int]
) -> list[dict]:
    """Run the six monthly cuts after an onset on a (possibly injected) cache."""
    rows = c.all_rows(cache=cache, site=site)
    first = add_months(moment=datetime(onset.year, onset.month, 1, tzinfo=UTC), months=1)
    out = []
    for k in range(MONTHS_AFTER):
        cut = add_months(moment=first, months=k)
        if cut >= c.END_OF_DATA:
            break
        result = one_cut(cache=cache, site=site, cut=cut, config=config, rows=rows)
        if result:
            out.append({"site": site, "scenario": label, "onset": str(onset.date()),
                        "cut": str(cut.date()), "month_after": k + 1, **result})
    return out


def _job(site: str) -> str:
    torch.set_num_threads(1)
    cache: FeatureCache = _STATE["cache"]
    config = _STATE["config"]
    target = c.EXPERIMENT_DIR / "gain" / f"{site}.parquet"
    if target.exists():
        return f"{site}: already done"
    rows: list[dict] = []
    if site == "D8":
        rows += scenario_rows(
            cache=cache, site=site, label="real_d8", onset=datetime(2023, 8, 15, tzinfo=UTC),
            config=config,
        )
    else:
        for year, month in EVEN_ONSETS:
            onset = datetime(year, month, 1, tzinfo=UTC)
            for size in STEP_SIZES:
                for shape in ("flat", "share"):
                    delta = delta_of(cache=cache, site=site, onset=onset, kind=shape, size=size, days=None)
                    injected = replace_site(cache=cache, site=site, y=cache.sites[site].y + delta)
                    rows += scenario_rows(
                        cache=injected, site=site, label=f"persistent_{shape}_{size:.2f}",
                        onset=onset, config=config,
                    )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return f"{site}: {frame.height} rows"


def main() -> None:
    """Run the gain scenarios in parallel by Primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sites", nargs="+", default=None)
    args = parser.parse_args()
    (c.EXPERIMENT_DIR / "gain").mkdir(parents=True, exist_ok=True)
    chosen = json.loads((c.EXPERIMENT_DIR / "chosen_config.json").read_text())
    _STATE["config"] = (float(chosen["beta"]), int(chosen["min_length"]))
    _STATE["cache"] = cache = c.load_cache()
    states = json.loads((c.EXPERIMENT_DIR / "usable_sites.json").read_text())
    sites = args.sites or states
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for line in pool.map(_job, sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
