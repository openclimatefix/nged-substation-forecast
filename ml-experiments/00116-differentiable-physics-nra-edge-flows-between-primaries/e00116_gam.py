"""Fit GAM-K's convex model without and with switching masks, for experiment 116.

One frozen fit per arm, substation, and cut, on the 24 months before the cut (no drift, capacity
pinned to experiment 45's real-time vintage for MW substations, dark rows only for MVA ones). Arm
`a0` trains on every row outside the lockdown window and keeps the hand-filtered switching events;
arm `a1` drops them, as experiment 111 does. Each job writes the fit's mean over every row with a
reading from the start of the training window to the end of the scored year, so the in-sample
residuals (needed for noise scales, filter hyperparameters, and placebo checks) and the
out-of-sample residuals of the scored year are both on file.

A job writes `work/gam/<arm>/<site>_<cut>.parquet` atomically and is skipped if it exists.

    uv run python ml-experiments/00116-*/e00116_gam.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import e00116_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import TrainingWindow, fit_gam, predict_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

ArmType = Literal["a0", "a1"]
"""`a0` keeps the switching events in training, `a1` drops them."""

ARMS: Final[tuple[ArmType, ...]] = ("a0", "a1")
SITES: Final[tuple[str, ...]] = (
    "D2",
    "D3",
    "D4",
    "D5",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D14",
    "D16",
    "D19",
    "D21",
    "D22",
)
"""Every Primary in the decision graph. The two series labelled G-c are fitted by a separate
sensitivity run."""


def arm_cache(*, cache: FeatureCache, site: str, arm: ArmType) -> FeatureCache:
    """Return the cache with the site's training rows set for an arm.

    Args:
        cache: The feature cache.
        site: The substation's label.
        arm: `a0` trains on the pool outside the lockdown window; `a1` also drops the switching
            events (the cache's default rows).

    Returns:
        A cache whose site has the arm's `default_rows`.
    """
    features = cache.sites[site]
    if arm == "a1":
        return cache
    rows = features.in_pool & ~features.lockdown
    return replace(cache, sites={**cache.sites, site: replace(features, default_rows=rows)})


def usable_months(*, cache: FeatureCache, site: str, cut: datetime) -> float:
    """Return how many months of pool rows the site has in the 24 months before a cut."""
    features = cache.sites[site]
    rows = np.zeros(len(cache.end_times), dtype=bool)
    rows[common.window_slice(cache=cache, cut=cut)] = True
    before = cache.end_times <= common.to_64(moment=cut)
    return float(np.sum(features.in_pool & features.present & rows & before)) / 1460.0


def job_path(*, arm: ArmType, site: str, cut: datetime) -> Path:
    """Return the file a job writes."""
    return common.WORK_DIR / "gam" / arm / f"{site}_{common.cut_label(cut=cut)}.parquet"


def fit_job(
    *, cache: FeatureCache, vintages: pl.DataFrame, arm: ArmType, site: str, cut: datetime
) -> pl.DataFrame:
    """Fit one arm at one site and cut, and return the mean over the window and scored year."""
    features = cache.sites[site]
    mva = features.table.units == "MVA"
    pinned = None
    if not mva:
        pinned = pinned_capacity(vintages=vintages, table=features.table, cut=cut)
    fit = fit_gam(
        cache=arm_cache(cache=cache, site=site, arm=arm),
        site=site,
        specs=[common.gam_spec(units=features.table.units)],
        cut=cut,
        window=TrainingWindow(months=common.WINDOW_MONTHS, dark_only=mva),
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    rows = np.zeros(len(cache.end_times), dtype=bool)
    rows[common.window_slice(cache=cache, cut=cut)] = True
    keep = rows & features.present & np.isfinite(features.y)
    if mva:
        keep &= features.dark
    index = np.flatnonzero(keep)
    mean = predict_gam(cache=cache, fit=fit, index=index)
    return pl.DataFrame(
        {
            "position": index.astype(np.int64),
            "y": features.y[index],
            "mean": mean,
            "default_row": features.default_rows[index],
            "switching_mask": features.switching_mask[index],
        }
    )


_STATE: dict[str, Any] = {}


def _worker(job: tuple[ArmType, str, datetime]) -> str:
    """Fit one job in a worker process with one thread, and write its file."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    arm, site, cut = job
    target = job_path(arm=arm, site=site, cut=cut)
    if target.exists():
        return f"skip {target.name}"
    if "cache" not in _STATE:
        _STATE["cache"] = common.load_cache()
        _STATE["vintages"] = read_vintages()
    cache = _STATE["cache"]
    if usable_months(cache=cache, site=site, cut=cut) < common.MIN_TRAIN_MONTHS:
        return f"too little history: {arm} {site} {common.cut_label(cut=cut)}"
    frame = fit_job(cache=cache, vintages=_STATE["vintages"], arm=arm, site=site, cut=cut)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.replace(target)
    return f"done {arm} {target.name}: {frame.height} rows"


def main() -> None:
    """Fit every job in parallel worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    jobs = [(arm, site, cut) for cut in common.CUTS.values() for arm in ARMS for site in SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
