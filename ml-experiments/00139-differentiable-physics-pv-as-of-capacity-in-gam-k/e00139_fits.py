"""Fit GAM-K at every monthly cut for each capacity arm of experiment 139.

The arms differ only in the `PinnedCapacity` given to `fit_gam` (see `e00139_adaptor.py`):

| Arm | Capacity |
|---|---|
| `pin` | experiment 45's `frozen` vintage, passed through the adaptor (the wiring identity check) |
| `asof` | experiment 129's arm 2, fitted before the latest 1 July origin at or before the cut |
| `hindsight` | experiment 129's published table (data to 2026-07-31); a ceiling |
| `nopv` | zero |

The spec, window (24 months), and cuts are experiment 111's `gamk`. D9 has no experiment 129 table,
so it gets `pin` and `nopv` only.

    uv run --no-sync python ml-experiments/00139-*/e00139_fits.py [--workers 6] [--arms pin asof]
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00139_adaptor as adaptor
import e00139_common as common
import e00139_run_common as run_common
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

ARMS: Final[tuple[str, ...]] = ("pin", "asof", "hindsight", "nopv")
POST_HOC_ARMS: Final[tuple[str, ...]] = ("asof_scaled", "asof_raw")
SCALE_FACTOR: Final[float] = 1.15
"""Post hoc: one over the known-answer recovery of 0.87 (the median over the farms at the five
Primaries in the scored year), which says an ensemble-mean-driven forecast removes 0.87 of a
capacity-unit of solar per unit of capacity. It is one value chosen from that test and not tuned on
forecast scores."""
TABLE_SITES: Final[tuple[str, ...]] = common.DECISION_SITES


def fit_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's monthly fits at one site."""
    return run_common.WORK_DIR / "fits" / arm / f"{site}.pkl"


def read_asof_table() -> pl.DataFrame:
    """Return every Primary's and origin's as-of block."""
    return pl.concat(
        [
            pl.read_parquet(path)
            for path in sorted((common.WORK_DIR / "asof").glob("asof_*.parquet"))
        ]
    )


def first_cut_index(*, cache: FeatureCache, site: str, cuts: list[datetime]) -> int:
    """Return the index of the first cut with `MIN_TRAINING_DAYS` days of usable rows."""
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    needed = run_common.MIN_TRAINING_DAYS * 48
    for index, cut in enumerate(cuts):
        if int((usable & (cache.end_times <= naive_utc(moment=cut))).sum()) >= needed:
            return index
    msg = f"{site} never has {run_common.MIN_TRAINING_DAYS} days of usable rows"
    raise ValueError(msg)


def capacity_for(*, arm: str, site: str, cut: datetime, pin, table, hindsight) -> tuple:  # noqa: ANN001
    """Return the arm's capacity at a cut, and the number of path weeks that took the pin."""
    if arm == "pin":
        return adaptor.pin_adaptor(pin=pin), 0
    if arm == "nopv":
        return adaptor.no_pv_adaptor(pin=pin), 0
    if arm == "hindsight":
        estimates = hindsight.estimates[site]
        return adaptor.table_adaptor(pin=pin, estimates=estimates, data_years=set(estimates))
    scale = SCALE_FACTOR if arm == "asof_scaled" else 1.0
    origin = adaptor.origin_for_cut(cut=cut)
    if origin is None:
        msg = f"no as-of origin at or before {cut:%Y-%m-%d}"
        raise ValueError(msg)
    column = "raw_points" if arm == "asof_raw" else "estimate_points"
    estimates, data_years = adaptor.asof_estimates(
        table=table, site=site, origin=origin, column=column
    )
    estimates = {year: scale * value for year, value in estimates.items()}
    return adaptor.table_adaptor(pin=pin, estimates=estimates, data_years=data_years)


def fit_site(*, arm: str, site: str) -> str:
    """Fit one arm at one site for every monthly cut from its first cut, and store the fits."""
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    cuts = run_common.monthly_cuts()
    spec = run_common.base_spec()
    first = first_cut_index(cache=cache, site=site, cuts=cuts)
    table = read_asof_table() if arm in ("asof", "asof_scaled", "asof_raw") else None
    hindsight = adaptor.HindsightTable.read() if arm == "hindsight" else None
    fits: list[GamFit] = []
    fallback_weeks = 0
    for cut in cuts[first:]:
        pin = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
        capacity, took_pin = capacity_for(
            arm=arm, site=site, cut=cut, pin=pin, table=table, hindsight=hindsight
        )
        fallback_weeks += took_pin
        fits.append(
            fit_gam(
                cache=cache,
                site=site,
                specs=[spec],
                cut=cut,
                window=run_common.base_window(),
                season_years=run_common.SEASON_YEARS,
                pinned=capacity,
            )
        )
    target = fit_path(arm=arm, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(target)
    return (
        f"{arm} {site}: {len(fits)} fits from {fits[0].cut:%Y-%m-%d}, "
        f"{fallback_weeks} path weeks took the pin, {time.monotonic() - started:.0f} s"
    )


def load_fits(*, arm: str, site: str) -> list[GamFit]:
    """Return the stored monthly fits of an arm at a site, ascending by cut."""
    return pickle.loads(fit_path(arm=arm, site=site).read_bytes())


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    return fit_site(arm=job[0], site=job[1])


def jobs_for(*, arms: list[str], sites: list[str]) -> list[tuple[str, str]]:
    """Return the (arm, site) jobs, leaving out the table arms at a site with no table."""
    return [
        (arm, site)
        for arm in arms
        for site in sites
        if arm in ("pin", "nopv") or site in TABLE_SITES
    ]


def main() -> None:
    """Fit every requested arm and Primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--sites", nargs="+", default=list(run_common.SCORED_SITES))
    arguments = parser.parse_args()
    jobs = jobs_for(arms=arguments.arms, sites=arguments.sites)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
