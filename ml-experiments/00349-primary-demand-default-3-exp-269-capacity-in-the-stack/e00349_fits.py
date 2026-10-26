"""Fit GAM-K at every monthly cut for each capacity arm of experiment 349.

The arms differ only in the `PinnedCapacity` given to `fit_gam` (see `e00349_adaptor.py`):

| Arm | Capacity |
|---|---|
| `pin` | experiment 45's `frozen` vintage through the adaptor (arm A, the wiring check) |
| `b` | experiment 269's pooled-empirical-shape estimate, fitted with rows before each cut |
| `c` | arm B's estimate held at its value at the first scored origin (2025-07-01) |

The spec, window (24 months), and cuts are experiment 111's `gamk`. D9 keeps the pin where its
history is too short for an as-of block.

    uv run --no-sync python ml-experiments/00349-*/e00349_fits.py [--workers 6] [--arms pin b c]
"""

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Final

import e00349_adaptor as adaptor
import e00349_common as common
import e00349_run_common as run_common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

ARMS: Final[tuple[str, ...]] = ("pin", "b", "c")
POST_HOC_ARMS: Final[tuple[str, ...]] = ()
TABLE_SITES: Final[tuple[str, ...]] = common.TABLE_SITES


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


def capacity_for(*, arm: str, site: str, cut: datetime, pin, table) -> tuple:  # noqa: ANN001
    """Return the arm's capacity at a cut, and the number of path weeks that took the pin.

    Arm B reads the estimate fitted with rows before the cut. Arm C reads the estimate at the cut
    or at `HELD_ORIGIN`, whichever is earlier. A Primary with no block at that origin (D9 before
    its history is long enough) keeps the pin, and the weeks are counted.
    """
    if arm == "pin":
        return adaptor.pin_adaptor(pin=pin), 0
    origin = cut if arm == "b" else min(cut, common.HELD_ORIGIN)
    estimates, data_years = adaptor.asof_estimates(table=table, site=site, origin=origin)
    if not any(np.isfinite(v) for v in estimates.values()) or not data_years:
        return adaptor.pin_adaptor(pin=pin), len(adaptor.WEEKS)
    return adaptor.table_adaptor(pin=pin, estimates=estimates, data_years=data_years)


def fit_site(*, arm: str, site: str) -> str:
    """Fit one arm at one site for every monthly cut from its first cut, and store the fits."""
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    cuts = run_common.monthly_cuts()
    spec = run_common.base_spec()
    first = first_cut_index(cache=cache, site=site, cuts=cuts)
    table = read_asof_table() if arm != "pin" else None
    fits: list[GamFit] = []
    fallback_weeks = 0
    for cut in cuts[first:]:
        pin = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
        capacity, took_pin = capacity_for(arm=arm, site=site, cut=cut, pin=pin, table=table)
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
    return [(arm, site) for arm in arms for site in sites if arm == "pin" or site in TABLE_SITES]


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
