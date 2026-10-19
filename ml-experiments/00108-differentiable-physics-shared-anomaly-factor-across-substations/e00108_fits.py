"""Fit the frozen convex model A0 of every contributor at each 1 July from 2021 to 2025.

A0 is experiment 106's arm of that name: the convex model of `studies.dp_gam` on all years with
the lockdown out, no drift, a free final knot, and the capacity pinned to the real-time vintage
(`studies.dp_pinned_capacity`). A site is fitted at a cut only if it has at least
`MIN_TRAINING_MONTHS` months of readings before it. Each fit stores:

- the scored year's rows (readings ending in the year after the cut) and its mean forecast there,
  so `actual - mean` is an out-of-sample residual;
- the in-sample residuals of the 12 months before the cut, which set the residual scale.

    uv run python ml-experiments/00108-*/e00108_fits.py [--workers 6]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00108_common as common
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

FIT_DIR: Final[Path] = common.WORK_DIR / "fits"
IN_SAMPLE_MONTHS: Final[int] = 12
DEFAULT_RIDGE: Final[float] = 0.1


def a0_spec() -> GamSpec:
    """Return experiment 106's A0 spec, built from its stored tuned settings."""
    tuned = json.loads((common.PART4_DIR / "work" / "tuned.json").read_text())
    base = GamSpec(
        ridge=DEFAULT_RIDGE,
        drift_step=tuned["drift_step"],
        step_penalty=tuned["step_penalty"],
        capacity="pinned",
        final_knot=tuned["final_knot"],
        final_knot_shrink=tuned["final_knot_shrink"],
    )
    return replace(base, drift="none", final_knot="free")


def months_before(*, moment: datetime, months: int) -> datetime:
    """Return `moment` moved back by whole months (the day of month is 1 for a 1 July cut)."""
    total = moment.year * 12 + moment.month - 1 - months
    return moment.replace(year=total // 12, month=total % 12 + 1)


def has_fit(*, cache: FeatureCache, site: str, year: int) -> bool:
    """Return True if the site has `MIN_TRAINING_MONTHS` months of readings before the cut."""
    cut = common.cut_of(year=year)
    readings = common.scoring_index(
        cache=cache, site=site, after=datetime(2019, 1, 1, tzinfo=UTC), through=cut
    )
    if len(readings) == 0:
        return False
    first = cache.end_times[readings[0]]
    limit = common.to_64(moment=months_before(moment=cut, months=common.MIN_TRAINING_MONTHS))
    return bool(first <= limit)


def fit_path(*, site: str, year: int) -> Path:
    """Return where a fit's results are stored."""
    return FIT_DIR / f"{site}_{year}.npz"


def fit_one(job: tuple[str, int]) -> str:
    """Fit one site at one cut and store the scored year's mean and the in-sample residuals."""
    import torch

    torch.set_num_threads(1)
    site, year = job
    cache = common.load_cache()
    cut = common.cut_of(year=year)
    vintages = read_vintages()
    spec = a0_spec()
    pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
    try:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=TrainingWindow(),
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
    except ValueError as error:
        return f"{site} {year}: no fit ({error})"
    index = common.scoring_index(
        cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
    )
    in_sample = common.scoring_index(
        cache=cache,
        site=site,
        after=months_before(moment=cut, months=IN_SAMPLE_MONTHS),
        through=cut,
    )
    y = cache.sites[site].y
    mean = predict_gam(cache=cache, fit=fit, index=index)
    in_sample_mean = predict_gam(cache=cache, fit=fit, index=in_sample)
    FIT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        fit_path(site=site, year=year),
        index=index,
        mean=mean,
        in_sample_index=in_sample,
        in_sample_residual=y[in_sample] - in_sample_mean,
    )
    return f"{site} {year}: {len(index)} rows, held capacity {pinned.held_points:.3f}"


def main() -> None:
    """Fit every contributor at every cut it qualifies for."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--sites", nargs="*", default=list(common.SCORED_SITES))
    parser.add_argument("--years", nargs="*", type=int, default=list(common.CUT_YEARS))
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.load_cache()
    jobs = [
        (site, year)
        for year in arguments.years
        for site in arguments.sites
        if has_fit(cache=cache, site=site, year=year)
        and not fit_path(site=site, year=year).exists()
    ]
    print(f"{len(jobs)} fits to run", flush=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for line in pool.map(fit_one, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
