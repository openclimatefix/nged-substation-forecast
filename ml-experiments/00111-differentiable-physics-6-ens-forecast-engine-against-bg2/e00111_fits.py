"""Fit the convex model at every monthly cut, per arm and substation, for experiment 111.

Both arms use Part 4's window of 24 months, the ridge of 0.1, and the capacity pinned to
experiment 45's real-time `frozen` vintage (`studies.dp_pinned_capacity`). They differ in the drift:

- `gam0` carries Part 4's drift (form D1, with Part 4's tuned drift step), as Part 4's window
  arm W does. It is the lagless arm of the plan's C1 and C4.
- `gamk` has no drift, as Part 4's C4 passed non-inferiority in both years. It is the main arm,
  and the filter is added to it afterwards.

A fit at cut `c` trains on rows that end no later than `c` and serves the forecasts of the runs
whose `power_fcst_init_time` is at or after `c` plus 3.5 hours and before the next cut. A site's
first cut is the first monthly cut with at least `MIN_TRAINING_DAYS` days of usable rows, which
applies experiment 103's start cuts, D18's commissioning months included.

    uv run python ml-experiments/00111-*/e00111_fits.py [--workers 4] [--arms gam0 gamk]
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Final

import e00111_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache, naive_utc
from studies.dp_gam import GamFit, GamSpec, TrainingWindow, fit_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

RIDGE: Final[float] = 0.1
"""Experiment 104's tuned ridge penalty."""
WINDOW_MONTHS: Final[int] = 24
ARMS: Final[tuple[str, ...]] = ("gam0", "gamk")


def tuned() -> dict:
    """Return Part 4's tuned settings (drift step, step penalty, final knot, window)."""
    return json.loads(common.PART_4_TUNED.read_text())


def arm_spec(*, arm: str, settings: dict) -> GamSpec:
    """Return an arm's penalties and drift form.

    Args:
        arm: `gam0` or `gamk`.
        settings: Part 4's tuned settings.

    Returns:
        The spec.

    Raises:
        ValueError: If the arm is unknown.
    """
    base = GamSpec(
        ridge=RIDGE,
        drift_step=settings["drift_step"],
        step_penalty=settings["step_penalty"],
        capacity="pinned",
        final_knot="free",
    )
    if arm == "gam0":
        return replace(base, drift="d1")
    if arm == "gamk":
        return replace(base, drift="none")
    msg = f"unknown arm {arm!r}"
    raise ValueError(msg)


def fit_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's fits at one site."""
    return common.FITS_DIR / arm / f"{site}.pkl"


def first_cut_index(*, cache: FeatureCache, site: str, cuts: list[datetime]) -> int:
    """Return the index of the first cut with `MIN_TRAINING_DAYS` days of usable rows.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cuts: The monthly cuts, ascending.

    Returns:
        The index into `cuts`.

    Raises:
        ValueError: If no cut has enough rows.
    """
    features = cache.sites[site]
    usable = features.default_rows & np.isfinite(features.y)
    needed = common.MIN_TRAINING_DAYS * 48
    for index, cut in enumerate(cuts):
        if int((usable & (cache.end_times <= naive_utc(moment=cut))).sum()) >= needed:
            return index
    msg = f"{site} never has {common.MIN_TRAINING_DAYS} days of usable rows"
    raise ValueError(msg)


def fit_site(*, arm: str, site: str) -> str:
    """Fit one arm at one site for every monthly cut from its first cut, and store the fits.

    Args:
        arm: `gam0` or `gamk`.
        site: The substation's label.

    Returns:
        A progress line.
    """
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    cuts = common.monthly_cuts()
    spec = arm_spec(arm=arm, settings=tuned())
    first = first_cut_index(cache=cache, site=site, cuts=cuts)
    fits: list[GamFit] = []
    for cut in cuts[first:]:
        pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
        fits.append(
            fit_gam(
                cache=cache,
                site=site,
                specs=[spec],
                cut=cut,
                window=TrainingWindow(months=WINDOW_MONTHS),
                season_years=common.SEASON_YEARS,
                pinned=pinned,
            )
        )
    target = fit_path(arm=arm, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(fits))
    temporary.rename(target)
    return (
        f"{arm} {site}: {len(fits)} fits from {fits[0].cut:%Y-%m-%d}, "
        f"{time.monotonic() - started:.0f} s"
    )


def load_fits(*, arm: str, site: str) -> list[GamFit]:
    """Return the stored monthly fits of an arm at a site, ascending by cut."""
    return pickle.loads(fit_path(arm=arm, site=site).read_bytes())


def _job(job: tuple[str, str]) -> str:
    torch.set_num_threads(1)
    return fit_site(arm=job[0], site=job[1])


def main() -> None:
    """Fit every arm and site in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    args = parser.parse_args()
    jobs = [(arm, site) for arm in args.arms for site in common.ALL_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
