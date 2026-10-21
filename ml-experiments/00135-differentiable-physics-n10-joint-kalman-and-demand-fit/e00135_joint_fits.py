"""Fit the convex model through the Kalman filter at every monthly cut, for experiment 135.

For each substation and monthly cut, one pass of the filter over the design columns
(`studies.dp_joint_fit`) gives five fits: J1 (no tie to the two-stage fit) and the four J2 arms,
which add a ridge of weight `lambda` towards the two-stage fit at the same cut. The two-stage fits
are experiment 111's `gamk`. The filter's settings are `e00135_arms.read_hyper`, the same for every
arm. The filter runs on the GPU when there is one. Each substation writes one file per arm,
atomically, and each substation's wall-clock time is appended to `work/gpu_hours.jsonl`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00135-*/e00135_joint_fits.py [--workers 3]
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import e00135_arms as arms
import e00135_common as common
import e00135_fits as fits_module
import torch
from studies.dp_gam import TrainingWindow
from studies.dp_joint_fit import fit_joint_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

ARM_NAMES: tuple[str, ...] = ("j1", *fits_module.J2_RIDGES)


def fit_site(*, site: str) -> str:
    """Fit every joint arm at one substation for every monthly cut from its first cut.

    Args:
        site: The substation's label.

    Returns:
        A progress line.
    """
    started = time.monotonic()
    cache = common.load_cache()
    vintages = read_vintages()
    two_stage = fits_module.load_fits(arm="gamk", site=site)
    spec = fits_module.gamk_spec()
    hyper = arms.read_hyper(site=site)
    ridges = (0.0, *fits_module.J2_RIDGES.values())
    out: dict[str, list] = {name: [] for name in ARM_NAMES}
    filter_seconds = 0.0
    for anchor in two_stage:
        began = time.monotonic()
        pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=anchor.cut)
        fitted = fit_joint_gam(
            cache=cache,
            site=site,
            spec=spec,
            cut=anchor.cut,
            window=TrainingWindow(months=fits_module.WINDOW_MONTHS),
            season_years=common.SEASON_YEARS,
            hyper=hyper,
            pinned=pinned,
            anchor=anchor,
            ridges=ridges,
        )
        for name, fit in zip(ARM_NAMES, fitted, strict=True):
            out[name].append(fit)
        filter_seconds += time.monotonic() - began
    for name, fits in out.items():
        target = fits_module.fit_path(arm=name, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_bytes(pickle.dumps(fits))
        temporary.rename(target)
    elapsed = time.monotonic() - started
    with (common.WORK_DIR / "gpu_hours.jsonl").open("a") as log:
        log.write(
            json.dumps({"site": site, "cuts": len(two_stage), "seconds": round(elapsed, 1)}) + "\n"
        )
    return f"{site}: {len(two_stage)} cuts x {len(ARM_NAMES)} arms, {elapsed:.0f} s"


def _job(site: str) -> str:
    torch.set_num_threads(2)
    return fit_site(site=site)


def main() -> None:
    """Fit the joint arms at every substation in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sites", nargs="+", default=list(common.ALL_SITES))
    args = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(_job, args.sites):
            print(line, flush=True)


if __name__ == "__main__":
    main()
