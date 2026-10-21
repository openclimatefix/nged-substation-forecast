"""Forecast the tuning months, the replicate months, and the scored year with experiment 135's arms.

For each arm and substation, writes one parquet file per period under `work/preds/`. The filter's
settings are `e00135_arms.read_hyper`, the spread is the arm's own (`e00135_tune.py`), and the
weather calibration and solar shrink are experiment 111's. The tuning-month rows are in-sample for
the spread and must not be quoted as results.

    uv run --no-sync python ml-experiments/00135-*/e00135_run.py [--workers 4] [--arms ts j1]
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00135_arms as arms
import e00135_common as common
import e00135_ens as ens
import torch
from studies.dp_assimilation import DEFAULT_DELAY_HOURS

PERIODS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
OUT_DIR: Final[Path] = common.WORK_DIR / "preds"


def out_path(*, arm: str, period: str, site: str) -> Path:
    """Return the file that holds an arm's forecasts for one period at one substation."""
    return OUT_DIR / f"{arm}_{period}_{site}.parquet"


def forecast_site(job: tuple[str, str]) -> str:
    """Forecast every period of one arm at one substation and write the files.

    Args:
        job: The arm's name and the substation's label.

    Returns:
        A progress line.
    """
    name, site = job
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    arm = arms.ARMS[name]
    tuned = arms.read_tuned()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=arms.load_calibration(site=site).with_solar_shrink(
            solar_shrink=tuned["solar_shrink"]
        )
    )
    fits = arms.load_arm_fits(arm=arm, site=site)
    trace = arms.build_trace(
        cache=cache, setup=setup, arm=arm, fits=fits, hyper=arms.read_hyper(site=site)
    )
    spread = arms.spread_with_factors(arm=name, site=site)
    rows = 0
    for period in PERIODS:
        first_usable_origin = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
        runs = [
            index
            for index in setup.indices(periods={period})
            if setup.origins[index] >= first_usable_origin
        ]
        if not runs:
            continue
        frame = ens.forecast_frame(
            cache=cache,
            setup=setup,
            fits=fits,
            spread=spread,
            trace=trace,
            run_indices=runs,
            arm=name,
        )
        target = out_path(arm=name, period=period, site=site)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        frame.write_parquet(temporary)
        temporary.rename(target)
        rows += frame.height
    return f"{name} {site}: {rows:,} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Forecast every arm and substation in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--arms", nargs="+", default=list(arms.ARMS))
    args = parser.parse_args()
    jobs = [(name, site) for name in args.arms for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(forecast_site, jobs):
            print(line, flush=True)


if __name__ == "__main__":
    main()
