"""Cache each run's member forecasts of GAM-K, so each arm of experiment 115 adds a correction.

**The correction is the same for every ensemble member**, so the DP forecast of each member never
changes between arms. This script runs `studies.dp_ens.member_means` once per run and substation,
with experiment 111's stored `gamk` fits, calibration, and solar shrink, and stores the means in
points of effective capacity. An arm then adds its correction to the cached means and builds the
mixture quantiles, which takes seconds instead of minutes.

Runs are every run from the first run a fit can serve: the tuning, replicate, scored, and stride
runs, and the warm-up runs before them, which only the 14-day profile of the recipe reads.

    uv run python ml-experiments/00115-*/e00162_s_g_members.py [--workers 6]
"""

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Final

import e00162_s_g_common as common
import e00162_s_g_ens as ens
import numpy as np
import torch  # ty: ignore[unresolved-import]
from studies.dp_assimilation import DEFAULT_DELAY_HOURS
from studies.dp_ens import member_means
from studies.dp_ens_weather import Calibration
from studies.dp_gam import GamFit

PERIODS: Final[frozenset[str]] = frozenset({"tuning", "replicate", "scored", "stride", "warm_up"})
MEMBERS_DIR: Final[Path] = common.SRC_115_WORK_DIR / "members"
GAMK_FITS_DIR: Final[Path] = common.FITS_DIR / "gamk"


@dataclass(frozen=True)
class RunMembers:
    """The cached member forecasts of one run.

    Attributes:
        index: The run's index in `common.run_dates()`.
        start: The grid position of the first target.
        means: The members' mean forecasts in points, shape (members, targets).
    """

    index: int
    start: int
    means: np.ndarray


def members_path(*, site: str) -> Path:
    """Return the file that holds a site's cached member forecasts."""
    return MEMBERS_DIR / f"{site}.pkl"


def load_fits(*, site: str) -> list[GamFit]:
    """Return experiment 111's stored monthly `gamk` fits at a site, ascending by cut."""
    return pickle.loads((GAMK_FITS_DIR / f"{site}.pkl").read_bytes())


def load_calibration(*, site: str) -> Calibration:
    """Return experiment 111's stored calibration at a site, with its tuned solar shrink."""
    with np.load(common.SRC_WORK_DIR / "calibration" / f"{site}.npz") as stored:
        calibration = Calibration(
            temperature_shift_c=stored["shift"],
            irradiance_ratio=stored["ratio"],
            solar_shrink=stored["shrink"],
        )
    shrink = json.loads((common.SRC_WORK_DIR / "tuned.json").read_text())["solar_shrink"]
    return calibration.with_solar_shrink(solar_shrink=shrink)


def build_site(site: str) -> str:
    """Cache one site's member forecasts and write them atomically.

    Args:
        site: The substation's label.

    Returns:
        A progress line.
    """
    torch.set_num_threads(1)
    started = time.monotonic()
    cache = common.load_cache()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=load_calibration(site=site)
    )
    fits = load_fits(site=site)
    first_usable_origin = fits[0].cut + timedelta(hours=DEFAULT_DELAY_HOURS)
    stored = []
    for index in setup.indices(periods=set(PERIODS)):
        run = setup.runs[index]
        origin = setup.origins[index]
        if origin < first_usable_origin:
            continue
        if len(cache.end_times) - run.origin_position - 1 < ens.MIN_WINDOW_HALF_HOURS:
            continue
        if index < common.PROXY_WARM_UP_RUNS:
            continue
        result = member_means(cache=cache, context=setup.context, run=run, origin=origin, fits=fits)
        stored.append((index, result.window.start, result.means))
    target = members_path(site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(pickle.dumps(stored))
    temporary.rename(target)
    return f"{site}: {len(stored)} runs, {time.monotonic() - started:.0f} s"


def load_members(*, site: str) -> dict[int, RunMembers]:
    """Return a site's cached member forecasts keyed by run index."""
    stored = pickle.loads(members_path(site=site).read_bytes())
    return {
        index: RunMembers(index=index, start=start, means=means) for index, start, means in stored
    }


def main() -> None:
    """Cache every scored site's member forecasts in parallel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(build_site, common.SCORED_SITES):
            print(line, flush=True)


if __name__ == "__main__":
    main()
