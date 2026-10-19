"""Fit the capacity-identification arms and their month-block bootstrap ranges.

For each K arm, decision substation, and cut, the script fits the arm once and under 20
month-block weightings (one batched Gram matrix, so the cost is one fit) and records the fitted
capacity of the last trained season-year, the 2nd and 19th of the 20 ordered bootstrap values, and
the vintage's mean capacity over that season-year. `e00110_report.py` reads the table and applies
the pre-registered capacity rule.

    uv run python ml-experiments/00110-*/e00110_capacity.py [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import e00110_arms as arms
import e00110_common as common
import polars as pl
from studies.dp_pinned_capacity import read_vintages

K_ARMS = ("k_base", "k_i", "k_ii", "k_iii", "k_iiiii")
_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache, vintages, and tuned settings once per worker process, with two threads."""
    import torch

    torch.set_num_threads(2)
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()
    _STATE["definitions"] = arms.arm_definitions(
        part_4=arms.read_part_4_tuned(), tuned=arms.read_tuned()
    )


def _job(job: tuple[str, str, str]) -> dict[str, Any]:
    """Return one arm's capacity result at one site and cut as a table row."""
    label, arm, site = job
    result = arms.capacity_with_bootstrap(
        cache=_STATE["cache"],
        vintages=_STATE["vintages"],
        site=site,
        cut=common.CUTS[label],
        definition=_STATE["definitions"][arm],
    )
    low, high = result.range_90
    return {
        "cut": label,
        "arm": arm,
        "site": site,
        "season_year": result.season_year,
        "capacity": result.capacity,
        "range_low": low,
        "range_high": high,
        "bootstrap": result.bootstrap.tolist(),
        "vintage_mean": result.vintage_mean,
        "vintage_source": result.vintage_source,
    }


def main() -> None:
    """Fit every K arm at every decision site and cut and write `work/capacity.parquet`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    jobs = [
        (label, arm, site)
        for label in common.CUTS
        for arm in K_ARMS
        for site in common.DECISION_SITES
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        rows = list(pool.map(_job, jobs))
    frame = pl.DataFrame(rows)
    frame.write_parquet(common.WORK_DIR / "capacity.parquet")
    print(frame.drop("bootstrap"))


if __name__ == "__main__":
    main()
