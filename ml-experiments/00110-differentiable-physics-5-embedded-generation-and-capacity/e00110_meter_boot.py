"""Bootstrap the learned meter fit's power factor and `q0` at each MVA substation.

For each MVA substation and cut, the script refits the learned meter fit (LQ) under 20 month-block
weightings of the training rows and records the power factor, `q0`, and capacity of each. The 2nd
and 19th of the 20 ordered power factors are the 90% range. A substation's power factor is
`recovered` when that range is narrower than 0.95 to 0.99, the width of the prior's 95% band, and
`not recovered` otherwise.

    uv run python ml-experiments/00110-*/e00110_meter_boot.py [--workers 4]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import e00110_arms as arms
import e00110_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import month_block_weights, prepare_fit
from studies.dp_meter import fit_gam_through_meter

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and arm definitions once per worker process, with two threads."""
    import torch

    torch.set_num_threads(2)
    _STATE["cache"] = common.load_cache()
    _STATE["definitions"] = arms.arm_definitions(part_4=arms.read_part_4_tuned(), tuned=None)


def _job(job: tuple[str, str]) -> list[dict[str, Any]]:
    """Return one substation's bootstrap rows at one cut."""
    label, site = job
    cache: FeatureCache = _STATE["cache"]
    definition: arms.ArmDefinition = _STATE["definitions"]["lq"]
    cut = common.CUTS[label]
    try:
        _, index = prepare_fit(
            cache=cache,
            site=site,
            spec=definition.spec,
            cut=cut,
            window=definition.window,
            season_years=common.SEASON_YEARS,
        )
    except ValueError:
        return []
    weights = month_block_weights(
        cache=cache,
        index=index,
        n_resamples=common.N_BOOTSTRAP_WEIGHTINGS,
        seed=common.BOOTSTRAP_SEED,
    )
    rows = []
    for draw in range(common.N_BOOTSTRAP_WEIGHTINGS):
        meter = fit_gam_through_meter(
            cache=cache,
            site=site,
            spec=definition.spec,
            cut=cut,
            window=definition.window,
            season_years=common.SEASON_YEARS,
            qmode="learned",
            weights=weights[draw],
        )
        rows.append(
            {
                "cut": label,
                "site": site,
                "draw": draw,
                "power_factor": meter.power_factor,
                "q0": meter.q0,
                "capacity": meter.fit.capacity_points()[meter.fit.last_season_year],
            }
        )
    return rows


def summarise(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return each site's and cut's 90% range of the power factor and whether it is recovered."""
    rows = []
    low_band, high_band = common.POWER_FACTOR_PRIOR_BAND
    for (label, site), group in table.group_by(["cut", "site"], maintain_order=True):
        factor = np.sort(group["power_factor"].to_numpy())
        q0 = np.sort(group["q0"].to_numpy())
        rows.append(
            {
                "cut": label,
                "site": site,
                "power_factor_low": float(factor[1]),
                "power_factor_median": float(np.median(factor)),
                "power_factor_high": float(factor[-2]),
                "q0_low": float(q0[1]),
                "q0_high": float(q0[-2]),
                "recovered": bool(factor[-2] - factor[1] < high_band - low_band),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    """Run the bootstrap and write `work/meter_boot.parquet` and the summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    jobs = [(label, site) for label in common.CAPACITY_CUTS for site in common.MVA_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        rows = [row for result in pool.map(_job, jobs) for row in result]
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "meter_boot.parquet")
    summary = summarise(table=table)
    summary.write_parquet(common.WORK_DIR / "meter_boot_summary.parquet")
    print(summary)


if __name__ == "__main__":
    main()
