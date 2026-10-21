"""Design check of experiment 123: coverage against the bootstrap block length.

How much does the placebo and matched-shape coverage move with a block of 7, 14, or 28 days?

Runs on the design winters 2023-24 and 2024-25 only, with 100 draws per cell, after the gate. The
plan fixes 14 days; this check is descriptive and changes nothing.
"""

import multiprocessing
from typing import Final

import e00123_common as common
import e00123_estimate as estimate
import e00123_gate as gate
import numpy as np
import polars as pl

BLOCK_LENGTHS: Final[tuple[int, ...]] = (7, 14, 28)
DRAWS: Final[int] = 100
CELLS: Final[tuple[tuple[str, float], ...]] = (("placebo", 0.0), ("matched", 4.0))


def run_series(job: gate.SeriesJob) -> pl.DataFrame:
    """Return coverage rows for one series at each block length."""
    common.cap_threads()
    table = pl.read_parquet(common.WORK_DIR / "night_tables.parquet")
    rows = []
    for block_days in BLOCK_LENGTHS:
        base = gate.residual_nights(
            nights=estimate.load_nights(
                table=table,
                site=job.site,
                capacity_mw=job.capacity_mw,
                winters=common.DESIGN_WINTERS,
                summer_years=gate.SUMMER_YEARS_FOR_GATE,
                block_days=block_days,
            )
        )
        rng = np.random.default_rng(job.seed + block_days)
        for _ in range(DRAWS):
            noise = gate.simulated_dataset(base=base, rng=rng)
            for shape, step in CELLS:
                sim = noise.with_arrays(
                    d=noise.d + gate.injection(shape=shape, step=step, nights=noise)
                )
                fit = estimate.fit_series(nights=sim, rng=rng)
                truth = gate.truth(shape=shape, step=step)
                rows.append(
                    {
                        "site": job.site,
                        "block_days": block_days,
                        "shape": shape,
                        "covered": fit.step.low <= truth <= fit.step.high,
                        "width": fit.step.high - fit.step.low,
                    }
                )
    return pl.DataFrame(rows)


def main() -> None:
    """Run the check and print coverage by block length."""
    infos = common.series_infos()
    jobs = [
        gate.SeriesJob(site=s, capacity_mw=infos[s].effective_capacity_mw, seed=5000 + i)
        for i, s in enumerate(common.FITTED_SERIES)
    ]
    with multiprocessing.get_context("spawn").Pool(gate.WORKERS) as workers:
        frames = workers.map(run_series, jobs, chunksize=1)
    frame = pl.concat(frames)
    frame.write_parquet(common.WORK_DIR / "block_length_draws.parquet")
    summary = (
        frame.group_by("shape", "block_days")
        .agg(coverage=pl.col("covered").mean(), mean_width=pl.col("width").mean())
        .sort("shape", "block_days")
    )
    with pl.Config(tbl_rows=-1):
        print(summary)
    summary.write_json(common.WORK_DIR / "block_length.json")


if __name__ == "__main__":
    main()
