"""Run the known-answer stage (S1): fleets of the truth cells, each fitted by its arms.

Each job draws one fleet of a truth cell, fits every arm of the cell for the full sample and 40
joint month-block resamples, and writes one parquet file of intervals. A job whose file exists is
skipped, so a crashed run resumes.

    uv run python ml-experiments/00117-*/e00117_gate.py --tag power --sets 1 --cells smooth unlike \
        --fleets 10 --workers 2 --threads 6 --gamma-lo 3000 --gamma-hi 10000
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import e00117_common as common
import e00117_synth as synth
import polars as pl

_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class Job:
    """One fleet of one truth cell in one set."""

    tag: str
    set_id: int
    cell: str
    index: int
    gamma_lo: float
    gamma_hi: float
    threads: int
    arms: tuple[str, ...]
    device: str = "cpu"


LOAD_LIMIT = float(os.environ.get("H2_LOAD_LIMIT", "35"))
"""A job waits while the machine's one-minute load average is above this."""


def wait_for_quiet_machine() -> None:
    """Sleep while the one-minute load average is above `LOAD_LIMIT`."""
    while os.getloadavg()[0] > LOAD_LIMIT:
        time.sleep(60)


def job_path(*, job: Job) -> Path:
    """Return the file a job writes."""
    return (
        common.WORK_DIR / "gate" / job.tag / f"set{job.set_id}_{job.cell}_{job.index:03d}.parquet"
    )


def run_job(job: Job) -> str:
    """Draw a fleet, fit its arms, and write the intervals; return a one-line report."""
    import torch

    torch.set_num_threads(job.threads)
    target = job_path(job=job)
    if target.exists():
        return f"skip {target.name}"
    wait_for_quiet_machine()
    if "templates" not in _STATE:
        cache = common.load_cache()
        _STATE["templates"] = synth.build_templates(cache=cache)
        _STATE["coverage"] = synth._coverage_table(templates=_STATE["templates"])
    cell_number = list(synth.CELL_ARMS).index(job.cell)
    seed = (job.set_id, cell_number, job.index)
    kind = "smooth" if job.cell in ("corr03", "corr06") else job.cell
    started = time.perf_counter()
    fleet = synth.draw_fleet(
        templates=_STATE["templates"],
        coverage=_STATE["coverage"],
        kind=kind,
        seed=seed,
        noise_rho=synth.CELL_NOISE_RHO.get(job.cell, 0.0),
    )
    drawn = time.perf_counter() - started
    counts = synth.fleet_counts(fleet=fleet, seed=seed)
    frames = []
    timings = {}
    for arm in job.arms:
        tick = time.perf_counter()
        frame = synth.run_arm(
            fleet=fleet,
            arm=arm,
            gamma_lo=job.gamma_lo,
            gamma_hi=job.gamma_hi,
            counts=counts,
            device=job.device,
        )
        timings[arm] = time.perf_counter() - tick
        frames.append(
            frame.with_columns(
                set_id=pl.lit(job.set_id),
                cell=pl.lit(job.cell),
                fleet=pl.lit(job.index),
                noise_sd=pl.lit(sum(fleet.noise_sd.values()) / len(fleet.noise_sd)),
            )
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    pl.concat(frames).write_parquet(temporary)
    temporary.replace(target)
    seconds = ", ".join(f"{arm} {t:.0f}s" for arm, t in timings.items())
    return f"done {target.name}: draw {drawn:.0f}s, {seconds}"


def main() -> None:
    """Run the requested jobs in worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--sets", type=int, nargs="+", default=[1])
    parser.add_argument("--cells", nargs="+", default=["smooth", "unlike"])
    parser.add_argument("--fleets", type=int, default=10)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--gamma-lo", type=float, required=True)
    parser.add_argument("--gamma-hi", type=float, required=True)
    parser.add_argument("--arms", nargs="*", default=None, help="overrides the cell's arms")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-arms", nargs="*", default=[], help="arms to leave out")
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(arguments.threads)
    common.load_cache()
    jobs = [
        Job(
            tag=arguments.tag,
            set_id=set_id,
            cell=cell,
            index=index,
            gamma_lo=arguments.gamma_lo,
            gamma_hi=arguments.gamma_hi,
            threads=arguments.threads,
            device=arguments.device,
            arms=tuple(
                arm
                for arm in (arguments.arms or synth.CELL_ARMS[cell])
                if arm not in arguments.skip_arms
            ),
        )
        for set_id in arguments.sets
        for cell in arguments.cells
        for index in range(arguments.first, arguments.first + arguments.fleets)
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(run_job, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
