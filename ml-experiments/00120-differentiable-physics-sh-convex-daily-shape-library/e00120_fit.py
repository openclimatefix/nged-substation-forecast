"""Fit every arm of experiment 120 at the replicate and scored cuts and store the predictions.

**Run only after the known-answer gate and the Opus review of the module.** Each (cut, arm, seed)
is one frozen fit, scored on the 12 months that follow. The tuned settings come from
`work/tuned.json` (`e00120_tune.py`, 2023-24 only).

**Arms.**

- Seedless: `g` (per-site convex), `n` (partial pooling at 3,000 rows), `n_full` (complete pooling).
- Library, five seeds each (seed 1 from the singular value decomposition, seeds 2 to 5 random
  Dirichlet): `sh` (the decision arm), `sh_k1` (K = 1), `sh_g0` (no orthogonality penalty),
  `sh_nodev` (a very large deviation weight), and the exploratory `sh_wd` (weights per day type)
  and `sh_all` (all years).

Each seed's frame is stored in `work/seed_frames/`, and `combine` writes the prediction set
`<cut>_<arm>` (all seeds in one set) once every seed has finished.

    uv run python ml-experiments/00120-*/e00120_fit.py run --cuts replicate scored [--workers 6]
    uv run python ml-experiments/00120-*/e00120_fit.py combine --cuts replicate scored
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00120_arms as arms
import e00120_common as common
import polars as pl
from studies.dp_shape_library import InitialShapesType, LibraryConfig

SEED_FRAME_DIR: Final[Path] = common.WORK_DIR / "seed_frames"
STARTS: Final[tuple[tuple[int, InitialShapesType], ...]] = (
    (1, "svd"),
    (2, "dirichlet"),
    (3, "dirichlet"),
    (4, "dirichlet"),
    (5, "dirichlet"),
)
NO_DEVIATION_WEIGHT: Final[float] = 1e8
"""A deviation weight so large that the deviation is zero: a pure library."""
LIBRARY_ARMS: Final[tuple[str, ...]] = ("sh", "sh_k1", "sh_g0", "sh_nodev", "sh_wd", "sh_all")
REFERENCE_ARMS: Final[tuple[str, ...]] = ("g", "n", "n_full")


@dataclass(frozen=True)
class FitJob:
    """One arm at one cut with one seed."""

    cut_name: str
    arm: str
    seed: int
    init_method: InitialShapesType


def spec_of(*, job: FitJob, tuned: dict[str, Any]) -> arms.ArmSpec:
    """Return the arm of a job from the tuned settings."""
    if job.arm in REFERENCE_ARMS:
        return arms.ArmSpec(name=job.arm, kind=job.arm)  # ty: ignore[invalid-argument-type]
    library = LibraryConfig(
        n_shapes=tuned["k"],
        deviation_weight=tuned["lambda_dev"],
        orthogonality=tuned["gamma"],
        other_strength=common.PRODUCTION_LAMBDA,
    )
    match job.arm:
        case "sh_k1":
            library = LibraryConfig(
                n_shapes=1,
                deviation_weight=tuned["lambda_dev"],
                other_strength=common.PRODUCTION_LAMBDA,
            )
        case "sh_g0":
            library = LibraryConfig(
                n_shapes=tuned["k"],
                deviation_weight=tuned["lambda_dev"],
                other_strength=common.PRODUCTION_LAMBDA,
            )
        case "sh_nodev":
            library = LibraryConfig(
                n_shapes=tuned["k"],
                deviation_weight=NO_DEVIATION_WEIGHT,
                orthogonality=tuned["gamma"],
                other_strength=common.PRODUCTION_LAMBDA,
            )
        case "sh_wd":
            library = LibraryConfig(
                n_shapes=tuned["k"],
                deviation_weight=tuned["lambda_dev"],
                orthogonality=tuned["gamma"],
                other_strength=common.PRODUCTION_LAMBDA,
                weights_per_day_type=True,
            )
    return arms.ArmSpec(
        name=job.arm,
        kind="sh",
        library=library,
        init_method=job.init_method,
        seed=job.seed,
        all_years=job.arm == "sh_all",
    )


def frame_path(*, job: FitJob) -> Path:
    """Return where one seed's predictions are stored."""
    return SEED_FRAME_DIR / f"{job.cut_name}_{job.arm}_s{job.seed}.parquet"


def run_job(job: FitJob) -> str:
    """Fit one job, store its frame and readouts, and skip a stored job."""
    import torch

    torch.set_num_threads(1)
    path = frame_path(job=job)
    if path.exists():
        return f"{path.name}: stored"
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    cache, vintages = arms.load_source()
    spec = spec_of(job=job, tuned=tuned)
    result = arms.fit_and_score(
        cache=cache, vintages=vintages, cut=common.CUTS[job.cut_name], spec=spec
    )
    if result.library is not None:
        arms.save_library_fit(
            name=f"{job.cut_name}_{job.arm}_s{job.seed}",
            library=result.library,
            extra={
                "cut": job.cut_name,
                "arm": job.arm,
                "seed": job.seed,
                "pool": result.pool,
                "capacity_sources": result.capacity_sources,
            },
        )
    SEED_FRAME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    result.frame.write_parquet(temporary)
    temporary.replace(path)
    return f"{path.name}: {len(result.frame)} rows, pool {result.pool}"


def jobs_for(*, cuts: list[str]) -> list[FitJob]:
    """Return every (cut, arm, seed) job."""
    jobs = []
    for cut_name in cuts:
        # The tuning cut needs the decision arm only: its fits give the disjoint-window stability
        # reading and the weights by winter.
        arms_here = ("sh",) if cut_name == "tuning" else LIBRARY_ARMS
        if cut_name != "tuning":
            jobs += [FitJob(cut_name, arm, 0, "svd") for arm in REFERENCE_ARMS]
        jobs += [FitJob(cut_name, arm, seed, init) for arm in arms_here for seed, init in STARTS]
    return jobs


def combine(*, cuts: list[str]) -> None:
    """Write one prediction set per (cut, arm), joining the seeds."""
    for cut_name in cuts:
        for arm in (*REFERENCE_ARMS, *LIBRARY_ARMS):
            files = sorted(SEED_FRAME_DIR.glob(f"{cut_name}_{arm}_s*.parquet"))
            expected = 1 if arm in REFERENCE_ARMS else len(STARTS)
            if len(files) != expected:
                print(f"{cut_name}_{arm}: {len(files)} of {expected} seeds, skipped")
                continue
            common.store_arm(
                name=f"{cut_name}_{arm}", frame=pl.concat([pl.read_parquet(f) for f in files])
            )
            print(f"{cut_name}_{arm}: stored")


def main() -> None:
    """Run the fits in worker processes, or combine the seeds into prediction sets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "combine"))
    parser.add_argument("--cuts", nargs="+", default=["replicate", "scored"])
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    if arguments.command == "combine":
        combine(cuts=arguments.cuts)
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    (common.WORK_DIR / f"fit_run_{'_'.join(arguments.cuts)}.json").write_text(
        json.dumps(common.run_metadata(), indent=2) + "\n"
    )
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(run_job, jobs_for(cuts=arguments.cuts)):
            print(message, flush=True)


if __name__ == "__main__":
    main()
