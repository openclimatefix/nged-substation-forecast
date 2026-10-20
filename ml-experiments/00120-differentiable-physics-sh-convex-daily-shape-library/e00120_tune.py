"""Tune the shape library on 2023-24: K, the deviation weight, and the orthogonality weight.

Every fit uses data before 2023-07-01 and is scored on 2023-07-01 to 2024-06-30, on rows with a
reading outside the switching masks. The replicate and scored years are never touched here.

**The grid** (plan): `lambda_dev` in {300, 1,000, 3,000, 10,000} and `gamma` in {0, 0.1, 1, 10}
(multiples of the shape's data curvature) at K = 3 and 4, three seeds each (seed 1 starts from the
singular value decomposition of the per-site profiles, seeds 2 and 3 from random Dirichlet
weights). The score of a configuration is the mean over seeds of the mean over decision sites of
the quantile score. K = 2 and 6 run at the chosen pair, exploratory, never scored later.

**The K rule.** Take the better of K = 3 and K = 4 by this score; scores within 0.023 points tie,
and a tie goes to the smaller K.

    uv run python ml-experiments/00120-*/e00120_tune.py [--workers 6]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import e00120_arms as arms
import e00120_common as common
import numpy as np
import polars as pl
from studies.dp_shape_library import InitialShapesType, LibraryConfig

K_GRID: Final[tuple[int, ...]] = (3, 4)
LAMBDA_DEV_GRID: Final[tuple[float, ...]] = (300.0, 1000.0, 3000.0, 10000.0)
GAMMA_GRID: Final[tuple[float, ...]] = (0.0, 0.1, 1.0, 10.0)
EXPLORATORY_K: Final[tuple[int, ...]] = (2, 6)
SEEDS: Final[tuple[tuple[int, InitialShapesType], ...]] = (
    (1, "svd"),
    (2, "dirichlet"),
    (3, "dirichlet"),
)
K_TIE_MARGIN: Final[float] = 0.023
"""Scores within this many points tie, and a tie goes to the smaller K."""


@dataclass(frozen=True)
class TuneJob:
    """One configuration and seed fitted at the tuning cut."""

    k: int
    lambda_dev: float
    gamma: float
    seed: int
    init_method: InitialShapesType
    exploratory: bool


def spec_of(*, job: TuneJob) -> arms.ArmSpec:
    """Return the arm of a tuning job."""
    name = f"k{job.k}_l{job.lambda_dev:g}_g{job.gamma:g}_s{job.seed}"
    return arms.ArmSpec(
        name=name,
        kind="sh",
        library=LibraryConfig(
            n_shapes=job.k,
            deviation_weight=job.lambda_dev,
            orthogonality=job.gamma,
            other_strength=common.PRODUCTION_LAMBDA,
        ),
        init_method=job.init_method,
        seed=job.seed,
        pool=common.POOL_SITES,
        tag="tuning",
    )


def scores_of(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the quantile score and the median's absolute error of the unmasked rows."""
    kept = frame.filter(~pl.col("switching_mask"))
    actual = kept["actual"].to_numpy().astype(np.float64)
    quantiles = kept.select(
        "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"
    ).to_numpy()
    return {
        "score": float(common.quantile_score(actual=actual, quantiles=quantiles).mean()),
        "mae_q50": float(np.abs(actual - quantiles[:, 5]).mean()),
    }


def run_job(job: TuneJob) -> list[dict[str, Any]]:
    """Fit one job at the tuning cut and return one row per decision site.

    Args:
        job: The configuration and seed.

    Returns:
        Rows with the job's labels, the site's scores, and the fit's convergence facts.
    """
    import torch

    torch.set_num_threads(1)
    cache, vintages = arms.load_source()
    spec = spec_of(job=job)
    result = arms.fit_and_score(
        cache=cache,
        vintages=vintages,
        cut=common.TUNING_CUT,
        spec=spec,
        score_sites=common.DECISION_SITES,
    )
    library = result.library
    assert library is not None
    rows = []
    for site in result.frame["time_series_id"].unique().to_list():
        part = result.frame.filter(pl.col("time_series_id") == site)
        rows.append(
            {
                "k": job.k,
                "lambda_dev": job.lambda_dev,
                "gamma": job.gamma,
                "seed": job.seed,
                "exploratory": job.exploratory,
                "time_series_id": site,
                "objective": library.objective_trace[-1],
                "n_sweeps": library.n_sweeps,
                "converged": library.converged,
                "pool": ",".join(result.pool),
                **scores_of(frame=part),
            }
        )
    print(
        f"{spec.name}: sweeps {library.n_sweeps}, converged {library.converged}, "
        f"score {np.mean([row['score'] for row in rows]):.3f}",
        flush=True,
    )
    return rows


def main_grid_jobs() -> list[TuneJob]:
    """Return the main grid: K = 3 and 4, 16 pairs, three seeds."""
    return [
        TuneJob(k, lam, gamma, seed, init, exploratory=False)
        for k in K_GRID
        for lam in LAMBDA_DEV_GRID
        for gamma in GAMMA_GRID
        for seed, init in SEEDS
    ]


def choose(*, table: pl.DataFrame) -> dict[str, Any]:
    """Apply the tuning rule to the main grid.

    Args:
        table: The scores of the main grid, one row per fit and site.

    Returns:
        The chosen `K`, `lambda_dev`, and `gamma`, the best pair per K, and every
        configuration's mean score.
    """
    per_fit = table.group_by("k", "lambda_dev", "gamma", "seed").agg(pl.col("score").mean())
    per_config = (
        per_fit.group_by("k", "lambda_dev", "gamma")
        .agg(pl.col("score").mean(), pl.col("score").std().alias("seed_sd"))
        .sort("score")
    )
    best_per_k = {
        k: per_config.filter(pl.col("k") == k).sort("score").row(0, named=True) for k in K_GRID
    }
    smallest = min(K_GRID)
    chosen_k = min(K_GRID, key=lambda k: best_per_k[k]["score"])
    if best_per_k[smallest]["score"] - best_per_k[chosen_k]["score"] <= K_TIE_MARGIN:
        chosen_k = smallest
    chosen = best_per_k[chosen_k]
    return {
        "k": chosen_k,
        "lambda_dev": chosen["lambda_dev"],
        "gamma": chosen["gamma"],
        "best_per_k": {str(k): v for k, v in best_per_k.items()},
        "configurations": per_config.to_dicts(),
    }


def run(*, jobs: list[TuneJob], workers: int) -> pl.DataFrame:
    """Run jobs in worker processes, one thread each, and return one row per fit and site."""
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_job, jobs))
    return pl.DataFrame([row for rows in results for row in rows])


def main() -> None:
    """Tune on the real data and write `work/tuned.json` and the score tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    table = run(jobs=main_grid_jobs(), workers=arguments.workers)
    table.write_parquet(common.WORK_DIR / "tuning_scores.parquet")
    chosen = choose(table=table)
    extra_jobs = [
        TuneJob(k, chosen["lambda_dev"], chosen["gamma"], seed, init, exploratory=True)
        for k in EXPLORATORY_K
        for seed, init in SEEDS
    ]
    extra = run(jobs=extra_jobs, workers=arguments.workers)
    extra.write_parquet(common.WORK_DIR / "tuning_scores_exploratory.parquet")
    chosen["exploratory_k"] = {
        str(k): float(extra.filter(pl.col("k") == k)["score"].to_numpy().mean())
        for k in EXPLORATORY_K
    }
    chosen["run"] = common.run_metadata()
    (common.WORK_DIR / "tuned.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print(json.dumps({k: v for k, v in chosen.items() if k != "configurations"}, indent=2))


if __name__ == "__main__":
    main()
