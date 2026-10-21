"""Tune the deviation penalty `lambda` on 2023-24 and the per-site comparator's short-history ridge.

Every fit uses data before 2023-07-01 and is scored on 2023-07-01 to 2024-06-30, on rows with a
reading outside the switching masks. The scored year (2025-07 onward) and the replicate year are
never touched here.

**The rule.** For each `lambda` in `common.LAMBDA_GRID` (rows of evidence), the partially pooled
model is fitted three ways: with every site on its full window (all rows), and with each decision
site in turn holding only its last 4 weeks or its last 3 months. `lambda*` minimises the mean of
the 4-week and 3-month ratios, subject to the all-rows score being within +0.02 points of the
per-site model's. A ratio is the quantile score at `lambda` over the score at `lambda = 0` with the
base ridge on the same rows, averaged over the decision sites. Each score's own best `lambda`, and
the `lambda` tuned on the median's mean absolute error, are reported; disagreement signals
misspecification.

The per-site comparator on a short history gets its own ridge from `common.G_SHORT_RIDGE_GRID`,
chosen by the quantile score on the same rows.

    uv run python ml-experiments/00150-*/e00150_tune.py [--workers 6]
"""

import argparse
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import e00150_arms as arms
import e00150_common as common
import numpy as np
import polars as pl
from studies.dp_gam_pooled import PoolingStrength

ALL_ROWS_MARGIN: Final[float] = 0.02
"""N's all-rows score may exceed the per-site model's by at most this many points at `lambda*`."""
HISTORIES: Final[tuple[common.History, ...]] = (common.FOUR_WEEKS, common.THREE_MONTHS)


@dataclass(frozen=True)
class TuneJob:
    """One cell of the tuning grid: an arm fitted at the tuning cut for every decision site."""

    source: arms.Source
    config: arms.ArmConfig
    cut_name: str
    targets: tuple[str, ...]
    decision: tuple[str, ...]
    kind: str
    lam: float
    ridge: float
    truth_path: Path | None = None
    """A saved noise-free series per synthetic site (`truth_<label>`), or `None` on real data."""


def truth_rmse(*, frame: pl.DataFrame, truth: np.ndarray, index: np.ndarray) -> float:
    """Return the root mean square error of the mean forecast against the noise-free truth.

    Args:
        frame: One site's predictions, one row per grid position in `index`.
        truth: The site's noise-free series over the whole grid.
        index: The grid positions the frame covers.

    Returns:
        The error in points, on rows outside the switching mask.
    """
    keep = ~frame["switching_mask"].to_numpy()
    error = frame["mean"].to_numpy().astype(np.float64) - truth[index]
    return float(np.sqrt(np.nanmean(error[keep] ** 2)))


def scores_of(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the quantile score and the median's absolute error of the unmasked rows.

    Args:
        frame: One site's predictions.

    Returns:
        The mean quantile score and the mean absolute error of `q50`, in points, and the row count.
    """
    kept = frame.filter(~pl.col("switching_mask"))
    actual = kept["actual"].to_numpy().astype(np.float64)
    quantiles = kept.select(
        "q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"
    ).to_numpy()
    return {
        "score": float(common.quantile_score(actual=actual, quantiles=quantiles).mean()),
        "mae_q50": float(np.abs(actual - quantiles[:, 5]).mean()),
        "n_rows": float(len(kept)),
    }


def run_job(job: TuneJob) -> list[dict[str, Any]]:
    """Fit one cell for every target and return one row per site.

    Args:
        job: The cell.

    Returns:
        Rows with the cell's labels and the site's scores.
    """
    import torch

    torch.set_num_threads(1)
    cache, vintages = arms.load_source(source=job.source)
    cut = common.CUTS[job.cut_name]
    truths = np.load(job.truth_path) if job.truth_path is not None else None
    rows = []
    for site in job.targets:
        frame, _ = arms.fit_and_score_site(
            cache=cache,
            vintages=vintages,
            cut=cut,
            config=job.config,
            target=site,
            decision_pool=job.decision,
        )
        rows.append(
            {
                "kind": job.kind,
                "history": job.config.history.label if job.config.history else "all",
                "lam": job.lam,
                "ridge": job.ridge,
                "site": site,
                **scores_of(frame=frame),
            }
        )
        if truths is not None:
            index = common.scoring_index(
                cache=cache, site=site, after=cut, through=common.year_end(cut=cut)
            )
            rows[-1]["truth_rmse"] = truth_rmse(frame=frame, truth=truths[site], index=index)
        print(
            f"{job.kind} {rows[-1]['history']} lam={job.lam:g} {site}: {rows[-1]['score']:.3f}",
            flush=True,
        )
    return rows


def grid_jobs(
    *,
    source: arms.Source,
    cut_name: str,
    pool: tuple[str, ...],
    decision: tuple[str, ...],
    tag: str,
    lambdas: Sequence[float] = common.LAMBDA_GRID,
    truth_path: Path | None = None,
) -> list[TuneJob]:
    """Return every cell of the tuning grid, and the per-site comparator's ridge cells.

    Args:
        source: Where workers read their data.
        cut_name: The tuning cut's name in `common.CUTS`.
        pool: The substations of the joint fit.
        decision: The substations scored and cross-fitted.
        tag: Names the data, so stored replicas never mix.
        lambdas: The deviation penalty weights.
        truth_path: A saved noise-free series per synthetic site, to score against the truth.

    Returns:
        The jobs.
    """
    jobs = []
    for lam in lambdas:
        for history in (None, *HISTORIES):
            label = history.label if history else "all"
            config = arms.ArmConfig(
                name=f"n_{label}_{lam:g}",
                strength=PoolingStrength.uniform(value=lam),
                history=history,
                tag=tag,
                pool=pool,
            )
            jobs.append(
                TuneJob(source, config, cut_name, decision, decision, "n", lam, 0.1, truth_path)
            )
    for ridge in common.G_SHORT_RIDGE_GRID:
        for history in HISTORIES:
            config = arms.ArmConfig(
                name=f"g_{history.label}_{ridge:g}",
                history=history,
                alone=True,
                short_ridge=ridge,
                tag=tag,
                pool=pool,
            )
            jobs.append(
                TuneJob(source, config, cut_name, decision, decision, "g", 0.0, ridge, truth_path)
            )
    return jobs


def run_grid(*, jobs: Sequence[TuneJob], workers: int) -> pl.DataFrame:
    """Run the jobs in worker processes and return one row per cell and site.

    Args:
        jobs: The cells.
        workers: The number of worker processes, each with one thread.

    Returns:
        The scores.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_job, jobs))
    return pl.DataFrame([row for rows in results for row in rows])


def choose(*, table: pl.DataFrame) -> dict[str, Any]:
    """Apply the tuning rule to a grid of scores.

    Args:
        table: The output of `run_grid`.

    Returns:
        `lambda*`, each score's own best `lambda`, the `lambda` tuned on the median's absolute
        error, the chosen ridge of the per-site comparator on each short history, and the grid.

    Raises:
        ValueError: If no `lambda` keeps the all-rows score within the margin of `lambda = 0`.
    """
    n = table.filter(pl.col("kind") == "n")
    lambdas = sorted(n["lam"].unique().to_list())

    def per_site(*, history: str, lam: float, column: str) -> dict[str, float]:
        part = n.filter((pl.col("history") == history) & (pl.col("lam") == lam))
        return dict(zip(part["site"].to_list(), part[column].to_list(), strict=True))

    def ratio(*, history: str, lam: float, column: str) -> float:
        base = per_site(history=history, lam=0.0, column=column)
        here = per_site(history=history, lam=lam, column=column)
        return float(np.mean([here[site] / base[site] for site in base]))

    def mean_of(*, history: str, lam: float, column: str) -> float:
        return float(np.mean(list(per_site(history=history, lam=lam, column=column).values())))

    summary = [
        {
            "lam": lam,
            "all_delta": mean_of(history="all", lam=lam, column="score")
            - mean_of(history="all", lam=0.0, column="score"),
            "ratio_4w": ratio(history="4w", lam=lam, column="score"),
            "ratio_3m": ratio(history="3m", lam=lam, column="score"),
            "mae_ratio_4w": ratio(history="4w", lam=lam, column="mae_q50"),
            "mae_ratio_3m": ratio(history="3m", lam=lam, column="mae_q50"),
            "mean_all": mean_of(history="all", lam=lam, column="score"),
            "mean_4w": mean_of(history="4w", lam=lam, column="score"),
            "mean_3m": mean_of(history="3m", lam=lam, column="score"),
        }
        for lam in lambdas
    ]
    feasible = [row for row in summary if row["all_delta"] <= ALL_ROWS_MARGIN]
    if not feasible:
        msg = "no lambda keeps the all-rows score within the margin of the per-site model"
        raise ValueError(msg)
    best = min(feasible, key=lambda row: (row["ratio_4w"] + row["ratio_3m"]) / 2.0)
    best_mae = min(summary, key=lambda row: (row["mae_ratio_4w"] + row["mae_ratio_3m"]) / 2.0)
    truth_best = None
    if "truth_rmse" in n.columns:
        truth_summary = [
            {
                "lam": lam,
                "all_delta": row["all_delta"],
                "ratio_4w": ratio(history="4w", lam=lam, column="truth_rmse"),
                "ratio_3m": ratio(history="3m", lam=lam, column="truth_rmse"),
            }
            for lam, row in zip(lambdas, summary, strict=True)
        ]
        truth_feasible = [row for row in truth_summary if row["all_delta"] <= ALL_ROWS_MARGIN]
        truth_best = min(truth_feasible, key=lambda r: (r["ratio_4w"] + r["ratio_3m"]) / 2.0)["lam"]
    g = table.filter(pl.col("kind") == "g")
    g_ridge = {}
    for history in ("4w", "3m"):
        part = g.filter(pl.col("history") == history)
        means = part.group_by("ridge").agg(pl.col("score").mean()).sort("ridge")
        g_ridge[history] = float(means.sort("score")["ridge"][0])
    return {
        "lambda_star": best["lam"],
        "lambda_star_mae": best_mae["lam"],
        "lambda_truth_best": truth_best,
        "own_best": {
            "all": min(summary, key=lambda row: row["mean_all"])["lam"],
            "4w": min(summary, key=lambda row: row["mean_4w"])["lam"],
            "3m": min(summary, key=lambda row: row["mean_3m"])["lam"],
        },
        "g_short_ridge": g_ridge,
        "summary": summary,
    }


def main() -> None:
    """Tune on the real data and write `work/tuned.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    jobs = grid_jobs(
        source=arms.Source(),
        cut_name="tuning",
        pool=common.POOL_SITES,
        decision=common.DECISION_SITES,
        tag="real",
    )
    table = run_grid(jobs=jobs, workers=arguments.workers)
    table.write_parquet(common.WORK_DIR / "tuning_scores.parquet")
    chosen = choose(table=table)
    chosen["run"] = common.run_metadata()
    (common.WORK_DIR / "tuned.json").write_text(json.dumps(chosen, indent=2) + "\n")
    print(json.dumps({k: v for k, v in chosen.items() if k != "summary"}, indent=2))
    print("\nlam   all_delta  ratio_4w  ratio_3m")
    for row in chosen["summary"]:
        columns = (row["all_delta"], row["ratio_4w"], row["ratio_3m"])
        print(f"{row['lam']:>9g} " + " ".join(f"{value:9.3f}" for value in columns))


if __name__ == "__main__":
    main()
