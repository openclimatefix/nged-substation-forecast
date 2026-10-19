"""Tune experiment 106's convex-model settings on 2023-07 to 2024-06, then fit the filters.

Every fit uses data before 2023-07-01 only and is scored on 2023-07-01 to 2024-06-30, by the
quantile score on rows outside the switching masks, at the five decision sites. D9 has no history at
that cut. **Every candidate is scored with one spread: the centred residual quantiles of arm A1's
default replica (a fit at 2022-07-01)**, so a candidate is not credited for a spread of its own.

The settings are chosen in order, each at the settings chosen before it:

1. the drift step, on D1, from 30 to 30,000;
2. the step penalty, on D1 plus steps (the paired arm A1s), from 30 to 10,000;
3. the final drift knot: free, shrunk by 0.25, 0.5, or 0.75, or tied; a variant other than `free`
   is adopted only if it beats `free` by at least 0.05 points;
4. the best window W: 12 months or 2 years, each with and without D1.

The filters' hyperparameters are then fitted by marginal likelihood on the out-of-sample residuals
of the 2023-07-01 fit of each base arm (A0, A1, A1s, and W), under a prior shared across sites. D9
takes the mean of the five sites' unconstrained values.

    uv run python ml-experiments/00106-*/e00106_tune.py [--workers 5]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Final

import e00106_common as common
import e00106_g as g
import numpy as np
from studies.dp_assimilation import fit_filter_hyperparameters
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, ResidualSpread, TrainingWindow, replica_residuals
from studies.dp_pinned_capacity import read_vintages

DRIFT_STEP_GRID: Final[tuple[float, ...]] = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
"""Seven values. The first run of the planned grid (300 to 30,000) chose 300, the grid's edge, so
30 and 100 were added before any replicate-year or scored-year fit."""
STEP_PENALTY_GRID: Final[tuple[float, ...]] = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
"""Six values. The planned grid (30 to 1,000) chose 1,000, the grid's edge, so 3,000 and 10,000
were added before any replicate-year or scored-year fit."""
SHRINK_GRID: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)
KNOT_ADOPTION_MARGIN: Final[float] = 0.05
"""A final-knot variant other than `free` must beat `free` by this many points to be adopted."""
WINDOWS: Final[tuple[tuple[int | None, bool], ...]] = (
    (12, False),
    (12, True),
    (24, False),
    (24, True),
)
"""The window candidates for W: months of history and whether the arm keeps D1."""
BASE_ARMS: Final[tuple[str, ...]] = ("a0", "a1", "a1s", "w")
"""The arms whose filters get their own hyperparameters."""

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    import torch

    torch.set_num_threads(1)
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def _reference_spread(*, site: str) -> ResidualSpread:
    """Return the centred spread of A1's default replica at the tuning cut for a site."""
    cache: FeatureCache = _STATE["cache"]
    spec = GamSpec(
        drift="d1", ridge=g.DEFAULT_RIDGE, drift_step=g.DEFAULT_DRIFT_STEP, capacity="pinned"
    )
    replica = replica_residuals(
        cache=cache,
        site=site,
        spec=spec,
        cut=common.TUNING_CUT,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
        capacity_vintages=_STATE["vintages"],
    )
    return ResidualSpread.fit(
        residuals=replica.residuals, band=replica.band, quarter=replica.quarter, centre=True
    )


def _score_job(job: tuple[str, dict[str, g.ArmDefinition]]) -> dict[str, float]:
    """Score candidate arms at one site with the reference spread; return mean score by name."""
    site, candidates = job
    cache: FeatureCache = _STATE["cache"]
    spread = _reference_spread(site=site)
    scores: dict[str, float] = {}
    for name, definition in candidates.items():
        run = g.run_site(
            cache=cache,
            vintages=_STATE["vintages"],
            site=site,
            cut=common.TUNING_CUT,
            definition=definition,
            with_replica=False,
        )
        keep = ~cache.sites[site].switching_mask[run.index]
        quantiles = spread.quantiles(
            mean=run.mean[keep],
            band=cache.calendar.band[run.index][keep],
            quarter=cache.calendar.quarter[run.index][keep],
        )
        actual = cache.sites[site].y[run.index][keep]
        scores[name] = float(common.quantile_score(actual=actual, quantiles=quantiles).mean())
    return scores


def _stage(
    *, pool: ProcessPoolExecutor, candidates: dict[str, g.ArmDefinition]
) -> dict[str, list[float]]:
    """Score candidates at every decision site; return the scores by name, one per site."""
    results = list(pool.map(_score_job, [(site, candidates) for site in common.DECISION_SITES]))
    return {name: [r[name] for r in results] for name in candidates}


def _best(*, scores: dict[str, list[float]]) -> str:
    """Return the name with the lowest mean score over sites."""
    return min(scores, key=lambda name: float(np.mean(scores[name])))


def _hyper_job(arm: str) -> dict[str, list[float]]:
    """Fit one base arm's filter hyperparameters from its 2023-07-01 residuals."""
    tuned = json.loads((common.WORK_DIR / "tuned_partial.json").read_text())
    definition = g.arm_definitions(tuned=tuned)[arm]
    cache: FeatureCache = _STATE["cache"]
    streams = {
        site: g.run_site(
            cache=cache,
            vintages=_STATE["vintages"],
            site=site,
            cut=common.TUNING_CUT,
            definition=definition,
            with_replica=False,
        ).residual_stream(cache=cache)
        for site in common.DECISION_SITES
    }
    fitted = fit_filter_hyperparameters(
        residuals=streams,
        band=cache.calendar.band,
        end_times=cache.end_times,
        window=(
            common.to_64(moment=common.TUNING_CUT),
            common.to_64(moment=common.year_end(cut=common.TUNING_CUT)),
        ),
    )
    values = {site: hyper.as_array() for site, hyper in fitted.items()}
    values["D9"] = np.mean(list(values.values()), axis=0)
    return {site: array.tolist() for site, array in values.items()}


def main() -> None:
    """Choose the settings, fit the filters, and write `work/tuned.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    base = GamSpec(ridge=g.DEFAULT_RIDGE, capacity="pinned")
    all_years = TrainingWindow()
    partial = common.WORK_DIR / "tuned_partial.json"
    tuned: dict[str, Any] = {
        "drift_step": g.DEFAULT_DRIFT_STEP,
        "step_penalty": 300.0,
        "final_knot": "free",
        "final_knot_shrink": 0.5,
        "window": {"months": None, "drift": False},
    }
    record: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        drift = {
            f"{value:g}": g.ArmDefinition(
                "a1", replace(base, drift="d1", drift_step=value), all_years
            )
            for value in DRIFT_STEP_GRID
        }
        record["drift_step"] = scores = _stage(pool=pool, candidates=drift)
        tuned["drift_step"] = float(_best(scores=scores))
        steps = {
            f"{value:g}": g.ArmDefinition(
                "a1s",
                replace(base, drift="d1_steps", drift_step=tuned["drift_step"], step_penalty=value),
                all_years,
            )
            for value in STEP_PENALTY_GRID
        }
        record["step_penalty"] = scores = _stage(pool=pool, candidates=steps)
        tuned["step_penalty"] = float(_best(scores=scores))
        a1 = replace(base, drift="d1", drift_step=tuned["drift_step"])
        knots: dict[str, g.ArmDefinition] = {
            "free": g.ArmDefinition("a1", a1, all_years),
            "pinned": g.ArmDefinition("a1", replace(a1, final_knot="pinned"), all_years),
            **{
                f"shrunk{round(100 * s)}": g.ArmDefinition(
                    "a1", replace(a1, final_knot="shrunk", final_knot_shrink=s), all_years
                )
                for s in SHRINK_GRID
            },
        }
        record["final_knot"] = scores = _stage(pool=pool, candidates=knots)
        best_knot = _best(scores=scores)
        free_mean = float(np.mean(scores["free"]))
        if best_knot != "free" and free_mean - float(np.mean(scores[best_knot])) < (
            KNOT_ADOPTION_MARGIN
        ):
            best_knot = "free"
        tuned["final_knot"] = "shrunk" if best_knot.startswith("shrunk") else best_knot
        if best_knot.startswith("shrunk"):
            tuned["final_knot_shrink"] = int(best_knot.removeprefix("shrunk")) / 100
        windows = {
            f"{months}m_{'d1' if drift_on else 'none'}": g.ArmDefinition(
                "w",
                replace(a1 if drift_on else replace(a1, drift="none"), final_knot="free"),
                TrainingWindow(months=months),
            )
            for months, drift_on in WINDOWS
        }
        record["window"] = scores = _stage(pool=pool, candidates=windows)
        months_text, drift_text = _best(scores=scores).split("_")
        tuned["window"] = {
            "months": int(months_text.removesuffix("m")),
            "drift": drift_text == "d1",
        }
        partial.write_text(json.dumps(tuned, indent=2) + "\n")
        hypers = list(pool.map(_hyper_job, BASE_ARMS))
    tuned["filter_hyper"] = dict(zip(BASE_ARMS, hypers, strict=True))
    (common.WORK_DIR / "tuned.json").write_text(
        json.dumps({**tuned, "scores": record, "sites": list(common.DECISION_SITES)}, indent=2)
        + "\n"
    )
    print("== tuning year (2023-07 to 2024-06): mean quantile score over five sites, points ==")
    for stage, scores in record.items():
        print(f"\n{stage}")
        for name, per_site in scores.items():
            print(
                f"  {name:>12} "
                + " ".join(f"{s:7.3f}" for s in per_site)
                + f"  mean {np.mean(per_site):7.3f}"
            )
    print(f"\nchosen: { {k: v for k, v in tuned.items() if k != 'filter_hyper'} }")
    print("\nfilter hyperparameters (log q, logit phi, log sigma_z, log sigma_eps) by arm and site")
    for arm, per_site in tuned["filter_hyper"].items():
        for site, values in per_site.items():
            print(f"  {arm:>4} {site:>4} " + " ".join(f"{v:7.3f}" for v in values))


if __name__ == "__main__":
    main()
