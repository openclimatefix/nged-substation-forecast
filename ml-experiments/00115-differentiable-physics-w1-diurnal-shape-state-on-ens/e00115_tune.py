"""Fit experiment 115's shape settings, choose its tuned values, and build every arm's forecasts.

Everything that picks a setting reads the tuning rows only: the odd months of 2024-07 to 2025-06,
with the origin and the valid time both in those months. The stages, in order:

    hypers       fit log q_s and log sigma_eps per site under a shared prior (KW; KS; KS1 in full)
    corrections  build the correction of the arms and the candidates
    spreads      fit each arm's spread on the tuning rows and score the candidates
    choose       pick the lead-band multipliers and the recipe's kappa; build those arms
    factors      score the spread factor grid and pick one factor per lead band per arm
    frames       forecast every cached run and write the arm's frames

    uv run python ml-experiments/00115-*/e00115_tune.py <stage> [--workers 6] [--arms ...]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Final

import e00115_arms as arms
import e00115_common as common
import e00115_shape as shape
import e00115_site as site_module
import numpy as np
import torch
from studies.dp_assimilation import MIN_PRIOR_SPREAD, usable_cutoff_positions  # noqa: F401
from studies.dp_ens_weather import N_LEAD_BANDS

SITES: Final[tuple[str, ...]] = common.SCORED_SITES
FRAMES_DIR: Final[Path] = common.WORK_DIR / "frames"
TUNED_PATH: Final[Path] = common.WORK_DIR / "tuned.json"
START_LOG_Q_S: Final[float] = float(np.log(0.01))
HYPER_FAMILIES: Final[tuple[str, ...]] = ("KW", "KS", "KS1")
REPORTED: Final[tuple[str, ...]] = (
    "K0",
    "KW",
    "KW-lb",
    "K0-lb",
    "KW-day",
    "R",
    "R75",
    "KS",
    "KW-nomid",
    "KS1",
    "KW-pl52",
    "KW-d0",
    "KW-d24",
    "KW-d33",
    "K0-d0",
    "K0-d24",
    "K0-d33",
)


@lru_cache(maxsize=2)
def get_site(site: str) -> site_module.SiteData:
    """Load a site once per worker process."""
    torch.set_num_threads(1)
    return site_module.load_site(site=site)


def tuning_stream(*, data: site_module.SiteData) -> np.ndarray:
    """Return the residual stream with every reading outside the tuning months set to `NaN`."""
    codes = np.array([100 * y + m for y, m in common.TUNING_MONTHS])
    return np.where(np.isin(data.month_code, codes), data.stream, np.nan)


def read_tuned() -> dict:
    """Return the stored tuned choices."""
    return json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}


def update_tuned(*, key: str, value: object) -> None:
    """Store one tuned choice."""
    tuned = read_tuned()
    tuned[key] = value
    common.write_json_atomic(path=TUNED_PATH, content=tuned)


# --- stage: hypers ---


def _fit_shape_one(job: tuple[str, str, list[float], list[float], list[float]]) -> tuple[str, list]:
    family, site, prior_mean, prior_spread, start = job
    data = get_site(site)
    structure = arms.structure_of(family=family, band_of_half_hour=data.band_of_half_hour)
    values = shape.fit_shape_hyper(
        residual=tuning_stream(data=data),
        half_hour=data.half_hour,
        structure=structure,
        base=data.kalman_hyper,
        prior_mean=np.array(prior_mean),
        prior_spread=np.array(prior_spread),
        start=np.array(start),
    )
    return site, values.tolist()


def _fit_free_one(job: tuple[str, list[float], list[float], list[float]]) -> tuple[str, list]:
    site, prior_mean, prior_spread, start = job
    data = get_site(site)
    structure = arms.structure_of(family="KS1", band_of_half_hour=data.band_of_half_hour)
    values = shape.fit_free_hyper(
        residual=tuning_stream(data=data),
        half_hour=data.half_hour,
        structure=structure,
        prior_mean=np.array(prior_mean),
        prior_spread=np.array(prior_spread),
        start=np.array(start),
    )
    return site, values.tolist()


def fit_hypers(*, family: str, pool: ProcessPoolExecutor) -> dict[str, list[float]]:
    """Fit a family's shape settings at every site under a prior shared across sites.

    Args:
        family: `KW`, `KS`, or `KS1`.
        pool: The worker pool.

    Returns:
        Each site's settings: (log q_s, log sigma_eps) for `KW` and `KS`, and all five for `KS1`.
    """
    base_noise = [get_site(site).kalman_hyper for site in SITES]
    if family == "KS1":
        stacked = np.array(
            [
                [h.log_q, h.logit_phi, h.log_sigma_z, h.log_sigma_eps, START_LOG_Q_S]
                for h in base_noise
            ]
        )
        prior_mean = stacked.mean(axis=0)
        prior_spread = np.array([1.5, 1.0, 1.0, 1.0, 2.0])
        fitted: dict[str, list] = {}
        for _ in range(shape.N_EMPIRICAL_BAYES_ROUNDS):
            jobs = [
                (
                    site,
                    prior_mean.tolist(),
                    prior_spread.tolist(),
                    fitted.get(site, prior_mean.tolist()),
                )
                for site in SITES
            ]
            fitted = dict(pool.map(_fit_free_one, jobs))
            values = np.array(list(fitted.values()))
            prior_mean = values.mean(axis=0)
            prior_spread = np.maximum(values.std(axis=0, ddof=1), MIN_PRIOR_SPREAD)
        return fitted
    noise = float(np.mean([h.log_sigma_eps for h in base_noise]))
    prior_mean = np.array([START_LOG_Q_S, noise])
    prior_spread = np.array([2.0, 0.5])
    fitted = {}
    for _ in range(shape.N_EMPIRICAL_BAYES_ROUNDS):
        jobs = [
            (
                family,
                site,
                prior_mean.tolist(),
                prior_spread.tolist(),
                fitted.get(site, prior_mean.tolist()),
            )
            for site in SITES
        ]
        fitted = dict(pool.map(_fit_shape_one, jobs))
        values = np.array(list(fitted.values()))
        prior_mean = values.mean(axis=0)
        prior_spread = np.maximum(values.std(axis=0, ddof=1), shape.MIN_SHAPE_PRIOR_SPREAD)
    return fitted


def stage_hypers(*, pool: ProcessPoolExecutor, families: list[str]) -> None:
    """Fit the shape settings of the chosen families and store them."""
    stored = arms.read_hypers() if arms.hypers_path().exists() else {}
    for family in families:
        started = time.monotonic()
        stored[family] = fit_hypers(family=family, pool=pool)
        common.write_json_atomic(path=arms.hypers_path(), content=stored)
        print(f"{family}: {stored[family]} ({time.monotonic() - started:.0f} s)", flush=True)


# --- stage: corrections ---


def _correction_job(job: tuple[str, str]) -> str:
    name, site = job
    data = get_site(site)
    spec = arms.arm_table()[name]
    traces = arms.TraceCache(data=data, hypers=arms.read_hypers())
    k0_q50 = None
    if spec.family == "R":
        k0 = arms.load_correction(arm="K0", site=site)
        spread = arms.load_spread(arm="K0", site=site)
        quantiles, rows = arms.quantiles_for_runs(
            data=data,
            correction=k0.total,
            spread=spread,
            run_indices=np.arange(len(data.runs)),
        )
        k0_q50 = np.full(data.n_rows, np.nan)
        k0_q50[rows] = quantiles[:, 5]
    correction = arms.compute_correction(spec=spec, traces=traces, k0_q50=k0_q50)
    arms.save_correction(arm=name, site=site, correction=correction)
    return f"{name} {site}"


# --- stage: spreads ---


def _spread_job(job: tuple[str, str]) -> dict:
    name, site = job
    data = get_site(site)
    correction = arms.load_correction(arm=name, site=site)
    spread = arms.fit_spread(data=data, correction=correction.total)
    if spread is None:
        return {"arm": name, "site": site}
    arms.save_spread(arm=name, site=site, spread=spread)
    scores = arms.tuning_band_scores(data=data, correction=correction.total, spread=spread)
    d1_sum, d1_count = arms.d1_tuning_score(data=data, correction=correction.total, spread=spread)
    return {
        "arm": name,
        "site": site,
        "band_scores": scores.tolist(),
        "d1": [d1_sum, d1_count],
    }


def pooled_mean(*, results: list[dict], arm: str, key: str, sites: tuple[str, ...]) -> np.ndarray:
    """Return the pooled sum and count of a stored score over the decision sites."""
    total = None
    for result in results:
        if result["arm"] == arm and result["site"] in sites and key in result:
            value = np.array(result[key], dtype=float)
            total = value if total is None else total + value
    if total is None:
        msg = f"no {key} scores for {arm}"
        raise ValueError(msg)
    return total


# --- stage: factors ---


def _factor_job(job: tuple[str, str]) -> dict:
    name, site = job
    data = get_site(site)
    correction = arms.load_correction(arm=name, site=site)
    spread = arms.load_spread(arm=name, site=site)
    out: dict = {"arm": name, "site": site, "scores": {}}
    for factor in arms.FACTOR_GRID:
        scored = arms.tuning_band_scores(
            data=data,
            correction=correction.total,
            spread=spread.with_factors(factors=[factor] * N_LEAD_BANDS),
        )
        out["scores"][str(factor)] = scored.tolist()
    return out


# --- stage: frames ---


def frame_path(*, arm: str, site: str) -> Path:
    """Return the file that holds an arm's frame at a site."""
    return FRAMES_DIR / f"{arm}_{site}.parquet"


def _frame_job(job: tuple[str, str]) -> str:
    name, site = job
    data = get_site(site)
    correction = arms.load_correction(arm=name, site=site)
    factors = read_tuned()["spread_factors"][name]
    spread = arms.load_spread(arm=name, site=site).with_factors(factors=factors)
    frame = arms.frame_for_arm(data=data, arm=name, correction=correction, spread=spread)
    frame = arms.replace_period_masks(frame=frame, data=data)
    target = frame_path(arm=name, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return f"{name} {site}: {frame.height:,} rows"


def _run(pool: ProcessPoolExecutor, function, jobs: list) -> list:
    started = time.monotonic()
    results = list(pool.map(function, jobs))
    print(f"{function.__name__}: {len(jobs)} jobs, {time.monotonic() - started:.0f} s", flush=True)
    return results


def main() -> None:
    """Run one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["hypers", "corrections", "spreads", "choose", "factors", "frames"]
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", default=None)
    parser.add_argument("--families", nargs="+", default=["KW"])
    args = parser.parse_args()
    table = arms.arm_table()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.stage == "hypers":
            stage_hypers(pool=pool, families=args.families)
            return
        names = args.arms or list(table)
        jobs = [(name, site) for name in names for site in SITES]
        if args.stage == "corrections":
            _run(pool, _correction_job, jobs)
        elif args.stage == "spreads":
            results = _run(pool, _spread_job, jobs)
            stored = read_tuned().get("spread_scores", {})
            for result in results:
                stored.setdefault(result["arm"], {})[result["site"]] = result
            update_tuned(key="spread_scores", value=stored)
        elif args.stage == "factors":
            results = _run(pool, _factor_job, jobs)
            stored = read_tuned().get("factor_scores", {})
            for result in results:
                stored.setdefault(result["arm"], {})[result["site"]] = result["scores"]
            update_tuned(key="factor_scores", value=stored)
            factors = read_tuned().get("spread_factors", {})
            for name in names:
                factors[name] = choose_factors(scores=stored[name])
            update_tuned(key="spread_factors", value=factors)
            print(json.dumps(factors, indent=1))
        elif args.stage == "frames":
            for line in _run(pool, _frame_job, jobs):
                print(line)
        elif args.stage == "choose":
            choose(pool=pool)


def choose_factors(*, scores: dict[str, dict[str, list]]) -> list[float]:
    """Return, per lead band, the factor with the lowest pooled tuning score at the decision sites."""
    totals = np.zeros((len(arms.FACTOR_GRID), N_LEAD_BANDS, 2))
    for site, per_factor in scores.items():
        if site not in common.DECISION_SITES:
            continue
        for position, factor in enumerate(arms.FACTOR_GRID):
            totals[position] += np.array(per_factor[str(factor)])
    mean = totals[:, :, 0] / totals[:, :, 1]
    return [float(arms.FACTOR_GRID[i]) for i in mean.argmin(axis=0)]


def choose(*, pool: ProcessPoolExecutor) -> None:
    """Choose the lead-band multipliers and the recipe's kappa, and build those arms.

    The multipliers come from the candidates' tuning scores per lead band (K0 and KW at multiplier
    1 are the arms themselves); kappa comes from the D+1 tuning score of the recipe candidates.
    """
    scores = read_tuned()["spread_scores"]
    results = [r for per_site in scores.values() for r in per_site.values()]
    chosen: dict[str, list[float]] = {}
    for family in ("K0", "KW"):
        totals = []
        for multiplier in arms.MULTIPLIER_GRID:
            name = (
                family
                if multiplier == 1.0
                else arms.candidate_name(family=family, multiplier=multiplier)
            )
            totals.append(
                pooled_mean(
                    results=results, arm=name, key="band_scores", sites=common.DECISION_SITES
                )
            )
        totals = np.array(totals)
        mean = totals[:, :, 0] / totals[:, :, 1]
        chosen[family] = [float(arms.MULTIPLIER_GRID[i]) for i in mean.argmin(axis=0)]
    kappa_means = []
    for kappa in arms.KAPPA_GRID:
        total = pooled_mean(
            results=results, arm=f"R_k{kappa:g}", key="d1", sites=common.DECISION_SITES
        )
        kappa_means.append(total[0] / total[1])
    best_kappa = float(arms.KAPPA_GRID[int(np.argmin(kappa_means))])
    update_tuned(key="multipliers", value=chosen)
    update_tuned(key="kappa", value={"grid_scores": kappa_means, "chosen": best_kappa})
    print(f"multipliers {chosen}; kappa {best_kappa} from {kappa_means}")
    for family in ("K0", "KW"):
        name = f"{family}-lb"
        for site in SITES:
            data = get_site(site)
            correction = arms.lead_band_correction(
                data=data, family=family, multipliers=tuple(chosen[family])
            )
            arms.save_correction(arm=name, site=site, correction=correction)
    for site in SITES:
        chosen_correction = arms.load_correction(arm=f"R_k{best_kappa:g}", site=site)
        arms.save_correction(arm="R", site=site, correction=chosen_correction)


if __name__ == "__main__":
    main()
