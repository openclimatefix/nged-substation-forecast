"""Tune experiment 121's shrinkage multiplier and filter hyperparameters on the tuning year.

Every fit uses data before 2023-07-01 (the tuning cut), and the replica of the tuning year is the
fit at 2022-07-01. The scored rows are 2023-07-01 to 2024-06-30 at the five decision sites. D9 has
no history at that cut.

1. **The shrinkage multiplier m** of MH-sh is chosen from `common.SHRINK_GRID`, one value for all
   sites, by the day-ahead quantile score after the filter on rows outside the switching masks.
   Each candidate has its own filter hyperparameters.
2. **Each arm's filter hyperparameters** are fitted by marginal likelihood on that arm's
   out-of-sample tuning-year residuals, under a prior shared across the five sites. D9 takes the
   mean of the five sites' unconstrained values.

The choices are written to `work/decisions.json` with a SHA-256 hash in `work/decisions.sha256`,
before any replicate-year or scored-year row is read. The added columns' effective degrees of
freedom at each multiplier are written to `work/tuning.json`.

    uv run python ml-experiments/00121-*/e00121_tune.py [--workers 6]
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00121_common as common
import e00121_design as design
import e00121_pipeline as pipeline
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import FilterHyperparameters, fit_filter_hyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

TUNING_LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning")
ArmKey = str
"""An arm's name, a key of `design.all_arms()`."""

_STATE: dict[str, Any] = {}


def arm_keys() -> list[ArmKey]:
    """Return every candidate arm."""
    return list(design.all_arms())


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def _fit_site(site: str) -> dict[tuple[str, str], pipeline.YearRun]:
    """Fit every candidate arm at one site at the replica and tuning cuts."""
    cache: FeatureCache = _STATE["cache"]
    runs: dict[tuple[str, str], pipeline.YearRun] = {}
    for label in TUNING_LABELS:
        for name in arm_keys():
            run = pipeline.fit_year(
                cache=cache,
                vintages=_STATE["vintages"],
                site=site,
                fit_label=label,
                arm_name=name,
            )
            if run is not None:
                runs[(name, label)] = run
    return runs


def _hyper_job(
    job: tuple[ArmKey, dict[str, np.ndarray]],
) -> tuple[ArmKey, dict[str, list[float]]]:
    """Fit one arm's filter hyperparameters from its tuning-year residual streams."""
    key, streams = job
    cache: FeatureCache = _STATE["cache"]
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
    return key, {site: array.tolist() for site, array in values.items()}


def _edf_job(site: str) -> dict[str, float]:
    """Return the added columns' effective degrees of freedom of each arm at one site.

    The count is `trace((G + P)^-1 G)` of the arm minus B0's, at the tuning cut, ignoring the sign
    bounds.
    """
    cache: FeatureCache = _STATE["cache"]
    base = design.build_base(
        cache=cache, vintages=_STATE["vintages"], site=site, cut=common.TUNING_CUT
    )
    reference = _edf(cache=cache, base=base, arm=design.B0)
    return {
        name: _edf(cache=cache, base=base, arm=arm) - reference
        for name, arm in design.all_arms().items()
    }


def _edf(*, cache: FeatureCache, base: design.BaseFit, arm: design.ArmSpec) -> float:
    """Return an arm's effective degrees of freedom at the base fit."""
    x = design.arm_matrix(
        cache=cache,
        layout=base.layout,
        site=base.site,
        index=base.index,
        x0=base.x0,
        arm=arm,
        cut=base.cut,
    )
    xt = torch.as_tensor(x)
    penalty = design.penalty_for(cache=cache, base=base, arm=arm, x=x)
    return design.effective_degrees_of_freedom(gram=xt.T @ xt, penalty=penalty)


def _score_job(
    job: tuple[ArmKey, str, dict[str, pipeline.YearRun], list[float]],
) -> tuple[ArmKey, str, float, float]:
    """Return the mean tuning-year quantile score of an arm at a site, unfiltered and filtered."""
    key, site, runs, hyper_values = job
    cache: FeatureCache = _STATE["cache"]
    scored = pipeline.score_year(
        cache=cache,
        runs=runs,
        label="tuning",
        hyper=FilterHyperparameters.from_array(values=np.array(hyper_values)),
    )
    keep = ~cache.sites[site].switching_mask[scored.index]
    actual = cache.sites[site].y[scored.index][keep]
    plain = common.quantile_score(actual=actual, quantiles=scored.quantiles[keep]).mean()
    filtered = common.quantile_score(
        actual=actual, quantiles=scored.filtered_quantiles[keep]
    ).mean()
    return key, site, float(plain), float(filtered)


def main() -> None:
    """Choose the multiplier, fit the filters, and write `decisions.json` with its hash."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    if (common.WORK_DIR / "decisions.sha256").exists() and not arguments.force:
        msg = "decisions.sha256 exists; pass --force to replace the hashed decisions"
        raise SystemExit(msg)
    common.load_cache()
    keys = arm_keys()
    sites = common.DECISION_SITES
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        by_site = dict(zip(sites, pool.map(_fit_site, sites), strict=True))
        edf = dict(zip(sites, pool.map(_edf_job, sites), strict=True))
        cache = common.load_cache()
        hyper_jobs = [
            (
                key,
                {
                    site: by_site[site][(key, "tuning")].residual_stream(cache=cache)
                    for site in sites
                },
            )
            for key in keys
        ]
        hypers = dict(pool.map(_hyper_job, hyper_jobs))
        score_jobs = [
            (
                key,
                site,
                {label: by_site[site][(key, label)] for label in TUNING_LABELS},
                hypers[key][site],
            )
            for key in keys
            for site in sites
        ]
        scores: dict[ArmKey, dict[str, tuple[float, float]]] = {key: {} for key in keys}
        for key, site, plain, filtered in pool.map(_score_job, score_jobs):
            scores[key][site] = (plain, filtered)
    mean_plain = {k: float(np.mean([s[0] for s in v.values()])) for k, v in scores.items()}
    mean_filtered = {k: float(np.mean([s[1] for s in v.values()])) for k, v in scores.items()}
    candidates = [design.shrunk_arm(shrink=m).name for m in common.SHRINK_GRID]
    chosen_name = min(candidates, key=lambda k: mean_filtered[k])
    chosen_shrink = common.SHRINK_GRID[candidates.index(chosen_name)]
    decisions = {
        "shrink": chosen_shrink,
        "filter_hyper": {key: hypers[key] for key in keys},
        "sites": list(sites),
    }
    text = json.dumps(decisions, indent=2, sort_keys=True) + "\n"
    (common.WORK_DIR / "decisions.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (common.WORK_DIR / "decisions.sha256").write_text(digest + "\n")
    record = {
        "score_unfiltered": mean_plain,
        "score_filtered": mean_filtered,
        "per_site": {k: {s: list(v) for s, v in per.items()} for k, per in scores.items()},
        "added_column_edf_by_site": edf,
    }
    (common.WORK_DIR / "tuning.json").write_text(json.dumps(record, indent=2) + "\n")
    print("== tuning year (2023-07 to 2024-06): day-ahead quantile score, points, five sites ==")
    print(f"{'arm':>10} {'unfiltered':>11} {'filtered':>9} {'added edf (median over sites)':>30}")
    for key in keys:
        edf_median = float(np.median([edf[s][key] for s in sites]))
        print(f"{key:>10} {mean_plain[key]:11.4f} {mean_filtered[key]:9.4f} {edf_median:>30.1f}")
    print(f"\nchosen shrinkage multiplier: {chosen_shrink:g}")
    print(f"decisions.json sha256: {digest}")
    frame = pl.DataFrame(
        [
            {"arm": k, "site": s, "unfiltered": v[0], "filtered": v[1]}
            for k, per in scores.items()
            for s, v in per.items()
        ]
    )
    print(frame.pivot(on="site", index="arm", values="filtered"))


if __name__ == "__main__":
    main()
