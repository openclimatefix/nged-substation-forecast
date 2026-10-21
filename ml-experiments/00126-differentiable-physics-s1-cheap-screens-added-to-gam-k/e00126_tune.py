"""Tune experiment 126's filter hyperparameters on the tuning year.

Every fit uses data before 2023-07-01 (the tuning cut), and the replica of the tuning year is the
fit at 2022-07-01. The scored rows are 2023-07-01 to 2024-06-30 at the five decision sites. D9 has
no history at that cut.

1. **The penalty weight** is T-all's, 1.0, chosen by experiment 114 on the same tuning year; it is
   read from 114's hashed `decisions.json` and not re-tuned, so every arm differs from T-all by its
   added columns only.
2. **Each arm's filter hyperparameters** are fitted by marginal likelihood on that arm's
   out-of-sample tuning-year residuals, under a prior shared across the five sites. D9 takes the
   mean of the five sites' unconstrained values. The `-b0` runs use T-all's hyperparameters.

The choices are written to `work/decisions.json` with a SHA-256 hash in `work/decisions.sha256`,
before any replicate-year or scored-year row is read. The new columns' effective degrees of freedom
are written to `work/tuning.json`.

    uv run --no-sync python ml-experiments/00126-*/e00126_tune.py [--workers 6]
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Final

import e00126_common as common
import e00126_design as design
import e00126_pipeline as pipeline
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import FilterHyperparameters, fit_filter_hyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

TUNING_LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning")
ArmKey = tuple[str, float]
"""An arm's name and penalty weight."""
T_ALL_DECISIONS: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00114-differentiable-physics-n3-n4-interactions-and-heating-threshold"
    / "work"
    / "decisions.json"
)

_STATE: dict[str, Any] = {}


def t_all_weight() -> float:
    """Return T-all's penalty weight from experiment 114's decisions, after checking its hash."""
    text = T_ALL_DECISIONS.read_text()
    recorded = T_ALL_DECISIONS.with_name("decisions.sha256").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != recorded:
        msg = "experiment 114's decisions.json no longer matches its recorded hash"
        raise ValueError(msg)
    return float(json.loads(text)["lambda"]["t_all"])


def arm_keys() -> list[ArmKey]:
    """Return every arm: T-all and the eleven arms that change its columns, at T-all's weight."""
    weight = t_all_weight()
    return [(name, weight) for name in pipeline.ARM_NAMES]


def key_text(*, key: ArmKey) -> str:
    """Return a printable name for an arm key."""
    name, lam = key
    return name if lam == 0.0 else f"{name}@{lam:g}"


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def _fit_site(site: str) -> dict[tuple[str, float, str], pipeline.YearRun]:
    """Fit every candidate arm at one site at the replica and tuning cuts."""
    cache: FeatureCache = _STATE["cache"]
    runs: dict[tuple[str, float, str], pipeline.YearRun] = {}
    for label in TUNING_LABELS:
        for name, lam in arm_keys():
            run = pipeline.fit_year(
                cache=cache,
                vintages=_STATE["vintages"],
                site=site,
                fit_label=label,
                arm_name=name,
                penalty_weight=lam,
            )
            if run is not None:
                runs[(name, lam, label)] = run
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
    """Return each arm's effective degrees of freedom above T-all's at one site.

    The count is `trace((G + P)^-1 G)` of the arm minus T-all's, at the tuning cut, ignoring the
    sign bounds.
    """
    cache: FeatureCache = _STATE["cache"]
    vintages = _STATE["vintages"]
    base = design.build_base(cache=cache, vintages=vintages, site=site, cut=common.TUNING_CUT)
    weight = t_all_weight()
    reference = _edf(
        cache=cache,
        base=base,
        arm=pipeline.arm_spec(arm_name="t_all", penalty_weight=weight),
        vintages=vintages,
    )
    out: dict[str, float] = {}
    for name in pipeline.ARM_NAMES:
        arm = pipeline.arm_spec(arm_name=name, penalty_weight=weight)
        out[name] = _edf(cache=cache, base=base, arm=arm, vintages=vintages) - reference
    return out


def _edf(
    *,
    cache: FeatureCache,
    base: design.BaseFit,
    arm: design.ArmSpec,
    vintages: pl.DataFrame,
) -> float:
    """Return an arm's effective degrees of freedom at the base fit."""
    context = pipeline.arm_context(
        cache=cache, vintages=vintages, site=base.site, cut=common.TUNING_CUT, arm_name=arm.name
    )
    x = design.arm_matrix(
        cache=cache,
        layout=base.layout,
        site=base.site,
        index=base.index,
        x0=base.x0,
        arm=arm,
        context=context,
    )
    x = x[np.isfinite(x).all(axis=1)]
    xt = torch.as_tensor(x)
    penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
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
    """Choose the penalty weights, fit the filters, and write `decisions.json` with its hash."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
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
                    site: by_site[site][(*key, "tuning")].residual_stream(cache=cache)
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
                {label: by_site[site][(*key, label)] for label in TUNING_LABELS},
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
    weight = t_all_weight()
    decisions = {
        "lambda": {"t_all": weight},
        "filter_hyper": {key[0]: hypers[key] for key in keys},
        "sites": list(sites),
    }
    text = json.dumps(decisions, indent=2, sort_keys=True) + "\n"
    (common.WORK_DIR / "decisions.json").write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    (common.WORK_DIR / "decisions.sha256").write_text(digest + "\n")
    record = {
        "score_unfiltered": {key_text(key=k): v for k, v in mean_plain.items()},
        "score_filtered": {key_text(key=k): v for k, v in mean_filtered.items()},
        "per_site": {
            key_text(key=k): {s: list(v) for s, v in per.items()} for k, per in scores.items()
        },
        "new_column_edf_by_site": edf,
    }
    (common.WORK_DIR / "tuning.json").write_text(json.dumps(record, indent=2) + "\n")
    print("== tuning year (2023-07 to 2024-06): day-ahead quantile score, points, five sites ==")
    print(f"{'arm':>10} {'unfiltered':>11} {'filtered':>9} {'new edf (median over sites)':>28}")
    for key in keys:
        text_key = key_text(key=key)
        edf_values = [edf[s][text_key] for s in sites if text_key in edf[s]]
        edf_text = f"{float(np.median(edf_values)):.1f}" if edf_values else "-"
        print(f"{text_key:>10} {mean_plain[key]:11.4f} {mean_filtered[key]:9.4f} {edf_text:>28}")
    print(f"\nT-all penalty weight: {weight}")
    print(f"decisions.json sha256: {digest}")
    frame = pl.DataFrame(
        [
            {"arm": key_text(key=k), "site": s, "unfiltered": v[0], "filtered": v[1]}
            for k, per in scores.items()
            for s, v in per.items()
        ]
    )
    print(frame.pivot(on="site", index="arm", values="filtered"))


if __name__ == "__main__":
    main()
