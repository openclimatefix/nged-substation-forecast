"""Fit experiment 126's arms at every cut and write the prediction sets.

**For each substation,** the script fits T-all and the eleven arms that change its columns at the
four cuts, forecasts the year after each, and writes day-ahead predictions with and without the
Kalman filter for the tuning, replicate, and scored years. The penalty weight and the filters'
hyperparameters come from `work/decisions.json`, whose hash must still match the one recorded
before the gates.

**Sets.** For an arm `a` and year `y`: `<y>_<a>` (unfiltered), `<y>_<a>_k` (the filter fitted on
that arm's tuning residuals), and, except for T-all, `<y>_<a>_kb0` (T-all's filter hyperparameters,
so only the terms differ). T-all alone also gets `<y>_t_all_fmix_k` (F-mix) and `<y>_t_all_gskf_k`
(G-skf). Both use T-all's plain hyperparameters for the normal regime; F-mix's factors come from
`work/decisions_filters.json`, and G-skf's transition matrix is fitted by expectation-maximisation
on the replica's out-of-sample residuals, with the hand-filtered event rows masked.

    uv run --no-sync python ml-experiments/00126-*/e00126_fit.py [--workers 5] [--years tuning]
"""

import argparse
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00126_common as common
import e00126_fmix as fmix
import e00126_pipeline as pipeline
import e00126_skf as skf
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

FIT_LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning", "replicate", "scored")
SCORED_LABELS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def verified(name: str) -> dict[str, Any]:
    """Return `work/<name>.json` after checking its hash against `<name>.sha256`."""
    text = (common.WORK_DIR / f"{name}.json").read_text()
    recorded = (common.WORK_DIR / f"{name}.sha256").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != recorded:
        msg = f"{name}.json no longer matches the hash recorded before the fits"
        raise ValueError(msg)
    return json.loads(text)


def _hyper(*, decisions: dict[str, Any], arm: str, site: str) -> FilterHyperparameters:
    """Return an arm's filter hyperparameters at a site, from the hashed decisions."""
    return FilterHyperparameters.from_array(values=np.array(decisions["filter_hyper"][arm][site]))


def _gskf_transition(
    *,
    cache: FeatureCache,
    runs: dict[str, pipeline.YearRun],
    label: str,
    hyper: FilterHyperparameters,
) -> np.ndarray:
    """Fit G-skf's transition matrix on the replica year's out-of-sample residuals.

    The replica of a scored year is the fit 12 months earlier, forecasting [cut - 12 months, cut).
    The rows inside the padded hand-filtered events are masked (set to `NaN`), so the matrix sees
    no event label and no injection.
    """
    replica = runs[pipeline.PREVIOUS_LABEL[label]]
    stream = replica.residual_stream(cache=cache)
    stream[cache.sites[replica.site].switching_mask] = np.nan
    return skf.fit_transition(residual=stream[None, :], band=cache.calendar.band, hyper=hyper)


def fit_site(job: tuple[str, dict[str, Any], dict[str, Any], tuple[str, ...]]) -> dict[str, Any]:
    """Fit every arm of one substation and return its prediction frames.

    Args:
        job: The substation's label, the verified decisions, the verified filter decisions, and
            the scored years to write.

    Returns:
        A dict with `frames` (set name to predictions), `cards` (coefficients), and `transitions`.
    """
    site, decisions, filter_decisions, years = job
    cache: FeatureCache = _STATE["cache"]
    vintages = _STATE["vintages"]
    weight = float(decisions["lambda"]["t_all"])
    needed = {pipeline.PREVIOUS_LABEL[y] for y in years} | set(years)
    runs: dict[str, dict[str, pipeline.YearRun]] = {}
    for arm_name in pipeline.ARM_NAMES:
        runs[arm_name] = {}
        for label in FIT_LABELS:
            if label not in needed:
                continue
            run = pipeline.fit_year(
                cache=cache,
                vintages=vintages,
                site=site,
                fit_label=label,
                arm_name=arm_name,
                penalty_weight=weight,
            )
            if run is not None:
                runs[arm_name][label] = run
    regimes = fmix.regime_map(cache=cache, site=site).regime
    log_factors = np.array(filter_decisions["log_factors"][site])
    base_hyper = _hyper(decisions=decisions, arm="t_all", site=site)
    frames: dict[str, pl.DataFrame] = {}
    transitions: dict[str, list[list[float]]] = {}
    for label in years:
        for arm_name in pipeline.ARM_NAMES:
            arm_runs = runs[arm_name]
            if label not in arm_runs or pipeline.PREVIOUS_LABEL[label] not in arm_runs:
                continue
            own = pipeline.score_year(
                cache=cache,
                runs=arm_runs,
                label=label,
                hyper=_hyper(decisions=decisions, arm=arm_name, site=site),
            )
            for name, frame in pipeline.scored_frames(
                cache=cache, site=site, arm_name=arm_name, scored=own
            ).items():
                frames[f"{label}_{name}"] = frame
            if arm_name == "t_all":
                continue
            control = pipeline.score_year(cache=cache, runs=arm_runs, label=label, hyper=base_hyper)
            frames[f"{label}_{arm_name}_kb0"] = pipeline.scored_frames(
                cache=cache, site=site, arm_name=arm_name, scored=control
            )[f"{arm_name}_k"].with_columns(arm=pl.lit(f"{arm_name}_kb0"))
        t_all_runs = runs["t_all"]
        if label not in t_all_runs or pipeline.PREVIOUS_LABEL[label] not in t_all_runs:
            continue
        mixed = pipeline.score_year(
            cache=cache,
            runs=t_all_runs,
            label=label,
            hyper=base_hyper,
            build=fmix.regime_builder(regimes=regimes, log_factors=log_factors, hyper=base_hyper),
        )
        frames[f"{label}_t_all_fmix_k"] = pipeline.scored_frames(
            cache=cache, site=site, arm_name="t_all_fmix", scored=mixed
        )["t_all_fmix_k"]
        transition = _gskf_transition(cache=cache, runs=t_all_runs, label=label, hyper=base_hyper)
        transitions[label] = transition.tolist()
        switching = pipeline.score_year(
            cache=cache,
            runs=t_all_runs,
            label=label,
            hyper=base_hyper,
            build=skf.switching_builder(hyper=base_hyper, transition=transition),
        )
        frames[f"{label}_t_all_gskf_k"] = pipeline.scored_frames(
            cache=cache, site=site, arm_name="t_all_gskf", scored=switching
        )["t_all_gskf_k"]
    cards = {
        arm_name: {label: run.fit.theta for label, run in arm_runs.items()}
        for arm_name, arm_runs in runs.items()
    }
    print(f"{site}: {len(frames)} sets", flush=True)
    return {"frames": frames, "cards": cards, "transitions": transitions}


def main() -> None:
    """Fit all substations and write the prediction sets and result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument(
        "--years",
        nargs="+",
        default=list(SCORED_LABELS),
        choices=SCORED_LABELS,
        help="the scored years to fit and write",
    )
    arguments = parser.parse_args()
    decisions = verified("decisions")
    filter_decisions = verified("decisions_filters")
    common.load_cache()
    years = tuple(arguments.years)
    jobs = [(site, decisions, filter_decisions, years) for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        results = list(pool.map(fit_site, jobs))
    names = dict.fromkeys(name for result in results for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in results if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    suffix = "_".join(years)
    cards = {site: r["cards"] for site, r in zip(common.SCORED_SITES, results, strict=True)}
    (common.WORK_DIR / f"coefficients_{suffix}.pkl").write_bytes(pickle.dumps(cards))
    transitions = {
        site: r["transitions"] for site, r in zip(common.SCORED_SITES, results, strict=True)
    }
    (common.WORK_DIR / f"transitions_{suffix}.json").write_text(
        json.dumps(transitions, indent=2) + "\n"
    )
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
