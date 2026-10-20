"""Fit experiment 114's arms at every cut, write the prediction sets, profiles, and capacity check.

**For each substation and arm,** the script fits the frozen model at the four cuts, forecasts the
year after each, and writes day-ahead predictions with and without the Kalman filter for the
tuning, replicate, and scored years. The penalty weights and the filters' hyperparameters come from
`work/decisions.json`, whose hash must still match the one recorded before the gates.

**Arms.** `b0`; `t1`, `t2`, `t3`, and `t_all` at their chosen weights; `h_site` (B0 with the
heating hinge at the threshold that minimises the unpenalised training loss at each cut). The
filtered version of an arm has the suffix `_k`. `t_all_kb0` is T-all run through B0's filter
hyperparameters (the filter-fit control).

**Profiles.** At every cut each substation's loss is profiled over the 15 thresholds, with the
full sample and 50 quarter-stratified month-block resamples. The scored cut's profile also gets the
per-site printing rule.

**Capacity check (descriptive, scored cut only).** B0 and T-all with the PV capacity fitted
instead of pinned, 50 month-block weightings each; the latest season-year's capacity is compared.

    uv run python ml-experiments/00114-*/e00114_fit.py [--workers 6]
"""

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00114_common as common
import e00114_design as design
import e00114_gate2 as gate2
import e00114_pipeline as pipeline
import e00114_profile as profile
import numpy as np
import polars as pl
import torch
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

FIT_LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning", "replicate", "scored")
SCORED_LABELS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
CAPACITY_FLAG_POINTS: Final[float] = 2.0
PROFILE_SEED_TAG: Final[int] = 5
CAPACITY_SEED_TAG: Final[int] = 6

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def _hyper(*, decisions: dict[str, Any], arm: str, site: str) -> FilterHyperparameters:
    """Return an arm's filter hyperparameters at a site, from the hashed decisions."""
    return FilterHyperparameters.from_array(values=np.array(decisions["filter_hyper"][arm][site]))


def _profile_record(
    *,
    cache: FeatureCache,
    base: design.BaseFit,
    label: str,
    tau: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Profile one substation's loss over the thresholds at one cut and apply the printing rule."""
    union = profile.build_union(cache=cache, base=base)
    counts = profile.weighting_set(quarter_of_month=base.quarter_of_month, rng=rng)
    result = profile.run_profile(union=union, counts=counts, target=base.target)
    argmins = result.argmin_thresholds()
    low, high = profile.argmin_interval(argmins=argmins[1:])
    nats = profile.effective_nats(loss=result.loss[0], n_rows=base.n, inflation=tau)
    base_position = profile.BASE_THRESHOLD_POSITION
    flat_low, flat_high = common.THRESHOLD_FLAT_RANGE_C
    outside = high < flat_low or low > flat_high
    printable = bool(outside and nats[base_position] >= 2.0)
    slopes = result.theta[base_position] @ profile.band_slope_functionals(
        layout_heating_start=base.layout.slices["heating"].start, p0=base.x0.shape[1]
    )
    widen = float(np.sqrt(len(base.months) / (len(base.months) - 1)))
    p_low, p_high = np.percentile(slopes[1:], profile.ARGMIN_INTERVAL_PERCENTILES, axis=0)
    within_2_nats = [common.HINGE_GRID_C[i] for i in range(len(nats)) if nats[i] <= 2.0]
    return {
        "site": base.site,
        "label": label,
        "n_rows": base.n,
        "argmin_c": float(argmins[0]),
        "interval_low_c": low,
        "interval_high_c": high,
        "nats": nats.tolist(),
        "loss": result.loss[0].tolist(),
        "penalised_loss": result.penalised_loss[0].tolist(),
        "nats_at_base": float(nats[base_position]),
        "within_2_nats_low_c": float(min(within_2_nats)),
        "within_2_nats_high_c": float(max(within_2_nats)),
        "printable": printable,
        "band_slope": slopes[0].tolist(),
        "band_slope_low": (slopes[0] - widen * (slopes[0] - p_low)).tolist(),
        "band_slope_high": (slopes[0] + widen * (p_high - slopes[0])).tolist(),
        "bootstrap_argmins": argmins[1:].tolist(),
    }


def _capacity_record(
    *, cache: FeatureCache, vintages: Any, site: str, t_all_weight: float
) -> dict[str, Any]:
    """Return the fitted latest capacity of B0 and T-all at the scored cut, with 50 resamples."""
    base = design.build_base(
        cache=cache, vintages=vintages, site=site, cut=common.SCORED_CUT, fitted=True
    )
    capacity = base.layout.slices["capacity"]
    rng = np.random.default_rng(
        [common.EXPERIMENT_ID, CAPACITY_SEED_TAG, common.SCORED_SITES.index(site)]
    )
    counts = profile.weighting_set(quarter_of_month=base.quarter_of_month, rng=rng)
    latest: dict[str, np.ndarray] = {}
    for name, arm in (
        ("b0", design.B0),
        ("t_all", design.interaction_arm(name="t_all", penalty_weight=t_all_weight)),
    ):
        x = design.arm_matrix(
            cache=cache, layout=base.layout, site=site, index=base.index, x0=base.x0, arm=arm
        )
        grams = design.MonthGrams.build(
            x=x, month_of_row=base.month_of_row, n_months=len(base.months)
        )
        month_cross = grams.cross_products(x=x, target=base.target)
        penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
        theta = design.solve(
            gram=grams.weighted(counts=counts),
            rhs=torch.as_tensor(counts) @ month_cross,
            penalty=penalty,
        )
        drop = {column for column, _ in design.replaced_columns(layout=base.layout, arm=arm)}
        kept = [j for j in range(base.x0.shape[1]) if j not in drop]
        positions = [kept.index(j) for j in range(capacity.start, capacity.stop)]
        latest[name] = theta[:, positions].sum(axis=1)
    difference = latest["t_all"] - latest["b0"]
    low, high = np.percentile(difference[1:], profile.ARGMIN_INTERVAL_PERCENTILES)
    return {
        "site": site,
        "b0_latest_capacity_points": float(latest["b0"][0]),
        "t_all_latest_capacity_points": float(latest["t_all"][0]),
        "difference_points": float(difference[0]),
        "difference_low_points": float(low),
        "difference_high_points": float(high),
        "flagged": bool(low > CAPACITY_FLAG_POINTS or high < -CAPACITY_FLAG_POINTS),
    }


def fit_site(job: tuple[str, dict[str, Any], float]) -> dict[str, Any]:
    """Fit every arm of one substation at the cuts and return frames and side tables.

    Args:
        job: The substation's label, the verified decisions, and the variance inflation `tau`.

    Returns:
        A dict with `frames` (set name to predictions), `profiles`, `capacity`, and `cards`.
    """
    site, decisions, tau = job
    cache: FeatureCache = _STATE["cache"]
    vintages = _STATE["vintages"]
    weights = {name: float(decisions["lambda"].get(name, 0.0)) for name in pipeline.ARM_NAMES}
    runs: dict[str, dict[str, pipeline.YearRun]] = {}
    for arm_name in pipeline.ARM_NAMES:
        runs[arm_name] = {}
        for label in FIT_LABELS:
            run = pipeline.fit_year(
                cache=cache,
                vintages=vintages,
                site=site,
                fit_label=label,
                arm_name=arm_name,
                penalty_weight=weights[arm_name],
            )
            if run is not None:
                runs[arm_name][label] = run
    frames: dict[str, pl.DataFrame] = {}
    for label in SCORED_LABELS:
        for arm_name in pipeline.ARM_NAMES:
            arm_runs = runs[arm_name]
            if label not in arm_runs or pipeline.PREVIOUS_LABEL[label] not in arm_runs:
                continue
            scored = pipeline.score_year(
                cache=cache,
                runs=arm_runs,
                label=label,
                hyper=_hyper(decisions=decisions, arm=arm_name, site=site),
            )
            for name, frame in pipeline.scored_frames(
                cache=cache, site=site, arm_name=arm_name, scored=scored
            ).items():
                frames[f"{label}_{name}"] = frame
            if arm_name == "t_all":
                control = pipeline.score_year(
                    cache=cache,
                    runs=arm_runs,
                    label=label,
                    hyper=_hyper(decisions=decisions, arm="b0", site=site),
                )
                frames[f"{label}_t_all_kb0"] = pipeline.scored_frames(
                    cache=cache, site=site, arm_name="t_all", scored=control
                )["t_all_k"].with_columns(arm=pl.lit("t_all_kb0"))
    profiles = []
    for label in FIT_LABELS:
        try:
            base = design.build_base(
                cache=cache, vintages=vintages, site=site, cut=common.FIT_CUTS[label]
            )
        except ValueError:
            continue
        if base.n < pipeline.MIN_TRAINING_ROWS:
            continue
        rng = np.random.default_rng(
            [
                common.EXPERIMENT_ID,
                PROFILE_SEED_TAG,
                common.SCORED_SITES.index(site),
                FIT_LABELS.index(label),
            ]
        )
        profiles.append(_profile_record(cache=cache, base=base, label=label, tau=tau, rng=rng))
    capacity = _capacity_record(
        cache=cache, vintages=vintages, site=site, t_all_weight=weights["t_all"]
    )
    cards = {
        arm_name: {label: run.fit.theta for label, run in arm_runs.items()}
        for arm_name, arm_runs in runs.items()
    }
    print(f"{site}: {len(frames)} sets, {len(profiles)} profiles", flush=True)
    return {"frames": frames, "profiles": profiles, "capacity": capacity, "cards": cards}


def main() -> None:
    """Fit all substations and write the prediction sets and result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    decisions = gate2.verified_decisions()
    gate3 = json.loads((common.WORK_DIR / "gate3.json").read_text())
    tau = float(gate3["tau_median"])
    common.load_cache()
    jobs = [(site, decisions, tau) for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        results = list(pool.map(fit_site, jobs))
    names = dict.fromkeys(name for result in results for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in results if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    records = [row for r in results for row in r["profiles"]]
    (common.WORK_DIR / "profiles.json").write_text(json.dumps(records) + "\n")
    (common.WORK_DIR / "capacity_check.json").write_text(
        json.dumps([r["capacity"] for r in results], indent=2) + "\n"
    )
    cards = {site: r["cards"] for site, r in zip(common.SCORED_SITES, results, strict=True)}
    (common.WORK_DIR / "coefficients.pkl").write_bytes(pickle.dumps(cards))
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
