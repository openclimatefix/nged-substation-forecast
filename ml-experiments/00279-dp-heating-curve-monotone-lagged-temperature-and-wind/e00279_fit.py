"""Fit experiment 279's arms at every cut and write the prediction sets.

**For each substation,** the script fits T-all (A0) and the five heating-curve arms at the four
cuts, forecasts the year after each, and writes day-ahead predictions with and without the Kalman
filter for the tuning, replicate, and scored years. T-all's penalty weight and its per-site filter
hyperparameters come from experiment 126's `decisions.json`, whose hash must still match the one
recorded there. Every arm uses T-all's filter hyperparameters, so the terms are the only
difference.

**Sets.** For an arm `a` and year `y`: `<y>_<a>` (unfiltered) and `<y>_<a>_k` (filtered).

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00279-*/e00279_fit.py [--workers 6]
"""

import argparse
import hashlib
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00279_common as common
import e00279_design as design
import e00279_pipeline as pipeline
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_pinned_capacity import read_vintages

FIT_LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning", "replicate", "scored")
SCORED_LABELS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
CURVE_GRID_C: Final[np.ndarray] = np.arange(-4.0, 18.5, 1.0)
"""The temperatures at which a network's fitted heating curve is stored for the card figure."""

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def verified_decisions() -> dict[str, Any]:
    """Return experiment 126's `decisions.json` after checking its recorded hash."""
    text = (common.DECISIONS_126_DIR / "decisions.json").read_text()
    recorded = (common.DECISIONS_126_DIR / "decisions.sha256").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != recorded:
        msg = "experiment 126's decisions.json no longer matches its recorded hash"
        raise ValueError(msg)
    return json.loads(text)


def network_curves(*, run: pipeline.YearRun, cache: FeatureCache) -> dict[str, list[float]]:
    """Return a network arm's heating power (points) against temperature, by group, at mean wind.

    Every one of the three temperatures is set to the grid value, the wind to the training mean.
    Groups are the hour band times 4 plus twice the day type plus the half of the band.
    """
    network = run.fit.network
    if network is None:
        return {}
    import e00279_nn as nn_module

    uses_wind = run.arm_name == "a3"
    wind = float(np.nanmean(design.site_aux(cache=cache, site=run.site).wind[run.index, 0]))
    out: dict[str, list[float]] = {}
    for group in range(nn_module.N_GROUPS):
        temperatures = np.repeat(CURVE_GRID_C[:, None], 3, axis=1)
        x = nn_module.inputs(
            temperatures_c=temperatures,
            wind_m_s=np.full(len(CURVE_GRID_C), wind) if uses_wind else None,
        )
        g = network.predict(x=x, group=np.full(len(CURVE_GRID_C), group))
        out[str(group)] = [float(v) for v in g]
    return out


def fit_site(job: tuple[str, dict[str, Any], tuple[str, ...]]) -> dict[str, Any]:
    """Fit every arm of one substation and return its prediction frames.

    Args:
        job: The substation's label, the verified decisions, and the scored years to write.

    Returns:
        A dict with `frames` (set name to predictions), `cards` (coefficients), and `curves`.
    """
    site, decisions, years = job
    cache: FeatureCache = _STATE["cache"]
    vintages = _STATE["vintages"]
    weight = float(decisions["lambda"]["t_all"])
    hyper = FilterHyperparameters.from_array(
        values=np.array(decisions["filter_hyper"]["t_all"][site])
    )
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
            print(f"{site} {arm_name} {label} fitted", flush=True)
    frames: dict[str, pl.DataFrame] = {}
    for label in years:
        for arm_name in pipeline.ARM_NAMES:
            arm_runs = runs[arm_name]
            if label not in arm_runs or pipeline.PREVIOUS_LABEL[label] not in arm_runs:
                continue
            scored = pipeline.score_year(cache=cache, runs=arm_runs, label=label, hyper=hyper)
            for name, frame in pipeline.scored_frames(
                cache=cache, site=site, arm_name=arm_name, scored=scored
            ).items():
                frames[f"{label}_{name}"] = frame
    cards = {
        arm_name: {label: run.fit.theta for label, run in arm_runs.items()}
        for arm_name, arm_runs in runs.items()
    }
    curves = {
        arm_name: {label: network_curves(run=run, cache=cache) for label, run in arm_runs.items()}
        for arm_name, arm_runs in runs.items()
        if arm_name in design.NETWORK_EXTRAS
    }
    print(f"{site}: {len(frames)} sets", flush=True)
    return {"frames": frames, "cards": cards, "curves": curves}


def main() -> None:
    """Fit all substations and write the prediction sets and result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--years", nargs="+", default=list(SCORED_LABELS), choices=SCORED_LABELS)
    parser.add_argument("--sites", nargs="+", default=list(common.SCORED_SITES))
    arguments = parser.parse_args()
    decisions = verified_decisions()
    common.load_cache()
    years = tuple(arguments.years)
    jobs = [(site, decisions, years) for site in arguments.sites]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        results = list(pool.map(fit_site, jobs))
    names = dict.fromkeys(name for result in results for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in results if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(years)
    cards = {site: r["cards"] for site, r in zip(arguments.sites, results, strict=True)}
    (common.WORK_DIR / f"coefficients_{suffix}.pkl").write_bytes(pickle.dumps(cards))
    curves = {site: r["curves"] for site, r in zip(arguments.sites, results, strict=True)}
    (common.WORK_DIR / f"curves_{suffix}.json").write_text(json.dumps(curves) + "\n")
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
