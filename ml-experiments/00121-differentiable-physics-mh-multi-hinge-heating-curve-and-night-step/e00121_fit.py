"""Fit experiment 121's arms at every cut and write the prediction sets.

**For each substation and arm,** the script fits the frozen model at the four cuts, forecasts the
year after each, and writes day-ahead predictions with and without the Kalman filter for the
tuning, replicate, and scored years. The shrinkage multiplier and the filters' hyperparameters
come from `work/decisions.json`, whose hash must still match the one recorded before the gates.

**Arms.** `b0`, `mh`, `mh5`, MH-sh at the chosen multiplier, `ns_l`, `ns_u`, `ns_h`, and `ns_hw`.
The filtered version of an arm has the suffix `_k`. Every arm except B0 and MH-5 is also run
through B0's filter hyperparameters (suffix `_kb0`, the filter-fit control).

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        uv run python ml-experiments/00121-*/e00121_fit.py [--workers 6]
"""

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00121_common as common
import e00121_gate2 as gate2
import e00121_pipeline as pipeline
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_pinned_capacity import read_vintages

FIT_LABELS: Final[tuple[str, ...]] = ("replica_tuning", "tuning", "replicate", "scored")
SCORED_LABELS: Final[tuple[str, ...]] = ("tuning", "replicate", "scored")
NO_CONTROL_ARMS: Final[tuple[str, ...]] = ("b0", "mh5")

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache and vintages once per worker process, with one thread."""
    common.configure_threads()
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()


def arm_names(*, decisions: dict[str, Any]) -> list[str]:
    """Return the fitted arms, with MH-sh at the chosen multiplier."""
    shrink = float(decisions["shrink"])
    return ["b0", "mh", "mh5", f"mh_sh{shrink:g}", "ns_l", "ns_u", "ns_h", "ns_hw", "ns_ut"]


def _hyper(*, decisions: dict[str, Any], arm: str, site: str) -> FilterHyperparameters:
    """Return an arm's filter hyperparameters at a site, from the hashed decisions."""
    return FilterHyperparameters.from_array(values=np.array(decisions["filter_hyper"][arm][site]))


def fit_site(job: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    """Fit every arm of one substation at the cuts and return frames and coefficients.

    Args:
        job: The substation's label and the verified decisions.

    Returns:
        A dict with `frames` (set name to predictions) and `cards` (coefficients by arm and cut).
    """
    site, decisions = job
    cache = _STATE["cache"]
    vintages = _STATE["vintages"]
    runs: dict[str, dict[str, pipeline.YearRun]] = {}
    for arm_name in arm_names(decisions=decisions):
        runs[arm_name] = {}
        for label in FIT_LABELS:
            run = pipeline.fit_year(
                cache=cache, vintages=vintages, site=site, fit_label=label, arm_name=arm_name
            )
            if run is not None:
                runs[arm_name][label] = run
    frames: dict[str, pl.DataFrame] = {}
    for label in SCORED_LABELS:
        for arm_name, arm_runs in runs.items():
            if label not in arm_runs or pipeline.PREVIOUS_LABEL[label] not in arm_runs:
                continue
            own_hyper = _hyper(decisions=decisions, arm=arm_name, site=site)
            scored = pipeline.score_year(cache=cache, runs=arm_runs, label=label, hyper=own_hyper)
            for name, frame in pipeline.scored_frames(
                cache=cache, site=site, arm_name=arm_name, scored=scored
            ).items():
                frames[f"{label}_{name}"] = frame
            if arm_name in NO_CONTROL_ARMS:
                continue
            control = pipeline.score_year(
                cache=cache,
                runs=arm_runs,
                label=label,
                hyper=_hyper(decisions=decisions, arm="b0", site=site),
            )
            frames[f"{label}_{arm_name}_kb0"] = pipeline.scored_frames(
                cache=cache, site=site, arm_name=arm_name, scored=control
            )[f"{arm_name}_k"].with_columns(arm=pl.lit(f"{arm_name}_kb0"))
    cards = {
        arm_name: {label: run.fit.theta for label, run in arm_runs.items()}
        for arm_name, arm_runs in runs.items()
    }
    print(f"{site}: {len(frames)} sets", flush=True)
    return {"frames": frames, "cards": cards}


def main() -> None:
    """Fit all substations and write the prediction sets and coefficients."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    decisions = gate2.verified_decisions()
    common.load_cache()
    jobs = [(site, decisions) for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        results = list(pool.map(fit_site, jobs))
    names = dict.fromkeys(name for result in results for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in results if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    cards = {site: r["cards"] for site, r in zip(common.SCORED_SITES, results, strict=True)}
    (common.WORK_DIR / "coefficients.pkl").write_bytes(pickle.dumps(cards))
    (common.WORK_DIR / "fit_arms.json").write_text(
        json.dumps(arm_names(decisions=decisions)) + "\n"
    )
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
