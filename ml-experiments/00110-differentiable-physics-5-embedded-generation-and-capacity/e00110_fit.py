"""Fit experiment 110's arms at the replicate and scored cuts and write their predictions.

One frozen fit per arm, substation, and cut. Each arm writes two prediction sets' worth of rows in
one set called `<cut>_<arm>`: the unfiltered forecast (the frozen mean plus the replica's centred
residual quantiles) and the filtered forecast (the Kalman filter's correction added to the mean,
with the replica run through the same filter for the spread), distinguished by the `filtered`
column. The filters' hyperparameters come from `e00110_tune.py`.

A site with no usable training rows at a cut is skipped and listed in `work/fit_skipped.json`.

    uv run python ml-experiments/00110-*/e00110_fit.py --cuts replicate scored [--arms b b_d1] \
        [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import e00110_arms as arms
import e00110_common as common
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_gam import generator_evidence, generator_path
from studies.dp_pinned_capacity import read_vintages

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache, vintages, and tuned settings once per worker process, with two threads."""
    import torch

    torch.set_num_threads(2)
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()
    tuned = arms.read_tuned()
    _STATE["tuned"] = tuned
    _STATE["definitions"] = arms.arm_definitions(part_4=arms.read_part_4_tuned(), tuned=tuned)


def _job(job: tuple[str, str, str]) -> dict[str, Any]:
    """Fit one arm at one site and cut; return its filtered and unfiltered prediction rows."""
    label, arm, site = job
    cache: FeatureCache = _STATE["cache"]
    definition: arms.ArmDefinition = _STATE["definitions"][arm]
    try:
        run = arms.run_site(
            cache=cache,
            vintages=_STATE["vintages"],
            site=site,
            cut=common.CUTS[label],
            definition=definition,
        )
    except ValueError as error:
        if "no training row" not in str(error):
            raise
        return {"job": job, "skipped": str(error), "frame": None}
    extra: dict[str, np.ndarray] = {"dark": cache.sites[site].dark[run.index]}
    extra.update(run.extra_columns)
    unfiltered = common.prediction_frame(
        cache=cache,
        site=site,
        index=run.index,
        quantiles=arms.uncorrected_quantiles(cache=cache, run=run),
        mean=run.mean,
        arm=arm,
        filtered=False,
        extra=extra,
    )
    hyper = FilterHyperparameters.from_array(
        values=np.array(_STATE["tuned"]["filter_hyper"][arm][site])
    )
    forecast = arms.filtered_forecast(cache=cache, run=run, hyper=hyper)
    filtered = common.prediction_frame(
        cache=cache,
        site=site,
        index=run.index,
        quantiles=forecast.quantiles,
        mean=forecast.corrected_mean,
        arm=arm,
        filtered=True,
        extra={**extra, "level": forecast.correction.level, "anomaly": forecast.correction.anomaly},
    )
    unfiltered = unfiltered.with_columns(
        level=pl.lit(0.0, dtype=pl.Float32), anomaly=pl.lit(0.0, dtype=pl.Float32)
    )
    return {
        "job": job,
        "skipped": None,
        "frame": pl.concat([unfiltered, filtered]),
        "details": _details(run=run),
    }


def _details(*, run: arms.SiteRun) -> dict[str, Any]:
    """Return the fitted quantities the report and cards need, as plain numbers."""
    details: dict[str, Any] = {"n_training_rows": run.fit.n_rows}
    if run.spec.capacity == "fitted":
        details["capacity"] = run.fit.capacity_points()
    if run.pinned is not None:
        details["pinned_held"] = run.pinned.held_points
    if run.spec.clip is not None:
        details["clip"] = {
            "multiplier": run.spec.clip.multiplier,
            "p_max_share": run.spec.clip.p_max_share,
        }
    if run.meter is not None:
        details["meter"] = {
            "tan_phi": run.meter.tan_phi,
            "q0": run.meter.q0,
            "power_factor": run.meter.power_factor,
            "reactive": run.meter.reactive,
            "reverse_flow_share": run.meter.reverse_flow_share,
            "converged": run.meter.converged,
            "loss": run.meter.loss,
            "starts": [
                {"name": s.name, "loss": s.loss, "capacity": s.final_capacity, "tan_phi": s.tan_phi}
                for s in run.meter.starts
            ],
        }
    if run.spec.node is not None:
        path = generator_path(fit=run.fit)
        details["node"] = {
            "first_day": run.fit.layout.node_first_day,
            "step_days": run.spec.node.step_days,
            "evidence": generator_evidence(path=path),
            "path": path.tolist(),
        }
    return details


def _check_gate(*, ignore: bool) -> None:
    """Refuse to fit the scored arms before the gate has run, unless told to ignore it.

    A failed blocking test stops scoring of the component it tests, so the gate's summary must
    exist; which tests passed is read by the report, and the plan's rule decides what is scored.
    """
    path = common.WORK_DIR / "gate.json"
    if ignore:
        return
    if not path.exists():
        msg = "the known-answer gate has not run: work/gate.json is missing"
        raise SystemExit(msg)
    summary = json.loads(path.read_text())["summary"]
    failed = [name for name in ("test_3", "test_4", "test_5") if not summary[name]["passed"]]
    if failed:
        print(f"WARNING: blocking tests failed: {failed}; their components count as failed")


def main() -> None:
    """Fit the arms and write one prediction set per cut and arm."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuts", nargs="+", default=["replicate", "scored"])
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ignore-gate", action="store_true")
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    _check_gate(ignore=arguments.ignore_gate)
    common.load_cache()
    definitions = arms.arm_definitions(part_4=arms.read_part_4_tuned(), tuned=arms.read_tuned())
    wanted = arguments.arms or list(definitions)
    skipped: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        for label in arguments.cuts:
            for arm in wanted:
                jobs = [(label, arm, site) for site in definitions[arm].sites]
                frames = []
                for result in pool.map(_job, jobs):
                    if result["frame"] is None:
                        skipped.append({"job": result["job"], "reason": result["skipped"]})
                        continue
                    frames.append(result["frame"])
                    details[f"{label}/{arm}/{result['job'][2]}"] = result["details"]
                if not frames:
                    continue
                is_mva = definitions[arm].sites == common.MVA_SITES
                folder = common.store_arm(
                    name=f"{label}_{arm}",
                    frame=pl.concat(frames),
                    row_set_id=common.MVA_ROW_SET_ID if is_mva else common.ROW_SET_ID,
                )
                print(f"wrote {folder}")
    skipped_path = common.WORK_DIR / "fit_skipped.json"
    details_path = common.WORK_DIR / "fit_details.json"
    if skipped_path.exists():
        skipped = json.loads(skipped_path.read_text()) + skipped
    if details_path.exists():
        details = {**json.loads(details_path.read_text()), **details}
    skipped_path.write_text(json.dumps(skipped, indent=2) + "\n")
    details_path.write_text(json.dumps(details, default=float) + "\n")


if __name__ == "__main__":
    main()
