"""Fit experiment 106's convex-model grid at the replicate and scored cuts and write the sets.

One frozen fit per arm, substation, and cut, with the capacity pinned, scored on the year after
the cut. The uncorrected arms take their spread from the replica. The filter arms add the
recent-level correction of `studies.dp_assimilation` and take their spread from the replica run
through the same correction. Sets are named `<cut>_<arm>` for the day-ahead forecast (leads of 15.5
to 39 hours) and `<cut>_<arm>_d<k>` for the per-lead curves of the Kalman arm A3, the exponentially
weighted mean A4, and the Kalman arm without drift A2.

Filter arms (the base arm's hyperparameters from `tuned.json`):

- `a2`: no drift, Kalman; `a3`: D1, Kalman; `a3s`: D1 plus steps, Kalman
- `a4`: D1, exponentially weighted mean; `a4_0`: no drift, exponentially weighted mean
- `a3_level`, `a3_anomaly`: A3 with only the level, or only the transient, in the correction
- `wk`: the best window with the Kalman filter; `a3_fc`: A3 with the capacity fitted

Substations are fitted in parallel worker processes with one thread each.

    uv run python ml-experiments/00106-*/e00106_grid.py [--workers 6]
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import e00106_common as common
import e00106_g as g
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters, post_filter_lead_spread
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS
from studies.dp_pinned_capacity import read_vintages

CUT_LABELS: Final[tuple[str, ...]] = ("replicate", "scored")
STATIC_ARMS: Final[tuple[str, ...]] = (
    "a0",
    "a1",
    "a1s",
    "w",
    "a1_fc",
    "a1_cov",
    "a1_smooth",
    "a1s_smooth",
    "a1_knot_shrunk25",
    "a1_knot_shrunk50",
    "a1_knot_shrunk75",
    "a1_knot_pinned",
)
LEAD_CURVE_ARMS: Final[tuple[str, ...]] = ("a2", "a3", "a4")
MAX_DAYS_AHEAD: Final[int] = 7


@dataclass(frozen=True)
class FilterArm:
    """A recent-level filter arm.

    Attributes:
        base: The static arm whose frozen forecast the filter corrects.
        kind: Which correction.
        hyper_from: The base arm whose filter hyperparameters the arm uses.
        part: Which hidden states of the Kalman filter enter the correction.
    """

    base: str
    kind: g.FilterKindType
    hyper_from: str
    part: g.FilterPartType = "both"


FILTER_ARMS: Final[dict[str, FilterArm]] = {
    "a2": FilterArm("a0", "kalman", "a0"),
    "a3": FilterArm("a1", "kalman", "a1"),
    "a3s": FilterArm("a1s", "kalman", "a1s"),
    "a4": FilterArm("a1", "ewma", "a1"),
    "a4_0": FilterArm("a0", "ewma", "a0"),
    "a3_level": FilterArm("a1", "kalman", "a1", "level"),
    "a3_anomaly": FilterArm("a1", "kalman", "a1", "anomaly"),
    "wk": FilterArm("w", "kalman", "w"),
    "a3_fc": FilterArm("a1_fc", "kalman", "a1"),
}


def fit_site(*, site: str) -> dict[str, Any]:
    """Fit every arm of one substation at both cuts and return frames and side tables.

    Args:
        site: The substation's label.

    Returns:
        A dict with `frames` (set name to predictions), `capacities`, and `drift` rows.
    """
    cache: FeatureCache = common.load_cache()
    vintages = read_vintages()
    tuned = g.read_tuned()
    definitions = g.arm_definitions(tuned=tuned)
    frames: dict[str, pl.DataFrame] = {}
    capacities: list[dict] = []
    drift: list[dict] = []
    for label in CUT_LABELS:
        cut = common.CUTS[label]
        runs = {
            name: g.run_site(
                cache=cache, vintages=vintages, site=site, cut=cut, definition=definitions[name]
            )
            for name in dict.fromkeys([*STATIC_ARMS, *(arm.base for arm in FILTER_ARMS.values())])
        }
        for name in STATIC_ARMS:
            run = runs[name]
            frames[f"{label}_{name}"] = common.prediction_frame(
                cache=cache,
                site=site,
                index=run.index,
                quantiles=g.uncorrected_quantiles(cache=cache, run=run),
                mean=run.mean,
                arm=name,
            )
            capacities += _capacity_rows(cache=cache, run=run, arm=name)
            drift += _drift_rows(run=run, arm=name)
        for name, arm in FILTER_ARMS.items():
            run = runs[arm.base]
            hyper = FilterHyperparameters.from_array(
                values=np.array(tuned["filter_hyper"][arm.hyper_from][site])
            )
            spread = post_filter_lead_spread(
                end_times=cache.end_times,
                replica=run.replica,
                run=g.trace_builder(kind=arm.kind, hyper=hyper, part=arm.part),
            )
            days = range(1, MAX_DAYS_AHEAD + 1) if name in LEAD_CURVE_ARMS else (1,)
            for days_ahead in days:
                result = g.filtered_forecast(
                    cache=cache,
                    run=run,
                    kind=arm.kind,
                    hyper=hyper,
                    days_ahead=days_ahead,
                    lead_spread=spread,
                    part=arm.part,
                )
                suffix = "" if days_ahead == 1 else f"_d{days_ahead}"
                frames[f"{label}_{name}{suffix}"] = common.prediction_frame(
                    cache=cache,
                    site=site,
                    index=run.index,
                    quantiles=result.quantiles,
                    mean=result.corrected_mean,
                    arm=name,
                    days_ahead=days_ahead,
                    level=result.correction.level,
                    anomaly=result.correction.anomaly,
                )
        print(f"{site} {label}: {len(runs)} fits, {len(frames)} sets so far", flush=True)
    return {"frames": frames, "capacities": capacities, "drift": drift}


def _capacity_rows(*, cache: FeatureCache, run: g.SiteRun, arm: str) -> list[dict]:
    """Return the capacity a run used, in points: the pinned value, or the fitted season-years."""
    effective = cache.sites[run.site].table.effective_capacity_mw
    label = next(name for name, cut in common.CUTS.items() if cut == run.cut)
    if run.pinned is not None:
        return [
            {
                "site": run.site,
                "cut": label,
                "arm": arm,
                "kind": "pinned",
                "season_year": 0,
                "source": run.pinned.source,
                "capacity_points": run.pinned.held_points,
                "capacity_mw": run.pinned.held_points * effective / 100.0,
            }
        ]
    return [
        {
            "site": run.site,
            "cut": label,
            "arm": arm,
            "kind": "fitted",
            "season_year": year,
            "source": "fit",
            "capacity_points": points,
            "capacity_mw": points * effective / 100.0,
        }
        for year, points in run.fit.capacity_points().items()
    ]


def _drift_rows(*, run: g.SiteRun, arm: str) -> list[dict]:
    """Return the drift path of a run as rows, one per hour band and knot."""
    layout = run.fit.layout
    if "drift" not in layout.slices:
        return []
    n_knots = len(layout.knot_days)
    drift = run.fit.theta[0, layout.slices["drift"]].reshape(N_BANDS, n_knots)
    steps = np.zeros((N_BANDS, n_knots))
    if "steps" in layout.slices:
        block = run.fit.theta[0, layout.slices["steps"]]
        signed = (block[: block.size // 2] - block[block.size // 2 :]).reshape(N_BANDS, n_knots - 1)
        steps[:, 1:] = np.cumsum(signed, axis=1)
    label = next(name for name, cut in common.CUTS.items() if cut == run.cut)
    return [
        {
            "site": run.site,
            "cut": label,
            "arm": arm,
            "band": band,
            "day": float(layout.knot_days[knot]),
            "drift": float(drift[band, knot]),
            "step_level": float(steps[band, knot]),
        }
        for band in range(N_BANDS)
        for knot in range(n_knots)
    ]


def _worker(site: str) -> dict[str, Any]:
    """Run `fit_site` in a worker process with one thread."""
    import torch

    torch.set_num_threads(1)
    return fit_site(site=site)


def main() -> None:
    """Fit all substations and write the prediction sets and result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_worker, common.SCORED_SITES))
    names = dict.fromkeys(name for result in results for name in result["frames"])
    for name in names:
        frame = pl.concat([r["frames"][name] for r in results if name in r["frames"]])
        common.store_arm(name=name, frame=frame)
    pl.DataFrame([row for r in results for row in r["capacities"]]).write_parquet(
        common.WORK_DIR / "grid_capacities.parquet"
    )
    pl.DataFrame([row for r in results for row in r["drift"]]).write_parquet(
        common.WORK_DIR / "grid_drift.parquet"
    )
    print(f"wrote {len(names)} prediction sets")


if __name__ == "__main__":
    main()
