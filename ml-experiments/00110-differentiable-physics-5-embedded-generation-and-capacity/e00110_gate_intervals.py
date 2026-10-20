"""The calibration gate of the capacity intervals: do 80% intervals cover the truth 70 to 90%?

The design session's ruling after experiment 112: where a gate uses a point tolerance, it also
reports whether the interval a card or a label will print is calibrated. For each capacity arm
(K-base, K-i, K-ii, K-iii) the script makes synthetic substations whose truth is the arm's own
fit to a real decision substation at the tuning cut, adds noise, refits the arm, and asks whether
the 80% month-block bootstrap interval of the last season-year's capacity covers the truth.

- **Twenty synthetic substations per arm and noise:** the five decision substations times four
  noise draws.
- **Two kinds of noise:** independent normal noise of 5.0 points (the standard deviation of
  experiment 112's gate), and noise made of the real residuals: each calendar month's residuals
  are replaced by those of a randomly chosen month from the same quarter of the year, so the
  synthetic noise keeps the real residuals' autocorrelation and their size.
- **The interval:** the 10th and 90th percentiles of 20 stratified month-block weightings (each
  calendar quarter's months kept), widened about the median by sqrt(m / (m - 1)) for m months in
  the window (experiment 113's advice).
- **Pass:** coverage between 70 and 90% over the 20 substations, reported with the mean width and
  the mean bias. The point tolerances of the main gate stay as the pre-registered record.

The truth is the arm's own model, so the test measures the interval's calibration and not the
bias of a missing term; the main gate's mismatched plants and farms measure bias.

    uv run python ml-experiments/00110-*/e00110_gate_intervals.py [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Final

import e00110_arms as arms
import e00110_common as common
import numpy as np
from studies.dp_features import FeatureCache
from studies.dp_gam import fit_gam, predict_gam, prepare_fit
from studies.dp_pinned_capacity import read_vintages

INTERVAL_ARMS: Final[tuple[str, ...]] = ("k_base", "k_i", "k_ii", "k_iii")
N_REPLICATES: Final[int] = 4
IID_NOISE_POINTS: Final[float] = 5.0
COVERAGE_BAND: Final[tuple[float, float]] = (0.70, 0.90)
INTERVAL_QUANTILES: Final[tuple[float, float]] = (0.10, 0.90)
NOISE_KINDS: Final[tuple[str, ...]] = ("iid", "real")

_STATE: dict[str, Any] = {}


def _init_worker() -> None:
    """Load the cache, vintages, and arm definitions once per worker process, with two threads."""
    import torch

    torch.set_num_threads(2)
    _STATE["cache"] = common.load_cache()
    _STATE["vintages"] = read_vintages()
    _STATE["definitions"] = arms.arm_definitions(
        part_4=arms.read_part_4_tuned(), tuned=arms.read_tuned()
    )


def real_residual_noise(
    *, cache: FeatureCache, index: np.ndarray, residual: np.ndarray, generator: np.random.Generator
) -> np.ndarray:
    """Return noise whose calendar months are the real residuals of random months of the quarter.

    Args:
        cache: The cache, for the rows' times.
        index: The training rows.
        residual: The real residual at each row of `index`.
        generator: Draws the donor months.

    Returns:
        A noise array of the same length as `index`; a donor month shorter or longer than the
        month it replaces is tiled or cut to fit.
    """
    month = cache.end_times[index].astype("datetime64[M]")
    months, inverse = np.unique(month, return_inverse=True)
    quarter = (months.astype(np.int64) % 12) // 3
    noise = np.empty(len(index))
    for position in range(len(months)):
        donors = np.flatnonzero(quarter == quarter[position])
        donor = int(generator.choice(donors))
        source = residual[inverse == donor]
        target = inverse == position
        noise[target] = np.resize(source, int(target.sum()))
    return noise


def _job(job: tuple[str, str, int, str]) -> dict[str, Any]:
    """Make one synthetic substation, refit the arm, and score the interval's coverage."""
    arm, site, replicate, noise_kind = job
    cache: FeatureCache = _STATE["cache"]
    definition: arms.ArmDefinition = _STATE["definitions"][arm]
    cut = common.TUNING_CUT
    truth_fit = fit_gam(
        cache=cache,
        site=site,
        specs=[definition.spec],
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    _, index = prepare_fit(
        cache=cache,
        site=site,
        spec=definition.spec,
        cut=cut,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    truth = truth_fit.capacity_points()[truth_fit.last_season_year]
    predicted = predict_gam(cache=cache, fit=truth_fit, index=index)
    features = cache.sites[site]
    generator = np.random.default_rng(
        [common.BOOTSTRAP_SEED, replicate, common.DECISION_SITES.index(site)]
    )
    if noise_kind == "iid":
        noise = IID_NOISE_POINTS * generator.normal(size=len(index))
    else:
        noise = real_residual_noise(
            cache=cache, index=index, residual=features.y[index] - predicted, generator=generator
        )
    y = features.y.copy()
    y[index] = predicted + noise
    synthetic = replace(cache, sites={**cache.sites, site: replace(features, y=y)})
    result = arms.capacity_with_bootstrap(
        cache=synthetic,
        vintages=_STATE["vintages"],
        site=site,
        cut=cut,
        definition=definition,
    )
    ordered = np.sort(result.bootstrap)
    median = float(np.median(ordered))
    widen = float(np.sqrt(result.n_months / max(result.n_months - 1, 1)))
    low, high = (
        median + widen * (float(np.quantile(ordered, q)) - median) for q in INTERVAL_QUANTILES
    )
    return {
        "arm": arm,
        "site": site,
        "replicate": replicate,
        "noise": noise_kind,
        "truth": truth,
        "estimate": result.capacity,
        "low": low,
        "high": high,
        "covered": bool(low <= truth <= high),
    }


def summarise(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return coverage, width, and bias per arm and noise, with the 70 to 90% verdict."""
    summary = []
    for arm in INTERVAL_ARMS:
        for kind in NOISE_KINDS:
            part = [r for r in rows if r["arm"] == arm and r["noise"] == kind]
            if not part:
                continue
            coverage = float(np.mean([r["covered"] for r in part]))
            summary.append(
                {
                    "arm": arm,
                    "noise": kind,
                    "n_substations": len(part),
                    "coverage": coverage,
                    "mean_width_points": float(np.mean([r["high"] - r["low"] for r in part])),
                    "mean_bias_points": float(np.mean([r["estimate"] - r["truth"] for r in part])),
                    "calibrated": COVERAGE_BAND[0] <= coverage <= COVERAGE_BAND[1],
                }
            )
    return summary


def main() -> None:
    """Run the calibration gate and write `work/gate_intervals.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    jobs = [
        (arm, site, replicate, noise)
        for arm in INTERVAL_ARMS
        for noise in NOISE_KINDS
        for site in common.DECISION_SITES
        for replicate in range(N_REPLICATES)
    ]
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        rows = list(pool.map(_job, jobs))
    summary = summarise(rows=rows)
    (common.WORK_DIR / "gate_intervals.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n"
    )
    for line in summary:
        print(
            f"{line['arm']:>8} {line['noise']:>5}: coverage {line['coverage']:.2f}, "
            f"width {line['mean_width_points']:.1f}, bias {line['mean_bias_points']:+.2f}, "
            f"calibrated {line['calibrated']}"
        )


if __name__ == "__main__":
    main()
