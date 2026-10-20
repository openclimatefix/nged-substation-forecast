"""Gate 4'' of experiment 121: can the night arms find a 4-point step, with a stated power target?

Gate 4' as first written (`e00121_gate4.py`) failed with thresholds that stated no effect size and
no power target. Gate 4'' is redefined after seeing that failure and before any replicate-year or
scored-year fit. The plan's "Clarifications" section fixes its thresholds.

**Synthetic substations.** As in `e00121_gate4`: B0's real fit at the 2024-07-01 cut, plus an
injected step, plus whole donor months of real noise inflated by the tuning-year out-of-sample over
in-sample RMSE ratio. Steps of 4 and 7 points of effective capacity (percent, not MW) in
the half-hour starting 00:30: `utc` (flat, UTC clock), `local` (flat, local clock), and `cold`
(a constant times the 15.5 degree hinge of the first smoothed temperature, scaled so its mean over
the winter rows of that half-hour equals the size), and a `placebo` with no step.

**Statistic and critical value.** An arm's statistic is its unfiltered mean quantile score minus
B0's on the label's own rows (British Summer Time nights for NS-U, winter nights for NS-H and
NS-HW). The critical value is the 5th percentile of the statistic over the 100 calibration placebo
draws (draws 0 to 19 at each of the 5 substations). An arm detects a step when its statistic is
below the critical value. The false-positive rate is checked on 100 further placebo draws (draws
100 to 119).

**Pass.** (a) NS-U detects the 4-point `utc` step in at least 80% of draws and the 4-point `local` step
in at most 10%; (b) NS-HW detects the 4-point `cold` step in at least 80%; (c) NS-H detects it in at
least 80%; and the verification placebo false-positive rate is at most 8% for each
(target 5%; 100 draws carry a Monte Carlo error of about 2 points).

    OMP_NUM_THREADS=1 uv run python ml-experiments/00121-*/e00121_gate4b.py run [--workers 5]
    uv run python ml-experiments/00121-*/e00121_gate4b.py summary
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00121_common as common
import e00121_gate4 as base_gate
import numpy as np
import polars as pl

SIZES_POINTS: Final[tuple[float, ...]] = (4.0, 7.0)
KINDS: Final[tuple[str, ...]] = ("utc", "local", "cold")
ARMS: Final[tuple[str, ...]] = ("b0", "ns_u", "ns_h", "ns_hw")
CALIBRATION_DRAWS: Final[range] = range(base_gate.N_DRAWS_PER_SITE)
VERIFICATION_DRAWS: Final[range] = range(100, 100 + base_gate.N_DRAWS_PER_SITE)
GATE_4B_SEED_TAG: Final[int] = 5
"""A seed tag of its own, so the noise differs from gate 4' as first written (tag 4)."""
FALSE_POSITIVE_RATE: Final[float] = 0.05
MAX_VERIFIED_FALSE_POSITIVE: Final[float] = 0.08
MIN_POWER: Final[float] = 0.80
MAX_WRONG_CLOCK_DETECTION: Final[float] = 0.10
ROWS_OF: Final[dict[str, tuple[str, str]]] = {
    "ns_u": ("bst_night", "step_bst_utc"),
    "ns_h": ("winter_night", "step_winter_utc"),
    "ns_hw": ("winter_night", "step_winter_utc"),
}
"""Each arm's label rows and the rows of the injected half-hour that go with them."""

_STATE = base_gate._STATE


def one_draw(
    *, setup: base_gate.SiteSetup, site_position: int, draw: int, with_steps: bool
) -> dict[str, float]:
    """Run one draw (a placebo, or the placebo plus every step) and return the scores."""
    rng = np.random.default_rng([common.EXPERIMENT_ID, GATE_4B_SEED_TAG, site_position, draw])
    noise_values = setup.inflation * _STATE["pool"].draw(site=setup.site, index=setup.rows, rng=rng)
    base_mean = setup.matrices[("b0", "all")] @ setup.theta_b0
    truths: dict[str, np.ndarray] = {"placebo": np.zeros_like(base_mean)}
    if with_steps:
        for kind in KINDS:
            for size in SIZES_POINTS:
                scale = size / base_gate.INJECTED_POINTS
                truths[f"{kind}{size:g}"] = scale * setup.steps[kind]
    out: dict[str, float] = {}
    for truth, step in truths.items():
        z = base_mean + step + noise_values
        for arm_name in ARMS:
            scores = base_gate.fit_and_score(setup=setup, arm_name=arm_name, z=z)
            for subset in (
                "bst_night",
                "winter_night",
                "step_bst_utc",
                "step_bst_local",
                "step_winter_utc",
            ):
                out[f"{truth}_{arm_name}_{subset}"] = scores[subset]
    return out


def _init_worker() -> None:
    """Load the cache, vintages, and donors once per worker process."""
    base_gate._init_worker()


def _site_job(job: tuple[str, int]) -> list[dict[str, Any]]:
    """Run one substation's calibration and verification draws."""
    site, position = job
    setup = base_gate.build_setup(cache=_STATE["cache"], vintages=_STATE["vintages"], site=site)
    rows = []
    for draw in CALIBRATION_DRAWS:
        scores = one_draw(setup=setup, site_position=position, draw=draw, with_steps=True)
        rows.append({"site": site, "draw": draw, "role": "calibration", **scores})
    for draw in VERIFICATION_DRAWS:
        scores = one_draw(setup=setup, site_position=position, draw=draw, with_steps=False)
        rows.append({"site": site, "draw": draw, "role": "verification", **scores})
    print(f"{site} done", flush=True)
    return rows


def run(*, workers: int) -> None:
    """Run the draws and store them."""
    common.load_cache()
    jobs = list(enumerate(common.DECISION_SITES))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        results = list(pool.map(_site_job, [(site, position) for position, site in jobs]))
    pl.DataFrame(
        [row for site_rows in results for row in site_rows], infer_schema_length=None
    ).write_parquet(common.WORK_DIR / "gate4b_draws.parquet")


def statistic(*, frame: pl.DataFrame, truth: str, arm: str, subset: str) -> np.ndarray:
    """Return the arm's score minus B0's on a row subset, per draw."""
    return (frame[f"{truth}_{arm}_{subset}"] - frame[f"{truth}_b0_{subset}"]).to_numpy()


def summary() -> dict[str, Any]:
    """Print and return gate 4'' from the stored draws."""
    frame = pl.read_parquet(common.WORK_DIR / "gate4b_draws.parquet")
    calibration = frame.filter(pl.col("role") == "calibration")
    verification = frame.filter(pl.col("role") == "verification")
    report: dict[str, Any] = {"arms": {}}
    for arm, (label_rows, step_rows) in ROWS_OF.items():
        entry: dict[str, Any] = {}
        for rows_name, subset in (
            ("label_rows", label_rows),
            ("injected_half_hour_rows", step_rows),
        ):
            placebo = statistic(frame=calibration, truth="placebo", arm=arm, subset=subset)
            critical = float(np.percentile(placebo, 100.0 * FALSE_POSITIVE_RATE))
            checked = statistic(frame=verification, truth="placebo", arm=arm, subset=subset)
            result: dict[str, Any] = {
                "critical_value": critical,
                "verification_false_positive": float((checked < critical).mean()),
            }
            for kind in KINDS:
                for size in SIZES_POINTS:
                    draws = statistic(
                        frame=calibration, truth=f"{kind}{size:g}", arm=arm, subset=subset
                    )
                    result[f"detects_{kind}{size:g}"] = float((draws < critical).mean())
                    for site in sorted(set(calibration["site"].to_list())):
                        mask = (calibration["site"] == site).to_numpy()
                        result[f"detects_{kind}{size:g}_{site}"] = float(
                            (draws[mask] < critical).mean()
                        )
            entry[rows_name] = result
        report["arms"][arm] = entry
    label = {arm: report["arms"][arm]["label_rows"] for arm in ROWS_OF}
    verified = all(
        label[arm]["verification_false_positive"] <= MAX_VERIFIED_FALSE_POSITIVE for arm in ROWS_OF
    )
    report["a_pass"] = bool(
        label["ns_u"]["detects_utc4"] >= MIN_POWER
        and label["ns_u"]["detects_local4"] <= MAX_WRONG_CLOCK_DETECTION
        and label["ns_u"]["verification_false_positive"] <= MAX_VERIFIED_FALSE_POSITIVE
    )
    report["b_pass"] = bool(
        label["ns_hw"]["detects_cold4"] >= MIN_POWER
        and label["ns_hw"]["verification_false_positive"] <= MAX_VERIFIED_FALSE_POSITIVE
    )
    report["c_pass"] = bool(
        label["ns_h"]["detects_cold4"] >= MIN_POWER
        and label["ns_h"]["verification_false_positive"] <= MAX_VERIFIED_FALSE_POSITIVE
    )
    report["verification_ok"] = verified
    report["passed"] = bool(report["a_pass"] and report["b_pass"] and report["c_pass"])
    (common.WORK_DIR / "gate4b.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Run the draws or print the summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "summary"])
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.action == "run":
        run(workers=arguments.workers)
    summary()


if __name__ == "__main__":
    main()
