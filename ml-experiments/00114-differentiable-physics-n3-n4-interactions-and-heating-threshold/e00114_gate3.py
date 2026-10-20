"""Gate 3 of experiment 114: can the profile rule recover a heating threshold at real noise?

**The synthetic substations.** The truth is B0's real fit at the 2024-07-01 cut with the heating
block replaced by a hinge at a true threshold drawn from U(12.5, 18.5) degrees C, whose slopes are
B0's heating slopes times a factor from U(0.7, 1.3) per hour band. The noise is whole donor months
of the other decision substations' B0 residuals (`e00114_noise`). Twenty draws per decision
substation (100 in all) test recovery, and four more per substation (20 in all) put the truth at
15.5 as a placebo.

**The rule under test is H-site's:** the threshold is the arg-min over the grid of the unpenalised
training loss, and its interval is the 10th to 90th percentile of the arg-mins of 50
quarter-stratified month-block resamples, widened by 0.25 degrees C each way.

**The gate (`identified`)** holds if all of these hold:

- (a) the interval covers the true threshold in 70 to 90% of the 100 draws;
- (b) the median interval width is at most 3 degrees C;
- (c) among the draws whose truth lies 2 degrees C or more from 15.5, the arg-min is within 1 degree
  C of the truth in at least 60%.

Otherwise the label is `not identified`. **The placebo** passes if the arg-min lies within 1
degree C of 15.5 in at least 70% of its 20 draws. **Tau** is the median over draws of the
bootstrap variance of the band heating slopes at 15.5 degrees C over their model-based variance.

    uv run python ml-experiments/00114-*/e00114_gate3.py run [--workers 5]
    uv run python ml-experiments/00114-*/e00114_gate3.py summary
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final

import e00114_common as common
import e00114_design as design
import e00114_noise as noise
import e00114_profile as profile
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import N_BANDS
from studies.dp_pinned_capacity import read_vintages

N_DRAWS_PER_SITE: Final[int] = 20
N_PLACEBO_PER_SITE: Final[int] = 4
TRUE_THRESHOLD_RANGE_C: Final[tuple[float, float]] = (12.5, 18.5)
SLOPE_FACTOR_RANGE: Final[tuple[float, float]] = (0.7, 1.3)
FAR_FROM_BASE_C: Final[float] = 2.0
NEAR_C: Final[float] = 1.0
COVERAGE_RANGE: Final[tuple[float, float]] = (0.70, 0.90)
MAX_MEDIAN_WIDTH_C: Final[float] = 3.0
MIN_FAR_RECOVERY: Final[float] = 0.60
MIN_PLACEBO_RECOVERY: Final[float] = 0.70
GATE_SEED_TAG: Final[int] = 3

_STATE: dict[str, Any] = {}


def results_path() -> object:
    """Return the file that holds all gate 3 draws."""
    return common.WORK_DIR / "gate3_draws.parquet"


def _init_worker() -> None:
    """Load the cache, the vintages, and the donors' residuals once per worker process."""
    common.configure_threads()
    cache: FeatureCache = common.load_cache()
    _STATE["cache"] = cache
    _STATE["vintages"] = read_vintages()
    _STATE["pool"] = noise.build_pool(cache=cache)


def truth_target(
    *,
    cache: FeatureCache,
    base: design.BaseFit,
    theta_b0: np.ndarray,
    true_threshold_c: float,
    band_factors: np.ndarray,
    noise_values: np.ndarray,
) -> np.ndarray:
    """Return the synthetic target: B0's other terms, a hinge at the truth, and the noise.

    Args:
        cache: The feature cache.
        base: B0's inputs at the cut.
        theta_b0: B0's real coefficients.
        true_threshold_c: The true heating threshold.
        band_factors: The factor on B0's heating slope in each hour band.
        noise_values: The noise per training row.

    Returns:
        The target per training row, in points.
    """
    heating = base.layout.slices["heating"]
    other = base.x0.copy()
    other[:, heating] = 0.0
    slopes = theta_b0[heating] * np.tile(band_factors, len(theta_b0[heating]) // N_BANDS)
    hinge = design.hinge_block(
        cache=cache, site=base.site, index=base.index, threshold_c=true_threshold_c
    )
    return other @ theta_b0 + hinge @ slopes + noise_values


def _site_draws(site: str) -> list[dict[str, Any]]:
    """Run every gate 3 draw of one substation and return one row per draw."""
    cache: FeatureCache = _STATE["cache"]
    pool: noise.ResidualPool = _STATE["pool"]
    base = design.build_base(
        cache=cache, vintages=_STATE["vintages"], site=site, cut=common.REPLICATE_CUT
    )
    theta_b0 = design.fit_arm(cache=cache, base=base, arm=design.B0).theta
    union = profile.build_union(cache=cache, base=base)
    site_position = common.DECISION_SITES.index(site)
    rows: list[dict[str, Any]] = []
    plan = [("main", d) for d in range(N_DRAWS_PER_SITE)] + [
        ("placebo", N_DRAWS_PER_SITE + d) for d in range(N_PLACEBO_PER_SITE)
    ]
    for kind, draw in plan:
        rng = np.random.default_rng([common.EXPERIMENT_ID, GATE_SEED_TAG, site_position, draw])
        truth = (
            common.BASE_THRESHOLD_C
            if kind == "placebo"
            else float(rng.uniform(*TRUE_THRESHOLD_RANGE_C))
        )
        factors = rng.uniform(*SLOPE_FACTOR_RANGE, size=N_BANDS)
        values = pool.draw(site=site, index=base.index, rng=rng)
        target = truth_target(
            cache=cache,
            base=base,
            theta_b0=theta_b0,
            true_threshold_c=truth,
            band_factors=factors,
            noise_values=values,
        )
        counts = profile.weighting_set(quarter_of_month=base.quarter_of_month, rng=rng)
        result = profile.run_profile(union=union, counts=counts, target=target)
        point = float(result.argmin_thresholds()[0])
        low, high = profile.argmin_interval(argmins=result.argmin_thresholds()[1:])
        sigma_squared = float(result.loss[0].min()) / base.n
        inflation = profile.variance_inflation(
            union=union, result=result, noise_variance=sigma_squared
        )
        rows.append(
            {
                "site": site,
                "kind": kind,
                "draw": draw,
                "true_threshold_c": truth,
                "argmin_c": point,
                "low_c": low,
                "high_c": high,
                "covers": bool(low <= truth <= high),
                "width_c": high - low,
                "tau": inflation,
                "noise_sd": float(values.std()),
            }
        )
        print(
            f"{site} {kind} {draw}: truth {truth:.2f} argmin {point:.1f} [{low:.2f}, {high:.2f}]",
            flush=True,
        )
    return rows


def run(*, workers: int) -> None:
    """Run all draws in parallel and store them."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        results = list(pool.map(_site_draws, common.DECISION_SITES))
    frame = pl.DataFrame([row for site_rows in results for row in site_rows])
    frame.write_parquet(common.WORK_DIR / "gate3_draws.parquet")


def summary() -> dict[str, Any]:
    """Print and return gate 3's criteria from the stored draws."""
    frame = pl.read_parquet(common.WORK_DIR / "gate3_draws.parquet")
    main = frame.filter(pl.col("kind") == "main")
    placebo = frame.filter(pl.col("kind") == "placebo")
    coverage = common.as_float(main["covers"].mean())
    median_width = common.as_float(main["width_c"].median())
    far = main.filter(
        (pl.col("true_threshold_c") - common.BASE_THRESHOLD_C).abs() >= FAR_FROM_BASE_C
    )
    far_recovery = common.as_float(
        ((far["argmin_c"] - far["true_threshold_c"]).abs() <= NEAR_C).mean()
    )
    placebo_recovery = common.as_float(
        ((placebo["argmin_c"] - common.BASE_THRESHOLD_C).abs() <= NEAR_C).mean()
    )
    checks = {
        "a_coverage": coverage,
        "a_pass": COVERAGE_RANGE[0] <= coverage <= COVERAGE_RANGE[1],
        "b_median_width_c": median_width,
        "b_pass": median_width <= MAX_MEDIAN_WIDTH_C,
        "c_far_draws": far.height,
        "c_far_recovery": far_recovery,
        "c_pass": far_recovery >= MIN_FAR_RECOVERY,
        "placebo_recovery": placebo_recovery,
        "placebo_pass": placebo_recovery >= MIN_PLACEBO_RECOVERY,
        "tau_median": common.as_float(main["tau"].median()),
        "tau_by_site": {
            site: common.as_float(main.filter(pl.col("site") == site)["tau"].median())
            for site in common.DECISION_SITES
        },
        "n_main": main.height,
        "n_placebo": placebo.height,
    }
    checks["identified"] = bool(checks["a_pass"] and checks["b_pass"] and checks["c_pass"])
    checks["label"] = "identified" if checks["identified"] else "not identified"
    (common.WORK_DIR / "gate3.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    return checks


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
