"""Gate 3 of experiment 126: can each arm's screen see an effect of the size that would matter?

**The synthetic substations.** The truth is T-all's real fit at the 2023-07-01 cut, written as
fitted values on every row up to the end of the tuning year. For an arm, the effect is the real
fitted values minus T-all's, scaled by a factor `s` so that the *oracle gain* equals M = 0.03
points: the drop in mean quantile score on the tuning year when the forecast is the truth with the
effect instead of the truth without it, with a spread taken from the noise alone. The factor is set
on 5 pilot draws per substation, apart from the main draws.

**The noise** is whole donor months of real B0 training residuals from the other decision
substations (`e00126_noise`), on the training and the held-out rows alike, so the noise keeps its
autocorrelation and the Kalman filter has something to do. The synthetic target replaces the real
power in a copy of the feature cache, so that each arm is fitted and filtered by exactly the code
that fits and filters the real data, at the replica cut (2022-07-01) and the tuning cut
(2023-07-01). W-basis keeps the basis learned from real data.

**Pass** for an arm if the mean of arm-K minus T-all-K over the 100 draws (20 per substation, five
substations) is at most -M/2 and the 99.44% bootstrap interval over draws lies below zero.
Otherwise the arm is `not powered`.

    uv run --no-sync python ml-experiments/00126-*/e00126_gate3.py run [--workers 5]
    uv run --no-sync python ml-experiments/00126-*/e00126_gate3.py summary
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any, Final

import e00126_common as common
import e00126_design as design
import e00126_noise as noise
import e00126_pipeline as pipeline
import e00126_tune as tune
import numpy as np
import polars as pl
from studies.dp_assimilation import FilterHyperparameters
from studies.dp_features import FeatureCache
from studies.dp_gam import ResidualSpread, design_matrix, pinned_solar_points
from studies.dp_pinned_capacity import read_vintages

TARGET_GAIN_POINTS: Final[float] = 0.03
N_DRAWS_PER_SITE: Final[int] = 20
N_PILOT_DRAWS: Final[int] = 5
SCALE_BRACKET: Final[tuple[float, float]] = (0.0, 40.0)
BISECTION_STEPS: Final[int] = 14
GATE_SEED_TAG: Final[int] = 3
PILOT_DRAW_ID: Final[int] = 99
N_BOOTSTRAP: Final[int] = 5000
GATE_LEVEL: Final[float] = 99.44
ARMS: Final[tuple[str, ...]] = tuple(a for a in pipeline.ARM_NAMES if a != "t_all")

_STATE: dict[Any, Any] = {}


def _init_worker() -> None:
    """Load the cache, vintages, and donor pool once per worker process."""
    common.configure_threads()
    cache = common.load_cache()
    _STATE["cache"] = cache
    _STATE["vintages"] = read_vintages()
    _STATE["pool"] = noise.build_pool(cache=cache)


def _decisions() -> dict[str, Any]:
    """Return the hashed decisions."""
    return json.loads((common.WORK_DIR / "decisions.json").read_text())


class SiteSetup:
    """Everything about one substation that does not change from draw to draw."""

    def __init__(self, *, site: str) -> None:
        """Fit T-all and every arm on the real tuning-cut data and keep their fitted values."""
        cache: FeatureCache = _STATE["cache"]
        vintages = _STATE["vintages"]
        weight = tune.t_all_weight()
        self.site = site
        features = cache.sites[site]
        year_end = common.to_64(moment=common.year_end(cut=common.TUNING_CUT))
        self.index = np.flatnonzero(np.isfinite(features.y) & (cache.end_times <= year_end))
        base = design.build_base(cache=cache, vintages=vintages, site=site, cut=common.TUNING_CUT)
        assert base.pinned is not None
        self.solar = pinned_solar_points(
            cache=cache, features=features, pinned=base.pinned, index=self.index
        )
        x0 = design_matrix(cache=cache, features=features, layout=base.layout, index=self.index)

        def fitted(arm_name: str) -> np.ndarray:
            arm = pipeline.arm_spec(arm_name=arm_name, penalty_weight=weight)
            context = pipeline.arm_context(
                cache=cache,
                vintages=vintages,
                site=site,
                cut=common.TUNING_CUT,
                arm_name=arm_name,
            )
            fit = design.fit_arm(cache=cache, base=base, arm=arm, context=context)
            matrix = design.arm_matrix(
                cache=cache,
                layout=base.layout,
                site=site,
                index=self.index,
                x0=x0,
                arm=arm,
                context=context,
            )
            return matrix @ fit.theta

        self.truth = fitted("t_all")
        self.effect = {name: np.nan_to_num(fitted(name) - self.truth) for name in ARMS}
        test = common.scoring_index(
            cache=cache,
            site=site,
            after=common.TUNING_CUT,
            through=common.year_end(cut=common.TUNING_CUT),
        )
        self.test_position = np.searchsorted(self.index, test)
        self.test = test
        self.keep = ~features.switching_mask[test]


def synthetic_cache(*, setup: SiteSetup, z: np.ndarray) -> FeatureCache:
    """Return a copy of the cache whose power at the setup's rows is `z` minus the pinned solar."""
    cache: FeatureCache = _STATE["cache"]
    features = cache.sites[setup.site]
    y = np.full(len(features.y), np.nan)
    y[setup.index] = z - setup.solar
    sites = {**cache.sites, setup.site: replace(features, y=y)}
    return replace(cache, sites=sites)


def oracle_gain(*, setup: SiteSetup, scale: float, noise_values: np.ndarray, arm: str) -> float:
    """Return the drop in mean quantile score from forecasting with the effect."""
    cache: FeatureCache = _STATE["cache"]
    position = setup.test_position
    keep = setup.keep
    band = cache.calendar.band[setup.test]
    quarter = cache.calendar.quarter[setup.test]
    spread = ResidualSpread.fit(
        residuals=noise_values[position][keep], band=band[keep], quarter=quarter[keep], centre=True
    )
    z = setup.truth + scale * setup.effect[arm] + noise_values
    actual = z[position]
    scores = []
    for mean in (setup.truth[position], (setup.truth + scale * setup.effect[arm])[position]):
        quantiles = spread.quantiles(mean=mean, band=band, quarter=quarter)
        scores.append(
            common.as_float(
                common.quantile_score(actual=actual[keep], quantiles=quantiles[keep]).mean()
            )
        )
    return scores[0] - scores[1]


def calibrate(*, setup: SiteSetup, arm: str, pool: noise.ResidualPool, position: int) -> float:
    """Return the scale at which the pilot draws' mean oracle gain equals the target."""
    draws = [
        pool.draw(
            site=setup.site,
            index=setup.index,
            rng=np.random.default_rng(
                [common.EXPERIMENT_ID, GATE_SEED_TAG, position, PILOT_DRAW_ID, k]
            ),
        )
        for k in range(N_PILOT_DRAWS)
    ]

    def gain(scale: float) -> float:
        return float(
            np.mean([oracle_gain(setup=setup, scale=scale, noise_values=n, arm=arm) for n in draws])
        )

    low, high = SCALE_BRACKET
    if gain(high) < TARGET_GAIN_POINTS:
        return float("nan")
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if gain(middle) < TARGET_GAIN_POINTS:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _filtered_score(
    *, cache: FeatureCache, site: str, arm_name: str, weight: float, hyper: FilterHyperparameters
) -> float:
    """Fit an arm at the replica and tuning cuts of a synthetic cache; return its filtered score."""
    runs = {}
    for label in ("replica_tuning", "tuning"):
        run = pipeline.fit_year(
            cache=cache,
            vintages=_STATE["vintages"],
            site=site,
            fit_label=label,
            arm_name=arm_name,
            penalty_weight=weight,
        )
        assert run is not None
        runs[label] = run
    scored = pipeline.score_year(cache=cache, runs=runs, label="tuning", hyper=hyper)
    features = cache.sites[site]
    keep = ~features.switching_mask[scored.index]
    actual = features.y[scored.index]
    return common.as_float(
        common.quantile_score(actual=actual[keep], quantiles=scored.filtered_quantiles[keep]).mean()
    )


def _site_job(job: tuple[str, int]) -> list[dict[str, Any]]:
    """Run one substation's pilot calibration and main draws for every arm."""
    site, position = job
    decisions = _decisions()
    weight = tune.t_all_weight()
    setup = SiteSetup(site=site)
    pool: noise.ResidualPool = _STATE["pool"]
    scales = {arm: calibrate(setup=setup, arm=arm, pool=pool, position=position) for arm in ARMS}
    print(f"{site}: scales {json.dumps({k: round(v, 3) for k, v in scales.items()})}", flush=True)

    def hyper(arm: str) -> FilterHyperparameters:
        return FilterHyperparameters.from_array(
            values=np.array(decisions["filter_hyper"][arm][site])
        )

    rows: list[dict[str, Any]] = []
    for draw in range(N_DRAWS_PER_SITE):
        rng = np.random.default_rng([common.EXPERIMENT_ID, GATE_SEED_TAG, position, draw])
        noise_values = pool.draw(site=site, index=setup.index, rng=rng)
        t_all_score = None
        for arm in ARMS:
            scale = scales[arm]
            if not np.isfinite(scale):
                rows.append({"site": site, "draw": draw, "arm": arm, "difference": None})
                continue
            z = setup.truth + scale * setup.effect[arm] + noise_values
            cache = synthetic_cache(setup=setup, z=z)
            t_all_score = _filtered_score(
                cache=cache, site=site, arm_name="t_all", weight=weight, hyper=hyper("t_all")
            )
            arm_score = _filtered_score(
                cache=cache, site=site, arm_name=arm, weight=weight, hyper=hyper(arm)
            )
            rows.append(
                {
                    "site": site,
                    "draw": draw,
                    "arm": arm,
                    "difference": arm_score - t_all_score,
                    "oracle": oracle_gain(
                        setup=setup, scale=scale, noise_values=noise_values, arm=arm
                    ),
                }
            )
        print(f"{site} draw {draw} done", flush=True)
    return rows


def run(*, workers: int) -> None:
    """Calibrate the scales, run the draws, and store them."""
    common.load_cache()
    jobs = [(site, position) for position, site in enumerate(common.DECISION_SITES)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        results = list(pool.map(_site_job, jobs))
    frame = pl.DataFrame([row for site_rows in results for row in site_rows])
    frame.write_parquet(common.WORK_DIR / "gate3_draws.parquet")


def summary() -> dict[str, Any]:
    """Print and store each arm's gate-3 verdict."""
    frame = pl.read_parquet(common.WORK_DIR / "gate3_draws.parquet")
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    half = (100.0 - GATE_LEVEL) / 2.0
    report: dict[str, Any] = {}
    for arm in ARMS:
        part = frame.filter((pl.col("arm") == arm) & pl.col("difference").is_not_null())
        if part.height == 0:
            report[arm] = {"powered": False, "reason": "no scale reaches the target gain"}
            continue
        difference = part["difference"].to_numpy()
        resampled = rng.choice(difference, size=(N_BOOTSTRAP, len(difference))).mean(axis=1)
        low, high = np.percentile(resampled, [half, 100.0 - half])
        mean = float(difference.mean())
        report[arm] = {
            "mean_difference": mean,
            "low": float(low),
            "high": float(high),
            "draws": len(difference),
            "mean_oracle_gain": common.as_float(part["oracle"].mean()),
            "powered": bool(mean <= -TARGET_GAIN_POINTS / 2.0 and high < 0.0),
        }
    (common.WORK_DIR / "gate3.json").write_text(json.dumps(report, indent=2) + "\n")
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
