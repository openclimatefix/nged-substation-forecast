"""The known-answer gate of experiment 110, run after tuning and before any scored fit.

Every test injects a quantity of known size into real demand at the injection sites, fits the
model at the tuning cut (2023-07-01, never scored), and checks that the model recovers the
quantity. A failed blocking test stops scoring of the component it tests.

1. **PV from a mismatched plant (blocking for each K arm).** Synthetic PV of 10, 30, and 60 points
   of AC capacity, built from real CAMS irradiance through an east-west plant and a south-west
   plant unlike the model's fleet, is subtracted from each site's power. The fitted capacity must
   rise by the injected AC capacity within 20% at every site, level, and plant.
2. **Injected farms (blocking for each K arm).** Five real solar farms' output, scaled to 20 points
   of injected 99th percentile, is subtracted from each site's power. The median ratio of recovered
   to injected capacity over sites and farms must lie in [0.75, 1.25]. For K-iii the ratio is also
   reported divided by K-base's.
3. **The meter recovers sign, capacity, and power factor (blocking for the layer).** A farm at
   experiment 79's level 1.0 (the median absolute midday power) is subtracted, the reading becomes
   `sqrt(P^2 + Q^2)` with `Q = tan(phi) D` at power factor 0.97 and `D` from the pinned B fit, and
   L0 and LQ are fitted. LQ's balanced accuracy of flow direction must be at least 0.78, each
   site's capacity within 15% of the plain fit to the signed series, and the median recovered
   power factor within 0.01 of 0.97. Level 0.5 is reported only.
4. **The node recovers a generator (blocking for the node).** A flat generator of 10 points with
   a +5-point step, a 14-day outage, and a 1-day outage is subtracted. The step must be placed
   within 7 days and sized within 30%, the 14-day outage found, and its level within 20%. The
   1-day outage is reported only. The plan's outage is 7 days; the node's blocks are 7 days long, so
   a 7-day outage that straddles two blocks would be read at half its depth.
5. **The clip recovers an injected clip (blocking for the clip).** The model's own rooftop output
   times 1.25 (a capacity multiplier inside the profile's grid, so the truth has the model's form)
   is clipped at 60% of its peak. The profiled ceiling must be within 10% of the injected clip at 3
   or more of 4 sites, on both legs: with the clip profiled on B-D1 (as B-gen does) and on B (as
   B-clip does).

Tests 1 and 3 run at the five decision substations, and tests 2, 4, and 5 at D10, D11, D14, and
D16. Every fit uses the window of the scored arm it certifies (all years for the K arms, the meter
arms, the node, and the clip), so the gate tests the configuration the scored arms use. A farm
half-hour with no reading or a dropout makes the power `NaN`, so that row leaves training.

Only ratios leave the private folder, because the farms are metered generators.

    uv run python ml-experiments/00110-*/e00110_gate.py [--workers 4]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Final

import e00110_arms as arms
import e00110_common as common
import e00110_inject as inject
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    CLIP_BETA_PER_POINT,
    LAST_TRAINED_MIN_SHAPE,
    GamSpec,
    NodeSpec,
    derated_unit_shape,
    fit_gam,
    pinned_solar_points,
    predict_gam,
    profile_clip,
)
from studies.dp_meter import fit_gam_through_meter, predict_meter
from studies.dp_operator import tan_phi_from_power_factor
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages

CAPACITY_LEVELS: Final[tuple[float, ...]] = (10.0, 30.0, 60.0)
PLANT_SETS: Final[dict[str, tuple[Any, ...]]] = {
    "east-west": inject.EAST_WEST_PLANTS,
    "south-west": inject.SOUTH_WEST_PLANTS,
}
CLIP_SCALE: Final[float] = 1.25
"""The injected rooftop multiplier of gate test 5, inside the clip profile's grid of 0.7 to 1.3."""
FARM_POINTS: Final[float] = 20.0
"""The injected 99th percentile of a farm, in points (experiment 39's headline size)."""
POWER_FACTOR_TRUTH: Final[float] = 0.97
METER_LEVELS: Final[tuple[float, ...]] = (1.0, 0.5)
NODE_GENERATOR: Final[dict[str, Any]] = {
    "level": 10.0,
    "step": 5.0,
    "step_date": (2022, 3, 15),
    "long_outage": (2022, 9, 12, 14),
    "short_outage": (2023, 1, 17, 1),
}
CLIP_BASES: Final[tuple[str, ...]] = ("b_d1", "b")
"""The bases the clip is profiled on: B-D1 for B-gen and B for B-clip."""
CLIP_SHARE_OF_PEAK: Final[float] = 0.6
CAPACITY_TOLERANCE: Final[float] = 0.20
FARM_RATIO_BAND: Final[tuple[float, float]] = (0.75, 1.25)
DIRECTION_ACCURACY_BAR: Final[float] = 0.779
METER_CAPACITY_TOLERANCE: Final[float] = 0.15
POWER_FACTOR_TOLERANCE: Final[float] = 0.01
NODE_STEP_TOLERANCE_DAYS: Final[float] = 7.0
NODE_STEP_SIZE_TOLERANCE: Final[float] = 0.30
NODE_LEVEL_TOLERANCE: Final[float] = 0.20
CLIP_TOLERANCE: Final[float] = 0.10
K_ARMS: Final[tuple[str, ...]] = ("k_base", "k_i", "k_ii", "k_iii", "k_iiiii")

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


@lru_cache(maxsize=64)
def _plant_shape(site: str, plant_name: str) -> np.ndarray:
    return inject.plant_shape(cache=_STATE["cache"], site=site, plants=PLANT_SETS[plant_name])


@lru_cache(maxsize=64)
def _farm_norm(farm: str) -> np.ndarray:
    return inject.farm_norm(cache=_STATE["cache"], farm=farm)


def _capacity(*, cache: FeatureCache, site: str, arm: str) -> float:
    """Return a K arm's fitted capacity of the last trained season-year at the tuning cut."""
    definition: arms.ArmDefinition = _STATE["definitions"][arm]
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[definition.spec],
        cut=common.TUNING_CUT,
        window=definition.window,
        season_years=common.SEASON_YEARS,
    )
    return fit.capacity_points()[fit.last_season_year]


def _shape_p99(*, cache: FeatureCache, site: str) -> float:
    """Return the 99th percentile of the derated unit shape over the site's daylight half-hours."""
    features = cache.sites[site]
    index = np.flatnonzero(np.isfinite(features.pv_unit_shape))
    shape = derated_unit_shape(features=features, index=index)
    return float(np.percentile(shape[shape > LAST_TRAINED_MIN_SHAPE], 99))


def _mismatched_job(job: tuple[str, str]) -> list[dict[str, Any]]:
    """Gate test 1 for one site and K arm: every level and plant."""
    site, arm = job
    cache: FeatureCache = _STATE["cache"]
    base = _capacity(cache=cache, site=site, arm=arm)
    rows = []
    for plant in PLANT_SETS:
        for level in CAPACITY_LEVELS:
            injected = inject.inject_pv(
                cache=cache, site=site, shape=_plant_shape(site, plant), capacity_points=level
            )
            rise = _capacity(cache=injected, site=site, arm=arm) - base
            rows.append(
                {
                    "test": 1,
                    "site": site,
                    "arm": arm,
                    "plant": plant,
                    "injected_points": level,
                    "ratio": rise / level,
                    "passed": abs(rise / level - 1.0) <= CAPACITY_TOLERANCE,
                }
            )
    return rows


def _farm_job(job: tuple[str, str]) -> list[dict[str, Any]]:
    """Gate test 2 for one site and K arm: every farm."""
    site, arm = job
    cache: FeatureCache = _STATE["cache"]
    target = FARM_POINTS / _shape_p99(cache=cache, site=site)
    rows = []
    for farm in inject.FARMS:
        # The base fit drops the same rows as the injected fit (an injection of zero capacity
        # carries the farm's missing half-hours as NaN), so the rise is the injection alone.
        base_cache = inject.inject_pv(
            cache=cache, site=site, shape=_farm_norm(farm), capacity_points=0.0
        )
        injected = inject.inject_pv(
            cache=cache, site=site, shape=_farm_norm(farm), capacity_points=FARM_POINTS
        )
        _check_rows_remain(base=cache, injected=base_cache, site=site)
        base = _capacity(cache=base_cache, site=site, arm=arm)
        rise = _capacity(cache=injected, site=site, arm=arm) - base
        rows.append({"test": 2, "site": site, "arm": arm, "farm": farm, "ratio": rise / target})
    return rows


def _check_rows_remain(*, base: FeatureCache, injected: FeatureCache, site: str) -> None:
    """Raise if an injection leaves under half of a site's readings."""
    before = int(np.isfinite(base.sites[site].y).sum())
    after = int(np.isfinite(injected.sites[site].y).sum())
    if after < 0.5 * before:
        msg = f"an injection left {after} of {before} readings at {site}"
        raise ValueError(msg)


def _pinned_b(*, cache: FeatureCache, site: str) -> tuple[PinnedCapacity, GamSpec]:
    spec = arms.base_specs(part_4=arms.read_part_4_tuned())["b"]
    pinned = pinned_capacity(
        vintages=_STATE["vintages"], table=cache.sites[site].table, cut=common.TUNING_CUT
    )
    return pinned, spec


def _meter_job(site: str) -> list[dict[str, Any]]:
    """Gate test 3 at one site, at each injection level."""
    cache: FeatureCache = _STATE["cache"]
    pinned, spec_b = _pinned_b(cache=cache, site=site)
    fit_b = fit_gam(
        cache=cache,
        site=site,
        specs=[spec_b],
        cut=common.TUNING_CUT,
        window=_STATE["definitions"]["b"].window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    features = cache.sites[site]
    usable = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
    usable = usable[cache.end_times[usable] <= common.to_64(moment=common.TUNING_CUT)]
    demand = np.full(len(cache.end_times), np.nan)
    demand[usable] = predict_gam(cache=cache, fit=fit_b, index=usable) + pinned_solar_points(
        cache=cache, features=features, pinned=pinned, index=usable
    )
    scale = inject.midday_median_absolute(cache=cache, site=site, before=common.TUNING_CUT)
    spec = replace(spec_b, capacity="fitted")
    rows = []
    for level in METER_LEVELS:
        signed_cache = inject.inject_farm(
            cache=cache, site=site, farm="A", capacity_points=level * scale
        )
        signed = signed_cache.sites[site].y
        metered = inject.inject_meter(
            cache=signed_cache,
            site=site,
            signed=signed,
            demand=demand,
            tan_phi=tan_phi_from_power_factor(power_factor=POWER_FACTOR_TRUTH),
        )
        reference = fit_gam(
            cache=signed_cache,
            site=site,
            specs=[spec],
            cut=common.TUNING_CUT,
            window=_STATE["definitions"]["lq"].window,
            season_years=common.SEASON_YEARS,
        )
        reference_capacity = reference.capacity_points()[reference.last_season_year]
        for qmode in ("zero", "learned"):
            meter = fit_gam_through_meter(
                cache=metered,
                site=site,
                spec=spec,
                cut=common.TUNING_CUT,
                window=_STATE["definitions"]["lq"].window,
                season_years=common.SEASON_YEARS,
                qmode=qmode,
            )
            capacity = meter.fit.capacity_points()[meter.fit.last_season_year]
            index = np.flatnonzero(
                np.isfinite(signed[: len(cache.end_times)]) & np.isfinite(demand)
            )
            predicted = predict_meter(cache=metered, meter=meter, index=index)
            truth_positive = signed[index] > 0
            predicted_positive = predicted.real_power > 0
            accuracy = 0.5 * (
                float(predicted_positive[truth_positive].mean())
                + float((~predicted_positive[~truth_positive]).mean())
            )
            rows.append(
                {
                    "test": 3,
                    "site": site,
                    "level": level,
                    "qmode": qmode,
                    "reverse_flow_share": float((~truth_positive).mean()),
                    "balanced_accuracy": accuracy,
                    "capacity_ratio_to_signed": capacity / reference_capacity,
                    "power_factor": meter.power_factor,
                    "reactive": meter.reactive,
                    "starts_capacity_spread": meter.capacity_spread,
                    "converged": meter.converged,
                }
            )
    return rows


def _node_job(site: str) -> dict[str, Any]:
    """Gate test 4 at one site."""
    cache: FeatureCache = _STATE["cache"]
    tuned = _STATE["tuned"]
    spec = replace(
        arms.base_specs(part_4=arms.read_part_4_tuned())["b_d1"], node=NodeSpec(**tuned["node"])
    )
    pinned = pinned_capacity(
        vintages=_STATE["vintages"], table=cache.sites[site].table, cut=common.TUNING_CUT
    )
    generator = NODE_GENERATOR
    step_day = inject.day_of(moment=datetime(*generator["step_date"], tzinfo=UTC))
    long_first = inject.day_of(moment=datetime(*generator["long_outage"][:3], tzinfo=UTC))
    short_first = inject.day_of(moment=datetime(*generator["short_outage"][:3], tzinfo=UTC))
    long_length = generator["long_outage"][3]
    short_length = generator["short_outage"][3]
    output = inject.generator_level(
        cache=cache,
        level=generator["level"],
        step_day=step_day,
        step_size=generator["step"],
        outages=((long_first, long_length), (short_first, short_length)),
    )
    injected = inject.inject_generator(cache=cache, site=site, output=output)
    fit = fit_gam(
        cache=injected,
        site=site,
        specs=[spec],
        cut=common.TUNING_CUT,
        window=_STATE["definitions"]["b_gen"].window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    baseline = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=common.TUNING_CUT,
        window=_STATE["definitions"]["b_gen"].window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    long = inject.node_readout(
        fit=fit, step_day=step_day, outage=(long_first, long_length), baseline=baseline
    )
    short = inject.node_readout(
        fit=fit, step_day=step_day, outage=(short_first, short_length), baseline=baseline
    )
    truth_level = generator["level"] + generator["step"]
    return {
        "test": 4,
        "site": site,
        "step_error_days": long.step_error_days,
        "step_size": long.step_size,
        "long_outage_found": long.outage_found,
        "long_outage_level": long.outage_level,
        "short_outage_found": short.outage_found,
        "short_outage_level": short.outage_level,
        "truth_level": truth_level,
        "passed": (
            abs(long.step_error_days) <= NODE_STEP_TOLERANCE_DAYS
            and abs(long.step_size - generator["step"])
            <= NODE_STEP_SIZE_TOLERANCE * generator["step"]
            and long.outage_found
            and abs(long.outage_level - truth_level) <= NODE_LEVEL_TOLERANCE * truth_level
        ),
    }


def _clip_job(job: tuple[str, str]) -> dict[str, Any]:
    """Gate test 5 at one site, with the clip profiled on B-D1 (B-gen) or on B (B-clip)."""
    site, base = job
    cache: FeatureCache = _STATE["cache"]
    pinned, _ = _pinned_b(cache=cache, site=site)
    # B-gen profiles its clip on B-D1 without the node and B-clip profiles it on B, so the gate
    # runs one leg for each.
    spec = arms.base_specs(part_4=arms.read_part_4_tuned())[base]
    features = cache.sites[site]
    rows = np.flatnonzero(np.isfinite(features.y) & np.isfinite(features.pv_unit_shape))
    rooftop = np.full(len(cache.end_times), np.nan)
    rooftop[rows] = pinned_solar_points(cache=cache, features=features, pinned=pinned, index=rows)
    clipped, level = inject.clipped_total(
        pinned_pv=rooftop,
        scale=CLIP_SCALE,
        share_of_peak=CLIP_SHARE_OF_PEAK,
        beta=CLIP_BETA_PER_POINT,
        peak_rows=rows[cache.end_times[rows] <= common.to_64(moment=common.TUNING_CUT)],
    )
    demand = features.y + np.nan_to_num(rooftop, nan=0.0)
    injected = inject.with_reading(cache=cache, site=site, y=demand - clipped)
    profile = profile_clip(
        cache=injected,
        site=site,
        spec=spec,
        cut=common.TUNING_CUT,
        window=_STATE["definitions"]["b_clip"].window,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    recovered = profile.best.p_max_share * profile.best.multiplier * pinned.held_points
    return {
        "test": 5,
        "site": site,
        "base": base,
        "injected_clip_points": level,
        "recovered_clip_points": recovered,
        "ratio": recovered / level,
        "multiplier": profile.best.multiplier,
        "share": profile.best.p_max_share,
        "passed": abs(recovered / level - 1.0) <= CLIP_TOLERANCE,
    }


def _summarise(*, results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Return each test's verdict from its rows."""
    summary: dict[str, Any] = {}
    summary["test_1"] = {
        arm: all(row["passed"] for row in results["1"] if row["arm"] == arm) for arm in K_ARMS
    }
    ratios = {
        arm: float(np.median([row["ratio"] for row in results["2"] if row["arm"] == arm]))
        for arm in K_ARMS
    }
    summary["test_2"] = {
        arm: {
            "median_ratio": ratios[arm],
            "passed": FARM_RATIO_BAND[0] <= ratios[arm] <= FARM_RATIO_BAND[1],
            "ratio_to_k_base": ratios[arm] / ratios["k_base"],
        }
        for arm in K_ARMS
    }
    three = [row for row in results["3"] if row["level"] == 1.0 and row["qmode"] == "learned"]
    accuracy = np.array([row["balanced_accuracy"] for row in three])
    capacity = np.array([row["capacity_ratio_to_signed"] for row in three])
    power_factor = float(np.median([row["power_factor"] for row in three]))
    summary["test_3"] = {
        "balanced_accuracy_min": float(accuracy.min()),
        "capacity_ratio_range": [float(capacity.min()), float(capacity.max())],
        "median_power_factor": power_factor,
        "all_learned": all(row["reactive"] == "learned" for row in three),
        "passed": bool(
            all(row["reactive"] == "learned" for row in three)
            and (accuracy >= DIRECTION_ACCURACY_BAR).all()
            and (np.abs(capacity - 1.0) <= METER_CAPACITY_TOLERANCE).all()
            and abs(power_factor - POWER_FACTOR_TRUTH) <= POWER_FACTOR_TOLERANCE
        ),
    }
    summary["test_4"] = {"passed": all(row["passed"] for row in results["4"])}
    legs = {
        base: sum(row["passed"] for row in results["5"] if row["base"] == base)
        for base in CLIP_BASES
    }
    summary["test_5"] = {
        "passed": all(count >= 3 for count in legs.values()),
        "n_passed_by_base": legs,
    }
    return summary


def main() -> None:
    """Run the gate and write `work/gate.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "2"
    common.load_cache()
    sites = common.INJECTION_SITES
    decision_sites = common.DECISION_SITES
    results: dict[str, list[dict[str, Any]]] = {"1": [], "2": [], "3": [], "4": [], "5": []}
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=_init_worker) as pool:
        mismatched = [(site, arm) for site in decision_sites for arm in K_ARMS]
        farms = [(site, arm) for site in sites for arm in K_ARMS]
        for rows in pool.map(_mismatched_job, mismatched):
            results["1"] += rows
        for rows in pool.map(_farm_job, farms):
            results["2"] += rows
        for rows in pool.map(_meter_job, decision_sites):
            results["3"] += rows
        results["4"] += list(pool.map(_node_job, sites))
        results["5"] += list(
            pool.map(_clip_job, [(site, base) for site in sites for base in CLIP_BASES])
        )
    summary = _summarise(results=results)
    (common.WORK_DIR / "gate.json").write_text(
        json.dumps({"summary": summary, "rows": results}, indent=2, default=float) + "\n"
    )
    for number, rows in results.items():
        print(f"\n== gate test {number} ==")
        print(pl.DataFrame(rows))
    print("\n== gate summary ==")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
