"""The known-answer gate of experiment 115, run on the tuning months before any replicate or scored run.

Five tests, each able to fail. A failed blocking reading stops scoring, and the verdict is capped at
`inconclusive` if it cannot be fixed inside the experiment. The tests run on real streams at the five
decision substations:

1. **An injected persistent profile change is recovered** (`test_recovery`).
2. **No false shape on clean data** (`test_clean`): permuted real streams and synthetic streams.
3. **Reproduction** (`test_reproduction`): with no shape state, KW is GAM-K as experiment 111 stored it.
4. **Causality** (`test_causality`): readings and ensemble runs after the cut change nothing.
5. **Zero mean** (`test_zero_mean`): the shape sums to zero in its blocks after every update.

    uv run python ml-experiments/00115-*/e00115_gate.py [--workers 6]
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import date
from typing import Final

import e00115_arms as arms
import e00115_common as common
import e00115_ens as ens
import e00115_members as members
import e00115_shape as shape
import e00115_site as site_module
import e00115_tune as tune
import numpy as np
import polars as pl
from studies.dp_assimilation import (
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_ens import proxy_residual_stream
from studies.dp_ens_weather import AnalysisProxy, EnsSiteRun, proxy_cache

SITES: Final[tuple[str, ...]] = common.DECISION_SITES
INJECTION_STARTS: Final[tuple[date, ...]] = (date(2024, 11, 1), date(2025, 1, 1))
INJECTION_DAYS: Final[int] = 60
DAYS_READ: Final[range] = range(7, 14)
"""Days 8 to 14 of the injection, as zero-based offsets from its first day."""
EVENING_HALF_HOURS: Final[tuple[int, ...]] = (34, 35, 36, 37, 38, 39)
"""Half-hours starting 17:00 to 19:30 local, inside the evening band."""
MORNING_PLUS: Final[tuple[int, ...]] = (13, 14)
MORNING_MINUS: Final[tuple[int, ...]] = (16, 17)
SHAPES: Final[dict[str, dict[int, float]]] = {
    "a_evening_plus3": dict.fromkeys(EVENING_HALF_HOURS, 3.0),
    "b_morning_earlier": {**dict.fromkeys(MORNING_PLUS, 2.0), **dict.fromkeys(MORNING_MINUS, -2.0)},
}
RECOVERY_TOLERANCE: Final[float] = 0.15
RECOVERY_FLOOR: Final[float] = 0.4
RECOVERY_CEILING: Final[float] = 1.25
SPILLOVER_LIMIT: Final[float] = 0.25
BRIEF_TARGET: Final[tuple[float, float]] = (0.6, 1.0)
CLEAN_Q_S_RATIO: Final[float] = 0.1
CLEAN_RMS_LIMIT: Final[float] = 0.3
CLEAN_SCORE_LIMIT: Final[float] = 0.02
TUNED_RMS_FACTOR: Final[float] = 1.2
GATE_PATH = common.WORK_DIR / "gate.json"


def kw_setup(
    *, data: site_module.SiteData, hypers: dict, family: str = "KW"
) -> tuple[shape.ShapeStructure, shape.ShapeHyper]:
    """Return a shape family's structure and tuned hyperparameters at a site."""
    structure = arms.structure_of(family=family, band_of_half_hour=data.band_of_half_hour)
    return structure, arms.family_hyper(data=data, family=family, hypers=hypers)


def d1_total(
    *,
    data: site_module.SiteData,
    trace: shape.ShapeTrace,
    structure: shape.ShapeStructure,
    rows: np.ndarray,
    delay_hours: float = 3.5,
) -> np.ndarray:
    """Return the D+1 correction of a trace at grid rows.

    Each row is forecast from 09:00 UTC on the day before its UTC day, from readings that ended at
    least `delay_hours` before.
    """
    valid = data.cache.end_times[rows]
    initialisation = day_ahead_initialisation(end_times=valid, days_ahead=1)
    cutoff = usable_cutoff_positions(
        end_times=data.cache.end_times, initialisation=initialisation, delay_hours=delay_hours
    )
    return shape.lookup(
        trace=trace, structure=structure, cutoff=cutoff, target=rows, half_hour=data.half_hour[rows]
    ).total


def injection(*, data: site_module.SiteData, start: date, amounts: dict[int, float]) -> np.ndarray:
    """Return the injected amount at every grid row (zero outside the window and half-hours)."""
    local_date = data.cache.calendar.local_date
    first = np.datetime64(start, "D")
    inside = (local_date >= first) & (local_date < first + np.timedelta64(INJECTION_DAYS, "D"))
    out = np.zeros(len(local_date))
    for half_hour, amount in amounts.items():
        out[inside & (data.half_hour == half_hour)] = amount
    return out


def read_rows(*, data: site_module.SiteData, start: date) -> np.ndarray:
    """Return the grid rows of days 8 to 14 of an injection that have a reading outside the masks."""
    local_date = data.cache.calendar.local_date
    first = np.datetime64(start, "D")
    days = (local_date >= first + np.timedelta64(DAYS_READ.start, "D")) & (
        local_date < first + np.timedelta64(DAYS_READ.stop, "D")
    )
    usable = data.features.present & np.isfinite(data.features.y) & ~data.features.switching_mask
    return np.flatnonzero(days & usable)


def recovery_of(*, delta: np.ndarray, injected: np.ndarray) -> float:
    """Return the fitted share of the injection that a correction carries at the injected rows."""
    chosen = injected != 0
    return float((delta[chosen] * injected[chosen]).sum() / (injected[chosen] ** 2).sum())


def spillover_of(
    *, delta: np.ndarray, rows: np.ndarray, data: site_module.SiteData, amounts: dict[int, float]
) -> float:
    """Return the largest mean correction at an un-injected half-hour, over the amplitude."""
    amplitude = max(abs(a) for a in amounts.values())
    half_hours = data.half_hour[rows]
    worst = 0.0
    for half_hour in range(48):
        if half_hour in amounts:
            continue
        chosen = half_hours == half_hour
        if chosen.any():
            worst = max(worst, abs(float(delta[chosen].mean())))
    return worst / amplitude


def recovery_site(job: tuple[str, dict, str]) -> dict:
    """Test 1 at one site: recovery of each injection by a shape family, K0, and the recipe.

    The family is KW for the pre-registered gate and `KWfast` for the post hoc reading.
    """
    site, hypers, family = job
    data = tune.get_site(site)
    structure, hyper = kw_setup(data=data, hypers=hypers, family=family)
    k0_hyper = data.kalman_hyper
    results: list[dict] = []
    ideal = np.zeros(len(data.stream))
    kw_clean = shape.run_shape_filter(
        residual=data.stream, half_hour=data.half_hour, structure=structure, hyper=hyper
    )
    k0_clean = shape.run_shape_filter(
        residual=data.stream, half_hour=data.half_hour, structure=structure, hyper=k0_hyper
    )
    ideal_clean = shape.run_shape_filter(
        residual=ideal, half_hour=data.half_hour, structure=structure, hyper=hyper
    )
    base_shift = arms.profile_correction(
        data=data, k0_q50=arms_k0_q50(data=data), kappa=0.75, delay_hours=3.5
    ).shape
    for start in INJECTION_STARTS:
        rows = read_rows(data=data, start=start)
        for name, amounts in SHAPES.items():
            added = injection(data=data, start=start, amounts=amounts)
            stream = np.where(np.isfinite(data.stream), data.stream + added, np.nan)
            kw = shape.run_shape_filter(
                residual=stream, half_hour=data.half_hour, structure=structure, hyper=hyper
            )
            k0 = shape.run_shape_filter(
                residual=stream, half_hour=data.half_hour, structure=structure, hyper=k0_hyper
            )
            ideal_trace = shape.run_shape_filter(
                residual=ideal + added, half_hour=data.half_hour, structure=structure, hyper=hyper
            )
            kw_delta = d1_total(data=data, trace=kw, structure=structure, rows=rows) - d1_total(
                data=data, trace=kw_clean, structure=structure, rows=rows
            )
            k0_delta = d1_total(data=data, trace=k0, structure=structure, rows=rows) - d1_total(
                data=data, trace=k0_clean, structure=structure, rows=rows
            )
            expected_delta = d1_total(
                data=data, trace=ideal_trace, structure=structure, rows=rows
            ) - d1_total(data=data, trace=ideal_clean, structure=structure, rows=rows)
            r_delta = recipe_delta(data=data, added=added, rows=rows, base_shift=base_shift)
            injected = added[rows]
            results.append(
                {
                    "site": site,
                    "start": start.isoformat(),
                    "shape": name,
                    "kw": recovery_of(delta=kw_delta, injected=injected),
                    "k0": recovery_of(delta=k0_delta, injected=injected),
                    "r": recovery_of(delta=r_delta, injected=injected),
                    "expected": recovery_of(delta=expected_delta, injected=injected),
                    "spillover": spillover_of(
                        delta=kw_delta, rows=rows, data=data, amounts=amounts
                    ),
                    "r_spillover": spillover_of(
                        delta=r_delta, rows=rows, data=data, amounts=amounts
                    ),
                }
            )
    return {"site": site, "results": results}


def arms_k0_q50(*, data: site_module.SiteData) -> np.ndarray:
    """Return K0's median forecast at every row, from its stored correction and spread."""
    correction = arms.load_correction(arm="K0", site=data.site)
    spread = arms.load_spread(arm="K0", site=data.site)
    quantiles, rows = arms.quantiles_for_runs(
        data=data,
        correction=correction.total,
        spread=spread,
        run_indices=np.arange(len(data.runs)),
    )
    q50 = np.full(data.n_rows, np.nan)
    q50[rows] = quantiles[:, 5]
    return q50


def recipe_delta(
    *, data: site_module.SiteData, added: np.ndarray, rows: np.ndarray, base_shift: np.ndarray
) -> np.ndarray:
    """Return the change in the recipe's D+1 correction that an injection causes at grid rows.

    The injection raises each reading by `added`, so the errors that the 14-day profile averages
    fall by the same amount where the reading was injected.
    """
    actual = data.actual + added[data.position]
    shifted = replace(data, actual=actual)
    shift = arms.profile_correction(
        data=shifted, k0_q50=arms_k0_q50(data=data), kappa=0.75, delay_hours=3.5
    ).shape
    change = shift - base_shift
    d1_rows = data.d1_rows()
    by_position = np.zeros(len(data.stream))
    counted = np.zeros(len(data.stream), dtype=bool)
    chosen = np.flatnonzero(d1_rows)
    by_position[data.position[chosen]] = change[chosen]
    counted[data.position[chosen]] = True
    return np.where(counted[rows], by_position[rows], 0.0)


def summarise_recovery(*, per_site: list[dict]) -> dict:
    """Return test 1's readings and verdict, per shape."""
    flat = [item for site in per_site for item in site["results"]]
    out: dict = {"shapes": {}, "passed": True}
    for name in SHAPES:
        items = [i for i in flat if i["shape"] == name]
        kw = np.array([i["kw"] for i in items])
        expected = np.array([i["expected"] for i in items])
        median, median_expected = float(np.median(kw)), float(np.median(expected))
        checks = {
            "within_tolerance": abs(median - median_expected) <= RECOVERY_TOLERANCE,
            "at_least_floor": median >= RECOVERY_FLOOR,
            "no_site_above_ceiling": bool(kw.max() <= RECOVERY_CEILING),
            "spillover_below_limit": max(i["spillover"] for i in items) < SPILLOVER_LIMIT,
        }
        out["shapes"][name] = {
            "kw_median": median,
            "expected_median": median_expected,
            "kw_max": float(kw.max()),
            "k0_median": float(np.median([i["k0"] for i in items])),
            "r_median": float(np.median([i["r"] for i in items])),
            "spillover_max": max(i["spillover"] for i in items),
            "r_spillover_max": max(i["r_spillover"] for i in items),
            "in_brief_target": BRIEF_TARGET[0] <= median <= BRIEF_TARGET[1],
            "checks": checks,
            "per_site": items,
        }
        out["passed"] = out["passed"] and all(checks.values())
    return out


def reproduction_site(site: str) -> dict:
    """Test 3 at one site: KW with no shape equals GAM-K as experiment 111 stored it."""
    data = tune.get_site(site)
    structure = arms.structure_of(family="KW", band_of_half_hour=data.band_of_half_hour)
    hyper = data.kalman_hyper
    trace = shape.run_shape_filter(
        residual=data.stream, half_hour=data.half_hour, structure=structure, hyper=hyper
    )
    reference = run_kalman(
        residual=data.stream,
        band=data.cache.calendar.band,
        hyper=FilterHyperparameters(
            log_q=hyper.log_q,
            logit_phi=hyper.logit_phi,
            log_sigma_z=hyper.log_sigma_z,
            log_sigma_eps=hyper.log_sigma_eps,
        ),
    )
    cutoff = data.cutoffs(delay_hours=3.5)[data.run_of_row]
    mine = shape.lookup(
        trace=trace,
        structure=structure,
        cutoff=cutoff,
        target=data.position,
        half_hour=data.row_half_hour,
    )
    theirs = correction_at(
        trace=reference, cutoff=cutoff, target=data.position, band=data.hour_band
    )
    out = {
        "site": site,
        "level_vs_code_max_abs": float(np.abs(mine.level - theirs.level).max()),
        "anomaly_vs_code_max_abs": float(np.abs(mine.anomaly - theirs.anomaly).max()),
    }
    spread = site_module.stored_spread(site=site)
    correction = arms.Correction(level=mine.level, anomaly=mine.anomaly, shape=mine.shape)
    for period in ("tuning", "scored"):
        path = common.SRC_WORK_DIR / "dp_preds" / f"gamk_{period}_{site}.parquet"
        stored = pl.read_parquet(path)
        runs = np.flatnonzero(data.run_period == period)
        frame = arms.frame_for_arm(
            data=data, arm="K0", correction=correction, spread=spread, run_indices=runs
        )
        joined = frame.join(stored, on=["run", "valid_time"], suffix="_stored")
        out[f"{period}_rows"] = [frame.height, stored.height, joined.height]
        out[f"{period}_level_max_abs"] = float(
            (joined["level"] - joined["level_stored"]).abs().max()
        )
        out[f"{period}_anomaly_max_abs"] = float(
            (joined["anomaly"] - joined["anomaly_stored"]).abs().max()
        )
        out[f"{period}_quantile_max_abs"] = max(
            float((joined[q] - joined[f"{q}_stored"]).abs().max()) for q in arms.QUANTILE_COLUMNS
        )
    return out


def summarise_reproduction(*, per_site: list[dict]) -> dict:
    """Return test 3's readings and verdict."""
    worst_code = max(
        max(s["level_vs_code_max_abs"], s["anomaly_vs_code_max_abs"]) for s in per_site
    )
    worst_stored = max(max(s[f"{p}_level_max_abs"] for p in ("tuning", "scored")) for s in per_site)
    worst_quantile = max(s[f"{p}_quantile_max_abs"] for s in per_site for p in ("tuning", "scored"))
    rows_match = all(
        s[f"{p}_rows"][0] == s[f"{p}_rows"][1] == s[f"{p}_rows"][2]
        for s in per_site
        for p in ("tuning", "scored")
    )
    return {
        "worst_vs_code_float64": worst_code,
        "worst_level_vs_stored": worst_stored,
        "worst_quantile_vs_stored": worst_quantile,
        "rows_match": rows_match,
        "passed": bool(worst_code <= 1e-9 and worst_quantile <= 1e-4 and rows_match),
        "per_site": per_site,
    }


def causality_site(job: tuple[str, dict]) -> dict:
    """Test 4 at one site: a change after the cut leaves every arm's correction at the origin alone."""
    site, hypers = job
    data = tune.get_site(site)
    rng = np.random.default_rng(115)
    structure, hyper = kw_setup(data=data, hypers=hypers)
    runs = [
        int(
            np.flatnonzero(data.run_period == period)[
                len(np.flatnonzero(data.run_period == period)) // 2
            ]
        )
        for period in ("tuning", "replicate", "scored")
    ]
    out: dict = {"site": site, "readings": [], "recipe": [], "ensemble": []}
    base = {
        name: shape.run_shape_filter(
            residual=data.stream, half_hour=data.half_hour, structure=structure, hyper=h
        )
        for name, h in (("KW", hyper), ("K0", data.kalman_hyper))
    }
    for run in runs:
        rows = np.arange(data.offsets[run], data.offsets[run + 1])
        for delay, day_only, label in ((3.5, False, "KW"), (3.5, True, "KW-day")):
            cutoff = data.cutoffs(delay_hours=delay, day_only=day_only)[run]
            full_cutoff = data.cutoffs(delay_hours=delay)[run]
            perturbed = data.stream.copy()
            after = np.arange(len(perturbed)) > cutoff
            perturbed[after] = np.where(
                np.isfinite(perturbed[after]), rng.normal(0, 5, after.sum()), np.nan
            )
            trace = shape.run_shape_filter(
                residual=perturbed, half_hour=data.half_hour, structure=structure, hyper=hyper
            )
            for name, reference in (("KW", base["KW"]),):
                before = shape.lookup(
                    trace=reference,
                    structure=structure,
                    cutoff=np.full(len(rows), cutoff),
                    target=data.position[rows],
                    half_hour=data.row_half_hour[rows],
                ).total
                changed = shape.lookup(
                    trace=trace,
                    structure=structure,
                    cutoff=np.full(len(rows), cutoff),
                    target=data.position[rows],
                    half_hour=data.row_half_hour[rows],
                ).total
                out["readings"].append(
                    {
                        "arm": label,
                        "run": run,
                        "cutoff": int(cutoff),
                        "full_cutoff": int(full_cutoff),
                        "max_abs_change": float(np.abs(before - changed).max()),
                    }
                )
        recipe_rows = tune_recipe_causality(data=data, run=run, rng=rng)
        out["recipe"].append(recipe_rows)
    out["ensemble"] = ensemble_causality(site=site, hypers=hypers, runs=runs[:2])
    out["day_only_sensitivity"] = day_only_sensitivity(
        data=data, structure=structure, hyper=hyper, run=runs[2], rng=rng
    )
    return out


def day_only_sensitivity(
    *,
    data: site_module.SiteData,
    structure: shape.ShapeStructure,
    hyper: shape.ShapeHyper,
    run: int,
    rng: np.random.Generator,
) -> dict:
    """Show that a reading between local midnight and the cutoff moves KW but not KW-day."""
    rows = np.arange(data.offsets[run], data.offsets[run + 1])
    midnight = data.cutoffs(delay_hours=3.5, day_only=True)[run]
    cutoff = data.cutoffs(delay_hours=3.5)[run]
    perturbed = data.stream.copy()
    between = (np.arange(len(perturbed)) > midnight) & (np.arange(len(perturbed)) <= cutoff)
    perturbed[between] = np.where(
        np.isfinite(perturbed[between]), rng.normal(0, 5, between.sum()), np.nan
    )
    both = {}
    for label, residual in (("clean", data.stream), ("perturbed", perturbed)):
        trace = shape.run_shape_filter(
            residual=residual, half_hour=data.half_hour, structure=structure, hyper=hyper
        )
        for arm_label, stop in (("KW", cutoff), ("KW-day", midnight)):
            both[(label, arm_label)] = shape.lookup(
                trace=trace,
                structure=structure,
                cutoff=np.full(len(rows), stop),
                target=data.position[rows],
                half_hour=data.row_half_hour[rows],
            ).total
    return {
        "kw_change": float(np.abs(both[("clean", "KW")] - both[("perturbed", "KW")]).max()),
        "kw_day_change": float(
            np.abs(both[("clean", "KW-day")] - both[("perturbed", "KW-day")]).max()
        ),
    }


def tune_recipe_causality(
    *, data: site_module.SiteData, run: int, rng: np.random.Generator
) -> dict:
    """Perturb everything after a run's cut and check the recipe's correction at the run."""
    k0_q50 = arms_k0_q50(data=data)
    cutoff = data.cutoffs(delay_hours=3.5)[run]
    reference = arms.profile_correction(data=data, k0_q50=k0_q50, kappa=0.75, delay_hours=3.5).shape
    later_rows = data.run_of_row > run
    actual = np.where(
        data.position > cutoff, data.actual + rng.normal(0, 5, data.n_rows), data.actual
    )
    k0_changed = np.where(later_rows, k0_q50 + rng.normal(0, 5, data.n_rows), k0_q50)
    changed = arms.profile_correction(
        data=replace(data, actual=actual), k0_q50=k0_changed, kappa=0.75, delay_hours=3.5
    ).shape
    rows = np.arange(data.offsets[run], data.offsets[run + 1])
    return {"run": run, "max_abs_change": float(np.abs(reference[rows] - changed[rows]).max())}


def ensemble_causality(*, site: str, hypers: dict, runs: list[int]) -> list[dict]:
    """Change every ensemble run issued after a run's origin and check the filter's inputs."""
    cache = common.load_cache()
    setup = ens.load_setup(cache=cache, site=site).with_calibration(
        calibration=members.load_calibration(site=site)
    )
    fits = members.load_fits(site=site)
    reference = proxy_residual_stream(
        cache=proxy_cache(cache=cache, context=setup.context), fits=fits, site=site
    )
    data = tune.get_site(site)
    rng = np.random.default_rng(1150)
    out = []
    for run in runs:
        origin = data.run_origins[run]
        cutoff = data.cutoffs(delay_hours=3.5)[run]
        changed_runs = []
        for index, item in enumerate(setup.runs):
            if setup.origins[index].replace(tzinfo=None) > origin.astype("datetime64[us]").astype(
                type(setup.origins[index].replace(tzinfo=None))
            ):
                changed_runs.append(
                    EnsSiteRun(
                        origin_position=item.origin_position,
                        temperature_c=item.temperature_c
                        + rng.normal(0, 5, item.temperature_c.shape),
                        shortwave_w_m2=item.shortwave_w_m2 * rng.uniform(0.2, 1.8),
                    )
                )
            else:
                changed_runs.append(item)
        context = replace(
            setup.context, proxy=AnalysisProxy.build(runs=changed_runs, n_rows=len(cache.end_times))
        )
        stream = proxy_residual_stream(
            cache=proxy_cache(cache=cache, context=context), fits=fits, site=site
        )
        usable = np.arange(len(stream)) <= cutoff
        same = np.array_equal(reference[usable], stream[usable], equal_nan=True)
        out.append(
            {
                "run": run,
                "inputs_identical_up_to_cutoff": bool(same),
                "inputs_changed_after": bool(
                    not np.array_equal(reference[~usable], stream[~usable], equal_nan=True)
                ),
            }
        )
    return out


def summarise_causality(*, per_site: list[dict]) -> dict:
    """Return test 4's readings and verdict."""
    readings = [r for s in per_site for r in s["readings"]]
    recipe = [r for s in per_site for r in s["recipe"]]
    ensemble = [r for s in per_site for r in s["ensemble"]]
    sensitivity = [s["day_only_sensitivity"] for s in per_site]
    day_midnight_earlier = [r for r in readings if r["arm"] == "KW-day"]
    return {
        "worst_reading_change": max(r["max_abs_change"] for r in readings),
        "worst_recipe_change": max(r["max_abs_change"] for r in recipe),
        "ensemble_inputs_identical": all(r["inputs_identical_up_to_cutoff"] for r in ensemble),
        "ensemble_inputs_changed_after": all(r["inputs_changed_after"] for r in ensemble),
        "kw_day_ignores_readings_after_midnight": max(s["kw_day_change"] for s in sensitivity)
        == 0.0,
        "kw_sees_readings_after_midnight": min(s["kw_change"] for s in sensitivity) > 0.0,
        "n_kw_day_readings": len(day_midnight_earlier),
        "passed": bool(
            max(r["max_abs_change"] for r in readings) == 0.0
            and max(r["max_abs_change"] for r in recipe) == 0.0
            and all(r["inputs_identical_up_to_cutoff"] for r in ensemble)
            and max(s["kw_day_change"] for s in sensitivity) == 0.0
        ),
        "per_site": per_site,
    }


def zero_mean_site(job: tuple[str, dict]) -> dict:
    """Test 5 at one site: the shape sums to zero in each block after every update."""
    site, hypers = job
    data = tune.get_site(site)
    out: dict = {"site": site, "worst": {}}
    for family in ("KW", "KS", "KS1", "KWnomid"):
        if family not in hypers and family not in ("KWnomid",):
            continue
        structure = arms.structure_of(family=family, band_of_half_hour=data.band_of_half_hour)
        hyper = arms.family_hyper(data=data, family=family, hypers=hypers)
        trace = shape.run_shape_filter(
            residual=data.stream, half_hour=data.half_hour, structure=structure, hyper=hyper
        )
        block = trace.state[:, 2 * trace.n_groups :]
        worst = 0.0
        for members_of_block in blocks_of(structure=structure):
            worst = max(worst, float(np.abs(block[:, members_of_block].sum(axis=1)).max()))
        out["worst"][family] = worst
    return out


def blocks_of(*, structure: shape.ShapeStructure) -> list[np.ndarray]:
    """Return the half-hour indices of each block of the projection that has any shape."""
    nonzero = np.flatnonzero(np.abs(structure.projection).sum(axis=1) > 0)
    seen: set[int] = set()
    blocks = []
    for index in nonzero:
        if int(index) in seen:
            continue
        members_of_block = np.flatnonzero(np.abs(structure.projection[index]) > 0)
        seen.update(int(m) for m in members_of_block)
        blocks.append(members_of_block)
    return blocks


def summarise_zero_mean(*, per_site: list[dict]) -> dict:
    """Return test 5's readings and verdict."""
    worst = max(v for s in per_site for v in s["worst"].values())
    return {"worst_abs_block_sum": worst, "passed": bool(worst <= 1e-9), "per_site": per_site}


# --- test 2: clean data ---


def permuted_stream(*, data: site_module.SiteData, seed: int) -> np.ndarray:
    """Return the real stream with each day's readings permuted within their hour band.

    A day is a local date; readings keep their place in the NaN pattern, so only which half-hour
    carries which value changes.
    """
    rng = np.random.default_rng(seed)
    stream = data.stream
    out = stream.copy()
    local_date = data.cache.calendar.local_date.astype("datetime64[D]").astype(np.int64)
    band = data.cache.calendar.band.astype(np.int64)
    key = local_date * 4 + band
    finite = np.flatnonzero(np.isfinite(stream))
    order = np.argsort(key[finite], kind="stable")
    sorted_rows = finite[order]
    boundaries = np.flatnonzero(np.diff(key[sorted_rows])) + 1
    for group in np.split(sorted_rows, boundaries):
        out[group] = stream[rng.permutation(group)]
    return out


def synthetic_stream(*, data: site_module.SiteData, seed: int) -> np.ndarray:
    """Return an AR(1) stream with a random-walk level and no diurnal term.

    The level, the transient, and the noise use the site's fitted GAM-K settings, and the stream
    is finite exactly where the real stream is.
    """
    rng = np.random.default_rng(seed)
    hyper = data.kalman_hyper
    n = len(data.stream)
    level = np.cumsum(rng.normal(0.0, np.sqrt(hyper.q / 48.0), n))
    transient = np.zeros(n)
    innovations = rng.normal(0.0, hyper.sigma_z, n)
    for i in range(1, n):
        transient[i] = hyper.phi * transient[i - 1] + innovations[i]
    noise = rng.normal(0.0, hyper.sigma_eps, n)
    return np.where(np.isfinite(data.stream), level + transient + noise, np.nan)


def clean_streams(*, kind: str, seed: int) -> dict[str, np.ndarray]:
    """Return each site's clean stream of a kind."""
    out = {}
    for site in common.SCORED_SITES:
        data = tune.get_site(site)
        out[site] = (
            permuted_stream(data=data, seed=seed)
            if kind == "permuted"
            else synthetic_stream(data=data, seed=seed)
        )
    return out


def _clean_fit(
    job: tuple[str, str, int, list[float], list[float], list[float]],
) -> tuple[str, list]:
    kind, site, seed, prior_mean, prior_spread, start = job
    data = tune.get_site(site)
    stream = clean_streams_one(kind=kind, data=data, seed=seed)
    codes = np.array([100 * y + m for y, m in common.TUNING_MONTHS])
    masked = np.where(np.isin(data.month_code, codes), stream, np.nan)
    structure = arms.structure_of(family="KW", band_of_half_hour=data.band_of_half_hour)
    values = shape.fit_shape_hyper(
        residual=masked,
        half_hour=data.half_hour,
        structure=structure,
        base=data.kalman_hyper,
        prior_mean=np.array(prior_mean),
        prior_spread=np.array(prior_spread),
        start=np.array(start),
    )
    return site, values.tolist()


def clean_streams_one(*, kind: str, data: site_module.SiteData, seed: int) -> np.ndarray:
    """Return one site's clean stream of a kind."""
    return (
        permuted_stream(data=data, seed=seed + int(data.site[1:]))
        if kind == "permuted"
        else synthetic_stream(data=data, seed=seed + int(data.site[1:]))
    )


def refit_clean(*, kind: str, pool: ProcessPoolExecutor, seed: int) -> dict[str, list[float]]:
    """Refit (log q_s, log sigma_eps) on a clean stream by the tuning procedure."""
    sites = common.SCORED_SITES
    noise = float(np.mean([tune.get_site(s).kalman_hyper.log_sigma_eps for s in sites]))
    prior_mean = np.array([tune.START_LOG_Q_S, noise])
    prior_spread = np.array([2.0, 0.5])
    fitted: dict[str, list] = {}
    for _ in range(shape.N_EMPIRICAL_BAYES_ROUNDS):
        jobs = [
            (
                kind,
                s,
                seed,
                prior_mean.tolist(),
                prior_spread.tolist(),
                fitted.get(s, prior_mean.tolist()),
            )
            for s in sites
        ]
        fitted = dict(pool.map(_clean_fit, jobs))
        values = np.array(list(fitted.values()))
        prior_mean = values.mean(axis=0)
        prior_spread = np.maximum(values.std(axis=0, ddof=1), shape.MIN_SHAPE_PRIOR_SPREAD)
    return fitted


def clean_readings_site(job: tuple[str, str, int, list[float], dict]) -> dict:
    """Test 2's readings at one site: the refit's shape RMS and the score of KW against K0."""
    kind, site, seed, refit, hypers = job
    data = tune.get_site(site)
    stream = clean_streams_one(kind=kind, data=data, seed=seed)
    structure = arms.structure_of(family="KW", band_of_half_hour=data.band_of_half_hour)
    refit_hyper = data.kalman_hyper.with_shape(log_q_s=refit[0], log_sigma_eps=refit[1])
    tuned_hyper = arms.family_hyper(data=data, family="KW", hypers=hypers)
    codes = np.array([100 * y + m for y, m in common.TUNING_MONTHS])
    in_tuning = np.isin(data.month_code, codes)
    results: dict = {
        "site": site,
        "kind": kind,
        "refit_q_s": float(np.exp(refit[0])),
        "refit_sigma_eps": float(np.exp(refit[1])),
    }
    traces = {}
    for name, hyper in (("refit", refit_hyper), ("tuned", tuned_hyper)):
        trace = shape.run_shape_filter(
            residual=stream,
            half_hour=data.half_hour,
            structure=structure,
            hyper=hyper,
            keep_variance=True,
        )
        traces[name] = trace
        usable = in_tuning[trace.positions]
        s_state = trace.state[usable][:, 2 * trace.n_groups :]
        results[f"{name}_s_rms"] = float(np.sqrt((s_state**2).mean()))
        results[f"{name}_p_s_rms"] = float(np.sqrt(trace.shape_variance[usable].mean()))
    results["tuned_q_s"] = float(tuned_hyper.q_s)
    # D+1 score of KW against K0 when the stream is the clean one: the readings the forecasts
    # are scored against are the proxy's forecast plus the clean residual.
    proxy_forecast = data.features.y - data.stream
    clean_actual_grid = np.where(np.isfinite(stream), proxy_forecast + stream, np.nan)
    actual = clean_actual_grid[data.position]
    clean_data = replace(
        data,
        actual=actual,
        scorable=np.isfinite(actual) & ~data.features.switching_mask[data.position],
    )
    scores = {}
    for name, hyper in (("K0", data.kalman_hyper), ("KW", refit_hyper)):
        trace = shape.run_shape_filter(
            residual=stream, half_hour=data.half_hour, structure=structure, hyper=hyper
        )
        cutoff = data.cutoffs(delay_hours=3.5)[data.run_of_row]
        found = shape.lookup(
            trace=trace,
            structure=structure,
            cutoff=cutoff,
            target=data.position,
            half_hour=data.row_half_hour,
        )
        correction = found.total
        spread = arms.fit_spread(data=clean_data, correction=correction)
        total, count = arms.d1_tuning_score(data=clean_data, correction=correction, spread=spread)
        scores[name] = [total, count]
    results["d1_scores"] = scores
    return results


def summarise_clean(*, readings: list[dict]) -> dict:
    """Return test 2's verdict per kind of clean stream."""
    out: dict = {"kinds": {}, "passed": True}
    for kind in ("permuted", "synthetic"):
        rows = [r for r in readings if r["kind"] == kind and r["site"] in SITES]
        q_s_ok = all(r["refit_q_s"] < CLEAN_Q_S_RATIO * r["tuned_q_s"] for r in rows)
        rms_ok = all(r["refit_s_rms"] < CLEAN_RMS_LIMIT for r in rows)
        totals = {
            name: np.sum([r["d1_scores"][name] for r in rows], axis=0) for name in ("K0", "KW")
        }
        gap = float(totals["KW"][0] / totals["KW"][1] - totals["K0"][0] / totals["K0"][1])
        score_ok = gap <= CLEAN_SCORE_LIMIT
        tuned_ok = all(r["tuned_s_rms"] <= TUNED_RMS_FACTOR * r["tuned_p_s_rms"] for r in rows)
        checks = {
            "refit_q_s_under_tenth_of_tuned": q_s_ok,
            "refit_s_rms_under_limit": rms_ok,
            "kw_score_within_limit_of_k0": score_ok,
            "tuned_rms_at_most_1.2_prior_sd": tuned_ok,
        }
        out["kinds"][kind] = {"score_gap_kw_minus_k0": gap, "checks": checks, "per_site": rows}
        out["passed"] = out["passed"] and all(checks.values())
    return out


def main() -> None:
    """Run the gate and write `work/gate.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1150)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--post-hoc-fast", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    hypers = arms.read_hypers()
    report: dict = json.loads(GATE_PATH.read_text()) if GATE_PATH.exists() else {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if "3" not in args.skip:
            per_site = list(pool.map(reproduction_site, common.SCORED_SITES))
            report["test_3_reproduction"] = summarise_reproduction(per_site=per_site)
            print("test 3", report["test_3_reproduction"]["passed"], flush=True)
        if "4" not in args.skip:
            per_site = list(pool.map(causality_site, [(s, hypers) for s in SITES]))
            report["test_4_causality"] = summarise_causality(per_site=per_site)
            print("test 4", report["test_4_causality"]["passed"], flush=True)
        if "5" not in args.skip:
            per_site = list(pool.map(zero_mean_site, [(s, hypers) for s in SITES]))
            report["test_5_zero_mean"] = summarise_zero_mean(per_site=per_site)
            print("test 5", report["test_5_zero_mean"]["passed"], flush=True)
        if "1" not in args.skip:
            per_site = list(pool.map(recovery_site, [(s, hypers, "KW") for s in SITES]))
            report["test_1_recovery"] = summarise_recovery(per_site=per_site)
            print("test 1", report["test_1_recovery"]["passed"], flush=True)
        if args.post_hoc_fast:
            per_site = list(pool.map(recovery_site, [(s, hypers, "KWfast") for s in SITES]))
            report["post_hoc_test_1_kw_fast"] = summarise_recovery(per_site=per_site)
            print("post hoc test 1 (KW-fast)", report["post_hoc_test_1_kw_fast"]["passed"])
        if "2" not in args.skip:
            readings = []
            for kind in ("permuted", "synthetic"):
                refit = refit_clean(kind=kind, pool=pool, seed=args.seed)
                jobs = [(kind, s, args.seed, refit[s], hypers) for s in common.SCORED_SITES]
                readings.extend(pool.map(clean_readings_site, jobs))
            report["test_2_clean"] = summarise_clean(readings=readings)
            print("test 2", report["test_2_clean"]["passed"], flush=True)
    tests = [k for k in report if k.startswith("test_")]  # the post hoc reading is not a gate test
    report["failed"] = sum(1 for k in tests if not report[k]["passed"])
    report["tests_run"] = tests
    common.write_json_atomic(path=GATE_PATH, content=report)
    print(f"gate: {report['failed']} failed of {len(tests)} ({time.monotonic() - started:.0f} s)")


if __name__ == "__main__":
    main()
