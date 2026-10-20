"""Arm B of experiment 124: the audit-calibrated hint.

    uv run python ml-experiments/00124-*/e00124_armb.py freeze
    uv run python ml-experiments/00124-*/e00124_armb.py plumbing
    uv run python ml-experiments/00124-*/e00124_armb.py tune

`freeze` writes the closure audit's multipliers at the tuning cut to `armb_multipliers.json`, once,
from rows before the cut. `plumbing` is the blocking synthetic check. `tune` fits the calibrated
hint over a weight grid that reaches far below the first arm's, injects the same errors, and applies
the plan's decision rule.
"""

import json
import sys
from datetime import datetime
from typing import Final

import numpy as np
import torch
from e00124_common import TUNING_CUT, WORK_DIR, load_bsps, load_cache, to_64
from e00124_gate import build_truth
from e00124_inject import ScoreRows, mae, monthly_mae, score_rows
from e00124_pilot import SEASON_YEARS, Pilot, build_pilot, closure_multipliers
from e00124_run import TUNING_SEEDS, inject_all, setup
from studies.dp_features import FeatureCache
from studies.dp_hierarchy import RemainderSpec, assemble_joint, solve_joint

WEIGHT_GRID: Final[tuple[float, ...]] = (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
"""Weight 0 and the plan's eight weights."""
CLEAN_LIMIT: Final[float] = 0.023
MIN_REDUCTION: Final[float] = 0.30
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
DRAWS: Final[int] = 2000
N_SYSTEMS: Final[int] = 20
WRONG_BY: Final[float] = 1.15
PLUMBING_WEIGHT: Final[float] = 0.1
PLUMBING_INDEX: Final[int] = WEIGHT_GRID.index(PLUMBING_WEIGHT)
DEVICE: Final[str] = "cuda" if torch.cuda.is_available() else "cpu"
FROZEN: Final = WORK_DIR / "armb_multipliers.json"


def freeze() -> None:
    """Write the audit's multipliers at the tuning cut, refusing to overwrite them."""
    if FROZEN.exists():
        msg = f"{FROZEN} exists; the multipliers are frozen"
        raise FileExistsError(msg)
    cache = load_cache()
    multipliers = closure_multipliers(cache=cache, bsps=load_bsps(cache=cache), cut=TUNING_CUT)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(multipliers, indent=2))
    sys.stdout.write(json.dumps(multipliers, indent=2) + "\n")


def _multipliers() -> dict[str, float]:
    values = json.loads(FROZEN.read_text())
    return {k: values[k] for k in ("a_P", "a_G", "c")}


def plumbing() -> None:
    """Fit exact synthetic systems with the audit's multipliers and with them 15% wrong."""
    multipliers = _multipliers()
    cache = load_cache()
    bsps = load_bsps(cache=cache)
    pilot = build_pilot(
        cache=cache, bsps=bsps, cut=TUNING_CUT, multipliers=multipliers, sigma_b_from="fitted"
    )
    base = build_pilot(cache=cache, bsps=bsps, cut=TUNING_CUT, sigma_b_from="fitted")
    assembly = assemble_joint(
        cache=cache,
        specs=base.specs,
        bsps=[base.bsp],
        remainder_spec=RemainderSpec(),
        sigma_points=base.sigma_points,
        cut=TUNING_CUT,
        window=base.windows,
        season_years=SEASON_YEARS,
        device=DEVICE,
    )
    truth = build_truth(cache=cache, pilot=base, assembly=assembly)
    wrong = {k: v * WRONG_BY for k, v in multipliers.items()}
    arms = {"exact": multipliers, "wrong by 15%": wrong}
    built = {
        name: build_pilot(
            cache=cache, bsps=bsps, cut=TUNING_CUT, multipliers=m, sigma_b_from="fitted"
        )
        for name, m in arms.items()
    }
    assemblies = {
        name: assemble_joint(
            cache=cache,
            specs=p.specs,
            bsps=[p.bsp],
            remainder_spec=RemainderSpec(),
            sigma_points=p.sigma_points,
            cut=TUNING_CUT,
            window=p.windows,
            season_years=SEASON_YEARS,
            device=DEVICE,
        )
        for name, p in built.items()
    }
    probe = solve_joint(assembly=assemblies["exact"], weights=[0.0])
    scores = {
        s: score_rows(cache=cache, assembly=assemblies["exact"], fit=probe, site=s, cut=TUNING_CUT)
        for s in pilot.mw_sites
    }
    rng_master = np.random.default_rng(7)
    rows = []
    n = len(cache.end_times)
    for system in range(N_SYSTEMS):
        rng = np.random.default_rng(rng_master.integers(2**32))
        from e00124_gate import ar1

        power = {
            m.site: truth.net[m.site]
            + ar1(n=n, sd=truth.sd[m.site], rho=truth.rho[m.site], rng=rng)
            for m in pilot.bsp.members
        }
        reading = truth.r_truth - multipliers["a_G"] * pilot.bsp.generators_mw
        for m in pilot.bsp.members:
            noisy = power[m.site] + (
                0.0 if m.kind == "mw" else truth.demand[m.site] - truth.net[m.site]
            )
            kind_multiplier = multipliers["a_P"] if m.kind == "mw" else multipliers["c"]
            reading = reading + kind_multiplier * truth.capacity[m.site] * noisy
        reading = reading + ar1(n=n, sd=truth.bsp_sd, rho=truth.bsp_rho, rng=rng)
        row: dict[str, object] = {"system": system}
        for name, asm in assemblies.items():
            fit = solve_joint(
                assembly=asm,
                weights=list(WEIGHT_GRID),
                primary_power=power,
                bsp_readings={"pilot": reading},
            )
            errors = []
            for site in pilot.mw_sites:
                observed = ScoreRows(
                    design=scores[site].design,
                    observed=power[site][scores[site].rows],
                    rows=scores[site].rows,
                )
                errors.append(mae(score=observed, fit=fit, site=site))
            mean = np.mean(errors, axis=0)
            row[name] = float(mean[PLUMBING_INDEX] - mean[0])
            row[name + " curve"] = (mean - mean[0]).tolist()
        rows.append(row)
        sys.stdout.write(f"system {system}\n")
        sys.stdout.flush()
    exact = np.array([r["exact"] for r in rows])
    wrong_cost = np.array([r["wrong by 15%"] for r in rows])
    result = {
        "weight": PLUMBING_WEIGHT,
        "median_rise_exact": float(np.median(exact)),
        "median_rise_wrong": float(np.median(wrong_cost)),
        "exact_passes": bool(np.median(exact) <= CLEAN_LIMIT),
        "control_detects": bool(np.median(wrong_cost) > np.median(exact)),
        "median_curve_exact": np.median([r["exact curve"] for r in rows], axis=0).tolist(),
        "median_curve_wrong": np.median([r["wrong by 15% curve"] for r in rows], axis=0).tolist(),
        "rows": rows,
    }
    (WORK_DIR / "armb_plumbing.json").write_text(json.dumps(result, indent=1))
    sys.stdout.write(
        f"median curve over the grid, exact: {np.round(result['median_curve_exact'], 4).tolist()}\n"
    )
    sys.stdout.write(
        f"median clean rise at weight {PLUMBING_WEIGHT}: exact {result['median_rise_exact']:.4f}, "
        f"multipliers 15% wrong {result['median_rise_wrong']:.4f}; exact passes "
        f"{result['exact_passes']}; control detects {result['control_detects']}\n"
    )


def _cold_snap_days(*, cache: FeatureCache, pilot: Pilot, cut: datetime) -> int:
    """Count the cold-snap days of the year after the cut, by the audit's temperature threshold.

    The threshold is the 5% quantile of winter-day mean temperatures before the cut.
    """
    days = cache.end_times.astype("datetime64[D]")
    month = cache.end_times.astype("datetime64[M]").astype(int) % 12 + 1
    winter = np.isin(month, sorted(WINTER_MONTHS))
    temperature = pilot.bsp.temperature_c
    before = winter & (cache.end_times <= to_64(moment=cut))
    after = (
        winter
        & (cache.end_times > to_64(moment=cut))
        & (cache.end_times <= to_64(moment=cut.replace(year=cut.year + 1)))
    )
    means = {d: float(np.nanmean(temperature[days == d])) for d in np.unique(days[before | after])}
    threshold = float(np.quantile([means[d] for d in np.unique(days[before])], 0.05))
    return sum(1 for d in np.unique(days[after]) if means[d] <= threshold)


def tune() -> None:
    """Fit the calibrated hint over the grid, inject the tuning-year errors, and apply the rule."""
    cache, pilot, assembly, cut = setup(
        cut_name="tuning", multipliers=_multipliers(), sigma_b_from="fitted"
    )
    weights = list(WEIGHT_GRID)
    clean = solve_joint(assembly=assembly, weights=weights)
    scores = {
        s: score_rows(cache=cache, assembly=assembly, fit=clean, site=s, cut=cut)
        for s in pilot.mw_sites
    }
    monthly = {
        s: monthly_mae(cache=cache, score=scores[s], fit=clean, site=s) for s in pilot.mw_sites
    }
    clean_mae = {s: mae(score=scores[s], fit=clean, site=s) for s in pilot.mw_sites}
    path = WORK_DIR / "armb_injections.jsonl"
    with path.open("w") as out:
        inject_all(
            cache=cache,
            pilot=pilot,
            assembly=assembly,
            cut=cut,
            weights=weights,
            seeds=TUNING_SEEDS,
            seed_offset=1000,
            scores=scores,
            out=out,
            with_oracle=False,
        )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    damage = np.array([np.array(r["mae"]) - clean_mae[r["site"]] for r in rows])
    months = sorted(next(iter(monthly.values())))
    rise = np.array(
        [
            np.mean(
                [np.array(monthly[s][m]) - np.array(monthly[s][m])[0] for s in pilot.mw_sites],
                axis=0,
            )
            for m in months
        ]
    )
    winter = np.array([int(m[5:7]) in WINTER_MONTHS for m in months])
    rng = np.random.default_rng(3)
    result = {"weights": weights, "months": months, "passing": []}
    for k, w in enumerate(weights):
        if w == 0.0:
            continue
        month_draws = np.array(
            [rise[rng.integers(0, len(months), len(months)), k].mean() for _ in range(DRAWS)]
        )
        upper = float(np.percentile(month_draws, 97.5))
        winter_rise = float(rise[winter, k].mean())
        total0 = damage[:, 0].sum()
        reduction = 1.0 - damage[:, k].sum() / total0
        boot = np.array(
            [
                1.0 - damage[idx, k].sum() / damage[idx, 0].sum()
                for idx in (rng.integers(0, len(damage), len(damage)) for _ in range(DRAWS))
            ]
        )
        lower = float(np.percentile(boot, 2.5))
        passes = (
            upper <= CLEAN_LIMIT
            and winter_rise <= CLEAN_LIMIT
            and reduction >= MIN_REDUCTION
            and lower > 0.0
        )
        entry = {
            "weight": w,
            "clean_rise": float(rise[:, k].mean()),
            "clean_rise_upper": upper,
            "winter_rise": winter_rise,
            "n_winter_months": int(winter.sum()),
            "reduction": float(reduction),
            "reduction_lower": lower,
            "passes": bool(passes),
        }
        result.setdefault("table", []).append(entry)
        if passes:
            result["passing"].append(entry)
    result["cold_snap_days_in_scored_year"] = _cold_snap_days(cache=cache, pilot=pilot, cut=cut)
    best = max(result["passing"], key=lambda e: e["reduction"], default=None)
    result["chosen_weight"] = None if best is None else best["weight"]
    (WORK_DIR / "armb_tuning.json").write_text(json.dumps(result, indent=2))
    sys.stdout.write(
        json.dumps(result["table"], indent=1) + f"\nchosen weight: {result['chosen_weight']}\n"
    )


if __name__ == "__main__":
    {"freeze": freeze, "plumbing": plumbing, "tune": tune}[sys.argv[1]]()
