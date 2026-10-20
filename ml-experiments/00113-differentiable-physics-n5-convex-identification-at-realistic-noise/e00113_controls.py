"""Experiment 113's blocking controls, run before any noisy cell is read.

    uv run python ml-experiments/00113-*/e00113_controls.py static   # controls 1, 4, and 5
    uv run python ml-experiments/00113-*/e00113_controls.py noisy    # controls 2 and 3

1. The G-full and no-drift subsets reproduce `dp_gam.fit_gam` on each real target to 1e-8
   relative; the no-harmonic-no-drift (G-0) and pinned subsets match a dense bounded least-squares
   fit of the same columns on the adjusted target to 1e-8.
2. Generator F with no noise: every fit that contains the truth's columns returns the truth plus
   the analytic penalty bias for every scored functional, to 1e-6.
3. G-0 on T-0 with iid sd 5.0 noise on 500 draws (one solve each): the posterior interval's
   coverage is within 77 to 83% for both families; I1's coverage on the cell's draws is within 70
   to 90%.
4. Every stratified resample keeps each calendar quarter's month count.
5. No training row ends after its cut, and nothing on or after 2026-07-01.
"""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime

import e00113_cell as cell_module
import e00113_common as common
import e00113_design as design_module
import e00113_noise as noise_module
import numpy as np
import polars as pl
import torch
from scipy.optimize import lsq_linear
from studies.dp_features import FeatureCache
from studies.dp_gam import TrainingWindow, fit_gam

REPORT: str = "controls.json"
HELD_OUT_START: datetime = datetime(2026, 7, 1, tzinfo=UTC)


def _real_solve(
    *, design: design_module.Design, target: np.ndarray, arm_name: str, pinned: np.ndarray | None
) -> np.ndarray:
    counts = np.ones((1, len(design.months)))
    gram = design_module.weighted_gram(design=design, counts=counts)
    rhs = design_module.weighted_rhs(
        counts=counts, month_rhs=design_module.month_cross_products(design=design, target=target)
    )
    return design_module.solve_arm(
        design=design,
        arm=design_module.ARMS[arm_name],
        gram=gram,
        rhs=rhs,
        pinned_increments=pinned,
    )[0]


def _dense_reference(
    *,
    design: design_module.Design,
    target: np.ndarray,
    arm: design_module.Arm,
    pinned: np.ndarray | None,
) -> np.ndarray:
    """Solve the arm's penalised, bounded least squares densely with scipy (float64)."""
    keep = design_module.selection(design=design, arm=arm)
    chosen = np.flatnonzero(keep)
    adjusted = target.copy()
    capacity = design.layout.slices["capacity"]
    pinned_theta = np.zeros(design.p)
    if pinned is not None:
        pinned_theta[capacity] = arm.pin * pinned
        adjusted = adjusted - design.x[:, capacity] @ pinned_theta[capacity]
    penalty = design.penalties[arm.drift_step][np.ix_(chosen, chosen)]
    values, vectors = np.linalg.eigh(penalty)
    root = (vectors * np.sqrt(np.clip(values, 0.0, None))).T
    stacked = np.vstack([design.x[:, chosen], root])
    response = np.concatenate([adjusted, np.zeros(root.shape[0])])
    lower = np.where(design.layout.constrained[chosen], 0.0, -np.inf)
    solution = lsq_linear(
        stacked,
        response,
        bounds=(lower, np.full(len(chosen), np.inf)),
        method="bvls",
        tol=1e-14,
        max_iter=5000,
    )
    theta = pinned_theta.copy()
    theta[chosen] = solution.x
    return theta


def control_1(*, cache: FeatureCache) -> dict:
    """Compare the month-Gram solve with `fit_gam` and with a dense reference."""
    out = {}
    for site in common.DECISION_SITES:
        design = design_module.build_design(
            cache=cache, site=site, cut=common.REPLICATE_CUT, window="all"
        )
        target = cache.sites[site].y[design.index]
        mine = _real_solve(design=design, target=target, arm_name="g_full", pinned=None)
        reference = fit_gam(
            cache=cache,
            site=site,
            specs=[design_module.full_spec()],
            cut=common.REPLICATE_CUT,
            window=TrainingWindow(),
            season_years=design.layout.season_years,
        ).theta[0]
        scale = np.abs(reference).max()
        out[f"{site}_g_full_vs_fit_gam"] = common.as_float(np.abs(mine - reference).max() / scale)
        # The no-drift subset: G-0 plus harmonics (g_h) is the no-drift fit; compare with fit_gam
        # laid out without the drift.
        mine_h = _real_solve(design=design, target=target, arm_name="g_h", pinned=None)
        spec_nodrift = replace(design_module.full_spec(), drift="none")
        ref_nodrift = fit_gam(
            cache=cache,
            site=site,
            specs=[spec_nodrift],
            cut=common.REPLICATE_CUT,
            window=TrainingWindow(),
            season_years=design.layout.season_years,
        ).theta[0]
        keep = design_module.selection(design=design, arm=design_module.ARMS["g_h"])
        out[f"{site}_no_drift_vs_fit_gam"] = common.as_float(
            np.abs(mine_h[keep] - ref_nodrift).max() / np.abs(ref_nodrift).max()
        )
        increments = mine[design.layout.slices["capacity"]]
        for name, pin in (("g_0", None), ("g_0_pin", increments), ("g_full_pin", increments)):
            solved = _real_solve(design=design, target=target, arm_name=name, pinned=pin)
            dense = _dense_reference(
                design=design, target=target, arm=design_module.ARMS[name], pinned=pin
            )
            out[f"{site}_{name}_vs_dense"] = common.as_float(
                np.abs(solved - dense).max() / np.abs(dense).max()
            )
    return out


def control_4(*, cache: FeatureCache) -> dict:
    """Check that each stratified resample keeps every quarter's month count."""
    worst = 0
    for site in common.DECISION_SITES:
        for window in ("all", "2y"):
            design = design_module.build_design(
                cache=cache, site=site, cut=common.REPLICATE_CUT, window=window
            )
            counts = design_module.stratified_counts(
                design=design, n_resamples=500, rng=np.random.default_rng(1)
            )
            for quarter in range(4):
                members = design.quarter_of_month == quarter
                drawn = counts[:, members].sum(axis=1)
                worst = max(worst, int(np.abs(drawn - members.sum()).max()))
    return {"max_quarter_count_error": worst}


def control_5(*, cache: FeatureCache) -> dict:
    """Check that no training row ends after its cut and none lies in the held-out period."""
    out = {}
    for site in common.DECISION_SITES:
        for cut in (common.REPLICATE_CUT, common.SCORED_CUT):
            for window in ("all", "2y"):
                design = design_module.build_design(cache=cache, site=site, cut=cut, window=window)
                last = cache.end_times[design.index].max()
                out[f"{site}_{cut.year}_{window}"] = bool(
                    last <= np.datetime64(cut.replace(tzinfo=None), "us")
                    and last < np.datetime64(HELD_OUT_START.replace(tzinfo=None), "us")
                )
    return out


def control_2() -> dict:
    """Check that the noise-free draw returns the truth plus the analytic bias (1e-6)."""
    frames = [
        pl.read_parquet(path) for path in sorted(common.CELLS_DIR.glob("control_*_100.parquet"))
    ]
    frame = pl.concat(frames).filter(
        (pl.col("noise") == "none") & pl.col("analytic_bias").is_not_nan()
    )
    frame = frame.with_columns(
        expected=pl.col("truth_value") + pl.col("analytic_bias"),
        error=(pl.col("point") - pl.col("truth_value") - pl.col("analytic_bias")).abs(),
    )
    worst = (
        frame.sort("error", descending=True)
        .head(5)
        .select("site", "truth", "arm", "functional", "error")
    )
    return {
        "max_abs_error": common.as_float(frame["error"].max()),
        "n_checked": frame.height,
        "worst": worst.to_dicts(),
    }


def control_3(*, cache: FeatureCache, draws_per_site: int = 100) -> dict:
    """Check I2's calibration for G-0 on T-0 with iid sd 5.0 noise on 500 draws."""
    arm = design_module.ARMS["g_0"]
    covered = {name: [] for name in design_module.FUNCTIONAL_NAMES}
    stored = cell_module.load_real_fits()
    for site_position, site in enumerate(common.DECISION_SITES):
        design = design_module.build_design(
            cache=cache, site=site, cut=common.REPLICATE_CUT, window="all"
        )
        analytic = cell_module.arm_analytics(design=design, arm=arm)
        counts = np.ones((1, len(design.months)))
        gram = design_module.weighted_gram(design=design, counts=counts)
        for draw in range(draws_per_site):
            rng = np.random.default_rng(
                np.random.SeedSequence([common.EXPERIMENT_ID, 77, site_position, draw])
            )
            truth, _ = cell_module.make_truth(
                design=design, theta_real=stored[f"theta_{site}"], kind="t0", rng=rng
            )
            noise = noise_module.iid_noise(n=design.n, sd=common.IID_SD, rng=rng)
            month_rhs = torch.einsum("mpq,q->mp", design.month_gram, torch.as_tensor(truth)) + (
                design_module.month_cross_products(design=design, target=noise)
            )
            rhs = design_module.weighted_rhs(counts=counts, month_rhs=month_rhs)
            theta = design_module.solve_arm(
                design=design,
                arm=arm,
                gram=gram,
                rhs=rhs,
                pinned_increments=None,
            )
            residual = design.x @ truth + noise - design.x @ theta[0]
            sigma2 = float(residual @ residual) / (design.n - analytic.edf)
            values = design_module.functional_values(design=design, theta=theta)[0]
            truth_values = design_module.functional_values(design=design, theta=truth)
            sd = np.sqrt(analytic.posterior * sigma2)
            bias = design_module.analytic_bias(design=design, arm=arm, theta_truth=truth)
            inside = np.abs(values - (truth_values + bias)) <= common.INTERVAL_Z * sd
            for functional, name in enumerate(design_module.FUNCTIONAL_NAMES):
                covered[name].append(bool(inside[functional]))
    bands = np.concatenate([covered[name] for name in design_module.FUNCTIONAL_NAMES[:4]])
    return {
        "bands_i2_coverage_of_noise_free_value": common.as_float(bands.mean()),
        "capacity_i2_coverage_of_noise_free_value": float(np.mean(covered["capacity"])),
        "n_draws": len(covered["capacity"]),
    }


def main() -> None:
    """Run the requested controls and write `controls.json`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("static", "noisy"))
    args = parser.parse_args()
    common.configure_threads(threads=4)
    cache = common.load_cache()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    path = common.WORK_DIR / REPORT
    report = json.loads(path.read_text()) if path.exists() else {}
    if args.stage == "static":
        report["control_1"] = control_1(cache=cache)
        report["control_4"] = control_4(cache=cache)
        report["control_5"] = control_5(cache=cache)
    else:
        report["control_2"] = control_2()
        report["control_3_i2"] = control_3(cache=cache)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
