"""The known-answer gate of experiment 125, run on the tuning months before any other period.

A failed blocking test stops scoring. Tests 1 and 2 use the synthetic twin of `e00125_twin`
at the five decision sites; tests 3 to 5 use the real cache.

1. **Variance transfer.** The members' variance (σ_w²), pooled per lead band, is within 10% of the
   500-draw variance of GAM-K on the injected weather, in every lead band whose reference sd
   exceeds 0.5 points.
2. **Known σ_w and σ_m are recovered and the double count has its predicted size.**
   (a) The operational σ_m² (the variance of the proxy-residual table's 11 atoms) is within 10% of
   the injected value on the same 11 atoms. (b) K0's total variance over MP's total variance is
   within 10% of the analytic ratio per lead band. (c) MP is calibrated in at least 14 of 16
   cells. (d) MO under-covers the central 90% interval by more than the analytic amount minus 2
   points. (e) With the members' weather error halved against the truth, MP-s recovers the true
   demand-space factor within 25% in at least 3 of 4 lead bands.
3. **Reproduction.** K0 rebuilt from the cache reproduces experiment 111's stored tuning
   quantiles to 1e-4; 51 identical members give one member's quantiles; MP with a zero table
   equals MO; MP-s with s = 1 equals MP; a table fitted on zero residuals is zero.
4. **Causality.** Changing every actual after the filter's cutoff, or every ENS run issued after the
   origin, leaves the corrected member means and the filter's correction unchanged.
5. **Mean and order.** The mean of each arm's centres equals the corrected ensemble mean to 1e-9,
   and every arm's quantiles are non-decreasing.

    uv run python ml-experiments/00125-*/e00125_gate.py [--workers 3]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from statistics import NormalDist
from typing import Any, Final

import e00125_arms as arms
import e00125_cache as cache_module
import e00125_common as common
import e00125_post as post
import e00125_twin as twin_module
import numpy as np
import polars as pl
from studies.dp_assimilation import usable_cutoff_positions
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import (
    N_LEAD_BANDS,
    AnalysisProxy,
    EnsSiteRun,
    lead_band_of_hours,
    replace_site,
)
from studies.dp_features import naive_utc
from studies.ens_members import MIXTURE_LEVELS, mixture_quantiles

TOLERANCE: Final[float] = 0.10
MC_SD_FLOOR_POINTS: Final[float] = 0.5
_LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (_LEVELS[:-1] + _LEVELS[1:]) / 2, [1.0]])
)
"""The probability mass each of the 11 atoms carries (as `mixture_quantiles` weights them)."""
Z_LEVELS: Final[np.ndarray] = np.array([NormalDist().inv_cdf(p) for p in MIXTURE_LEVELS])
GAUSSIAN_11_ATOM_FACTOR: Final[float] = float((MASSES * Z_LEVELS**2).sum())
"""The variance of the 11 Gaussian atoms, as a multiple of the Gaussian's variance."""
TAIL_POSITIONS: Final[tuple[int, ...]] = post.TAIL_LEVEL_POSITIONS
Q05, Q95 = 1, 9
"""Positions of q05 and q95 among the 11 levels."""
N_SITES_NEEDED: Final[int] = 5
CELLS_NEEDED: Final[int] = 14


def table_atom_variance(*, spread: LeadHourSpread, rows: cache_module.RunCache) -> np.ndarray:
    """Return, per row, the mass-weighted variance of the table's 11 atoms (factor included)."""
    out = np.empty(len(rows.actual))
    band = lead_band_of_hours(lead_hours=rows.lead_hours)
    for index in range(N_LEAD_BANDS):
        chosen = band == index
        if not chosen.any():
            continue
        table = spread.by_lead_band[index].table[
            rows.hour_band[chosen].astype(np.int64), rows.quarter[chosen].astype(np.int64)
        ]
        mean = (table * MASSES).sum(axis=1, keepdims=True)
        out[chosen] = spread.factors[index] ** 2 * (MASSES * (table - mean) ** 2).sum(axis=1)
    return out


def band_of(*, runs: list[cache_module.RunCache]) -> np.ndarray:
    """Return the lead band of every stacked row."""
    return lead_band_of_hours(lead_hours=np.concatenate([run.lead_hours for run in runs]))


def twin_cells(*, site: str, seed: int) -> dict:
    """Run the twin checks of one site and return per-lead-band sums to pool over sites.

    Args:
        site: The substation's label.
        seed: The twin's seed.

    Returns:
        Sums and counts per lead band for tests 1, 2(a) to 2(d), and the full-spread twin's
        reference for 2(e).
    """
    twin = twin_module.build_twin(site=site, seed=seed)
    runs = twin.runs
    k0_table = post.residual_table(runs=runs, centre="mean")
    mp_table = post.residual_table(runs=runs, centre="proxy")
    band = band_of(runs=runs)
    out: dict = {"site": site, "eta_sd": twin.eta_sd}
    member_var1 = np.concatenate([run.corrected.var(axis=0, ddof=1) for run in runs])
    member_var0 = np.concatenate([run.corrected.var(axis=0, ddof=0) for run in runs])
    # Reference rows: the MC runs, whose rows are a stride of all rows.
    starts = np.cumsum([0, *[len(run.actual) for run in runs]])
    mc_rows = np.concatenate(
        [np.arange(starts[position], starts[position + 1]) for position in twin.reference_run]
    )
    mc_var = np.concatenate(twin.reference_variance)
    mc_var_half = np.concatenate(twin.reference_variance_half)
    out["mc"] = [
        [
            float(mc_var[band[mc_rows] == b].sum()),
            float(member_var1[mc_rows][band[mc_rows] == b].sum()),
            float(mc_var_half[band[mc_rows] == b].sum()),
            int((band[mc_rows] == b).sum()),
        ]
        for b in range(N_LEAD_BANDS)
    ]
    k0_v = np.concatenate([table_atom_variance(spread=k0_table, rows=run) for run in runs])
    mp_v = np.concatenate([table_atom_variance(spread=mp_table, rows=run) for run in runs])
    out["variance"] = [
        [
            float(member_var0[band == b].sum()),
            float(k0_v[band == b].sum()),
            float(mp_v[band == b].sum()),
            int((band == b).sum()),
        ]
        for b in range(N_LEAD_BANDS)
    ]
    mp_rows = post.build_rows(runs=runs, spread=mp_table, centre="members")
    mo_rows = post.build_rows(runs=runs, spread=None, centre="members")
    k0_rows = post.build_rows(runs=runs, spread=k0_table, centre="members")
    levels = [MIXTURE_LEVELS[i] for i in TAIL_POSITIONS]
    dates = (mp_rows.valid_time.astype("datetime64[D]")).astype(np.int64)
    out["mp_coverage"] = _coverage_cells(rows=mp_rows, band=band, dates=dates, levels=levels)
    out["k0_coverage"] = _coverage_cells(rows=k0_rows, band=band, dates=dates, levels=levels)
    inside = (mo_rows.actual >= mo_rows.quantiles[:, Q05]) & (
        mo_rows.actual <= mo_rows.quantiles[:, Q95]
    )
    out["mo_central90"] = [
        [int(inside[band == b].sum()), int((band == b).sum())] for b in range(N_LEAD_BANDS)
    ]
    # Halved members: MP-s over the s grid, factor 1.
    half = twin_module.build_twin(site=site, seed=seed, member_scale=0.5)
    half_table = post.residual_table(runs=half.runs, centre="proxy")
    half_band = band_of(runs=half.runs)
    out["half_mc"] = [
        [
            float(np.concatenate(half.reference_variance)[half_band_mc == b].sum()),
            float(np.concatenate(half.reference_variance_half)[half_band_mc == b].sum()),
            int((half_band_mc == b).sum()),
        ]
        for b in range(N_LEAD_BANDS)
        for half_band_mc in [_mc_band(twin=half, band=half_band)]
    ]
    out["s_surface"] = {}
    for s_value in post.S_GRID:
        rows = post.build_rows(
            runs=half.runs, spread=half_table, centre="members", s=[s_value] * N_LEAD_BANDS
        )
        tail = post.tail_pinball(rows=rows)
        out["s_surface"][str(s_value)] = [
            [float(tail[half_band == b].sum()), int((half_band == b).sum())]
            for b in range(N_LEAD_BANDS)
        ]
    return out


def _mc_band(*, twin: twin_module.Twin, band: np.ndarray) -> np.ndarray:
    starts = np.cumsum([0, *[len(run.actual) for run in twin.runs]])
    mc_rows = np.concatenate(
        [np.arange(starts[position], starts[position + 1]) for position in twin.reference_run]
    )
    return band[mc_rows]


def _coverage_cells(
    *, rows: post.Rows, band: np.ndarray, dates: np.ndarray, levels: list[float]
) -> list[list[list[float]]]:
    """Return, per lead band and tail level, the hits, rows, and distinct local dates."""
    cells = []
    for b in range(N_LEAD_BANDS):
        row_cells = []
        for position, level in zip(TAIL_POSITIONS, levels, strict=True):
            chosen = band == b
            hits = rows.actual[chosen] <= rows.quantiles[chosen, position]
            row_cells.append(
                [
                    float(hits.sum()),
                    float(chosen.sum()),
                    float(len(np.unique(dates[chosen]))),
                    level,
                ]
            )
        cells.append(row_cells)
    return cells


def _twin_job(site: str) -> dict:
    return twin_cells(site=site, seed=20_250_125)


def evaluate_twin(*, results: list[dict]) -> dict:
    """Pool the twin sums over the sites and apply the pass rules of tests 1 and 2."""
    mc = np.sum([np.array(r["mc"]) for r in results], axis=0)
    variance = np.sum([np.array(r["variance"]) for r in results], axis=0)
    eta_var = float(np.mean([r["eta_sd"] ** 2 for r in results]))
    sigma_w2 = mc[:, 0] / mc[:, 3]
    operational_w2 = mc[:, 1] / mc[:, 3]
    test1 = {
        "reference_sd": np.sqrt(sigma_w2).tolist(),
        "ratio": (operational_w2 / sigma_w2).tolist(),
    }
    tested = np.sqrt(sigma_w2) > MC_SD_FLOOR_POINTS
    test1["pass"] = bool(np.all(np.abs(np.array(test1["ratio"])[tested] - 1.0) <= TOLERANCE))
    members_var = variance[:, 0] / variance[:, 3]
    k0_atoms = variance[:, 1] / variance[:, 3]
    mp_atoms = variance[:, 2] / variance[:, 3]
    k0_total = members_var + k0_atoms
    mp_total = members_var + mp_atoms
    expected_mp_atoms = GAUSSIAN_11_ATOM_FACTOR * eta_var
    expected_k0_atoms = GAUSSIAN_11_ATOM_FACTOR * (sigma_w2 * (1 + 1 / 51) + eta_var)
    analytic_ratio = (sigma_w2 * (1 - 1 / 51) + expected_k0_atoms) / (
        sigma_w2 * (1 - 1 / 51) + expected_mp_atoms
    )
    ratio = k0_total / mp_total
    two_a = np.abs(mp_atoms / expected_mp_atoms - 1.0)
    two_b = np.abs(ratio / analytic_ratio - 1.0)
    # (c) calibration cells of MP, pooled over sites.
    cells = np.sum([np.array(r["mp_coverage"]) for r in results], axis=0)
    n_eff = cells[:, :, 2]
    coverage = cells[:, :, 0] / cells[:, :, 1]
    level = np.array(results[0]["mp_coverage"])[:, :, 3]
    in_band = np.abs(coverage - level) <= 2.5 * np.sqrt(level * (1 - level) / n_eff)
    cells_k0 = np.sum([np.array(r["k0_coverage"]) for r in results], axis=0)
    in_band_k0 = np.abs(cells_k0[:, :, 0] / cells_k0[:, :, 1] - level) <= 2.5 * np.sqrt(
        level * (1 - level) / cells_k0[:, :, 2]
    )
    # (d) MO shortfall.
    mo = np.sum([np.array(r["mo_central90"]) for r in results], axis=0)
    shortfall = 0.90 - mo[:, 0] / mo[:, 1]
    ratio_sd = np.sqrt(members_var / (sigma_w2 * (1 + 1 / 51) + eta_var))
    analytic_shortfall = 0.90 - (2 * np.array([NormalDist().cdf(1.645 * r) for r in ratio_sd]) - 1)
    two_d = shortfall > analytic_shortfall - 0.02
    # (e) recovery of the true factor.
    half_mc = np.sum([np.array(r["half_mc"]) for r in results], axis=0)
    s_star = np.sqrt(half_mc[:, 0] / half_mc[:, 1])
    surface = np.array(
        [
            [
                sum(r["s_surface"][str(s)][b][0] for r in results)
                / sum(r["s_surface"][str(s)][b][1] for r in results)
                for b in range(N_LEAD_BANDS)
            ]
            for s in post.S_GRID
        ]
    )
    s_hat = np.array(post.S_GRID)[surface.argmin(axis=0)]
    within = np.abs(s_hat - s_star) <= 0.25 * s_star
    in_grid = bool(np.all((s_star >= min(post.S_GRID)) & (s_star <= max(post.S_GRID))))
    return {
        "test1": test1,
        "test2a_ratio": (mp_atoms / expected_mp_atoms).tolist(),
        "test2a_pass": bool(np.all(two_a <= TOLERANCE)),
        "test2b_operational_ratio": ratio.tolist(),
        "test2b_analytic_ratio": analytic_ratio.tolist(),
        "test2b_pass": bool(np.all(two_b <= TOLERANCE)),
        "test2c_mp_cells_in_band": int(in_band.sum()),
        "test2c_k0_cells_in_band": int(in_band_k0.sum()),
        "test2c_mp_coverage": coverage.tolist(),
        "test2c_pass": bool(in_band.sum() >= CELLS_NEEDED),
        "test2d_shortfall": shortfall.tolist(),
        "test2d_analytic_shortfall": analytic_shortfall.tolist(),
        "test2d_pass": bool(two_d.all()),
        "test2e_s_star": s_star.tolist(),
        "test2e_s_hat": s_hat.tolist(),
        "test2e_s_star_in_grid": in_grid,
        "test2e_pass": bool(in_grid and within.sum() >= 3),
        "gaussian_11_atom_factor": GAUSSIAN_11_ATOM_FACTOR,
        "eta_variance": eta_var,
    }


def reproduction(*, sites: tuple[str, ...]) -> dict:
    """Run gate test 3 on the real cache."""
    tuned = arms.read_tuned()
    columns = ["q01", "q05", "q10", "q25", "q40", "q50", "q60", "q75", "q90", "q95", "q99"]
    diffs = {}
    for site in sites:
        runs = cache_module.read_cache(site=site, period="tuning")
        stored = arms.load_spread(arm="gamk", site=site)
        rows = post.build_rows(
            runs=runs,
            spread=stored.with_factors(factors=tuned["spread_factors"]["gamk"]),
            centre="members",
        )
        reference = pl.read_parquet(
            common.INPUT_WORK_DIR / "dp_preds" / f"gamk_tuning_{site}.parquet"
        )
        diffs[site] = float(np.abs(reference.select(columns).to_numpy() - rows.quantiles).max())
    rng = np.random.default_rng(0)
    one = np.sort(rng.normal(size=(7, 1, 11)), axis=2)
    identical = bool(
        np.allclose(mixture_quantiles(member_quantiles=np.repeat(one, 51, axis=1)), one[:, 0, :])
    )
    runs = cache_module.read_cache(site=sites[0], period="tuning")[:20]
    zero = replace(post.residual_table(runs=runs, centre="proxy"))
    zero = LeadHourSpread(
        by_lead_band=tuple(replace(band, table=band.table * 0.0) for band in zero.by_lead_band),
        factors=np.ones(N_LEAD_BANDS),
    )
    mp_zero = post.build_rows(runs=runs, spread=zero, centre="members").quantiles
    mo = post.build_rows(runs=runs, spread=None, centre="members").quantiles
    mp = post.residual_table(runs=runs, centre="proxy")
    s_one = post.build_rows(
        runs=runs, spread=mp, centre="members", s=[1.0] * N_LEAD_BANDS
    ).quantiles
    s_none = post.build_rows(runs=runs, spread=mp, centre="members").quantiles
    zero_fit = LeadHourSpread.fit(
        residuals=np.zeros(1000),
        band=np.zeros(1000, dtype=np.int64),
        quarter=np.zeros(1000, dtype=np.int64),
        lead_hours=np.linspace(0.5, 167.5, 1000),
    )
    return {
        "max_difference_to_experiment_111": diffs,
        "k0_pass": all(value < 1e-4 for value in diffs.values()),
        "identical_members_pass": identical,
        "mp_zero_table_equals_mo": bool(np.allclose(mp_zero, mo, atol=1e-9)),
        "mp_s1_equals_mp": bool(np.allclose(s_one, s_none, atol=1e-9)),
        "zero_residual_table_is_zero": bool(
            all(np.all(band.table == 0.0) for band in zero_fit.by_lead_band)
        ),
    }


def causality(*, site: str, indices: tuple[int, ...]) -> dict[str, Any]:
    """Run gate test 4: later actuals and later ENS runs must not move a run's forecast."""
    base = cache_module.prepare_site(site=site)
    results: dict[str, Any] = {}
    cache = base.cache
    features = cache.sites[site]
    runs_all = base.setup.indices(periods={"tuning"})
    for position in indices:
        index = runs_all[position]
        origin = base.setup.origins[index]
        cutoff = int(
            usable_cutoff_positions(
                end_times=cache.end_times,
                initialisation=np.array([naive_utc(moment=origin)]),
                delay_hours=3.5,
            )[0]
        )
        shifted_y = features.y.copy()
        shifted_y[cutoff + 1 :] += 10.0
        later = cache_module.prepare_site(
            site=site, cache=replace_site(cache=cache, site=site, y=shifted_y)
        )
        runs = [
            run
            if i <= index
            else EnsSiteRun(
                origin_position=run.origin_position,
                temperature_c=run.temperature_c + 3.0,
                shortwave_w_m2=run.shortwave_w_m2,
            )
            for i, run in enumerate(base.setup.runs)
        ]
        context = replace(
            base.setup.context, proxy=AnalysisProxy.build(runs=runs, n_rows=len(cache.end_times))
        )
        future_runs = cache_module.prepare_site(
            site=site, cache=cache, setup=replace(base.setup, runs=runs, context=context)
        )
        reference = cache_module.build_run(inputs=base, index=index)
        assert reference is not None
        changed = [
            cache_module.build_run(inputs=later, index=index),
            cache_module.build_run(inputs=future_runs, index=index),
        ]
        recent_y = features.y.copy()
        recent_y[cutoff - 47 : cutoff + 1] += 10.0
        control = cache_module.build_run(
            inputs=cache_module.prepare_site(
                site=site, cache=replace_site(cache=cache, site=site, y=recent_y)
            ),
            index=index,
        )
        changed = [run for run in changed if run is not None]
        results[str(index)] = {
            kind: bool(
                np.array_equal(reference.corrected, other.corrected)
                and np.array_equal(reference.level, other.level)
                and np.array_equal(reference.anomaly, other.anomaly)
            )
            for kind, other in zip(("later_actuals", "later_ens_runs"), changed, strict=True)
        }
        assert control is not None
        results[str(index)]["control_sees_a_change_before_the_cutoff"] = bool(
            not np.array_equal(reference.level, control.level)
        )
    results["pass"] = all(value for item in results.values() for value in item.values())
    return results


def mean_and_order(*, sites: tuple[str, ...]) -> dict:
    """Run gate test 5 on the real cache."""
    worst_mean = 0.0
    worst_order = 0.0
    for site in sites:
        runs = cache_module.read_cache(site=site, period="tuning")[:60]
        tables = {
            "k0": post.residual_table(runs=runs, centre="mean"),
            "mp": post.residual_table(runs=runs, centre="proxy"),
        }
        for run in runs:
            for scale in (1.0, 2.0, 3.0):
                moved = post.rescale_members(
                    corrected=run.corrected,
                    lead_hours=run.lead_hours,
                    s=[scale] * N_LEAD_BANDS,
                )
                worst_mean = max(
                    worst_mean, float(np.abs(moved.mean(axis=0) - run.corrected.mean(axis=0)).max())
                )
        arms_to_check: dict[str, dict[str, Any]] = {
            "k0": {"spread": tables["k0"], "centre": "members"},
            "mp": {"spread": tables["mp"], "centre": "members"},
            "mp_s": {"spread": tables["mp"], "centre": "members", "s": [2.0] * N_LEAD_BANDS},
            "mm": {"spread": tables["k0"], "centre": "mean"},
            "mo": {"spread": None, "centre": "members"},
            "mp_wide": {
                "spread": tables["mp"].with_factors(factors=[0.7, 1.5, 1.0, 1.3]),
                "centre": "members",
            },
        }
        for kwargs in arms_to_check.values():
            rows = post.build_rows(runs=runs, **kwargs)
            worst_order = min(worst_order, float(np.diff(rows.quantiles, axis=1).min()))
    return {
        "max_mean_difference": worst_mean,
        "min_quantile_step": worst_order,
        "pass": worst_mean < 1e-9 and worst_order >= 0.0,
    }


def main() -> None:
    """Run the gate and write `work/gate.json`; exit non-zero if a blocking test fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        twin_results = list(pool.map(_twin_job, common.DECISION_SITES))
    gate: dict = {"twin": evaluate_twin(results=twin_results)}
    gate["reproduction"] = reproduction(sites=common.SCORED_SITES)
    gate["causality"] = causality(site="D5", indices=(20, 100))
    gate["mean_and_order"] = mean_and_order(sites=("D5", "D11"))
    twin = gate["twin"]
    passes = {
        "1 variance transfer": twin["test1"]["pass"],
        "2a sigma_m recovered": twin["test2a_pass"],
        "2b double-count ratio": twin["test2b_pass"],
        "2c MP calibrated in 14 of 16": twin["test2c_pass"],
        "2d MO under-covers": twin["test2d_pass"],
        "2e s recovered": twin["test2e_pass"],
        "3 reproduction": all(
            value
            for key, value in gate["reproduction"].items()
            if key.endswith("pass") or key.startswith(("mp_", "identical", "zero"))
        ),
        "4 causality": gate["causality"]["pass"],
        "5 mean and order": gate["mean_and_order"]["pass"],
    }
    gate["passes"] = passes
    common.write_json_atomic(path=common.WORK_DIR / "gate.json", content=gate)
    print(json.dumps(passes, indent=2))
    sys.exit(0 if all(passes.values()) else 1)


if __name__ == "__main__":
    main()
