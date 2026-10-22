"""Experiment 244: score the deep ensembles, the band scaling E3, and the placebo P.

Reads `work/rows.parquet` and `work/preds_*.npz` (from `e00244_predict.py`) and writes
`work/report.json` and `work/eval.parquet`. Every printed number in the write-up comes from here.

    uv run --no-sync python e00244_score.py --workers 5
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00244_data as data_module
import e00244_fit as fit_module
import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import spearmanr

WORK_DIR = data_module.WORK_DIR
QUANTILES: Final = data_module.QUANTILES
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
FIT_POSITIONS: Final[tuple[int, ...]] = (0, 1, 2, 3, 7, 8, 9, 10)
TAIL_POSITIONS: Final = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}
DECISION: Final = ("D5", "D10", "D11", "D14", "D16")
WINTER: Final = (11, 12, 1, 2, 3)
NIGHT_BANDS: Final = (0, 3)
N_BOOT: Final[int] = 4000
N_SHUFFLES: Final[int] = 20
N_SHUFFLES_ACROSS_SITES: Final[int] = 5
INTERCEPT_RANGE: Final = (0.3, 3.0)
COEFFICIENT_MAX: Final[float] = 5.0
MINIMUM_OFFSET: Final[float] = 1e-3
KNOT_PERCENTILES: Final = (50.0, 90.0)
COMBOS: Final = (("ref", "seed"), ("ref", "bag"), ("err", "seed"), ("err", "bag"))


# ---------------------------------------------------------------- loading


def load_predictions(*, n_rows: int) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Per combination: out-of-sample c, s (20, n_rows) with NaN where the row is not predicted."""
    out = {}
    for kind, scheme in COMBOS:
        c = np.full((fit_module.N_SEEDS, n_rows), np.nan, np.float32)
        s = np.full_like(c, np.nan)
        ci = np.full_like(c, np.nan)
        si = np.full_like(c, np.nan)
        cin = np.full_like(c, np.nan)
        for group in ("full", "cfx", "cfy"):
            z = np.load(WORK_DIR / f"preds_{group}_{kind}_{scheme}.npz")
            c[:, z["row_id"]], s[:, z["row_id"]] = z["c"], z["s"]
            if group == "full":
                ci[:, z["row_id"]], si[:, z["row_id"]] = z["c_injected"], z["s_injected"]
        z = np.load(WORK_DIR / f"preds_fullin_{kind}_{scheme}.npz")
        cin[:, z["row_id"]] = z["c"]
        out[kind, scheme] = {"c": c, "s": s, "ci": ci, "si": si, "c_fullin": cin}
    return out


# ---------------------------------------------------------------- metrics


def pinball(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Twice the pinball loss per row and level."""
    diff = y[:, None] - q
    return 2 * np.maximum(LEVELS * diff, (LEVELS - 1) * diff)


def metrics(*, q: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Quantile score, and coverage and pinball loss at the four tail levels."""
    loss = pinball(q=q, y=y)
    out = {"score": float(loss.mean(axis=1).mean()), "n": len(y)}
    for name, position in TAIL_POSITIONS.items():
        out[f"cov_{name}"] = float((y <= q[:, position]).mean())
        out[f"pin_{name}"] = float(loss[:, position].mean())
    return out


class Days:
    """Resampling of valid days: a paired mean difference with a 95% interval."""

    def __init__(self, *, days: np.ndarray, seed: int = 0) -> None:  # noqa: D107
        self.codes = np.unique(days, return_inverse=True)[1]
        self.n = self.codes.max() + 1
        rng = np.random.default_rng(seed)
        self.draw = rng.integers(0, self.n, size=(N_BOOT, self.n))

    def mean_ci(self, *, values: np.ndarray, mask: np.ndarray) -> list[float]:
        """Mean of `values` over `mask` rows, with a day-resampled 95% interval."""
        sums = np.bincount(self.codes, weights=np.where(mask, values, 0.0), minlength=self.n)
        counts = np.bincount(self.codes, weights=mask.astype(float), minlength=self.n)
        total, number = sums[self.draw].sum(axis=1), counts[self.draw].sum(axis=1)
        est = total[number > 0] / number[number > 0]
        point = float(values[mask].mean())
        return [
            round(point, 4),
            round(float(np.percentile(est, 2.5)), 4),
            round(float(np.percentile(est, 97.5)), 4),
        ]

    def spearman_ci(self, *, a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> list[float]:
        """Spearman correlation over `mask` rows, with a day-resampled 95% interval."""
        idx_by_day = [np.flatnonzero(mask & (self.codes == d)) for d in range(self.n)]
        point = float(spearmanr(a[mask], b[mask]).statistic)
        rng = np.random.default_rng(1)
        est = []
        for _ in range(500):
            pick = rng.integers(0, self.n, size=self.n)
            idx = np.concatenate([idx_by_day[d] for d in pick])
            est.append(spearmanr(a[idx], b[idx]).statistic)
        return [
            round(point, 3),
            round(float(np.nanpercentile(est, 2.5)), 3),
            round(float(np.nanpercentile(est, 97.5)), 3),
        ]


# ---------------------------------------------------------------- E3 fit


def fit_level(*, level: float, residual, offset, design) -> np.ndarray:  # noqa: ANN001
    """Coefficients minimising the pinball loss of `mean + offset * (design @ w)` (experiment 169)."""
    n_rows, n_terms = design.shape
    tau = np.where(offset > 0, level, 1.0 - level)
    weight = np.abs(offset)
    cost = np.concatenate([np.zeros(n_terms), weight * tau, weight * (1.0 - tau)])
    identity = sparse.identity(n_rows, format="csr")
    equality = sparse.hstack([sparse.csr_matrix(design), identity, -identity], format="csr")
    bounds = (
        [INTERCEPT_RANGE] + [(0.0, COEFFICIENT_MAX)] * (n_terms - 1) + [(0.0, None)] * (2 * n_rows)
    )
    result = linprog(cost, A_eq=equality, b_eq=residual / offset, bounds=bounds, method="highs")
    if not result.success:
        msg = f"level {level}: {result.message}"
        raise RuntimeError(msg)
    return result.x[:n_terms]


def _job(task: tuple) -> tuple[tuple, list[float]]:
    key, level, residual, offset, design = task
    return key, fit_level(level=level, residual=residual, offset=offset, design=design).tolist()


def basis(*, z: np.ndarray, knots: tuple[float, float]) -> np.ndarray:
    """Constant plus one hinge per knot."""
    return np.stack(
        [np.ones_like(z), np.maximum(z - knots[0], 0), np.maximum(z - knots[1], 0)], axis=1
    )


def fit_many(
    *, pool: ProcessPoolExecutor, fits: dict[str, dict], tune: dict
) -> dict[str, dict[int, np.ndarray]]:
    """Fit E3-style scales for several spread columns at once; `fits[name]` holds `z` on tuning rows."""
    tasks = []
    for name, spec in fits.items():
        design = basis(z=spec["z"], knots=spec["knots"])
        for position in FIT_POSITIONS:
            offset = tune["offsets"][:, position]
            usable = np.abs(offset) > MINIMUM_OFFSET
            tasks.append(
                (
                    (name, position),
                    LEVELS[position],
                    tune["residual"][usable],
                    offset[usable],
                    design[usable],
                )
            )
    out: dict[str, dict[int, np.ndarray]] = {name: {} for name in fits}
    for (name, position), w in pool.map(_job, tasks):
        out[name][position] = np.array(w)
    return out


def apply_scale(
    *, mean: np.ndarray, offsets: np.ndarray, z: np.ndarray, coef: dict, knots
) -> np.ndarray:
    """Quantiles `mean + offset * scale`, sorted per row."""
    scale = np.ones_like(offsets)
    design = basis(z=z, knots=knots)
    for position, w in coef.items():
        scale[:, position] = design @ w
    return np.sort(mean[:, None] + offsets * scale, axis=1)


def shuffled(*, z: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute `z` among the rows of each group."""
    out = z.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = z[rng.permutation(idx)]
    return out


# ---------------------------------------------------------------- main


def main() -> None:  # noqa: PLR0915
    """Run the whole analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    workers = parser.parse_args().workers
    rows = pl.read_parquet(WORK_DIR / "rows.parquet")
    n = rows.height
    preds = load_predictions(n_rows=n)
    site = rows["site"].to_numpy()
    month = rows["month"].to_numpy()
    band = rows["hour_band"].to_numpy()
    y = rows["actual"].to_numpy().astype(np.float64)
    mean = rows["mean"].to_numpy().astype(np.float64)
    e0 = rows.select(list(QUANTILES)).to_numpy().astype(np.float64)
    e0_off = e0 - mean[:, None]
    d_q50 = rows["d_q50"].to_numpy().astype(np.float64)
    d_off = rows.select([f"d_off_{q}" for q in QUANTILES]).to_numpy().astype(np.float64)
    armd = d_q50[:, None] + d_off
    head_set = rows["head_set"].to_numpy()
    period = rows["period"].to_numpy()
    days = rows["day"].to_numpy().astype("datetime64[D]")
    cold_night = np.isin(month, WINTER) & rows["cold2"].to_numpy() & np.isin(band, NIGHT_BANDS)
    decision = np.isin(site, DECISION)
    scored = head_set == "full"
    tuning = period == "tuning"
    rep_even = (period == "replicate") & (head_set != "full")
    resampler = Days(days=days)
    report: dict = {"counts": {}}
    for name, m in (("scored", scored), ("replicate_even", rep_even), ("tuning", tuning)):
        report["counts"][name] = {
            "rows_decision": int((m & decision).sum()),
            "cold_night_rows": int((m & decision & cold_night).sum()),
            "cold_night_days": len(np.unique(days[m & decision & cold_night])),
        }

    # Spreads and centres.
    spread = {}
    for combo, p in preds.items():
        spread[combo] = {
            "c_std": np.nanstd(p["c"], axis=0),
            "s_std": np.nanstd(p["s"], axis=0),
            "c_mean": np.nanmean(p["c"], axis=0),
            "s_mean": np.nanmean(p["s"], axis=0),
            "c_inj_std": np.nanstd(p["ci"], axis=0),
            "c_in_std": np.nanstd(p["c_fullin"], axis=0),
        }

    # ---- Prediction 1: Spearman of spread against the absolute error of E0's median.
    abs_err = np.abs(y - e0[:, 5])
    p1: dict = {}
    for combo, sp in spread.items():
        for what in ("c_std", "s_std"):
            entry = {}
            for scope_name, m in (
                ("scored", scored),
                ("replicate_even", rep_even),
                ("tuning", tuning),
            ):
                mask = m & decision & cold_night
                entry[scope_name] = resampler.spearman_ci(a=sp[what], b=abs_err, mask=mask)
            p1[f"{combo[0]}_{combo[1]}_{what}"] = entry
    allrow = {}
    for scope_name, m in (("scored", scored), ("replicate_even", rep_even)):
        allrow[scope_name] = resampler.spearman_ci(
            a=spread["ref", "seed"]["c_std"], b=abs_err, mask=m & decision
        )
    report["prediction_1"] = p1
    report["spearman_all_rows_ref_seed"] = allrow
    print("prediction 1 (ref seed c_std, cold nights):", p1["ref_seed_c_std"], flush=True)

    # ---- Gate.
    gate = {}
    injected = rows["injected"].to_numpy()
    for combo in COMBOS:
        sp = spread[combo]
        entry = {}
        for name, mask in (
            ("registered_2pct", scored & injected),
            ("all_cold_night_scored", scored & cold_night),
        ):
            clean, inj = sp["c_std"], sp["c_inj_std"]
            ratio_vals = inj - clean
            entry[name] = {
                "rows": int(mask.sum()),
                "days": len(np.unique(days[mask])),
                "mean_clean": float(clean[mask].mean()),
                "mean_injected": float(inj[mask].mean()),
                "ratio_of_means": float(inj[mask].mean() / clean[mask].mean()),
                "mean_difference": resampler.mean_ci(values=ratio_vals, mask=mask),
            }
            # Ratio interval by resampling days.
            codes = resampler.codes
            sc = np.bincount(codes, weights=np.where(mask, clean, 0.0), minlength=resampler.n)
            si = np.bincount(codes, weights=np.where(mask, inj, 0.0), minlength=resampler.n)
            tot_c, tot_i = sc[resampler.draw].sum(axis=1), si[resampler.draw].sum(axis=1)
            ok = tot_c > 0
            ratios = tot_i[ok] / tot_c[ok]
            entry[name]["ratio_ci"] = [
                round(float(np.percentile(ratios, 2.5)), 3),
                round(float(np.percentile(ratios, 97.5)), 3),
            ]
        gate[f"{combo[0]}_{combo[1]}"] = entry
    report["gate"] = gate
    print("gate ref seed:", gate["ref_seed"], flush=True)

    # ---- Centres, E1 and the single networks (scored year).
    def e1_quantiles(combo: tuple[str, str]) -> np.ndarray:
        sp = spread[combo]
        return d_q50[:, None] + sp["c_mean"][:, None] + np.exp(sp["s_mean"])[:, None] * d_off

    scored_dec = scored & decision
    cold_dec = scored_dec & cold_night
    arms: dict[str, np.ndarray] = {"arm_d": armd, "E0": e0}
    for combo in COMBOS:
        arms[f"E1_{combo[0]}_{combo[1]}"] = e1_quantiles(combo)
    table: dict = {}
    for name, q in arms.items():
        ok = np.isfinite(q).all(axis=1)
        table[name] = {
            "all": metrics(q=q[scored_dec & ok], y=y[scored_dec & ok]),
            "cold": metrics(q=q[cold_dec & ok], y=y[cold_dec & ok]),
        }
    report["table"] = table

    # ---- Prediction 3: the single networks' gains over arm D.
    gains: dict = {}
    base = pinball(q=armd[scored_dec], y=y[scored_dec]).mean(axis=1)
    for kind in ("ref", "err"):
        p = preds[kind, "seed"]
        vals = []
        for j in range(fit_module.N_SEEDS):
            q = (
                d_q50[scored_dec][:, None]
                + p["c"][j, scored_dec][:, None]
                + np.exp(p["s"][j, scored_dec])[:, None] * d_off[scored_dec]
            )
            vals.append(float(base.mean() - pinball(q=q, y=y[scored_dec]).mean()))
        gains[kind] = {
            "gains": [round(v, 4) for v in vals],
            "sd": float(np.std(vals, ddof=1)),
            "mean": float(np.mean(vals)),
        }
    report["prediction_3"] = gains
    print("seed gains sd:", {k: v["sd"] for k, v in gains.items()}, flush=True)

    # ---- E3 fits.
    tune_mask = tuning & decision
    tune_mask_all = tuning  # the D9 rows are tuning rows too; E0's fit used the five decision sites
    tune = {
        "residual": (y - mean)[tune_mask],
        "offsets": e0_off[tune_mask],
    }
    del tune_mask_all
    specs: dict[str, dict] = {}
    norm: dict = {}
    for combo in COMBOS:
        sp = spread[combo]
        # Tuning rows use the cross-fit networks' spread; the scored year uses the full set's.
        med_cf = float(np.nanmedian(sp["c_std"][tune_mask]))
        med_full = float(np.nanmedian(sp["c_in_std"][tune_mask]))
        z = np.where(scored, sp["c_std"] / med_full, sp["c_std"] / med_cf)
        norm[f"{combo[0]}_{combo[1]}"] = {
            "median_crossfit_tuning": med_cf,
            "median_full_in_sample_tuning": med_full,
        }
        knots = tuple(float(np.percentile(z[tune_mask], p)) for p in KNOT_PERCENTILES)
        specs[f"{combo[0]}_{combo[1]}"] = {"z": z, "knots": knots}
    report["normalisation"] = norm
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fits = fit_many(
            pool=pool,
            fits={k: {"z": v["z"][tune_mask], "knots": v["knots"]} for k, v in specs.items()},
            tune=tune,
        )
        # Placebo fits: refit with shuffled tuning spreads.
        primary = specs["ref_seed"]
        group_within = np.array(
            [
                f"{a}|{b}|{c}|{d}"
                for a, b, c, d in zip(
                    site,
                    month,
                    band,
                    np.where(scored, "s", np.where(tuning, "t", "r")),
                    strict=True,
                )
            ]
        )
        group_across = np.array(
            [
                f"{b}|{c}|{d}"
                for b, c, d in zip(
                    month, band, np.where(scored, "s", np.where(tuning, "t", "r")), strict=True
                )
            ]
        )
        placebo_z: dict[str, np.ndarray] = {}
        for i in range(N_SHUFFLES):
            placebo_z[f"within_{i}"] = shuffled(
                z=primary["z"], groups=group_within, rng=np.random.default_rng(i)
            )
        for i in range(N_SHUFFLES_ACROSS_SITES):
            placebo_z[f"across_{i}"] = shuffled(
                z=primary["z"], groups=group_across, rng=np.random.default_rng(100 + i)
            )
        placebo_fits = fit_many(
            pool=pool,
            fits={k: {"z": v[tune_mask], "knots": primary["knots"]} for k, v in placebo_z.items()},
            tune=tune,
        )
    report["e3_coefficients"] = {
        k: {QUANTILES[pos]: [round(x, 4) for x in w] for pos, w in c.items()}
        for k, c in fits.items()
    }
    for key, spec in specs.items():
        arms[f"E3_{key}"] = apply_scale(
            mean=mean, offsets=e0_off, z=spec["z"], coef=fits[key], knots=spec["knots"]
        )
    # Placebo: per-row loss and coverage averaged over shuffles.
    placebo_loss = {"within": np.zeros(n), "across": np.zeros(n)}
    placebo_cov = {"within": np.zeros(n), "across": np.zeros(n)}
    placebo_all = {"within": np.zeros(n), "across": np.zeros(n)}
    placebo_metrics: dict = {}
    for kind_name, count in (("within", N_SHUFFLES), ("across", N_SHUFFLES_ACROSS_SITES)):
        per = []
        for i in range(count):
            key = f"{kind_name}_{i}"
            q = apply_scale(
                mean=mean,
                offsets=e0_off,
                z=placebo_z[key],
                coef=placebo_fits[key],
                knots=primary["knots"],
            )
            placebo_loss[kind_name] += pinball(q=q, y=y).mean(axis=1) / count
            placebo_cov[kind_name] += (y <= q[:, 9]) / count
            placebo_all[kind_name] += (y <= q[:, 1]) / count
            per.append(
                {
                    "cold_cov_q95": float((y <= q[:, 9])[cold_dec].mean()),
                    "all_score": float(pinball(q=q[scored_dec], y=y[scored_dec]).mean()),
                }
            )
        placebo_metrics[kind_name] = per
    report["placebo_per_shuffle"] = placebo_metrics

    # ---- Tables for E3 arms, contrasts.
    for name, q in arms.items():
        if name in table:
            continue
        table[name] = {
            "all": metrics(q=q[scored_dec], y=y[scored_dec]),
            "cold": metrics(q=q[cold_dec], y=y[cold_dec]),
        }
        table[name]["replicate_even_all"] = metrics(
            q=q[rep_even & decision], y=y[rep_even & decision]
        )
        table[name]["replicate_even_cold"] = metrics(
            q=q[rep_even & decision & cold_night], y=y[rep_even & decision & cold_night]
        )
    for name in ("E0",):
        table[name]["replicate_even_all"] = metrics(
            q=e0[rep_even & decision], y=y[rep_even & decision]
        )
        table[name]["replicate_even_cold"] = metrics(
            q=e0[rep_even & decision & cold_night], y=y[rep_even & decision & cold_night]
        )
    report["table"] = table

    loss = {
        k: pinball(q=q, y=y).mean(axis=1)
        for k, q in arms.items()
        if k.startswith("E3") or k == "E0"
    }
    cov95 = {
        k: (y <= q[:, 9]).astype(float) for k, q in arms.items() if k.startswith("E3") or k == "E0"
    }
    contrasts: dict = {}
    for scope_name, scope in (("scored", scored), ("replicate_even", rep_even)):
        mask_all, mask_cold = scope & decision, scope & decision & cold_night
        for key in ("ref_seed", "ref_bag", "err_seed", "err_bag"):
            e3 = f"E3_{key}"
            contrasts[f"{scope_name}/{e3}-E0"] = {
                "all_row_score": resampler.mean_ci(values=loss[e3] - loss["E0"], mask=mask_all),
                "cold_score": resampler.mean_ci(values=loss[e3] - loss["E0"], mask=mask_cold),
                "cold_q95_coverage": resampler.mean_ci(
                    values=cov95[e3] - cov95["E0"], mask=mask_cold
                ),
            }
        for kind_name in ("within", "across"):
            contrasts[f"{scope_name}/P_{kind_name}-E0"] = {
                "all_row_score": resampler.mean_ci(
                    values=placebo_loss[kind_name] - loss["E0"], mask=mask_all
                ),
                "cold_score": resampler.mean_ci(
                    values=placebo_loss[kind_name] - loss["E0"], mask=mask_cold
                ),
                "cold_q95_coverage": resampler.mean_ci(
                    values=placebo_cov[kind_name] - cov95["E0"], mask=mask_cold
                ),
            }
            contrasts[f"{scope_name}/E3_ref_seed-P_{kind_name}"] = {
                "all_row_score": resampler.mean_ci(
                    values=loss["E3_ref_seed"] - placebo_loss[kind_name], mask=mask_all
                ),
                "cold_q95_coverage": resampler.mean_ci(
                    values=cov95["E3_ref_seed"] - placebo_cov[kind_name], mask=mask_cold
                ),
            }
    report["contrasts"] = contrasts
    print("E3 ref seed vs E0 (scored):", contrasts["scored/E3_ref_seed-E0"], flush=True)

    # ---- Coverage by spread decile (scored year, primary spread).
    z0 = specs["ref_seed"]["z"]
    deciles: dict = {}
    for name, mask in (("cold_nights", cold_dec), ("all_rows", scored_dec)):
        edges = np.percentile(z0[mask], np.linspace(0, 100, 11))
        bins = np.clip(np.searchsorted(edges, z0[mask], side="right") - 1, 0, 9)
        idx = np.flatnonzero(mask)
        deciles[name] = [
            {
                "decile": d + 1,
                "rows": int((bins == d).sum()),
                "z_mean": float(z0[idx[bins == d]].mean()),
                "abs_error": float(abs_err[idx[bins == d]].mean()),
                "E0_cov_q95": float((y <= e0[:, 9])[idx[bins == d]].mean()),
                "E0_cov_q05": float((y <= e0[:, 1])[idx[bins == d]].mean()),
                "E3_cov_q95": float((y <= arms["E3_ref_seed"][:, 9])[idx[bins == d]].mean()),
            }
            for d in range(10)
        ]
    report["deciles"] = deciles

    # ---- Checks: centres per site, mean-forecast error, leave-one-site-out.
    checks: dict = {"median_minus_mean_by_site": {}, "centre_error": {}, "leave_one_site_out": {}}
    for s in sorted(set(site)):
        m = scored & (site == s)
        checks["median_minus_mean_by_site"][s] = {
            "E0_q50_minus_mean": float((e0[:, 5] - mean)[m].mean()),
            "E1_ref_seed_centre_minus_mean": float((arms["E1_ref_seed"][:, 5] - mean)[m].mean()),
        }
    for name, centre in (
        ("E0_mean", mean),
        ("E0_median", e0[:, 5]),
        ("E1_ref_seed_median", arms["E1_ref_seed"][:, 5]),
    ):
        err = y - centre
        checks["centre_error"][name] = {
            "mean_error": float(err[scored_dec].mean()),
            "mae": float(np.abs(err[scored_dec]).mean()),
            "cold_mean_error": float(err[cold_dec].mean()),
            "cold_mae": float(np.abs(err[cold_dec]).mean()),
        }
    for s in DECISION:
        m_all, m_cold = scored_dec & (site != s), cold_dec & (site != s)
        checks["leave_one_site_out"][s] = {
            "all_row_score_E3_minus_E0": float((loss["E3_ref_seed"] - loss["E0"])[m_all].mean()),
            "cold_q95_cov_E3_minus_E0": float((cov95["E3_ref_seed"] - cov95["E0"])[m_cold].mean()),
            "cold_q95_cov_E3": float(cov95["E3_ref_seed"][m_cold].mean()),
        }
    report["checks"] = checks
    per_site = {}
    for s in sorted(set(site)):
        m = scored & (site == s) & cold_night
        per_site[s] = {
            "cold_rows": int(m.sum()),
            "E0_cov_q95": float(cov95["E0"][m].mean()) if m.any() else None,
            "E3_cov_q95": float(cov95["E3_ref_seed"][m].mean()) if m.any() else None,
        }
    report["per_site_cold"] = per_site

    # ---- Decision.
    p1_ok = {
        scope: bool(p1["ref_seed_c_std"][scope][0] >= 0.2 and p1["ref_seed_c_std"][scope][1] > 0.1)
        for scope in ("scored", "replicate_even")
    }
    c3 = contrasts["scored/E3_ref_seed-E0"]
    cov_e3 = table["E3_ref_seed"]["cold"]["cov_q95"]
    cov_e0 = table["E0"]["cold"]["cov_q95"]
    lift = cov_e3 - cov_e0
    lift_p = float(np.mean([m["cold_cov_q95"] for m in placebo_metrics["within"]])) - cov_e0
    allcov = table["E3_ref_seed"]["all"]
    p2_ok = bool(
        cov_e3 >= 0.90
        and c3["all_row_score"][0] <= 0.01
        and abs(allcov["cov_q05"] - 0.05) <= 0.01
        and abs(allcov["cov_q95"] - 0.95) <= 0.01
    )
    gate_ok = bool(
        gate["ref_seed"]["registered_2pct"]["ratio_of_means"] >= 1.1
        and gate["ref_seed"]["registered_2pct"]["ratio_ci"][0] > 1.0
    )
    report["decision_inputs"] = {
        "gate_passes": gate_ok,
        "prediction_1_passes": p1_ok,
        "prediction_2_passes": p2_ok,
        "cold_q95_coverage_E0_E3": [cov_e0, cov_e3],
        "lift_E3": lift,
        "lift_P_within_mean": lift_p,
        "placebo_lift_under_half": bool(lift >= 0.005 and lift_p < lift / 2),
        "prediction_3_sd_ref": gains["ref"]["sd"],
    }
    print(json.dumps(report["decision_inputs"], indent=2), flush=True)
    (WORK_DIR / "report.json").write_text(json.dumps(report, indent=2))
    eval_frame = rows.select(
        [
            "row_id",
            "site",
            "period",
            "head_set",
            "valid",
            "day",
            "month",
            "hour_band",
            "lead_hours",
            "actual",
            "mean",
            "cold2",
        ]
    ).with_columns(
        cold_night=pl.Series(cold_night),
        z=pl.Series(z0),
        c_std=pl.Series(spread["ref", "seed"]["c_std"]),
        s_std=pl.Series(spread["ref", "seed"]["s_std"]),
        abs_err=pl.Series(abs_err),
        loss_E0=pl.Series(loss["E0"]),
        loss_E3=pl.Series(loss["E3_ref_seed"]),
        loss_P=pl.Series(placebo_loss["within"]),
        cov95_E0=pl.Series(cov95["E0"]),
        cov95_E3=pl.Series(cov95["E3_ref_seed"]),
        cov95_P=pl.Series(placebo_cov["within"]),
    )
    eval_frame.write_parquet(WORK_DIR / "eval.parquet")


if __name__ == "__main__":
    main()
