# ruff: noqa: E501
"""Step 3 of experiment 00348: score every arm and print `report.txt`, the numbers the page quotes.

Scores the 53 scored Tuesday runs (kept rows; headline scope true days 1 to 10, lead hours 24 to
240; days 1 to 2, days 3 to 10, and every lead beside). Intervals resample whole calendar months
(50,000 resamples) at 95% and at the level 1 - 0.01/5 for five arms. Per-seed scores are averaged
within each row before resampling. Reuses experiment 00347's scoring helpers.

    PYTHONPATH=<348>:<347>:<230>:<303>:<310>:<326>:<336 folders> \
    uv run --no-sync python ml-experiments/00348-*/e00348_score.py
"""

import json
import os
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_score as score326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_score as s4  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00348_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

LEVEL: Final[float] = 1 - 0.01 / 5
s4.LEVEL = LEVEL  # the shared resampler reads the Bonferroni level for five arms
ORDER: Final[tuple[str, ...]] = ("z4", "z5", "z3", "z2", "z1")
HEAD: Final[tuple[str, ...]] = ("y2", "y2p", "z1", "z2", "z3", "z4")
MIN_GAIN: Final[dict[str, float]] = {"z1": 0.03, "z2": 0.02, "z3": 0.03, "z4": 0.03, "z5": 0.02}
HEADLINE_TOL: Final[float] = 0.02
TAIL_TOL: Final[float] = 0.05
REFERENCES: Final[tuple[str, ...]] = ("default", "default_notable", "c", "W4", "W0")
LABELS: Final[dict[str, str]] = {
    "default": "default (Wind farms research default 1)",
    "default_notable": "XGBoost alone",
    "c": "XGBoost + adapter column (326 c)",
    "W4": "physics core + adapter + MDN (324 W4)",
    "W0": "climatology (324 W0)",
    "y2": "y2 refit (347)",
    "y2_347": "y2 as stored by 347",
    "y2p": "y2p (y2, pinball loss, weights 1)",
    "z1": "z1 weighted pinball",
    "z2": "z2 point mass + cap",
    "z3": "z3 monotone tail increments",
    "z4": "z4 = z1 + z2",
    "z5": "z5 blend (band + XGBoost)",
    "z5a": "z5a plant + offsets",
    "z5b": "z5b plant + residual band",
}


def fmt(entry: dict) -> str:
    """Return 'mean (95% low to high; 99.8% low to high)' for an interval entry."""
    low, high = entry["0.9500"]
    wide_low, wide_high = entry[f"{LEVEL:.4f}"]
    return f"{entry['mean']:+.3f} (95% {low:+.3f} to {high:+.3f}; 99.8% {wide_low:+.3f} to {wide_high:+.3f})"


def seeds_of(name: str) -> tuple[int, ...]:
    """Return the seeds stored for an arm: five for heads, one for z5 and its parts."""
    return (0,) if name.startswith("z5") else common.SEEDS


def load(*, name: str) -> np.ndarray:
    """Return an arm's stored quantiles (seeds, rows, 11)."""
    return np.stack([np.load(common.PARTS_DIR / f"{name}_h{s}.npy") for s in seeds_of(name)])


def mean_qs(*, q: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return per-row quantile score (points) averaged over seeds, as (1, rows)."""
    return (
        np.mean(
            [c303.quantile_score(quantiles=x.astype(np.float64), actual=actual) for x in q], axis=0
        )[None]
        * c303.POINTS_PER_FRACTION
    )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Score everything and write the report."""
    cases = data.scored_frame().with_columns(lead_day=pl.col("lead_day").clip(0, 10))
    actual = cases["actual_norm"].to_numpy().astype(np.float64)
    sites = cases["site"].to_numpy()
    arms: dict[str, np.ndarray] = {}
    for name in (*HEAD, "z5", "z5a", "z5b"):
        arms[name] = load(name=name)
        if name in HEAD:
            arms[f"{name}_placebo"] = load(name=f"{name}_placebo")
    for name in (*(a for a in HEAD),):
        arms[f"{name}_basefarm"] = load(name=f"{name}_basefarm")
    arms["y2_347"] = np.stack([np.load(data.PARTS_DIR / f"y2_h{s}.npy") for s in common.SEEDS])
    arms["default"] = np.load(data.E339_FINAL / "default.npy")
    arms["default_notable"] = np.load(data.E339_FINAL / "default_notable.npy")
    arms["c"] = np.load(data.E326_FINAL / "c.npy")
    arms["W4"] = np.load(data.E324_FINAL / "W4.npy")
    arms["W0"] = np.load(data.E324_FINAL / "W0.npy")
    flags = {
        "kept": ~cases["masked"].to_numpy(),
        "true_1_10": (~cases["masked"] & cases["lead_hours"].is_between(24, 240)).to_numpy(),
        "days_1_2": (~cases["masked"] & cases["lead_day"].is_between(1, 2)).to_numpy(),
        "days_3_10": (~cases["masked"] & cases["lead_day"].is_between(3, 10)).to_numpy(),
        "all": np.ones(cases.height, dtype=bool),
    }
    subsets = tuple(flags)
    resampler = s4.Intervals(month=cases["month"].to_numpy(), seeds=1)
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for name, q in arms.items():
        raw = {
            **score303.row_metrics(quantiles=q, actual=actual),
            **score326.extra_metrics(quantiles=q, actual=actual),
        }
        raw.update(s4.squared_errors(quantiles=q, actual=actual))
        metrics[name] = {k: v.mean(axis=0, keepdims=True) for k, v in raw.items()}
        metrics[name]["seed_qs_kept"] = raw["qs"][:, flags["kept"]].mean(axis=1)[None]
    qs = {n: m["qs"] for n, m in metrics.items()}
    lines = ["EXPERIMENT 348: upper-tail fixes for the frozen physics plant plus mixture head"]
    lines.append(
        f"Scored cases {cases.height} ({int(flags['kept'].sum())} kept, {int(flags['true_1_10'].sum())} kept at true days 1 to 10)."
    )
    report: dict = {}

    lines.append(
        "-- absolute quantile scores, points of capacity (seed mean); seed sd on kept rows --"
    )
    lines.append("arm".ljust(46) + "".join(s.rjust(11) for s in subsets) + "   seed sd")
    for name in LABELS:
        if name not in qs:
            continue
        row = "".join(f"{qs[name][0, flags[s]].mean():11.3f}" for s in subsets)
        seeds = metrics[name]["seed_qs_kept"][0]
        sd = f"{np.std(seeds, ddof=1):.3f}" if len(seeds) > 1 else "n/a"
        lines.append(f"{LABELS.get(name, name)[:45].ljust(46)}{row}   {sd}")
    report["scores"] = {n: {s: float(qs[n][0, flags[s]].mean()) for s in subsets} for n in qs}

    lines.append("-- by farm, true days 1 to 10 (points) --")
    for name in ("default", "y2", *ORDER[::-1], "z5a", "z5b"):
        cols = "  ".join(
            f"{f} {qs[name][0, flags['true_1_10'] & (sites == f)].mean():.3f}" for f in common.FARMS
        )
        lines.append(f"{LABELS.get(name, name)[:45].ljust(46)}{cols}")

    contrasts: dict = {}
    lines.append(
        "-- contrasts, arm minus comparison, quantile score in points (negative favours the arm) --"
    )
    for arm in ("y2", "y2p", "z1", "z2", "z3", "z4", "z5", "z5a", "z5b"):
        for ref in ("default", "y2", "default_notable", "c", "W4"):
            if ref == arm or (ref == "y2" and arm == "y2"):
                continue
            for subset in ("true_1_10", "days_1_2", "days_3_10", "kept"):
                entry = resampler.read(values=qs[arm] - qs[ref], keep=flags[subset])
                contrasts[f"{arm} - {ref} | {subset}"] = entry
                lines.append(f"{arm} minus {ref}, {subset}: {fmt(entry)}")
    report["contrasts"] = contrasts

    lines.append(
        "-- tail tables: shares below q01, q05 and above q95, q99 (percent; nominal 1, 5, 5, 1), pinball losses, 5-95 coverage (percent; nominal 90) --"
    )
    tails: dict = {}
    tail_names = (
        "default",
        "default_notable",
        "c",
        "W4",
        "y2",
        "y2p",
        "z1",
        "z2",
        "z3",
        "z4",
        "z5",
        "z5a",
        "z5b",
        "z1_basefarm",
        "z2_basefarm",
        "z3_basefarm",
        "z4_basefarm",
        "y2_basefarm",
        "z4_placebo",
    )
    for scope in ("true_1_10", "kept"):
        lines.append(f"scope {scope}")
        lines.append(
            "arm".ljust(46)
            + "".join(
                h.rjust(9)
                for h in ("<q01", "<q05", ">q95", ">q99", "p01", "p05", "p95", "p99", "cover")
            )
        )
        for name in tail_names:
            m, k = metrics[name], flags[scope]
            values = [
                100 * m[key][0, k].mean()
                for key in ("below_q01", "below_q05", "above_q95", "above_q99")
            ]
            values += [m[key][0, k].mean() for key in ("p01", "p05", "p95", "p99")] + [
                100 * m["covered"][0, k].mean()
            ]
            tails.setdefault(scope, {})[name] = [float(v) for v in values]
            lines.append(
                f"{LABELS.get(name, name)[:45].ljust(46)}" + "".join(f"{v:9.3f}" for v in values)
            )
    lines.append("share above q99 by farm, true days 1 to 10 (percent) and 0.99 pinball loss")
    by_farm: dict = {}
    for name in ("default", "y2", "z1", "z2", "z3", "z4", "z5"):
        cols = []
        for f in common.FARMS:
            k = flags["true_1_10"] & (sites == f)
            above, p99 = (
                100 * metrics[name]["above_q99"][0, k].mean(),
                metrics[name]["p99"][0, k].mean(),
            )
            by_farm[f"{name}|{f}"] = [float(above), float(p99)]
            cols.append(f"{f} {above:.2f}% / {p99:.3f}")
        lines.append(f"{LABELS.get(name, name)[:45].ljust(46)}" + "   ".join(cols))
    report["tails"], report["by_farm_tail"] = tails, by_farm

    lines.append(
        "-- checks: median forecast minus mean forecast per farm, true days 1 to 10 (points) --"
    )
    for name in ("default", "y2", "z1", "z2", "z3", "z4", "z5"):
        cols = "  ".join(
            f"{f} {metrics[name]['median_minus_mean'][0, flags['true_1_10'] & (sites == f)].mean():+.2f}"
            for f in common.FARMS
        )
        lines.append(f"{LABELS.get(name, name)[:45].ljust(46)}{cols}")
    lines.append(
        "-- checks: mean-forecast squared error contrast beside the quantile-score contrast (arm minus default, true days 1 to 10) --"
    )
    trigger3 = {}
    for arm in ("y2", "z1", "z2", "z3", "z4", "z5"):
        qsd = resampler.read(values=qs[arm] - qs["default"], keep=flags["true_1_10"])["mean"]
        sq = resampler.read(
            values=metrics[arm]["mean_sq_error"] - metrics["default"]["mean_sq_error"],
            keep=flags["true_1_10"],
        )["mean"]
        md = resampler.read(
            values=metrics[arm]["median_abs_error"] - metrics["default"]["median_abs_error"],
            keep=flags["true_1_10"],
        )["mean"]
        trigger3[arm] = {
            "qs": qsd,
            "mean_sq": sq,
            "median_abs": md,
            "opposite": bool((qsd < 0) != (sq < 0)),
        }
        lines.append(
            f"{arm}: quantile score {qsd:+.3f}; mean-forecast squared error {sq:+.2f}; median absolute error {md:+.3f}; opposite signs: {trigger3[arm]['opposite']}"
        )
    report["trigger3"] = trigger3

    lines.append("-- controls --")
    gate = abs(float(qs["y2"][0, flags["kept"]].mean() - qs["y2_347"][0, flags["kept"]].mean()))
    lines.append(
        f"reproduction gate: y2 refit {qs['y2'][0, flags['kept']].mean():.3f} vs 347 {qs['y2_347'][0, flags['kept']].mean():.3f} (difference {gate:.4f}, limit 0.01)"
    )
    report["gate"] = gate
    base_share = {}
    for arm in HEAD:
        gain = (qs[arm] - qs["default"])[0, flags["true_1_10"]].mean()
        base = (qs[f"{arm}_basefarm"] - qs["default"])[0, flags["true_1_10"]].mean()
        plc = (qs[f"{arm}_placebo"] - qs["default"])[0, flags["true_1_10"]].mean()
        base_share[arm] = {
            "gain": float(gain),
            "base_gain": float(base),
            "share": float(base / gain) if abs(gain) > 1e-9 else None,
            "placebo_minus_default": float(plc),
        }
        lines.append(
            f"{arm}: minus default {gain:+.3f}; base-only head {base:+.3f} ({100 * base / gain:.0f}% of the gain if the gain is negative); shuffled-summary placebo {plc:+.3f}"
        )
    report["controls"] = base_share

    lines.append(
        "-- replicate: heads trained on 2024-04 to 2024-11, scored on the daily runs 2025-01 to 2025-06  --"
    )
    rep_cases = pl.read_parquet(data.E336_FINAL / "replicate_cases.parquet").with_columns(
        lead_day=pl.col("lead_day").clip(0, 10)
    )
    rep_actual = rep_cases["actual_norm"].to_numpy().astype(np.float64)
    rep_true = (~rep_cases["masked"] & rep_cases["lead_hours"].is_between(24, 240)).to_numpy()
    rep_kept = (~rep_cases["masked"]).to_numpy()
    rep_resampler = s4.Intervals(month=rep_cases["month"].to_numpy(), seeds=1)
    default_rep = np.load(data.E339_FINAL / "default_rep.npy")
    rep_default = mean_qs(q=default_rep, actual=rep_actual)
    replicate = {}
    for arm in (*HEAD, "z5"):
        rep_seeds = (0,) if arm == "z5" else common.REPLICATE_SEEDS
        q_rep = np.stack([np.load(common.PARTS_DIR / f"{arm}_rep_h{s}.npy") for s in rep_seeds])
        rep_qs = mean_qs(q=q_rep, actual=rep_actual)
        months = np.unique(rep_cases["month"].to_numpy())
        monthly = np.array(
            [
                (rep_qs - rep_default)[0, rep_true & (rep_cases["month"].to_numpy() == m)].mean()
                for m in months
            ]
        )
        test = score326.t_test(monthly=monthly)
        entry = {
            "true_1_10": rep_resampler.read(values=rep_qs - rep_default, keep=rep_true),
            "kept": rep_resampler.read(values=rep_qs - rep_default, keep=rep_kept),
            "t_test": test,
            "monthly": monthly.tolist(),
            "score_true": float(rep_qs[0, rep_true].mean()),
            "default_true": float(rep_default[0, rep_true].mean()),
        }
        replicate[arm] = entry
        lines.append(
            f"{arm} replicate, true days 1 to 10: score {entry['score_true']:.3f} (default {entry['default_true']:.3f}); minus default {fmt(entry['true_1_10'])}; kept all leads {fmt(entry['kept'])}; monthly {[round(float(v), 3) for v in monthly]}; t {test['t']:+.2f}, p {test['p_two_sided']:.3f}"
        )
    report["replicate"] = replicate

    lines.append(
        "-- adoption conditions, in the order z4, z5, z3, z2, z1 (headline scope true days 1 to 10) --"
    )
    adopted, verdicts = None, {}
    for arm in ORDER:
        entry = contrasts[f"{arm} - default | true_1_10"]
        low, high = entry[f"{LEVEL:.4f}"]
        c1 = bool(high < 0 and -entry["mean"] >= MIN_GAIN[arm])
        harm = {}
        for key in ("p01", "p99"):
            diff = resampler.read(
                values=metrics[arm][key] - metrics["default"][key], keep=flags["true_1_10"]
            )
            harm[key] = diff["upper_95_one_sided"]
        c2 = bool(all(v <= TAIL_TOL for v in harm.values()))
        if arm in replicate:
            t = replicate[arm]["t_test"]
            c3 = bool(
                replicate[arm]["true_1_10"]["mean"] < 0  # ty: ignore[not-subscriptable]
                and not (t["t"] > 0 and t["p_two_sided"] < 0.05)  # ty: ignore[not-subscriptable]
            )
        else:
            c3 = False
        placebo_loses = (
            bool(
                (qs[f"{arm}_placebo"] - qs["default"])[0, flags["true_1_10"]].mean()
                > -0.5 * abs(entry["mean"])
            )
            if arm in HEAD
            else True
        )
        seed_sd = (
            float(np.std(metrics[arm]["seed_qs_kept"][0], ddof=1))
            if len(metrics[arm]["seed_qs_kept"][0]) > 1
            else 0.0
        )
        headline_harm = resampler.read(values=qs[arm] - qs["default"], keep=flags["true_1_10"])[
            "upper_95_one_sided"
        ]
        all_harm = resampler.read(values=qs[arm] - qs["default"], keep=flags["all"])[
            "upper_95_one_sided"
        ]
        c2 = c2 and bool(headline_harm <= HEADLINE_TOL and all_harm <= TAIL_TOL)
        c4 = bool(gate <= 0.01 and placebo_loses and seed_sd < 0.5 * abs(entry["mean"]))
        verdicts[arm] = {
            "c1": c1,
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "mean": entry["mean"],
            "interval": [low, high],
            "harm_upper": harm,
        }
        lines.append(
            f"{arm}: condition 1 {c1} (gain {-entry['mean']:+.3f}, minimum {MIN_GAIN[arm]}, 99.8% interval {low:+.3f} to {high:+.3f}); condition 2 {c2} (one-sided 95% upper bound of harm: p01 {harm['p01']:+.3f}, p99 {harm['p99']:+.3f}, limit {TAIL_TOL}); condition 3 {c3}; condition 4 gate {c4}"
        )
        if arm == "z5":
            lines.append(
                "z5's replicate was run after the main scores were read (post hoc), so z5 is at most a candidate"
            )
        if adopted is None and c1 and c2 and c3 and c4:
            adopted = arm
    lines.append(
        f"first arm passing conditions 1 to 4: {adopted} (condition 5 is the page's serving text; audit triggers are judged in the write-up)"
    )
    report["adoption"], report["adopted"] = verdicts, adopted

    best = min(("y2", "z1", "z2", "z3", "z4"), key=lambda a: qs[a][0, flags["true_1_10"]].mean())
    families = {
        n: float(qs[n][0, flags["true_1_10"]].mean())
        for n in ("default", "default_notable", "c", "W4")
    }
    best_family = min(families, key=lambda n: families[n])
    arm_score = float(qs[best][0, flags["true_1_10"]].mean())
    lines.append(
        f"standard architecture (physics plant + MDN head): best head arm {best} {arm_score:.3f}; best family row {best_family} {families[best_family]:.3f}; gap {arm_score - families[best_family]:+.3f} points, {100 * (arm_score - families[best_family]) / families[best_family]:+.1f}%"
    )
    report["standard"] = {
        "best": best,
        "score": arm_score,
        "best_family": best_family,
        "family_score": families[best_family],
    }

    out = common.DATA_DIR / "report.txt"
    out.write_text("\n".join(lines) + "\n")
    (common.DATA_DIR / "score.json").write_text(json.dumps(report, indent=1, default=float))
    keep_cols = {
        "site": cases["site"],
        "month": cases["month"],
        "lead_hours": cases["lead_hours"],
        "masked": cases["masked"],
        "actual": cases["actual_norm"],
    }
    for name in ("default", "y2", "z1", "z2", "z3", "z4", "z5"):
        keep_cols[f"qs__{name}"] = pl.Series(qs[name][0])
        for k, level in ((5, "q50"), (9, "q95"), (10, "q99")):
            keep_cols[f"{level}__{name}"] = pl.Series(
                arms[name].astype(np.float64).mean(axis=0)[:, k]
            )
    pl.DataFrame(keep_cols).write_parquet(common.DATA_DIR / "rows.parquet")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
