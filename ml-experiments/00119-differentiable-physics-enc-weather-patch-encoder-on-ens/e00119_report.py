"""Print every number the experiment 119 page quotes into `report.md`.

uv run python ml-experiments/00119-*/e00119_report.py
"""

import json
from pathlib import Path

import e00119_common as common

HERE = Path(__file__).resolve().parent
CONTRASTS = {
    "C1": "ENC-c minus GAM-K",
    "C2": "ENC-c minus ENC-c0",
    "C3": "ENC-c minus ENC-cpt",
    "C3b": "ENC-c minus ENC-cps",
    "C4": "ENC-r minus GAM-K",
}
SCOPES = (
    "0.5-168 h",
    "D+1 (15.5-39 h)",
    "0-24 h",
    "24-48 h",
    "48-144 h",
    "144-168 h",
    "winter (Nov-Mar)",
    "winter evening peak (16-20 UTC)",
    "summer (Apr-Oct)",
    "working days",
    "weekends and bank holidays",
)


def load(name: str) -> dict:
    """Return a stored JSON result."""
    return json.loads((common.WORK_DIR / name).read_text())


def fmt(item: dict) -> str:
    """Format an estimate with its interval."""
    return f"{item['difference']:+.3f} [{item['lower']:+.3f}, {item['upper']:+.3f}]"


def main() -> None:  # noqa: PLR0915
    """Write the report."""
    out: list[str] = []
    for direction in ("F1", "F2"):
        r = load(f"results_{direction}.json")
        e = load(f"extra_{direction}.json")
        out.append(
            f"## Direction {direction} ({r['n_runs']} runs, {r['n_rows']:,} decision-site rows)\n"
        )
        out.append("### Planned contrasts, 99% intervals, points of capacity\n")
        out.append(
            "| Contrast | Pair | Annual | Day ahead | Winter | MAE of q50 | Sites better "
            "| Leave-one-out C1 range |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for name, pair in CONTRASTS.items():
            c = r["contrasts"][name]
            better = sum(1 for s, v in c["per_site"].items() if s != "D9" and v < 0)
            loo = ""
            if name == "C1":
                values = [v["C1"]["difference"] for v in r["leave_one_site_out"].values()]
                loo = f"{min(values):+.3f} to {max(values):+.3f}"
            out.append(
                f"| {name} | {pair} | {fmt(c['by_scope']['0.5-168 h'])} "
                f"| {fmt(c['by_scope']['D+1 (15.5-39 h)'])} "
                f"| {fmt(c['by_scope']['winter (Nov-Mar)'])} "
                f"| {c['mae_median']['difference']:+.3f} | {better} of 5 | {loo} |"
            )
        out.append("\n### Every scope, C1 (ENC-c minus GAM-K)\n")
        out.append("| Scope | Estimate [99% interval] |")
        out.append("|---|---|")
        out.extend(f"| {s} | {fmt(r['contrasts']['C1']['by_scope'][s])} |" for s in SCOPES)
        out.append("\n### Each arm minus GAM-K (annual), and absolute scores\n")
        out.append(
            "| Arm | Annual [99% interval] | Winter | Absolute score, annual | Winter | Day ahead |"
        )
        out.append("|---|---|---|---|---|---|")
        a = r["absolute"]
        out.append(
            f"| GAM-K | 0 | 0 | {a['K0']['0.5-168 h']:.3f} | {a['K0']['winter (Nov-Mar)']:.3f} "
            f"| {a['K0']['D+1 (15.5-39 h)']:.3f} |"
        )
        for arm, v in e["arm_vs_k0"].items():
            out.append(
                f"| {arm} | {fmt(v['0.5-168 h'])} | {v['winter (Nov-Mar)']['difference']:+.3f} | "
                f"{a[arm]['0.5-168 h']:.3f} | {a[arm]['winter (Nov-Mar)']:.3f} "
                f"| {a[arm]['D+1 (15.5-39 h)']:.3f} |"
            )
        g = e["gam0"]
        out.append(
            "\nGAM-0 (experiment 111, no Kalman filter) on the same rows: "
            f"{g['gam0']:.3f} against GAM-K {g['k0']:.3f}.\n"
        )
        out.append("### Hard days (7-day block bootstrap, 99%), descriptive\n")
        out.append("| Subset | Days | C1 | C3 |")
        out.append("|---|---|---|---|")
        for key in ("coldest", "showery", "largest_error"):
            c1 = r["contrasts"]["C1"]["hard_days"][key]
            c3 = r["contrasts"]["C3"]["hard_days"][key]
            out.append(
                f"| {key} | {c1['n_days']} "
                f"| {c1['difference']:+.3f} [{c1['lower']:+.3f}, {c1['upper']:+.3f}] "
                f"| {c3['difference']:+.3f} [{c3['lower']:+.3f}, {c3['upper']:+.3f}] |"
            )
        out.append("\n### Tail coverage (share at or below the quantile), annual, decision sites\n")
        out.append("| Arm | q01 | q05 | q95 | q99 |")
        out.append("|---|---|---|---|---|")
        for arm in ("K0", "ENC-c", "ENC-cpt"):
            cells = " | ".join(f"{t['coverage']:.3f}" for t in r["tails"][arm]["0.5-168 h"])
            out.append(f"| {arm} | {cells} |")
        out.append(
            "\n### Median minus mean forecast, points, by site (GAM-K's, the same for every arm)\n"
        )
        mmm = r["median_minus_mean"]["K0"]
        out.append(", ".join(f"{s} {v:+.3f}" for s, v in mmm.items()) + "\n")
        out.append("### Five seeds, annual absolute score\n")
        for arm, values in r["per_seed_absolute"].items():
            out.append(f"- {arm}: " + ", ".join(f"{v:.3f}" for v in values))
        out.append("")
        sign = {
            n: (r["contrasts"][n]["by_scope"]["0.5-168 h"]["difference"] < 0)
            == (r["contrasts"][n]["mae_median"]["difference"] < 0)
            for n in ("C1", "C2", "C3", "C3b", "C4")
        }
        out.append(f"Quantile score and MAE of the median agree in sign: {sign}\n")
    decisions = load("decisions.json")
    out.append("## Settings chosen on each training year\n")
    for key in ("F1.ENC-c", "F1.ENC-r", "F2.ENC-c", "F2.ENC-r"):
        d = decisions[key]
        out.append(f"- {key}: c {d['c']}, epochs {d['epochs']}, lambda {d['shrink']}")
    out.append(f"- Parameters: {decisions['parameters']}")
    patches = json.loads(common.patch_path().read_text())["patches"]
    out.append("\n## Shared cells between the six patches (of 37)\n")
    names = common.SCORED_SITES
    for a_name in names:
        counts = [len(set(patches[a_name]["cells"]) & set(patches[b]["cells"])) for b in names]
        out.append(
            f"- {a_name}: " + ", ".join(f"{b} {c}" for b, c in zip(names, counts, strict=True))
        )
    out.append(
        f"\nPlacebo ENC-cps partner (farthest decision site): {decisions['placebo_distant_site']}"
    )
    for run in ("gate_run1.json", "gate_run2.json"):
        gate = json.loads((HERE / run).read_text())
        out.append(f"\n## Gate {run}\n")
        out.append("| Injection | Recovered share | Sites with half recovered | Pooled interval |")
        out.append("|---|---|---|---|")
        for name, v in gate["injections"].items():
            out.append(
                f"| {name} | {v['recovered_share']:.2f} | {v['sites_recovering_half']} of 5 | "
                f"[{v['contrast']['lower']:+.3f}, {v['contrast']['upper']:+.3f}] |"
            )
        out.extend(
            f"\n{key}: " + ", ".join(f"{a} {v['gain']:+.4f}" for a, v in gate[key].items())
            for key in ("clean", "null", "permuted")
            if key in gate
        )
        if "seeds" in gate:
            out.append(f"\nSeed spread {gate['seeds']['spread']:.4f}")
    (common.WORK_DIR / "report.md").write_text("\n".join(out) + "\n")
    (HERE / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
