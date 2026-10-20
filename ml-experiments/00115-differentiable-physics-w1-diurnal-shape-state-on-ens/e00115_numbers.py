"""Print every number that experiment 115's write-up quotes.

Reads `work/results.json`, `work/gate.json`, `work/hypers.json`, and `work/tuned.json`, which
`e00115_report.py`, `e00115_gate.py`, and `e00115_tune.py` write. The output is committed beside the
write-up as `report.txt`.

    uv run python ml-experiments/00115-*/e00115_numbers.py > docs/ml-experiments/00115-*/report.txt
"""

import json
import math

import e00115_common as common

WORK = common.WORK_DIR


def load(name: str) -> dict:
    """Return a stored JSON file of the work folder."""
    return json.loads((WORK / name).read_text())


def interval(entry: dict) -> str:
    """Return a contrast as `estimate [lower, upper] sites better`."""
    return (
        f"{entry['estimate']:+.4f} [{entry['lower']:+.4f}, {entry['upper']:+.4f}] "
        f"{entry['sites_better']}/{entry['n_sites']} sites better"
    )


def gate(report: dict) -> None:
    """Print the gate readings."""
    print("== Gate (first run kept) ==")
    print(f"tests failed: {report['failed']} of {len(report['tests_run'])}")
    for key in ("test_1_recovery", "post_hoc_test_1_kw_fast"):
        block = report[key]
        label = "pre-registered test 1" if key == "test_1_recovery" else "POST HOC test 1, KW-fast"
        print(f"{label}: passed={block['passed']}")
        for name, entry in block["shapes"].items():
            print(
                f"  {name}: KW median {entry['kw_median']:.3f} (expected "
                f"{entry['expected_median']:.3f}), "
                f"max site {entry['kw_max']:.3f}, K0 median {entry['k0_median']:.3f}, "
                f"R median {entry['r_median']:.3f}, spillover max {entry['spillover_max']:.3f}, "
                f"R spillover max {entry['r_spillover_max']:.3f}, in brief target "
                f"{entry['in_brief_target']}, checks {entry['checks']}"
            )
    t2 = report["test_2_clean"]
    print(f"test 2 (clean data): passed={t2['passed']}")
    for kind, block in t2["kinds"].items():
        print(
            f"  {kind}: score gap KW minus K0 {block['score_gap_kw_minus_k0']:+.4f}, "
            f"{block['checks']}"
        )
        for row in block["per_site"]:
            print(
                f"    {row['site']}: refit q_s {row['refit_q_s']:.2e} (tuned "
                f"{row['tuned_q_s']:.4f}), "
                f"refit s RMS {row['refit_s_rms']:.3f}, tuned s RMS {row['tuned_s_rms']:.3f}, "
                f"sqrt mean P_s {row['tuned_p_s_rms']:.3f}"
            )
    t3 = report["test_3_reproduction"]
    print(
        f"test 3 (reproduction): passed={t3['passed']}, worst vs code "
        f"{t3['worst_vs_code_float64']:.2e}, "
        f"worst level vs stored {t3['worst_level_vs_stored']:.2e}, worst quantile vs stored "
        f"{t3['worst_quantile_vs_stored']:.2e}, rows match {t3['rows_match']}"
    )
    t4 = report["test_4_causality"]
    print(
        f"test 4 (causality): passed={t4['passed']}, worst reading change "
        f"{t4['worst_reading_change']}, "
        f"worst recipe change {t4['worst_recipe_change']}, ensemble inputs identical "
        f"{t4['ensemble_inputs_identical']}"
    )
    t5 = report["test_5_zero_mean"]
    print(
        f"test 5 (zero mean): passed={t5['passed']}, worst block sum "
        f"{t5['worst_abs_block_sum']:.2e}"
    )


def settings(hypers: dict, tuned: dict) -> None:
    """Print the tuned settings."""
    print("== Tuned settings ==")
    print("KW (log q_s, log sigma_eps) per site:")
    for site, (log_q_s, log_eps) in hypers["KW"].items():
        print(
            f"  {site}: q_s {math.exp(log_q_s):.4f} points^2/day, sigma_eps "
            f"{math.exp(log_eps):.3f} points"
        )
    print(f"lead-band multipliers: {tuned['multipliers']}")
    print(
        f"recipe kappa grid scores {tuned['kappa']['grid_scores']}, chosen "
        f"{tuned['kappa']['chosen']}"
    )
    print("spread factors per lead band (0-24, 24-48, 48-144, 144-168 h):")
    for arm, factors in tuned["spread_factors"].items():
        print(f"  {arm}: {factors}")


def planned(results: dict) -> None:
    """Print the planned contrasts."""
    print("== Planned contrasts (98.75% intervals) ==")
    for period, block in results["planned"].items():
        for label, entry in block.items():
            for band in ("D+1", "0.5-168 h", "0-24 h", "24-48 h", "48-144 h", "144-168 h"):
                if band not in entry["bands"]:
                    continue
                data = entry["bands"][band]
                line = (
                    f"{period} | {label} | {band} | qs {interval(data['qs'])} | MAE median "
                    f"{data['mae_median']['estimate']:+.4f}"
                )
                if "mae_mean" in data:
                    line += f" | MAE mean {data['mae_mean']['estimate']:+.4f}"
                print(line)
                if band == "D+1":
                    print(f"    absolute qs {data['absolute_qs']}, rows {data['qs']['rows']}")
            for kind, cells in entry["robust"].items():
                for name, cell in cells.items():
                    if cell is None or (kind == "leave_one_site_out" and name == "D9"):
                        continue
                    print(
                        f"    robust {kind} {name}: {cell['estimate']:+.4f} "
                        f"({cell['sites_better']}/{cell['n_sites']})"
                    )


def exploratory(results: dict) -> None:
    """Print the exploratory contrasts (95% intervals)."""
    print("== Exploratory contrasts (95% intervals), D+1 and all leads ==")
    for period, block in results["exploratory"].items():
        for label, entry in block.items():
            d1, full = entry["bands"]["D+1"], entry["bands"]["0.5-168 h"]
            print(
                f"{period} | {label} | D+1 {interval(d1['qs'])} | all leads "
                f"{interval(full['qs'])} | "
                f"MAE median D+1 {d1['mae_median']['estimate']:+.4f}"
            )


def absolute(results: dict) -> None:
    """Print absolute scores."""
    print("== Absolute quantile scores (points), arms on their own common rows ==")
    for period, block in results["absolute"].items():
        for arm, bands in block.items():
            cells = ", ".join(f"{b} {v['qs']:.3f}" for b, v in bands.items())
            print(f"{period} | {arm} | {cells} | MAE median D+1 {bands['D+1']['mae_median']:.3f}")
    print("D9 beside the decision substations:")
    for period, block in results["absolute_d9"].items():
        for arm, bands in block.items():
            d1, full = bands["D+1"]["qs"], bands["0.5-168 h"]["qs"]
            print(f"{period} | D9 | {arm} | D+1 {d1:.3f}, all leads {full:.3f}")
    print("== References on rows every listed arm holds ==")
    for period, block in results["references"].items():
        print(f"{period}: {block['rows']:,} rows")
        for arm, bands in block["absolute"].items():
            print(f"  {arm}: " + ", ".join(f"{b} {v:.3f}" for b, v in bands.items()))
    print("== Contrasts against references (95%) ==")
    for period, block in results["reference_contrasts"].items():
        for label, entry in block.items():
            d1, full = entry["bands"]["D+1"], entry["bands"]["0.5-168 h"]
            print(
                f"{period} | {label} | D+1 {interval(d1['qs'])} | all leads {interval(full['qs'])}"
            )


def winter(results: dict) -> None:
    """Print the winter breakdown (post hoc)."""
    print("== Winter breakdown, November to March (POST HOC; scores were already read) ==")
    for period, block in results["winter"].items():
        print(f"{period}: cold-snap days per site {block['cold_days']}")
        for pair, entry in block["pairs"].items():
            for label, cell in entry.items():
                if isinstance(cell, str):
                    print(f"  {pair} | {label} | {cell}")
                    continue
                print(
                    f"  {pair} | {label} | {interval(cell['qs'])} | absolute "
                    f"{ {k: round(v, 3) for k, v in cell['absolute'].items()} } | rows "
                    f"{cell['rows']:,}"
                )


def tails(results: dict) -> None:
    """Print tail coverage and pinball loss."""
    print("== Tails: coverage minus nominal (percentage points) and pinball loss ==")
    for period, block in results["tails"].items():
        for arm, bands in block.items():
            for band, entries in bands.items():
                cells = ", ".join(
                    f"q{round(e['level'] * 100):02d} gap {100 * e['gap']:+.2f} pinball "
                    f"{e['pinball']:.4f}"
                    for e in entries
                )
                print(f"{period} | {arm} | {band} | {cells}")
    kw_gaps = {
        band: {round(e["level"] * 100): 100 * e["gap"] for e in entries}
        for band, entries in results["tails"]["scored"]["KW"].items()
    }
    worst_outer = (
        max(abs(g[1]) for g in kw_gaps.values()),
        max(abs(g[99]) for g in kw_gaps.values()),
    )
    worst_inner = (
        max(abs(g[5]) for g in kw_gaps.values()),
        max(abs(g[95]) for g in kw_gaps.values()),
    )
    print(
        f"scored KW worst |gap| over lead bands: q01 {worst_outer[0]:.2f}, q99 "
        f"{worst_outer[1]:.2f}, q05 {worst_inner[0]:.2f}, q95 {worst_inner[1]:.2f}"
    )


def checks(results: dict) -> None:
    """Print the audit checks."""
    print("== Checks: median forecast minus mean forecast at D+1, per site (points) ==")
    for period, block in results["checks"].items():
        for arm, per_site in block.items():
            print(
                f"{period} | {arm} | "
                + ", ".join(f"{s} {v:+.3f}" for s, v in sorted(per_site.items()))
            )
    print("== Sister intervals at 98.75%, D+1 (trigger 6) ==")
    for period, block in results["sisters"].items():
        for label, entry in block.items():
            print(
                f"{period} | {label} | {entry['estimate']:+.4f} [{entry['lower']:+.4f}, "
                f"{entry['upper']:+.4f}] width {entry['width']:.4f}"
            )


def main() -> None:
    """Print every section."""
    results = load("results.json")
    gate(load("gate.json"))
    settings(load("hypers.json"), load("tuned.json"))
    planned(results)
    exploratory(results)
    absolute(results)
    winter(results)
    tails(results)
    checks(results)


if __name__ == "__main__":
    main()
