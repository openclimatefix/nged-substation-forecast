# ruff: noqa: E501
"""Print every number the experiment 277 page quotes, from `work/report_*.json`.

uv run --no-sync python ml-experiments/00277-*/e00277_report.py
"""

import json
from typing import Any

import e00277_common as common


def fmt(x: dict[str, float], scale: float = 1.0, nd: int = 2) -> str:
    """Return `est (lo to hi)`."""
    return f"{x['est'] / scale:.{nd}f} ({x['lo'] / scale:.{nd}f} to {x['hi'] / scale:.{nd}f})"


def main() -> None:  # noqa: PLR0912
    """Print the report."""
    a: dict[str, Any] = json.loads((common.OUT_DIR / "report_a.json").read_text())
    b: dict[str, Any] = json.loads((common.OUT_DIR / "report_b.json").read_text())
    c: dict[str, Any] = json.loads((common.OUT_DIR / "report_c.json").read_text())
    print("== A: Primaries, rows", a["A_rows"])
    print("quantile crossing share per arm", a["A_crossing_share"])
    print(
        "median minus mean per site (arm 1, points)",
        {k: round(v, 3) for k, v in a["A_median_minus_mean"]["arm1"].items()},
    )
    print("MAE of median / of mean", a["A_mae_of_mean_vs_median"])
    print("-- scores, all rows (G0)")
    for arm, m in a["A_metrics"].items():
        print(arm, {k: fmt(v, nd=3) for k, v in m["G0 all rows"].items()})
    print("-- score contrasts (first minus second), all rows")
    for pair, g in a["A_contrasts"].items():
        print(pair, {k: fmt(v, nd=3) for k, v in g["G0 all rows"].items()})
    for fraction in ("0.5", "0.75", "0.9"):
        cost = a["A_cost"][fraction]
        print(
            f"-- cost at f={fraction}: limits",
            [(x["node_id"], round(x["limit_mw"], 2)) for x in cost["limits_mw"]],
        )
        print(
            "census (n half-hours, n above limit)",
            [(x["node_id"], x["n_half_hours"], x["n_above_limit"]) for x in cost["census"]],
            "total above",
            sum(x["n_above_limit"] for x in cost["census"]),
        )
        print(
            "perfect-forecast cost GBP m",
            round(cost["all"]["perfect_cost"] / 1e6, 2),
            "node half-hours",
            cost["all"]["n_node_half_hours"],
        )
        for arm, v in cost["all"]["arms"].items():
            print(" ", arm, "cost", fmt(v["cost"], 1e6), "avoidable", fmt(v["avoidable"], 1e6))
        for pair, v in cost["all"]["contrasts"].items():
            print(" ", pair, fmt(v, 1e6))
    print("-- cost cuts at f=0.5 (GBP m): contrasts bg2-arm1, climatology-arm1")
    for g, v in a["A_cost"]["0.5"]["cuts"].items():
        print(
            " ",
            g,
            "bg2-arm1",
            fmt(v["contrasts"]["bg2-arm1"], 1e6),
            "| clim-arm1",
            fmt(v["contrasts"]["climatology-arm1"], 1e6),
            "| manual-bg2",
            fmt(v["contrasts"]["manual_heuristic-bg2"], 1e6),
        )
    print("-- leave one substation out")
    for fraction, d in a["A_leave_one_out"].items():
        for s, v in d.items():
            print(" ", fraction, s, {k: fmt(x, 1e6) for k, x in v.items()})
    print("-- detectability (observed, smallest resolvable, % of first forecast's own score)")
    for kind, d in a["A_detectability"].items():
        for pair, v in d.items():
            if kind == "cost":
                print(
                    " cost",
                    pair,
                    round(v["observed"] / 1e6, 2),
                    round(v["mde"] / 1e6, 2),
                    round(v["mde_relative_percent"], 1),
                )
            else:
                print(
                    " score",
                    pair,
                    {
                        m: (
                            round(x["observed"], 4),
                            round(x["mde"], 4),
                            round(x["mde_relative_percent"], 1),
                        )
                        for m, x in v.items()
                        if m in ("qs", "tw95", "tw99", "mae")
                    },
                )
    print("== B: bulk supply points", b["n_rows"])
    for arm, m in b["metrics"].items():
        print(
            arm,
            {
                g: {
                    k: round(v["est"], 3)
                    for k, v in x.items()
                    if k in ("qs", "mae", "tw95", "tw99")
                }
                for g, x in m.items()
            },
        )
    for fraction, cost in b["cost"].items():
        print(
            f" f={fraction}",
            "census",
            cost["census"],
            "limits",
            [(x["node_id"], round(x["limit_mw"], 2)) for x in cost["limits_mw"]],
        )
        for arm, v in cost["arms"].items():
            print("  ", arm, "avoidable", fmt(v["avoidable"], 1e6))
        for pair, v in cost["contrasts"].items():
            print("  ", pair, fmt(v, 1e6))
    print("== C: load-shift groups (rows, transfers)")
    for g, n in c["n_rows"].items():
        print(" ", g, n, c["n_cards"][g])
    for g, by_arm in c["metrics"].items():
        print(g)
        for arm in ("K0", "bg2", "manual_heuristic", "climatology"):
            print("  ", arm, {k: fmt(v, nd=2) for k, v in by_arm[arm].items()})
        print("   K0 MAE of mean forecast", fmt(by_arm["K0_mean_forecast_mae"]))
    print("-- contrasts by group (qs, pin95)")
    for g, by_pair in c["contrasts"].items():
        for pair, x in by_pair.items():
            print(
                " ",
                g,
                pair,
                "qs",
                fmt(x["qs"]),
                "pin95",
                fmt(x["pin95"]),
                "se(qs)",
                round(x["qs"]["se"], 3),
            )
    print(
        "-- detectability by row group: smallest resolvable difference, bg2 minus K0 (2.8 x standard error)"
    )
    for g in ("G1 ordinary rows", "G3 during", "G4 after the end", "G5 near (G2 + G3 + G4)"):
        qs = c["contrasts"][g]["bg2-K0"]["qs"]
        pin = c["contrasts"][g]["bg2-K0"]["pin95"]
        cost = c["cost"]["0.5"][g]["contrasts"]["bg2-K0"]
        print(
            " ",
            g,
            "CRPS",
            round(common.POWER_FOR_80_PERCENT * qs["se"], 3),
            "pinball q95",
            round(common.POWER_FOR_80_PERCENT * pin["se"], 3),
            "cost GBP thousand at 0.5",
            round(common.POWER_FOR_80_PERCENT * cost["se"] / 1e3, 1),
        )
    for fraction, d in c["cost"].items():
        for g, v in d.items():
            arms = {k: round(x["avoidable"]["est"] / 1e3, 1) for k, x in v["arms"].items()}
            print(
                f" cost f={fraction}",
                g,
                "blocks",
                v["n_blocks"],
                "half-hours",
                v["n_node_half_hours"],
                "avoidable GBP thousand",
                arms,
                "perfect",
                round(v["perfect_cost"] / 1e3, 1),
            )
            print(
                "    contrasts GBP thousand",
                {
                    k: fmt(x, 1e3, 1)
                    for k, x in v["contrasts"].items()
                    if k in ("bg2-K0", "manual_heuristic-K0", "climatology-K0")
                }
                if False
                else {k: fmt(x, 1e3, 1) for k, x in v["contrasts"].items()},
            )


if __name__ == "__main__":
    main()
