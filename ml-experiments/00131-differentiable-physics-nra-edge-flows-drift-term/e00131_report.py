"""Print every number that experiment 131's write-up quotes, from the stored result files.

uv run python ml-experiments/00131-*/e00131_report.py > data/.../work/report.txt
"""

import json
from typing import Final

import e00131_common as common
import numpy as np
import polars as pl

WORK: Final = common.WORK_DIR


def load(name: str) -> dict:
    """Read a JSON result file."""
    return json.loads((WORK / name).read_text())


def main() -> None:
    """Print the report."""
    decisions = load("decisions.json")
    print("== chosen settings (tuning year) ==")
    for group, result in decisions.items():
        print(
            group,
            result["config"],
            "recall",
            {k: round(v, 3) for k, v in result["recall"].items()},
            "worst full-year placebo",
            round(max(result["placebo_full"].values()), 4),
            "met limit",
            result["met_limit"],
        )
    tune = [json.loads(p.read_text()) for p in sorted((WORK / "tune").glob("*.json"))]
    print("== stage 1 ==")
    print(
        "settings screened:",
        len(tune),
        "registered:",
        sum(r["config"]["level_c"] in (0.3, 1.0) for r in tune),
        "extra:",
        sum(r["config"]["level_c"] in (0.5, 0.7) for r in tune),
    )
    under = [r for r in tune if max(r["placebo"].values()) < 0.05]
    with_drift = [r for r in under if r["config"]["drift_q"] is not None]
    print("under the 0.05 limit:", len(under), "of which with a drift term:", len(with_drift))
    print(
        "best margin among settings under the limit:",
        round(
            max(
                min(r["recall"]["3-24h"] / 0.5, r["recall"]["1-3 days"] / 0.77)
                for r in under
                if "recall" in r
            ),
            3,
        ),
    )
    print("recall >= targets among all settings read:")
    for r in tune:
        if "recall" in r and r["recall"]["3-24h"] >= 0.5 and r["recall"]["1-3 days"] >= 0.77:
            print(
                "  ",
                r["config"]["step_c"],
                r["config"]["level_c"],
                r["config"]["drift_q"],
                "placebo",
                round(max(r["placebo"].values()), 3),
            )
    print("matched-recall placebo, step 3 level 0.3 (tuning year, 4 tiles):")
    for r in tune:
        c = r["config"]
        if c["step_c"] == 3.0 and c["level_c"] == 0.3:
            print("  q", c["drift_q"], "placebo", round(max(r["placebo"].values()), 2), r["recall"])
    gate = load("gate.json")
    print("== gate ==")
    print(json.dumps(gate["gate_passes"]), "7-30 day recall", gate["recall_7_to_30_days"])
    for group, years in gate["readings"].items():
        for year, readings in years.items():
            row = {
                k: (v["value"] if k[0] != "3" else v["worst"])
                for k, v in readings.items()
                if isinstance(v, dict)
            }
            print(group, year, "n_big", readings["n_big_transfers"], row)
            print("   passes", {k: v["pass"] for k, v in readings.items() if isinstance(v, dict)})
            print("   release n_found", readings["4_release_share"]["n_found"])
    rows = pl.read_parquet(WORK / "gate_rows.parquet").filter(pl.col("set") == "G")
    print("== found transfers: fitted over true mean flow (shrinkage) ==")
    for group in ("drift", "none"):
        for year in ("tuning", "replicate"):
            sub = rows.filter(
                pl.col("group") == group, pl.col("year") == year, pl.col("recall") >= 0.5
            )
            if sub.height:
                print(
                    group,
                    year,
                    "found",
                    sub.height,
                    "median ratio",
                    round(float(np.median(sub["ratio"].to_numpy())), 3),
                )
    placebo = pl.read_parquet(WORK / "gate_placebo.parquet")
    print("== gate placebo (MW-days a year) ==")
    print(placebo.sort("group", "year", "pair", "order"))
    print("== where the replicate placebo flow falls (drift setting, placebo first) ==")
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if kk != "tiles"}
                for k, v in load("placebo_where.json").items()
            },
            indent=1,
        )
    )
    print("== big-event threshold (MW, at the tuning cut) ==")
    print(load("threshold.json") if (WORK / "threshold.json").exists() else "")
    if (WORK / "real_events.json").exists():
        real = load("real_events.json")
        print("== real transfers ==")
        print(real["summary"])
        events = pl.DataFrame(real["events"])
        print(
            "events by year:",
            events.group_by("year").agg(pl.len().alias("n"), pl.col("big").sum().alias("big")),
        )
        print(
            "direction unknown:",
            int((~events["direction_known"]).sum()),
            "labelled events 3 hours or longer on decision edges:",
            int((events["hours"] >= 3.0).sum()),
        )
        found = events.filter(pl.col("found"))
        print("found:", found.select("edge", "hours", "labelled_moved_mw", "fitted_mean_flow_mw"))
        runs = pl.DataFrame(real["runs"])
        print(runs.group_by("year").agg(pl.len().alias("runs"), pl.col("matched").sum()))


if __name__ == "__main__":
    main()
