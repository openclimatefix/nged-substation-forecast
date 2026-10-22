# ruff: noqa: D103
"""Print every number that experiment 216's `result.md` quotes, from the stored result files.

uv run --no-sync python ml-experiments/00216-*/e00216_report.py > printed_report.txt
"""

import json

import e00216_common as common
import polars as pl


def load(name: str) -> dict:
    return json.loads((common.WORK_DIR / name).read_text())


def main() -> None:
    score, deciles, surface, gate, tail = (
        load(n)
        for n in ("score.json", "deciles.json", "surface.json", "gate.json", "tail_ratio.json")
    )
    print("== tail ratio selection (tuning months; tail = outer deciles 1 and 10) ==")
    print(json.dumps(tail["errors"], indent=1))
    print("chosen:", tail["chosen"])
    for period in ("scored", "replicate"):
        p = score[period]
        print(f"== {period}: absolute (five decision substations) ==")
        for arm, v in p["absolute"].items():
            print(arm, {k: round(x, 4) for k, x in v.items()})
        print(f"== {period}: contrasts (treatment minus reference) ==")
        for k, v in p["contrasts"].items():
            print(k, {a: round(b, 4) for a, b in v.items()})
        print(f"== {period}: by lead band (vs r0) ==")
        for k, v in p["by_lead"].items():
            print(k, {a: round(b, 4) for a, b in v.items()})
        print(f"== {period}: by substation ==")
        for k, v in p["by_site"].items():
            print(k, {s: round(x, 4) for s, x in v.items()})
        print(f"== {period}: median minus mean per substation ==")
        for arm, v in p["checks"].items():
            print(arm, {s: round(x, 4) for s, x in v.items()})
        print(f"== {period}: tail calibration ==")
        for arm, rows in p["tail"].items():
            print(
                arm, [(r["level"], round(r["coverage"], 4), round(r["pinball"], 4)) for r in rows]
            )
    print("== stored network (arm 2 of experiment 201) minus arm D, per seed ==")
    for k, v in score["scored"]["network"].items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    print("== deciles ==")
    print(
        "rows",
        deciles["rows"],
        "rows share outer",
        round(deciles["rows_share_outer"], 4),
        "arm D score share outer",
        round(deciles["r0_score_share_outer"], 4),
    )
    print("outer share of gain", json.dumps(deciles["outer_share"], indent=1))
    for d, v in deciles["by_decile"].items():
        print(d, {a: round(b, 4) for a, b in v.items()})
    for k, v in deciles["by_hour_band"].items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    print("== arm S (convex surface on the network's median correction) ==")
    print(json.dumps(surface, indent=1))
    print("== gate (mean over substations and tuning-month cuts) ==")
    frame = pl.DataFrame(gate["rows"])
    print(
        frame.group_by("variant", "arm")
        .agg(
            pl.col("train_recovery").mean().round(3),
            pl.col("held_recovery").mean().round(3),
            pl.col("train_tail_change").mean().round(3),
            pl.col("held_tail_change").mean().round(3),
        )
        .sort("variant", "arm")
    )
    step = (
        frame.filter((pl.col("variant") == "step") & (pl.col("arm") == "f1_1"))
        .group_by("site")
        .agg(pl.col("train_recovery").mean().round(3))
        .sort("site")
    )
    print(step)


if __name__ == "__main__":
    main()
