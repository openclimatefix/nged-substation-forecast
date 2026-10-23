# ruff: noqa: E501
"""Print the tables the write-up quotes, from the stored report.

uv run --no-sync python e00308_report.py | tee printed_report.txt
"""

from __future__ import annotations

import json

import e00308_data as dat

DETECTORS = ("d_a", "d_b", "d_c6", "d_c24", "d24_a", "d24_b", "r6_a", "r24_a", "r24_b", "r72_a")
NAMES = {
    "d": "gap, leads 0-6 h",
    "d24": "gap, leads 0-24 h (post hoc)",
    "r6": "residual mean 6 h",
    "r24": "residual mean 24 h",
    "r72": "residual mean 72 h",
}
VAR = {"a": "plain", "b": "neighbour check", "c6": "6 h minimum", "c24": "24 h minimum"}


def label(key: str) -> str:
    """Readable name of a detector key such as d_a."""
    score, variant = key.split("_")
    return f"{NAMES[score]}, {VAR[variant]}"


def main() -> None:
    """Print the tables."""
    r = json.loads((dat.WORK_DIR / "results" / "report.json").read_text())
    print("CARRIED DETECTORS (35 real shifts over 1 day)")
    for src, v in r["carried"].items():
        print(
            f"{src}: matched {v['matched']} ({v['matched_lo']:.0f} to {v['matched_hi']:.0f}); after-end rule {v['matched_late']}; step rule {v['step_rule_matched']}; "
            f"MW share {100 * v['mw_share']:.0f}% ({100 * v['mw_lo']:.0f} to {100 * v['mw_hi']:.0f}); FA odd {v['fa_odd']:.2f}, even {v['fa_even']:.2f} ({v['fa_even_lo']:.2f} to {v['fa_even_hi']:.2f}) per Primary-year; "
            f"alarms/match {v['alarms_per_match']:.1f}; median lag {v['lag_median_hours']} h; D8 drops {v['drops_blocks']['drops']}/{v['drops_blocks']['n_drops']}, blocks {v['drops_blocks']['blocks']}/{v['drops_blocks']['n_blocks']}; "
            f"FA split { ({k: round(x, 2) for k, x in v['fa_split'].items()}) }; classes {v['fa_classes']}; alarms {v['n_alarms']}"
        )
    for src in r["carried"]:
        print(
            f"\nGAP DETECTORS AT {src}'s FALSE ALARM RATE (odd {r['carried'][src]['fa_odd']:.2f} per Primary-year)"
        )
        print(
            "detector | k | matched (95%) | vs carried | after-end rule vs carried | MW share | FA even | alarms/match | lag h | D8 drops | blocks | injected recall | inj FA/yr | even-onset"
        )
        for key in DETECTORS:
            v = r["gap"][f"{key}_{src}"]
            inj = v["injected"]
            inj_s = (
                f"{inj['matched']}/{inj['events']} ({100 * inj['matched'] / inj['events']:.0f}%), MW {100 * inj['mw_share']:.0f}%, FA {inj['fa_per_year']:.2f}"
                if inj
                else "n/a"
            )
            vc, vl = v["vs_carried"], v["vs_carried_late"]
            print(
                f"{label(key)} | {v['k']} | {v['matched']} ({v['matched_lo']:.0f} to {v['matched_hi']:.0f}); late {v['matched_late']} | {vc['diff']:+d} ({vc['lo']:.0f} to {vc['hi']:.0f}) | {vl['diff']:+d} ({vl['lo']:.0f} to {vl['hi']:.0f}) | "
                f"{100 * v['mw_share']:.0f}% ({100 * v['mw_lo']:.0f} to {100 * v['mw_hi']:.0f}) | {v['fa_even']:.2f} ({v['fa_even_lo']:.2f} to {v['fa_even_hi']:.2f}) | {v['alarms_per_match']:.1f} | {v['lag_median_hours']} | "
                f"{v['drops_blocks']['drops']}/{v['drops_blocks']['n_drops']} | {v['drops_blocks']['blocks']}/{v['drops_blocks']['n_blocks']} | {inj_s} | {v['matched_even_onset']}/{v['n_even_onset']}"
            )
            if key == "d_a":
                print(
                    f"    FA split per Primary-year: { ({k: round(x, 2) for k, x in v['fa_split'].items()}) }; classes {v['fa_classes']}"
                )


def injected_breakdown() -> None:
    """Recall of the plain 0-6 h gap detector on injected shifts, by moved megawatts and duration."""
    import polars as pl

    for src in ("LR_b20", "RO_b80"):
        t = pl.read_parquet(dat.WORK_DIR / "results" / f"injected_{src}.parquet")
        t = t.with_columns(
            size=pl.when(pl.col("mw") < 0.25)
            .then(pl.lit("under 0.25 MW"))
            .when(pl.col("mw") < 0.5)
            .then(pl.lit("0.25 to 0.5 MW"))
            .when(pl.col("mw") < 1.0)
            .then(pl.lit("0.5 to 1 MW"))
            .otherwise(pl.lit("1 MW or more")),
            duration=pl.when(pl.col("hours") <= 48)
            .then(pl.lit("up to 2 days"))
            .when(pl.col("hours") <= 120)
            .then(pl.lit("2 to 5 days"))
            .otherwise(pl.lit("over 5 days")),
            winter=pl.col("month").is_in([11, 12, 1, 2, 3]),
        )
        print(f"\nINJECTED RECALL, plain gap (0-6 h) at {src}'s rate")
        for col in ("size", "duration", "winter"):
            print(
                t.group_by(col)
                .agg(
                    events=pl.len(),
                    matched=pl.col("matched").sum(),
                    recall=pl.col("matched").mean(),
                    lag=pl.col("lag_hours").median(),
                )
                .sort(col)
            )


if __name__ == "__main__":
    main()
    injected_breakdown()
