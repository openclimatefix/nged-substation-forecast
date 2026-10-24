# ruff: noqa: E501
"""Print Part 1's real-record tables: tuned thresholds, same-threshold neighbour checks, timing.

uv run --no-sync python e00298_report_real.py > printed_real_summary.txt
"""

from __future__ import annotations

import json

import e00298_common as common
import e00298_detect as dt
import e00298_events as ev
import e00298_real as real
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_width_chars(200)
PAIRS = (("A3", "D-A3"), ("A3", "F-A3"), ("A7", "D-A7"), ("A7", "F-A7"), ("C", "D-C"), ("C", "F-C"))


def main() -> None:
    """Print the same-threshold comparison and timing."""
    curve = pl.read_parquet(common.EXPERIMENT_DIR / "real_curves.parquet")
    chosen = json.loads((common.EXPERIMENT_DIR / "chosen_k.json").read_text())
    streams = real.build_streams()
    events, closures = ev.load_events(), ev.load_closures()
    partnered = [
        e
        for e in events
        if e["cls"] == "transfer" and len(e["sites"]) == 2 and all(s in streams for s in e["sites"])
    ]
    print(
        f"transfers with two metered sides: {len(partnered)} of {sum(e['cls'] == 'transfer' for e in events)}"
    )
    print("== Tuned thresholds (2 false alarms per Primary-year in odd months) ==")
    print(
        curve.join(
            pl.DataFrame({"family": list(chosen), "k": list(chosen.values())}), on=["family", "k"]
        )
        .select(
            "family",
            "k",
            "fa_odd",
            "fa_even",
            "fa_unlogged_even",
            "fa_d8_even",
            "fa_d9_even",
            "block",
            "drop",
            "transfer",
        )
        .sort("family")
    )
    print("== Same threshold: base variant against its neighbour-confirmed wrapper ==")
    for base, wrapper in PAIRS:
        k = chosen[base]
        rows = []
        for name, kind_family in ((base, base), (wrapper, wrapper)):
            kwargs = dict(next(kw for f, kw in real.FAMILIES if f == kind_family))
            variant = dt.Variant(k=k, **kwargs)
            spans = real.run_family(streams=streams, variant=variant)
            table = real.classify(spans=spans, events=events, closures=closures)
            fa_even = table.filter(
                pl.col("false_strict"),
                ~pl.col("site").is_in(real.SPECIAL),
                pl.col("month") % 2 == 0,
            ).height
            fa_all = table.filter(
                pl.col("false_strict"), ~pl.col("site").is_in(real.SPECIAL)
            ).height
            found_partnered = sum(
                any(
                    s.start <= e["end"] + 1 and s.end >= e["start"] - 1
                    for site in e["sites"]
                    for s in spans.get(site, [])
                )
                for e in partnered
            )
            tr = real.recall(spans=spans, events=events, cls="transfer")[0]
            rows.append((name, fa_all, fa_even, found_partnered, tr))
        (n0, fa0, fe0, p0, t0), (n1, fa1, fe1, p1, t1) = rows
        print(
            f"k={k:4.2f} {n0:5s} FA all {fa0:3d} even {fe0:3d} partnered {p0}/{len(partnered)} transfers {t0}/35 | {n1:5s} FA all {fa1:3d} even {fe1:3d} partnered {p1}/{len(partnered)} transfers {t1}/35 | FA ratio {fa1 / max(fa0, 1):.2f}, partnered recall loss {1 - p1 / max(p0, 1):.2f}"
        )
    detail = pl.read_parquet(common.EXPERIMENT_DIR / "real_event_detail.parquet")
    print("== Timing among found events (median hours): onset error, end error, alarm lag ==")
    print(
        detail.filter(pl.col("found"))
        .group_by("family", "cls")
        .agg(
            n=pl.len(),
            onset=pl.col("onset_err_h").median(),
            end=pl.col("end_err_h").median(),
            lag=pl.col("alarm_lag_h").median(),
        )
        .sort("cls", "family")
    )
    _ = np


if __name__ == "__main__":
    main()
