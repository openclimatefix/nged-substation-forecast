"""Export the XL detector's intervals on the real history, for experiment 00086's labels.

XL (the logistic same-sign score on the refit-offset detector's candidates) is pickled in
`models.pkl`, whose classes live in `e00082_combined`, so the intervals are written from this
folder. For each XL alarm the script builds the alarm series' interval with the first-return end
rule, exactly as `e00082_run.build_detectors` does, and adds one interval for each partner: a
neighbour whose own 24-hour step has the opposite sign and exceeds `NEIGHBOUR_Z` noise units
(`e00082_detectors._balance`), whose own level then holds at least half that step for 24 hours
(`e00082_intervals.build_intervals`). No switching log is read.

Writes `xl_intervals.parquet` in experiment 00082's private folder, one row per interval (one per
series and onset, preferring the series' own alarm over a partner row):

- `series`: the series id;
- `role`: `alarm` (the series the alarm fired on) or `partner`;
- `alarm_series`: the series the alarm fired on;
- `onset`, `end`: half-hour stamps from 2022-01-01 00:00 UTC; the interval covers `[onset, end)`;
- `step_mw`: the signed 24-hour residual step of this series at the onset, in MW.

    uv run python ml-experiments/00082-*/e00082_export_labels.py
"""

from __future__ import annotations

import pickle

import numpy as np
import polars as pl
from e00082_combined import candidates
from e00082_common import WORK_DIR
from e00082_detectors import _balance
from e00082_endrule import first_return_intervals
from e00082_intervals import build_intervals
from e00082_run import CAP_DAYS, WINDOW, all_scales, read_json
from e00082_variants import load_both, tracked_relative, variant_inputs


def main() -> None:
    """Write the XL intervals on the real residual panel."""
    panel, offset_panel = load_both()
    scales = all_scales(panel=panel, offset_panel=offset_panel)
    residual, offset = panel["residual"], offset_panel["residual"]
    inputs = variant_inputs(variant="X", residual=residual, offset=offset, scale=scales["X"])
    relative = tracked_relative(
        variant="X", panel=panel, offset_panel=offset_panel, residual=residual, offset=offset
    )
    model = pickle.loads((WORK_DIR / "models.pkl").read_bytes())["XL"]
    threshold = read_json(out=WORK_DIR, name="thresholds.json")["XL"]
    found = candidates(inputs=inputs)
    kept = found.filter(pl.Series(model.score(table=found) > threshold)).select(
        "series", "stamp", "step_mw"
    )
    partner_steps: dict[tuple[int, int], dict[int, float]] = {}
    partner_text = []
    for series, stamp in kept.select("series", "stamp").iter_rows():
        _, partners = _balance(inputs=inputs, series=series, stamp=stamp)
        partner_steps[series, stamp] = partners
        partner_text.append(";".join(f"{p}:{step:.6f}" for p, step in partners.items()))
    alarm_table = kept.with_columns(partners=pl.Series(partner_text, dtype=pl.String))
    tracked = build_intervals(
        alarm_table=alarm_table, relative=relative, window=WINDOW, detector="D0"
    )
    intervals = first_return_intervals(intervals=tracked, inputs=inputs, cap_days=CAP_DAYS)
    own_step = {(r["series"], r["stamp"]): r["step_mw"] for r in kept.iter_rows(named=True)}
    rows = []
    for iv in intervals:
        key = (iv.alarm_series, iv.onset)
        step = own_step[key] if iv.role == "alarm" else partner_steps[key][iv.series]
        rows.append(
            {
                "series": iv.series,
                "role": iv.role,
                "alarm_series": iv.alarm_series,
                "onset": iv.onset,
                "end": iv.end,
                "step_mw": float(step),
            }
        )
    table = pl.DataFrame(
        rows,
        schema={
            "series": pl.Int64,
            "role": pl.String,
            "alarm_series": pl.Int64,
            "onset": pl.Int64,
            "end": pl.Int64,
            "step_mw": pl.Float64,
        },
    )
    # When both sides of a transfer alarm, each series holds an `alarm` and a `partner` interval
    # at the same onset; keep one per (series, onset), preferring the series' own alarm.
    table = (
        table.sort("series", "onset", pl.col("role") != "alarm")
        .unique(subset=["series", "onset"], keep="first", maintain_order=True)
        .sort("series", "onset")
    )
    target = WORK_DIR / "xl_intervals.parquet"
    table.write_parquet(target)
    print(f"{kept.height} XL alarms -> {table.height} intervals -> {target}")
    print(
        table.group_by("role").agg(pl.len(), hours=((pl.col("end") - pl.col("onset")).mean() / 2))
    )
    bad = int((~np.isfinite(table["step_mw"].to_numpy())).sum())
    print(f"intervals with a non-finite step: {bad}")


if __name__ == "__main__":
    main()
