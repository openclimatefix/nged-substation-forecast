# ruff: noqa: E501, D103
"""Interval tables of the three detectors over the whole history, with experiment 148's end rule.

RO-NC intervals are experiment 148's stored arm B intervals (6-hour-lookahead end rule, 24-hour
merge). The pair alarm (experiment 156's 4-hour window) and the learned re-ranker (experiment 195,
one alarm set per held-out fold) get intervals from copies of the same code. No switching log is
read.

    OMP_NUM_THREADS=2 systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00219-*/e00219_build.py
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import polars as pl
from e00219_common import ML_EXPERIMENTS_DATA_DIR, WORK_DIR
from e00219_endrule import filter_alarms, live_end_intervals
from e00219_intervals import build_intervals
from e00219_variants import load_both, scales_for, tracked_relative, variant_inputs

E148_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00148-switching-end-rule-lookahead-ro-nc" / "run"
E156_DIR: Final = (
    ML_EXPERIMENTS_DATA_DIR / "00156-switching-faster-onset-alarm-from-opposite-steps" / "detect"
)
E195_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00195-switching-neural-plus-detector-features"
E217_DIR: Final = ML_EXPERIMENTS_DATA_DIR / "00217-switching-label-free-thresholds-in-noise-units"
RULE: Final = "L6"
CAP_DAYS: Final = 14
WINDOW: Final = 48
MERGE: Final = 48
LEARNED_ARM: Final = "R re-ranker"
SCHEMA: Final = pl.Schema({
    "detector": pl.String, "variant": pl.String, "series": pl.Int64, "role": pl.String,
    "alarm_series": pl.Int64, "onset": pl.Int64, "end": pl.Int64, "step_mw": pl.Float64,
})  # fmt: skip


def intervals_frame(*, intervals: list, steps: dict, detector: str, variant: str) -> pl.DataFrame:
    rows = [
        {
            "detector": detector, "variant": variant, "series": iv.series, "role": iv.role,
            "alarm_series": iv.alarm_series, "onset": iv.onset, "end": iv.end,
            "step_mw": steps.get((iv.alarm_series, iv.onset), float("nan")),
        }
        for iv in intervals
    ]  # fmt: skip
    return pl.DataFrame(rows, schema=SCHEMA)


def from_alarms(
    *, table: pl.DataFrame, relative: dict, inputs: Any, merge: bool
) -> tuple[list, dict]:
    steps = {(r["series"], r["stamp"]): r["step_mw"] for r in table.iter_rows(named=True)}
    tracked = build_intervals(alarm_table=table, relative=relative, window=WINDOW, detector="D1")
    live = live_end_intervals(intervals=tracked, inputs=inputs, rule=RULE, cap_days=CAP_DAYS)
    if not merge:
        return live, steps
    kept = filter_alarms(table=table, live=live, tolerance=MERGE)
    rebuilt = build_intervals(alarm_table=kept, relative=relative, window=WINDOW, detector="D1")
    live = live_end_intervals(intervals=rebuilt, inputs=inputs, rule=RULE, cap_days=CAP_DAYS)
    return live, steps


def alarm_table(
    *, series: list[int], stamps: list[int], inputs: Any, partners: dict
) -> pl.DataFrame:
    rows = []
    for s, t in zip(series, stamps, strict=True):
        step = float(inputs.shift[s][t])
        z = step / inputs.scale[s]
        if not np.isfinite(step):
            continue
        rows.append(
            {"series": s, "stamp": t, "step_mw": step, "z": z, "partners": partners.get((s, t), "")}
        )
    return pl.DataFrame(
        rows,
        schema={"series": pl.Int64, "stamp": pl.Int64, "step_mw": pl.Float64, "z": pl.Float64, "partners": pl.String},
    )  # fmt: skip


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    panel, offset_panel = load_both()
    scale = scales_for(variant="X", residual=panel["residual"], offset=offset_panel["residual"])
    inputs = variant_inputs(
        variant="X", residual=panel["residual"], offset=offset_panel["residual"], scale=scale
    )
    relative = tracked_relative(
        variant="X", panel=panel, offset_panel=offset_panel,
        residual=panel["residual"], offset=offset_panel["residual"],
    )  # fmt: skip
    parts = []

    stored = pl.read_parquet(E148_DIR / "intervals.parquet")
    for setting, variant in (("L6_M48", "L6_M48"), ("L6", "L6")):
        sub = stored.filter(pl.col("setting") == setting)
        parts.append(
            sub.select(
                detector=pl.lit("RO-NC"), variant=pl.lit(variant), series="series", role="role",
                alarm_series="alarm_series", onset="onset", end="end", step_mw="step_mw",
            ).cast(SCHEMA)
        )  # fmt: skip
        print("RO-NC", setting, sub.filter(pl.col("role") == "alarm").height, "alarm intervals")

    pair = pl.read_parquet(E156_DIR / "real_W8_alarms.parquet")
    partners = {(r["series"], r["stamp"]): r["partners"] for r in pair.iter_rows(named=True)}
    table = alarm_table(
        series=pair["series"].to_list(),
        stamps=pair["stamp"].to_list(),
        inputs=inputs,
        partners=partners,
    )
    for merge, variant in ((True, "L6_M48"), (False, "L6")):
        live, steps = from_alarms(table=table, relative=relative, inputs=inputs, merge=merge)
        frame = intervals_frame(intervals=live, steps=steps, detector="pair", variant=variant)
        parts.append(frame)
        print("pair", variant, frame.filter(pl.col("role") == "alarm").height, "alarm intervals")

    for fold in range(3):
        a = pl.read_parquet(E195_DIR / f"alarms_f{fold}.parquet").filter(
            pl.col("arm") == LEARNED_ARM
        )
        a = a.unique(["series", "stamp"]).sort("stamp")
        table = alarm_table(
            series=a["series"].to_list(), stamps=a["stamp"].to_list(), inputs=inputs, partners={}
        )
        for merge, variant in ((True, "L6_M48"), (False, "L6")):
            live, steps = from_alarms(table=table, relative=relative, inputs=inputs, merge=merge)
            frame = intervals_frame(
                intervals=live, steps=steps, detector=f"learned_f{fold}", variant=variant
            )
            parts.append(frame)
            print(
                f"learned_f{fold}", variant, frame.height, "intervals from", table.height, "alarms"
            )

    out = pl.concat(parts)
    out.write_parquet(WORK_DIR / "intervals.parquet")
    p99 = {
        s: float(np.nanpercentile(v, 99)) for s, v in panel["power"].items() if np.isfinite(v).any()
    }
    pl.DataFrame({"series": list(p99), "p99_mw": list(p99.values())}).write_parquet(
        WORK_DIR / "p99.parquet"
    )
    print("p99 MW by series:", {s: round(v, 2) for s, v in p99.items()})


if __name__ == "__main__":
    main()
