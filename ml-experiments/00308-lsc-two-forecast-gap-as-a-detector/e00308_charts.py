# ruff: noqa: E501
"""Draw the figures of experiment 308 as SVG beside the write-up.

uv run --no-sync python e00308_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00308_data as dat
import e00308_detect as det
import e00308_harness as h
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00308-lsc-two-forecast-gap-as-a-detector"
)
W: Final[int] = CONTENT_WIDTH_PX - 120
LR_K: Final[float] = 3.2
"""The threshold of the plain 0-6 hour gap detector at LR_b20's false alarm rate (read from the report)."""


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def fig_matches() -> None:
    """Matched shifts against false alarms per Primary-year, for the carried and gap detectors."""
    r = json.loads((dat.WORK_DIR / "results" / "report.json").read_text())
    rows = []
    names = {"LR_b20": "LR_b20 (learned score)", "RO_b80": "RO-NC", "PG_g70": "pair alarm"}
    for src, v in r["carried"].items():
        rows.append(
            {
                "detector": names[src],
                "family": "carried",
                "fa": v["fa_even"],
                "matched": v["matched"],
                "lo": v["matched_lo"],
                "hi": v["matched_hi"],
                "target": names[src],
            }
        )
    gap_names = {
        "d_a": "gap, 0 to 6 h",
        "d24_a": "gap, 0 to 24 h (post hoc)",
        "r24_a": "24 h mean residual",
    }
    for key, label in gap_names.items():
        for src in r["carried"]:
            v = r["gap"][f"{key}_{src}"]
            rows.append(
                {
                    "detector": label,
                    "family": "gap",
                    "fa": v["fa_even"],
                    "matched": v["matched"],
                    "lo": v["matched_lo"],
                    "hi": v["matched_hi"],
                    "target": names[src],
                }
            )
    t = pl.DataFrame(rows)
    order = [
        "LR_b20 (learned score)",
        "RO-NC",
        "pair alarm",
        "gap, 0 to 6 h",
        "gap, 0 to 24 h (post hoc)",
        "24 h mean residual",
    ]
    colours = [
        ocf.DATA_PURPLE,
        ocf.BRAND_ORANGE,
        ocf.DATA_SKY,
        ocf.DATA_BLUE,
        ocf.DATA_GREEN,
        "#1B1B1B",
    ]
    base = alt.Chart(t).encode(
        x=alt.X("fa:Q", title="False alarms per Primary-year, even months (fewer is better)"),
        y=alt.Y(
            "matched:Q",
            title="Real load shifts matched, of 35 (more is better)",
            scale=alt.Scale(domain=[0, 30]),
        ),
        color=alt.Color(
            "detector:N",
            scale=alt.Scale(domain=order, range=colours),
            legend=alt.Legend(title=None, orient="top", columns=2),
        ),
    )
    chart = (
        base.mark_errorbar(thickness=1.5).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("lo:Q", title="Real load shifts matched, of 35 (more is better)"), y2="hi:Q"
        )
        + base.mark_point(filled=True, size=90, aria=False)
    ).properties(
        width=W,
        height=300,
        title=alt.TitleParams(
            [
                "Each gap detector is set to the false alarm rate of one carried detector,",
                "so the points form three columns",
            ],
            subtitle=[
                "Dot: shifts whose alarm opened while the shift was on. Line: 95% interval from resampling whole shifts.",
                "Threshold fixed on odd months without labels; false alarm rate read on even months.",
            ],
            anchor="start",
        ),
    )
    save(chart, "figure_1_matched_shifts_against_false_alarms")


def window(*, series: int, centre: int, before: int, after: int) -> pl.DataFrame:
    """The gap scores of one series around a stamp, in MW, with days from the centre."""
    caps = h.capacities()
    z = np.load(det.GAP / f"real_{series}.npz")
    sel = (z["origin"] >= centre - before * 48) & (z["origin"] <= centre + after * 48)
    cap = caps[series]
    return pl.DataFrame(
        {
            "day": (z["origin"][sel] - centre) / 48.0,
            "Gap, leads 0 to 6 h": z["d"][sel] * cap / 100.0,
            "Mean residual, last 24 h": z["r24"][sel] * cap / 100.0,
        }
    ).unpivot(index="day", variable_name="score", value_name="mw")


def fig_examples() -> None:
    """Three stacked panels: a clear shift, a false alarm on D9, and a D8 multi-day drop."""
    ctx = det.build_context()
    shifts = ctx.shifts
    # Panel A: the shift with the largest step (series 13), onset at day 0.
    row = max(shifts.iter_rows(named=True), key=lambda r: det.shift_weight(ctx=ctx, row=r))
    series_a = next(s for s in row["series"] if s in ctx.real)
    opens_a = det.opens_of(s=ctx.real[series_a], score="d", variant="a", k=LR_K, confirm=None)
    # Panel B: first D9 false alarm of the plain gap detector.
    v9 = ctx.real[det.D9]
    o9 = det.opens_of(s=v9, score="d", variant="a", k=LR_K, confirm=None)
    fa9 = int(o9[v9.ordinary_stamp[o9]][0])
    # Panel C: D8's longest multi-day drop.
    drops = pl.read_parquet(dat.D8_DROPS).sort("last_stamp", descending=False)
    drop = max(drops.iter_rows(named=True), key=lambda r: r["last_stamp"] - r["first_stamp"])
    v8 = ctx.real[det.D8]
    o8 = det.opens_of(s=v8, score="d", variant="a", k=LR_K, confirm=None)
    panels = [
        (
            "A shift over 1 day at one Primary: the gap and the residual both step",
            series_a,
            int(row["onset"]),
            int(row["end"]),
            opens_a,
        ),
        (
            "A false alarm at D9: the gap crosses the threshold on an ordinary day",
            det.D9,
            fa9,
            fa9,
            o9,
        ),
        (
            "D8's longest multi-day drop",
            det.D8,
            int(drop["first_stamp"]),
            int(drop["last_stamp"]),
            o8,
        ),
    ]
    layers = []
    for title, series, start, end, opens in panels:
        data = window(
            series=series, centre=start, before=3, after=max(5, int((end - start) / 48) + 3)
        )
        k_mw = LR_K * ctx.real[series].sigma["d"] * ctx.real[series].cap / 100.0
        lines = (
            alt.Chart(data)  # ty: ignore[unresolved-attribute]
            .mark_line(clip=True, aria=False)
            .encode(
                x=alt.X("day:Q", title="Days from the start of the shift (or of the false alarm)"),
                y=alt.Y("mw:Q", title="MW (positive: more power than the no-lag forecast)"),
                color=alt.Color(
                    "score:N",
                    scale=alt.Scale(
                        domain=["Gap, leads 0 to 6 h", "Mean residual, last 24 h"],
                        range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
        )
        span = (
            alt.Chart(pl.DataFrame({"a": [0.0], "b": [(end - start) / 48.0]}))  # ty: ignore[unresolved-attribute]
            .mark_rect(opacity=0.12, color=ocf.DATA_PURPLE)
            .encode(x="a:Q", x2="b:Q")
        )
        rules = (
            alt.Chart(pl.DataFrame({"y": [k_mw, -k_mw]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 3], color="#555555")
            .encode(y="y:Q")
        )
        marks = (
            alt.Chart(  # ty: ignore[unresolved-attribute]
                pl.DataFrame(
                    {"x": [(float(o) - start) / 48.0 for o in opens if abs(o - start) < 8 * 48]}
                )
            )
            .mark_rule(color=ocf.DATA_GREEN, strokeWidth=2)
            .encode(x="x:Q")
        )
        layers.append(
            (span + rules + marks + lines).properties(
                width=W, height=150, title=alt.TitleParams(title, anchor="start")
            )
        )
    chart = (
        alt.vconcat(*layers)
        .resolve_scale(color="shared")
        .properties(
            title=alt.TitleParams(
                [
                    "The gap between the two forecasts steps at a real shift and trails the residual,",
                    "and it also crosses the threshold on some ordinary days",
                ],
                subtitle=[
                    "Purple band: the shift or drop. Dashed lines: the alarm threshold of the plain gap detector at LR_b20's false alarm rate.",
                    "Green lines: origins where this detector opens an alarm episode. The panel for the false alarm has no purple band.",
                ],
                anchor="start",
            )
        )
    )
    save(chart, "figure_2_example_gap_and_residual")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_matches()
    fig_examples()
