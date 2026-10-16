"""Draw experiment 61's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00060-*/e00061_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00061_common import WORK_DIR
from e00061_profile import SHAPE_NAMES, SHAPES
from e00061_report import SUPPORT_AT_LEAST, contrast, population, recall_rows, wide

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00061-sw2-6-paired-step-intervals"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
STEP_EDGES: Final[list[float]] = [0.0, 2.0, 4.0, 6.0, 10.0, 1000.0]
STEP_LABELS: Final[list[str]] = ["0-2", "2-4", "4-6", "6-10", "10+"]
D0M: Final[str] = "Constant step (D0M)"
D3: Final[str] = "Profile-weighted step (D3)"


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def _label(shape: str) -> str:
    return f"{shape}: {SHAPE_NAMES[shape]}"


def figure_difference(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Gain in MW-weighted recall of D3 over D0M for each shape, with intervals."""
    rows = []
    for shape in SHAPES:
        point, low, high = contrast(frames=[wide(data=data, shape=shape)])
        rows.append({"shape": _label(shape), "point": point, "low": low, "high": high})
    base = alt.Chart(pl.DataFrame(rows))
    y = alt.Y("shape:N", sort=[_label(s) for s in SHAPES], title=None)
    grey = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888")
    orange = alt.Chart(pl.DataFrame({"x": [SUPPORT_AT_LEAST]})).mark_rule(
        color=ocf.BRAND_ORANGE, strokeDash=[4, 3]
    )
    rules = grey.encode(x="x:Q") + orange.encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    bars = base.mark_rule(color=ocf.DATA_BLUE, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Gain in MW-weighted recall (D3 minus D0M; larger is better)"),
        x2="high:Q",
        y=y,
    )
    dots = base.mark_point(color=ocf.DATA_BLUE, filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y
    )
    return (rules + bars + dots).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="Weighting by the daily profile gains nothing on any shape of transfer",
            lines=[
                (
                    "Dot: estimate. Line: 98.33% interval from resampling whole calendar months. "
                    "Grey rule: no gain. Dashed orange rule: the +0.01 that would count as a gain. "
                    "620 injected transfers of 25% to 40% of the receiving load, over 1 day, "
                    "starting in even months. S1 matches D3's model; S5 works against it."
                )
            ],
        ),
    )


def figure_absolute(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of both detectors for each shape."""
    rows = []
    for shape in SHAPES:
        rec = recall_rows(frame=wide(data=data, shape=shape))
        rows.append({"shape": _label(shape), "detector": D0M, "recall": rec["f0_mw"]})
        rows.append({"shape": _label(shape), "detector": D3, "recall": rec["f3_mw"]})
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("shape:N", sort=[_label(s) for s in SHAPES], title=None),
        x=alt.X("recall:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "detector:N",
            title=None,
            scale=alt.Scale(domain=[D0M, D3], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top"),
        ),
        yOffset="detector:N",
    )
    bars = base.mark_bar()
    text = base.mark_text(align="left", dx=4, fontSize=10).encode(  # ty: ignore[unresolved-attribute]
        text=alt.Text("recall:Q", format=".2f")
    )
    return (bars + text).properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="Both detectors miss about half of the moved MW, whatever the shape",
            lines=[
                (
                    "MW-weighted recall at the same rate of unexplained alarms, 24-hour window, "
                    "620 injected transfers starting in even months. Larger is better; the "
                    "missed share is one minus the bar."
                )
            ],
        ),
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share of transfers found by donor step in noise units, pooled over shapes S2, S3 and S4."""
    pooled = pl.concat([wide(data=data, shape=s) for s in ("S2", "S3", "S4")])
    edges = np.array(STEP_EDGES[1:-1])
    index = np.searchsorted(edges, pooled["donor_step_z"].abs().to_numpy(), side="right")
    pooled = pooled.with_columns(step_bin=pl.Series([STEP_LABELS[i] for i in index]))
    rows = []
    for (bin_,), part in pooled.group_by("step_bin"):
        rows.append(
            {"step_bin": bin_, "detector": D0M, "found": part["f0"].mean(), "n": part.height}
        )
        rows.append(
            {"step_bin": bin_, "detector": D3, "found": part["f3"].mean(), "n": part.height}
        )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        x=alt.X(
            "step_bin:N",
            sort=STEP_LABELS,
            title="Donor's step (noise units of the 24-hour statistic)",
            axis=alt.Axis(labelAngle=0),
        ),
        y=alt.Y("found:Q", title="Share of transfers found", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "detector:N",
            title=None,
            scale=alt.Scale(domain=[D0M, D3], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top"),
        ),
        xOffset="detector:N",
    )
    return base.mark_bar().properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH,
        height=220,
        title=_title(
            text="Transfers are found when the step is large, not when its shape is right",
            lines=[
                (
                    "Share of injected transfers found, by the donor's step in noise units; "
                    "shapes S2, S3 and S4 pooled (1,860 transfer-shape pairs). Larger is better."
                )
            ],
        ),
    )


def main() -> None:
    """Write the three figures."""
    data = population(out=WORK_DIR)
    _save(figure_difference(data=data), name="figure_1_gain_by_shape.svg")
    _save(figure_absolute(data=data), name="figure_2_recall_by_shape.svg")
    _save(figure_by_step(data=data), name="figure_3_found_by_step.svg")


if __name__ == "__main__":
    main()
