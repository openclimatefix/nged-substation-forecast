"""Draw experiment 67's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00067-*/e00067_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00067_common import WORK_DIR
from e00067_profile import NON_MATCHING, SHAPE_NAMES
from e00067_report import ARMS, STEP_EDGES, contrast, population, recall

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00067-sw2-15-joint-edge-flow-estimator"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "EF": "EF: joint edge-flow estimator",
    "D0M": "D0M: single-series step",
    "LOGIT_SAME": "LOGIT + same-sign count (post hoc)",
}
COLOURS: Final[list[str]] = [ocf.BRAND_ORANGE, "#888888", ocf.DATA_BLUE]
STEP_LABELS: Final[list[str]] = ["0-2.5", "2.5-4", "4-6", "6-10", "10+"]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_recall(*, data: pl.DataFrame, headline: str) -> alt.TopLevelMixin:
    """MW-weighted recall of each arm for each shape, as dots."""
    rows = [
        {
            "arm": LABELS[arm],
            "shape": f"{shape}: {SHAPE_NAMES[shape]}",
            "recall": recall(frame=data.filter(pl.col("shape") == shape), column=arm)[0],
        }
        for arm in ARMS
        for shape in NON_MATCHING
    ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "recall:Q",
                title="Share of the moved MW found (MW-weighted recall)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("shape:N", title=None),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=[LABELS[a] for a in ARMS], range=COLOURS),
                legend=alt.Legend(orient="top", columns=1),
            ),
        )
        .properties(
            width=WIDTH,
            height=150,
            title=_title(
                text=headline,
                lines=[
                    "620 injected transfers of 25% to 40% of the receiving load, over 1 day, "
                    "starting in even months, all arms held to 80 unexplained alarms on odd months, "
                    "interval end from the strongest opposite step (EF: the end of its run)."
                ],
            ),
        )
    )


def figure_gain(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Gain in MW-weighted recall of EF over D0M and over LOGIT_SAME, with intervals."""
    frames = [data.filter(pl.col("shape") == s) for s in NON_MATCHING]
    rows = []
    for base in ("D0M", "LOGIT_SAME"):
        point, low, high = contrast(frames=frames, arm="EF", base=base)
        rows.append({"base": f"over {LABELS[base]}", "point": point, "low": low, "high": high})
    chart = alt.Chart(pl.DataFrame(rows))
    y = alt.Y("base:N", title=None)
    grey = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")
    bars = chart.mark_rule(strokeWidth=2, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Gain in MW-weighted recall of EF (larger is better)"),
        x2="high:Q",
        y=y,
    )
    dots = chart.mark_point(filled=True, size=70, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y
    )
    return (grey + bars + dots).properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH,
        height=90,
        title=_title(
            text="Gain of the joint estimator over the single-series step detector",
            lines=[
                "Dot: estimate. Line: 97.5% interval from resampling whole calendar months. "
                "Grey rule: no gain."
            ],
        ),
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share found by donor step in noise units, split by metered receiver, shapes S2 to S4."""
    rows = []
    for shape in NON_MATCHING:
        frame = data.filter(pl.col("shape") == shape)
        for flag, group in ((True, "Metered receiver"), (False, "No metered receiver")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            for low, high, label in zip(STEP_EDGES[:-1], STEP_EDGES[1:], STEP_LABELS, strict=True):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                rows.extend(
                    {
                        "group": group,
                        "step": label,
                        "arm": LABELS[arm],
                        "found": int(bin_[arm].sum()),
                        "n": bin_.height,
                    }
                    for arm in ARMS
                )
    table = (
        pl.DataFrame(rows)
        .group_by("group", "step", "arm")
        .agg(pl.col("found").sum(), pl.col("n").sum())
        .with_columns(share=pl.col("found") / pl.col("n"))
        .filter(pl.col("n") > 0)
    )
    chart = (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("step:N", sort=STEP_LABELS, title="Donor step (noise units)"),
            y=alt.Y("share:Q", title="Share of transfers found", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=[LABELS[a] for a in ARMS], range=COLOURS),
                legend=alt.Legend(orient="top", columns=1),
            ),
        )
        .properties(width=WIDTH, height=150)
        .facet(row=alt.Row("group:N", title=None, sort=["Metered receiver", "No metered receiver"]))
    )
    return chart.properties(
        title=_title(
            text="Share of transfers found by the size of the donor's step",
            lines=[
                "Unweighted share found, pooled over three shapes and counted once per shape, "
                "even months."
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    data = population(out=WORK_DIR)
    _save(figure_recall(data=data, headline="HEADLINE"), name="figure_1_recall_by_arm.svg")
    _save(figure_gain(data=data), name="figure_2_gain_over_d0m.svg")
    _save(figure_by_step(data=data), name="figure_3_share_found_by_step.svg")


if __name__ == "__main__":
    main()
