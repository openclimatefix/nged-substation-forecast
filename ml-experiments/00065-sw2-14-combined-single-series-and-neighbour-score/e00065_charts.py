"""Draw experiment 65's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00065-*/e00065_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00065_common import WORK_DIR
from e00065_profile import NON_MATCHING
from e00065_report import contrast, population, wide
from e00065_run import setting

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00065-sw2-14-combined-single-series-and-neighbour-score"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "D1": "D1: neighbour gate",
    "D2": "D2: neighbour veto",
    "LR": "LR: likelihood-ratio sum",
    "LOGIT": "LOGIT: logistic regression",
    "LOGIT_SAME": "LOGIT + same-sign count",
}
GAIN_ORDER: Final[tuple[str, ...]] = ("D1", "D2", "LR", "LOGIT", "LOGIT_SAME")
COLOURS: Final[list[str]] = [
    "#888888",
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE,
]
STEP_EDGES: Final[list[float]] = [0.0, 2.5, 4.0, 6.0, 10.0, 1000.0]
STEP_LABELS: Final[list[str]] = ["0-2.5", "2.5-4", "4-6", "6-10", "10+"]
RECALL_ARMS: Final[tuple[str, ...]] = ("D0M", "D2", "LOGIT", "LOGIT_SAME")
RECALL_LABELS: Final[dict[str, str]] = {"D0M": "D0M: single series", **LABELS}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_gain(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Gain in MW-weighted recall over D0M, pooled over S2, S3, and S4, under the PE end rule."""
    frames = [wide(data=data, shape=s) for s in NON_MATCHING]
    rows = []
    for arm in GAIN_ORDER:
        point, low, high = contrast(
            frames=frames, arm=setting(arm=arm, rule="PE"), base=setting(arm="D0M", rule="PE")
        )
        rows.append({"arm": LABELS[arm], "point": point, "low": low, "high": high})
    base = alt.Chart(pl.DataFrame(rows))
    y = alt.Y("arm:N", sort=[LABELS[a] for a in GAIN_ORDER], title=None)
    color = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=[LABELS[a] for a in GAIN_ORDER], range=COLOURS),
        legend=None,
    )
    grey = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888")
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Gain in MW-weighted recall over D0M (larger is better)"),
        x2="high:Q",
        y=y,
        color=color,
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y, color=color
    )
    return (grey.encode(x="x:Q") + bars + dots).properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH,
        height=190,
        title=_title(
            text="Neighbour evidence adds little; counting same-sign neighbours adds the most",
            lines=[
                (
                    "Last row is post hoc. Dot: estimate. Line: 99.17% interval from resampling "
                    "whole calendar months. "
                    "Grey rule: no gain over D0M. 620 injected transfers of 25% to 40% of the "
                    "receiving load, over 1 day, starting in even months, pooled over three "
                    "shapes, all arms at 80 unexplained alarms on odd months, interval end from "
                    "the strongest opposite step."
                )
            ],
        ),
    )


def figure_by_step(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Share found by donor step in noise units, split by metered receiver, shapes S2 to S4."""
    rows = []
    for shape in NON_MATCHING:
        frame = wide(data=data, shape=shape)
        for flag, group in ((True, "Metered receiver"), (False, "No metered receiver")):
            part = frame.filter((pl.col("metered_receivers") > 0) == flag)
            for low, high, label in zip(STEP_EDGES[:-1], STEP_EDGES[1:], STEP_LABELS, strict=True):
                bin_ = part.filter(pl.col("donor_step_z").abs().is_between(low, high, "left"))
                rows.extend(
                    {
                        "group": group,
                        "step": label,
                        "arm": RECALL_LABELS[arm],
                        "shape": shape,
                        "found": int(bin_[setting(arm=arm, rule="PE")].sum()),
                        "n": bin_.height,
                    }
                    for arm in RECALL_ARMS
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
                scale=alt.Scale(
                    domain=[RECALL_LABELS[a] for a in RECALL_ARMS],
                    range=["#888888", ocf.DATA_BLUE, ocf.DATA_GREEN, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(orient="top", columns=2),
            ),
        )
        .properties(width=WIDTH, height=160)
        .facet(
            row=alt.Row(
                "group:N",
                title=None,
                sort=["Metered receiver", "No metered receiver"],
            )
        )
    )
    return chart.properties(
        title=_title(
            text="The scores gain where the donor step is 2.5 to 6 noise units",
            lines=[
                (
                    "Unweighted share found, transfers pooled over three shapes and counted once "
                    "per shape, even months, interval end from the strongest opposite step."
                )
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    data = population(out=WORK_DIR)
    _save(figure_gain(data=data), name="figure_1_gain_over_d0m.svg")
    _save(figure_by_step(data=data), name="figure_2_share_found_by_step.svg")


if __name__ == "__main__":
    main()
