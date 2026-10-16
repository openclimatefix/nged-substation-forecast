"""Draw experiment 66's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00060-*/e00066_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00066_common import WORK_DIR
from e00066_profile import SHAPE_NAMES, SHAPES
from e00066_report import ARMS, NAMES, contrast, population, recall_rows, wide

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00066-sw2-16-end-rule-as-baseline"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
STEP_EDGES: Final[list[float]] = [0.0, 2.0, 4.0, 6.0, 10.0, 1000.0]
STEP_LABELS: Final[list[str]] = ["0-2", "2-4", "4-6", "6-10", "10+"]
LABELS: Final[dict[str, str]] = {
    "f0": "D0M: tracked end",
    "fe": "PE: D0M alarms, new end",
    "fl": "PL: pairs, labelled prior",
    "ff": "PF: pairs, flat prior",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, "#888888"]


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
    """Gain in MW-weighted recall of PE, PL and PF over D0M for each shape, with intervals."""
    rows = []
    for shape in SHAPES:
        for arm in ("fe", "fl", "ff"):
            point, low, high = contrast(frames=[wide(data=data, shape=shape)], arm=arm, base="f0")
            rows.append(
                {
                    "shape": _label(shape),
                    "arm": LABELS[arm],
                    "point": point,
                    "low": low,
                    "high": high,
                }
            )
    base = alt.Chart(pl.DataFrame(rows))
    y = alt.Y("shape:N", sort=[_label(s) for s in SHAPES], title=None)
    color = alt.Color(
        "arm:N",
        title=None,
        scale=alt.Scale(domain=[LABELS[a] for a in ("fe", "fl", "ff")], range=COLOURS[1:]),
        legend=alt.Legend(orient="top", columns=1),
    )
    grey = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888")
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Gain in MW-weighted recall over D0M (larger is better)"),
        x2="high:Q",
        y=y,
        yOffset="arm:N",
        color=color,
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y, yOffset="arm:N", color=color
    )
    return (grey.encode(x="x:Q") + bars + dots).properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH,
        height=330,
        title=_title(
            text="Finding the end from the return step helps; pairing the alarm itself hurts",
            lines=[
                (
                    "Dot: estimate. Line: 98.33% interval from resampling whole calendar months. "
                    "Grey rule: no gain. 620 injected transfers of 25% to 40% of the receiving "
                    "load, over 1 day, starting in even months, all arms at the same number of "
                    "unexplained alarms on odd months."
                )
            ],
        ),
    )


def figure_absolute(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of every arm for each shape."""
    rows = []
    for shape in SHAPES:
        rec = recall_rows(frame=wide(data=data, shape=shape))
        rows.extend(
            {"shape": _label(shape), "arm": LABELS[arm], "recall": rec[f"{arm}_mw"]}
            for arm in NAMES
        )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("shape:N", sort=[_label(s) for s in SHAPES], title=None),
        x=alt.X("recall:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "arm:N",
            title=None,
            scale=alt.Scale(domain=[LABELS[a] for a in NAMES], range=COLOURS),
            legend=alt.Legend(orient="top", columns=1),
        ),
        yOffset="arm:N",
    )
    text = base.mark_text(align="left", dx=4, fontSize=9).encode(  # ty: ignore[unresolved-attribute]
        text=alt.Text("recall:Q", format=".2f")
    )
    return (base.mark_bar() + text).properties(
        width=WIDTH,
        height=420,
        title=_title(
            text="The best arm still misses about a third of the moved MW",
            lines=[
                (
                    "MW-weighted recall, 24-hour window, 620 injected transfers starting in even "
                    "months. Larger is better; the missed share is one minus the bar."
                )
            ],
        ),
    )


def figure_outcomes(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """How the transfers with a donor step of 6 noise units or more end up, shape S2."""
    rows = []
    for setting, arm in ARMS.items():
        part = data.filter(
            pl.col("shape") == "S2",
            pl.col("setting") == setting,
            pl.col("donor_step_z").abs() >= 6,
        )
        n = part.height
        found = int(part["found"].sum())
        wrong = part.filter(pl.col("onset_found") & ~pl.col("found")).height
        for outcome, count in (
            ("Found", found),
            ("Right start, wrong end", wrong),
            ("No alarm at the start", n - found - wrong),
        ):
            rows.append({"arm": LABELS[arm], "outcome": outcome, "count": count})
    order = ["Found", "Right start, wrong end", "No alarm at the start"]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("arm:N", sort=[LABELS[a] for a in NAMES], title=None),
            x=alt.X("count:Q", title="Transfers (constant-MW shape S2)"),
            color=alt.Color(
                "outcome:N",
                title=None,
                sort=order,
                scale=alt.Scale(domain=order, range=[ocf.DATA_GREEN, ocf.BRAND_ORANGE, "#888888"]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=170,
            title=_title(
                text="Pairing the alarm loses starts; the return step repairs ends",
                lines=[
                    (
                        "The 215 injected constant-MW transfers whose donor step is 6 noise "
                        "units or more, scored on even months."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    data = population(out=WORK_DIR)
    _save(figure_difference(data=data), name="figure_1_gain_by_arm.svg")
    _save(figure_absolute(data=data), name="figure_2_recall_by_arm.svg")
    _save(figure_outcomes(data=data), name="figure_3_outcomes_big_steps.svg")


if __name__ == "__main__":
    main()
