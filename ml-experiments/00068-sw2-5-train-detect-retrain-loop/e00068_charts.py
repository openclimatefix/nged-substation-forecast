"""Draw experiment 68's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00068-*/e00068_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00068_common import WORK_DIR
from e00068_report import error_contrast, event_frame, recall, recall_contrast

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00068-sw2-5-train-detect-retrain-loop"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
WORLDS: Final[dict[str, str]] = {
    "B_in_training": "World B: transfers in the training data",
    "A_into_residual": "World A: transfers added to the residual",
}
ROUND_LABELS: Final[list[str]] = ["Round 1", "Round 2", "Round 3"]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_recall_gain(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """Change in MW-weighted recall from round 0, by round and world, with intervals."""
    rows = []
    for world, label in WORLDS.items():
        for k, round_label in enumerate(ROUND_LABELS, start=1):
            point, low, high = recall_contrast(
                data=data, world=world, arm=f"{world[0]}_r{k}", base=f"{world[0]}_r0"
            )
            rows.append(
                {"world": label, "round": round_label, "point": point, "low": low, "high": high}
            )
    base = alt.Chart(pl.DataFrame(rows))
    y = alt.Y("round:N", sort=ROUND_LABELS, title=None)
    color = alt.Color(
        "world:N",
        title=None,
        scale=alt.Scale(domain=list(WORLDS.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(orient="top", columns=1, labelLimit=500),
    )
    grey = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888")
    bars = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("low:Q", title="Change in MW-weighted recall from round 0 (larger is better)"),
        x2="high:Q",
        y=y,
        yOffset="world:N",
        color=color,
    )
    dots = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y, yOffset="world:N", color=color
    )
    return (grey.encode(x="x:Q") + bars + dots).properties(  # ty: ignore[unresolved-attribute]
        width=WIDTH,
        height=200,
        title=_title(
            text="Refitting without the detected switching changes recall by 0.03 at most",
            lines=[
                (
                    "Dot: estimate. Line: 97.5% interval from resampling whole calendar months. "
                    "Grey rule: no change. Recall is the share of the MW moved by injected "
                    "load transfers that sits in transfers the detector found, pooled over "
                    "three transfer shapes, even months, at 80 or fewer unexplained alarms on "
                    "odd months."
                )
            ],
        ),
    )


def figure_error(*, errors: pl.DataFrame) -> alt.TopLevelMixin:
    """Relative change in the forecast's mean absolute error from round 0, with intervals."""
    rows = []
    for name, label in (
        ("A_r1", "Round 1"),
        ("A_r2", "Round 2"),
        ("A_r3", "Round 3"),
        ("A_rc1", "Control: random mask, round 1"),
    ):
        change, low, high, _, _ = error_contrast(errors=errors, arm=name, base="A_r0")
        rows.append({"round": label, "point": 100 * change, "low": 100 * low, "high": 100 * high})
    order = [str(r["round"]) for r in rows]
    base = alt.Chart(pl.DataFrame(rows))
    y = alt.Y("round:N", sort=order, title=None)
    rules = alt.Chart(pl.DataFrame({"x": [0.0, -1.0], "label": ["no change", "1% fall needed"]}))
    zero = rules.mark_rule(color="#888888").encode(  # ty: ignore[unresolved-attribute]
        x="x:Q",
        strokeDash=alt.StrokeDash(
            "label:N",
            scale=alt.Scale(domain=["no change", "1% fall needed"], range=[[1, 0], [4, 4]]),
            legend=alt.Legend(orient="top", title=None),
        ),
    )
    bars = base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "low:Q",
            title="Change in mean absolute error from round 0 (%; more negative is better)",
        ),
        x2="high:Q",
        y=y,
    )
    dots = base.mark_point(filled=True, size=60, color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
        x="point:Q", y=y
    )
    return (zero + bars + dots).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="The forecast error falls by under 0.5% and the control does as well",
            lines=[
                (
                    "Dot: estimate. Line: 97.5% interval from resampling whole calendar months. "
                    "Even-month half-hours more than 24 hours from every logged closure and "
                    "labelled transfer, 16 Primary substation series, equal weight per series. "
                    "Dashed rule: the 1% fall the decision rule required."
                )
            ],
        ),
    )


def figure_absolute(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall by round and world (absolute skill)."""
    rows = []
    for world, label in WORLDS.items():
        for k in range(4):
            weighted, _ = recall(frame=event_frame(data=data, world=world, name=f"{world[0]}_r{k}"))
            rows.append({"world": label, "round": f"Round {k}", "recall": weighted})
    order = [f"Round {k}" for k in range(4)]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_point(filled=True, size=70)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("round:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "recall:Q",
                scale=alt.Scale(domain=[0.0, 1.0]),
                title="MW-weighted recall (larger is better)",
            ),
            color=alt.Color(
                "world:N",
                title=None,
                scale=alt.Scale(
                    domain=list(WORLDS.values()), range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top", columns=1, labelLimit=500),
            ),
        )
        .properties(
            width=WIDTH,
            height=220,
            title=_title(
                text="The detector finds 0.5 to 0.7 of the moved MW, whatever the round",
                lines=[
                    (
                        "Share of the MW moved by injected load transfers that sits in transfers "
                        "the detector found, pooled over three shapes. World B has 130 "
                        "transfers and world A has 620."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Draw the three figures."""
    data = pl.read_parquet(WORK_DIR / "event_results.parquet")
    errors = pl.read_parquet(WORK_DIR / "errors.parquet")
    _save(figure_recall_gain(data=data), name="figure_1_recall_gain.svg")
    _save(figure_error(errors=errors), name="figure_2_error_change.svg")
    _save(figure_absolute(data=data), name="figure_3_absolute_recall.svg")


if __name__ == "__main__":
    main()
