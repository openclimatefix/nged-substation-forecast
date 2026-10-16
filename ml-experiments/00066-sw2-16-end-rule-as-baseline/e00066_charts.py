"""Draw experiment 66's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00066-*/e00066_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00066_common import WORK_DIR
from e00066_profile import NON_MATCHING
from e00066_report import BENCHMARK, contrast, pooled, populations, recall_by_duration

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00066-sw2-16-end-rule-as-baseline"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "D0M_48": "D0M: tracked end",
    "PE14_48": "End rule, 14-day cap (benchmark)",
    "PE30_48": "End rule, 30-day cap",
    "PE60_48": "End rule, 60-day cap",
    "PF14_48": "First-return rule, 14-day cap (post hoc)",
}
COLOURS: Final[list[str]] = [ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, "#888888", "#7A3E9D"]
POPULATIONS: Final[dict[str, str]] = {
    "base": "620 transfers of experiment 58 (1.5% last over 14 days)",
    "long": "215 long transfers (15.6 to 60 days)",
}


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_recall(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of each 24-hour arm on the two populations, with intervals."""
    rows = []
    for population, label in POPULATIONS.items():
        frames = pooled(data=data, population=population, parity="even")
        for arm, name in LABELS.items():
            point, low, high = contrast(frames=frames, arm=arm)
            rows.append(
                {"population": label, "arm": name, "point": point, "low": low, "high": high}
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=list(LABELS.values()), title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color(
            "arm:N",
            legend=None,
            scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS),
        ),
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=130)
    return (
        chart.facet(
            row=alt.Row(
                "population:N",
                title=None,
                sort=list(POPULATIONS.values()),
                header=alt.Header(
                    labelOrient="top", labelAngle=0, labelLimit=520, labelFontSize=12
                ),
            )
        )
        .properties(
            title=_title(
                text="A 14-day cap fails on long transfers; a 30-day cap loses a little elsewhere",
                lines=[
                    (
                        "MW-weighted recall, 24-hour window, shapes S2, S3, and S4 averaged, "
                        "transfers starting in even months. Dot: estimate. Line: 99% interval "
                        "from resampling whole calendar months. The first-return rule is post hoc."
                    )
                ],
            )
        )
        .resolve_scale(y="independent")
    )


def figure_contrasts(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """The five planned contrasts and the post hoc contrasts, with intervals."""
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    items = [
        ("C1: 30-day cap, long transfers", long, "PE30_48", BENCHMARK, "Planned"),
        ("C2: 60-day cap, long transfers", long, "PE60_48", BENCHMARK, "Planned"),
        ("C3: 30-day cap, 620 transfers", base, "PE30_48", BENCHMARK, "Planned"),
        ("C4: 60-day cap, 620 transfers", base, "PE60_48", BENCHMARK, "Planned"),
        ("C5: 4-hour window, 620 transfers", base, "PE14_8m", BENCHMARK, "Planned"),
        ("First-return rule, long transfers", long, "PF14_48", BENCHMARK, "Post hoc"),
        ("First-return rule, 620 transfers", base, "PF14_48", BENCHMARK, "Post hoc"),
    ]
    rows = []
    for label, frames, arm, ref, kind in items:
        point, low, high = contrast(frames=frames, arm=arm, base=ref)
        rows.append({"contrast": label, "kind": kind, "point": point, "low": low, "high": high})
    chart = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y(
            "contrast:N",
            sort=[str(r["contrast"]) for r in rows],
            title=None,
            axis=alt.Axis(labelLimit=320),
        ),
        color=alt.Color(
            "kind:N",
            title=None,
            scale=alt.Scale(domain=["Planned", "Post hoc"], range=[ocf.DATA_BLUE, "#7A3E9D"]),
            legend=alt.Legend(orient="top"),
        ),
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#888888").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return (
        rule
        + chart.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Change in MW-weighted recall"), x2="high:Q"
        )
        + chart.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text="The 4-hour window loses recall; longer caps gain on long transfers only",
            lines=[
                (
                    "Change from the 24-hour window with a 14-day cap, S2, S3, and S4 averaged, "
                    "even months. Line: 99% interval. Grey rule: no change."
                )
            ],
        ),
    )


def figure_duration(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall by transfer duration for the 24-hour arms."""
    rows = recall_by_duration(data=data, arms=list(LABELS))
    labelled = [
        {**r, "bin": f"{r['low']:g} to {min(r['high'], 60):g}", "arm": LABELS[r["arm"]]}
        for r in rows
    ]
    frame = pl.DataFrame(labelled)
    order = list(dict.fromkeys(r["bin"] for r in labelled))
    return (
        alt.Chart(frame)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "bin:N",
                sort=order,
                title="Duration of the transfer (days)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("recall:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "arm:N",
                title=None,
                scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS),
                legend=alt.Legend(orient="top", columns=1, labelLimit=400),
            ),
        )
        .properties(
            width=WIDTH,
            height=260,
            title=_title(
                text="The 14-day cap finds almost nothing beyond 15.6 days",
                lines=[
                    (
                        f"Both populations together, shapes {', '.join(NON_MATCHING)} averaged, "
                        "even months, 24-hour window. Few transfers last over 14 days in the "
                        "620-transfer set, so the bins from 14 days upward are mostly the long set."
                    )
                ],
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    data = populations(out=WORK_DIR)
    _save(figure_recall(data=data), name="figure_1_recall_by_arm.svg")
    _save(figure_contrasts(data=data), name="figure_2_contrasts.svg")
    _save(figure_duration(data=data), name="figure_3_recall_by_duration.svg")


if __name__ == "__main__":
    main()
