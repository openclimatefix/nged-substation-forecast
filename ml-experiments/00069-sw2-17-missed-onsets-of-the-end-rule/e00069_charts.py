"""Draw experiment 69's figures as SVG.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.

The series are Primary substations, not NGED generators.

    uv run python ml-experiments/00069-*/e00069_charts.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00069_classify import CLASSES
from e00069_common import WORK_DIR
from e00069_report import contrast, pooled, populations

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00069-sw2-17-missed-onsets-of-the-end-rule"
)
WIDTH: Final[int] = 520
WRAP: Final[int] = 80
LABELS: Final[dict[str, str]] = {
    "D0M_48": "D0M: tracked end",
    "PE14_48": "End rule PE14 (strongest step in 14 days)",
    "PF14_48": "End rule PF14 (first return step): benchmark",
    "SF14_50": "Strong alarms only at 6.55 (PF14 end)",
    "PPF14_50": "PP, weak alarms at half the threshold (PF14 end)",
    "PPF14_75": "PP, weak alarms at three quarters (PF14 end)",
}
COLOURS: Final[list[str]] = [
    "#888888",
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    "#7A3E9D",
    "#B8860B",
]
POPULATIONS: Final[dict[str, str]] = {
    "base": "1,160 fresh headline-band transfers (625 start in even months)",
    "long": "400 fresh long transfers (15.6 to 60 days)",
}
BENCHMARK: Final[str] = "PF14_48"


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / name)
    print("wrote", name)


def figure_classes() -> alt.TopLevelMixin:
    """The class of each of the 24 unalarmed real transfers, and how many stand above noise."""
    classes = pl.read_parquet(WORK_DIR / "classes.parquet")
    rows = []
    for label in CLASSES:
        part = classes.filter(pl.col("class") == label)
        above = int(part["near_above_control"].sum())
        rows += [
            {"class": label, "kind": "Largest step above noise", "n": above},
            {"class": label, "kind": "Largest step within noise", "n": part.height - above},
        ]
    return (
        alt.Chart(pl.DataFrame(rows))
        .mark_bar()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("class:N", sort=list(CLASSES), title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("n:Q", title="Labelled real transfers over 1 day"),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=["Largest step above noise", "Largest step within noise"],
                    range=[ocf.BRAND_ORANGE, "#BBBBBB"],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=200,
            title=_title(
                text="Only 8 of the 24 unalarmed transfers have a sharp step under the threshold",
                lines=[
                    (
                        "The 24 labelled real transfers over 1 day with no D0M alarm within 2 "
                        "hours of the start, by the first class that applies. Above noise: the "
                        "largest "
                        "step within 2 hours of the start exceeds the 95th percentile of the "
                        "largest steps in control windows with no transfer."
                    )
                ],
            ),
        )
    )


def figure_recall(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """MW-weighted recall of each arm on the two fresh populations, with intervals."""
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
            "arm:N", legend=None, scale=alt.Scale(domain=list(LABELS.values()), range=COLOURS)
        ),
    )
    chart = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Share of moved MW found", scale=alt.Scale(domain=[0, 1])),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=70).encode(x="point:Q")  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH, height=150)
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
                text="Weak alarms recover what the raised strong threshold loses, and no more",
                lines=[
                    (
                        "MW-weighted recall at 80 odd-month unexplained alarms, shapes S2, S3, and "
                        "S4 averaged, transfers starting in even months. Dot: estimate. Line: "
                        "98.75% interval from resampling whole calendar months."
                    )
                ],
            )
        )
        .resolve_scale(y="independent")
    )


def figure_contrasts(*, data: pl.DataFrame) -> alt.TopLevelMixin:
    """The four planned contrasts and the descriptive contrasts, with intervals."""
    base = pooled(data=data, population="base", parity="even")
    long = pooled(data=data, population="long", parity="even")
    items = [
        ("A1: PF14 minus PE14, headline band", base, "PF14_48", "PE14_48", "Planned"),
        ("A2: PF14 minus PE14, long transfers", long, "PF14_48", "PE14_48", "Planned"),
        ("C1: PP minus PF14, headline band", base, "PPF14_50", BENCHMARK, "Planned"),
        ("C2: PP minus PF14, long transfers", long, "PPF14_50", BENCHMARK, "Planned"),
        ("Strong alarms only minus PF14, headline band", base, "SF14_50", BENCHMARK, "Descriptive"),
        ("PP minus strong alarms only, headline band", base, "PPF14_50", "SF14_50", "Descriptive"),
        ("D0M minus PF14, long transfers", long, "D0M_48", BENCHMARK, "Descriptive"),
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
            scale=alt.Scale(domain=["Planned", "Descriptive"], range=[ocf.DATA_BLUE, "#7A3E9D"]),
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
            text="The first-return rule is confirmed; the paired onset gains nothing",
            lines=[
                (
                    "Change in MW-weighted recall, shapes S2, S3, and S4 averaged, even months. "
                    "Line: 98.75% interval. Grey rule: no change."
                )
            ],
        ),
    )


def figure_tuning() -> alt.TopLevelMixin:
    """Odd-month unexplained alarms against the strong threshold, split into strong and weak."""
    curves = pl.read_parquet(WORK_DIR / "curves.parquet").filter(pl.col("weak_share") == 0.5)
    frame = pl.concat(
        [
            curves.select("threshold", n=pl.col("odd_weak")).with_columns(
                kind=pl.lit("Weak alarms")
            ),
            curves.select(
                "threshold", n=pl.col("odd_unexplained") - pl.col("odd_weak")
            ).with_columns(kind=pl.lit("Strong alarms")),
        ]
    )
    target = alt.Chart(pl.DataFrame({"y": [80.0]})).mark_rule(color="#888888").encode(y="y:Q")  # ty: ignore[unresolved-attribute]
    lines = (
        alt.Chart(frame)
        .mark_line()
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("threshold:Q", title="Strong threshold T (noise units)"),
            y=alt.Y(
                "n:Q",
                title="Odd-month unexplained alarms",
                scale=alt.Scale(type="symlog"),
                axis=alt.Axis(values=[0, 10, 30, 80, 200, 700]),
            ),
            color=alt.Color(
                "kind:N",
                title=None,
                scale=alt.Scale(
                    domain=["Strong alarms", "Weak alarms"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    return (target + lines).properties(
        width=WIDTH,
        height=240,
        title=_title(
            text="At 4.0 nearly all paired alarms would be weak, so the threshold rises to 6.55",
            lines=[
                (
                    "Alarms of PP with the weak threshold at half of T, on odd months, against T. "
                    "Grey rule: the 80 alarms of D0M at 4.0 that PP is matched to."
                )
            ],
        ),
    )


def main() -> None:
    """Write the four figures."""
    data = populations(out=WORK_DIR)
    _save(figure_classes(), name="figure_1_classes.svg")
    _save(figure_recall(data=data), name="figure_3_recall_by_arm.svg")
    _save(figure_contrasts(data=data), name="figure_2_contrasts.svg")
    _save(figure_tuning(), name="figure_4_tuning.svg")


if __name__ == "__main__":
    main()
