# ruff: noqa: E501
"""Draw the figures of experiment 202's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00202-*/e00202_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00202_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
REPORT: Final[Path] = common.WORK_DIR / "report"
WIDTH: Final[int] = 680
ARM_LABELS: Final[dict[str, str]] = {
    "m": "M: 51 members (fixed factors)",
    "q2": "Q2: two numbers (experiment 193, fixed factors)",
    "r2": "R2: two float32 columns, factors refitted monthly",
    "r2o": "R2o: factors fitted on the scored month (ceiling)",
    "rr": "RR: placebo, no member information",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    "#808080",
]


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    print("wrote", name)


def _interval_chart(
    *,
    rows: list[dict],
    colour_field: str,
    domain: list[str],
    colours: list[str],
    title: alt.TitleParams,
    height: int,
    y_title: str | None = None,
) -> alt.TopLevelMixin:
    frame = pl.DataFrame(rows)
    base = alt.Chart(frame)
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    line = base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("label:N", sort=None, title=y_title, axis=alt.Axis(labelLimit=400)),
        x=alt.X(
            "low:Q",
            title="Change in daylight score from arm A (%; negative is better)",
        ),
        x2="high:Q",
        color=alt.Color(
            f"{colour_field}:N", scale=alt.Scale(domain=domain, range=colours), legend=None
        ),
    )
    dot = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("label:N", sort=None),
        x="estimate:Q",
        color=alt.Color(
            f"{colour_field}:N", scale=alt.Scale(domain=domain, range=colours), legend=None
        ),
    )
    return (rule + line + dot).properties(width=WIDTH - 300, height=height, title=title)


def _rows(*, report: dict, cuts: list[tuple[str, str]], arms: list[str]) -> list[dict]:
    rows = []
    for cut, cut_label in cuts:
        for arm in arms:
            c = report["subsets"][cut][arm]["change_pct"]
            rows.append(
                {
                    "label": f"{cut_label}, {ARM_LABELS[arm].split(':')[0]}"
                    if cut_label
                    else ARM_LABELS[arm],
                    "series": ARM_LABELS[arm],
                    "estimate": c["point"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    return rows


def figure_headline(*, report: dict) -> None:
    """Change in the daylight score for every arm."""
    arms = list(ARM_LABELS)
    chart = _interval_chart(
        rows=_rows(report=report, cuts=[("daylight", "")], arms=arms),
        colour_field="series",
        domain=list(ARM_LABELS.values()),
        colours=ARM_COLOURS,
        title=_title(
            text="Refitting the factors each month on completed history keeps the whole two-number gain",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours (sun 5 degrees or more up), 5 substations, leads 0.5 to 168 hours, scored year. Dashed rule: same as arm A.",
        ),
        height=300,
    )
    _save(chart, name="figure_1_headline")


def figure_substation(*, report: dict) -> None:
    """Change in the daylight score by substation for M, Q2, and R2."""
    cuts = [(f"daylight, {site}", site) for site in common.DECISION_SITES]
    arms = ["m", "q2", "r2"]
    chart = _interval_chart(
        rows=_rows(report=report, cuts=cuts, arms=arms),
        colour_field="series",
        domain=[ARM_LABELS[a] for a in arms],
        colours=ARM_COLOURS[:1] + ARM_COLOURS[1:3],
        title=_title(
            text="The replay gains at every one of the five substations",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours, scored year. Dashed rule: same as arm A.",
        ),
        height=640,
    )
    _save(chart, name="figure_2_substation")


def figure_lead(*, report: dict) -> None:
    """Change in the daylight score by lead band for M, Q2, and R2."""
    cuts = [
        ("daylight, under 24 h", "under 24 h"),
        ("daylight, 24 to 72 h", "24 to 72 h"),
        ("daylight, 72 to 168 h", "72 to 168 h"),
    ]
    arms = ["m", "q2", "r2"]
    chart = _interval_chart(
        rows=_rows(report=report, cuts=cuts, arms=arms),
        colour_field="series",
        domain=[ARM_LABELS[a] for a in arms],
        colours=ARM_COLOURS[:1] + ARM_COLOURS[1:3],
        title=_title(
            text="The replay gains in every lead band, most at 72 to 168 hours",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours, 5 substations, scored year. Dashed rule: same as arm A.",
        ),
        height=420,
    )
    _save(chart, name="figure_3_lead")


def figure_month(*, report: dict) -> None:
    """Change in the daylight score of R2 in each scored month."""
    frame = pl.DataFrame(
        {
            "month": list(report["monthly_daylight_change_pct"]),
            "change": list(report["monthly_daylight_change_pct"].values()),
        }
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(y="y:Q")
    )  # ty: ignore[unresolved-attribute]
    bars = (
        alt.Chart(frame)
        .mark_bar(color=ocf.BRAND_ORANGE)  # ty: ignore[unresolved-attribute]
        .encode(
            x=alt.X("month:N", sort=None, title="Month of the forecast origin"),
            y=alt.Y(
                "change:Q",
                title="Change in daylight score from arm A (%; negative is better)",
            ),
        )
    )
    chart = (bars + zero).properties(
        width=WIDTH - 120,
        height=260,
        title=_title(
            text="Eleven of the 12 months improve; December 2025 got 0.9% worse",
            subtitle="Bars: R2's change in the daylight score of the five substations, by month of the forecast origin. Dashed rule: same as arm A.",
        ),
    )
    _save(chart, name="figure_4_month")


def main() -> None:
    """Draw every figure."""
    report = json.loads((REPORT / "report.json").read_text())
    figure_headline(report=report)
    figure_substation(report=report)
    figure_lead(report=report)
    figure_month(report=report)


if __name__ == "__main__":
    main()
