# ruff: noqa: E501
"""Draw the figures of experiment 193's write-up from `work/report/*.json`.

uv run --no-sync python ml-experiments/00193-*/e00193_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00193_common as common
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
    "m": "M: 51 members",
    "q2": "Q2: mean and spread (2 numbers)",
    "q3": "Q3: plus 10th and 90th percentiles (4 numbers)",
    "mp_strong": "Placebo: other day's numbers (lead, cloud band)",
    "mp": "Placebo: other day's numbers (month, hour)",
    "mr": "Placebo: rescale only, no members",
}
ARM_COLOURS: Final[list[str]] = [
    ocf.DATA_BLUE,
    ocf.BRAND_ORANGE,
    ocf.DATA_GREEN,
    ocf.DATA_PURPLE,
    ocf.DATA_SKY,
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
            title="Change in the daylight quantile score from arm A (%; more negative is better)",
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
    for period in ("scored", "replicate"):
        for cut, cut_label in cuts:
            for arm in arms:
                c = report[period]["subsets"][cut][arm]["change_pct"]
                rows.append(
                    {
                        "label": f"{cut_label}, {ARM_LABELS[arm]}, {period}",
                        "series": ARM_LABELS[arm],
                        "estimate": c["point"],
                        "low": c["lower"],
                        "high": c["upper"],
                    }
                )
    return rows


def figure_headline(*, report: dict) -> None:
    """Change in the daylight score for arms M, Q2, Q3, and the placebos, both periods."""
    arms = list(ARM_LABELS)
    chart = _interval_chart(
        rows=_rows(report=report, cuts=[("daylight", "daylight")], arms=arms),
        colour_field="series",
        domain=list(ARM_LABELS.values()),
        colours=ARM_COLOURS,
        title=_title(
            text="Two numbers per half-hour keep 98% of the 51-member gain in the daylight score",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours (sun 5 degrees or more up), 5 substations, leads 0.5 to 168 hours. Dashed rule: same as arm A.",
        ),
        height=420,
    )
    _save(chart, name="figure_1_headline")


def figure_lead(*, report: dict) -> None:
    """Change in the daylight score by lead band for M and Q2, scored year and replicate months."""
    cuts = [
        ("daylight, under 24 h", "under 24 h"),
        ("daylight, 24 to 72 h", "24 to 72 h"),
        ("daylight, 72 to 168 h", "72 to 168 h"),
    ]
    chart = _interval_chart(
        rows=_rows(report=report, cuts=cuts, arms=["m", "q2"]),
        colour_field="series",
        domain=[ARM_LABELS["m"], ARM_LABELS["q2"]],
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        title=_title(
            text="The two-number arm follows the 51-member arm in every lead band",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours, 5 substations. Dashed rule: same as arm A.",
        ),
        height=400,
    )
    _save(chart, name="figure_2_lead")


def figure_substation(*, report: dict) -> None:
    """Change in the daylight score by substation for M and Q2, scored year and replicate months."""
    cuts = [(f"daylight, {site}", site) for site in common.DECISION_SITES]
    chart = _interval_chart(
        rows=_rows(report=report, cuts=cuts, arms=["m", "q2"]),
        colour_field="series",
        domain=[ARM_LABELS["m"], ARM_LABELS["q2"]],
        colours=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
        title=_title(
            text="The two-number arm gains at the same substations as the 51-member arm",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours. Dashed rule: same as arm A.",
        ),
        height=620,
    )
    _save(chart, name="figure_3_substation")


def main() -> None:
    """Draw every figure."""
    report = json.loads((REPORT / "report.json").read_text())
    figure_headline(report=report)
    figure_lead(report=report)
    figure_substation(report=report)


if __name__ == "__main__":
    main()
