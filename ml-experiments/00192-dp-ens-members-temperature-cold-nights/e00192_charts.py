# ruff: noqa: E501
"""Draw the figures of experiment 192's write-up from `work/report/*.json`.

uv run --no-sync python ml-experiments/00192-*/e00192_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00192_common as common
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
PLACEBO_LABELS: Final[dict[str, str]] = {
    "m": "M: the day's own members",
    "mp_strong": "Other day's members (lead, cloud band)",
    "mp": "Other day's members (month, hour)",
    "gauss": "Gaussian, row's own mean and spread",
    "flip": "Responses reflected about mean",
}
PLACEBO_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
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


def figure_headline(*, placebos: dict) -> None:
    """Change in the daylight score for arm M and each placebo, both periods."""
    rows = []
    for period, label in (("scored", "scored year"), ("replicate", "replicate months")):
        for arm, arm_label in PLACEBO_LABELS.items():
            c = placebos[period]["daylight"][arm]
            rows.append(
                {
                    "label": f"{arm_label}, {label.split()[0]}",
                    "series": arm_label,
                    "estimate": c["point"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    chart = _interval_chart(
        rows=rows,
        colour_field="series",
        domain=list(PLACEBO_LABELS.values()),
        colours=PLACEBO_COLOURS,
        title=_title(
            text="Arm M lowers the daylight score by 2.2% in the scored year; the day's own members carry most of it",
            subtitle="Dot: estimate. Line: 95% interval from resampling whole months. Daylight half-hours (sun 5 degrees or more up), 5 substations, leads 0.5 to 168 hours. Dashed rule: same as arm A.",
        ),
        height=330,
    )
    _save(chart, name="figure_1_headline")


def figure_lead(*, report: dict) -> None:
    """Change in the daylight score by lead band, both periods."""
    cuts = [
        ("daylight, under 24 h", "under 24 h"),
        ("daylight, 24 to 72 h", "24 to 72 h"),
        ("daylight, 72 to 168 h", "72 to 168 h"),
    ]
    rows = []
    for period, label in (("scored", "scored year"), ("replicate", "replicate months")):
        for cut, cut_label in cuts:
            c = report[period]["subsets"][cut]["m"]["change_pct"]
            rows.append(
                {
                    "label": f"{cut_label}, {label.split()[0]}",
                    "series": label,
                    "estimate": c["point"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    chart = _interval_chart(
        rows=rows,
        colour_field="series",
        domain=["scored year", "replicate months"],
        colours=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
        title=_title(
            text="The gain from arm M grows with lead time and holds in both periods",
            subtitle="Dot: estimate. Line: 97.5% interval from resampling whole months. Daylight half-hours, 5 substations. Dashed rule: same as arm A.",
        ),
        height=230,
    )
    _save(chart, name="figure_2_lead")


def figure_substation(*, report: dict) -> None:
    """Change in the daylight score by substation, scored year and replicate months."""
    rows = []
    for period, label in (("scored", "scored year"), ("replicate", "replicate months")):
        for site in common.DECISION_SITES:
            c = report[period]["subsets"][f"daylight, {site}"]["m"]["change_pct"]
            rows.append(
                {
                    "label": f"{site}, {label.split()[0]}",
                    "series": label,
                    "estimate": c["point"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    chart = _interval_chart(
        rows=rows,
        colour_field="series",
        domain=["scored year", "replicate months"],
        colours=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
        title=_title(
            text="Arm M gains at every substation in the scored year; D11's replicate interval includes zero",
            subtitle="Dot: estimate. Line: 97.5% interval from resampling whole months. Daylight half-hours. Dashed rule: same as arm A.",
        ),
        height=300,
    )
    _save(chart, name="figure_3_substation")


def main() -> None:
    """Draw every figure."""
    report = json.loads((REPORT / "report.json").read_text())
    placebos = json.loads((REPORT / "placebos.json").read_text())
    figure_headline(placebos=placebos)
    figure_lead(report=report)
    figure_substation(report=report)


if __name__ == "__main__":
    main()
