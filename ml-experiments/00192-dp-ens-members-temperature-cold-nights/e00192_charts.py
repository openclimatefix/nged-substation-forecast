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
WIDTH: Final[int] = 480
ARM_LABELS: Final[dict[str, str]] = {
    "t": "T: members through temperature",
    "st": "ST: members through solar and temperature",
    "m": "M: members through solar only (experiment 187)",
    "mr": "Rescale only, no members",
    "tp": "T with another day's members",
}
CUT_LABELS: Final[dict[str, str]] = {
    "cold nights": "Cold nights",
    "winter evenings (Nov to Feb, 17:00 to 23:00)": "Winter evenings",
    "non-cold rows": "All other rows",
    "daylight": "Daylight",
}


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


def _one_panel(
    *, frame: pl.DataFrame, height: int, title: str | alt.TitleParams | None
) -> alt.TopLevelMixin:
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color="#808080", strokeDash=[4, 3])
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    colour = alt.Color(
        "period:N",
        scale=alt.Scale(
            domain=["scored year", "replicate months"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
        ),
        title=None,
    )
    line = (
        alt.Chart(frame)
        .mark_rule(strokeWidth=2)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
            x=alt.X(
                "low:Q",
                scale=alt.Scale(domain=[-3.5, 3.5]),
                title="Change in the quantile score from arm A (%; more negative is better)",
            ),
            x2="high:Q",
            color=colour,
            yOffset="period:N",
        )
    )
    point = (
        alt.Chart(frame)
        .mark_point(filled=True, size=50)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=None),
            x=alt.X("point:Q", scale=alt.Scale(domain=[-3.5, 3.5])),
            color=colour,
            yOffset="period:N",
        )
    )
    return alt.layer(zero, line, point).properties(width=WIDTH, height=height, title=title or "")


def _interval_chart(
    *, rows: list[dict], title: alt.TitleParams, row_field: str, height: int
) -> alt.TopLevelMixin:
    frame = pl.DataFrame(rows)
    if not row_field:
        return _one_panel(frame=frame, height=height, title=title)
    panels = [
        _one_panel(frame=frame.filter(pl.col(row_field) == name), height=height, title=name)
        for name in frame[row_field].unique(maintain_order=True)
    ]
    return alt.vconcat(*panels).properties(title=title).resolve_scale(color="shared")  # ty: ignore[invalid-argument-type]


def main() -> None:
    """Write the three figures."""
    report = json.loads((REPORT / "report.json").read_text())
    extra = json.loads((REPORT / "extra.json").read_text())
    period_names = {"scored": "scored year", "replicate": "replicate months"}

    rows = []
    for cut, cut_label in CUT_LABELS.items():
        for arm, arm_label in ARM_LABELS.items():
            for period, period_label in period_names.items():
                change = report[period]["subsets"][cut][arm]["change_pct"]
                rows.append(
                    {
                        "cut": cut_label,
                        "label": arm_label,
                        "period": period_label,
                        "point": change["point"],
                        "low": change["lower"],
                        "high": change["upper"],
                    }
                )
    _save(
        _interval_chart(
            rows=rows,
            title=_title(
                text="Members' temperature does not improve cold-night bands",
                subtitle="Cold nights: ensemble-mean temperature under 2 C, local 17:00 to 08:00. Month-block 95% intervals (97.5% for T and ST).",
            ),
            row_field="cut",
            height=170,
        ),
        name="figure_1_headline",
    )

    rows = []
    for site in common.DECISION_SITES:
        for period, period_label in period_names.items():
            change = report[period]["subsets"][f"cold nights, {site}"]["t"]["change_pct"]
            rows.append(
                {
                    "cut": "",
                    "label": site,
                    "period": period_label,
                    "point": change["point"],
                    "low": change["lower"],
                    "high": change["upper"],
                }
            )
    _save(
        _interval_chart(
            rows=rows,
            title=_title(
                text="Cold-night change by substation, arm T",
                subtitle="No substation shows a gain in both periods with an interval below zero.",
            ),
            row_field="",
            height=260,
        ),
        name="figure_2_substation",
    )

    scored = extra["scored"]
    spread = pl.DataFrame(
        {
            "label": [
                "Arm A's half-width, q05 to q95, cold nights",
                "Spread of the members' temperature responses, cold nights",
                "Spread of the members' solar responses, daylight",
            ],
            "points": [
                scored["arm_a_cold_half_width_q05_q95"],
                scored["spread"]["delta_t"]["cold_std_of_members"],
                scored["spread"]["delta_m"]["daylight_std_of_members"],
            ],
        }
    )
    bars = (
        alt.Chart(spread)
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
            x=alt.X("points:Q", title="Points of effective capacity (scored year)"),
        )
    )
    _save(
        bars.properties(
            width=WIDTH,
            height=120,
            title=_title(
                text="The members' temperatures disagree too little to matter on cold nights",
                subtitle="Mean over rows of the standard deviation of the 51 members' responses.",
            ),
        ),
        name="figure_3_spread",
    )


if __name__ == "__main__":
    main()
