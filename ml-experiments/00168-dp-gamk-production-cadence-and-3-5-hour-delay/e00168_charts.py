"""Draw the figures of experiment 168's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00168-*/e00168_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00168_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
WRAP: Final[int] = 105
GREY: Final[str] = "#808080"
PERIOD_COLOURS: Final[dict[str, str]] = {
    "even months (replicate)": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
}
ARM_LABELS: Final[dict[str, str]] = {
    "d": "Daily, with delay",
    "di0": "Daily, no delay",
    "dp": "6-hourly, with delay",
    "dp0": "6-hourly, no delay",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "d": ocf.DATA_BLUE,
    "di0": ocf.DATA_SKY,
    "dp": ocf.BRAND_ORANGE,
    "dp0": ocf.DATA_PURPLE,
}
CONTRAST_NAMES: Final[dict[str, str]] = {
    "C1": "C1 6-hourly minus daily, same lead",
    "C4": "C4 6-hourly minus daily, same valid time",
    "C3": "C3 3.5-hour delay, 6-hourly",
    "C7": "C7 3.5-hour delay, daily at 09:00",
    "C5": "C5 offset and R, 6-hourly",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _forest(
    *, rows: list[dict], order: list[str], title: alt.TitleParams, height: int
) -> alt.Chart:
    present = {row["period"] for row in rows}
    colours = {k: v for k, v in PERIOD_COLOURS.items() if k in present}
    base = alt.Chart(pl.DataFrame(rows)).encode(  # ty: ignore[unresolved-attribute]
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=320)),
        yOffset=alt.YOffset("period:N", sort=list(colours)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", direction="horizontal"),
        ),
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    rule = base.mark_rule(strokeWidth=2).encode(x="low:Q", x2="high:Q")  # ty: ignore[unresolved-attribute]
    dot = base.mark_point(filled=True, size=60).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X("estimate:Q", title="Difference in quantile score (points; below zero is better)")
    )
    return (zero + rule + dot).properties(width=WIDTH - 330, height=height, title=title)


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The planned and post hoc contrasts over leads under 24 hours, with 99% intervals."""
    rows = []
    for period, label in (("replicate", "even months (replicate)"), ("scored", "scored year")):
        for key, name in CONTRAST_NAMES.items():
            c = data[period]["contrasts"][key]["bands"]["under 24 h"]["qs"]
            rows.append(
                {
                    "label": name,
                    "period": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    return _forest(
        rows=rows,
        order=list(CONTRAST_NAMES.values()),
        height=320,
        title=_title(
            text="Same lead: 6-hourly scores 0.08 points worse. Same valid time: no worse",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 24 hours, five decision "
                "substations, outside switching events. Line: 99% interval from resampling whole "
                "calendar months. The line at zero means no difference."
            ),
        ),
    )


def figure_by_hour(*, data: dict) -> alt.TopLevelMixin:
    """Mean score at leads under 24 hours by UTC hour of the valid time."""
    rows = [
        {"arm": ARM_LABELS[arm], "hour": item["hour"], "score": item["qs"]}
        for arm in ARM_LABELS
        for item in data["scored"]["tables"]["by_valid_hour_under_24"][arm]
    ]
    labels = list(ARM_LABELS.values())
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "hour:Q",
                title="Hour of the valid time (UTC)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 23]),
            ),
            y=alt.Y("score:Q", title="Mean quantile score (points; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in ARM_LABELS]),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(
            width=WIDTH - 60,
            height=260,
            title=_title(
                text="Hour of day sets the score; the daily forecast is best just after 09:00",
                subtitle=(
                    "Mean quantile score over leads 0.5 to 24 hours, by valid-time hour. "
                    "Scored year, five decision substations. Daily: one origin at 09:00 UTC. "
                    "6-hourly: four origins. Every arm has the same weather input."
                ),
            ),
        )
    )


def figure_by_slot(*, data: dict) -> alt.TopLevelMixin:
    """C1 and C3 by production origin, leads under 24 hours."""
    rows = []
    for key, name in (
        ("C1", "C1 same lead"),
        ("C3", "C3 delay"),
    ):
        for slot, c in data["scored"]["contrasts"][key]["by_slot_under_24"].items():
            rows.append(
                {
                    "label": f"{name}, origin {slot}",
                    "period": "scored year",
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    order = [r["label"] for r in rows]
    return _forest(
        rows=rows,
        order=order,
        height=260,
        title=_title(
            text="The same-lead gap is smallest at the 12:00 origin",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 24 hours, scored "
                "year, five decision substations. Line: 99% interval from resampling whole "
                "calendar months. Origins at 00:00 and 06:00 UTC fall on the day after the run."
            ),
        ),
    )


def figure_delay_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """C3, the 3.5-hour delay at the production origins, by lead band."""
    bands = ["0.5-3.5 h", "3.5-9.5 h", "9.5-24 h", "24-72 h", "72-147 h"]
    rows = []
    for period, label in (("scored", "scored year"),):
        for band in bands:
            c = data[period]["contrasts"]["C3"]["bands"][band]["qs"]
            rows.append(
                {
                    "label": f"leads {band}",
                    "period": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    return _forest(
        rows=rows,
        order=[f"leads {b}" for b in bands],
        height=200,
        title=_title(
            text="The 3.5-hour delay costs 0.3 points in the first 3.5 hours, then little",
            subtitle=(
                "Dot: score with the 3.5-hour delay minus the same forecasts with no delay, "
                "every 6 hours, scored year, five decision substations. Line: 99% interval "
                "from resampling whole calendar months."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_by_hour(data=data), name="figure_2_by_hour")
    _save(figure_by_slot(data=data), name="figure_3_by_origin")
    _save(figure_delay_by_lead(data=data), name="figure_4_delay_by_lead")


if __name__ == "__main__":
    main()
