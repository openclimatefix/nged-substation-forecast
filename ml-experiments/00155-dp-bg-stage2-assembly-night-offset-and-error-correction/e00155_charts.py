"""Draw the figures of experiment 155's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00155-*/e00155_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00155_common as common
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
DARK: Final[str] = "#333333"
ARM_LABELS: Final[dict[str, str]] = {
    "a": "A stage 1 (experiment 136's A2)",
    "b": "B A + winter night offset",
    "c": "C A + recipe R",
    "d": "D B + recipe R",
    "e": "E B + analog correction",
    "k111": "Experiment 111's GAM-K",
    "bg2": "BG2",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "a": GREY,
    "b": ocf.DATA_SKY,
    "c": ocf.BRAND_ORANGE,
    "d": ocf.DATA_BLUE,
    "e": ocf.DATA_PURPLE,
    "k111": DARK,
    "bg2": ocf.DATA_GREEN,
    "manual_heuristic": "#999999",
    "climatology": "#BBBBBB",
}
PERIOD_COLOURS: Final[dict[str, str]] = {
    "even months (replicate)": ocf.DATA_BLUE,
    "scored year": ocf.BRAND_ORANGE,
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


def figure_headline(*, data: dict) -> alt.TopLevelMixin:
    """The planned contrasts over all leads, with 99% intervals."""
    names = {
        "C3": "C3 D minus A (offset and R)",
        "C2": "C2 C minus A (R)",
        "C1": "C1 B minus A (offset)",
        "C4": "C4 D minus C (offset on top of R)",
        "C5": "C5 D minus B (R on top of offset)",
        "C7": "C7 E minus D (analog in place of R)",
        "C8": "C8 C48 minus C (R to 48 h only)",
        "C6": "C6 D minus BG2",
    }
    rows = []
    for period, label in (("replicate", "even months (replicate)"), ("scored", "scored year")):
        for key, name in names.items():
            if key not in data[period]["contrasts"]:
                continue
            c = data[period]["contrasts"][key]["all"]["qs"]
            rows.append(
                {
                    "contrast": name,
                    "period": label,
                    "estimate": c["estimate"],
                    "low": c["lower"],
                    "high": c["upper"],
                }
            )
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=300)),
        yOffset=alt.YOffset("period:N", sort=list(PERIOD_COLOURS)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=list(PERIOD_COLOURS), range=list(PERIOD_COLOURS.values())),
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
        x=alt.X(
            "estimate:Q",
            title="Difference in quantile score (points; below zero favours the first arm)",
        )
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 260,
        height=300,
        title=_title(
            text="Recipe R carries the gain; the night offset adds nothing once R is in",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours, five decision "
                "substations (D5, D10, D11, D14, D16), outside switching events. Line: 99% "
                "interval from resampling whole calendar months."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Each arm's absolute quantile score by lead band, scored year."""
    order = list(ARM_LABELS)
    bands = ("under 24 h", "24-72 h", "72-168 h", "all")
    rows = []
    for band in bands:
        rows.extend(
            {"band": band, "arm": ARM_LABELS[item["arm"]], "score": item["qs"]}
            for item in data["scored"]["tables"]["by_lead"][band]
            if item["arm"] in ARM_LABELS
        )
    labels = [ARM_LABELS[a] for a in order]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("score:Q", title="Mean quantile score (points of capacity; lower is better)"),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in order]),
                legend=None,
            ),
            row=alt.Row("band:N", sort=list(bands), title=None),
        )
        .properties(
            width=WIDTH - 240,
            height=130,
            title=_title(
                text="R's gain is largest below 72 hours; BG2 stays ahead beyond 72 hours",
                subtitle="Scored year, five decision substations, outside switching events.",
            ),
        )
    )


def figure_breakouts(*, data: dict) -> alt.TopLevelMixin:
    """Each arm's score minus arm A's, by breakout."""
    rows = []
    for label, entry in data["scored"]["tables"]["breakouts"].items():
        if label in ("cold nights (3 C or below, as experiment 136)", "holidays and Christmas"):
            continue
        scores = {s["arm"]: s["qs"] for s in entry["scores"]}
        rows.extend(
            {
                "subset": f"{label} ({entry['rows']:,} rows)",
                "arm": ARM_LABELS[arm],
                "difference": scores[arm] - scores["a"],
            }
            for arm in ("b", "c", "d", "e", "bg2")
        )
    arms = ("b", "c", "d", "e", "bg2")
    labels = [ARM_LABELS[a] for a in arms]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(
                "difference:Q",
                title="Quantile score minus arm A (points; below zero is better)",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in arms]),
                legend=None,
            ),
            row=alt.Row("subset:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 260,
            height=70,
            title=_title(
                text="Both corrections help winter nights; BG2 is far ahead only at Christmas",
                subtitle=(
                    "Scored year, all leads, five decision substations. Each bar is one arm's mean "
                    "quantile score minus arm A's on the same rows."
                ),
            ),
        )
    )


def figure_half_hour(*, data: dict) -> alt.TopLevelMixin:
    """Mean score by UTC half-hour on winter nights, for arms A, B, and D."""
    rows = [
        {"arm": ARM_LABELS[x["arm"]], "hour": (x["minute"] / 60 + 3) % 24 - 3, "score": x["qs"]}
        for x in data["scored"]["winter_night_by_half_hour"]
    ]
    arms = ("a", "b", "d")
    labels = [ARM_LABELS[a] for a in arms]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "hour:Q", title="UTC hour at which the half-hour ends (21 to 9, wrapping midnight)"
            ),
            y=alt.Y("score:Q", title="Mean quantile score (points)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "arm:N",
                sort=labels,
                scale=alt.Scale(domain=labels, range=[ARM_COLOURS[a] for a in arms]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 100,
            height=240,
            title=_title(
                text="The offset lowers the score from 01:00 UTC and leaves 22:00 to 23:00 level",
                subtitle=(
                    "Scored year, November to March, five decision substations, all leads, "
                    "half-hours ending 21:00 to 09:00 UTC."
                ),
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads((common.WORK_DIR / "report" / "report.json").read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_by_lead(data=data), name="figure_2_by_lead")
    _save(figure_breakouts(data=data), name="figure_3_breakouts")
    _save(figure_half_hour(data=data), name="figure_4_half_hour")


if __name__ == "__main__":
    main()
