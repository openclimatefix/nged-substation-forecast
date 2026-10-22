"""Draw the figures of experiment 170's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00170-*/e00155_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00170_stage2_common as common
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
    "a155": "A stage 1, no drift (experiment 155)",
    "c155": "C A + recipe R, no drift",
    "d155": "D A + offset + R, no drift",
    "s24_a": "A + drift, 24 months",
    "s24_c": "C + drift, 24 months",
    "s24_d": "D + drift, 24 months",
    "sall_a": "A + drift, all years",
    "sall_c": "C + drift, all years",
    "sall_d": "D + drift, all years",
    "bg2": "BG2",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "a155": GREY,
    "c155": GREY,
    "d155": DARK,
    "s24_a": ocf.DATA_SKY,
    "s24_c": ocf.DATA_BLUE,
    "s24_d": ocf.DATA_BLUE,
    "sall_a": ocf.BRAND_ORANGE,
    "sall_c": ocf.BRAND_ORANGE,
    "sall_d": ocf.BRAND_ORANGE,
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
        "C1": "C1 D + drift (24 months) minus D",
        "C2": "C2 D + drift, all years minus 24 months",
        "C3": "C3 D + drift, all years minus D",
        "C4": "C4 drop the offset (24 months)",
        "C5": "C5 drop recipe R (24 months)",
        "C6": "C6 drop the offset (all years)",
        "C7": "C7 drop recipe R (all years)",
        "C8": "C8 drift alone (24 months) minus A",
        "C12": "C12 D + drift (24 months) minus BG2",
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
        height=330,
        title=_title(
            text="The drift term raised the score; the night offset still adds nothing",
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours, five decision "
                "substations (D5, D10, D11, D14, D16), outside switching events. Line: 99% "
                "interval from resampling whole calendar months. C4 to C7 are each the full "
                "assembly minus the assembly with one part dropped."
            ),
        ),
    )


def figure_absolute(*, data: dict) -> alt.TopLevelMixin:
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
            height=170,
            title=_title(
                text="The drift arms sit just above their no-drift twins at every lead band",
                subtitle="Scored year, five decision substations, outside switching events.",
            ),
        )
    )


def figure_filter(*, nofilter: dict) -> alt.TopLevelMixin:
    """Quantile score of the three stage 1 fit sets with and without the Kalman filter."""
    fit_sets = {
        "a2_repro": "No drift, 24 months",
        "m24_d1": "Drift, 24 months",
        "all_d1": "Drift, all years",
    }
    rows = []
    for item in nofilter["table"]:
        arm = item["arm"].removesuffix("_nf")
        rows.append(
            {
                "fit": fit_sets[arm],
                "filter": "without the filter"
                if item["arm"].endswith("_nf")
                else "with the filter",
                "score": item["qs"],
            }
        )
    colours = {"without the filter": ocf.DATA_SKY, "with the filter": ocf.BRAND_ORANGE}
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("fit:N", sort=list(fit_sets.values()), title=None),
            yOffset=alt.YOffset("filter:N", sort=list(colours)),
            x=alt.X("score:Q", title="Mean quantile score (points of capacity; lower is better)"),
            color=alt.Color(
                "filter:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", direction="horizontal"),
            ),
        )
        .properties(
            width=WIDTH - 200,
            height=180,
            title=_title(
                text="The Kalman filter removes the level error that the drift term would fix",
                subtitle=(
                    "Stage 1 forecasts before the offset and recipe R, scored year, leads 0.5 to "
                    "168 hours, five decision substations. Without the filter, drift lowers the "
                    "score by about 0.2 points; with it, drift raises the score by about 0.03."
                ),
            ),
        )
    )


def figure_events(*, events: dict) -> alt.TopLevelMixin:
    """Event-day score as a multiple of ordinary weekdays at the same half-hours, by arm."""
    arms = ("a155", "c155", "s24_a", "s24_c")
    rows = [
        {
            "event": {
                "christmas": "24 Dec to 2 Jan",
                "easter": "Easter",
                "bank_holiday": "Other bank holidays",
            }[r["event"]],
            "period": {"replicate": "2024-25", "scored": "2025-26"}[r["period"]],
            "arm": ARM_LABELS[r["arm"]],
            "ratio": r["ratio"],
        }
        for r in events["ratios"]
        if r["scope"] == "all"
        and r["reference"] == "weekday"
        and r["period"] in ("replicate", "scored")
        and r["arm"] in arms
        and "ratio" in r
    ]
    labels = [ARM_LABELS[a] for a in arms]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(domain=["2024-25", "2025-26"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(title="Christmas or year", orient="top", direction="horizontal"),
        ),
    )
    dots = base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "ratio:Q",
            title="Quantile score on event days / ordinary weekdays, same half-hours",
            scale=alt.Scale(domain=[0.9, 1.3]),
            axis=alt.Axis(values=[0.9, 1.0, 1.1, 1.2, 1.3]),
        )
    )
    return dots.encode(
        row=alt.Row("event:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left"))
    ).properties(
        width=WIDTH - 260,
        height=90,
        title=_title(
            text="Christmas is 1.1 to 1.2 times as hard to forecast as a weekday, not 1.5",
            subtitle=(
                "Five decision substations, leads 0.5 to 168 hours. A ratio of 1 means event "
                "days score the same as ordinary weekdays at the same local half-hours of the "
                "day. The 2024-25 Christmas has December origins only."
            ),
        ),
    )


def main() -> None:
    """Draw every figure."""
    folder = common.WORK_DIR / "report"
    data = json.loads((folder / "report.json").read_text())
    nofilter = json.loads((folder / "nofilter.json").read_text())
    events = json.loads((folder / "events.json").read_text())
    _save(figure_headline(data=data), name="figure_1_headline")
    _save(figure_absolute(data=data), name="figure_2_absolute")
    _save(figure_filter(nofilter=nofilter), name="figure_3_filter")
    _save(figure_events(events=events), name="figure_4_events")


if __name__ == "__main__":
    main()
