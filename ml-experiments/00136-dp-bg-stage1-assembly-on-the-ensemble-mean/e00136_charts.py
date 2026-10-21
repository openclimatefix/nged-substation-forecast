"""Draw the figures of experiment 136's write-up from `work/report/report.json`.

uv run python ml-experiments/00136-*/e00136_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00136_common as common
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
    "a0": "A0 GAM-K on the ensemble mean",
    "a1": "A1 A0 + T-all",
    "a2": "A2 A1 + POOL noise",
    "a3": "A3 A2 refitted weekly",
    "k111": "Experiment 111's GAM-K",
    "bg2": "BG2",
    "manual_heuristic": "Manual heuristic",
    "climatology": "Climatology",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "a0": ocf.DATA_SKY,
    "a1": ocf.DATA_BLUE,
    "a2": ocf.BRAND_ORANGE,
    "a3": ocf.DATA_PURPLE,
    "k111": DARK,
    "bg2": ocf.DATA_GREEN,
    "manual_heuristic": GREY,
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
    """The six planned contrasts over all leads, with 99% intervals."""
    names = {
        "C6": "C6 A2 minus experiment 111's GAM-K",
        "C5": "C5 A2 minus BG2",
        "C1": "C1 A0 minus experiment 111's GAM-K",
        "C2": "C2 A1 minus A0 (T-all)",
        "C3": "C3 A2 minus A1 (POOL noise)",
        "C4": "C4 A3 minus A2 (weekly refits)",
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
            title=(
                "Difference in quantile score (points of capacity; "
                "below zero favours the first arm)"
            ),
            axis=alt.Axis(values=[-0.2, -0.1, 0.0, 0.1, 0.2]),
        )
    )
    return (zero + rule + dot).properties(
        width=WIDTH - 260,
        height=260,
        title=_title(
            text=(
                "Adding T-all helped; the assembled arm is not distinguishable from "
                "BG2 or from GAM-K"
            ),
            subtitle=(
                "Dot: difference in mean quantile score over leads 0.5 to 168 hours, five decision "
                "substations (D5, D10, D11, D14, D16), outside switching events. Line: 99% "
                "interval from resampling whole calendar months. BG2 exists only for the "
                "scored year."
            ),
        ),
    )


def figure_by_lead(*, data: dict) -> alt.TopLevelMixin:
    """Each arm's absolute quantile score by lead band, scored year."""
    order = list(ARM_LABELS)
    rows = []
    for band in ("under 24 h", "24-72 h", "72-168 h", "all"):
        rows.extend(
            {"band": band, "arm": ARM_LABELS[item["arm"]], "score": item["qs"]}
            for item in data["scored"]["tables"]["by_lead"][band]
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
            row=alt.Row("band:N", sort=["under 24 h", "24-72 h", "72-168 h", "all"], title=None),
        )
        .properties(
            width=WIDTH - 240,
            height=110,
            title=_title(
                text="The assembled arm matches BG2 and is far ahead of the references",
                subtitle=(
                    "Scored year, five decision substations, outside switching events. Rows are "
                    "lead bands."
                ),
            ),
        )
    )


def figure_breakouts(*, data: dict) -> alt.TopLevelMixin:
    """Each DP arm's score minus experiment 111's GAM-K, by breakout."""
    rows = []
    for label, entry in data["scored"]["tables"]["breakouts"].items():
        scores = {s["arm"]: s["qs"] for s in entry["scores"]}
        rows.extend(
            {
                "subset": f"{label} ({entry['rows']:,} rows)",
                "arm": ARM_LABELS[arm],
                "difference": scores[arm] - scores["k111"],
            }
            for arm in ("a0", "a1", "a2", "a3", "bg2")
        )
    labels = [ARM_LABELS[a] for a in ("a0", "a1", "a2", "a3", "bg2")]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("arm:N", sort=labels, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(
                "difference:Q",
                title="Quantile score minus experiment 111's GAM-K (points; below zero is better)",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=labels, range=[ARM_COLOURS[a] for a in ("a0", "a1", "a2", "a3", "bg2")]
                ),
                legend=None,
            ),
            row=alt.Row("subset:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        )
        .properties(
            width=WIDTH - 260,
            height=70,
            title=_title(
                text="The gains concentrate in winter and on cold nights; holidays lose",
                subtitle=(
                    "Scored year, all leads, five decision substations. Each bar is one arm's "
                    "mean quantile score minus experiment 111's GAM-K on the same rows."
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


if __name__ == "__main__":
    main()
