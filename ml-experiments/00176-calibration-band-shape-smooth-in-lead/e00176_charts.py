"""Draw the figures of experiment 176's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00176-*/e00176_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00176_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
REPORT: Final[Path] = common.WORK_DIR / "report" / "report.json"
WIDTH: Final[int] = 680
GREY: Final[str] = "#808080"
ARM_NAMES: Final[dict[str, str]] = {
    "a": "A: reference",
    "pb": "PB: growth per hour band",
    "p": "P: growth per hour band and quarter",
    "u": "U: growth per substation",
    "s": "S: shrunk cells",
    "ps": "PS: P with shrunk scales",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "A: reference": GREY,
    "PB: growth per hour band": ocf.DATA_SKY,
    "P: growth per hour band and quarter": ocf.BRAND_ORANGE,
    "U: growth per substation": ocf.DATA_PURPLE,
    "S: shrunk cells": ocf.DATA_BLUE,
    "PS: P with shrunk scales": ocf.DATA_GREEN,
}
DAYS: Final[list[str]] = [f"{d}" for d in range(common.N_LEAD_DAYS)]


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, 105),
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


def _colour(*, title: str = "Arm") -> alt.Color:
    return alt.Color(
        "arm:N",
        title=title,
        scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
        legend=alt.Legend(orient="top", columns=2, labelLimit=400),
    )


def width_frame(*, report: dict, site: str, label: str) -> pl.DataFrame:
    """Return the median width by lead day of every arm at one substation and row set."""
    rows = [
        {"arm": ARM_NAMES[arm], "day": day, "width": value, "label": label}
        for arm in common.ARMS
        for day, value in enumerate(report["scored"]["widths"][f"{arm}|{site}|{label}|median"])
    ]
    return pl.DataFrame(rows)


def figure_d5(*, report: dict) -> alt.TopLevelMixin:
    """Figure 1: D5's median width by lead day on the scoring rows and on all rows."""
    panels = []
    for label, heading in (("kept", "Rows outside the switching events"), ("all", "All rows")):
        data = width_frame(report=report, site="D5", label=label)
        panels.append(
            alt.Chart(data, title=heading, width=WIDTH - 90, height=170)  # ty: ignore[unresolved-attribute]
            .mark_line(point=True, clip=True)
            .encode(
                x=alt.X(
                    "day:O",
                    title="Lead day (0 = up to 24 hours ahead)",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    "width:Q", title="Median q05 to q95 width (points)", scale=alt.Scale(zero=False)
                ),
                color=_colour(),
            )
        )
    return alt.vconcat(*panels).properties(
        title=_title(
            text="D5's band width rises with lead on the scored rows; the dips appear on all rows",
            subtitle=(
                "Scored year (2025-07-02 to 2026-06-29). Width is the q05 to q95 interval, "
                "in points of effective capacity."
            ),
        )
    )


def figure_all_sites(*, report: dict) -> alt.TopLevelMixin:
    """Figure 2: the same curve for the five decision substations and D9, all rows."""
    frames = [
        width_frame(report=report, site=s, label="all").with_columns(site=pl.lit(s))
        for s in common.SCORED_SITES
    ]
    data = pl.concat(frames)
    return (
        alt.Chart(data, width=(WIDTH - 120) // 2, height=120)  # ty: ignore[unresolved-attribute]
        .mark_line(point=False, clip=True)
        .encode(
            x=alt.X("day:O", title="Lead day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("width:Q", title="Median width (points)", scale=alt.Scale(zero=False)),
            color=_colour(),
            facet=alt.Facet("site:N", columns=2, title=None),
        )
        .properties(
            title=_title(
                text="Only D5 shows dips; the other substations' widths rise smoothly with lead",
                subtitle=(
                    "Median q05 to q95 width, in points of effective capacity, "
                    "over the rows of each substation."
                ),
            )
        )
    )


def figure_contrasts(*, report: dict) -> alt.TopLevelMixin:
    """Figure 3: the quantile-score difference from arm A at every substation, with intervals."""
    rows = []
    for period in ("scored", "replicate"):
        for arm in ("p", "s", "ps"):
            for site, c in report[period]["contrasts"][arm].items():
                rows.append({"period": period, "arm": ARM_NAMES[arm], "site": site, **c})
    data = pl.DataFrame(rows)
    base = alt.Chart(data, width=WIDTH - 150, height=150)
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="black").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    limit = (
        alt.Chart(pl.DataFrame({"x": [0.005]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=GREY, strokeDash=[4, 3])
        .encode(x="x:Q")
    )
    layers = (
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("lower:Q", title="Quantile score minus arm A's (points; negative is better)"),
            x2="upper:Q",
            y=alt.Y("site:N", title=None, sort=["pooled", "D5", "D10", "D11", "D14", "D16", "D9"]),
            color=_colour(),
            yOffset="arm:N",
        )
        + base.mark_point(filled=True, size=40).encode(  # ty: ignore[unresolved-attribute]
            x="estimate:Q", y="site:N", color=_colour(), yOffset="arm:N"
        )
        + zero
        + limit
    )
    return layers.facet(row=alt.Row("period:N", title=None)).properties(
        title=_title(
            text="Shrinking the stored cells helps D10; refitting a smooth curve costs score",
            subtitle=(
                "Dot: difference. Line: 95% interval from resampling whole months. "
                "Dashed rule: the +0.005 limit."
            ),
        )
    )


def figure_coverage(*, report: dict) -> alt.TopLevelMixin:
    """Figure 4: q95 coverage by lead day at the five decision substations."""
    rows = [
        {"arm": ARM_NAMES[arm], "day": day, "coverage": value}
        for arm in common.ARMS
        for day, value in enumerate(report["scored"]["by_site"][arm]["cov_by_day"]["q95"])
    ]
    data = pl.DataFrame(rows)
    line = (
        alt.Chart(data, width=WIDTH - 90, height=200)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("day:O", title="Lead day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "coverage:Q", title="Share of readings at or under q95", scale=alt.Scale(zero=False)
            ),
            color=_colour(),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"y": [0.95]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="black", strokeDash=[4, 3])
        .encode(y="y:Q")
    )
    return (line + rule).properties(
        title=_title(
            text="q95 coverage by lead day stays near 0.95 for every arm",
            subtitle=(
                "Five decision substations, scored year, rows outside the switching events. "
                "Dashed rule: nominal 0.95."
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    report = json.loads(REPORT.read_text())
    alt.theme.enable("ocf") if hasattr(alt, "theme") else None
    _save(figure_d5(report=report), name="figure_1_d5_width_by_lead_day")
    _save(figure_all_sites(report=report), name="figure_2_width_all_substations")
    _save(figure_contrasts(report=report), name="figure_3_score_contrasts")
    _save(figure_coverage(report=report), name="figure_4_q95_coverage_by_lead_day")


if __name__ == "__main__":
    main()
