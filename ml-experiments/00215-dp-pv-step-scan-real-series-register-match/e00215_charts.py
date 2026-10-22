"""Draw the figures of experiment 215's write-up from `work/paths.parquet` and `work/steps.parquet`.

uv run --no-sync python ml-experiments/00215-*/e00215_charts.py
"""

import json
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00215_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 600
GREY: Final[str] = "#444444"
SITE_ORDER: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16"]
EXPERIMENT_207_FALSE_STEPS: Final[dict[str, float]] = {
    "D5": 0.00, "D10": 0.60, "D11": 0.67, "D14": 0.53, "D16": 0.13,
}  # fmt: skip
"""Experiment 207's false steps above 0.5 MW per no-injection year at weight 3 (its Checks)."""
REGISTER_END: Final[date] = date(2021, 3, 24)
RULE_WEIGHT: Final[float] = 3.0
REAL_LABEL: Final[str] = "Real series"
NO_STEP_LABEL: Final[str] = "Experiment 207, no step"
FULL_MONTHS: Final[int] = 60
"""The months from 2021-07 to 2026-06; D9 has fewer."""


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG beside the page and as PNG in the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def month_start(*, index: int) -> date:
    """Return the first day of a month given as months since 1970-01."""
    return date(1970 + index // 12, index % 12 + 1, 1)


def paths_chart(*, paths: pl.DataFrame, steps: pl.DataFrame) -> alt.Chart:
    """Figure 1: the fitted solar capacity of each Primary, with the steps above 0.5 MW marked."""
    line = paths.filter(pl.col("weight") == RULE_WEIGHT).with_columns(
        date=pl.col("month_index").map_elements(
            lambda i: month_start(index=i), return_dtype=pl.Date
        )
    )
    step_keys = steps.filter(
        (pl.col("weight") == RULE_WEIGHT) & (pl.col("kind") == "month")
    ).select("site", "month_index", is_step=pl.lit(True))
    line = line.join(step_keys, on=["site", "month_index"], how="left").with_columns(
        is_step=pl.col("is_step").fill_null(False)
    )
    base = alt.Chart(line).encode(
        x=alt.X("date:T", title="Month (April to August months carry the data)"),
        y=alt.Y("level_mw:Q", title="Fitted solar capacity (MW)"),
    )
    lines = base.mark_line(color=ocf.DATA_BLUE, strokeWidth=2)
    points = base.mark_point(color=ocf.BRAND_ORANGE, filled=True, size=70).transform_filter(  # ty: ignore[unresolved-attribute]
        "datum.is_step"
    )
    return (
        alt.layer(lines, points)  # ty: ignore[invalid-argument-type]
        .properties(width=250, height=110)
        .facet(facet=alt.Facet("site:N", title=None, sort=SITE_ORDER), columns=2)
        .resolve_scale(y="independent")
        .properties(
            title=[
                "Blue: capacity fitted with the step path at weight 3 on April to August rows.",
                "Orange dots: month-to-month steps above 0.5 MW (August-to-April: see table).",
            ]
        )
    )


def rate_chart(*, steps: pl.DataFrame, summary: dict) -> alt.LayerChart:
    """Figure 2: steps above 0.5 MW per year in the real series against experiment 207's rate."""
    rows = []
    for site in SITE_ORDER:
        years = summary[site]["months"] / 12.0
        count = steps.filter((pl.col("weight") == RULE_WEIGHT) & (pl.col("site") == site)).height
        rows.append({"site": site, "series": REAL_LABEL, "rate": count / years})
        if site in EXPERIMENT_207_FALSE_STEPS:
            rows.append(
                {"site": site, "series": NO_STEP_LABEL, "rate": EXPERIMENT_207_FALSE_STEPS[site]}
            )
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90, opacity=1.0)
        .encode(
            y=alt.Y("site:N", sort=SITE_ORDER, title=None),
            yOffset=alt.YOffset("series:N", sort=[REAL_LABEL, NO_STEP_LABEL]),
            x=alt.X("rate:Q", title="Steps above 0.5 MW per year (fewer is quieter)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[REAL_LABEL, NO_STEP_LABEL],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
    )
    return chart.properties(
        width=WIDTH,
        height=170,
        title=[
            "Weight 3. Real series: July 2021 to June 2026 (D9: April 2024 onward).",
            "D9 has no experiment 207 value.",
        ],
    )


def window_chart(*, summary: dict) -> alt.LayerChart:
    """Figure 3: the months the fit trains on against the end of the feed-in tariff register."""
    full = date(2021, 7, 1)
    late = date(2024, 4, 1)
    rows = [
        {
            "site": site,
            "start": full if summary[site]["months"] == FULL_MONTHS else late,
            "end": date(2026, 6, 30),
        }
        for site in SITE_ORDER
    ]
    bars = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE, height=14)
        .encode(
            y=alt.Y("site:N", sort=SITE_ORDER, title=None),
            x=alt.X(
                "start:T",
                title="Date",
                axis=alt.Axis(format="%b %Y"),
                scale=alt.Scale(domain=["2019-09-01", "2026-07-01"]),
            ),
            x2="end:T",
        )
    )
    rule_data = pl.DataFrame({"date": [REGISTER_END], "t": ["Register ends 2021-03-24"]})
    rule = (
        alt.Chart(rule_data)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color=ocf.BRAND_ORANGE, size=2)
        .encode(x="date:T")
    )
    label = (
        alt.Chart(rule_data)  # ty: ignore[unresolved-attribute]
        .mark_text(align="left", dx=4, dy=-4, baseline="bottom", fontSize=10, color=GREY)
        .encode(x="date:T", y=alt.value(0), text="t:N")
    )
    return (bars + rule + label).properties(
        width=WIDTH,
        height=170,
        title=[
            "Blue: months of data the fit trains on at each Primary. Orange line: the last",
            "commissioning date in the feed-in tariff register. The two never overlap.",
        ],
    )


def main() -> None:
    """Write the figures."""
    paths = pl.read_parquet(common.WORK_DIR / "paths.parquet")
    steps = pl.read_parquet(common.WORK_DIR / "steps.parquet")
    summary = json.loads((common.WORK_DIR / "summary.json").read_text())
    save(chart=paths_chart(paths=paths, steps=steps), name="figure_01_paths")
    save(chart=rate_chart(steps=steps, summary=summary), name="figure_02_step_rate")
    save(chart=window_chart(summary=summary), name="figure_03_register_window")


if __name__ == "__main__":
    main()
