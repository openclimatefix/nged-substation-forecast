"""Draw the figures of experiment 172's write-up from the stored day tables.

uv run --no-sync python ml-experiments/00172-*/e00172_charts.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00172_common as common
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
WRAP: Final[int] = 100
GREY: Final[str] = "#808080"
SITE_COLOURS: Final[dict[str, str]] = {
    "D5": ocf.DATA_BLUE,
    "D10": ocf.DATA_SKY,
    "D11": ocf.DATA_PURPLE,
    "D14": ocf.DATA_GREEN,
    "D16": ocf.BRAND_ORANGE,
}
LABEL_NAMES: Final[dict[str, str]] = {
    "switching": "Switching event",
    "flagged": "Flagged reading",
    "cold": "Cold snap",
    "economy7": "Economy 7 step",
    "holiday": "Holiday",
    "labelled": "Any label",
}


def lines(*, text: str) -> list[str]:
    """Wrap text to the plot width."""
    return textwrap.wrap(text, WRAP)


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG into the docs folder and PNG into the private folder."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def curve_chart() -> alt.TopLevelMixin:
    """Return the cumulative share of excess loss against the fraction of days, worst first."""
    frames = [
        pl.read_parquet(common.WORK_DIR / f"curve_{site}.parquet").with_columns(site=pl.lit(site))
        for site in common.DECISION_SITES
    ]
    pooled = pl.read_parquet(common.WORK_DIR / "curve_pooled.parquet")
    x = alt.X(
        "fraction_of_days:Q",
        title="Fraction of days, worst first (each substation's days ranked on its own)",
        axis=alt.Axis(format="%", values=[0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]),
        scale=alt.Scale(domain=[0, 1]),
    )
    y = alt.Y(
        "share_of_excess:Q",
        title="Cumulative share of excess loss",
        axis=alt.Axis(format="%"),
        scale=alt.Scale(domain=[0, 1]),
    )
    sites = (
        alt.Chart(pl.concat(frames))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2, opacity=0.8)
        .encode(
            x=x,
            y=y,
            color=alt.Color(
                "site:N",
                scale=alt.Scale(domain=list(SITE_COLOURS), range=list(SITE_COLOURS.values())),
                legend=alt.Legend(title="Substation", orient="top"),
            ),
        )
    )
    flat = (
        alt.Chart(pl.DataFrame({"fraction_of_days": [0.0, 1.0], "share_of_excess": [0.0, 1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeDash=[4, 4], color=GREY)
        .encode(x=x, y=y)
    )
    mark = alt.Chart(pl.DataFrame({"f": [0.05]})).mark_rule(color=GREY).encode(x="f:Q")  # ty: ignore[unresolved-attribute]
    headline = alt.Chart(pooled).mark_line(strokeWidth=3, color="#333333").encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
    return (flat + mark + sites + headline).properties(
        width=WIDTH,
        height=300,
        title=alt.TitleParams(
            "The worst 5% of days carry more than 40% of the excess loss",
            subtitle=lines(
                text="Excess loss is a day's quantile score (points of effective capacity, D+1 "
                "rows, 2025-07-02 to 2026-06-29) minus the substation's median day. Thick black "
                "line: all five substations pooled. Dashed line: loss spread evenly over days. "
                "Vertical rule: the worst 5%. Higher and further left is more concentrated."
            ),
        ),
    )


def label_chart() -> alt.TopLevelMixin:
    """Return the share of the worst days and of all days that carry each label."""
    table = pl.read_parquet(common.WORK_DIR / "label_table.parquet")
    long = pl.concat(
        [
            table.select(
                "label", share=pl.col("share_of_worst_days"), group=pl.lit("Worst 5% of days")
            ),
            table.select("label", share=pl.col("share_of_all_days"), group=pl.lit("All days")),
        ]
    ).with_columns(label=pl.col("label").replace(LABEL_NAMES))
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("label:N", title=None, sort=list(LABEL_NAMES.values())),
            x=alt.X("share:Q", title="Share of days carrying the label", axis=alt.Axis(format="%")),
            yOffset="group:N",
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["Worst 5% of days", "All days"], range=[ocf.BRAND_ORANGE, GREY]
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=alt.TitleParams(
                "Switching events label the most worst days; half the worst days have no label",
                subtitle=lines(
                    text="Five substations, 72 worst days of 1,392 substation-days. A day can carry several "
                    "labels, so the bars do not sum to the 'Any label' bar."
                ),
            ),
        )
    )


def example_days() -> list[dict]:
    """Return one worst day per label class (switching, holiday, cold) and one with no label."""
    worst = pl.read_parquet(common.WORK_DIR / "worst_days.parquet").filter(
        pl.col("site").is_in(common.DECISION_SITES)
    )
    picks = []
    for name, condition in (
        ("switching event", pl.col("switching")),
        ("holiday", pl.col("holiday") & ~pl.col("switching")),
        ("cold snap", pl.col("cold") & ~pl.col("switching") & ~pl.col("holiday")),
        ("no label", ~pl.col("labelled")),
    ):
        row = (
            worst.filter(condition, pl.col("n_rows") == 48)
            .sort("excess", descending=True)
            .row(0, named=True)
        )
        picks.append({"name": name, "site": row["site"], "date": row["local_date"]})
    return picks


def example_chart() -> alt.TopLevelMixin:
    """Return forecast against actual for four typical worst days, one panel each."""
    rows = common.read_rows(lead_band=common.D1_LEAD_HOURS, drop_masked=True)
    panels = []
    for pick in example_days():
        day = rows.filter(
            pl.col("site") == pick["site"], pl.col("local_date") == pick["date"]
        ).with_columns(
            hour=(
                pl.col("valid_time").dt.convert_time_zone(common.LOCAL_TZ).dt.hour()
                + pl.col("valid_time").dt.convert_time_zone(common.LOCAL_TZ).dt.minute() / 60
            )
        )
        x = alt.X(
            "hour:Q",
            title="Local hour of day",
            axis=alt.Axis(values=[0, 6, 12, 18, 24]),
            scale=alt.Scale(domain=[0, 24]),
        )
        y = alt.Y("q50:Q", title="Power (points of capacity)", scale=alt.Scale(zero=False))
        band = (
            alt.Chart(day)  # ty: ignore[unresolved-attribute]
            .mark_area(opacity=0.25, color=ocf.DATA_BLUE)
            .encode(x=x, y="q05:Q", y2="q95:Q")
        )
        forecast = alt.Chart(day).mark_line(color=ocf.DATA_BLUE).encode(x=x, y=y)  # ty: ignore[unresolved-attribute]
        actual = (
            alt.Chart(day)  # ty: ignore[unresolved-attribute]
            .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=2)
            .encode(x=x, y="actual:Q")
        )
        title = f"{pick['site']}, {pick['date']}: {pick['name']}"
        panels.append((band + forecast + actual).properties(width=WIDTH, height=130, title=title))
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            "Four typical worst days: the actual power sits far from the median forecast",
            subtitle=lines(
                text="Orange: actual power. Blue line: median forecast. Blue band: 5% to 95% "
                "quantiles. Forecasts made 15 to 39 hours ahead. One worst day per label class "
                "(the highest-loss day of each class among days with all 48 half-hours scored)."
            ),
        )
    )


def main() -> None:
    """Write the three figures."""
    save(chart=curve_chart(), name="fig1_cumulative_share")
    save(chart=label_chart(), name="fig2_labels")
    save(chart=example_chart(), name="fig3_worst_days")


if __name__ == "__main__":
    main()
