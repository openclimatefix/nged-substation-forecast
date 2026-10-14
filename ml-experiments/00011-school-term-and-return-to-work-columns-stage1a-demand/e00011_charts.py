"""Draw experiment 11's figures from the tables `e00011_report.py` wrote.

cd ml-experiments/00011-school-term-and-return-to-work-columns-stage1a-demand
uv run python e00011_charts.py
"""

import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import e00011_report as report
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

OUT_DIR: Final[Path] = (
    Path(__file__).parents[2]
    / "docs"
    / "ml-experiments"
    / "00011-school-term-and-return-to-work-columns-stage1a-demand"
)
CHARACTERS_PER_LINE: Final[int] = 105
ARM_LABELS: Final[dict[str, str]] = {
    "both": "Return and term columns",
    "term": "Term columns only",
    "return": "Return column only",
    "one_harmonic": "One harmonic of time of year",
}
ARM_COLOURS: Final[dict[str, str]] = {
    "both": ocf.BRAND_ORANGE,
    "term": ocf.DATA_BLUE,
    "return": ocf.DATA_SKY,
    "one_harmonic": ocf.DATA_PURPLE,
}
WINDOWS: Final[tuple[str, ...]] = ("Return", "Easter", "Half-term holidays")
WINDOW_LABELS: Final[dict[str, str]] = {
    "Return": "2 to 20 January",
    "Easter": "Easter holiday and first week of term 5",
    "Half-term holidays": "Autumn, February, and May half terms",
}


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, width=80),
        subtitle=textwrap.wrap(subtitle, width=CHARACTERS_PER_LINE),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def figure_1(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the Primary change in error by window and column set."""
    rows = []
    for window in WINDOWS:
        for arm, label in ARM_LABELS.items():
            if arm == "both":
                query = pl.col("label") == window
            else:
                query = pl.col("label") == f"{arm} vs base, {window}"
            row = table.filter(pl.col("type") == "Primary", query).row(0, named=True)
            rows.append(
                {
                    "window": WINDOW_LABELS[window],
                    "arm": label,
                    "change": row["change"],
                    "lower": row["lower"],
                    "upper": row["upper"],
                }
            )
    data = pl.DataFrame(rows)
    base = alt.Chart(data).encode(
        y=alt.Y(
            "window:N",
            sort=[WINDOW_LABELS[w] for w in WINDOWS],
            title=None,
            axis=alt.Axis(labelLimit=260),
        ),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_LABELS.values())),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_LABELS.values()), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", columns=2, labelLimit=300),
        ),
    )
    rules = base.mark_rule(strokeWidth=1.5).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "lower:Q",
            title=(
                "Change in error against the base arm (points of capacity; more negative is better)"
            ),
        ),
        x2="upper:Q",
    )
    dots = base.mark_circle(size=60).encode(x="change:Q")  # ty: ignore[unresolved-attribute]
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0], "t": ["Same as base arm"]}))
        .mark_rule(color="#222222")
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(rules, dots, zero).properties(
        width=CONTENT_WIDTH_PX - 200,
        height=250,
        title=_title(
            text="No column set is shown to lower the Primary error in the three windows",
            subtitle=(
                "Mean absolute error of the median, 13 or 14 Primary substations in Lincolnshire, "
                "three validation years (2023-24 to 2025-26). Dot: estimate. Line: 99.2% "
                "interval resampling 7-day blocks of days. The vertical line is the base arm "
                "(weather block, hour, weekday, and Christmas column)."
            ),
        ),
    )


def figure_2(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the per-year change of the Both arm against the base arm, Primary substations."""
    names = {
        "select_2023": "2023-24",
        "select_2024": "2024-25",
        "confirm_2025": "2025-26",
    }
    rows = []
    for window in WINDOWS:
        row = table.filter(pl.col("type") == "Primary", pl.col("label") == window).row(
            0, named=True
        )
        rows += [
            {"window": WINDOW_LABELS[window], "year": label, "change": row[f"change_{fold}"]}
            for fold, label in names.items()
        ]
    data = pl.DataFrame(rows)
    colours = [ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE]
    dots = (
        alt.Chart(data)
        .mark_circle(size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y(
                "window:N",
                sort=[WINDOW_LABELS[w] for w in WINDOWS],
                title=None,
                axis=alt.Axis(labelLimit=260),
            ),
            x=alt.X(
                "change:Q",
                title="Change in mean absolute error against the base arm (points of capacity)",
            ),
            color=alt.Color(
                "year:N",
                scale=alt.Scale(domain=list(names.values()), range=colours),
                legend=alt.Legend(title="Validation year", orient="top"),
            ),
        )
    )
    zero = (
        alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#222222").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(zero, dots).properties(
        width=CONTENT_WIDTH_PX - 200,
        height=170,
        title=_title(
            text="The sign of the change differs between years in the Return and Easter windows",
            subtitle=(
                "Return and term columns against the base arm, Primary substations. More "
                "negative is better; the vertical line is no change."
            ),
        ),
    )


def figure_3(*, daily: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the Primary daily error around Christmas and New Year, averaged over the years."""
    frame = (
        daily.filter(pl.col("substation_type") == "Primary")
        .with_columns(
            christmas=pl.struct("local_date", "fold").map_elements(
                lambda s: (
                    (
                        s["local_date"]
                        - date(
                            s["local_date"].year - (1 if s["local_date"].month < 7 else 0), 12, 25
                        )
                    ).days
                ),
                return_dtype=pl.Int64,
            )
        )
        .filter(pl.col("christmas").is_between(-10, 32))
        .group_by("christmas")
        .agg(pl.col("base", "both", "one_harmonic").mean())
        .sort("christmas")
    )
    long = frame.unpivot(index="christmas", variable_name="arm", value_name="error").with_columns(
        arm=pl.col("arm").replace_strict(
            {
                "base": "Base arm",
                "both": "Return and term columns",
                "one_harmonic": "One harmonic of time of year",
            }
        )
    )
    colours = {
        "Base arm": "#666666",
        "Return and term columns": ocf.BRAND_ORANGE,
        "One harmonic of time of year": ocf.DATA_PURPLE,
    }
    return (
        alt.Chart(long)
        .mark_line(strokeWidth=1.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "christmas:Q",
                title="Days from 25 December (0 is Christmas Day, 7 is the first week of January)",
                axis=alt.Axis(values=[-7, 0, 7, 14, 21, 28]),
            ),
            y=alt.Y(
                "error:Q",
                title="Mean absolute error (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", columns=3, labelLimit=260),
            ),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 60,
            height=230,
            title=_title(
                text="The new columns do not visibly change the error after Christmas",
                subtitle=(
                    "Mean absolute error of the median over the Primary substations, averaged "
                    "over the three validation years, for each day from 10 days before to 32 "
                    "days after 25 December."
                ),
            ),
        )
    )


def main() -> None:
    """Draw and save the figures as SVG and PNG."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pl.read_parquet(report.RESULTS_DIR / "window_contrasts.parquet")
    daily = pl.read_parquet(report.RESULTS_DIR / "daily_errors.parquet")
    for number, chart in (
        (1, figure_1(table=table)),
        (2, figure_2(table=table)),
        (3, figure_3(daily=daily)),
    ):
        chart.save(OUT_DIR / f"figure_{number}.svg")
        chart.save(report.RESULTS_DIR / f"figure_{number}.png", scale_factor=2)


if __name__ == "__main__":
    main()
