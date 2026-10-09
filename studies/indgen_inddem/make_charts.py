"""Draw the INDDEM and INDGEN page's figures as SVG under `docs/studies/assets/`.

Run after `build_tables.py`: `uv run python studies/indgen_inddem/make_charts.py`. Every number a
title quotes is computed from the tables that `build_tables.py` wrote, so a figure cannot disagree
with `report.md`. All of the data is public, so the charts name the GSP groups and show megawatts
on calendar dates.
"""

import subprocess
from datetime import UTC, datetime, timedelta
from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from indgen_inddem_common import (
    ASSETS_DIR,
    STUDY_DIR,
    ZONES,
)
from studies.charts import CONTENT_WIDTH_PX, figure

WEEKS: Final[dict[str, datetime]] = {
    "winter": datetime(2025, 12, 15, tzinfo=UTC),
    "summer": datetime(2026, 6, 15, tzinfo=UTC),
}
"""The Monday that starts the week containing each solstice, fixed by the calendar and not chosen by
how the data looks."""
SERIES_COLOURS: Final[dict[str, str]] = {
    "INDDEM (demand, sign reversed)": ocf.BRAND_ORANGE,
    "INDGEN (generation)": ocf.DATA_BLUE,
    "National demand outturn (INDO)": ocf.DATA_BLUE,
    "AGV, summed over the 14 groups": ocf.BRAND_ORANGE,
}
ROW_HEIGHT_PX: Final[int] = 130
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 100
LONDON: Final[str] = "Europe/London"


def colour_scale(*, names: list[str]) -> alt.Scale:
    """Return a colour scale that gives each name in `names` its colour from `SERIES_COLOURS`."""
    return alt.Scale(domain=names, range=[SERIES_COLOURS[name] for name in names])


def week_slice(*, frame: pl.DataFrame, season: str) -> pl.DataFrame:
    """Keep the rows in the seven UTC days that start on the season's fixed Monday."""
    start = WEEKS[season]
    return frame.filter(pl.col("time").is_between(start, start + timedelta(days=7), closed="left"))


def line_panel(
    *,
    data: pl.DataFrame,
    names: list[str],
    title: str,
    y_title: str,
    last: bool,
    x_title: str = "Day (UTC)",
) -> alt.Chart:
    """Draw one row of lines with a UTC time axis, coloured by series."""
    return (
        alt.Chart(data, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
        .mark_line(strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T",
                axis=alt.Axis(
                    format="%a %-d %b",
                    tickCount="day",
                    labels=last,
                    ticks=last,
                    title=x_title if last else None,
                ),
            ),
            y=alt.Y("megawatts:Q", axis=alt.Axis(tickCount=4, title=y_title)),
            color=alt.Color("series:N", scale=colour_scale(names=names), legend=None),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX)
    )


def unit_check_figure(*, number: int) -> alt.VConcatChart:
    """Draw AGV beside the initial national demand outturn, as a week and as a scatter."""
    check = pl.read_parquet(STUDY_DIR / "agv_against_indo.parquet")
    ratio = check["ratio"].median()
    correlation = check.select(pl.corr("agv_mw", "indo_mw")).item()
    names = ["National demand outturn (INDO)", "AGV, summed over the 14 groups"]
    long = pl.concat(
        [
            check.select("time", megawatts="indo_mw", series=pl.lit(names[0])),
            check.select("time", megawatts="agv_mw", series=pl.lit(names[1])),
        ]
    )
    week = line_panel(
        data=week_slice(frame=long, season="winter"),
        names=names,
        title="One winter week",
        y_title="MW",
        last=True,
    )
    scatter = (
        alt.Chart(
            check.gather_every(7).select(indo="indo_mw", agv="agv_mw"),
            title=alt.TitleParams(
                "Every seventh half-hour of 2025-09-01 to 2026-09-20",
                anchor="start",
                fontSize=11,
                offset=2,
            ),
        )
        .mark_circle(size=6, opacity=0.35, color=ocf.DATA_BLUE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("indo:Q", axis=alt.Axis(title="National demand outturn (MW)")),
            y=alt.Y("agv:Q", axis=alt.Axis(title="AGV summed over the 14 groups, in MW")),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * 2)
    )
    line = (
        alt.Chart(pl.DataFrame({"x": [10_000.0, 45_000.0], "y": [10_000.0, 45_000.0]}))
        .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
        .encode(x="x:Q", y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return figure(
        panels=[week, cast(alt.LayerChart, alt.layer(scatter, line))],
        number=number,
        title=(
            f"AGV, doubled, is {ratio:.0%} of the national demand outturn, so AGV is in "
            "megawatt-hours per half-hour"
        ),
        subtitle=[
            (
                "Half-hourly values for the 14 GSP groups of Great Britain. Orange: AGV, the "
                "energy the groups take from the transmission system, times 2 to give megawatts. "
                "Blue: the national demand outturn (INDO). In the scatter, the dashed line is "
                "equality."
            ),
            f"The correlation of the {len(check)} half-hours is {correlation:.3f}.",
        ],
        figure_planning=None,
    )


def zone_sign_figure(*, number: int) -> alt.VConcatChart:
    """Draw, for each zone, the share of half-hours in which the derived zone has the wrong sign."""
    signs = pl.read_parquet(STUDY_DIR / "zone_signs.parquet").filter(pl.col("view") == "latest")
    overall = signs["wrong_sign"].sum() / signs["half_hours"].sum()
    names = {"inddem": "INDDEM (demand, sign reversed)", "indgen": "INDGEN (generation)"}
    data = signs.with_columns(
        series=pl.col("dataset").replace_strict(names), count=pl.col("wrong_sign")
    )
    chart = (
        alt.Chart(data)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("zone:N", sort=list(ZONES), axis=alt.Axis(title="Study zone", labelAngle=0)),
            xOffset="series:N",
            y=alt.Y(
                "count:Q",
                axis=alt.Axis(title="Half-hours with the wrong sign (of 18,960)"),
            ),
            color=alt.Color(
                "series:N",
                scale=colour_scale(names=list(names.values())),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * 1.5)
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            f"The zones recovered from the boundaries have the expected sign in "
            f"{1 - overall:.2%} of zone half-hours"
        ),
        subtitle=[
            (
                "Each zone is a combination of the national total and the 17 boundaries, by the "
                "formulas of Elexon CVA Change Circular 235. Demand zones should be zero or "
                "negative and generation zones zero or positive, within 1 MW of rounding."
            ),
            "The latest issue before each half-hour, 2025-09-01 to 2026-09-30.",
        ],
        figure_planning=None,
    )


def save(*, chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG under `ASSETS_DIR` and optimise it with `svgo`."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / f"{name}.svg"
    chart.save(path)
    subprocess.run(
        ["npx", "svgo@4", "--multipass", "--precision=1", "--final-newline", str(path)],
        check=True,
        capture_output=True,
    )
    print(f"Wrote {path}")


def main() -> None:
    """Write every chart."""
    charts = {
        "indgen_inddem_unit_check": unit_check_figure(number=1),
        "indgen_inddem_zone_signs": zone_sign_figure(number=2),
    }
    for name, chart in charts.items():
        save(chart=chart, name=name)


if __name__ == "__main__":
    main()
