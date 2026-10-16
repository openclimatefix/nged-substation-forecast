"""Draw the capacity study's figures from the tables the two build scripts wrote.

    uv run python studies/embedded_pv_capacity/capacity_charts.py [--text-only]

Run `build_capacity_table.py` and `build_raw_gaps.py` first. Writes one SVG per figure under
`docs/studies/assets/` and the text of every figure to `figure_text.md` in the study folder, for
the text review that comes before the first render. The Primaries are substations, so each chart
may name them. The solar farms are not drawn.
"""

import json
import sys
from pathlib import Path
from typing import Any, Final

import altair as alt
import embedded_pv_inputs as inputs
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX, figure

ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
BOUNDARY_PATH: Final[Path] = Path(
    "packages/geo/src/geo/great_britain/england_scotland_wales.geojson"
)
STATUS_COLOURS: Final[dict[str, str]] = {
    "clean": ocf.DATA_BLUE,
    "low-power": ocf.DATA_PURPLE,
    "flagged": ocf.BRAND_ORANGE,
}
STATUS_SCALE: Final[alt.Scale] = alt.Scale(
    domain=list(STATUS_COLOURS), range=list(STATUS_COLOURS.values())
)
SKY_COLOURS: Final[dict[str, str]] = {"clear": ocf.DATA_SKY, "overcast": ocf.BLACK_1}
HOUR_AXIS: Final[alt.Axis] = alt.Axis(values=[6, 12, 18], labelFontSize=10, titleFontSize=10)
HOUR_DOMAIN: Final[list[int]] = [4, 20]
PANEL_WIDTH_PX: Final[int] = 172
SINGLE_WIDTH_PX: Final[int] = 625
PAIR_WIDTH_PX: Final[int] = 280
PANEL_HEIGHT_PX: Final[int] = 150
FACET_PRIMARIES: Final[tuple[str, ...]] = ("D10", "D11", "D14", "D16", "D18", "D9")
"""The six Primaries with a capacity: clean or low-power in the latest season-year."""
LATEST: Final[int] = inputs.LAST_SEASON_YEAR
MAP_SCALE: Final[int] = 26_000
MAP_HEIGHT_PX: Final[int] = 480
MAP_CENTRE: Final[tuple[float, float]] = (0.04, 53.12)


def season_label(year: int) -> str:
    """Return the label of a season-year, such as `2025/26`."""
    return f"{year}/{(year + 1) % 100:02d}"


def name_of(site: str) -> str:
    """Return the Primary's name."""
    return inputs.PRIMARY_NAMES[site]


def facet_header(title: str, *, on_top: bool = False) -> alt.Header:
    """Return the header style of a facet panel; `on_top` puts a row header above its panel."""
    if on_top:
        return alt.Header(
            title=title,
            labelFontSize=11,
            labelAnchor="start",
            labelPadding=4,
            labelOrient="top",
            labelAngle=0,
        )
    return alt.Header(title=title, labelFontSize=11, labelAnchor="start", labelPadding=4)


def headline_figure(*, table: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw each Primary's capacity in 2025/26 with its interval and its undetected-load bound."""
    latest = table.filter((pl.col("season_year") == LATEST) & (pl.col("status") != "left-out"))
    order = (
        latest.with_columns(sort_key=pl.col("capacity_mw").fill_null(0.0))
        .sort("sort_key", descending=True)["primary"]
        .to_list()
    )
    has_capacity = latest.filter(pl.col("status") != "flagged").with_columns(
        low_bound=pl.col("capacity_mw") / (1 + pl.col("bound_dayL_relative"))
    )
    flagged = latest.filter(pl.col("status") == "flagged")
    y = alt.Y("primary:N", sort=order, title=None, axis=alt.Axis(labelFontSize=11))
    x_title = "Capacity C in 2025/26 (MW of 99th-percentile AC output)"
    bound = (
        alt.Chart(has_capacity)
        .mark_bar(height=16, opacity=0.25, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x=alt.X("low_bound:Q", title=x_title, scale=alt.Scale(domain=[0, 6.5])),
            x2="capacity_mw:Q",
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=list(STATUS_COLOURS), range=list(STATUS_COLOURS.values())),
                legend=alt.Legend(title="Status in 2025/26"),
            ),
        )
    )
    interval = (
        alt.Chart(has_capacity)
        .mark_rule(strokeWidth=2.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x="capacity_low_mw:Q",
            x2="capacity_high_mw:Q",
            color=alt.Color("status:N", scale=STATUS_SCALE, legend=None),
        )
    )
    dot = (
        alt.Chart(has_capacity)
        .mark_point(size=90, filled=True, opacity=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x="capacity_mw:Q",
            color=alt.Color("status:N", scale=STATUS_SCALE, legend=None),
            fill=alt.condition(
                alt.datum.status == "low-power",
                alt.value("white"),
                alt.Color("status:N", scale=STATUS_SCALE, legend=None),
            ),
            stroke=alt.Color("status:N", scale=STATUS_SCALE, legend=None),
        )
    )
    library = (
        alt.Chart(flagged)
        .mark_rule(strokeWidth=7, opacity=0.35, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x="library_low_mw:Q",
            x2="library_high_mw:Q",
            color=alt.Color("status:N", scale=STATUS_SCALE, legend=None),
        )
    )
    library_text = (
        alt.Chart(flagged)
        .mark_text(align="left", dx=6, fontSize=10, color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            y=y,
            x="library_high_mw:Q",
            text=alt.value("range over seven load shapes; no capacity"),
        )
    )
    panel = alt.layer(bound, library, interval, dot, library_text).properties(
        width=CONTENT_WIDTH_PX - 150, height=26 * len(order)
    )
    return figure(
        panels=[panel],
        number=number,
        title="Each of the six Primary substations we can size has 2.5 to 4.5 MW of embedded solar",
        subtitle=[
            (
                "Capacity C is the 99th percentile of the half-hourly AC output of the embedded "
                "solar, in MW, over the season-year 2025/26 (September 2025 to August 2026). A "
                "Primary substation is a substation that steps power down to the local distribution"
                " network."
            ),
            (
                "Dot: C. Dark line: 95% interval from 200 resamples of the year's days, drawn in "
                "blocks of seven consecutive days. Pale bar: C divided by one plus the typical "
                "error (median of the undetected resamples) when a day-long load of a quarter of "
                "the true capacity (half, for a hollow dot) goes undetected. Larger loads would "
                "pull the true capacity lower."
            ),
            (
                "Hollow purple dot: the test flags a day-long load of half the capacity in under "
                "80% of resamples, so a large such load could go unnoticed. Orange bar (Kirton): "
                "the test found a demand that rises on cloudy days, so no capacity is given; the "
                "bar is the range of capacities across seven assumed demand shapes."
            ),
        ],
        figure_planning=None,
    )


def profile_figure(*, profile: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw summer demand by half-hour on clear and overcast days, first year against 2025/26."""
    summer = profile.filter(
        (pl.col("season") == "summer") & pl.col("site").is_in(list(FACET_PRIMARIES))
    )
    first = summer.group_by("site").agg(first_year=pl.col("season_year").min())
    labels = {
        row["site"]: f"{name_of(row['site'])} (from {season_label(int(row['first_year']))})"
        for row in first.iter_rows(named=True)
    }
    rows = (
        summer.join(first, on="site")
        .filter((pl.col("season_year") == pl.col("first_year")) | (pl.col("season_year") == LATEST))
        .with_columns(
            period=pl.when(pl.col("season_year") == LATEST)
            .then(pl.lit(season_label(LATEST)))
            .otherwise(pl.lit("first season-year")),
            hour=pl.col("half_hour") / 2,
            panel=pl.col("site").replace_strict(labels),
            sky=pl.col("sky_class"),
        )
    )
    order = [labels[s] for s in FACET_PRIMARIES]
    chart = (
        alt.Chart(rows.select("panel", "hour", "mean_mw", "sky", "period"))
        .mark_line(strokeWidth=1.8, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=HOUR_DOMAIN),
                axis=HOUR_AXIS,
                title="Hour of day, half-hour start (local)",
            ),
            y=alt.Y("mean_mw:Q", title="Mean power drawn (MW)"),
            color=alt.Color(
                "sky:N",
                scale=alt.Scale(domain=list(SKY_COLOURS), range=list(SKY_COLOURS.values())),
                legend=alt.Legend(title="Day type"),
            ),
            strokeDash=alt.StrokeDash(
                "period:N",
                scale=alt.Scale(
                    domain=["first season-year", season_label(LATEST)],
                    range=[[5, 3], [1, 0]],
                ),
                legend=alt.Legend(title="Season-year", labelLimit=300),
            ),
        )
        .properties(width=PANEL_WIDTH_PX, height=PANEL_HEIGHT_PX)
        .facet(facet=alt.Facet("panel:N", sort=order, header=facet_header("")), columns=3)
        .resolve_scale(y="independent")
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            "Net summer power drawn by time of day, on clear and overcast days, at six Primary "
            "substations"
        ),
        subtitle=[
            (
                "Mean half-hourly power drawn from the electricity network (MW), after embedded "
                "solar has supplied part of the local demand, April to September."
            ),
            (
                "Clear day: daily surface sunlight at least 0.8 times its clear-sky value, from "
                "CAMS (the Copernicus Atmosphere Monitoring Service). Overcast day: at most 0.3 "
                "times. Dashed line: the first season-year of each Primary substation's record. "
                "Solid line: 2025/26."
            ),
            (
                "On clear days the midday dip is deeper, partly because embedded solar supplies "
                "local demand. Each panel has its own y axis."
            ),
        ],
        figure_planning=None,
    )


def gap_figure(*, gaps: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the summer midday mean power on clear and overcast days for each season-year."""
    rows = gaps.filter(pl.col("site").is_in(list(FACET_PRIMARIES))).with_columns(
        label=pl.col("season_year").map_elements(season_label, return_dtype=pl.String),
        panel=pl.col("site").replace_strict({s: name_of(s) for s in FACET_PRIMARIES}),
        low=pl.col("mean_mw") - pl.col("ci_half_mw"),
        high=pl.col("mean_mw") + pl.col("ci_half_mw"),
    )
    base = alt.Chart(rows).encode(
        x=alt.X(
            "label:O",
            title="Season-year",
            axis=alt.Axis(labelAngle=-90, labelFontSize=9, titleFontSize=10),
        ),
        color=alt.Color(
            "sky_class:N",
            scale=alt.Scale(domain=list(SKY_COLOURS), range=list(SKY_COLOURS.values())),
            legend=alt.Legend(title="Day type"),
        ),
    )
    layers = alt.layer(
        base.mark_rule(strokeWidth=1.2, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low:Q", title="Midday mean power drawn (MW)"), y2="high:Q"
        ),
        base.mark_line(strokeWidth=1.5, aria=False).encode(y="mean_mw:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=26, aria=False).encode(y="mean_mw:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=PANEL_WIDTH_PX, height=PANEL_HEIGHT_PX)
    chart = layers.facet(
        facet=alt.Facet(
            "panel:N", sort=[name_of(s) for s in FACET_PRIMARIES], header=facet_header("")
        ),
        columns=3,
    ).resolve_scale(y="independent")
    return figure(
        panels=[chart],
        number=number,
        title=(
            "At midday in summer 2025/26, power drawn is 1.9 to 3.0 MW lower on clear days than"
            " on overcast days"
        ),
        subtitle=[
            (
                "Mean power drawn from the electricity network (MW) from 11:00 to 14:00 local time "
                "on summer days (April to September), per season-year, at six Primary substations. "
                "No weather or statistical model is involved."
            ),
            (
                "Dot: mean over days. Vertical line: 95% interval for the mean (t distribution "
                "across days, which are treated as independent). Overcast days are few: 4 to 15 "
                "per summer. Summer here is September 2025 and April to August 2026 for 2025/26."
            ),
            (
                "Overcast days also differ from clear days in temperature and demand, so the gap is"
                " more than the solar output alone. Each panel has its own y axis."
            ),
        ],
        figure_planning=None,
    )


def kirton_figure(*, pooled: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw overcast minus clear demand by half-hour at Kirton and Donington, winter and summer."""
    rows = pooled.with_columns(
        hour=pl.col("half_hour") / 2,
        primary=pl.col("site").replace_strict(
            {inputs.KIRTON: name_of(inputs.KIRTON), inputs.DONINGTON: name_of(inputs.DONINGTON)}
        ),
        panel=pl.col("season").replace_strict(
            {"summer": "Summer, April to September", "winter": "Winter: November and February days"}
        ),
    )
    colours = {name_of(inputs.KIRTON): ocf.BRAND_ORANGE, name_of(inputs.DONINGTON): ocf.DATA_BLUE}
    chart = (
        alt.Chart(rows.select("panel", "hour", "gap_mw", "primary"))
        .mark_line(strokeWidth=2, clip=True, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=HOUR_DOMAIN),
                axis=HOUR_AXIS,
                title="Hour of day, half-hour start (local)",
            ),
            y=alt.Y("gap_mw:Q", title="Overcast minus clear (MW)"),
            color=alt.Color(
                "primary:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title="Primary"),
            ),
        )
        .properties(width=SINGLE_WIDTH_PX, height=PANEL_HEIGHT_PX)
        .facet(
            row=alt.Row(
                "panel:N",
                sort=["Summer, April to September", "Winter: November and February days"],
                header=facet_header("", on_top=True),
            )
        )
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            "At Kirton the extra power drawn on overcast days at midday is larger in November "
            "and February than in summer; at Donington it is the same size"
        ),
        subtitle=[
            (
                "Mean power drawn on overcast days minus clear days, by half-hour (MW), at the "
                "Kirton and Donington Primary substations. A positive value means more power drawn "
                "on overcast days."
            ),
            (
                "Season-years 2020/21 to 2025/26 pooled, each weighted by its number of days. "
                "Winter holds only November and February days, because the sun stays below 15 "
                "degrees in December and January: at Kirton 30 clear and 33 overcast days, at "
                "Donington 26 and 31. Summer holds 342 and 51 at Kirton, 319 and 52 at Donington. "
                "Solar output is far lower in winter, so if solar alone caused the gap it would be "
                "smaller in winter. A larger winter gap may be cold weather instead, so it is a "
                "difference and not a proof."
            ),
        ],
        figure_planning=None,
    )


def bias_figure(*, bias: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the noise-free bias of C and of the daily fit against the size of an undetected load."""
    names = {
        "dayL": "A load on dull days, the same all day",
        "sunload": "A load that rises with sunshine",
    }
    zero = pl.DataFrame(
        {
            "shape": [s for s in names for _ in range(2)],
            "estimator": ["C (six-arm)", "Daily fit"] * len(names),
            "amplitude": [0.0] * 2 * len(names),
            "median": [0.0] * 2 * len(names),
            "low": [0.0] * 2 * len(names),
            "high": [0.0] * 2 * len(names),
        }
    )
    long = pl.concat(
        [
            bias.select(
                "shape",
                "amplitude",
                estimator=pl.lit(name),
                value=pl.col(column),
            )
            for name, column in (("C (six-arm)", "bias_cross_arm"), ("Daily fit", "bias_daily_fit"))
        ]
    )
    summary = long.group_by("shape", "estimator", "amplitude").agg(
        median=pl.col("value").median(),
        low=pl.col("value").quantile(0.1),
        high=pl.col("value").quantile(0.9),
    )
    rows = pl.concat([zero, summary.select(zero.columns)]).with_columns(
        panel=pl.col("shape").replace_strict(names)
    )
    colours = {"C (six-arm)": ocf.DATA_BLUE, "Daily fit": ocf.DATA_PURPLE}
    base = alt.Chart(rows).encode(
        x=alt.X(
            "amplitude:Q",
            title="Undetected load (share of true capacity)",
            axis=alt.Axis(values=[0, 0.1, 0.25, 0.5], format=".2f"),
        ),
        color=alt.Color(
            "estimator:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title="Estimate of capacity"),
        ),
    )
    layers = alt.layer(
        base.mark_area(opacity=0.18, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low:Q", title="Estimate over truth minus 1"), y2="high:Q"
        ),
        base.mark_line(strokeWidth=2, aria=False).encode(y="median:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(rows).mark_rule(color=ocf.BLACK_1, strokeDash=[2, 2]).encode(y=alt.datum(0)),  # ty: ignore[unresolved-attribute]
    ).properties(width=PAIR_WIDTH_PX, height=PANEL_HEIGHT_PX)
    chart = layers.facet(
        column=alt.Column("panel:N", sort=list(names.values()), header=facet_header(""))
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            "An undetected demand load on dull days biases the daily-fit capacity about 2.6 times "
            "as much as the six-arm capacity C"
        ),
        subtitle=[
            (
                "Relative error of each capacity estimate, (estimate over true capacity) minus 1, "
                "when a demand the test cannot see is added to injected solar output. No "
                "measurement noise. The six-arm capacity C comes from six fits of the same series; "
                "the daily fit is a seventh, made on whole days."
            ),
            (
                "The load is sized by the capacity the first of the six fits would report for it, "
                "as a share of the true capacity. Line: median over 218 cases, each using the "
                "load's "
                "arm vector measured on one solar farm's rows at one Primary substation in one "
                "season-year, added to the Primary substation's typical solar output. Band: 10th "
                "to 90th percentile. Positive means too high. For a load that rises with "
                "sunshine, C is biased low by more than the daily fit."
            ),
        ],
        figure_planning=None,
    )


def crosscheck_figure(*, table: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the daily-fit capacity against the six-arm capacity, one dot per Primary-year."""
    rows = table.filter(pl.col("status") != "left-out").with_columns(
        label=pl.col("season_year").map_elements(season_label, return_dtype=pl.String)
    )
    top = float(rows.select(pl.max_horizontal("capacity_mw", "capacity_daily_fit_mw").max()).item())
    top *= 1.05
    diagonal = (
        alt.Chart(pl.DataFrame({"a": [0.0, top], "b": [0.0, top]}))
        .mark_line(color=ocf.BLACK_1, strokeDash=[3, 3], aria=False)
        .encode(x="a:Q", y="b:Q")  # ty: ignore[unresolved-attribute]
    )
    points = (
        alt.Chart(rows.select("primary", "label", "status", "capacity_mw", "capacity_daily_fit_mw"))
        .mark_point(filled=True, size=55, opacity=0.9, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "capacity_mw:Q", title="Six-arm capacity C (MW)", scale=alt.Scale(domain=[0, top])
            ),
            y=alt.Y(
                "capacity_daily_fit_mw:Q",
                title="Daily-fit capacity (MW)",
                scale=alt.Scale(domain=[0, top]),
            ),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=list(STATUS_COLOURS), range=list(STATUS_COLOURS.values())),
                legend=alt.Legend(title="Status"),
            ),
        )
    )
    panel = alt.layer(diagonal, points).properties(width=SINGLE_WIDTH_PX, height=300)
    return figure(
        panels=[panel],
        number=number,
        title=(
            "At Kirton the daily-fit capacity is above the six-arm capacity C in all 7 season-"
            "years; elsewhere it is mostly below"
        ),
        subtitle=[
            (
                "One dot per Primary substation and season-year. Dotted line: the two capacity "
                "estimates are equal."
            ),
            (
                "A demand that rises on cloudy days, and lasts all day, would lift the daily fit "
                "above the six-arm capacity C. All 7 Kirton season-years sit above the line."
            ),
        ],
        figure_planning=None,
    )


def weeks_figure(*, weeks: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw one week each of summer and winter, with the fitted solar output added back."""
    starts = {
        r["season"]: r["week_start"]
        for r in weeks.select("season", "week_start").unique().iter_rows(named=True)
    }
    panels_text = {
        "summer": f"Summer week from {starts['summer']:%d %B %Y}",
        "winter": f"Winter week from {starts['winter']:%d %B %Y}",
    }
    long = pl.concat(
        [
            weeks.select(
                "season",
                "day",
                "half_hour",
                series=pl.lit(name),
                mw=pl.col(column).cast(pl.Float64),
            )
            for name, column in (
                ("Metered flow", "power"),
                ("Fitted solar output", "solar_mw"),
                ("Metered flow plus fitted solar", "gross_mw"),
            )
        ]
    ).with_columns(
        x=(pl.col("day") - 1) + (pl.col("half_hour") + 0.5) / 48,
        panel=pl.col("season").replace_strict(panels_text),
    )
    colours = {
        "Metered flow": ocf.BLACK_1,
        "Fitted solar output": ocf.DATA_SKY,
        "Metered flow plus fitted solar": ocf.DATA_BLUE,
    }
    chart = (
        alt.Chart(long.select("panel", "x", "mw", "series"))
        .mark_line(strokeWidth=1.6, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[0, 7]),
                axis=alt.Axis(values=list(range(7)), labelExpr="datum.value + 1", labelFontSize=10),
                title="Day of the week (1 to 7)",
            ),
            y=alt.Y("mw:Q", title="Power (MW)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, labelLimit=300),
            ),
        )
        .properties(width=SINGLE_WIDTH_PX, height=170)
        .facet(
            row=alt.Row(
                "panel:N",
                sort=list(panels_text.values()),
                header=facet_header("", on_top=True),
            )
        )
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            "At Donington, metered power plus fitted solar output is the demand without solar, "
            "if the fitted capacity is right"
        ),
        subtitle=[
            (
                "Donington Primary substation, season-year 2025/26. For each season, the seven "
                "consecutive days with the highest mean clear-sky index (CAMS surface sunlight over"
                " its clear-sky value). Day 1 to day 7 are those days in order."
            ),
            (
                "Fitted solar output is capacity C (4.5 MW) times the output per MW that CAMS "
                "sunlight implies. Metered power is power drawn from the electricity network. "
                "Adding the fitted solar output to the metered power gives the demand that the "
                "solar hid from the meter."
            ),
        ],
        figure_planning=None,
    )


def trend_figure(*, table: pl.DataFrame, trend: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the capacity per season-year with its interval and the fitted growth rate."""
    growth = {
        r["site"]: r for r in trend.filter(pl.col("estimator") == "six-arm C").iter_rows(named=True)
    }
    names = {site: name_of(site) for site in FACET_PRIMARIES}
    first_line: dict[str, str] = {}
    second_line: dict[str, str] = {}
    for site in FACET_PRIMARIES:
        if site in growth:
            g = growth[site]
            first_line[site] = f"{g['growth_per_year']:+.0%} a year"
            second_line[site] = f"95%: {g['season_low']:+.0%} to {g['season_high']:+.0%}"
        else:
            first_line[site] = "record too short"
            second_line[site] = "for a trend"
    rows = table.filter(
        pl.col("site").is_in(list(FACET_PRIMARIES)) & pl.col("status").is_in(["clean", "low-power"])
    )
    fits = []
    for site in growth:
        sub = rows.filter(pl.col("site") == site).sort("season_year")
        years = sub["season_year"].to_numpy().astype(float)
        slope, intercept = np.polyfit(years, np.log(sub["capacity_mw"].to_numpy()), 1)
        fits.extend(
            {"site": site, "season_year": y, "fit": float(np.exp(intercept + slope * y))}
            for y in years
        )
    panel_of = pl.col("site").replace_strict(names)
    rows = rows.with_columns(
        panel=panel_of,
        first_line=pl.col("site").replace_strict(first_line),
        second_line=pl.col("site").replace_strict(second_line),
    )
    rows = rows.join(
        pl.DataFrame(fits, schema={"site": pl.String, "season_year": pl.Int64, "fit": pl.Float64}),
        on=["site", "season_year"],
        how="left",
    )
    x = alt.X(
        "season_year:O",
        title="Season-year (September to August)",
        axis=alt.Axis(
            labelExpr="datum.value + '/' + substring(toString(datum.value + 1), 2)",
            labelAngle=-90,
            labelFontSize=9,
            titleFontSize=10,
        ),
    )
    status_scale = alt.Scale(
        domain=["clean", "low-power"], range=[STATUS_COLOURS["clean"], STATUS_COLOURS["low-power"]]
    )
    base = alt.Chart(rows)
    layers = alt.layer(
        base.mark_rule(strokeWidth=1.6, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("capacity_low_mw:Q", title="Capacity C (MW)", scale=alt.Scale(domain=[0, 5.5])),
            y2="capacity_high_mw:Q",
            color=alt.Color("status:N", scale=status_scale, legend=alt.Legend(title="Status")),
        ),
        base.mark_point(size=50, filled=True, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y="capacity_mw:Q",
            color=alt.Color("status:N", scale=status_scale, legend=None),
            fill=alt.condition(
                alt.datum.status == "low-power",
                alt.value("white"),
                alt.Color("status:N", scale=status_scale, legend=None),
            ),
            stroke=alt.Color("status:N", scale=status_scale, legend=None),
            strokeWidth=alt.value(1.6),
        ),
        base.mark_text(
            align="left", baseline="top", fontSize=11, color=ocf.BLACK_1, aria=False
        ).encode(  # ty: ignore[unresolved-attribute]
            x=alt.value(6), y=alt.value(4), text="first_line:N"
        ),
        base.mark_text(
            align="left", baseline="top", fontSize=10, color=ocf.BLACK_1, aria=False
        ).encode(  # ty: ignore[unresolved-attribute]
            x=alt.value(6), y=alt.value(19), text="second_line:N"
        ),
        alt.Chart(rows)
        .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x, y="fit:Q"
        ),
    ).properties(width=PANEL_WIDTH_PX, height=PANEL_HEIGHT_PX)
    chart = layers.facet(
        facet=alt.Facet(
            "panel:N", sort=[names[s] for s in FACET_PRIMARIES], header=facet_header("")
        ),
        columns=3,
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            "Embedded solar capacity has grown 7% to 16% a year at the four Primary substations"
            " with 6 or more season-years"
        ),
        subtitle=[
            (
                "Capacity C per season-year, with the 95% interval from 200 resamples of seven-day "
                "blocks of days. Hollow dot: the test misses a day-long load of a quarter of the "
                "capacity and flags half the capacity in under 80% of resamples. Dashed line: a "
                "straight line fitted to the logarithm of C, "
                "which is constant percentage growth."
            ),
            (
                "Text in each panel: fitted growth per year, with the 95% interval from resampling "
                "the season-years. Six points per line: the interval is wide and the line is a "
                "summary, not a forecast. Kirton (no capacity given) and Wrangle T1 (too small) "
                "are not shown."
            ),
        ],
        figure_planning=None,
    )


def map_figure(*, table: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the Primaries on a map, circle area proportional to the 2025/26 capacity."""
    sites = pl.read_parquet(inputs.SITES_PATH).filter(
        pl.col("site").is_in(list(inputs.PRIMARY_NAMES))
    )
    latest = table.filter(pl.col("season_year") == LATEST)
    rows = (
        sites.join(latest.select("site", "primary", "status", "capacity_mw"), on="site")
        .filter(pl.col("site") != "D15")
        .with_columns(
            longitude=pl.col("longitude").cast(pl.Float64),
            latitude=pl.col("latitude").cast(pl.Float64),
            drawn=pl.col("capacity_mw").fill_null(2.0),
        )
    )
    capacity_rows = rows.filter(pl.col("status") != "flagged")
    flagged_rows = rows.filter(pl.col("status") == "flagged")
    boundary = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    width = CONTENT_WIDTH_PX - 40
    base = (
        alt.Chart(alt.Data(values=boundary["features"]))
        .mark_geoshape(fill=ocf.GREY_2, stroke=ocf.GREY_3, strokeWidth=0.5, aria=False)
        .properties(width=width, height=MAP_HEIGHT_PX)  # ty: ignore[unresolved-attribute]
    )
    scale = alt.Scale(domain=[0, 5], range=[0, 2400])
    circles = (
        alt.Chart(capacity_rows.select("longitude", "latitude", "status", "capacity_mw", "primary"))
        .mark_circle(opacity=0.8, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            longitude="longitude:Q",
            latitude="latitude:Q",
            size=alt.Size(
                "capacity_mw:Q",
                scale=scale,
                legend=alt.Legend(title="Capacity C (MW)", values=[1, 2, 3, 4]),
            ),
            color=alt.Color(
                "status:N",
                scale=alt.Scale(domain=list(STATUS_COLOURS), range=list(STATUS_COLOURS.values())),
                legend=alt.Legend(title="Status"),
            ),
        )
    )
    rings = (
        alt.Chart(flagged_rows.select("longitude", "latitude"))
        .mark_point(size=500, filled=False, strokeWidth=2.5, color=ocf.BRAND_ORANGE, aria=False)
        .encode(longitude="longitude:Q", latitude="latitude:Q")  # ty: ignore[unresolved-attribute]
    )
    left_label = pl.col("site") == inputs.DONINGTON
    labels = [
        alt.Chart(
            rows.filter(left_label if on_left else ~left_label).select(
                "longitude", "latitude", "primary"
            )
        )
        .mark_text(
            align="right" if on_left else "left",
            dx=-16 if on_left else 14,
            fontSize=10,
            color=ocf.BLACK_1,
            aria=False,
        )
        .encode(longitude="longitude:Q", latitude="latitude:Q", text="primary:N")  # ty: ignore[unresolved-attribute]
        for on_left in (True, False)
    ]
    panel = alt.layer(base, circles, rings, *labels).project(
        type="mercator",
        center=list(MAP_CENTRE),
        scale=MAP_SCALE,
        translate=[width / 2, MAP_HEIGHT_PX / 2],
    )
    return figure(
        panels=[panel],
        number=number,
        title=(
            "The eight Primary substations, at seven sites, lie in Lincolnshire, no two sites "
            "more than 55 km apart"
        ),
        subtitle=[
            (
                "Circle area is proportional to capacity C in 2025/26 (MW, 99th percentile of AC "
                "output). Orange ring: Kirton, where the test found a demand that rises on cloudy "
                "days, so no capacity is estimated."
            ),
            (
                "Wrangle T1 and Wrangle T2 are two transformers at one site. Wrangle T1 is not "
                "sized, because its power is too small and noisy for the test."
            ),
        ],
        figure_planning=None,
    )


def spec_text(node: Any, found: list[str]) -> None:
    """Collect every title, subtitle and text string of a Vega-Lite spec, in document order."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"text", "subtitle", "title"} and isinstance(value, str | list):
                found.extend(value if isinstance(value, list) else [value])
            elif key == "value" and isinstance(value, str) and " " in value:
                found.append(value)
            else:
                spec_text(value, found)
    elif isinstance(node, list):
        for item in node:
            spec_text(item, found)


def build_all() -> dict[str, alt.TopLevelMixin]:
    """Return every figure by asset name."""
    table = pl.read_parquet(inputs.TABLE_PATH)
    trend = pl.read_parquet(inputs.TREND_PATH)
    bias = pl.read_parquet(inputs.BIAS_PATH)
    profile = pl.read_parquet(inputs.RAW_PATH)
    gaps = pl.read_parquet(inputs.STUDY_DIR / "midday_gaps.parquet")
    weeks = pl.read_parquet(inputs.STUDY_DIR / "weeks.parquet")
    pooled = pl.read_parquet(inputs.STUDY_DIR / "pooled_gaps.parquet")
    return {
        "embedded_pv_capacity_headline": headline_figure(table=table, number=1),
        "embedded_pv_capacity_map": map_figure(table=table, number=2),
        "embedded_pv_capacity_profiles": profile_figure(profile=profile, number=3),
        "embedded_pv_capacity_gaps": gap_figure(gaps=gaps, number=4),
        "embedded_pv_capacity_kirton": kirton_figure(pooled=pooled, number=5),
        "embedded_pv_capacity_bias": bias_figure(bias=bias, number=6),
        "embedded_pv_capacity_crosscheck": crosscheck_figure(table=table, number=7),
        "embedded_pv_capacity_weeks": weeks_figure(weeks=weeks, number=8),
        "embedded_pv_capacity_trend": trend_figure(table=table, trend=trend, number=9),
    }


def main() -> None:
    """Write the figure text, and the SVGs unless `--text-only` is given."""
    charts = build_all()
    lines = ["# Figure text", ""]
    for name, chart in charts.items():
        found: list[str] = []
        spec_text(chart.to_dict(), found)
        seen = list(dict.fromkeys(found))
        lines += [f"## {name}", "", *[f"- {line}" for line in seen], ""]
    (inputs.STUDY_DIR / "figure_text.md").write_text("\n".join(lines))
    if "--text-only" in sys.argv:
        return
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in charts.items():
        chart.save(ASSETS_DIR / f"{name}.svg")
        print(f"Wrote {ASSETS_DIR / f'{name}.svg'}")


if __name__ == "__main__":
    main()
