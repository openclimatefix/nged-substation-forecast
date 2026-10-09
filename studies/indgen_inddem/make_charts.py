"""Draw the INDDEM and INDGEN page's figures as SVG under `docs/studies/assets/`.

Run after `build_tables.py`: `uv run python studies/indgen_inddem/make_charts.py`. Every number a
title quotes is computed from the tables that `build_tables.py` wrote, so a figure cannot disagree
with `report.md`. All of the data is public, so the charts name the GSP groups and show megawatts
on calendar dates.
"""

import subprocess
from datetime import UTC, datetime, timedelta
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from indgen_inddem_common import (
    ASSETS_DIR,
    INDDEM_MATCH_TOLERANCE_MW,
    LONDON,
    NGED_GROUPS,
    PV_LIVE_PATH,
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
INDDEM_NAME: Final[str] = "INDDEM (demand, sign reversed)"
INDGEN_NAME: Final[str] = "INDGEN (generation)"
PN_IMPORT_NAME: Final[str] = "Sum of the sampled PNs, import"
PN_EXPORT_NAME: Final[str] = "Sum of the sampled PNs, export"
SERIES_COLOURS: Final[dict[str, str]] = {
    INDDEM_NAME: ocf.BRAND_ORANGE,
    INDGEN_NAME: ocf.DATA_BLUE,
    PN_IMPORT_NAME: ocf.BRAND_ORANGE,
    PN_EXPORT_NAME: ocf.DATA_BLUE,
    "National demand outturn (INDO)": ocf.DATA_BLUE,
    "AGV, summed over the 14 groups": ocf.BRAND_ORANGE,
    "AGV (net import)": ocf.BRAND_ORANGE,
    "INDDEM of the best-matching zone": ocf.DATA_BLUE,
    "AGV plus PV_Live solar (gross demand)": ocf.DATA_PURPLE,
    "PV_Live solar generation": ocf.DATA_GREEN,
}
DASHED: Final[tuple[str, ...]] = (PN_IMPORT_NAME, PN_EXPORT_NAME)
ROW_HEIGHT_PX: Final[int] = 130
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 100
ZONE_LABELS: Final[dict[str, str]] = {
    "Z1": "Z1 (= B1)",
    "Z3": "Z3 (= B3, Sloy)",
    "Z11": "Z11 (= B17, West Midlands)",
    "Z14": "Z14 (= B14, London)",
    "Z15": "Z15 (= B15, Thames Estuary)",
    "Z17": "Z17 (= B13, South West)",
}
"""Zone labels. A zone that equals one boundary names the boundary."""
LATEST_FILTER: Final[pl.Expr] = pl.col("view") == "latest"


def layered(*charts: alt.Chart | alt.LayerChart) -> alt.LayerChart:
    """Layer charts, typed as the layer chart that `studies.charts.figure` accepts as a panel."""
    return cast(alt.LayerChart, alt.layer(*charts))


def colour_scale(*, names: list[str]) -> alt.Scale:
    """Return a colour scale that gives each name in `names` its colour from `SERIES_COLOURS`."""
    return alt.Scale(domain=names, range=[SERIES_COLOURS[name] for name in names])


def dash_scale(*, names: list[str]) -> alt.Scale:
    """Return a stroke-dash scale that dashes the names in `DASHED` and no others."""
    return alt.Scale(domain=names, range=[[4, 3] if name in DASHED else [1, 0] for name in names])


def title(text: str) -> alt.TitleParams:
    """Return the small left-aligned title of one panel."""
    return alt.TitleParams(text, anchor="start", fontSize=11, offset=2)


def week_slice(*, frame: pl.DataFrame, season: str) -> pl.DataFrame:
    """Keep the rows in the seven UTC days that start on the season's fixed Monday."""
    start = WEEKS[season]
    return frame.filter(pl.col("time").is_between(start, start + timedelta(days=7), closed="left"))


def line_panel(
    *,
    data: pl.DataFrame,
    names: list[str],
    panel_title: str,
    y_title: str,
    last: bool,
    width: int = PLOT_WIDTH_PX,
    x_format: str = "%a %-d %b",
    x_title: str = "Day (UTC)",
    tick_count: Any = "day",
) -> alt.LayerChart:
    """Draw one row of lines against UTC time, coloured by series.

    The data needs `time`, `series`, and `megawatts` columns.
    """
    chart = (
        alt.Chart(data, title=title(panel_title))
        .mark_line(strokeWidth=1.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T",
                axis=alt.Axis(
                    format=x_format,
                    tickCount=tick_count,
                    labels=last,
                    ticks=last,
                    title=x_title if last else None,
                ),
            ),
            y=alt.Y("megawatts:Q", axis=alt.Axis(tickCount=4, title=y_title)),
            color=alt.Color("series:N", scale=colour_scale(names=names), legend=None),
            strokeDash=alt.StrokeDash("series:N", scale=dash_scale(names=names), legend=None),
        )
        .properties(width=width, height=ROW_HEIGHT_PX)
    )
    return layered(chart)


def stack(*, series: dict[str, pl.DataFrame], value: str) -> pl.DataFrame:
    """Stack tables that each have `time` and the column `value` into `time, series, megawatts`."""
    return pl.concat(
        [
            frame.select("time", series=pl.lit(name), megawatts=pl.col(value))
            for name, frame in series.items()
        ]
    )


def unit_check_figure(*, number: int) -> alt.VConcatChart:
    """Draw AGV beside the initial national demand outturn, as a week and as a scatter."""
    check = pl.read_parquet(STUDY_DIR / "agv_against_indo.parquet")
    ratio = check["ratio"].median()
    correlation = check.select(pl.corr("agv_mw", "indo_mw")).item()
    names = ["National demand outturn (INDO)", "AGV, summed over the 14 groups"]
    long = pl.concat(
        [
            check.select("time", series=pl.lit(names[0]), megawatts="indo_mw"),
            check.select("time", series=pl.lit(names[1]), megawatts="agv_mw"),
        ]
    )
    week = line_panel(
        data=week_slice(frame=long, season="winter"),
        names=names,
        panel_title="One winter week, 15 to 21 December 2025",
        y_title="MW",
        last=True,
    )
    scatter = (
        alt.Chart(
            check.gather_every(7).select(indo="indo_mw", agv="agv_mw"),
            title=title("Every seventh half-hour, 1 September 2025 to 20 September 2026"),
        )
        .mark_circle(size=6, opacity=0.35, color=ocf.DATA_BLUE, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("indo:Q", axis=alt.Axis(title="National demand outturn (MW)")),
            y=alt.Y("agv:Q", axis=alt.Axis(title="AGV summed over the 14 groups (MW)")),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * 2)
    )
    line = (
        alt.Chart(pl.DataFrame({"x": [10_000.0, 45_000.0], "y": [10_000.0, 45_000.0]}))
        .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
        .encode(x="x:Q", y="y:Q")  # ty: ignore[unresolved-attribute]
    )
    return figure(
        panels=[week, layered(scatter, line)],
        number=number,
        title=(
            f"AGV, doubled, is {ratio:.0%} of the national demand outturn, so AGV is in "
            "megawatt-hours per half-hour"
        ),
        subtitle=[
            (
                "Half-hourly values. Orange: AGV, the energy the 14 GSP groups take from the "
                "transmission system, times 2 to give megawatts. Blue: the national demand "
                "outturn (INDO). In the scatter, the dashed line is equality."
            ),
            f"The correlation of the {len(check)} half-hours is {correlation:.3f}.",
        ],
        figure_planning=None,
    )


def zone_sign_figure(*, number: int) -> alt.VConcatChart:
    """Draw, for each zone, the half-hours in which the derived zone has the wrong sign."""
    signs = pl.read_parquet(STUDY_DIR / "zone_signs.parquet").filter(LATEST_FILTER)
    total = int(signs["wrong_sign"].sum())
    half_hours = int(signs["half_hours"].sum())
    names = {"inddem": INDDEM_NAME, "indgen": INDGEN_NAME}
    data = signs.with_columns(series=pl.col("dataset").replace_strict(names))
    chart = (
        alt.Chart(data)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("zone:N", sort=list(ZONES), axis=alt.Axis(title="Study zone", labelAngle=0)),
            xOffset="series:N",
            y=alt.Y(
                "wrong_sign:Q",
                axis=alt.Axis(
                    title=f"Half-hours with the wrong sign (of {signs['half_hours'][0]})"
                ),
            ),
            color=alt.Color(
                "series:N",
                scale=colour_scale(names=list(names.values())),
                legend=alt.Legend(title=None, labelLimit=400),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=int(ROW_HEIGHT_PX * 1.5))
    )
    return figure(
        panels=[chart],
        number=number,
        title=(
            f"The zones recovered from the boundaries have the expected sign in all but {total} of "
            f"{half_hours:,} zone half-hours"
        ),
        subtitle=[
            (
                "Each zone is a combination of the national total and the 17 boundaries, by the "
                "formulas of Elexon CVA Change Circular 235. A demand zone should be zero or "
                "negative and a generation zone zero or positive, within 1 MW of rounding."
            ),
            "The latest issue before each half-hour, 1 September 2025 to 30 September 2026.",
        ],
        figure_planning=None,
    )


def national_series() -> pl.DataFrame:
    """Return the national INDDEM (sign reversed) and INDGEN, in the `latest` view, as `stack`."""
    views = pl.read_parquet(STUDY_DIR / "views.parquet").filter(
        LATEST_FILTER & (pl.col("boundary") == "N")
    )
    return stack(
        series={
            INDDEM_NAME: views.filter(pl.col("dataset") == "inddem").with_columns(
                megawatts=-pl.col("value_mw")
            ),
            INDGEN_NAME: views.filter(pl.col("dataset") == "indgen").with_columns(
                megawatts=pl.col("value_mw")
            ),
        },
        value="megawatts",
    )


def national_figure(*, number: int) -> alt.VConcatChart:
    """Draw the national INDDEM and INDGEN over the year, and over a winter and a summer week."""
    long = national_series()
    names = [INDDEM_NAME, INDGEN_NAME]
    daily = (
        long.sort("series", "time")
        .group_by_dynamic("time", every="1d", group_by="series")
        .agg(megawatts=pl.col("megawatts").mean())
    )
    year = line_panel(
        data=daily,
        names=names,
        panel_title="Daily mean, 1 September 2025 to 30 September 2026",
        y_title="MW",
        last=True,
        x_format="%b %Y",
        tick_count="month",
    )
    weeks = [
        line_panel(
            data=week_slice(frame=long, season=season),
            names=names,
            panel_title=f"{season.capitalize()} week, from Monday {WEEKS[season]:%-d %B %Y}",
            y_title="MW",
            last=season == "summer",
        )
        for season in ("winter", "summer")
    ]
    means = long.group_by("series").agg(mean=pl.col("megawatts").mean())
    demand = means.filter(pl.col("series") == INDDEM_NAME)["mean"].item()
    generation = means.filter(pl.col("series") == INDGEN_NAME)["mean"].item()
    return figure(
        panels=[year, *weeks],
        number=number,
        title=(
            f"National INDDEM averages {demand / 1000:.1f} GW and INDGEN {generation / 1000:.1f} "
            "GW over the year, and both follow the daily cycle"
        ),
        subtitle=[
            (
                "Half-hourly national totals of the Physical Notifications (PNs) in the latest "
                "issue published before each half-hour. Orange: INDDEM, the PNs of the BMUs "
                "that plan to import, with the sign reversed to show demand as positive. Blue: "
                "INDGEN, the PNs of the BMUs that plan to export."
            ),
        ],
        figure_planning=None,
    )


def local_hour(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `hour`, the UK local time of day in hours, from the UTC `time` column."""
    local = pl.col("time").dt.convert_time_zone(LONDON)
    return frame.with_columns(
        hour=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60
    )


def zone_profile_panel(*, zones: pl.DataFrame, zone: str) -> alt.LayerChart:
    """Draw one zone's mean INDDEM (sign reversed) and INDGEN by UK local time of day."""
    names = [INDDEM_NAME, INDGEN_NAME]
    frame = local_hour(frame=zones.filter(pl.col("zone") == zone))
    profile = (
        frame.group_by("dataset", "hour")
        .agg(megawatts=pl.col("value_mw").mean())
        .with_columns(
            series=pl.col("dataset").replace_strict({"inddem": INDDEM_NAME, "indgen": INDGEN_NAME}),
            megawatts=pl.when(pl.col("dataset") == "inddem")
            .then(-pl.col("megawatts"))
            .otherwise(pl.col("megawatts")),
        )
        .sort("series", "hour")
    )
    chart = (
        alt.Chart(profile, title=title(ZONE_LABELS.get(zone, zone)))
        .mark_line(strokeWidth=1.3, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=[0, 24], nice=False),
                axis=alt.Axis(values=[0, 6, 12, 18, 24], title=None),
            ),
            y=alt.Y("megawatts:Q", axis=alt.Axis(tickCount=3, title=None)),
            color=alt.Color("series:N", scale=colour_scale(names=names), legend=None),
        )
        .properties(width=PLOT_WIDTH_PX // 3 - 40, height=80)
    )
    return layered(chart)


def zone_profiles_figure(*, number: int) -> alt.VConcatChart:
    """Draw the 17 zones' mean daily profiles as small multiples."""
    zones = pl.read_parquet(STUDY_DIR / "zones.parquet").filter(LATEST_FILTER)
    panels = [zone_profile_panel(zones=zones, zone=zone) for zone in ZONES]
    rows = [
        alt.hconcat(*panels[start : start + 3], spacing=14) for start in range(0, len(ZONES), 3)
    ]
    means = (
        zones.filter(pl.col("dataset") == "inddem")
        .group_by("zone")
        .agg(mean=-pl.col("value_mw").mean())
        .sort("mean", descending=True)
    )
    largest = means.row(0, named=True)
    return figure(
        panels=rows,
        number=number,
        title=(
            f"Zone {largest['zone'][1:]} holds the most INDDEM, {largest['mean'] / 1000:.1f} GW on "
            "average, and the zones differ in the shape of their day"
        ),
        subtitle=[
            (
                "Mean by UK local time of day, 1 September 2025 to 30 September 2026. Orange: "
                "INDDEM, sign reversed. Blue: INDGEN. Each panel has its own vertical scale, in "
                "megawatts. Zones come from the latest issue before each half-hour."
            ),
        ],
        figure_planning=None,
    )


def reach_figure(*, number: int) -> alt.VConcatChart:
    """Draw how many hours ahead each INDDEM issue reaches, by UK local publication time."""
    reach = pl.read_parquet(STUDY_DIR / "issue_reach.parquet").filter(
        (pl.col("dataset") == "inddem") & (pl.col("reach_hours") > 1)
    )
    local = pl.col("publish_time").dt.convert_time_zone(LONDON)
    data = reach.select(
        hour=local.dt.hour().cast(pl.Float64) + local.dt.minute().cast(pl.Float64) / 60,
        reach_hours="reach_hours",
        clocks=pl.when(local.dt.dst_offset().dt.total_hours() == 0)
        .then(pl.lit("GMT"))
        .otherwise(pl.lit("BST")),
    )
    chart = (
        alt.Chart(data)
        .mark_circle(size=14, opacity=0.5, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=[0, 24], nice=False),
                axis=alt.Axis(values=list(range(0, 25, 3)), title="UK local time of publication"),
            ),
            y=alt.Y("reach_hours:Q", axis=alt.Axis(title="Hours ahead of publication")),
            color=alt.Color(
                "clocks:N",
                scale=alt.Scale(domain=["GMT", "BST"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title="UK clocks"),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * 2)
    )
    longest = data["reach_hours"].max()
    shortest = data["reach_hours"].min()
    return figure(
        panels=[chart],
        number=number,
        title=(
            f"An INDDEM issue reaches {shortest:.0f} to {longest:.0f} hours ahead, "
            "far short of the 14 days of the forecast horizon"
        ),
        subtitle=[
            (
                f"Each dot is one of {len(data)} issues of the national total. The reach is the "
                "time from publication to the end of the last half-hour in the issue. INDGEN's "
                "issues have the same reach."
            ),
        ],
        figure_planning=None,
    )


def difference_panel(*, frame: pl.DataFrame, dataset: str, name: str, last: bool) -> alt.LayerChart:
    """Draw the mean (00:00 UTC issue - latest issue) by UK local target half-hour and clocks."""
    sign = -1.0 if dataset == "inddem" else 1.0
    stats = (
        frame.filter(pl.col("dataset") == dataset)
        .group_by("clocks", "local_half_hour")
        .agg(mean=(sign * pl.col("difference_mw")).mean())
        .with_columns(hour=pl.col("local_half_hour") / 2)
        .sort("clocks", "hour")
    )
    chart = (
        alt.Chart(stats, title=title(name))
        .mark_line(strokeWidth=1.8, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "hour:Q",
                scale=alt.Scale(domain=[0, 24], nice=False),
                axis=alt.Axis(
                    values=list(range(0, 25, 3)),
                    title="Target half-hour, UK local time" if last else None,
                    labels=last,
                    ticks=last,
                ),
            ),
            y=alt.Y("mean:Q", axis=alt.Axis(title="MW", tickCount=4)),
            color=alt.Color(
                "clocks:N",
                scale=alt.Scale(domain=["GMT", "BST"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title="UK clocks"),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX)
    )
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=0.8, aria=False)
        .encode(y="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    return layered(zero, chart)


def first_versus_latest_figure(*, number: int) -> alt.VConcatChart:
    """Draw how far the 00:00 UTC issue sits from the latest issue, by target half-hour."""
    frame = pl.read_parquet(STUDY_DIR / "first_versus_latest.parquet")
    mean_abs = frame.group_by("dataset").agg(m=pl.col("difference_mw").abs().mean())
    inddem_abs = mean_abs.filter(pl.col("dataset") == "inddem")["m"].item()
    indgen_abs = mean_abs.filter(pl.col("dataset") == "indgen")["m"].item()
    panels = [
        difference_panel(
            frame=frame,
            dataset="inddem",
            name="INDDEM (demand, sign reversed): 00:00 UTC issue minus latest issue",
            last=False,
        ),
        difference_panel(
            frame=frame,
            dataset="indgen",
            name="INDGEN (generation): 00:00 UTC issue minus latest issue",
            last=True,
        ),
    ]
    return figure(
        panels=panels,
        number=number,
        title=(
            f"The 00:00 UTC issue differs from the latest issue by a mean absolute "
            f"{inddem_abs:,.0f} MW (INDDEM) and {indgen_abs:,.0f} MW (INDGEN), with its largest "
            "step at 23:00 UK local time"
        ),
        subtitle=[
            (
                "National total, 1 September 2025 to 30 September 2026. The 00:00 UTC issue is the "
                "latest one published at or before 00:00 UTC on the target's UTC day, which is the "
                "23:47 UTC issue of the day before. The lines are the signed mean difference."
            ),
        ],
        figure_planning=None,
    )


def pn_against_inddem_figure(*, number: int) -> alt.VConcatChart:
    """Draw the sampled PN sums beside INDDEM and INDGEN on each of the four sample days."""
    joined = pl.read_parquet(STUDY_DIR / "pn_against_inddem.parquet").sort("time")
    names = [INDDEM_NAME, PN_IMPORT_NAME, INDGEN_NAME, PN_EXPORT_NAME]
    local_day = pl.col("time").dt.convert_time_zone(LONDON).dt.date()
    days = sorted(set(joined.select(local_day.alias("day"))["day"]))
    difference = (joined["inddem"] - joined["pn_import_mw"]).abs()
    reproduced = int((difference <= INDDEM_MATCH_TOLERANCE_MW).sum())
    largest_gap = joined.select((pl.col("inddem") - pl.col("pn_import_mw")).abs().max()).item()
    panels = []
    for index, day in enumerate(days):
        sub = joined.filter(local_day == day)
        long = pl.concat(
            [
                sub.select("time", series=pl.lit(names[0]), megawatts=-pl.col("inddem")),
                sub.select("time", series=pl.lit(names[1]), megawatts=-pl.col("pn_import_mw")),
                sub.select("time", series=pl.lit(names[2]), megawatts="indgen"),
                sub.select("time", series=pl.lit(names[3]), megawatts="pn_export_mw"),
            ]
        )
        panels.append(
            line_panel(
                data=long,
                names=names,
                panel_title=f"{day:%A %-d %B %Y}",
                y_title="MW",
                last=index == len(days) - 1,
                x_format="%H:%M",
                x_title="Time (UTC)",
                tick_count=8,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=(
            "Summed Physical Notifications reproduce INDDEM to within "
            f"{INDDEM_MATCH_TOLERANCE_MW:g} MW in {reproduced} of {joined.height} half-hours "
            f"and miss it by up to {largest_gap / 1000:.1f} GW"
        ),
        subtitle=[
            (
                "Solid: INDDEM (orange, sign reversed) and INDGEN (blue), the latest issue "
                "before each half-hour. Dashed: the same quantity summed from the final PNs of "
                "every BMU, which Elexon serves now. The sum for a half-hour is the time-weighted "
                "mean of each BMU's notified level, with imports and exports summed apart."
            ),
        ],
        figure_planning=None,
    )


def ordered_groups(*, present: set[str]) -> list[str]:
    """Order the columns: NGED's four GSP groups, the other groups, the interconnectors, `none`."""
    letters = sorted(
        group for group in present if group.startswith("_") and group not in NGED_GROUPS
    )
    interconnectors = sorted(group for group in present if group.startswith("IC "))
    tail = ["none"] if "none" in present else []
    return [*NGED_GROUPS, *letters, *interconnectors, *tail]


def weights_figure(*, number: int) -> alt.VConcatChart:
    """Draw the fitted share of each GSP group's import PNs in each zone as a heat map."""
    fractions = pl.read_parquet(STUDY_DIR / "zone_group_fractions.parquet")
    groups = ordered_groups(present=set(fractions["gsp_group"]))
    heat = (
        alt.Chart(fractions)
        .mark_rect(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "gsp_group:N",
                sort=groups,
                axis=alt.Axis(title="Column (GSP group or interconnector)", labelAngle=-90),
            ),
            y=alt.Y("zone:N", sort=list(ZONES), axis=alt.Axis(title="Study zone")),
            color=alt.Color(
                "fraction:Q",
                scale=alt.Scale(
                    domain=[0, 1],
                    range=[ocf.WHITE, ocf.BRAND_ORANGE],
                    clamp=True,
                    interpolate="rgb",
                ),
                legend=alt.Legend(title="Fitted share"),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=int(ROW_HEIGHT_PX * 2.6))
    )
    labels = (
        alt.Chart(fractions.filter(pl.col("fraction") >= 0.3))
        .mark_text(fontSize=9, color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("gsp_group:N", sort=groups),
            y=alt.Y("zone:N", sort=list(ZONES)),
            text=alt.Text("fraction:Q", format=".1f"),
        )
    )
    return figure(
        panels=[layered(heat, labels)],
        number=number,
        title=(
            "Each interconnector lands in the zone where it is known to land, and no GSP group "
            f"is settled by a fit on {fractions['half_hours'][0]} half-hours"
        ),
        subtitle=[
            (
                "Each zone's INDDEM is fitted as the sum, over columns, of the column's summed "
                "import PNs times a share, with non-negative shares that sum to 1 over the 17 "
                "zones for each column. A column wholly inside one zone would have a share near 1 "
                "there. The columns are the 14 GSP groups, the 10 interconnectors (each netted), "
                "and `none`, the BMUs that Elexon's register gives no GSP group. Fitted on the "
                "half-hours where the sampled PNs reproduce INDDEM's national total. Shares of "
                "0.3 or more are printed."
            ),
        ],
        figure_planning=None,
    )


def correlation_figure(*, number: int) -> alt.VConcatChart:
    """Draw each GSP group's AGV against each zone's INDDEM, with the national anomaly removed."""
    correlations = pl.read_parquet(STUDY_DIR / "anomaly_correlations.parquet")
    groups = ordered_groups(present=set(correlations["gsp_group"]))
    heat = (
        alt.Chart(correlations)
        .mark_rect(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("zone:N", sort=list(ZONES), axis=alt.Axis(title="Study zone", labelAngle=0)),
            y=alt.Y("gsp_group:N", sort=groups, axis=alt.Axis(title="GSP group")),
            color=alt.Color(
                "correlation_without_national:Q",
                scale=alt.Scale(
                    domain=[-0.5, 0, 0.5],
                    range=[ocf.BRAND_ORANGE, ocf.WHITE, ocf.DATA_BLUE],
                    clamp=True,
                    interpolate="rgb",
                ),
                legend=alt.Legend(title="Correlation"),
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=ROW_HEIGHT_PX * 2)
    )
    labels = (
        alt.Chart(correlations)
        .mark_text(fontSize=8, color=ocf.BLACK_1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("zone:N", sort=list(ZONES)),
            y=alt.Y("gsp_group:N", sort=groups),
            text=alt.Text("correlation_without_national:Q", format=".1f"),
        )
    )
    highest = correlations["correlation_without_national"].max()
    return figure(
        panels=[layered(heat, labels)],
        number=number,
        title=(
            "With the shared cycles and the national anomaly removed, the highest correlation "
            f"between a GSP group's AGV and a zone's INDDEM is {highest:.2f}"
        ),
        subtitle=[
            (
                "Correlation of the half-hourly anomalies of each GSP group's AGV and each zone's "
                "INDDEM (latest issue), 1 September 2025 to 20 September 2026, after removing "
                "from each the multiple of the national anomaly that a least-squares fit gives. "
                "An anomaly is a series minus its mean at the same UK local half-hour, day type, "
                "and month. NGED's four groups are the top rows."
            ),
            (
                "The median over all 238 cells is "
                f"{correlations['correlation_raw'].median():.2f} for the raw series, "
                f"{correlations['correlation_anomaly'].median():.2f} for the anomalies, and "
                f"{correlations['correlation_without_national'].median():.2f} with the national "
                "anomaly removed."
            ),
        ],
        figure_planning=None,
    )


def best_zones() -> dict[str, tuple[str, float]]:
    """Return, for each NGED group, the zone whose INDDEM anomaly correlates best with its AGV."""
    correlations = pl.read_parquet(STUDY_DIR / "anomaly_correlations.parquet")
    best = {}
    for group in NGED_GROUPS:
        top = correlations.filter(pl.col("gsp_group") == group).sort(
            "correlation_without_national", descending=True
        )
        best[group] = (top["zone"][0], top["correlation_without_national"][0])
    return best


def nged_figure(*, number: int) -> alt.VConcatChart:
    """Draw each NGED group's AGV beside the INDDEM of its best-correlated zone, in two weeks."""
    agv = pl.read_parquet(STUDY_DIR / "agv_groups.parquet")
    zones = pl.read_parquet(STUDY_DIR / "zones.parquet").filter(
        LATEST_FILTER & (pl.col("dataset") == "inddem")
    )
    names = ["AGV (net import)", "INDDEM of the best-matching zone"]
    rows = []
    best = best_zones()
    ratios = pl.read_parquet(STUDY_DIR / "nged_levels.parquet")["zone_to_agv_ratio"]
    for index, (group, group_name) in enumerate(NGED_GROUPS.items()):
        zone, correlation = best[group]
        halves = []
        for season in ("winter", "summer"):
            long = stack(
                series={
                    names[0]: week_slice(
                        frame=agv.filter(pl.col("gsp_group") == group), season=season
                    ),
                    names[1]: week_slice(
                        frame=zones.filter(pl.col("zone") == zone), season=season
                    ).with_columns(import_mw=-pl.col("value_mw")),
                },
                value="import_mw",
            )
            halves.append(
                line_panel(
                    data=long,
                    names=names,
                    panel_title=(
                        f"{group} {group_name}, winter week"
                        if season == "winter"
                        else f"summer week; best zone {zone} (correlation {correlation:.2f})"
                    ),
                    y_title="MW",
                    last=index == len(NGED_GROUPS) - 1,
                    width=PLOT_WIDTH_PX // 2 - 20,
                    x_format="%a",
                    x_title="Day of the week",
                )
            )
        rows.append(alt.hconcat(*halves, spacing=16))
    return figure(
        panels=rows,
        number=number,
        title=(
            f"The zone that correlates best with an NGED GSP group has {min(ratios):.1f} to "
            f"{max(ratios):.1f} times the group's mean AGV, so the two series differ in level"
        ),
        subtitle=[
            (
                "Orange: AGV, the energy the group takes from the transmission system, in "
                "megawatts. Blue: INDDEM, sign reversed, of the zone whose correlation with the "
                "group is highest after the national anomaly is removed (named in the right-hand "
                "title). Left: the winter week starting Monday 15 December 2025. Right: the "
                "summer week starting Monday 15 June 2026."
            ),
        ],
        figure_planning=None,
    )


def pv_live_figure(*, number: int) -> alt.VConcatChart:
    """Draw AGV with and without PV_Live's solar for NGED's four groups in the summer week."""
    agv = pl.read_parquet(STUDY_DIR / "agv_groups.parquet")
    pv = pl.read_parquet(PV_LIVE_PATH).select("time", "gsp_group", "generation_mw")
    names = [
        "AGV (net import)",
        "AGV plus PV_Live solar (gross demand)",
        "PV_Live solar generation",
    ]
    panels = []
    shares = pl.read_parquet(STUDY_DIR / "nged_levels.parquet")["solar_over_agv"]
    for index, (group, group_name) in enumerate(NGED_GROUPS.items()):
        joined = agv.filter(pl.col("gsp_group") == group).join(
            pv.filter(pl.col("gsp_group") == group), on=["time", "gsp_group"]
        )
        week = week_slice(frame=joined, season="summer")
        long = stack(
            series={
                names[0]: week.with_columns(megawatts=pl.col("import_mw")),
                names[1]: week.with_columns(
                    megawatts=pl.col("import_mw") + pl.col("generation_mw")
                ),
                names[2]: week.with_columns(megawatts=pl.col("generation_mw")),
            },
            value="megawatts",
        )
        panels.append(
            line_panel(
                data=long,
                names=names,
                panel_title=f"{group} {group_name}",
                y_title="MW",
                last=index == len(NGED_GROUPS) - 1,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=(
            f"PV_Live's solar equals {shares.min():.0%} to {shares.max():.0%} of the settled take "
            "of NGED's four GSP groups over the study window"
        ),
        subtitle=[
            (
                "Summer week starting Monday 15 June 2026. Orange: AGV, the net import from the "
                "transmission system. Green: PV_Live's estimate of the solar generation of the "
                "distribution licence area with the same letter, which is embedded in the "
                "distribution network. Purple: their sum, which still leaves out other embedded "
                "generation. The shares in the title are each group's solar divided by its AGV, "
                "summed over 1 September 2025 to 20 September 2026."
            ),
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
        "indgen_inddem_correlations": correlation_figure(number=1),
        "indgen_inddem_unit_check": unit_check_figure(number=2),
        "indgen_inddem_zone_signs": zone_sign_figure(number=3),
        "indgen_inddem_national": national_figure(number=4),
        "indgen_inddem_zone_profiles": zone_profiles_figure(number=5),
        "indgen_inddem_reach": reach_figure(number=6),
        "indgen_inddem_first_versus_latest": first_versus_latest_figure(number=7),
        "indgen_inddem_pn_against_inddem": pn_against_inddem_figure(number=8),
        "indgen_inddem_weights": weights_figure(number=9),
        "indgen_inddem_nged": nged_figure(number=10),
        "indgen_inddem_pv_live": pv_live_figure(number=11),
    }
    for name, chart in charts.items():
        save(chart=chart, name=name)


if __name__ == "__main__":
    main()
