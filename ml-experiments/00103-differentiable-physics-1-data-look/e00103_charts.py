"""Draw experiment 103's figures 1 to 12 as SVG into `draft/`.

The two metered generators appear only in figure 9, as G-a and G-b, as aggregate statistics with no
dates. Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=4 uv run python ml-experiments/00103-*/e00103_charts.py
"""

from __future__ import annotations

import textwrap
from datetime import date
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00103_common import (
    FIGURE_DIR,
    GENERATOR_SERIES,
    MVA_PRIMARIES,
    MW_LONG_HISTORY,
    PRIMARIES,
    WORK_DIR,
    hand_filtered_events,
    load_series,
)
from e00103_stats import load_panel

TABLES: Final[Path] = WORK_DIR / "tables"
WIDTH: Final[int] = 560
"""Plot-area width: the axis labels bring a chart to the 680 px text column."""
WRAP: Final[int] = 95
SITES: Final[dict[int, str]] = dict(load_series().select("time_series_id", "site").iter_rows())
SERIES: Final[pl.DataFrame] = load_series()
MW_ORDER: Final[list[str]] = ["D5", "D9", "D10", "D11", "D14", "D16", "D18", "D15"]
BAND_COLOURS: Final[dict[str, str]] = {
    "night": ocf.DATA_BLUE,
    "evening": ocf.BRAND_ORANGE,
    "summer midday": ocf.DATA_SKY,
    "winter midday": ocf.DATA_PURPLE,
}
SHADE: Final[str] = "#EAEAEA"


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURE_DIR / f"{name}.svg")
    chart.save(FIGURE_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_1_availability() -> alt.TopLevelMixin:
    """Share of half-hours present and unflagged per month for 20 series, with switching ticks."""
    table = (
        pl.read_parquet(TABLES / "availability.parquet")
        .with_columns(site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String))
        .filter(~pl.col("site").is_in(list(GENERATOR_SERIES)))
        .with_columns(month_end=pl.col("month").dt.offset_by("1mo"))
    )
    order = sorted(table["site"].unique().to_list(), key=lambda s: int(s[1:]))
    heat = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount="year")),
            x2="month_end:T",
            y=alt.Y("site:N", sort=order, title=None),
            color=alt.Color(
                "share:Q",
                scale=alt.Scale(domain=[0, 1], range=["#FFFFFF", ocf.DATA_BLUE]),
                legend=alt.Legend(title="Share present and unflagged", orient="top", format=".0%"),
            ),
        )
    )
    events = (
        hand_filtered_events()
        .explode("series_ids")
        .rename({"series_ids": "time_series_id"})
        .with_columns(
            site=pl.col("time_series_id").replace_strict(
                SITES, return_dtype=pl.String, default=None
            )
        )
        .drop_nulls("site")
        .filter(~pl.col("site").is_in(list(GENERATOR_SERIES)))
    )
    ticks = (
        alt.Chart(events)  # ty: ignore[unresolved-attribute]
        .mark_tick(color=ocf.BRAND_ORANGE, thickness=1.5, size=14)
        .encode(x="onset_utc:T", y=alt.Y("site:N", sort=order))
    )
    return (heat + ticks).properties(
        width=WIDTH,
        height=18 * len(order),
        title=_title(
            text="Availability of the 20 series that are not metered generators, 2019 to 2026",
            lines=[
                (
                    "Shade: share of half-hours present and unflagged in each month (darker is "
                    "more). "
                    "Orange tick: the start of one of the 66 hand-filtered switching events that "
                    "touches the series. The two metered generators are left out."
                )
            ],
        ),
    )


def figure_2_raw_weeks() -> alt.TopLevelMixin:
    """Half-hourly power in four weeks at the 8 MW Primaries."""
    ids = {v: k for k, v in SITES.items()}
    panel = load_panel().filter(
        pl.col("clean"), pl.col("time_series_id").is_in([ids[s] for s in MW_ORDER])
    )
    weeks = {
        "Jul 2020": date(2020, 7, 6),
        "Jan 2021": date(2021, 1, 11),
        "Jul 2025": date(2025, 7, 7),
        "Jan 2026": date(2026, 1, 12),
    }
    frames = []
    for name, start in weeks.items():
        part = panel.filter(
            pl.col("local_date") >= start,
            pl.col("local_date") < pl.lit(start) + pl.duration(days=7),
        ).with_columns(
            week=pl.lit(name),
            day=(pl.col("local_date") - pl.lit(start)).dt.total_days()
            + pl.col("local_hour_start") / 24.0,
        )
        frames.append(part.select("site", "week", "day", "points"))
    table = pl.concat(frames)
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1)
        .encode(
            x=alt.X(
                "day:Q",
                title="Days since Monday 00:00 local time",
                axis=alt.Axis(values=list(range(8))),
                scale=alt.Scale(domain=[0, 7]),
            ),
            y=alt.Y("points:Q", title=None),
            color=alt.Color(
                "week:N",
                scale=alt.Scale(
                    domain=list(weeks),
                    range=[ocf.DATA_SKY, ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(title="Week", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 40, height=70)
        .facet(
            row=alt.Row(
                "site:N",
                sort=MW_ORDER,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            )
        )
        .properties(
            title=_title(
                text="Net power at the 8 MW Primaries in four weeks",
                lines=[
                    (
                        "Half-hourly net power, as a percentage of each site's effective capacity "
                        "(negative is export; the vertical axis of every panel is this "
                        "percentage). A "
                        "July and a January week in 2020-21 and in 2025-26; a site with no line in"
                        " a "
                        "week had no clean data then."
                    )
                ],
            )
        )
    )


def figure_3_monthly_levels() -> alt.TopLevelMixin:
    """Monthly level by hour band per MW Primary, with the lockdown and cut dates marked."""
    ids = {v: k for k, v in SITES.items()}
    levels = pl.read_parquet(TABLES / "monthly_levels.parquet").filter(
        pl.col("time_series_id").is_in([ids[s] for s in MW_ORDER])
    )
    long = (
        levels.unpivot(
            on=["night", "evening", "summer_midday", "winter_midday"],
            index=["time_series_id", "month"],
            variable_name="band",
            value_name="points",
        )
        .drop_nulls("points")
        .with_columns(
            site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String),
            band=pl.col("band").str.replace("_", " "),
        )
    )
    cuts = (
        pl.read_parquet(TABLES / "cuts.parquet")
        .with_columns(site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String))
        .filter(pl.col("site").is_in(MW_ORDER) & (pl.col("cut_months") > 0))
    )
    shade = pl.DataFrame(
        {
            "start": [date(2020, 3, 23), date(2022, 10, 1)],
            "end": [date(2021, 7, 19), date(2022, 11, 1)],
        }
    )
    panels = []
    for k, site in enumerate(MW_ORDER):
        last = k == len(MW_ORDER) - 1
        rects = alt.Chart(shade).mark_rect(color=SHADE, opacity=0.8).encode(x="start:T", x2="end:T")  # ty: ignore[unresolved-attribute]
        lines = (
            alt.Chart(long.filter(pl.col("site") == site))  # ty: ignore[unresolved-attribute]
            .mark_line(strokeWidth=1.2)
            .encode(
                x=alt.X(
                    "month:T",
                    title=None,
                    axis=alt.Axis(format="%Y", tickCount="year", labels=last, ticks=last),
                    scale=alt.Scale(domain=[date(2019, 9, 1), date(2026, 7, 1)]),
                ),
                y=alt.Y("points:Q", title=site),
                color=alt.Color(
                    "band:N",
                    scale=alt.Scale(domain=list(BAND_COLOURS), range=list(BAND_COLOURS.values())),
                    legend=alt.Legend(
                        title="Hour band (local time)", orient="top", direction="horizontal"
                    ),
                ),
            )
        )
        layers = [rects, lines]
        cut = cuts.filter(pl.col("site") == site)
        if cut.height:
            layers.append(
                alt.Chart(cut)  # ty: ignore[unresolved-attribute]
                .mark_rule(color="black", strokeDash=[4, 3])
                .encode(x="first_usable:T")
            )
        panels.append(alt.layer(*layers).properties(width=WIDTH - 40, height=70))
    return alt.vconcat(*panels, spacing=4).properties(
        title=_title(
            text="Monthly level by hour band at the 8 MW Primaries",
            lines=[
                (
                    "Mean net power in each month, as a share of effective capacity (axis title: "
                    "the "
                    "site). Night: 00:00 to 04:59; evening: 17:00 to 20:59; midday: 11:00 to "
                    "14:59, "
                    "split into April to September (summer) and October to March (winter). Grey: "
                    "March"
                    " 2020 to June 2021, and October 2022. Dashed line: the first usable date "
                    "after "
                    "the early-calibration cut."
                )
            ],
        )
    )


def figure_4_early_calibration() -> alt.TopLevelMixin:
    """Level in each of a series' first 12 months minus the same calendar month a year later."""
    table = (
        pl.read_parquet(TABLES / "early_calibration.parquet")
        .with_columns(site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String))
        .filter(pl.col("site").is_in(list(PRIMARIES)))
        .unpivot(
            on=["night_diff", "weekday_day_diff"],
            index=["site", "month_index"],
            variable_name="band",
            value_name="difference",
        )
        .with_columns(
            band=pl.col("band").replace(
                {"night_diff": "night", "weekday_day_diff": "weekday 09:00-16:59"}
            )
        )
    )
    guides = (
        alt.Chart(pl.DataFrame({"y": [-15, 0, 15]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[3, 3], color="grey")
        .encode(y="y:Q")
    )
    order = sorted(PRIMARIES, key=lambda s: int(s[1:]))
    rows = []
    for r in range(4):
        cells = []
        for site in order[4 * r : 4 * r + 4]:
            lines = (
                alt.Chart(table.filter(pl.col("site") == site))  # ty: ignore[unresolved-attribute]
                .mark_line(point=True, strokeWidth=1.2)
                .encode(
                    x=alt.X(
                        "month_index:Q",
                        title="Months since start" if r == 3 else None,
                        axis=alt.Axis(values=[0, 6, 12]),
                        scale=alt.Scale(domain=[0, 12]),
                    ),
                    y=alt.Y(
                        "difference:Q",
                        title="Points" if site == order[4 * r] else None,
                        scale=alt.Scale(domain=[-45, 45]),
                        axis=alt.Axis(values=[-30, -15, 0, 15, 30]),
                    ),
                    color=alt.Color(
                        "band:N",
                        scale=alt.Scale(
                            domain=["night", "weekday 09:00-16:59"],
                            range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                        ),
                        legend=alt.Legend(title="Hour band", orient="top", direction="horizontal"),
                    ),
                )
            )
            cells.append(
                alt.layer(guides, lines).properties(
                    width=115, height=80, title=alt.TitleParams(site, fontSize=11, anchor="start")
                )
            )
        rows.append(alt.hconcat(*cells, spacing=6))
    return alt.vconcat(*rows, spacing=6).properties(
        title=_title(
            text="Early calibration: the first 12 months against the same months a year later",
            lines=[
                (
                    "Difference in mean level (percentage points of effective capacity) between "
                    "month "
                    "k of the series and the same calendar month a year later; where that falls in "
                    "the fixed lockdown window the first later year outside it is used; for D15 "
                    "and "
                    "D18 the comparison month may fall as late as September 2026. Dotted lines: "
                    "plus "
                    "and minus 15 points."
                )
            ],
        )
    )


def figure_5_profiles() -> alt.TopLevelMixin:
    """Mean weekday and weekend half-hourly profile per MW Primary per season-year."""
    ids = {v: k for k, v in SITES.items()}
    panel = (
        load_panel()
        .filter(
            pl.col("clean"),
            ~pl.col("switching_mask"),
            pl.col("time_series_id").is_in([ids[s] for s in MW_ORDER]),
        )
        .with_columns(
            kind=pl.when(pl.col("weekday") <= 5)
            .then(pl.lit("weekday"))
            .otherwise(pl.lit("weekend")),
            season=(
                pl.when(pl.col("local_date").dt.month() >= 7)
                .then(pl.col("local_date").dt.year())
                .otherwise(pl.col("local_date").dt.year() - 1)
            ),
        )
    )
    table = (
        panel.group_by("site", "kind", "season", "local_hour_start")
        .agg(points=pl.col("points").mean(), n=pl.len())
        .filter(pl.col("n") >= 40)
        .with_columns(local_hour_start=pl.col("local_hour_start").floor())
        .group_by("site", "kind", "season", "local_hour_start")
        .agg(points=pl.col("points").mean())
    )
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("season:O", title="Season-year (July to June)"),
            y=alt.Y(
                "local_hour_start:Q",
                bin=alt.Bin(step=1),
                title="Local hour",
                axis=alt.Axis(values=[0, 6, 12, 18, 24], format="d"),
                scale=alt.Scale(domain=[0, 24]),
            ),
            color=alt.Color(
                "points:Q",
                scale=alt.Scale(scheme="blueorange", domainMid=0),
                legend=alt.Legend(title="Mean net power (% of capacity)", orient="top"),
            ),
        )
        .properties(width=140, height=70)
        .facet(
            row=alt.Row(
                "site:N",
                sort=MW_ORDER,
                title=None,
                header=alt.Header(labelAngle=0, labelAlign="left"),
            ),
            column=alt.Column("kind:N", title=None),
        )
        .properties(
            title=_title(
                text="Mean daily profile at the 8 MW Primaries by season-year",
                lines=[
                    (
                        "Each cell: mean net power over the half-hour of the local day, for the "
                        "weekdays or the weekend days of one season-year (July to June); cells "
                        "with "
                        "fewer than 40 observations are blank."
                    )
                ],
            )
        )
    )


def figure_6_clip() -> alt.TopLevelMixin:
    """Summer weekday midday power against CAMS GHI, with the latest summer's fitted slopes."""
    ids = {v: k for k, v in SITES.items()}
    panel = load_panel().filter(
        pl.col("clean"),
        ~pl.col("switching_mask"),
        pl.col("time_series_id").is_in([ids[s] for s in MW_ORDER]),
        pl.col("local_date").dt.month().is_between(5, 8),
        pl.col("weekday") <= 5,
        pl.col("cams_hour_end").dt.hour().is_between(11, 14),
    )
    hourly = (
        panel.group_by("site", "cams_hour_end")
        .agg(power=pl.col("points").mean(), ghi=pl.col("ghi_w_m2").first(), n=pl.len())
        .filter(pl.col("n") == 2)
        .with_columns(
            summer=pl.when(pl.col("cams_hour_end").dt.year() == 2025)
            .then(pl.lit("2025"))
            .otherwise(pl.lit("2019-2024 and 2026"))
        )
    )
    # Thin the dots so the SVG stays small: every twelfth hour of the earlier summers and every
    # third hour of 2025. The fits use every hour.
    thinned = pl.concat(
        [
            hourly.filter(pl.col("summer") == "2025")
            .sort("site", "cams_hour_end")
            .with_row_index()
            .filter(pl.col("index") % 3 == 0)
            .drop("index"),
            hourly.filter(pl.col("summer") != "2025")
            .sort("site", "cams_hour_end")
            .with_row_index()
            .filter(pl.col("index") % 12 == 0)
            .drop("index"),
        ]
    )
    fits = []
    for site in MW_ORDER:
        part = hourly.filter(pl.col("site") == site, pl.col("summer") == "2025")
        for name, low, high in (("below 500", 0, 500), ("above 600", 600, 1000)):
            sub = part.filter(pl.col("ghi") >= low, pl.col("ghi") <= high)
            if sub.height >= 30:
                slope, intercept = np.polyfit(sub["ghi"].to_numpy(), sub["power"].to_numpy(), 1)
                fits.append(
                    {
                        "site": site,
                        "fit": name,
                        "ghi": float(low),
                        "power": float(slope * low + intercept),
                    }
                )
                fits.append(
                    {
                        "site": site,
                        "fit": name,
                        "ghi": float(min(high, sub["ghi"].to_numpy().max())),
                        "power": float(slope * min(high, sub["ghi"].to_numpy().max()) + intercept),
                    }
                )
    fit_table = pl.DataFrame(fits)
    rows = []
    for r in range(2):
        cells = []
        for site in MW_ORDER[4 * r : 4 * r + 4]:
            points = (
                alt.Chart(  # ty: ignore[unresolved-attribute]
                    thinned.filter(pl.col("site") == site).with_columns(
                        pl.col("power", "ghi").round(1)
                    )
                )
                .mark_circle(size=8, opacity=0.35)
                .encode(
                    x=alt.X(
                        "ghi:Q",
                        title="CAMS irradiance (W/m\u00b2)" if r == 1 else None,
                        scale=alt.Scale(domain=[0, 1000]),
                        axis=alt.Axis(values=[0, 250, 500, 750, 1000]),
                    ),
                    y=alt.Y(
                        "power:Q",
                        title="Net power (% of capacity)" if site == MW_ORDER[4 * r] else None,
                    ),
                    color=alt.Color(
                        "summer:N",
                        scale=alt.Scale(
                            domain=["2019-2024 and 2026", "2025"],
                            range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE],
                        ),
                        legend=alt.Legend(
                            title="Summer (May to August)", orient="top", direction="horizontal"
                        ),
                    ),
                )
            )
            lines = (
                alt.Chart(fit_table.filter(pl.col("site") == site))  # ty: ignore[unresolved-attribute]
                .mark_line(color="black", strokeWidth=1.5)
                .encode(x="ghi:Q", y="power:Q", detail="fit:N")
            )
            cells.append(
                alt.layer(points, lines).properties(
                    width=120, height=110, title=alt.TitleParams(site, fontSize=11, anchor="start")
                )
            )
        rows.append(alt.hconcat(*cells, spacing=6))
    return alt.vconcat(*rows, spacing=8).properties(
        title=_title(
            text="Clip look: summer weekday midday power against irradiance at the 8 MW Primaries",
            lines=[
                (
                    "Each dot: the mean net power over one weekday hour (CAMS hour ending 11:00 to "
                    "14:00 UTC, May to August) against that hour's CAMS irradiance. Black lines: "
                    "least-squares fits below 500 W/m\u00b2 and above 600 W/m\u00b2 for summer "
                    "2025."
                )
            ],
        )
    )


def figure_7_heating() -> alt.TopLevelMixin:
    """Night power against temperature each winter, and each winter's slope with its interval."""
    daily = (
        pl.read_parquet(TABLES / "daily.parquet")
        .with_columns(
            site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String),
            winter=pl.when(pl.col("local_date").dt.month() >= 11)
            .then(pl.col("local_date").dt.year())
            .when(pl.col("local_date").dt.month() <= 2)
            .then(pl.col("local_date").dt.year() - 1),
        )
        .filter(
            pl.col("site").is_in(MW_ORDER),
            pl.col("winter").is_not_null(),
            pl.col("temperature_c") < 12,
            pl.col("night_n") >= 8,
            pl.col("night_mean").is_not_null(),
        )
    )
    daily = daily.with_columns(
        night_mean=pl.col("night_mean").round(1), temperature_c=pl.col("temperature_c").round(1)
    )
    first_last = daily.group_by("site").agg(
        first=pl.col("winter").min(), last=pl.col("winter").max()
    )
    daily = daily.join(first_last, on="site").with_columns(
        which=pl.when(pl.col("winter") == pl.col("first"))
        .then(pl.lit("first winter"))
        .when(pl.col("winter") == pl.col("last"))
        .then(pl.lit("last winter"))
        .otherwise(pl.lit("other winters"))
    )
    # Thin the other winters (every sixth day) so the SVG stays small. The slopes use every day.
    daily = pl.concat(
        [
            daily.filter(pl.col("which") != "other winters"),
            daily.filter(pl.col("which") == "other winters")
            .sort("site", "local_date")
            .with_row_index()
            .filter(pl.col("index") % 6 == 0)
            .drop("index"),
        ]
    )
    rows = []
    for r in range(2):
        cells = []
        for site in MW_ORDER[4 * r : 4 * r + 4]:
            cells.append(  # noqa: PERF401  (the other panel loops build their cells the same way)
                alt.Chart(daily.filter(pl.col("site") == site))  # ty: ignore[unresolved-attribute]
                .mark_circle(size=8, opacity=0.5)
                .encode(
                    x=alt.X(
                        "temperature_c:Q",
                        title="Daily-mean temp. (\u00b0C)" if r == 1 else None,
                        scale=alt.Scale(domain=[-2, 12]),
                        axis=alt.Axis(values=[0, 4, 8, 12]),
                    ),
                    y=alt.Y(
                        "night_mean:Q",
                        title="Night mean (% of cap.)" if site == MW_ORDER[4 * r] else None,
                    ),
                    color=alt.Color(
                        "which:N",
                        scale=alt.Scale(
                            domain=["first winter", "other winters", "last winter"],
                            range=[ocf.DATA_BLUE, "#C9C9C9", ocf.BRAND_ORANGE],
                        ),
                        legend=alt.Legend(
                            title="Winter (November to February)",
                            orient="top",
                            direction="horizontal",
                        ),
                    ),
                )
                .properties(
                    width=100, height=100, title=alt.TitleParams(site, fontSize=11, anchor="start")
                )
            )
        rows.append(alt.hconcat(*cells, spacing=6))
    scatter = alt.vconcat(*rows, spacing=8)
    slopes = (
        pl.read_parquet(TABLES / "heating.parquet")
        .with_columns(site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String))
        .filter(pl.col("site").is_in(MW_ORDER), pl.col("days") >= 60)
        .drop_nulls("low")
    )
    zero = (
        alt.Chart(pl.DataFrame({"y": [0.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[3, 3])
        .encode(y="y:Q")
    )
    interval = (
        alt.Chart(slopes)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=1.5, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X(
                "winter:O",
                title="Winter (year of its November)",
                axis=alt.Axis(labelAngle=0, labelOverlap="parity"),
            ),
            y=alt.Y("low:Q", title="Slope (points/\u00b0C)"),
            y2="high:Q",
        )
    )
    dot = (
        alt.Chart(slopes)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=40, color=ocf.DATA_BLUE)
        .encode(x="winter:O", y="slope:Q")
    )
    panels = []
    for r in range(2):
        cells = [
            alt.layer(
                zero,
                interval.transform_filter(alt.datum.site == site),
                dot.transform_filter(alt.datum.site == site),
            ).properties(
                width=100, height=90, title=alt.TitleParams(site, fontSize=11, anchor="start")
            )
            for site in MW_ORDER[4 * r : 4 * r + 4]
        ]
        panels.append(alt.hconcat(*cells, spacing=6))
    slope_chart = alt.vconcat(*panels, spacing=8)
    return alt.vconcat(scatter, slope_chart, spacing=16).properties(
        title=_title(
            text="Night power against temperature in winter at the 8 MW Primaries",
            lines=[
                (
                    "Top: each dot is one winter day (November to February, daily-mean ERA5 "
                    "temperature below 12 \u00b0C); blue is the site's first winter after its "
                    "first "
                    "usable date, orange its last. Bottom: each winter's least-squares slope with "
                    "a "
                    "95% interval from resampling whole calendar months (points of effective "
                    "capacity "
                    "per \u00b0C; more negative means more heating load per degree)."
                )
            ],
        )
    )


def figure_8_correlation() -> alt.TopLevelMixin:
    """Pairwise correlation of monthly night and evening levels across the long-history sites."""
    frames = []
    for band in ("night", "evening"):
        corr = pl.read_parquet(TABLES / f"level_correlation_{band}.parquet")
        names = [c for c in corr.columns if c not in ("series", "months")]
        long = corr.unpivot(
            on=names, index="series", variable_name="other", value_name="correlation"
        ).with_columns(
            band=pl.lit(band),
            a=pl.col("series").cast(pl.Int64).replace_strict(SITES, return_dtype=pl.String),
            b=pl.col("other").cast(pl.Int64).replace_strict(SITES, return_dtype=pl.String),
        )
        frames.append(long)
    table = pl.concat(frames)
    order = ["D5", "D10", "D11", "D14", "D16"]
    base = alt.Chart(table).encode(
        x=alt.X("a:N", sort=order, title=None), y=alt.Y("b:N", sort=order, title=None)
    )
    heat = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "correlation:Q",
            scale=alt.Scale(domain=[-1, 0, 1], range=[ocf.BRAND_ORANGE, "#FFFFFF", ocf.DATA_BLUE]),
            legend=alt.Legend(title="Correlation of monthly level", orient="top"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text=alt.Text("correlation:Q", format=".2f"))  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(heat, text)
        .properties(width=190, height=190)
        .facet(column=alt.Column("band:N", title=None, sort=["night", "evening"]))
        .properties(
            title=_title(
                text="Correlation of monthly levels across the five long-history sites",
                lines=[
                    (
                        "Each cell: the correlation, across the months with data at all five sites "
                        "(outside the lockdown window), of two sites' monthly mean level after "
                        "each site's mean seasonal cycle is removed. D9 is left out because its "
                        "history starts in 2023."
                    )
                ],
            )
        )
    )


def figure_9_generator_screen() -> alt.TopLevelMixin:
    """The screen's inputs for all 22 series, and D15 and D18's nights beside their sum."""
    screen = (
        pl.read_parquet(TABLES / "generator_screen.parquet")
        .with_columns(
            label=pl.col("site").replace(GENERATOR_SERIES),
            kind=pl.when(pl.col("site").is_in(list(GENERATOR_SERIES)) | (pl.col("site") == "D15"))
            .then(pl.lit("positive control"))
            .when(pl.col("group") == "grid flow")
            .then(pl.lit("grid or bulk supply point"))
            .otherwise(pl.lit("Primary substation")),
            share_percent=pl.col("negative_dark_share") * 100,
        )
        .with_columns(
            range_ratio=pl.col("range_over_mean").cast(pl.Float64),
            labelled=pl.col("kind") != "Primary substation",
        )
    )
    colour = alt.Color(
        "kind:N",
        scale=alt.Scale(
            domain=["Primary substation", "grid or bulk supply point", "positive control"],
            range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE],
        ),
        legend=alt.Legend(title="Kind of series", orient="top", direction="horizontal"),
    )
    limit = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color="grey", strokeDash=[4, 3])
        .encode(x="x:Q")
    )

    def panel(y: str, title: str, *, log: bool = False) -> alt.LayerChart:
        base = alt.Chart(screen).encode(
            x=alt.X(
                "share_percent:Q",
                title="Dark half-hours with negative power (%)" if y == "range_ratio" else None,
                scale=alt.Scale(domain=[-3, 103]),
                axis=alt.Axis(values=[0, 1, 25, 50, 75, 100]),
            ),
            y=alt.Y(
                f"{y}:Q",
                title=title,
                scale=alt.Scale(type="symlog") if log else alt.Scale(zero=False),
            ),
        )
        return alt.layer(  # ty: ignore[invalid-return-type]
            limit,
            base.mark_circle(size=60, aria=False).encode(color=colour),  # ty: ignore[unresolved-attribute]
            base.transform_filter(alt.datum.labelled)  # ty: ignore[unresolved-attribute]
            .mark_text(dx=7, dy=-5, align="left", fontSize=9, aria=False)
            .encode(text="label:N"),
        ).properties(width=WIDTH - 40, height=110)

    scatter = alt.vconcat(
        panel("winter_night_slope", "Winter night slope (points/\u00b0C)"),
        panel("range_over_capacity", "Daily range (share of capacity)"),
        panel("range_ratio", "Daily range / |mean| (log)", log=True),
        spacing=6,
    )
    ids = {v: k for k, v in SITES.items()}
    panel_data = load_panel().filter(
        pl.col("clean"), pl.col("time_series_id").is_in([ids["D15"], ids["D18"]])
    )
    night = (
        panel_data.filter((pl.col("local_hour_start") >= 0) & (pl.col("local_hour_start") < 5))
        .group_by("site", "local_date")
        .agg(night=pl.col("power").mean(), n=pl.len())
        .filter(pl.col("n") >= 8)
    )
    wide = (
        night.pivot(on="site", index="local_date", values="night")
        .sort("local_date")
        .with_columns(total=pl.col("D15") + pl.col("D18"))
    )
    long = (
        wide.unpivot(
            on=["D15", "D18", "total"],
            index="local_date",
            variable_name="series",
            value_name="night_mw",
        )
        .drop_nulls("night_mw")
        .filter(pl.col("local_date") >= date(2025, 2, 1))
    )
    long = long.with_columns(series=pl.col("series").replace({"total": "D15 + D18"}))
    cuts = pl.DataFrame(
        {
            "date": [date(2025, 2, 1), date(2025, 10, 1)],
            "what": ["D18 first usable date", "D15 first usable date"],
        }
    )
    line = (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=1.2)
        .encode(
            x=alt.X("local_date:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=8)),
            y=alt.Y("night_mw:Q", title="Night mean (MW or MVA as metered)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["D15", "D18", "D15 + D18"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE, "black"],
                ),
                legend=alt.Legend(
                    title="Daily night mean of", orient="top", direction="horizontal"
                ),
            ),
        )
    )
    rules = alt.Chart(cuts).mark_rule(strokeDash=[4, 3], color="grey").encode(x="date:T")  # ty: ignore[unresolved-attribute]
    nights = alt.layer(line, rules).properties(width=WIDTH - 40, height=130)
    return (
        alt.vconcat(scatter, nights, spacing=18)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Generator screen for all 22 series, and the nights at D15 and D18",
                lines=[
                    (
                        "Top three panels: for each series (only the series that are not Primary "
                        "substations, and D15, are labelled), the share of its dark half-hours "
                        "(CAMS "
                        "irradiance below 1 W/m\u00b2) with power below zero against, in turn, the "
                        "slope of its winter night level on daily-mean temperature, its median "
                        "daily "
                        "range as a share of effective capacity, and its median daily range "
                        "divided by"
                        " the absolute daily mean. Dashed line: the 1% flag. G-a and G-b are the "
                        "two "
                        "metered generators, shown with no dates. Bottom: the daily night mean of "
                        "D15 "
                        "and D18 and their sum, February 2025 to June 2026."
                    )
                ],
            )
        )
    )


def _timestamp_panel(*, table_name: str, panel_title: str, after_label: str) -> alt.LayerChart:
    """One panel of figure 10: correlation against stamp shift, before and after the fix."""
    before_label = "before the correction"
    table = (
        pl.read_parquet(TABLES / f"{table_name}.parquet")
        .with_columns(
            site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String),
            side=pl.when(pl.col("after_fix"))
            .then(pl.lit(after_label))
            .otherwise(pl.lit(before_label)),
        )
        .filter(pl.col("site").is_in(MW_ORDER))
    )
    mean = table.group_by("side", "shift_minutes").agg(correlation=pl.col("correlation").mean())
    colour = alt.Color(
        "side:N",
        scale=alt.Scale(
            domain=[before_label, after_label], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
        ),
        legend=alt.Legend(title="Rows", orient="top", direction="vertical", labelLimit=400),
    )
    x = alt.X(
        "shift_minutes:Q",
        title="Shift applied to the power half-hour stamps (minutes)",
        axis=alt.Axis(values=[-90, -60, -30, 0, 30, 60, 90]),
    )
    thin = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=0.8, opacity=0.35)
        .encode(
            x=x,
            y=alt.Y("correlation:Q", title="Correlation with CAMS irradiance"),
            color=colour,
            detail="site:N",
        )
    )
    bold = (
        alt.Chart(mean)  # ty: ignore[unresolved-attribute]
        .mark_line(strokeWidth=3, point=True)
        .encode(x=x, y="correlation:Q", color=colour)
    )
    return alt.layer(thin, bold).properties(  # ty: ignore[invalid-return-type]
        width=WIDTH - 40,
        height=170,
        title=alt.TitleParams(panel_title, fontSize=12, anchor="start"),
    )


def figure_10_timestamp() -> alt.TopLevelMixin:
    """Correlation of hourly power with CAMS irradiance by stamp shift, in two tests."""
    top = _timestamp_panel(
        table_name="timestamp",
        panel_title="Clear-sky hours in April to September",
        after_label="after the correction (April to September 2026)",
    )
    bottom = _timestamp_panel(
        table_name="timestamp_summer_middays",
        panel_title="Summer middays: May to August, 10:00 to 14:00 UTC",
        after_label="after the correction (May to August 2026)",
    )
    return (
        alt.vconcat(top, bottom, spacing=14)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Timestamp convention: the correlation is most negative between -30 and +30",
                lines=[
                    (
                        "Correlation of the hourly mean net power at the 8 MW Primaries with CAMS "
                        "irradiance when the power stamps are shifted by the amount on the axis. "
                        "Net power falls as irradiance rises, so the correlation is negative and "
                        "the right shift makes it most negative. Top: clear-sky hours (clear-sky "
                        "index above 0.8, clear-sky irradiance above 100 W/m\u00b2). Bottom: the "
                        "plan's stricter test, with clear-sky irradiance above 400 W/m\u00b2. "
                        "The bottom panel is nearly flat. Thin lines: one site; thick line: "
                        "the mean over sites."
                    )
                ],
            )
        )
    )


def figure_11_mva_dark() -> alt.TopLevelMixin:
    """Dark site-years per MVA Primary and season-year, and the dark-row daily profiles."""
    season = (
        pl.read_parquet(TABLES / "dark_by_season.parquet")
        .with_columns(site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String))
        .filter(pl.col("site").is_in(MVA_PRIMARIES))
    )
    bars = (
        alt.Chart(season)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X(
                "season:O",
                title="Season-year (July to June; 2019 and 2026 are part-years)",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("site_years:Q", title="Dark site-years"),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(
                    domain=list(MVA_PRIMARIES),
                    range=[
                        ocf.DATA_BLUE,
                        ocf.BRAND_ORANGE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.DATA_BLUE_LIGHT,
                        ocf.BRAND_ORANGE_LIGHT,
                        ocf.DATA_SKY_LIGHT,
                    ],
                ),
                legend=alt.Legend(title="MVA Primary", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 40, height=120)
    )
    ids = {v: k for k, v in SITES.items()}
    wanted = [ids[s] for s in (*MVA_PRIMARIES, *MW_LONG_HISTORY, "D18")]
    panel = load_panel().filter(
        pl.col("clean"),
        pl.col("dark"),
        ~pl.col("switching_mask"),
        pl.col("time_series_id").is_in(wanted),
    )
    profile = (
        panel.group_by("site", "local_hour_start")
        .agg(power=pl.col("power").mean(), n=pl.len())
        .filter(pl.col("n") >= 100)
    )
    profile = profile.with_columns(
        normalised=pl.col("power") / pl.col("power").mean().over("site"),
        kind=pl.when(pl.col("site").is_in(list(MVA_PRIMARIES)))
        .then(pl.lit("MVA Primary (|S|)"))
        .otherwise(pl.lit("MW Primary (net MW)")),
    )
    lines = (
        alt.Chart(profile)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=14, opacity=0.8)
        .encode(
            x=alt.X(
                "local_hour_start:Q",
                title="Local hour of day (dark half-hours only)",
                axis=alt.Axis(values=[0, 6, 12, 18, 24]),
                scale=alt.Scale(domain=[0, 24]),
            ),
            y=alt.Y("normalised:Q", title="Mean dark-row power / site mean"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["MVA Primary (|S|)", "MW Primary (net MW)"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title="Series", orient="top", direction="horizontal"),
            ),
        )
        .properties(width=WIDTH - 40, height=150)
    )
    return (
        alt.vconcat(bars, lines, spacing=16)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Dark rows at the 8 MVA Primaries",
                lines=[
                    (
                        "Top: half-hours with CAMS irradiance below 1 W/m\u00b2 (clean rows after "
                        "the "
                        "early-calibration cut and outside switching masks), in site-years of "
                        "17,532 "
                        "half-hours. Bottom: the mean power over those dark half-hours by local "
                        "hour, "
                        "each site divided by its own mean; each dot is one site and half-hour, "
                        "shown "
                        "only where the site has at least 100 dark half-hours. D15, which carries "
                        "an "
                        "embedded generator, is left out."
                    )
                ],
            )
        )
    )


def figure_12_lockdown() -> alt.TopLevelMixin:
    """Departure of the weekday daytime level from a no-lockdown fit at four sites."""
    table = pl.read_parquet(TABLES / "lockdown_departures.parquet").with_columns(
        site=pl.col("time_series_id").replace_strict(SITES, return_dtype=pl.String)
    )
    shade = pl.DataFrame({"start": [date(2020, 3, 23)], "end": [date(2021, 7, 19)]})
    rect = alt.Chart(shade).mark_rect(color=SHADE, opacity=0.8).encode(x="start:T", x2="end:T")  # ty: ignore[unresolved-attribute]
    guides = (
        alt.Chart(pl.DataFrame({"y": [-5, 0, 5]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[3, 3], color="grey")
        .encode(y="y:Q")
    )
    lines = (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, strokeWidth=1.3)
        .encode(
            x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %Y", tickCount=8)),
            y=alt.Y("departure:Q", title="Departure from the no-lockdown fit (points)"),
            color=alt.Color(
                "site:N",
                scale=alt.Scale(
                    domain=["D5", "D10", "D11", "D14"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE, ocf.DATA_SKY],
                ),
                legend=alt.Legend(title="Site", orient="top", direction="horizontal"),
            ),
        )
    )
    return alt.layer(rect, guides, lines).properties(
        width=WIDTH - 40,
        height=200,
        title=_title(
            text="Lockdown: weekday daytime level against a fit that leaves the window out",
            lines=[
                (
                    "Monthly mean net power on weekdays 09:00 to 16:59 minus a per-site fit "
                    "(calendar-"
                    "month effect plus linear trend) to the months outside the grey window, in "
                    "points of"
                    " effective capacity. D14 enters in July 2020 after its early-calibration cut. "
                    "Dotted lines: plus and minus 5 points."
                )
            ],
        ),
    )


def main() -> None:
    """Draw figures 1 to 12."""
    for name, function in (
        ("figure_01_availability", figure_1_availability),
        ("figure_02_raw_weeks", figure_2_raw_weeks),
        ("figure_03_monthly_levels", figure_3_monthly_levels),
        ("figure_04_early_calibration", figure_4_early_calibration),
        ("figure_05_profiles", figure_5_profiles),
        ("figure_06_clip", figure_6_clip),
        ("figure_07_heating", figure_7_heating),
        ("figure_08_correlation", figure_8_correlation),
        ("figure_09_generator_screen", figure_9_generator_screen),
        ("figure_10_timestamp", figure_10_timestamp),
        ("figure_11_mva_dark", figure_11_mva_dark),
        ("figure_12_lockdown", figure_12_lockdown),
    ):
        _save(function(), name=name)


if __name__ == "__main__":
    main()
