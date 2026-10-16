"""Draw experiment 00039's 19 figures from the saved results, as SVG (raw data first).

Every farm is "Solar farm A" to "Solar farm F"; no chart carries a calendar date beside a farm's
output, and the one chart of injected farm output counts days 1 to 7. The SVGs are afterwards
optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run python ml-experiments/00039-pv-disaggregation-cams-injection/e00039_charts.py
"""

import textwrap
from typing import Final

import altair as alt
import e00039_common as common
import e00039_model as model
import e00039_report as report
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 560
WRAP: Final[int] = 88
NAMES: Final[dict[str, str]] = common.PRIMARY_NAMES
SCOPE: Final[str] = (
    "8 directional Primary substations; 6 solar farms (A to F); season-years run September to "
    "August."
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _season(*, month: pl.Expr) -> pl.Expr:
    return (
        pl.when(month.is_in(common.SUMMER_MONTHS))
        .then(pl.lit("April to September"))
        .otherwise(pl.lit("October to March"))
    )


SEASON_SCALE: Final[alt.Scale] = alt.Scale(
    domain=["April to September", "October to March"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
)


def _prepared() -> pl.DataFrame:
    return model.prepare(frame=pl.read_parquet(common.FRAME_PATH)).with_columns(
        name=pl.col("site").replace(NAMES)
    )


def figure_01(*, rows: dict[str, dict[str, float]], units: pl.DataFrame) -> alt.TopLevelMixin:
    """Headline: median recovery ratio with 95% intervals, units as faint dots."""
    labels = {
        "M1_sub": "M1, CAMS at the substation",
        "M1d_sub": "M1d (daily), CAMS at the substation",
        "M1L_sub": "M1L (cloud term), CAMS at the substation",
        "M1_own": "M1, CAMS at the farm",
        "M1d_own": "M1d (daily), CAMS at the farm",
    }
    order = list(labels.values())
    summary = pl.DataFrame(
        [{"label": labels[k], **{f: rows[k][f] for f in ("median", "low", "high")}} for k in labels]
    )
    dots = units.select(
        pl.col("config").replace(dict(labels)).alias("label"),
        "R",
    )
    y = alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300))
    band = (
        alt.Chart(pl.DataFrame({"a": [0.75], "b": [1.25]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_GREEN_LIGHT, opacity=0.6)
        .encode(x=alt.X("a:Q", scale=alt.Scale(domain=[0.4, 1.3])), x2="b:Q")
    )
    one = alt.Chart(pl.DataFrame({"x": [1.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    faint = (
        alt.Chart(dots)
        .mark_circle(size=12, opacity=0.25, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "R:Q",
                scale=alt.Scale(domain=[0.4, 1.3]),
                title="Recovered capacity over injected capacity",
            ),
            y=y,
        )
    )
    bars = (
        alt.Chart(summary)
        .mark_rule(color=ocf.BRAND_ORANGE, strokeWidth=3)
        .encode(  # ty: ignore[unresolved-attribute]
            x="low:Q", x2="high:Q", y=y
        )
    )
    points = (
        alt.Chart(summary)
        .mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE)
        .encode(  # ty: ignore[unresolved-attribute]
            x="median:Q", y=y
        )
    )
    return alt.layer(band, one, faint, bars, points).properties(
        width=WIDTH,
        height=190,
        title=_title(
            text="A CAMS-driven fit recovers 77% to 88% of injected solar capacity",
            lines=[
                (
                    "Orange dot: median over 218 units. Orange line: 95% interval from resampling "
                    "Primaries and farms. Faint dots: single units. Green band: the "
                    "pre-registered 0.75 to 1.25."
                ),
                "Injected capacity is 20% of each Primary's effective capacity. " + SCOPE,
            ],
        ),
    )


def figure_02(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw: Donington's midday power against CAMS irradiance, 2025/26, by season."""
    d = prepared.filter(
        (pl.col("site") == "D16")
        & (pl.col("season_year") == 2025)
        & pl.col("power").is_not_null()
        & pl.col("half_hour").is_between(22, 27)
        & (pl.col("off") == 0)
    ).with_columns(season=_season(month=pl.col("month")))
    chart = (
        alt.Chart(d.select("power", "ghi_u", "season"))
        .mark_circle(size=14, opacity=0.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("ghi_u:Q", title="CAMS global horizontal irradiance (W/m²)"),
            y=alt.Y("power:Q", title="Substation power (MW; negative is export)"),
            color=alt.Color(
                "season:N", scale=SEASON_SCALE, title=None, legend=alt.Legend(orient="top")
            ),
        )
    )
    return chart.properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="At Donington, midday power falls as sunshine rises",
            lines=[
                (
                    "Each dot is one weekday half-hour between 11:00 and 14:00 local in the "
                    "2025/26 season-year. The downward slope is embedded solar generation."
                ),
            ],
        ),
    )


def figure_03(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw: midday mean power by calendar month, one line per season-year, per Primary."""
    d = (
        prepared.filter(
            pl.col("site").is_in(common.DIRECTIONAL_PRIMARIES)
            & pl.col("power").is_not_null()
            & pl.col("half_hour").is_between(22, 27)
        )
        .group_by("name", "season_year", "month")
        .agg(pl.col("power").mean(), pl.len().alias("n"))
        .filter(pl.col("n") > 60)
        .with_columns(season=pl.col("season_year").cast(pl.String))
    )
    return (
        alt.Chart(d.select("name", "season", "month", "power"))  # ty: ignore[unresolved-attribute]
        .mark_line(point=False)
        .encode(
            x=alt.X("month:O", title="Calendar month", axis=alt.Axis(values=[3, 6, 9, 12])),
            y=alt.Y("power:Q", title="Midday mean power (MW)"),
            color=alt.Color(
                "season:N",
                title="Season-year",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=250, height=120)
        .facet(
            facet=alt.Facet(
                "name:N", title=None, sort=[NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
            ),
            columns=2,
        )
        .properties(
            title=_title(
                text="The summer dip in midday demand deepens over the years",
                lines=[
                    (
                        "Mean power from 11:00 to 14:00 local by calendar month, one line per "
                        "season-year starting in the year shown; months with fewer than 60 "
                        "half-hours are omitted."
                    )
                ],
            )
        )
    )


def figure_04(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw: clear-sky midday power minus overcast midday power, per Primary and season-year."""
    daily = (
        prepared.filter(pl.col("site").is_in(common.DIRECTIONAL_PRIMARIES))
        .group_by("site", "name", "season_year", "date")
        .agg(
            (pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum()).alias("csi"),
            pl.col("power").filter(pl.col("half_hour").is_between(24, 25)).mean().alias("midday"),
            pl.col("month").first(),
        )
        .filter(pl.col("month").is_between(5, 8) & pl.col("midday").is_not_null())
    )
    clear = (
        daily.filter(pl.col("csi") >= 0.8)
        .group_by("name", "season_year")
        .agg(pl.col("midday").mean().alias("clear"), pl.len().alias("n_clear"))
    )
    dull = (
        daily.filter(pl.col("csi") <= 0.3)
        .group_by("name", "season_year")
        .agg(pl.col("midday").mean().alias("overcast"), pl.len().alias("n_overcast"))
    )
    d = (
        clear.join(dull, on=["name", "season_year"])
        .with_columns(
            gap=pl.col("clear") - pl.col("overcast"),
            season=pl.col("season_year").cast(pl.String),
        )
        .filter((pl.col("n_clear") >= 5) & (pl.col("n_overcast") >= 3))
    )
    return (
        alt.Chart(d.select("name", "season", "gap"))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=60)
        .encode(
            x=alt.X("season:O", title="Season-year (starting September)"),
            y=alt.Y("gap:Q", title="Clear minus overcast (MW)"),
            color=alt.value(ocf.DATA_BLUE),
        )
        .properties(width=250, height=110)
        .facet(
            facet=alt.Facet(
                "name:N", title=None, sort=[NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
            ),
            columns=2,
        )
        .properties(
            title=_title(
                text="Without any model, clear days already draw less power than overcast days",
                lines=[
                    (
                        "Mean power from 12:00 to 13:00 local in May to August: days with a "
                        "clear-sky index of 0.8 or more minus days of 0.3 or less (needs 5 and 3 "
                        "such days). More negative means more embedded solar."
                    )
                ],
            )
        )
    )


def figure_05() -> alt.TopLevelMixin:
    """Data quality: CAMS daylight hours dropped by month, and power coverage by season-year."""
    cams = pl.read_parquet(common.CAMS_DEMAND_POINTS_PATH).filter(
        (pl.col("site") == "D16") & (pl.col("clear_sky_ghi_w_m2") > 20)
    )
    dropped = (
        cams.with_columns(month=pl.col("time").dt.month())
        .group_by("month")
        .agg((pl.col("reliability") < common.MIN_RELIABILITY).mean().alias("share"))
        .sort("month")
    )
    low, high = float(dropped["share"].min()), float(dropped["share"].max())  # ty: ignore[invalid-argument-type]
    top = (
        alt.Chart(dropped)  # ty: ignore[unresolved-attribute]
        .mark_bar(color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("month:O", title="Calendar month"),
            y=alt.Y("share:Q", title="Share of daylight hours dropped", axis=alt.Axis(format="%")),
        )
        .properties(width=WIDTH, height=120)
    )
    frame = pl.read_parquet(common.FRAME_PATH).with_columns(
        season_year=(
            pl.col("time").dt.offset_by("-15m").dt.year()
            - (pl.col("time").dt.offset_by("-15m").dt.month() < 9).cast(pl.Int32)
        )
    )
    expected = (
        frame.filter(pl.col("site") == "D5").group_by("season_year").len().rename({"len": "all"})
    )
    coverage = (
        frame.filter(pl.col("site").is_in([*common.DIRECTIONAL_PRIMARIES, *common.FARMS]))
        .group_by("site", "season_year")
        .agg(pl.col("power").is_not_null().sum().alias("have"))
        .filter(pl.col("season_year") <= 2025)
        .join(expected, on="season_year")
        .with_columns(
            share=pl.col("have") / pl.col("all"),
            label=pl.col("site").replace({**NAMES, **common.FARM_NAMES}),
            season=pl.col("season_year").cast(pl.String),
        )
    )
    order = [NAMES[p] for p in common.DIRECTIONAL_PRIMARIES] + list(common.FARM_NAMES.values())
    bottom = (
        alt.Chart(coverage.select("label", "season", "share"))  # ty: ignore[unresolved-attribute]
        .mark_rect()
        .encode(
            x=alt.X("season:O", title="Season-year (starting September)"),
            y=alt.Y("label:N", sort=order, title=None),
            color=alt.Color(
                "share:Q",
                title="Share of half-hours with power",
                scale=alt.Scale(domain=[0, 1], range=["#f0f0f0", ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top", format="%"),
            ),
        )
        .properties(width=WIDTH, height=220)
    )
    return alt.vconcat(top, bottom).properties(
        title=_title(
            text=f"CAMS drops {low:.0%} to {high:.0%} of daylight hours; several series start late",
            lines=[
                (
                    "Top: share of daylight hours (clear-sky irradiance above 20 W/m²) with CAMS "
                    "reliability below 0.9 at one demand point. Bottom: share of half-hours in "
                    "each season-year with a power reading."
                )
            ],
        )
    )


def figure_06(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Method: u and clear-sky u at one Primary through a summer and a winter week."""
    d = prepared.filter(pl.col("site") == "D16")
    weeks = {
        "Summer week (June 2024)": (pl.date(2024, 6, 10), pl.date(2024, 6, 17)),
        "Winter week (December 2024)": (pl.date(2024, 12, 9), pl.date(2024, 12, 16)),
    }
    parts = []
    for label, (start, end) in weeks.items():
        parts.append(
            d.filter((pl.col("date") >= start) & (pl.col("date") < end))
            .select("time", "u", "u_clear")
            .with_columns(week=pl.lit(label))
        )
    long = (
        pl.concat(parts)
        .unpivot(
            on=["u", "u_clear"], index=["time", "week"], variable_name="shape", value_name="value"
        )
        .with_columns(
            shape=pl.col("shape").replace({"u": "All-sky (CAMS)", "u_clear": "Clear-sky"})
        )
    )
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(point=False, interpolate="step-after")
        .encode(
            x=alt.X("time:T", title="Time (UTC)"),
            y=alt.Y("value:Q", title="Unit PV shape (MW per MW of AC capacity)"),
            color=alt.Color(
                "shape:N",
                title=None,
                scale=alt.Scale(
                    domain=["All-sky (CAMS)", "Clear-sky"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=WIDTH, height=130)
        .facet(
            row=alt.Row("week:N", title=None),
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="The fit's PV shape is the output of one generic fleet under CAMS irradiance",
                lines=[
                    (
                        "Unit PV shape `u` at Donington (tilt 35°, facing south, 1.15 MW of "
                        "panels per MW of inverter). Daylight half-hours only, with CAMS "
                        "reliability of 0.9 or more."
                    )
                ],
            )
        )
    )


def _week_table(*, half_hourly: pl.DataFrame) -> pl.DataFrame:
    """Return one unit's first June week (Primary D16, farm A, 2024, CAMS at the substation)."""
    d = half_hourly.filter(
        (pl.col("primary") == "D16")
        & (pl.col("farm") == "A")
        & (pl.col("year") == 2024)
        & (pl.col("which") == "sub")
    ).sort("time")
    local = d.with_columns(local=pl.col("time").dt.convert_time_zone("Europe/London"))
    june = local.filter(pl.col("local").dt.month() == 6)
    first = june["local"].dt.date().min()
    week = june.filter(pl.col("local").dt.date() < first + pl.duration(days=7))
    return week.with_columns(
        day=(pl.col("local").dt.date() - first).dt.total_days() + 1,
        hour=pl.col("local").dt.hour() + pl.col("local").dt.minute() / 60.0,
    ).with_columns(x=pl.col("day") - 1 + pl.col("hour") / 24.0)


def figure_07(*, half_hourly: pl.DataFrame) -> alt.TopLevelMixin:
    """Building an injection: a June week of demand, Solar farm A scaled to 20%, and the sum."""
    week = _week_table(half_hourly=half_hourly)
    names = [
        "Observed demand",
        "Injected Solar farm A (negative)",
        "Sum: demand with the injection",
    ]
    long = week.select(
        "x",
        "day",
        pl.col("observed").alias(names[0]),
        (-pl.col("truth")).alias(names[1]),
        pl.col("observed_plus_injection").alias(names[2]),
    ).unpivot(index=["x", "day"], variable_name="series", value_name="MW")
    return (
        alt.Chart(long)  # ty: ignore[unresolved-attribute]
        .mark_line(point=False, interpolate="step-after", clip=True, aria=False)
        .encode(
            x=alt.X(
                "x:Q",
                title="Day (1 to 7)",
                axis=alt.Axis(values=[0, 1, 2, 3, 4, 5, 6], labelExpr="datum.value + 1"),
                scale=alt.Scale(domain=[0, 7]),
            ),
            y=alt.Y("MW:Q", title="Power (MW; negative is generation)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=names, range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_PURPLE]
                ),
                legend=alt.Legend(orient="top", columns=1, labelLimit=400),
            ),
            detail="day:N",
        )
        .properties(width=WIDTH, height=240)
        .properties(
            title=_title(
                text="An injection adds a real solar farm's shape, scaled, to a real demand series",
                lines=[
                    (
                        "One week of June 2024 at Donington. Solar farm A is scaled so its 99th "
                        "percentile is 20% of Donington's effective capacity. Daylight half-hours "
                        "only."
                    )
                ],
            )
        )
    )


def figure_08(*, half_hourly: pl.DataFrame) -> alt.TopLevelMixin:
    """Recovery: the recovered PV beside the truth, and demand plus PV plus residual."""
    week = _week_table(half_hourly=half_hourly)
    top_names = ["Injected (truth)", "Recovered by M1"]
    top_long = week.select(
        "x", "day", pl.col("truth").alias(top_names[0]), pl.col("estimate").alias(top_names[1])
    ).unpivot(index=["x", "day"], variable_name="series", value_name="MW")
    scale_x = alt.X(
        "x:Q",
        title="Day (1 to 7)",
        axis=alt.Axis(values=list(range(7)), labelExpr="datum.value + 1"),
        scale=alt.Scale(domain=[0, 7]),
    )
    top = (
        alt.Chart(top_long)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after", clip=True, aria=False)
        .encode(
            x=scale_x,
            y=alt.Y("MW:Q", title="PV output (MW)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=top_names, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
                legend=alt.Legend(orient="top"),
            ),
            detail="day:N",
        )
        .properties(width=WIDTH, height=150)
    )
    bottom_names = ["Demand with the injection", "Demand term", "Fitted PV (negative)", "Residual"]
    bottom_long = week.select(
        "x",
        "day",
        pl.col("observed_plus_injection").alias(bottom_names[0]),
        pl.col("demand").alias(bottom_names[1]),
        (-pl.col("pv")).alias(bottom_names[2]),
        pl.col("residual").alias(bottom_names[3]),
    ).unpivot(index=["x", "day"], variable_name="series", value_name="MW")
    bottom = (
        alt.Chart(bottom_long)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after", clip=True, aria=False)
        .encode(
            x=scale_x,
            y=alt.Y("MW:Q", title="Power (MW)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=bottom_names,
                    range=["#808080", ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN],
                ),
                legend=alt.Legend(orient="top"),
            ),
            detail="day:N",
        )
        .properties(width=WIDTH, height=170)
    )
    return (
        alt.vconcat(top, bottom)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text=(
                    "The recovered PV follows the injected shape; "
                    "the fit's three parts add to the series"
                ),
                lines=[
                    (
                        "Same week as the previous figure. Bottom: the demand term, minus the "
                        "fitted PV, plus the residual add to the demand with the injection (grey)."
                    )
                ],
            )
        )
    )


def figure_09(*, half_hourly: pl.DataFrame) -> alt.TopLevelMixin:
    """Recovered against true half-hourly PV, binned, at both irradiance points."""
    labels = {"sub": "CAMS at the substation", "own": "CAMS at the farm"}
    line = (
        alt.Chart(pl.DataFrame({"x": [0.0, 2.5], "y": [0.0, 2.5]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.BRAND_ORANGE)
        .encode(x="x:Q", y="y:Q")
    )
    panels = []
    for which, label in labels.items():
        sample = half_hourly.filter((pl.col("which") == which) & (pl.col("truth") > 0.01)).select(
            "truth", "estimate"
        )
        heat = (
            alt.Chart(sample)  # ty: ignore[unresolved-attribute]
            .mark_rect()
            .encode(
                x=alt.X("truth:Q", bin=alt.Bin(maxbins=40), title="Injected output (MW)"),
                y=alt.Y("estimate:Q", bin=alt.Bin(maxbins=40), title="Recovered output (MW)"),
                color=alt.Color(
                    "count():Q", scale=alt.Scale(type="sqrt", scheme="blues"), title="Half-hours"
                ),
            )
            .properties(width=240, height=240, title=label)
        )
        panels.append(alt.layer(heat, line))
    return alt.hconcat(*panels).properties(
        title=_title(
            text="Recovered output follows injected output but falls short at the peaks",
            lines=[
                (
                    "All 218 units at the 20% size: each cell counts half-hours; orange line is "
                    "1:1. Half-hours with injected output above 0.01 MW."
                )
            ],
        )
    )


def figure_10(*, injection: pl.DataFrame) -> alt.TopLevelMixin:
    """R by farm and by distance band; no Primary is named."""
    head = injection.filter(
        (pl.col("k") == common.HEADLINE_SIZE) & (pl.col("arm") == "M1")
    ).with_columns(
        farm_name=pl.col("farm").replace(common.FARM_NAMES),
        band=pl.col("km")
        .cut(  # ty: ignore[deprecated]
            [15.0, 20.0, 25.0, 30.0],
            labels=["under 15 km", "15 to 20 km", "20 to 25 km", "25 to 30 km", "over 30 km"],
        )
        .cast(pl.String),
        which=pl.col("which").replace({"sub": "CAMS at the substation", "own": "CAMS at the farm"}),
    )
    colour = alt.Color(
        "which:N",
        title=None,
        scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
        legend=alt.Legend(orient="top"),
    )
    by_farm = (
        alt.Chart(head.select("farm_name", "R", "which"))  # ty: ignore[unresolved-attribute]
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("farm_name:N", title=None),
            y=alt.Y("R:Q", title="Recovery ratio", scale=alt.Scale(domain=[0.4, 1.1])),
            color=colour,
            xOffset="which:N",
        )
        .properties(width=WIDTH, height=170)
    )
    by_band = (
        alt.Chart(head.select("band", "R", "which"))  # ty: ignore[unresolved-attribute]
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X(
                "band:N",
                title="Distance from Primary to farm",
                sort=["under 15 km", "15 to 20 km", "20 to 25 km", "25 to 30 km", "over 30 km"],
            ),
            y=alt.Y("R:Q", title="Recovery ratio", scale=alt.Scale(domain=[0.4, 1.1])),
            color=colour,
            xOffset="which:N",
        )
        .properties(width=WIDTH, height=170)
    )
    return alt.vconcat(by_farm, by_band).properties(
        title=_title(
            text="Recovery is similar across farms; Solar farm E is the worst case",
            lines=[
                (
                    "Boxes: quartiles over units; whiskers: range. Substations lie 16 to 24 km "
                    "from the farms on average, so the distance bands are narrow."
                )
            ],
        )
    )


def figure_11(*, injection: pl.DataFrame) -> alt.TopLevelMixin:
    """Detection limit: twice the standard error of the fitted capacity against injected size."""
    d = (
        injection.filter((pl.col("arm") == "M1") & (pl.col("which") == "sub"))
        .with_columns(two_se=2.0 * pl.col("se_ratio") * pl.col("k"))
        .group_by("k")
        .agg(
            pl.col("two_se").median(),
            pl.col("two_se").quantile(0.1).alias("lo"),
            pl.col("two_se").quantile(0.9).alias("hi"),
        )
    )
    x = alt.X(
        "k:Q",
        title="Injected capacity (share of the Primary's effective capacity)",
        axis=alt.Axis(format="%", values=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),
        scale=alt.Scale(domain=[0.0, 0.4]),
    )
    line = (
        alt.Chart(pl.DataFrame({"k": [0.0, 0.4], "y": [0.0, 0.4]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#999999", strokeDash=[4, 3])
        .encode(x=x, y=alt.Y("y:Q", scale=alt.Scale(domain=[0.0, 0.1])))
    )
    band = (
        alt.Chart(d).mark_area(opacity=0.25, color=ocf.DATA_BLUE).encode(x=x, y="lo:Q", y2="hi:Q")  # ty: ignore[unresolved-attribute]
    )
    mid = (
        alt.Chart(d)
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y(
                "two_se:Q",
                title="Twice the standard error (share of capacity)",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0.0, 0.1]),
            ),
        )
    )
    return alt.layer(line, band, mid).properties(
        width=WIDTH,
        height=220,
        title=_title(
            text=(
                "A change of about 2.5% of capacity per season-year is the smallest the fit can see"
            ),
            lines=[
                (
                    "Blue: median over units of twice the day-clustered standard error (band: "
                    "10th to 90th percentile). Dashed line: injected capacity. Where the blue "
                    "line meets the dashed line, an injection is just twice its standard error."
                )
            ],
        ),
    )


def figure_12(*, injection: pl.DataFrame) -> alt.TopLevelMixin:
    """Summer and winter R for injected PV, the expectation K2 uses."""
    d = injection.filter(
        (pl.col("k") == common.HEADLINE_SIZE) & (pl.col("arm") == "M1")
    ).with_columns(
        which=pl.col("which").replace({"sub": "CAMS at the substation", "own": "CAMS at the farm"})
    )
    dots = (
        alt.Chart(d.select("R_summer", "R_winter", "which"))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=18, opacity=0.5)
        .encode(
            x=alt.X(
                "R_summer:Q",
                title="Recovery ratio, April to September",
                scale=alt.Scale(domain=[0.0, 1.1]),
            ),
            y=alt.Y(
                "R_winter:Q",
                title="Recovery ratio, October to March",
                scale=alt.Scale(domain=[0.0, 1.1]),
            ),
            color=alt.Color(
                "which:N",
                title=None,
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=300, height=300)
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.1], "y": [0.0, 1.1]}))  # ty: ignore[unresolved-attribute]
        .mark_line(color="#999999")
        .encode(x="x:Q", y="y:Q")
    )
    return alt.layer(diagonal, dots).properties(
        title=_title(
            text="True PV is recovered equally well in summer and winter",
            lines=["Each dot is one injected unit (20% size). Grey line: equal ratios."],
        )
    )


def figure_13(*, real: pl.DataFrame) -> alt.TopLevelMixin:
    """Real directional Primaries: fitted capacity per season-year with 95% intervals."""
    d = real.filter(pl.col("role") == "directional").with_columns(
        name=pl.col("site").replace(NAMES),
        low=pl.col("c") - 1.96 * pl.col("se"),
        high=pl.col("c") + 1.96 * pl.col("se"),
        season=pl.col("year").cast(pl.String),
    )
    base = alt.Chart(d.select("name", "season", "c", "low", "high")).encode(
        x=alt.X("season:O", title="Season-year (starting September)")
    )
    chart = alt.layer(
        base.mark_rule(color=ocf.DATA_BLUE).encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("low:Q", title="Fitted PV capacity (MW)"), y2="high:Q"
        ),
        base.mark_point(filled=True, color=ocf.DATA_BLUE, size=40).encode(y="c:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=250, height=110)
    return chart.facet(
        facet=alt.Facet(
            "name:N", title=None, sort=[NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
        ),
        columns=2,
    ).properties(
        title=_title(
            text="The fitted capacity is 0.9 to 3.2 MW per Primary and mostly rises",
            lines=[
                (
                    "M1 with CAMS at the substation. Dot: the 99th percentile of the fitted PV "
                    "component in MW. Line: 95% interval (1.96 day-clustered standard errors). "
                    "The Embedded Capacity Register comparison was not run."
                )
            ],
        )
    )


def figure_14(*, checks: pl.DataFrame, placebo: pl.DataFrame) -> alt.TopLevelMixin:
    """K1: real capacity against 20 placebo values per Primary, latest season-year."""
    latest = checks.sort("year").group_by("site").last().select("site", "year", "c", "k1")
    d = placebo.join(latest.select("site", "year"), on=["site", "year"]).with_columns(
        name=pl.col("site").replace(NAMES)
    )
    real = latest.with_columns(name=pl.col("site").replace(NAMES), kind=pl.lit("Real days"))
    order = [NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
    placebo_dots = (
        alt.Chart(d.select("name", "c"))
        .mark_circle(size=25, opacity=0.5, color="#808080")
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("name:N", sort=order, title=None),
            x=alt.X("c:Q", title="Fitted PV capacity (MW)"),
        )
    )
    real_dots = (
        alt.Chart(real.select("name", "c", "k1"))
        .mark_point(filled=True, size=100)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("name:N", sort=order),
            x="c:Q",
            color=alt.Color(
                "k1:N",
                title="K1 passes",
                scale=alt.Scale(domain=[True, False], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(orient="top"),
            ),
        )
    )
    passing = int(real["k1"].sum())
    return alt.layer(placebo_dots, real_dots).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text=(
                "Real days give a clearly larger capacity than shuffled days "
                f"at {passing} of 8 Primaries"
            ),
            lines=[
                (
                    "Grey: 20 fits with the days of each month shuffled (so the sky no longer "
                    "matches the day). Coloured: the real fit. Latest season-year (2025/26) of "
                    "each Primary."
                )
            ],
        ),
    )


def figure_15(*, checks: pl.DataFrame, real: pl.DataFrame) -> alt.TopLevelMixin:
    """K2: winter against summer capacity, directional coloured and MVA grey."""
    d = real.filter(pl.col("role").is_in(["directional", "mva"])).with_columns(
        meter=pl.when(pl.col("role") == "directional")
        .then(pl.lit("Directional (MW)"))
        .otherwise(pl.lit("Non-directional (MVA)")),
    )
    dots = (
        alt.Chart(d.select("c_summer", "c_winter", "meter"))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=40, opacity=0.7)
        .encode(
            x=alt.X("c_summer:Q", title="Capacity fitted to April to September (MW)"),
            y=alt.Y("c_winter:Q", title="Capacity fitted to October to March (MW)"),
            color=alt.Color(
                "meter:N",
                title=None,
                scale=alt.Scale(
                    range=[ocf.DATA_BLUE, "#999999"],
                    domain=["Directional (MW)", "Non-directional (MVA)"],
                ),
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=330, height=330)
    )
    top = float(d["c_summer"].max()) * 1.05  # ty: ignore[invalid-argument-type]
    top = max(top, 1.0)
    lines = pl.DataFrame(
        {"x": [0.0, top], "one": [0.0, top], "lo": [0.0, top * 0.85], "hi": [0.0, top * 1.15]}
    )
    layers = [
        alt.Chart(lines).mark_line(color="#666666").encode(x="x:Q", y="one:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(lines).mark_line(color="#999999", strokeDash=[4, 3]).encode(x="x:Q", y="lo:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(lines).mark_line(color="#999999", strokeDash=[4, 3]).encode(x="x:Q", y="hi:Q"),  # ty: ignore[unresolved-attribute]
    ]
    return alt.layer(*layers, dots).properties(
        title=_title(
            text=(
                "Directional Primaries have similar summer and winter capacity; "
                "MVA Primaries do not"
            ),
            lines=[
                (
                    "Every Primary-year. Solid line: equal; dashed lines: winter 15% above or "
                    "below summer. Negative values are a Primary whose meter folds PV into "
                    "positive demand."
                )
            ],
        )
    )


def figure_16(*, checks: pl.DataFrame) -> alt.TopLevelMixin:
    """K3: capacity with the cloud term over capacity without it, per Primary-year."""
    expected = float(checks["expected_k3"][0])
    d = checks.with_columns(
        name=pl.col("site").replace(NAMES),
        season=pl.col("year").cast(pl.String),
        group=pl.when(pl.col("site") == "D5")
        .then(pl.lit("Kirton"))
        .otherwise(pl.lit("Other Primaries")),
    )
    band = (
        alt.Chart(pl.DataFrame({"a": [expected - 0.2], "b": [expected + 0.2]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_GREEN_LIGHT, opacity=0.6)
        .encode(y="a:Q", y2="b:Q")
    )
    dots = (
        alt.Chart(d.select("season", "k3_statistic", "group", "name"))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=50, opacity=0.8)
        .encode(
            x=alt.X("season:O", title="Season-year (starting September)"),
            y=alt.Y("k3_statistic:Q", title="Capacity with cloud term / capacity without"),
            color=alt.Color(
                "group:N",
                title=None,
                scale=alt.Scale(
                    domain=["Kirton", "Other Primaries"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["name"],
        )
        .properties(width=WIDTH, height=260)
    )
    return alt.layer(band, dots).properties(
        title=_title(
            text=(
                "A cloud-demand term cuts Kirton's capacity by 20% to 60%; "
                "most others stay in the band"
            ),
            lines=[
                (
                    f"Green band: the injected units' median ({expected:.2f}) plus or minus 0.20. "
                    "A ratio below the band means part of the fitted PV is demand "
                    "that follows cloud."
                )
            ],
        )
    )


def figure_17(*, real: pl.DataFrame) -> alt.TopLevelMixin:
    """K4: weekday against weekend capacity."""
    d = real.filter(pl.col("role") == "directional").with_columns(
        name=pl.col("site").replace(NAMES)
    )
    top = float(max(d["c_weekday"].max(), d["c_weekend"].max())) * 1.05  # ty: ignore[invalid-argument-type]
    dots = (
        alt.Chart(d.select("c_weekday", "c_weekend", "name"))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=40, opacity=0.8, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("c_weekday:Q", title="Capacity fitted to weekdays (MW)"),
            y=alt.Y("c_weekend:Q", title="Capacity fitted to weekends and bank holidays (MW)"),
            tooltip=["name"],
        )
        .properties(width=320, height=320)
    )
    lines = pl.DataFrame(
        {"x": [0.0, top], "one": [0.0, top], "lo": [0.0, top * 0.8], "hi": [0.0, top * 1.2]}
    )
    layers = [
        alt.Chart(lines).mark_line(color="#666666").encode(x="x:Q", y="one:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(lines).mark_line(color="#999999", strokeDash=[4, 3]).encode(x="x:Q", y="lo:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart(lines).mark_line(color="#999999", strokeDash=[4, 3]).encode(x="x:Q", y="hi:Q"),  # ty: ignore[unresolved-attribute]
    ]
    return alt.layer(*layers, dots).properties(
        title=_title(
            text="Capacity fitted to weekends matches weekdays, as the sun does not know the day",
            lines=["Every directional Primary-year. Solid line: equal; dashed: 20% apart."],
        )
    )


def figure_18(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """MVA folding: Stickney's absolute midday power against CAMS irradiance."""
    d = prepared.filter(
        (pl.col("site") == "D12")
        & (pl.col("season_year") == 2025)
        & pl.col("power").is_not_null()
        & pl.col("half_hour").is_between(22, 27)
        & (pl.col("off") == 0)
    ).with_columns(season=_season(month=pl.col("month")))
    chart = (
        alt.Chart(d.select("power", "ghi_u", "season"))
        .mark_circle(size=14, opacity=0.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("ghi_u:Q", title="CAMS global horizontal irradiance (W/m²)"),
            y=alt.Y("power:Q", title="Substation power as metered (MVA, always positive)"),
            color=alt.Color(
                "season:N", scale=SEASON_SCALE, title=None, legend=alt.Legend(orient="top")
            ),
        )
    )
    return chart.properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="At Stickney, whose meter reads MVA, strong sunshine pushes power back up",
            lines=[
                (
                    "Weekday half-hours from 11:00 to 14:00 local in 2025/26. An MVA meter cannot "
                    "read export, so output above demand shows as positive power and the sign of "
                    "the slope is lost: the V shape is the fold."
                )
            ],
        ),
    )


def figure_19(*, real: pl.DataFrame) -> alt.TopLevelMixin:
    """BSP identity (descriptive): capacity at a bulk supply point against the parts below it."""
    out = []
    for bsp, members in {
        "D17": ["D5", "D14", "D15", "D16", "D18"],
        "D13": ["D9", "D10", "D11"],
    }.items():
        own = real.filter(pl.col("site") == bsp).select("year", pl.col("c").alias("whole"))
        prim = (
            real.filter(pl.col("site").is_in(members))
            .group_by("year")
            .agg(pl.col("c").sum().alias("Directional Primaries below"))
        )
        farm = (
            real.filter(pl.col("site").is_in(common.BSP_FARMS[bsp]))
            .group_by("year")
            .agg(pl.col("c").sum().alias("Metered solar farms below"))
        )
        out.append(
            own.join(prim, on="year", how="left")
            .join(farm, on="year", how="left")
            .with_columns(
                parts=pl.col("Directional Primaries below").fill_null(0.0)
                + pl.col("Metered solar farms below").fill_null(0.0),
                bsp=pl.lit(NAMES[bsp]),
            )
        )
    d = pl.concat(out)
    dots = (
        alt.Chart(d.select("bsp", "whole", "parts", "year"))
        .mark_circle(size=60, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("parts:Q", title="Sum of Primaries' and farms' capacity below (MW)"),
            y=alt.Y("whole:Q", title="Capacity fitted to the bulk supply point (MW)"),
            color="bsp:N",
            tooltip=["year"],
        )
        .properties(width=WIDTH, height=260)
    )
    return dots.properties(
        title=_title(
            text="Bulk supply points show more solar than the parts we fitted, as expected",
            lines=[
                (
                    "Descriptive only. Each dot is one season-year. The bulk supply point also "
                    "carries the MVA Primaries and the unmetered solar below the directional "
                    "ones, so dots lie above the sum."
                )
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    injection = pl.read_parquet(common.INJECTION_PATH)
    half_hourly = pl.read_parquet(common.INJECTION_HALF_HOURLY_PATH)
    real = pl.read_parquet(common.REAL_PATH)
    placebo = pl.read_parquet(common.PLACEBO_PATH)
    checks = pl.read_parquet(common.WORK_DIR / "report_checks.parquet")
    prepared = _prepared()
    table = report.UnitTable(injection=injection, k=common.HEADLINE_SIZE)
    rows = report.contrasts(table=table)
    long_units = injection.filter(pl.col("k") == common.HEADLINE_SIZE).with_columns(
        config=pl.col("arm") + "_" + pl.col("which")
    )
    figures: dict[str, alt.TopLevelMixin] = {
        "figure_01": figure_01(rows=rows, units=long_units.select("config", "R")),
        "figure_02": figure_02(prepared=prepared),
        "figure_03": figure_03(prepared=prepared),
        "figure_04": figure_04(prepared=prepared),
        "figure_05": figure_05(),
        "figure_06": figure_06(prepared=prepared),
        "figure_07": figure_07(half_hourly=half_hourly),
        "figure_08": figure_08(half_hourly=half_hourly),
        "figure_09": figure_09(half_hourly=half_hourly),
        "figure_10": figure_10(injection=injection),
        "figure_11": figure_11(injection=injection),
        "figure_12": figure_12(injection=injection),
        "figure_13": figure_13(real=real),
        "figure_14": figure_14(checks=checks, placebo=placebo),
        "figure_15": figure_15(checks=checks, real=real),
        "figure_16": figure_16(checks=checks),
        "figure_17": figure_17(real=real),
        "figure_18": figure_18(prepared=prepared),
        "figure_19": figure_19(real=real),
    }
    for name, chart in figures.items():
        chart.save(common.CHART_DIR / f"{name}.svg")
        print("wrote", name)


if __name__ == "__main__":
    main()
