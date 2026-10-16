"""Draw experiment 00040's 15 figures from the saved results, as SVG (raw data first).

Every farm is "Solar farm A" to "Solar farm F"; no chart pairs a farm label with a name or an
identifier, and no chart carries calendar dates beside a farm's output. Substations are named. The
SVGs are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
        uv run --with holidays python ml-experiments/00040-pv-capacity-error-bars/e00040_charts.py
"""

import textwrap
from typing import Final

import altair as alt
import e00040_common as common
import e00040_detector as detector
import e00040_inputs as inputs
import e00040_model as model
import e00040_score as score
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 560
WRAP: Final[int] = 88
NAMES: Final[dict[str, str]] = common.PRIMARY_NAMES
ORDER: Final[list[str]] = [NAMES[p] for p in common.DIRECTIONAL_PRIMARIES]
SCOPE: Final[str] = (
    "8 directional Primary substations; 6 solar farms (A to F); season-years run September to "
    "August, and 2025/26 is labelled 2025."
)
SHAPE_COLOURS: Final[dict[str, str]] = {
    "lowsun": ocf.DATA_BLUE,
    "dayL": ocf.BRAND_ORANGE,
    "evening": ocf.DATA_PURPLE,
    "threshold": ocf.DATA_GREEN,
    "mixture": ocf.DATA_SKY,
    "flat": ocf.DATA_BLUE_LIGHT,
    "winter": ocf.DATA_GREEN_LIGHT,
    "sunload": "#000000",
}
SHAPE_SCALE: Final[alt.Scale] = alt.Scale(
    domain=list(SHAPE_COLOURS), range=list(SHAPE_COLOURS.values())
)
SEASON_SCALE: Final[alt.Scale] = alt.Scale(
    domain=["April to September", "October to March"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
)


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _rule(*, value: float, axis: str = "x", colour: str = "#999999") -> alt.Chart:
    return (
        alt.Chart(pl.DataFrame({"v": [value]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(color=colour, strokeDash=[4, 3])
        .encode(**{axis: "v:Q"})
    )


def _season(*, month: pl.Expr) -> pl.Expr:
    return (
        pl.when(month.is_in(common.SUMMER_MONTHS))
        .then(pl.lit("April to September"))
        .otherwise(pl.lit("October to March"))
    )


def _full_rows() -> pl.DataFrame:
    """Return every row (day and night) of the directional Primaries with calendar columns."""
    frame = pl.read_parquet(common.FRAME_PATH).filter(
        pl.col("site").is_in(common.DIRECTIONAL_PRIMARIES)
    )
    return (
        frame.with_columns(
            local=pl.col("time").dt.offset_by("-15m").dt.convert_time_zone("Europe/London")
        )
        .with_columns(
            half_hour=pl.col("local").dt.hour().cast(pl.Int64) * 2
            + pl.col("local").dt.minute().cast(pl.Int64) // 30,
            off=(
                (pl.col("local").dt.weekday() >= 6)
                | pl.col("local").dt.date().is_in(model.bank_holidays().implode())
            ).cast(pl.Int64),
            month=pl.col("local").dt.month().cast(pl.Int64),
            date=pl.col("local").dt.date(),
            season_year=pl.col("local").dt.year() - (pl.col("local").dt.month() < 9).cast(pl.Int32),
            cloud=1.0
            - (pl.col("ghi_u") / pl.col("ghi_u_clear")).clip(upper_bound=1.0).fill_nan(0.0),
        )
        .filter(pl.col("power").is_not_null())
        .with_columns(name=pl.col("site").replace(NAMES))
    )


def figure_01(*, real: pl.DataFrame) -> alt.TopLevelMixin:
    """Headline: 2025/26 capacity per Primary with its interval and bound."""
    latest = real.sort("year").group_by("site").last()
    d = latest.with_columns(
        name=pl.col("site").replace(NAMES),
        status=pl.when(pl.col("flagged")).then(pl.lit("flagged")).otherwise(pl.lit("clean")),
        bound_low=pl.col("headline") * (1 - pl.col("B")),
        bound_high=pl.col("headline") * (1 + pl.col("B")),
    )
    y = alt.Y("name:N", sort=ORDER, title=None, axis=alt.Axis(labelLimit=200))
    colour = alt.Color(
        "status:N",
        scale=alt.Scale(domain=["clean", "flagged"], range=[ocf.DATA_BLUE, "#808080"]),
        legend=alt.Legend(orient="top", title=None),
    )
    x_title = "Embedded solar capacity (MW)"
    bound = (
        alt.Chart(d)
        .mark_rule(strokeWidth=9, opacity=0.25)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("bound_low:Q", title=x_title), x2="bound_high:Q", y=y, color=colour
        )
    )
    interval = (
        alt.Chart(d)
        .mark_rule(strokeWidth=2)
        .encode(x="headline_low:Q", x2="headline_high:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
    )
    dot = (
        alt.Chart(d).mark_point(filled=True, size=70).encode(x="headline:Q", y=y, color=colour)  # ty: ignore[unresolved-attribute]
    )
    spread = (
        alt.Chart(d.filter(pl.col("flagged")))
        .mark_rule(strokeWidth=3, color="#000000", strokeDash=[2, 2])
        .encode(x="range_low:Q", x2="range_high:Q", y=y)  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(bound, interval, dot, spread).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="Seven of eight Primaries are unflagged, but the bound is at least 18%",
            lines=[
                "Dot: capacity from the daily fit divided by its injected recovery ratio. Thin "
                "line: 95% interval from resampling 7-day blocks. Wide band: the bound B on the "
                "capacity from the misfit test. Dashed black line: range over the library of "
                "confound shapes at a flagged Primary. " + SCOPE,
            ],
        ),
    )


def figure_02(*, full: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw: power on overcast days minus power on clear days, by half-hour, two Primaries."""
    daylight = full.filter(pl.col("zenith") < common.MAX_ZENITH_DEG)
    index = daylight.group_by("site", "date").agg(
        k=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum(), n=pl.len()
    )
    index = index.filter(pl.col("n") >= common.MIN_DAILY_ROWS)
    d = (
        daylight.join(index, on=["site", "date"])
        .filter(pl.col("site").is_in(["D5", "D16"]) & (pl.col("off") == 0))
        .with_columns(
            season=_season(month=pl.col("month")),
            sky=pl.when(pl.col("k") <= 0.3)
            .then(pl.lit("overcast"))
            .when(pl.col("k") >= 0.8)
            .then(pl.lit("clear"))
            .otherwise(pl.lit("other")),
        )
        .filter(pl.col("sky") != "other")
        .group_by("name", "season", "sky", "half_hour")
        .agg(pl.col("power").mean(), pl.len().alias("n"))
        .filter(pl.col("n") >= 15)
        .pivot(on="sky", index=["name", "season", "half_hour"], values="power")
        .with_columns(difference=pl.col("overcast") - pl.col("clear"), hour=pl.col("half_hour") / 2)
    )
    return (
        alt.Chart(d.select("name", "season", "hour", "difference"))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X(
                "hour:Q",
                title="Local hour of the day (daylight hours only)",
                scale=alt.Scale(domain=[4, 20]),
                axis=alt.Axis(values=[4, 8, 12, 16, 20]),
            ),
            y=alt.Y("difference:Q", title="Overcast minus clear (MW)"),
            color=alt.Color("name:N", title=None, legend=alt.Legend(orient="top")),
        )
        .properties(width=WIDTH, height=130)
        .facet(row=alt.Row("season:N", title=None))
        .properties(
            title=_title(
                text="Kirton's overcast-minus-clear power is larger and flatter than Donington's",
                lines=[
                    (
                        "Mean weekday power on overcast days (daily clear-sky index at most 0.3) "
                        "minus clear days (at least 0.8), by local half-hour, all years. Solar "
                        "generation gives a dip shaped like the clear-sky curve and a cloud-driven "
                        "load gives a flat offset. Cells with under 15 days are omitted."
                    )
                ],
            )
        )
    )


def figure_03(*, full: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw: evening power against the day's mean cloud, all Primaries."""
    day_cloud = (
        full.filter(pl.col("zenith") < common.MAX_ZENITH_DEG)
        .group_by("site", "date")
        .agg(L=pl.col("cloud").mean(), n=pl.len())
        .filter(pl.col("n") >= common.MIN_DAILY_ROWS)
    )
    evening = (
        full.filter(
            (pl.col("zenith") >= 100)
            & pl.col("half_hour").is_between(36, 44)
            & pl.col("temperature_2m").is_not_null()
        )
        .group_by("site", "name", "date", "month", "off")
        .agg(power=pl.col("power").mean(), temperature=pl.col("temperature_2m").mean())
        .join(day_cloud, on=["site", "date"])
        .with_columns(
            residual=pl.col("power") - pl.col("power").mean().over("site", "month", "off"),
            bin=(pl.col("L") * 5).floor().clip(0, 4) / 5 + 0.1,
        )
        .group_by("name", "bin")
        .agg(pl.col("residual").mean(), pl.len().alias("n"))
        .filter(pl.col("n") >= 20)
    )
    return (
        alt.Chart(evening.select("name", "bin", "residual"))  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("bin:Q", title="Day's mean cloud (0 clear, 1 overcast)"),
            y=alt.Y("residual:Q", title="MW from month mean"),
        )
        .properties(width=250, height=110)
        .facet(facet=alt.Facet("name:N", sort=ORDER, title=None), columns=2)
        .properties(
            title=_title(
                text="Evening demand rises with the day's cloud at some Primaries",
                lines=[
                    (
                        "Mean power from 18:00 to 22:30 local, sun at least 10° below the horizon, "
                        "minus the mean of the same month and day type, in 5 bins of the "
                        "day's mean "
                        ""
                        "cloud response. A cloud response after dark cannot be solar generation."
                    )
                ],
            )
        )
    )


def figure_04(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Raw: Kirton midday power against CAMS irradiance, by season, 2019/20 and 2025/26."""
    d = (
        prepared.filter(
            (pl.col("site") == "D5")
            & pl.col("season_year").is_in([2019, 2025])
            & pl.col("power").is_not_null()
            & pl.col("half_hour").is_between(24, 25)
            & (pl.col("off") == 0)
        )
        .with_columns(season=_season(month=pl.col("month")), year=pl.col("season_year").cast(str))
        .select(pl.col("power").round(2), pl.col("ghi_u").round(0), "season", "year")
    )
    return (
        alt.Chart(d)
        .mark_circle(size=14, opacity=0.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("ghi_u:Q", title="CAMS global horizontal irradiance (W/m²)"),
            y=alt.Y("power:Q", title="Substation power (MW; negative is export)"),
            color=alt.Color(
                "season:N", scale=SEASON_SCALE, title=None, legend=alt.Legend(orient="top")
            ),
        )
        .properties(width=250, height=190)
        .facet(column=alt.Column("year:N", title="Season-year starting in"))
        .properties(
            title=_title(
                text="At Kirton, power falls with irradiance in winter as well as in summer",
                lines=[
                    (
                        "Each dot is one weekday half-hour from 12:00 to 13:00 local. Solar "
                        "generation would give a steeper fall in summer than in winter, for the "
                        "same irradiance."
                    )
                ],
            )
        )
    )


def figure_05(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Method: the confound shapes and `u` through one summer day and one winter day."""
    import e00040_arms as arms

    rows = inputs.primary_rows(prepared=prepared).filter(
        (pl.col("site") == "D16") & (pl.col("season_year") == 2024)
    )
    window = model.window_from(rows=rows)
    shapes = {**arms.confound_shapes(window=window)}
    dates = rows["date"].to_numpy()
    cloud = window.cloud
    chosen = {}
    for label, months in (("summer day", common.SUMMER_MONTHS), ("winter day", (11, 12, 1, 2))):
        best = None
        for day in np.unique(dates[np.isin(window.month, months)]):
            mask = dates == day
            if mask.sum() < common.MIN_DAILY_ROWS:
                continue
            late = mask & (window.half_hour >= window.half_hour[mask].max() - 5)
            spread = float(np.std(cloud[mask]) * np.mean(cloud[late]))
            if best is None or spread > best[0]:
                best = (spread, day)
        assert best is not None
        chosen[label] = best[1]
    records = []
    for label, day in chosen.items():
        mask = dates == day
        for name, values in {**shapes, "u (PV shape)": window.u}.items():
            for hour, value in zip(window.half_hour[mask] / 2, values[mask], strict=True):
                records.append({"day": label, "shape": name, "hour": hour, "value": float(value)})
    d = pl.DataFrame(records)
    return (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("hour:Q", title="Local hour", axis=alt.Axis(values=[6, 12, 18])),
            y=alt.Y("value:Q", title="Load (per unit)"),
            color=alt.Color(
                "day:N",
                scale=alt.Scale(
                    domain=["summer day", "winter day"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
                ),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=170, height=100)
        .facet(
            facet=alt.Facet(
                "shape:N", title=None, sort=[*common.SHAPES[:-1], "u (PV shape)", "sunload"]
            ),
            columns=3,
        )
        .properties(
            title=_title(
                text="Eight cloud-driven loads and the PV shape through two days",
                lines=[
                    (
                        "Each shape is divided by its mean over the season-year's daylight "
                        "rows. The "
                        ""
                        "days are the summer day and the winter day with the most variable "
                        "cloud at "
                        ""
                        "Donington in 2024/25. The load sunload has the shape of u."
                    )
                ],
            )
        )
    )


def figure_06(*, real_vectors: pl.DataFrame) -> alt.TopLevelMixin:
    """Method: leakage heat map, arm by shape."""
    d = (
        real_vectors.filter(pl.col("kind") != "real")
        .unpivot(index=["kind"], on=list(common.ALL_FITS), variable_name="arm", value_name="c")
        .group_by("kind", "arm")
        .agg(pl.col("c").median())
        .with_columns(label=(pl.col("c").round(2) + 0.0).cast(pl.String))
    )
    base = alt.Chart(d).encode(
        x=alt.X("arm:N", sort=list(common.ALL_FITS), title="Fit"),
        y=alt.Y("kind:N", sort=list(common.SHAPES), title="Confound shape"),
    )
    rect = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "c:Q",
            scale=alt.Scale(domain=[-7, 0, 7], range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top", title="MW per MW of load"),
        )
    )
    text = base.mark_text(fontSize=10).encode(text="label:N")  # ty: ignore[unresolved-attribute]
    return (rect + text).properties(
        width=WIDTH,
        height=230,
        title=_title(
            text="Every cloud-driven load leaks into the half-hourly capacity",
            lines=[
                (
                    "Median over the 41 Primary-years of the capacity each fit assigns to a "
                    "load of "
                    ""
                    "1 MW mean over daylight rows. Positive means the load looks like solar "
                    "generation; zero means the fit contains the shape. Only sunload leaks "
                    "the same amount into every fit, with a negative sign."
                )
            ],
        ),
    )


def figure_07(*, prepared: pl.DataFrame) -> alt.TopLevelMixin:
    """Method: daily sums of `u` and `L` after the cells are removed."""
    rows = inputs.primary_rows(prepared=prepared).filter(
        (pl.col("site") == "D16") & (pl.col("season_year") == 2024)
    )
    window = model.window_from(rows=rows)
    y_daily, daily = model.daily_sums(window=window, y=window.cloud)
    u = model.demean(values=daily.u, cells=daily.cells)
    cloud = model.demean(values=y_daily, cells=daily.cells)
    correlations = []
    for primary in common.DIRECTIONAL_PRIMARIES:
        for _, r in inputs.season_year_rows(
            rows=inputs.primary_rows(prepared=prepared).filter(pl.col("site") == primary)
        ):
            w = model.window_from(rows=r)
            yd, dw = model.daily_sums(window=w, y=w.cloud)
            correlations.append(
                np.corrcoef(
                    model.demean(values=dw.u, cells=dw.cells),
                    model.demean(values=yd, cells=dw.cells),
                )[0, 1]
            )
    d = pl.DataFrame({"u": u, "L": cloud})
    return (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_circle(size=18, opacity=0.6, color=ocf.DATA_BLUE)
        .encode(
            x=alt.X("u:Q", title="Daily sum of PV shape u, minus cell mean"),
            y=alt.Y("L:Q", title="Daily sum of cloud L, minus cell mean"),
        )
        .properties(
            width=WIDTH,
            height=240,
            title=_title(
                text="The daily fit cannot tell solar generation from a cloud load",
                lines=[
                    (
                        f"Donington, 2024/25; each dot is one day. Across all {len(correlations)} "
                        f"Primary-years the correlation runs from {min(correlations):.3f} to "
                        f"{max(correlations):.3f}, so a daily cloud term has no room."
                    )
                ],
            ),
        )
    )


def figure_08(*, units: pl.DataFrame) -> alt.TopLevelMixin:
    """V1: capacity over truth by solar farm, leave-one-farm-out."""
    pure = units.filter(pl.col("shape") == "pure").with_columns(
        farm_name=pl.lit("Solar farm ") + pl.col("farm")
    )
    summary = pure.group_by("farm_name").agg(
        pl.col("median_ratio").median().alias("median"),
        pl.col("power").mean().alias("false_positive_rate"),
    )
    y = alt.Y("farm_name:N", title=None, sort=[f"Solar farm {f}" for f in common.FARMS])
    band = (
        alt.Chart(pl.DataFrame({"a": [0.9], "b": [1.1]}))  # ty: ignore[unresolved-attribute]
        .mark_rect(color=ocf.DATA_GREEN_LIGHT, opacity=0.6)
        .encode(x=alt.X("a:Q", scale=alt.Scale(domain=[0.5, 1.4])), x2="b:Q")
    )
    dots = (
        alt.Chart(pure.select("farm_name", "median_ratio"))
        .mark_circle(size=14, opacity=0.35, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("median_ratio:Q", title="Recovered over true capacity (leave-one-farm-out)"),
            y=y,
        )
    )
    med = (
        alt.Chart(summary)
        .mark_point(filled=True, size=90, color=ocf.BRAND_ORANGE)
        .encode(x="median:Q", y=y)  # ty: ignore[unresolved-attribute]
    )
    text = (
        alt.Chart(summary.with_columns(label=(pl.col("false_positive_rate") * 100).round(1)))
        .mark_text(align="left", dx=8, fontSize=10)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.value(WIDTH - 90),
            y=y,
            text=alt.Text("label:Q", format=".1f"),
        )
    )
    return alt.layer(band, dots, med, text).properties(
        width=WIDTH,
        height=170,
        title=_title(
            text="Solar farm E does not transfer to the other farms: its ratios fall 17% short",
            lines=[
                (
                    "Dots: the median over 200 noise draws for one unit of the farm injected at a "
                    "Primary, with the ratio vector taken from the other five farms. Orange: the "
                    "median over the farm's units. Green band: 0.9 to 1.1. The number at the right "
                    "is the farm's false-positive rate in percent."
                )
            ],
        ),
    )


def figure_09(*, units: pl.DataFrame) -> alt.TopLevelMixin:
    """V2: power against amplitude, one line per shape."""
    latest = units.group_by("site").agg(pl.col("year").max().alias("latest"))
    d = (
        units.join(latest, on="site")
        .filter((pl.col("year") == pl.col("latest")) & (pl.col("shape") != "pure"))
        .group_by("site", "shape", "amplitude")
        .agg(pl.col("power").mean())
        .group_by("shape", "amplitude")
        .agg(pl.col("power").median())
    )
    x = alt.X(
        "amplitude:Q",
        title="Confound amplitude (share of true capacity added to the half-hourly fit)",
        scale=alt.Scale(domain=[0.1, 1.0]),
        axis=alt.Axis(values=list(common.AMPLITUDES), format=".2~f"),
    )
    lines = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("power:Q", title="Share of draws flagged", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "shape:N", scale=SHAPE_SCALE, legend=alt.Legend(orient="top", title=None)
            ),
        )
    )
    rule_y = (
        alt.Chart(pl.DataFrame({"y": [common.MIN_POWER]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#999999")
        .encode(y="y:Q")
    )
    rule_x = (
        alt.Chart(pl.DataFrame({"x": [common.POWER_AMPLITUDE]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 3], color="#999999")
        .encode(x="x:Q")
    )
    return alt.layer(rule_y, rule_x, lines).properties(
        width=WIDTH,
        height=260,
        title=_title(
            text="The test finds four of five loads at 25% of capacity, but not a day-long load",
            lines=[
                (
                    "Median over the 8 Primaries' latest season-years of the share of noise draws "
                    "flagged. The pale lines are the two shapes that some fit contains, and the "
                    "black line is a load shaped like solar generation. Dashed rules: power 0.8 "
                    "and amplitude 0.25."
                )
            ],
        ),
    )


def figure_10(*, units: pl.DataFrame) -> alt.TopLevelMixin:
    """V2: power at amplitude 0.25 per Primary, one panel per out-of-arm shape."""
    latest = units.group_by("site").agg(pl.col("year").max().alias("latest"))
    d = (
        units.join(latest, on="site")
        .filter(
            (pl.col("year") == pl.col("latest"))
            & pl.col("shape").is_in(common.OUT_OF_ARM_SHAPES)
            & (pl.col("amplitude") == common.POWER_AMPLITUDE)
        )
        .group_by("site", "shape")
        .agg(pl.col("power").mean())
        .with_columns(name=pl.col("site").replace(NAMES))
    )
    bars = (
        alt.Chart(d.select("name", "shape", "power"))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("power:Q", title="Share flagged", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("name:N", sort=ORDER, title=None),
            color=alt.Color("shape:N", scale=SHAPE_SCALE, legend=None),
        )
        .properties(width=150, height=140)
    )
    return bars.facet(facet=alt.Facet("shape:N", title=None), columns=3).properties(
        title=_title(
            text="Wrangle T1 cannot detect any load, and a day-long load escapes most Primaries",
            lines=[
                (
                    "Share of noise draws flagged at an amplitude of 0.25, per Primary, in its "
                    "latest season-year, for the five shapes that no fit contains."
                )
            ],
        )
    )


def figure_11(*, real: pl.DataFrame) -> alt.TopLevelMixin:
    """Real: misfit T against the threshold, per Primary-year."""
    d = real.with_columns(
        name=pl.col("site").replace(NAMES),
        label=pl.col("site").replace(NAMES) + " " + pl.col("year").cast(pl.String),
        status=pl.when(pl.col("flagged")).then(pl.lit("flagged")).otherwise(pl.lit("clean")),
    )
    scale = alt.Scale(type="log", domain=[0.01, 2.0])
    base = alt.Chart(d).encode(
        x=alt.X("tau:Q", scale=scale, title="Flag threshold (95th percentile of the null misfit)"),
        y=alt.Y("T:Q", scale=scale, title="Real relative misfit T"),
    )
    points = base.mark_circle(size=40).encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["clean", "flagged"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
            legend=alt.Legend(orient="top", title=None),
        ),
        tooltip=["label:N"],
    )
    labels = (
        base.transform_filter(alt.datum.flagged)
        .mark_text(align="left", dx=6, fontSize=9)
        .encode(text="label:N")  # ty: ignore[unresolved-attribute]
    )
    diagonal = (
        alt.Chart(pl.DataFrame({"a": [0.01, 2.0], "b": [0.01, 2.0]}))  # ty: ignore[unresolved-attribute]
        .mark_line(strokeDash=[4, 3], color="#999999")
        .encode(
            x=alt.X("a:Q", scale=scale),
            y=alt.Y("b:Q", scale=scale),
        )
    )
    return alt.layer(diagonal, points, labels).properties(
        width=WIDTH,
        height=320,
        title=_title(
            text="Only Kirton sits above its threshold",
            lines=[
                (
                    "Each dot is a directional Primary-year (41). Above the dashed line the misfit "
                    "exceeds the threshold built from injected solar-farm output and the series' "
                    "own block-bootstrap noise, and the Primary-year is flagged."
                )
            ],
        ),
    )


def figure_12(*, real: pl.DataFrame) -> alt.TopLevelMixin:
    """Real: headline capacity per Primary and season-year; flagged years hollow."""
    d = real.with_columns(
        name=pl.col("site").replace(NAMES),
        flag=pl.when(pl.col("flagged")).then(pl.lit("flagged")).otherwise(pl.lit("clean")),
    )
    x = alt.X("year:O", title="Season-year starting in")
    rule = (
        alt.Chart(d)
        .mark_rule(color=ocf.DATA_BLUE_LIGHT)
        .encode(x=x, y="headline_low:Q", y2="headline_high:Q")  # ty: ignore[unresolved-attribute]
    )
    dots = (
        alt.Chart(d)
        .mark_point(size=50, strokeWidth=2, color=ocf.DATA_BLUE)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            y=alt.Y("headline:Q", title="Capacity (MW)"),
            fill=alt.Fill(
                "flag:N",
                scale=alt.Scale(domain=["clean", "flagged"], range=[ocf.DATA_BLUE, "#FFFFFF"]),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
    )
    return (
        (rule + dots)
        .properties(width=250, height=120)
        .facet(facet=alt.Facet("name:N", sort=ORDER, title=None), columns=2)
        .properties(
            title=_title(
                text="Capacity grows at every Primary, whether or not a year is flagged",
                lines=[
                    (
                        "Daily-fit capacity divided by its recovery ratio, with a 95% "
                        "interval from "
                        ""
                        "resampling 7-day blocks. Hollow dots: Primary-years the misfit test flags."
                    )
                ],
            )
        )
    )


def figure_13(*, det: detector.Detector, real: pl.DataFrame) -> alt.TopLevelMixin:
    """Real: the six arm capacities per Primary in 2025/26 against `R * C`."""
    records = []
    for site in common.DIRECTIONAL_PRIMARIES:
        year = int(np.max(real.filter(pl.col("site") == site)["year"].to_numpy()))
        ratios = det.ratios(site=site, excluded=frozenset())
        assert ratios is not None
        scale = det.capacity(site=site, year=year, excluded=frozenset())
        for i, arm in enumerate(common.ARMS):
            records.append(
                {
                    "name": NAMES[site],
                    "arm": arm,
                    "fitted": float(det.real[site, year][i]),
                    "expected": float(ratios[i] * scale),
                }
            )
    d = pl.DataFrame(records)
    x = alt.X("arm:N", sort=list(common.ARMS), title=None)
    expected = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_line(color=ocf.DATA_BLUE_LIGHT, strokeWidth=3)
        .encode(x=x, y=alt.Y("expected:Q", title="Capacity (MW)"))
    )
    fitted = (
        alt.Chart(d)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=50, color=ocf.BRAND_ORANGE)
        .encode(x=x, y="fitted:Q")
    )
    return (
        (expected + fitted)
        .properties(width=250, height=120)
        .facet(facet=alt.Facet("name:N", sort=ORDER, title=None), columns=2)
        .properties(
            title=_title(
                text="Kirton's six fits disagree; the other Primaries' fits follow solar output",
                lines=[
                    (
                        "Orange dots: the capacity each fit gives the real series in 2025/26. Blue "
                        "line: what the same fits would give for solar generation alone, R "
                        "times the "
                        ""
                        "no-confound capacity C. The gap between dots and line is the misfit."
                    )
                ],
            )
        )
    )


def figure_14(*, menu: pl.DataFrame) -> alt.TopLevelMixin:
    """Descriptive: east, south, and west capacities per Primary-year."""
    d = (
        menu.unpivot(index=["site", "year"], on=["east", "south", "west"], variable_name="side")
        .with_columns(name=pl.col("site").replace(NAMES))
        .rename({"value": "capacity"})
    )
    return (
        alt.Chart(d.select("name", "year", "side", "capacity"))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            x=alt.X("year:O", title="Season-year starting in"),
            y=alt.Y("capacity:Q", title="Capacity (MW)", stack=True),
            color=alt.Color(
                "side:N",
                scale=alt.Scale(
                    domain=["east", "south", "west"],
                    range=[ocf.DATA_SKY, ocf.BRAND_ORANGE, ocf.DATA_PURPLE],
                ),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
        )
        .properties(width=250, height=120)
        .facet(facet=alt.Facet("name:N", sort=ORDER, title=None), columns=2)
        .properties(
            title=_title(
                text="The split between east, south, and west changes from year to year",
                lines=[
                    (
                        "Non-negative capacities of three panel orientations, fitted to April to "
                        "September rows with the M2h cloud terms free. The split is not "
                        "identified, "
                        ""
                        "so the menu decides nothing."
                    )
                ],
            )
        )
    )


def figure_15(*, units: pl.DataFrame) -> alt.TopLevelMixin:
    """The blind spot: detection power and bias against amplitude, sunload against lowsun."""
    d = (
        units.filter(pl.col("shape").is_in(["sunload", "lowsun"]))
        .group_by("shape", "amplitude")
        .agg(
            pl.col("power").median(),
            (pl.col("median_ratio") - 1.0).median().alias("bias"),
        )
    )
    colour = alt.Color(
        "shape:N",
        scale=alt.Scale(
            domain=["sunload", "lowsun"], range=[SHAPE_COLOURS["sunload"], SHAPE_COLOURS["lowsun"]]
        ),
        legend=alt.Legend(orient="top", title=None),
    )
    x = alt.X(
        "amplitude:Q",
        title="Confound amplitude (share of true capacity added to the half-hourly fit)",
        axis=alt.Axis(values=list(common.AMPLITUDES), format=".2~f"),
    )
    power = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(x=x, y=alt.Y("power:Q", title="Share of draws flagged"), color=colour)  # ty: ignore[unresolved-attribute]
        .properties(width=WIDTH, height=120)
    )
    bias = (
        alt.Chart(d)
        .mark_line(point=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y("bias:Q", title="Median capacity over truth, minus 1"), color=colour
        )
        .properties(width=WIDTH, height=120)
    )
    return alt.vconcat(power, bias).properties(
        title=_title(
            text="A load that rises with sunshine biases the capacity low and escapes the test",
            lines=[
                (
                    "Median over all units. Black: the load sunload, shaped like solar "
                    "output, which "
                    ""
                    "removes capacity from every fit alike. Blue: the load lowsun, for comparison. "
                    "The "
                    "lower panel shows the bias of the capacity C before any flag."
                )
            ],
        )
    )


def main() -> None:
    """Write the figures."""
    common.CHART_DIR.mkdir(parents=True, exist_ok=True)
    full = _full_rows()
    prepared = inputs.load_prepared().with_columns(name=pl.col("site").replace(NAMES))
    det = score.load_detector()
    real = pl.read_parquet(common.REAL_SCORES_PATH)
    units = pl.read_parquet(common.UNIT_SCORES_PATH)
    real_vectors = pl.read_parquet(common.REAL_VECTORS_PATH)
    menu = pl.read_parquet(common.MENU_PATH)
    figures: dict[str, alt.TopLevelMixin] = {
        "figure_01": figure_01(real=real),
        "figure_02": figure_02(full=full),
        "figure_03": figure_03(full=full),
        "figure_04": figure_04(prepared=prepared),
        "figure_05": figure_05(prepared=prepared),
        "figure_06": figure_06(real_vectors=real_vectors),
        "figure_07": figure_07(prepared=prepared),
        "figure_08": figure_08(units=units),
        "figure_09": figure_09(units=units),
        "figure_10": figure_10(units=units),
        "figure_11": figure_11(real=real),
        "figure_12": figure_12(real=real),
        "figure_13": figure_13(det=det, real=real),
        "figure_14": figure_14(menu=menu),
        "figure_15": figure_15(units=units),
    }
    for name, chart in figures.items():
        chart.save(common.CHART_DIR / f"{name}.svg")
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
