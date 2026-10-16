"""Draw experiment 00045's figures 1 to 4, the raw-data figures, as SVG.

No chart here shows a solar farm. Substations are named, in megawatts, on calendar dates. The SVGs
are afterwards optimised with `npx svgo@4 --multipass --precision=1 --final-newline`.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 uv run python \
        ml-experiments/00045-pvdis-4-weekly-capacity-asymmetric-l1/e00045_raw_charts.py
"""

import textwrap
from typing import Final

import altair as alt
import e00045_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

WIDTH: Final[int] = 560
WRAP: Final[int] = 88
ORDER: Final[list[str]] = [common.PRIMARY_NAMES[p] for p in common.PRIMARIES]
MIDDAY_UTC_HOURS: Final[tuple[int, int]] = (10, 14)
"""Stamps whose half-hour starts from 10:00 up to (not including) 14:00 UTC."""
MIN_MIDDAY_ROWS: Final[int] = 6
ROLLING_WEEKS: Final[int] = 8
HARMONICS: Final[int] = 3
DETECTION_FACTOR: Final[float] = (1.96 + 0.84) * np.sqrt(2.0)
"""Two independent weekly estimates differ detectably (5% test, 80% power) at this many standard
errors."""
MONTHS: Final[list[str]] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
DAY_LABELS: Final[str] = "['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][floor(datum.value / 24)]"


def _label(*, site: str) -> str:
    status = common.PRIMARY_STATUS[site]
    name = common.PRIMARY_NAMES[site]
    return name if status == "clean" else f"{name} ({status})"


LABELS: Final[dict[str, str]] = {site: _label(site=site) for site in common.PRIMARIES}
LABEL_ORDER: Final[list[str]] = [LABELS[site] for site in common.PRIMARIES]


def _title(*, text: str, lines: list[str]) -> alt.TitleParams:
    wrapped = [row for line in lines for row in textwrap.wrap(line, WRAP)]
    return alt.TitleParams(
        text=text, subtitle=wrapped, anchor="start", fontSize=14, subtitleFontSize=11
    )


def _usable_rows() -> pl.DataFrame:
    """Return the usable daylight half-hours with local-free calendar columns."""
    return (
        pl.read_parquet(common.HALF_HOURLY_PATH)
        .filter(pl.col("usable"))
        .with_columns(
            stamp_start=pl.col("time").dt.offset_by("-30m"),
            day=common.day_expr(time=pl.col("time")),
        )
        .with_columns(hour=pl.col("stamp_start").dt.hour(), month=pl.col("day").dt.month())
    )


def clear_overcast_gap() -> pl.DataFrame:
    """Return, per Primary-week, the midday power gap between overcast and clear days.

    A day's clear-sky index is the CAMS irradiance over the CAMS clear-sky irradiance, summed over
    the day's usable half-hours. Clear days (index 0.8 or more) and overcast days (0.3 or less) in
    April to September are pooled over the week and the 7 weeks before it, and the gap is the mean
    power from 10:00 to 14:00 UTC on the overcast days minus that on the clear days, in MW. A
    window with fewer than 3 days of either kind gives no value.
    """
    rows = _usable_rows().filter(pl.col("month").is_in(common.SUMMER_MONTHS))
    midday = pl.col("hour").is_between(MIDDAY_UTC_HOURS[0], MIDDAY_UTC_HOURS[1] - 1)
    daily = (
        rows.group_by("site", "week", "day")
        .agg(
            index=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum(),
            midday_power=pl.col("power").filter(midday).mean().cast(pl.Float64),
            midday_rows=midday.sum(),
        )
        .filter(pl.col("midday_rows") >= MIN_MIDDAY_ROWS)
    )
    clear = pl.col("index") >= common.CLEAR_INDEX
    overcast = pl.col("index") <= common.OVERCAST_INDEX
    weekly = daily.group_by("site", "week").agg(
        clear_sum=pl.col("midday_power").filter(clear).sum(),
        overcast_sum=pl.col("midday_power").filter(overcast).sum(),
        n_clear=clear.sum().cast(pl.Int64),
        n_overcast=overcast.sum().cast(pl.Int64),
    )
    grid = pl.DataFrame({"site": list(common.PRIMARIES)}).join(
        pl.DataFrame({"week": pl.int_range(common.N_WEEKS, eager=True, dtype=pl.Int32)}),
        how="cross",
    )
    return (
        grid.join(weekly, on=["site", "week"], how="left")
        .sort("site", "week")
        .with_columns(pl.col("clear_sum", "overcast_sum", "n_clear", "n_overcast").fill_null(0))
        .with_columns(
            pl.col("clear_sum", "overcast_sum", "n_clear", "n_overcast")
            .rolling_sum(window_size=ROLLING_WEEKS, min_samples=1)
            .over("site")
            .name.prefix("w_")
        )
        .with_columns(
            gap=pl.when((pl.col("w_n_clear") >= 3) & (pl.col("w_n_overcast") >= 3))
            .then(
                pl.col("w_overcast_sum") / pl.col("w_n_overcast")
                - pl.col("w_clear_sum") / pl.col("w_n_clear")
            )
            .otherwise(None),
            week_start=pl.lit(common.FIRST_WEEK_START) + pl.duration(weeks=pl.col("week")),
        )
    )


def figure_01(*, gap: pl.DataFrame) -> alt.FacetChart:
    """Raw: the model-free trend in the overcast-minus-clear midday power gap."""
    d = gap.with_columns(name=pl.col("site").replace(LABELS))
    x = alt.X(
        "week_start:T",
        title=None,
        axis=alt.Axis(format="%Y", tickCount="year", labelOverlap=True),
        scale=alt.Scale(
            domain=[common.FIRST_WEEK_START.isoformat(), common.LAST_DATA_END.isoformat()]
        ),
    )
    line = (
        alt.Chart()
        .mark_line(color=ocf.DATA_BLUE, strokeWidth=2)
        .encode(x=x, y=alt.Y("gap:Q", title=None, scale=alt.Scale(zero=True)))  # ty: ignore[unresolved-attribute]
    )
    zero = alt.Chart().mark_rule(color="#999999", strokeDash=[4, 3]).encode(y=alt.datum(0))  # ty: ignore[unresolved-attribute]
    panel = alt.layer(zero, line).properties(width=WIDTH, height=58)
    return panel.facet(
        row=alt.Row(
            "name:N",
            sort=LABEL_ORDER,
            title=None,
            header=alt.Header(labelAngle=0, labelAlign="left", labelFontSize=10, labelPadding=6),
        ),
        data=d,
    ).properties(
        title=_title(
            text="Where the overcast-minus-clear gap rises, solar capacity has risen",
            lines=[
                (
                    "Line (MW): the mean power from 10:00 to 14:00 UTC, in MW, on overcast days "
                    "(daily clear-sky index 0.3 or less) minus that on clear days (0.8 or more), "
                    "pooled over 8 weeks. A positive gap means less power is drawn on clear days, "
                    "so solar output is offsetting demand."
                ),
                (
                    "Scope: April to September only (the sun is too low in winter); days inside a "
                    "logged switching closure are removed; 8 Primary substations; windows with "
                    "fewer than 3 days of either kind give no line."
                ),
            ],
        ),
    )


def _week_means(*, rows: pl.DataFrame) -> pl.DataFrame:
    return rows.group_by("week").agg(
        n_rows=pl.len(),
        index=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum(),
    )


def pick_weeks() -> dict[str, int]:
    """Return the Donington weeks drawn in figure 2: the clearest and most overcast of each season.

    Candidate weeks hold at least 300 half-hours and no masked daylight half-hour. A summer week
    starts in June or July of 2022 or later, a winter week in December or January.
    """
    donington = pl.read_parquet(common.HALF_HOURLY_PATH).filter(pl.col("site") == "D16")
    masked_weeks = donington.filter(pl.col("masked")).select("week").unique().to_series().to_list()
    daytime = donington.filter(pl.col("u_clear") > 0.0)
    weeks = (
        daytime.group_by("week")
        .agg(
            index=pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum(),
            n_daytime=pl.len(),
        )
        .join(
            donington.group_by("week").agg(n_rows=pl.len(), n_power=pl.col("power").count()),
            on="week",
        )
        .with_columns(
            month=(
                pl.lit(common.FIRST_WEEK_START) + pl.duration(weeks=pl.col("week"), days=3)
            ).dt.month(),
            year=(
                pl.lit(common.FIRST_WEEK_START) + pl.duration(weeks=pl.col("week"), days=3)
            ).dt.year(),
        )
        .filter(
            (pl.col("n_rows") >= 300)
            & (pl.col("n_power") >= 300)
            & ~pl.col("week").is_in(masked_weeks)
            & (pl.col("year") >= 2022)
        )
    )
    summer = weeks.filter(pl.col("month").is_in([6, 7]))
    winter = weeks.filter(pl.col("month").is_in([12, 1]))
    return {
        "Summer, clear week": int(summer.sort("index")["week"][-1]),
        "Summer, overcast week": int(summer.sort("index")["week"][0]),
        "Winter, clear week": int(winter.sort("index")["week"][-1]),
        "Winter, overcast week": int(winter.sort("index")["week"][0]),
    }


def figure_02() -> tuple[alt.VConcatChart, dict[str, int]]:
    """Raw: one clear and one overcast week in summer and in winter at Donington."""
    chosen = pick_weeks()
    donington = pl.read_parquet(common.HALF_HOURLY_PATH).filter(pl.col("site") == "D16")
    panels = []
    for name, week in chosen.items():
        start = common.week_start(week=week)
        rows = donington.filter(pl.col("week") == week).with_columns(
            hours=(pl.col("time") - pl.lit(start)).dt.total_minutes() / 60.0 - 0.5,
            power_mw=pl.col("power").cast(pl.Float64),
        )
        index = (
            rows.filter(pl.col("u_clear") > 0.0)
            .select((pl.col("ghi_u").sum() / pl.col("ghi_u_clear").sum()).alias("i"))
            .item()
        )
        title = f"{name}: week of {start:%-d %B %Y}, clear-sky index {index:.2f}"
        x = alt.X(
            "hours:Q",
            title=None,
            scale=alt.Scale(domain=[0, 168]),
            axis=alt.Axis(
                values=[12 + 24 * i for i in range(7)],
                labelExpr=DAY_LABELS.replace("datum.value / 24", "(datum.value - 12) / 24"),
                grid=False,
            ),
        )
        power = (
            alt.Chart(rows)
            .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.5)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x,
                y=alt.Y(
                    "power_mw:Q",
                    title="Power (MW)",
                    scale=alt.Scale(zero=False),
                    axis=alt.Axis(titleColor=ocf.DATA_BLUE),
                ),
            )
        )
        sun = (
            alt.Chart(rows)
            .mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.5)
            .encode(  # ty: ignore[unresolved-attribute]
                x=x,
                y=alt.Y(
                    "ghi_u:Q",
                    title="CAMS irradiance (W/m²)",
                    axis=alt.Axis(titleColor=ocf.BRAND_ORANGE),
                ),
            )
        )
        panels.append(
            alt.layer(power, sun)
            .resolve_scale(y="independent")
            .properties(
                width=WIDTH, height=105, title=alt.TitleParams(title, fontSize=11, anchor="start")
            )
        )
    chart = alt.vconcat(*panels).properties(
        title=_title(
            text="At Donington, summer power falls through the middle of a sunny day",
            lines=[
                (
                    "Blue (left axis): half-hourly power at Donington Primary substation, in MW "
                    "(negative flow would be export). Orange (right axis): CAMS solar irradiance "
                    "arriving on a horizontal plane, in W/m². Days run from Monday (left) to "
                    "Sunday."
                ),
                (
                    "Scope: the four weeks with the highest and lowest clear-sky index (CAMS "
                    "irradiance over CAMS clear-sky irradiance) among June, July, December, and "
                    "January weeks since 2022 with no logged switching closure."
                ),
            ],
        )
    )
    return chart, chosen


def _design(*, rows: pl.DataFrame) -> np.ndarray:
    """Return the weekly design: constant, weekend, temperature, daily harmonics, minus u."""
    stamp = rows["time"].dt.offset_by("-30m")
    fraction = (stamp.dt.hour() * 2 + stamp.dt.minute() // 30).to_numpy() / 48.0
    weekend = (stamp.dt.weekday() >= 6).cast(pl.Float64).to_numpy()
    columns = [np.ones(rows.height), weekend, rows["temperature_2m"].cast(pl.Float64).to_numpy()]
    for k in range(1, HARMONICS + 1):
        columns += [np.sin(2 * np.pi * k * fraction), np.cos(2 * np.pi * k * fraction)]
    columns.append(-rows["u"].to_numpy())
    return np.column_stack(columns)


def free_weekly_capacity() -> pl.DataFrame:
    """Fit each observed Primary-week alone and return the capacity and its standard error.

    The fit is power on a constant, a weekend indicator, temperature, three daily harmonics, and
    minus the CAMS unit shape u, by ordinary least squares on the week's usable half-hours. The
    capacity is the coefficient on minus u, in MW; the standard error is the classical one, which
    ignores serial correlation and so understates the real uncertainty.
    """
    rows = _usable_rows()
    weeks = pl.read_parquet(common.INPUTS_PATH).filter(~pl.col("unobserved"))
    out = []
    for (site, week), block in rows.group_by("site", "week", maintain_order=True):
        if weeks.filter((pl.col("site") == site) & (pl.col("week") == week)).height == 0:
            continue
        design = _design(rows=block)
        target = block["power"].cast(pl.Float64).to_numpy()
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        degrees = block.height - rank
        if degrees < 5:
            continue
        residual = target - design @ coefficients
        sigma2 = float(residual @ residual / degrees)
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        out.append(
            {
                "site": site,
                "week": week,
                "capacity_mw": float(coefficients[-1]),
                "se_mw": float(np.sqrt(covariance[-1, -1])),
                "n_rows": block.height,
            }
        )
    return (
        pl.DataFrame(out)
        .with_columns(
            week_start=pl.lit(common.FIRST_WEEK_START) + pl.duration(weeks=pl.col("week")),
        )
        .with_columns(month=(pl.col("week_start") + pl.duration(days=3)).dt.month())
        .sort("site", "week")
    )


def _p95(values: pl.Series) -> float:
    """Return the 95th percentile of a numeric series."""
    return float(np.percentile(values.drop_nulls().to_numpy(), 95))


def _heat_text(
    *, d: pl.DataFrame, value: str, title: str, label_format: str, max_colour: float
) -> alt.LayerChart:
    y = alt.Y("name:N", sort=LABEL_ORDER, title=None, axis=alt.Axis(labelLimit=220))
    x = alt.X("month_name:N", sort=MONTHS, title=None, axis=alt.Axis(labelAngle=0))
    base = alt.Chart(d).encode(x=x, y=y)
    cells = base.mark_rect().encode(  # ty: ignore[unresolved-attribute]
        color=alt.Color(
            f"{value}:Q",
            scale=alt.Scale(domain=[0, max_colour], clamp=True),
            legend=None,
        )
    )
    text = base.mark_text(fontSize=9).encode(  # ty: ignore[unresolved-attribute]
        text=alt.Text(f"{value}:Q", format=label_format),
        color=alt.condition(
            alt.datum[value] > max_colour * 0.6, alt.value("#FFFFFF"), alt.value("#000000")
        ),
    )
    return (cells + text).properties(
        width=WIDTH, height=130, title=alt.TitleParams(title, fontSize=11, anchor="start")
    )


def figure_04(*, free: pl.DataFrame) -> tuple[alt.VConcatChart, pl.DataFrame]:
    """Raw: the free weekly capacity at Donington, and its standard error by calendar month."""
    donington = free.filter(pl.col("site") == "D16").with_columns(
        low=pl.col("capacity_mw") - 1.96 * pl.col("se_mw"),
        high=pl.col("capacity_mw") + 1.96 * pl.col("se_mw"),
    )
    x = alt.X("week_start:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    band = (
        alt.Chart(donington)
        .mark_rule(color=ocf.DATA_BLUE_LIGHT, strokeWidth=2)
        .encode(x=x, y=alt.Y("low:Q", title="Capacity (MW)"), y2="high:Q")  # ty: ignore[unresolved-attribute]
    )
    line = (
        alt.Chart(donington)
        .mark_point(color=ocf.DATA_BLUE, filled=True, size=14)
        .encode(x=x, y="capacity_mw:Q")  # ty: ignore[unresolved-attribute]
    )
    zero = (
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(color="#999999", strokeDash=[4, 3])
        .encode(y="v:Q")  # ty: ignore[unresolved-attribute]
    )
    top = alt.layer(zero, band, line).properties(
        width=WIDTH,
        height=170,
        title=alt.TitleParams(
            "Donington: free weekly capacity (dots) with its 95% range (1.96 standard errors)",
            fontSize=11,
            anchor="start",
        ),
    )
    by_month = (
        free.group_by("site", "month")
        .agg(se=pl.col("se_mw").median(), weeks=pl.len())
        .with_columns(
            step=pl.col("se") * DETECTION_FACTOR,
            name=pl.col("site").replace(LABELS),
            month_name=pl.col("month").map_elements(
                lambda m: MONTHS[m - 1], return_dtype=pl.String
            ),
        )
    )
    bottom_se = _heat_text(
        d=by_month,
        value="se",
        title="Median standard error of the weekly capacity, by calendar month (MW)",
        label_format=".2f",
        max_colour=_p95(by_month["se"]),
    )
    bottom_step = _heat_text(
        d=by_month,
        value="step",
        title="Implied smallest detectable step between two weeks (MW) = 3.96 x standard error",
        label_format=".2f",
        max_colour=_p95(by_month["step"]),
    )
    chart = alt.vconcat(top, bottom_se, bottom_step).properties(
        title=_title(
            text="One week of data fixes capacity only to within a few tenths of a megawatt",
            lines=[
                (
                    "Each week is fitted alone by least squares: power on a constant, a weekend "
                    "indicator, temperature, three daily harmonics, and minus the CAMS solar shape"
                    " (peak 1). The coefficient on the shape is the week's capacity in MW."
                ),
                (
                    "Standard errors are the classical ones, so they understate the real "
                    "uncertainty; blank cells are months with no observed week. 8 Primary "
                    "substations, 2019 to 2026."
                ),
            ],
        )
    )
    return chart, by_month


def _runs(*, d: pl.DataFrame, value: str, step: float) -> pl.DataFrame:
    """Round `value` to `step`, then merge consecutive equal weeks of a Primary into one rectangle.

    Merging keeps the SVG small: one rectangle per week would run to a megabyte.
    """
    d = d.sort("site", "week").with_columns(
        shown=(pl.col(value) / step).round() * step,
        state=pl.when(pl.col("before_start"))
        .then(pl.lit("before"))
        .when(pl.col("unobserved"))
        .then(pl.lit("unobserved"))
        .otherwise(pl.lit("observed")),
    )
    key = pl.struct(
        "state", pl.when(pl.col("state") == "observed").then(pl.col("shown")).otherwise(None)
    )
    d = d.with_columns(
        run=(key != key.shift(1)).fill_null(True).cast(pl.Int64).cum_sum().over("site")
    )
    return (
        d.group_by("site", "run", "state")
        .agg(
            week_start=pl.col("week_start").min(),
            week_end=pl.col("week_start").max() + pl.duration(days=7),
            shown=pl.col("shown").first(),
        )
        .with_columns(name=pl.col("site").replace(LABELS))
    )


def _heat_rect(
    *, d: pl.DataFrame, value: str, title: str, domain: list[float], legend_title: str, step: float
) -> alt.LayerChart:
    merged = _runs(d=d, value=value, step=step)
    y = alt.Y("name:N", sort=LABEL_ORDER, title=None, axis=alt.Axis(labelLimit=220))
    x = alt.X("week_start:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"))
    observed = (
        alt.Chart(merged.filter(pl.col("state") == "observed"))
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            x2="week_end:T",
            y=y,
            color=alt.Color(
                "shown:Q",
                scale=alt.Scale(domain=domain, clamp=True),
                legend=alt.Legend(orient="right", title=legend_title, gradientLength=60),
            ),
        )
    )
    unobserved = (
        alt.Chart(
            merged.filter(pl.col("state") == "unobserved").with_columns(
                label=pl.lit("Unobserved week")
            )
        )
        .mark_rect()
        .encode(  # ty: ignore[unresolved-attribute]
            x=x,
            x2="week_end:T",
            y=y,
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=["Unobserved week"], range=["#4D4D4D"]),
                legend=alt.Legend(orient="right", title=None),
            ),
        )
    )
    return (
        alt.layer(observed, unobserved)
        .resolve_scale(color="independent")
        .properties(
            width=WIDTH - 110, height=130, title=alt.TitleParams(title, fontSize=11, anchor="start")
        )
    )


def figure_03(*, weeks: pl.DataFrame) -> alt.LayerChart:
    """Raw: coverage heat maps per Primary-week."""
    a = _heat_rect(
        d=weeks,
        value="n_usable",
        title="Usable daylight half-hours per week",
        domain=[0.0, 180.0],
        legend_title="Half-hours",
        step=20.0,
    )
    b = _heat_rect(
        d=weeks,
        value="n_masked_days",
        title="Days per week with a masked closure half-hour (24 h padding)",
        domain=[0.0, 7.0],
        legend_title="Days",
        step=1.0,
    )
    c = _heat_rect(
        d=weeks,
        value="cams_dropped_share",
        title="Share of the week's daylight half-hours with CAMS dropped (reliability under 0.9)",
        domain=[0.0, 0.5],
        legend_title="Share",
        step=0.05,
    )
    return (
        alt.vconcat(a, b, c)
        .resolve_scale(color="independent")
        .properties(
            title=_title(
                text="Winter weeks and heavily switched Primaries have little usable evidence",
                lines=[
                    (
                        "Rows: the 8 Primary substations; columns: Monday-start weeks from "
                        "September 2019. A half-hour is usable when it is daylight (sun more than "
                        "15 degrees above the horizon), has power and temperature, and lies "
                        "outside a logged closure. A week is unobserved (dark grey) with fewer "
                        "than 3 days of 10 or more usable half-hours. White cells precede a "
                        "Primary's first observed week."
                    ),
                ],
            )
        )
    )


def main() -> None:
    """Draw, save, and describe figures 1 to 4."""
    common.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    weeks = pl.read_parquet(common.INPUTS_PATH)

    gap = clear_overcast_gap()
    print(
        "\nFigure 1: overcast-minus-clear midday gap (MW), pooled over 8 weeks, April to September"
    )
    for site in common.PRIMARIES:
        rows = gap.filter(pl.col("site") == site).drop_nulls("gap")
        by_year = (
            rows.filter(pl.col("week_start").dt.month().is_in([6, 7, 8]))
            .with_columns(year=pl.col("week_start").dt.year())
            .group_by("year")
            .agg(m=pl.col("gap").mean())
            .sort("year")
        )
        print(
            f"{LABELS[site]:30s} weeks with a gap {rows.height:3d}; June-August mean gap by year  "
            + "  ".join(f"{y}: {m:5.2f}" for y, m in by_year.iter_rows())
        )
    figure_01(gap=gap).save(common.DOCS_DIR / "figure_01.svg")

    chart, chosen = figure_02()
    print("\nFigure 2: Donington weeks drawn")
    for name, week in chosen.items():
        print(f"{name}: week {week}, starting {common.week_start(week=week):%Y-%m-%d}")
    chart.save(common.DOCS_DIR / "figure_02.svg")

    figure_03(weeks=weeks).save(common.DOCS_DIR / "figure_03.svg")
    print("\nFigure 3: coverage")
    for site in common.PRIMARIES:
        rows = weeks.filter((pl.col("site") == site) & ~pl.col("before_start"))
        print(
            f"{LABELS[site]:30s} median usable half-hours {rows['n_usable'].median():5.0f}  "
            f"weeks with a masked day {rows.filter(pl.col('n_masked_days') > 0).height:3d}  "
            f"unobserved {int(rows['unobserved'].sum()):3d}  "
            f"weeks with CAMS share over 0.1 "
            f"{rows.filter(pl.col('cams_dropped_share') > 0.1).height:3d}"
        )

    free = free_weekly_capacity()
    free.write_parquet(common.WEEKLY_FREE_PATH)
    chart4, by_month = figure_04(free=free)
    chart4.save(common.DOCS_DIR / "figure_04.svg")
    print("\nFigure 4: free weekly capacity (simplified weekly least squares)")
    print(f"weekly fits: {free.height}")
    for site in common.PRIMARIES:
        rows = free.filter(pl.col("site") == site)
        summer_se = float(
            rows.filter(pl.col("month").is_in(common.SUMMER_MONTHS))["se_mw"].median()  # ty: ignore[invalid-argument-type]
        )
        print(
            f"{LABELS[site]:30s} fits {rows.height:3d}  "
            f"median capacity {rows['capacity_mw'].median():5.2f} MW  "
            f"median SE {rows['se_mw'].median():5.2f} MW  "
            f"April-September median SE {summer_se:5.2f}  "
            f"smallest detectable step (summer) {DETECTION_FACTOR * summer_se:5.2f} MW"
        )
    print("median SE (MW) by calendar month, rows are Primaries:")
    pivot = by_month.pivot(on="month", index="site", values="se").sort("site")
    print(pivot.with_columns(pl.selectors.numeric().round(3)).write_csv())


if __name__ == "__main__":
    main()
