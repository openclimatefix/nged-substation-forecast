"""Draw the data-look figures of the unmetered-battery-capacity page (5 to 9).

These figures show what the inputs look like before any fit: the eight NGED primaries in MW (they
are substations, so the rule that bars a metered generator's name or series does not cover them,
and they carry their study labels S1 to S8), the two prices, NGED battery A inside its bulk supply
point (normalised by its 99th percentile, with the days numbered and no dates), and a simulated
merchant battery added to a primary at three sizes.

Run: `uv run python studies/unmetered_battery_capacity/capacity_charts_data.py`.
"""

from datetime import UTC, datetime
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from capacity_charts_common import PERCENT, draw_figure, save
from capacity_inputs import (
    OUTPUT_DIR,
    agile_prices,
    day_ahead_on_grid,
    nged_series,
    window_half_hours,
)
from capacity_runs import p99_flow, simulated_merchant_battery
from studies.charts import CONTENT_WIDTH_PX

PRIMARIES: Final[list[str]] = [f"S{n}" for n in range(1, 9)]
WINTER_WEEK_START: Final[datetime] = datetime(2026, 1, 5, tzinfo=UTC)
SUMMER_WEEK_START: Final[datetime] = datetime(2026, 7, 6, tzinfo=UTC)
HALF_HOURS_PER_WEEK: Final[int] = 7 * 48
WINDOW_EDGES: Final[list[float]] = [0.5, 2.0, 5.5, 16.0, 19.0, 23.5]
"""Tariff window edges and the red band's ends, in hours on the UK clock."""
SIMULATION_SEED: Final[int] = 20261009
SHOWN_SHARES: Final[tuple[float, ...]] = (0.4, 0.1, 0.02)
DAYS_PER_WEEK: Final[int] = 7
SMALL_PANEL_WIDTH_PX: Final[int] = 290
FIRST_WEEKEND_WEEKDAY: Final[int] = 6
"""Polars numbers the weekdays from Monday = 1."""


def _frame(*, values: dict[str, np.ndarray]) -> pl.DataFrame:
    """Return the series on the window grid, with each half-hour's start in UTC and UK time."""
    start = window_half_hours().dt.offset_by("-30m")
    return (
        pl.DataFrame({"start": start, **values})
        .with_columns(pl.col(pl.Float64).fill_nan(None))
        .with_columns(uk=pl.col("start").dt.convert_time_zone("Europe/London"))
    )


def _week(*, frame: pl.DataFrame, first: datetime) -> pl.DataFrame:
    """Return the 7 days of the frame that start at `first` (a midnight, UTC)."""
    start = pl.lit(first)
    return frame.filter(
        (pl.col("start") >= start) & (pl.col("start") < start + pl.duration(days=DAYS_PER_WEEK))
    )


def primaries_figure(*, number: int) -> alt.VConcatChart:
    """Draw two weeks of every primary's flow, one in winter and one in summer."""
    series = nged_series()
    frame = _frame(values={label: series[label] for label in PRIMARIES})
    long = pl.concat(
        [
            _week(frame=frame, first=first).with_columns(season=pl.lit(label))
            for first, label in (
                (WINTER_WEEK_START, "Week from 5 January 2026"),
                (SUMMER_WEEK_START, "Week from 6 July 2026"),
            )
        ]
    ).unpivot(index=["start", "season"], on=PRIMARIES, variable_name="primary", value_name="mw")
    base = (
        alt.Chart(long)
        .mark_line(color=ocf.DATA_BLUE, strokeWidth=1.0, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("start:T", axis=alt.Axis(format="%a", tickCount=7), title=None),
            y=alt.Y("mw:Q", title="MW", scale=alt.Scale(zero=False)),
        )
        .properties(width=SMALL_PANEL_WIDTH_PX, height=48)
    )
    panel = base.facet(
        row=alt.Row("primary:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        column=alt.Column("season:N", title=None, sort=["Week from 5 January 2026"]),
    ).resolve_scale(x="independent", y="independent")
    return draw_figure(
        panels=[panel],
        number=number,
        title="The primaries differ in level, noise, and daily shape",
        subtitle=[
            (
                "Half-hourly import in MW (positive is import) of the 8 NGED primaries "
                "metered in MW, for the first full Monday-to-Sunday week of January and"
                " of July 2026."
            ),
            (
                "Each row has its own vertical axis. The noise from one half-hour to "
                "the next, not the level, sets the smallest battery that can be seen."
            ),
        ],
        figure_planning=None,
    )


def profile_figure(*, number: int) -> alt.VConcatChart:
    """Draw each primary's daily profile and its mean half-hour change by time of day."""
    series = nged_series()
    frame = _frame(values={label: series[label] for label in PRIMARIES})
    long = (
        frame.unpivot(index=["start", "uk"], on=PRIMARIES, variable_name="primary", value_name="mw")
        .drop_nulls("mw")
        .sort("primary", "start")
        .with_columns(
            change=pl.col("mw").diff().over("primary"),
            hour=pl.col("uk").dt.hour().cast(pl.Float64) + pl.col("uk").dt.minute() / 60.0,
            day_type=pl.when(pl.col("uk").dt.weekday() >= FIRST_WEEKEND_WEEKDAY)
            .then(pl.lit("Weekends"))
            .otherwise(pl.lit("Weekdays")),
        )
    )
    profile = long.group_by("primary", "day_type", "hour").agg(pl.col("mw").mean())
    change = (
        long.drop_nulls("change")
        .group_by("primary", "hour")
        .agg(pl.col("change").mean())
        .with_columns(kind=pl.lit("All days"))
    )
    edges = pl.DataFrame({"hour": WINDOW_EDGES})

    def grid(*, data: pl.DataFrame, y: str, y_title: str, colour: str | None) -> alt.VConcatChart:
        x = alt.X(
            "hour:Q",
            scale=alt.Scale(domain=[0, 24]),
            axis=alt.Axis(values=[0, 12, 24]),
            title="Hour, UK clock",
        )
        rules = (
            alt.Chart(edges)
            .mark_rule(color=ocf.GREY_3, strokeDash=[2, 2], aria=False)
            .encode(x="hour:Q")  # ty: ignore[unresolved-attribute]
        )
        cells = []
        for label in PRIMARIES:
            part = data.filter(pl.col("primary") == label)
            encoding = {"x": x, "y": alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False))}
            if colour is None:
                line = (
                    alt.Chart(part)
                    .mark_line(strokeWidth=1.2, color=ocf.BRAND_ORANGE, aria=False)
                    .encode(**encoding)  # ty: ignore[unresolved-attribute]
                )
            else:
                line = (
                    alt.Chart(part)
                    .mark_line(strokeWidth=1.2, aria=False)
                    .encode(  # ty: ignore[unresolved-attribute]
                        **encoding,
                        color=alt.Color(
                            f"{colour}:N",
                            scale=alt.Scale(
                                domain=["Weekdays", "Weekends"],
                                range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                            ),
                            legend=alt.Legend(title=None, orient="bottom"),
                        ),
                    )
                )
            cells.append(
                alt.layer(rules, line).properties(
                    width=105, height=70, title=alt.TitleParams(label, anchor="start")
                )
            )
        half = len(cells) // 2
        return alt.vconcat(alt.hconcat(*cells[:half]), alt.hconcat(*cells[half:]), spacing=8)

    return draw_figure(
        panels=[
            grid(data=profile, y="mw", y_title="Mean MW", colour="day_type"),
            grid(data=change, y="change", y_title="Mean change, MW", colour=None),
        ],
        number=number,
        title=(
            "Daily shapes dwarf a battery; some primaries step near 05:30, the "
            "end of the cheap windows"
        ),
        subtitle=[
            (
                "Upper block: mean flow in MW by half-hour of the UK clock over the "
                "study year (September 2025 to August 2026), weekdays against weekends."
                " Lower block: mean change from one half-hour to the next."
            ),
            (
                "Grey dashed lines: the tariff window edges (00:30, 02:00, 05:30, "
                "23:30) and the ends of the red band (16:00 and 19:00). Each panel has "
                "its own vertical axis."
            ),
        ],
        figure_planning=None,
    )


def prices_figure(*, number: int) -> alt.VConcatChart:
    """Draw a week of the N2EX day-ahead price and the East Midlands Agile price."""
    n2ex = _frame(values={"N2EX day-ahead price": day_ahead_on_grid()})
    agile = (
        agile_prices()
        .with_columns(
            start=pl.col("time").dt.cast_time_unit("us"),
            **{"Octopus Agile import price": pl.col("price_inc_vat_p_per_kwh") * 10.0},
        )
        .select("start", "Octopus Agile import price")
    )
    week = _week(frame=n2ex.join(agile, on="start", how="left"), first=WINTER_WEEK_START).unpivot(
        index="start",
        on=["N2EX day-ahead price", "Octopus Agile import price"],
        variable_name="price",
        value_name="gbp_per_mwh",
    )
    chart = (
        alt.Chart(week)
        .mark_line(strokeWidth=1.4, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("start:T", axis=alt.Axis(format="%a %d %b", tickCount=7), title=None),
            y=alt.Y("gbp_per_mwh:Q", title="£ per MWh"),
            color=alt.Color(
                "price:N",
                scale=alt.Scale(
                    domain=["N2EX day-ahead price", "Octopus Agile import price"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 110, height=150)
    )
    return draw_figure(
        panels=[chart],
        number=number,
        title="The two prices share a daily shape, and Agile's cheap slots move from day to day",
        subtitle=[
            (
                "Week from 5 January 2026 (UTC). N2EX is the hourly day-ahead price; "
                "Agile is the East Midlands half-hourly import price including VAT, "
                "converted from pence per kilowatt-hour to pounds per megawatt-hour."
            ),
        ],
        figure_planning=None,
    )


def battery_a_figure(*, number: int) -> alt.VConcatChart:
    """Draw NGED battery A beneath its bulk supply point's flow, with the days numbered."""
    series = nged_series()
    battery, flow = series["battery_A"], series["BSP1"]
    battery_p99, flow_p99 = p99_flow(battery), p99_flow(flow)
    frame = _frame(
        values={
            "output": battery / battery_p99,
            "flow": flow / flow_p99,
            "added_back": (flow + battery) / flow_p99,
        }
    )
    week = _week(frame=frame, first=WINTER_WEEK_START).with_columns(
        day=(pl.col("start") - pl.lit(WINTER_WEEK_START)).dt.total_hours() / 24.0 + 1.0
    )
    titles = {
        "output": "NGED battery A's output",
        "flow": "The bulk supply point's flow",
        "added_back": "The flow with the battery's output added back",
    }
    panels = []
    for (name, title), colour in zip(
        titles.items(), [ocf.BRAND_ORANGE, ocf.DATA_BLUE, ocf.DATA_SKY], strict=True
    ):
        panels.append(
            alt.Chart(week.select("day", name))
            .mark_line(strokeWidth=1.4, color=colour, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "day:Q",
                    scale=alt.Scale(domain=[1, 8]),
                    axis=alt.Axis(values=list(range(1, 9)), format="d"),
                    title="Day of the week plotted (1 to 7)",
                ),
                y=alt.Y(f"{name}:Q", title="Over own 99th percentile", scale=alt.Scale(zero=False)),
            )
            .properties(
                width=CONTENT_WIDTH_PX - 110,
                height=90,
                title=alt.TitleParams(title, anchor="start"),
            )
        )
    return draw_figure(
        panels=panels,
        number=number,
        title="A real battery of a few percent of a bulk supply point's flow is hard to see by eye",
        subtitle=[
            (
                "One week in winter, with days numbered rather than dated. Each series "
                "is divided by its own 99th percentile. The battery exports when its "
                "output is positive, which lowers the import."
            ),
            (
                "The two half-hour changes correlate at -0.29 over the study year, and "
                "at 0.07 or less with every primary: the evidence that the battery "
                "connects to this bulk supply point."
            ),
        ],
        figure_planning=None,
    )


def visibility_figure(*, number: int) -> alt.VConcatChart:
    """Draw a primary's week with a simulated merchant battery of three sizes subtracted."""
    series = nged_series()
    demand = series["S6"]
    p99 = p99_flow(demand)
    unit, _, _ = simulated_merchant_battery(nameplate_hours=2.0, seed=SIMULATION_SEED)
    panels = []
    for share in SHOWN_SHARES:
        battery = share * p99 * unit
        frame = _frame(
            values={
                "S6 with no battery": demand,
                "S6 with the battery subtracted": demand - battery,
                "The battery (export positive)": battery,
            }
        )
        week = _week(frame=frame, first=WINTER_WEEK_START).unpivot(
            index=["start", "uk"], variable_name="series", value_name="mw"
        )
        panels.append(
            alt.Chart(week)
            .mark_line(strokeWidth=1.2, aria=False)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X("start:T", axis=alt.Axis(format="%a", tickCount=7), title=None),
                y=alt.Y("mw:Q", title="MW", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    scale=alt.Scale(
                        domain=[
                            "S6 with no battery",
                            "S6 with the battery subtracted",
                            "The battery (export positive)",
                        ],
                        range=[ocf.BLACK_1, ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                    ),
                    legend=alt.Legend(title=None, columns=1, labelLimit=300),
                ),
            )
            .properties(
                width=CONTENT_WIDTH_PX - 280,
                height=100,
                title=alt.TitleParams(
                    f"A 2-hour battery at {share * PERCENT:g}% of the 99th-percentile flow",
                    anchor="start",
                ),
            )
        )
    return draw_figure(
        panels=panels,
        number=number,
        title="A simulated battery is visible by eye at a 40% share and invisible at 2%",
        subtitle=[
            (
                "Primary S6, week from 5 January 2026. The battery follows the "
                "day-ahead price (a linear-programme dispatch) and is subtracted from "
                "the flow in MW."
            ),
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write the data-look figures."""
    figures = {
        "primaries": primaries_figure(number=5),
        "profiles": profile_figure(number=6),
        "prices": prices_figure(number=7),
        "battery_a_in_bsp": battery_a_figure(number=8),
        "visibility": visibility_figure(number=9),
    }
    for name, chart in figures.items():
        print(f"Wrote {save(chart=chart, name=name)}")
    print(f"Previews in {OUTPUT_DIR / 'previews'}")


if __name__ == "__main__":
    main()
