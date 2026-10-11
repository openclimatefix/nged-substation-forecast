"""Draw the solar-disaggregation page's figures.

Every figure is built from the series and tables the other scripts wrote under the study's data
folder, so no figure refits anything. The BMU register, B1610, and CAMS are public, so the figures
name each BMU and show output in megawatts on calendar dates.

Run after `walkthrough_series.py` and `disaggregation_report.py`:
`uv run python studies/solar_disaggregation/disaggregation_charts.py`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from inputs import OUTPUT_DIR
from studies.charts import CONTENT_WIDTH_PX, figure

ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
WEEKS: Final[dict[str, datetime]] = {
    "summer": datetime(2026, 6, 15, tzinfo=UTC),
    "winter": datetime(2025, 12, 15, tzinfo=UTC),
}
"""The Monday of the week containing each solstice, fixed by the calendar and not chosen by how the
output looks. The census page uses the same weeks."""
PANEL_HEIGHT_PX: Final[int] = 105
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 95
SOLAR_COLOUR: Final[str] = ocf.BRAND_ORANGE
OTHER_COLOUR: Final[str] = ocf.DATA_BLUE
RECOVERED_COLOUR: Final[str] = ocf.DATA_GREEN
AGGREGATE_COLOUR: Final[str] = ocf.BLACK_1
GREY: Final[str] = ocf.GREY_3


@dataclass(frozen=True)
class Line:
    """One line in a time-series panel.

    Attributes:
        column: The column to plot.
        label: The legend name.
        colour: The line colour.
        width: The line width in pixels.
        dash: The dash pattern, or an empty tuple for a solid line.
    """

    column: str
    label: str
    colour: str
    width: float = 1.5
    dash: tuple[int, ...] = ()


def week_of(
    *, frame: pl.DataFrame, season: str, time_column: str = "half_hour_end_time"
) -> pl.DataFrame:
    """Return the rows of one fixed week, with the time shifted to the start of each half-hour.

    Args:
        frame: The rows to filter.
        season: A key of `WEEKS`: the week to draw.
        time_column: The column holding the end time of each half-hour.

    Returns:
        The rows in the week whose end time falls after the week's start and up to seven days later.
    """
    start = WEEKS[season]
    return frame.filter(
        (pl.col(time_column) > start) & (pl.col(time_column) <= start + timedelta(days=7))
    )


def time_panel(
    *,
    frame: pl.DataFrame,
    lines: Sequence[Line],
    title: str,
    y_title: str,
    last: bool,
    domain: tuple[float, float] | None = None,
    time_column: str = "half_hour_end_time",
    zero_rule: bool = True,
    height: int = PANEL_HEIGHT_PX,
    palette: dict[str, str] | None = None,
    date_format: str = "%a %-d %b",
    tick_count: str = "day",
    x_title: str = "Day (UTC)",
) -> alt.LayerChart:
    """Draw a panel of lines over time.

    Args:
        frame: The series.
        lines: The lines to draw.
        title: The panel's title.
        y_title: The y axis title.
        last: Whether the panel is the bottom one, which carries the time axis labels.
        domain: The y axis range, or None to let the data decide.
        time_column: The time column.
        zero_rule: Whether to draw a rule at zero.
        height: The panel's height in pixels.
        palette: Every series name in the whole figure with its colour. The panels of one figure
            share a colour scale, so each panel passes the figure's full palette.
        date_format: The format of the time axis labels.
        tick_count: The spacing of the time axis ticks.
        x_title: The time axis title, drawn on the bottom panel.

    Returns:
        The panel.
    """
    long = pl.concat(
        [
            frame.select(
                time=pl.col(time_column), value=pl.col(line.column), series=pl.lit(line.label)
            )
            for line in lines
        ]
    ).drop_nulls("value")
    names = [line.label for line in lines]
    full = palette or {line.label: line.colour for line in lines}
    dashes = {line.label: list(line.dash) for line in lines}
    widths = {line.label: line.width for line in lines}
    y_scale = alt.Scale(domain=list(domain), nice=False) if domain else alt.Scale(zero=False)
    chart = (
        alt.Chart(long, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
        .mark_line(interpolate="monotone", aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T",
                axis=alt.Axis(
                    format=date_format,
                    tickCount=tick_count,  # ty: ignore[invalid-argument-type]
                    labels=last,
                    ticks=last,
                    title=x_title if last else None,
                ),
            ),
            y=alt.Y("value:Q", scale=y_scale, axis=alt.Axis(tickCount=4, title=y_title)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(full), range=list(full.values())),
                legend=alt.Legend(title=None, columns=2, labelLimit=330, symbolLimit=0),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(domain=names, range=[dashes[name] or [1, 0] for name in names]),
                legend=None,
            ),
            strokeWidth=alt.StrokeWidth(
                "series:N",
                scale=alt.Scale(domain=names, range=[widths[name] for name in names]),
                legend=None,
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=height)
    )
    if not zero_rule:
        return cast(alt.LayerChart, alt.layer(chart))
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=0.6, opacity=0.6, aria=False)
        .encode(y="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    return cast(alt.LayerChart, alt.layer(zero, chart))


def week_subtitle(*, season: str) -> str:
    """Return the sentence that names a fixed week.

    Args:
        season: A key of `WEEKS`: the week to draw.

    Returns:
        The sentence.
    """
    start = WEEKS[season]
    end = start + timedelta(days=6)
    return (
        f"Half-hourly values, 00:00 UTC on {start:%-d %B %Y} to 23:30 UTC on {end:%-d %B %Y}, "
        "in megawatts unless stated."
    )


def scenarios() -> pl.DataFrame:
    """Return the walk-through scenarios' series.

    Returns:
        One row per half-hour per scenario.
    """
    return pl.read_parquet(OUTPUT_DIR / "walkthrough_scenarios.parquet")


def sites() -> pl.DataFrame:
    """Return the walk-through sites' forward-model series.

    Returns:
        One row per half-hour per site.
    """
    return pl.read_parquet(OUTPUT_DIR / "walkthrough_sites.parquet")


SCENARIO: Final[str] = "three_pure_pv_plus_offshore_wind_25"
"""The walk-through scenario: three real pure-PV BMUs plus a real offshore wind BMU."""
BATTERY_SCENARIO: Final[str] = "three_pure_pv_plus_battery_25"
LABELS: Final[dict[str, tuple[str, str]]] = {
    "solar_truth_mw": ("True solar (the answer key)", SOLAR_COLOUR),
    "non_solar_mw": ("Real output of the other generation", OTHER_COLOUR),
    "aggregate_mw": ("Synthetic aggregate", AGGREGATE_COLOUR),
    "separation_solar_mw": ("Recovered: calendar-baseline separation", ocf.DATA_PURPLE),
    "difference_solar_mw": ("Recovered: difference separation", ocf.DATA_SKY),
    "physical_solar_mw": ("Recovered: physical fit to changes", RECOVERED_COLOUR),
}
"""Column: legend name and colour, shared by every figure that draws the series."""


def _line(column: str, *, width: float = 1.5, label: str | None = None) -> Line:
    name, colour = LABELS[column]
    return Line(column=column, label=label or name, colour=colour, width=width)


def _palette(*columns: str) -> dict[str, str]:
    return {LABELS[column][0]: LABELS[column][1] for column in columns}


def build_figure(*, number: int, season: str = "summer") -> alt.VConcatChart:
    """Show how the synthetic aggregate is summed from real BMU series.

    Args:
        number: The figure's number on its page.
        season: A key of `WEEKS`: the week to draw.

    Returns:
        The figure.
    """
    frame = week_of(frame=scenarios().filter(pl.col("scenario") == SCENARIO), season=season)
    palette = _palette("solar_truth_mw", "non_solar_mw", "aggregate_mw")
    titles = {
        "solar_truth_mw": (
            "1. The real output of three pure-PV BMUs, added together: the solar part, known "
            "exactly"
        ),
        "non_solar_mw": "2. The real output of an offshore wind farm BMU: the other generation",
        "aggregate_mw": (
            "3. Their sum is the synthetic aggregate. The methods see only this and the weather."
        ),
    }
    panels = [
        time_panel(
            frame=frame,
            lines=[_line(column=column)],
            title=title,
            y_title="MW",
            last=column == "aggregate_mw",
            palette=palette,
        )
        for column, title in titles.items()
    ]
    return figure(
        panels=panels,
        number=number,
        title=(
            "A synthetic aggregate is real solar output plus real wind output, so the answer is "
            "known"
        ),
        subtitle=[
            week_subtitle(season=season),
            "Solar is scaled to 25% of the aggregate's 99th-percentile output, wind to 75%.",
        ],
        figure_planning=None,
    )


def forward_model_figure(*, number: int, bmu_id: str, season: str) -> alt.VConcatChart:
    """Show the forward model's steps for one real solar BMU, from CAMS irradiance to power.

    Args:
        number: The figure's number on its page.
        bmu_id: The validation BMU to draw.
        season: A key of `WEEKS`: the week to draw.

    Returns:
        The figure.
    """
    frame = week_of(frame=sites().filter(pl.col("bmu") == bmu_id), season=season)
    plant = frame.row(0, named=True)
    colours = {
        "CAMS global horizontal irradiance": ocf.DATA_SKY,
        "Direct normal irradiance": ocf.BRAND_ORANGE,
        "Diffuse horizontal irradiance": ocf.DATA_BLUE,
        "Irradiance on the fitted plane": ocf.DATA_PURPLE,
        "Measured output": AGGREGATE_COLOUR,
        "Fitted plant": RECOVERED_COLOUR,
    }
    panels = [
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="ghi_w_m2",
                    label="CAMS global horizontal irradiance",
                    colour=colours["CAMS global horizontal irradiance"],
                )
            ],
            title=(
                "1. CAMS hourly irradiance, shared between the two half-hours of each hour by "
                "the sun's height"
            ),
            y_title="W/m²",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="dni_w_m2",
                    label="Direct normal irradiance",
                    colour=colours["Direct normal irradiance"],
                ),
                Line(
                    column="dhi_w_m2",
                    label="Diffuse horizontal irradiance",
                    colour=colours["Diffuse horizontal irradiance"],
                ),
            ],
            title="2. The Erbs correlation splits it into a direct beam and diffuse sky light",
            y_title="W/m²",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="poa_w_m2",
                    label="Irradiance on the fitted plane",
                    colour=colours["Irradiance on the fitted plane"],
                )
            ],
            title=(
                f"3. Both are projected onto a plane tilted {plant['tilt_deg']:.0f}° and facing "
                f"{plant['azimuth_deg']:.0f}° (due south is 180°)"
            ),
            y_title="W/m²",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="measured_mw",
                    label="Measured output",
                    colour=colours["Measured output"],
                    width=2.5,
                ),
                Line(column="fitted_mw", label="Fitted plant", colour=colours["Fitted plant"]),
            ],
            title=(
                f"4. DC capacity {plant['dc_capacity_mw']:.0f} MW, clipped at "
                f"{plant['ac_capacity_mw']:.0f} MW AC "
                f"(DC:AC {plant['dc_capacity_mw'] / plant['ac_capacity_mw']:.2f}), against "
                "the measured output"
            ),
            y_title="MW",
            last=True,
            palette=colours,
            height=PANEL_HEIGHT_PX + 20,
        ),
    ]
    return figure(
        panels=panels,
        number=number,
        title=(
            f"The forward model turns CAMS irradiance into the output of {bmu_id}, a real pure-PV "
            "BMU"
        ),
        subtitle=[
            week_subtitle(season=season),
            (
                "The fit never saw the BMU's registered capacity: tilt, azimuth, DC capacity, "
                "and AC capacity are all fitted."
            ),
        ],
        figure_planning=None,
    )


def fit_weeks_figure(*, number: int) -> alt.VConcatChart:
    """Show measured and fitted output of two pure-PV BMUs across seasons.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    colours = {"Measured output": AGGREGATE_COLOUR, "Fitted plant": RECOVERED_COLOUR}
    panels = []
    cases = [
        ("T_BURWS-1", "summer"),
        ("T_BURWS-1", "winter"),
        ("C__ESTAT019", "summer"),
        ("C__ESTAT019", "winter"),
    ]
    for index, (bmu_id, season) in enumerate(cases):
        frame = week_of(frame=sites().filter(pl.col("bmu") == bmu_id), season=season)
        start = WEEKS[season]
        panels.append(
            time_panel(
                frame=frame,
                lines=[
                    Line(
                        column="measured_mw",
                        label="Measured output",
                        colour=colours["Measured output"],
                        width=2.5,
                    ),
                    Line(column="fitted_mw", label="Fitted plant", colour=colours["Fitted plant"]),
                ],
                title=f"{bmu_id}, week of {start:%-d %B %Y}",
                y_title="MW",
                last=index == len(cases) - 1,
                palette=colours,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title="The fitted plants follow two pure-PV BMUs through a summer and a winter week",
        subtitle=[
            (
                "Half-hourly output in megawatts for the weeks containing the 2026 summer and "
                "2025 winter solstices."
            ),
            (
                "Fits use the whole year and the BMU's own CAMS point. The registered capacity "
                "is not an input."
            ),
        ],
        figure_planning=None,
    )


def separation_figure(
    *,
    number: int,
    columns: Sequence[str],
    season: str,
    title: str,
    scenario: str = SCENARIO,
    show_other: bool = False,
) -> alt.VConcatChart:
    """Show the aggregate, then each chosen method's recovered solar drawn over the true solar.

    Args:
        number: The figure's number on its page.
        columns: The columns of the scenario frame holding each method's recovered solar, one panel
            each.
        season: A key of `WEEKS`: the week to draw.
        title: The figure's title.
        scenario: The walk-through scenario to draw.
        show_other: Whether to add a panel for the other generation.

    Returns:
        The figure.
    """
    frame = week_of(frame=scenarios().filter(pl.col("scenario") == scenario), season=season)
    palette = _palette("aggregate_mw", "solar_truth_mw", "non_solar_mw", *columns)
    other_title = (
        "The other generation in the aggregate (a real battery BMU, charging when the sun is up)"
        if "battery" in scenario
        else "The other generation in the aggregate"
    )
    panels = [
        time_panel(
            frame=frame,
            lines=[_line(column="aggregate_mw")],
            title="The aggregate, which is the only series the methods see",
            y_title="MW",
            last=False,
            palette=palette,
        )
    ]
    if show_other:
        panels.append(
            time_panel(
                frame=frame,
                lines=[_line(column="non_solar_mw")],
                title=other_title,
                y_title="MW",
                last=False,
                palette=palette,
            )
        )
    for index, column in enumerate(columns):
        panels.append(
            time_panel(
                frame=frame,
                lines=[_line(column="solar_truth_mw", width=2.5), _line(column=column)],
                title=LABELS[column][0],
                y_title="MW",
                last=index == len(columns) - 1,
                palette=palette,
                height=PANEL_HEIGHT_PX + 15,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[week_subtitle(season=season)],
        figure_planning=None,
    )


def method_inputs_figure(*, number: int, season: str = "summer") -> alt.VConcatChart:
    """Show the irradiance the methods use and the four solar curves built from it.

    Args:
        number: The figure's number on its page.
        season: A key of `WEEKS`: the week to draw.

    Returns:
        The figure.
    """
    frame = week_of(frame=scenarios().filter(pl.col("scenario") == SCENARIO), season=season)
    colours: dict[str, str] = {
        "East-facing plants": ocf.DATA_BLUE,
        "South-facing plants": ocf.BRAND_ORANGE,
        "West-facing plants": ocf.DATA_PURPLE,
        "Single-axis trackers": ocf.DATA_GREEN,
        "Regional CAMS irradiance": ocf.DATA_SKY,
    }
    panels = [
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="regional_ghi_w_m2",
                    label="Regional CAMS irradiance",
                    colour=colours["Regional CAMS irradiance"],
                )
            ],
            title=(
                "1. CAMS irradiance averaged over the three grid points nearest the aggregate "
                "(not the solar sites' own points)"
            ),
            y_title="W/m²",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="basis_east",
                    label="East-facing plants",
                    colour=colours["East-facing plants"],
                ),
                Line(
                    column="basis_south",
                    label="South-facing plants",
                    colour=colours["South-facing plants"],
                ),
                Line(
                    column="basis_west",
                    label="West-facing plants",
                    colour=colours["West-facing plants"],
                ),
                Line(
                    column="basis_tracker",
                    label="Single-axis trackers",
                    colour=colours["Single-axis trackers"],
                ),
            ],
            title=(
                "2. The output of 1 MW of AC capacity in each of four orientations, clipped at 1 "
                "MW (DC:AC 1.2 to 1.4)"
            ),
            y_title="MW per MW",
            last=True,
            palette=colours,
            height=PANEL_HEIGHT_PX + 25,
        ),
    ]
    return figure(
        panels=panels,
        number=number,
        title=(
            "The methods turn weather into four candidate solar curves and ask how much of each "
            "is in the aggregate"
        ),
        subtitle=[
            week_subtitle(season=season),
            (
                "The curves clip because inverters cap the output; the ratio of DC to AC "
                "capacity sets where."
            ),
        ],
        figure_planning=None,
    )


def decomposition_figure(*, number: int, season: str = "summer") -> alt.VConcatChart:
    """Show the aggregate split into recovered solar, calendar baseline, and residual.

    Args:
        number: The figure's number on its page.
        season: A key of `WEEKS`: the week to draw.

    Returns:
        The figure.
    """
    frame = week_of(
        frame=scenarios().filter(pl.col("scenario") == SCENARIO), season=season
    ).with_columns(recovered_solar=pl.col("difference_solar_mw"))
    colours: dict[str, str] = {
        "Aggregate": AGGREGATE_COLOUR,
        "Recovered solar": ocf.DATA_SKY,
        "Calendar baseline (the rest, as a daily pattern)": ocf.DATA_BLUE,
        "Residual (what neither part explains)": ocf.GREY_3,
    }
    panels = [
        time_panel(
            frame=frame,
            lines=[Line(column="aggregate_mw", label="Aggregate", colour=colours["Aggregate"])],
            title="The aggregate",
            y_title="MW",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="recovered_solar",
                    label="Recovered solar",
                    colour=colours["Recovered solar"],
                )
            ],
            title="= recovered solar",
            y_title="MW",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="difference_baseline_mw",
                    label="Calendar baseline (the rest, as a daily pattern)",
                    colour=colours["Calendar baseline (the rest, as a daily pattern)"],
                )
            ],
            title="+ calendar baseline",
            y_title="MW",
            last=False,
            palette=colours,
        ),
        time_panel(
            frame=frame,
            lines=[
                Line(
                    column="difference_residual_mw",
                    label="Residual (what neither part explains)",
                    colour=colours["Residual (what neither part explains)"],
                )
            ],
            title="+ residual (the wind's weather-driven swings, which no calendar can follow)",
            y_title="MW",
            last=True,
            palette=colours,
        ),
    ]
    return figure(
        panels=panels,
        number=number,
        title="The three parts add up to the aggregate by construction",
        subtitle=[week_subtitle(season=season)],
        figure_planning=None,
    )


def daily_energy(*, frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Return each series' daily energy in megawatt-hours, over the days where all are present.

    Args:
        frame: Half-hourly series in megawatts, with `half_hour_end_time`.
        columns: The series to total. A day counts only if every one is present in at least 40 half-
            hours.

    Returns:
        One row per day, with `day`, each series' energy in megawatt-hours, and the count `n` of
        half-hours.
    """
    day = pl.col("half_hour_end_time").dt.offset_by("-1us").dt.truncate("1d")
    complete = frame.with_columns(pl.col(list(columns)).fill_nan(None)).drop_nulls(list(columns))
    return (
        complete.with_columns(day=day)
        .group_by("day")
        .agg(*[(pl.col(c).sum() / 2.0).alias(c) for c in columns], pl.len().alias("n"))
        .filter(pl.col("n") >= 40)
        .sort("day")
    )


def year_figure(*, number: int) -> alt.VConcatChart:
    """Show daily solar energy, true and recovered, over the whole year.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    columns = ["solar_truth_mw", "separation_solar_mw", "difference_solar_mw", "physical_solar_mw"]
    palette = _palette(*columns)
    panels = []
    names = {
        "three_pure_pv_plus_offshore_wind_25": (
            "Offshore wind as the other generation, solar at 25%"
        ),
        "three_pure_pv_plus_gas_25": "Gas baseload as the other generation, solar at 25%",
        "three_pure_pv_plus_battery_25": "A battery as the other generation, solar at 25%",
    }
    for index, (scenario, label) in enumerate(names.items()):
        frame = scenarios().filter(pl.col("scenario") == scenario)
        daily = daily_energy(frame=frame, columns=columns).rename({"day": "half_hour_end_time"})
        panels.append(
            time_panel(
                frame=daily,
                lines=[
                    _line(column="solar_truth_mw", width=2.5),
                    *[_line(column=c, width=1.2) for c in columns[1:]],
                ],
                title=label,
                y_title="MWh per day",
                last=index == len(names) - 1,
                palette=palette,
                height=PANEL_HEIGHT_PX + 20,
                date_format="%b %Y",
                tick_count="month",
                x_title="Month (UTC)",
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=(
            "Over a year, the physical fit tracks the true daily solar energy except beside a "
            "battery"
        ),
        subtitle=[
            (
                "Daily solar energy in megawatt-hours, from 1 September 2025 to 1 September "
                "2026. Days with fewer than 40 usable half-hours are left out."
            )
        ],
        figure_planning=None,
    )


def scatter_figure(*, number: int) -> alt.VConcatChart:
    """Plot recovered against true half-hourly solar, three methods, two other-generation types.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    columns = ["separation_solar_mw", "difference_solar_mw", "physical_solar_mw"]
    short = {
        "separation_solar_mw": "Calendar-baseline separation",
        "difference_solar_mw": "Difference separation",
        "physical_solar_mw": "Physical fit to changes",
    }
    rows = []
    for scenario, label in (
        (SCENARIO, "Offshore wind as the other generation"),
        (BATTERY_SCENARIO, "A battery as the other generation"),
    ):
        frame = scenarios().filter(pl.col("scenario") == scenario)
        charts = []
        for column in columns:
            long = (
                frame.select(truth="solar_truth_mw", recovered=column)
                .drop_nulls()
                .filter(pl.col("truth") > 0.05)
                .gather_every(12)
                .with_columns(pl.col("truth", "recovered").round(2))
            )
            top = float(long["truth"].max())  # ty: ignore[invalid-argument-type]
            points = (
                alt.Chart(long, title=alt.TitleParams(short[column], fontSize=11, anchor="start"))
                .mark_circle(size=6, opacity=0.25, color=LABELS[column][1], aria=False)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "truth:Q",
                        title="True solar (MW)",
                        scale=alt.Scale(domain=[0, top * 1.05], nice=False),
                    ),
                    y=alt.Y(
                        "recovered:Q",
                        title="Recovered (MW)",
                        scale=alt.Scale(domain=[0, top * 1.05], nice=False),
                    ),
                )
                .properties(width=190, height=190)
            )
            line = (
                alt.Chart(pl.DataFrame({"a": [0.0, top * 1.05]}))
                .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
                .encode(x="a:Q", y="a:Q")  # ty: ignore[unresolved-attribute]
            )
            charts.append(alt.layer(points, line))
        rows.append(
            alt.hconcat(*charts).properties(
                title=alt.TitleParams(label, fontSize=11, anchor="start", offset=2)
            )
        )
    return figure(
        panels=rows,
        number=number,
        title="Recovered solar lies on the 1:1 line beside wind and falls off it beside a battery",
        subtitle=[
            (
                "Each dot is a half-hour with solar output above 0.05 MW (every twelfth half-hour, "
                "for legibility)."
            ),
            "The dashed line is perfect recovery.",
        ],
        figure_planning=None,
    )


def physical_fit(*, regressor: str = "regional") -> pl.DataFrame:
    """Return stage 2c's physical-fit results, fitted without priors.

    Args:
        regressor: `regional` or `gb_mean`, the sky the plant was fitted with.

    Returns:
        One row per synthetic aggregate.
    """
    return pl.read_parquet(OUTPUT_DIR / "stage2c_physical_fit.parquet").filter(
        (pl.col("prior") == "none") & (pl.col("regressor") == regressor)
    )


SHARE_COLOURS: Final[dict[str, str]] = {
    "0% (no solar)": ocf.BLACK_1,
    "10%": ocf.DATA_SKY,
    "25%": ocf.DATA_BLUE,
    "50%": ocf.BRAND_ORANGE,
}
NON_SOLAR_NAMES: Final[dict[str, str]] = {
    "offshore_wind": "Offshore wind",
    "gas_peaker": "Gas peaker",
    "gas_baseload": "Gas baseload",
    "pumped_storage": "Pumped storage",
    "battery_lakeside": "Battery A",
    "battery_dollymans": "Battery B",
}


def share_label(share: float) -> str:
    """Return the legend name of a solar share.

    Args:
        share: The solar share of the aggregate.

    Returns:
        The name.
    """
    return "0% (no solar)" if share == 0 else f"{share:.0%}"


def detection_figure(*, number: int) -> alt.VConcatChart:
    """Plot how much of the changes in output the fitted plant explains, by solar share.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    frame = (
        physical_fit()
        .filter(~pl.col("weather_demand"))
        .with_columns(
            share_name=pl.col("share").map_elements(share_label, return_dtype=pl.String),
            other=pl.col("non_solar").replace(NON_SOLAR_NAMES),
        )
    )
    threshold = float(frame.filter(pl.col("share") == 0)["variance_explained"].max())  # ty: ignore[invalid-argument-type]
    order = list(SHARE_COLOURS)
    base = alt.Chart(frame).encode(
        x=alt.X(
            "variance_explained:Q",
            title="Share of the four-hour changes in output that the fitted plant explains",
            scale=alt.Scale(domain=[-0.05, 0.9], nice=False),
        ),
        y=alt.Y("share_name:N", sort=order, title="Solar share of the aggregate"),
        color=alt.Color(
            "share_name:N",
            scale=alt.Scale(domain=order, range=list(SHARE_COLOURS.values())),
            legend=None,
        ),
        yOffset=alt.YOffset("jitter:Q") if False else alt.value(0),
    )
    points = base.mark_circle(size=60, opacity=0.7, aria=False, clip=True)
    rule = (
        alt.Chart(pl.DataFrame({"x": [threshold]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    panel = alt.layer(points, rule).properties(width=PLOT_WIDTH_PX, height=150)  # ty: ignore[invalid-argument-type]
    return figure(
        panels=[panel],
        number=number,
        title=(
            "Fitted solar explains none of an aggregate's changes without solar, and most of "
            "them from a 25% share"
        ),
        subtitle=[
            (
                "Each dot is one synthetic aggregate: 4 solar halves times 6 other-generation "
                "halves for each share."
            ),
            (
                "The dashed line is the largest value over the aggregates with no solar: a share "
                "above it counts as detected."
            ),
        ],
        figure_planning=None,
    )


def parameter_error_figure(*, number: int) -> alt.VConcatChart:
    """Plot the fitted plant's parameter errors against the direct fit to the solar half alone.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    threshold = float(
        physical_fit()  # ty: ignore[invalid-argument-type]
        .filter((pl.col("share") == 0) & ~pl.col("weather_demand"))["variance_explained"]
        .max()
    )
    frame = (
        physical_fit()
        .filter(
            (pl.col("share") > 0)
            & ~pl.col("weather_demand")
            & (pl.col("variance_explained") > threshold)
        )
        .with_columns(
            share_name=pl.col("share").map_elements(share_label, return_dtype=pl.String),
            other=pl.col("non_solar").replace(NON_SOLAR_NAMES),
            tilt_error=(pl.col("fitted_tilt_deg") - pl.col("reference_tilt_deg")),
            azimuth_error=(pl.col("fitted_azimuth_deg") - pl.col("reference_azimuth_deg")),
            ratio_error=(pl.col("fitted_dc_ac_ratio") - pl.col("reference_dc_ac_ratio")),
            ac_error=(pl.col("fitted_ac_mw") / pl.col("reference_ac_mw") - 1) * 100,
        )
    )
    order = list(NON_SOLAR_NAMES.values())
    specs = [
        ("tilt_error", "Tilt error (degrees)", [-40, 40]),
        ("azimuth_error", "Azimuth error (degrees)", [-100, 100]),
        ("ratio_error", "DC:AC ratio error", [-1.2, 1.2]),
        ("ac_error", "AC capacity error (%)", [-60, 120]),
    ]
    panels = []
    for index, (column, title, domain) in enumerate(specs):
        points = (
            alt.Chart(frame)
            .mark_circle(size=55, opacity=0.75, aria=False, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(f"{column}:Q", title=title, scale=alt.Scale(domain=domain, nice=False)),
                y=alt.Y("other:N", sort=order, title=None, axis=alt.Axis(labels=True)),
                color=alt.Color(
                    "share_name:N",
                    scale=alt.Scale(
                        domain=order_shares(), range=[SHARE_COLOURS[n] for n in order_shares()]
                    ),
                    legend=alt.Legend(title="Solar share") if index == 0 else None,
                ),
            )
        )
        zero = (
            alt.Chart(pl.DataFrame({"x": [0.0]}))
            .mark_rule(color=ocf.BLACK_1, strokeWidth=0.8, aria=False)
            .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(alt.layer(zero, points).properties(width=PLOT_WIDTH_PX - 20, height=125))
    return figure(
        panels=panels,
        number=number,
        title=(
            "Beside wind and gas the plant's tilt, azimuth, DC:AC ratio, and capacity come back "
            "close; beside storage they do not"
        ),
        subtitle=[
            (
                "Error of the plant fitted to the aggregate's changes, against the plant fitted "
                "directly to the solar half's own output through the same irradiance."
            ),
            "Each dot is one detected aggregate. Zero is a perfect match.",
        ],
        figure_planning=None,
    )


def order_shares() -> list[str]:
    """Return the legend names of the three non-zero solar shares.

    Returns:
        The names, in ascending order of share.
    """
    return [name for name in SHARE_COLOURS if not name.startswith("0%")]


def dot_figure_panel(
    *,
    frame: pl.DataFrame,
    x: str,
    y: str,
    colour: str,
    x_title: str,
    colours: dict[str, str],
    y_order: Sequence[str],
    height: int,
    x_domain: tuple[float, float] | None = None,
    rule: float | None = None,
    legend: bool = False,
    size: int = 55,
) -> alt.LayerChart:
    """Draw a dot per row, with one category per line, a colour per group, and an optional rule.

    Args:
        frame: One row per dot.
        x: The column on the horizontal axis.
        y: The column that names the line a dot sits on.
        colour: The column that sets a dot's colour.
        x_title: The horizontal axis title.
        colours: The colour of each value of `colour`.
        y_order: The lines, from top to bottom.
        height: The panel's height in pixels.
        x_domain: The horizontal axis range, or None to let the data set it.
        rule: The position of a dashed vertical rule, or None for no rule.
        legend: Whether to draw the colour legend.
        size: The dot area in square pixels.

    Returns:
        The panel.
    """
    scale = alt.Scale(domain=list(x_domain), nice=False) if x_domain else alt.Scale(zero=False)
    points = (
        alt.Chart(frame)
        .mark_circle(size=size, opacity=0.8, aria=False, clip=True)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(f"{x}:Q", title=x_title, scale=scale),
            y=alt.Y(f"{y}:N", sort=list(y_order), title=None),
            color=alt.Color(
                f"{colour}:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None) if legend else None,
            ),
        )
    )
    layers = [points]
    if rule is not None:
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"r": [rule]}))
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
            .encode(x="r:Q"),  # ty: ignore[unresolved-attribute]
        )
    return cast(
        alt.LayerChart, alt.layer(*layers).properties(width=PLOT_WIDTH_PX - 70, height=height)
    )


def parameter_recovery_figure(*, number: int) -> alt.VConcatChart:
    """Plot fitted against true parameters for simulated plants, from the direct physical fit.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    frame = pl.read_parquet(OUTPUT_DIR / "stage1b_recovery.parquet")
    specs = [
        ("tilt_deg", "Tilt (degrees)", [0, 60]),
        ("azimuth_deg", "Azimuth (degrees)", [120, 240]),
        ("dc_ac_ratio", "DC:AC ratio", [1.0, 1.8]),
        ("ac_capacity_mw", "AC capacity (MW)", [40, 60]),
    ]
    charts = []
    for name, title, domain in specs:
        points = (
            alt.Chart(frame, title=alt.TitleParams(title, fontSize=11, anchor="start"))
            .mark_circle(size=40, opacity=0.75, color=ocf.DATA_BLUE, aria=False, clip=True)
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(f"true_{name}:Q", title="True", scale=alt.Scale(domain=domain, nice=False)),
                y=alt.Y(
                    f"fitted_{name}:Q", title="Fitted", scale=alt.Scale(domain=domain, nice=False)
                ),
            )
            .properties(width=(PLOT_WIDTH_PX - 60) // 2, height=150)
        )
        line = (
            alt.Chart(pl.DataFrame({"a": [float(domain[0]), float(domain[1])]}))
            .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
            .encode(x="a:Q", y="a:Q")  # ty: ignore[unresolved-attribute]
        )
        charts.append(alt.layer(points, line))
    grid = [alt.hconcat(charts[0], charts[1]), alt.hconcat(charts[2], charts[3])]
    return figure(
        panels=grid,
        number=number,
        title=(
            "The fit recovers known azimuth, DC:AC ratio, and capacity from simulated output; "
            "tilt is recovered with a bias"
        ),
        subtitle=[
            (
                "44 simulated plants at the real sites' positions, generated with a different "
                "decomposition, transposition, and a temperature derate from the fit's, with "
                "real weather-correlated noise."
            ),
            (
                "The dashed line is perfect recovery. The fit has no temperature input, so tilt "
                "absorbs part of the temperature effect."
            ),
        ],
        figure_planning=None,
    )


CAPACITY_GROUP_COLOURS: Final[dict[str, str]] = {
    "Wind and gas": ocf.DATA_BLUE,
    "Pumped storage": ocf.DATA_PURPLE,
    "Batteries": ocf.BRAND_ORANGE,
}
CAPACITY_GROUPS: Final[dict[str, str]] = {
    "offshore_wind": "Wind and gas",
    "gas_peaker": "Wind and gas",
    "gas_baseload": "Wind and gas",
    "pumped_storage": "Pumped storage",
    "battery_lakeside": "Batteries",
    "battery_dollymans": "Batteries",
}
SKY_TITLES: Final[dict[str, str]] = {
    "regional": "Regional sky (3 nearest grid points)",
    "gb_mean": "Great Britain mean sky (all 18 grid points)",
}
CAPACITY_AXIS_MAX_MW: Final[float] = 80.0


def capacity_recovery_figure(*, number: int) -> alt.VConcatChart:
    """Plot fitted against true AC capacity on the synthetic aggregates, by sky and solar share.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    frame = pl.concat([physical_fit(regressor=name) for name in SKY_TITLES]).filter(
        pl.col("share").is_in([0.25, 0.5]) & ~pl.col("weather_demand")
    )
    frame = frame.with_columns(
        other=pl.col("non_solar").replace_strict(CAPACITY_GROUPS, return_dtype=pl.String)
    )
    domain = [0.0, CAPACITY_AXIS_MAX_MW]
    rows = []
    for share in (0.25, 0.5):
        charts = []
        for regressor, sky_title in SKY_TITLES.items():
            sub = frame.filter((pl.col("share") == share) & (pl.col("regressor") == regressor))
            points = (
                alt.Chart(
                    sub,
                    title=alt.TitleParams(
                        f"{sky_title}, {share:.0%} solar share", fontSize=11, anchor="start"
                    ),
                )
                .mark_circle(size=45, opacity=0.75, aria=False, clip=True)
                .encode(  # ty: ignore[unresolved-attribute]
                    x=alt.X(
                        "reference_ac_mw:Q",
                        title="True AC capacity (MW)",
                        scale=alt.Scale(domain=domain, nice=False),
                    ),
                    y=alt.Y(
                        "fitted_ac_mw:Q",
                        title="Fitted AC capacity (MW)",
                        scale=alt.Scale(domain=domain, nice=False),
                    ),
                    color=alt.Color(
                        "other:N",
                        scale=alt.Scale(
                            domain=list(CAPACITY_GROUP_COLOURS),
                            range=list(CAPACITY_GROUP_COLOURS.values()),
                        ),
                        legend=alt.Legend(title="Other generation")
                        if share == 0.25 and regressor == "gb_mean"
                        else None,
                    ),
                )
                .properties(width=(PLOT_WIDTH_PX - 60) // 2, height=150)
            )
            line = (
                alt.Chart(pl.DataFrame({"a": domain}))
                .mark_line(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=1, aria=False)
                .encode(x="a:Q", y="a:Q")  # ty: ignore[unresolved-attribute]
            )
            charts.append(alt.layer(points, line))
        rows.append(alt.hconcat(*charts))
    return figure(
        panels=rows,
        number=number,
        title=(
            "The fitted AC capacity falls below the true capacity, most of all beside pumped "
            "storage and batteries"
        ),
        subtitle=[
            (
                "Each dot is one synthetic aggregate, fitted without priors: 4 solar halves times "
                "6 other-generation halves for each share."
            ),
            (
                "The dashed line is perfect recovery. The true capacity is the direct fit to the "
                "solar half, scaled to the share."
            ),
        ],
        figure_planning=None,
    )


METHOD_COLOURS: Final[dict[str, str]] = {
    "Physical fit": RECOVERED_COLOUR,
    "Envelope fit, CAMS at the BMU's own point": ocf.DATA_BLUE,
    "99th percentile of output": ocf.DATA_PURPLE,
    "Largest output": ocf.BRAND_ORANGE,
}


def site_capacity_figure(*, number: int) -> alt.VConcatChart:
    """Plot each estimate of AC capacity at the 11 single-site BMUs, as a share of capacity.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    baselines = pl.read_parquet(OUTPUT_DIR / "stage1_baselines.parquet")
    fits = pl.read_parquet(OUTPUT_DIR / "stage1_fits.parquet").filter(
        (pl.col("fold") == -1) & (pl.col("arm") == "free")
    )
    frame = baselines.join(fits.select("bmu", "ac_capacity_mw"), on="bmu")
    long = pl.concat(
        [
            frame.select(
                bmu="bmu",
                method=pl.lit(label),
                ratio=pl.col(column) / pl.col("generation_capacity_mw"),
            )
            for label, column in (
                ("Physical fit", "ac_capacity_mw"),
                ("Envelope fit, CAMS at the BMU's own point", "envelope_cams_mw"),
                ("99th percentile of output", "p99_output_mw"),
                ("Largest output", "max_output_mw"),
            )
        ]
    )
    order = frame.sort("generation_capacity_mw")["bmu"].to_list()
    panel = dot_figure_panel(
        frame=long,
        x="ratio",
        y="bmu",
        colour="method",
        x_title="Estimate divided by the BMU's registered Generation Capacity",
        colours=METHOD_COLOURS,
        y_order=order,
        height=300,
        x_domain=(0.2, 1.3),
        rule=1.0,
        legend=True,
    )
    return figure(
        panels=[panel],
        number=number,
        title=(
            "The fitted AC capacity is close to the registered capacity at some single-site BMUs "
            "and well below it at others"
        ),
        subtitle=[
            (
                "AC capacity at the 11 single-site BMUs that follow the sun, each with the "
                "registered capacity hidden from every fit."
            ),
            (
                "The dashed line is the Generation Capacity. A site whose output never reaches "
                "it, or that is capped, sits left of the line."
            ),
        ],
        figure_planning=None,
    )


def heldout_figure(*, number: int) -> alt.VConcatChart:
    """Plot held-out error of the three orientation arms at each site.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    predictions = pl.read_parquet(OUTPUT_DIR / "stage1_predictions.parquet")
    capacities = pl.read_parquet(OUTPUT_DIR / "stage1_baselines.parquet").select(
        "bmu", "generation_capacity_mw"
    )
    by_site = (
        predictions.join(capacities, on="bmu")
        .with_columns(
            error=(pl.col("output_mw") - pl.col("predicted_mw")).abs()
            / pl.col("generation_capacity_mw")
            * 100
        )
        .group_by("bmu", "arm", "fold")
        .agg(pl.col("error").mean())
        .group_by("bmu", "arm")
        .agg(pl.col("error").mean())
        .with_columns(
            arm=pl.col("arm").replace(
                {
                    "free": "Tilt and azimuth fitted",
                    "fixed": "Tilt 30°, due south",
                    "tracker": "Single-axis tracker",
                }
            )
        )
    )
    colours = {
        "Tilt and azimuth fitted": RECOVERED_COLOUR,
        "Tilt 30°, due south": ocf.DATA_BLUE,
        "Single-axis tracker": ocf.BRAND_ORANGE,
    }
    order = (
        by_site.filter(pl.col("arm") == "Tilt and azimuth fitted").sort("error")["bmu"].to_list()
    )
    panel = dot_figure_panel(
        frame=by_site,
        x="error",
        y="bmu",
        colour="arm",
        x_title="Held-out mean absolute error (% of Generation Capacity; smaller is better)",
        colours=colours,
        y_order=order,
        height=300,
        legend=True,
    )
    return figure(
        panels=[panel],
        number=number,
        title=(
            "Fitting each site's tilt and azimuth gives the lowest held-out error at 10 of 11 sites"
        ),
        subtitle=[
            (
                "Each fit uses three of four contiguous three-month blocks and is scored on the "
                "block it did not see. Dots are means over the four blocks."
            ),
        ],
        figure_planning=None,
    )


def stage3_series() -> pl.DataFrame:
    """Return the real aggregates' output and recovered solar.

    Returns:
        One row per half-hour per aggregate BMU.
    """
    return pl.read_parquet(OUTPUT_DIR / "stage3_series.parquet")


LARGE_BMU_MW: Final[float] = 10.0
"""The smallest fitted AC capacity that the capacity figure draws."""
REAL_AGGREGATES: Final[tuple[str, ...]] = (
    "2__ATGPL000",
    "2__HTGPL000",
    "2__BTGPL000",
    "2__LTGPL000",
)
"""The four largest aggregate BMUs whose output the physical fit explains: the TotalEnergies
supplier BMUs with the highest 99th-percentile output among those with a detected solar part."""


def problem_figure(*, number: int, season: str = "summer") -> alt.VConcatChart:
    """Show real aggregate BMUs' output beside the irradiance.

    Args:
        number: The figure's number on its page.
        season: A key of `WEEKS`: the week to draw.

    Returns:
        The figure.
    """
    series = stage3_series()
    scenario = (
        scenarios()
        .filter(pl.col("scenario") == SCENARIO)
        .select("half_hour_end_time", "regional_ghi_w_m2")
    )
    palette = {"Output of the BMU": AGGREGATE_COLOUR, "CAMS irradiance (regional)": ocf.DATA_SKY}
    panels = []
    for bmu_id in REAL_AGGREGATES[:3]:
        frame = week_of(frame=series.filter(pl.col("bmu") == bmu_id), season=season)
        panels.append(
            time_panel(
                frame=frame,
                lines=[
                    Line(column="output_mw", label="Output of the BMU", colour=AGGREGATE_COLOUR)
                ],
                title=f"{bmu_id}: a supplier BMU, net of the demand and generation it carries",
                y_title="MW",
                last=False,
                palette=palette,
            )
        )
    panels.append(
        time_panel(
            frame=week_of(frame=scenario, season=season),
            lines=[
                Line(
                    column="regional_ghi_w_m2",
                    label="CAMS irradiance (regional)",
                    colour=palette["CAMS irradiance (regional)"],
                )
            ],
            title="CAMS irradiance, for comparison",
            y_title="W/m²",
            last=True,
            palette=palette,
        )
    )
    return figure(
        panels=panels,
        number=number,
        title=(
            "Real aggregate BMUs carry a daytime signal that no public source attributes to a "
            "technology"
        ),
        subtitle=[
            week_subtitle(season=season),
            "Supplier BMUs register no fuel type, so what they hold is not stated.",
        ],
        figure_planning=None,
    )


def real_weeks_figure(*, number: int, season: str, bmus: Sequence[str]) -> alt.VConcatChart:
    """Show real aggregate output, the recovered solar, and the output with that solar removed.

    Args:
        number: The figure's number on its page.
        season: A key of `WEEKS`: the week to draw.
        bmus: The aggregate BMUs to draw, one block each.

    Returns:
        The figure.
    """
    series = stage3_series().with_columns(
        without_solar_mw=pl.col("output_mw") - pl.col("physical_solar_mw")
    )
    colours = {
        "Output": ocf.GREY_3,
        "Output with the recovered solar removed": AGGREGATE_COLOUR,
        "Recovered solar (physical fit to changes)": RECOVERED_COLOUR,
    }
    panels = []
    for index, bmu_id in enumerate(bmus):
        frame = week_of(frame=series.filter(pl.col("bmu") == bmu_id), season=season)
        panels.append(
            time_panel(
                frame=frame,
                lines=[
                    Line(column="output_mw", label="Output", colour=colours["Output"], width=3),
                    Line(
                        column="without_solar_mw",
                        label="Output with the recovered solar removed",
                        colour=colours["Output with the recovered solar removed"],
                        width=1.2,
                    ),
                    Line(
                        column="physical_solar_mw",
                        label="Recovered solar (physical fit to changes)",
                        colour=colours["Recovered solar (physical fit to changes)"],
                    ),
                ],
                title=bmu_id,
                y_title="MW",
                last=index == len(bmus) - 1,
                palette=colours,
                height=PANEL_HEIGHT_PX + 25,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=(
            "The recovered solar accounts for the daytime hump of real supplier BMUs in a "
            f"{season} week"
        ),
        subtitle=[
            week_subtitle(season=season),
            (
                "There is no answer key: the check is whether the output with the solar removed "
                "still follows the sun."
            ),
        ],
        figure_planning=None,
    )


def real_controls_figure(*, number: int) -> alt.VConcatChart:
    """Plot the cloud increment for real BMUs of known technology, the aggregates, and replicas.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    frame = pl.read_parquet(OUTPUT_DIR / "stage3b_real_controls.parquet")
    frame = frame.with_columns(
        row=pl.when(pl.col("group") == "negative control")
        .then(pl.lit("Not solar: ") + pl.col("fuel").str.to_titlecase())
        .when(pl.col("group") == "positive control")
        .then(pl.lit("Solar BMUs (single site)"))
        .when(pl.col("group") == "calendar replica")
        .then(pl.lit("Calendar replicas"))
        .otherwise(pl.lit("Aggregate BMUs"))
    )
    order = [
        "Aggregate BMUs",
        "Calendar replicas",
        "Solar BMUs (single site)",
        "Not solar: Wind",
        "Not solar: Ccgt",
        "Not solar: Ps",
        "Not solar: Npshyd",
        "Not solar: Nuclear",
        "Not solar: Biomass",
        "Not solar: Battery",
    ]
    colours = {
        "aggregate": ocf.BRAND_ORANGE,
        "calendar replica": ocf.DATA_PURPLE,
        "positive control": RECOVERED_COLOUR,
        "negative control": ocf.BLACK_1,
    }
    panel = dot_figure_panel(
        frame=frame,
        x="cloud_increment",
        y="row",
        colour="group",
        x_title=(
            "Cloud increment: share of four-hour changes explained with cloud, minus with clear sky"
        ),
        colours=colours,
        y_order=order,
        height=290,
        x_domain=(-0.15, 0.8),
        rule=0.0,
    )
    return figure(
        panels=[panel],
        number=number,
        title="Cloud information adds little to the fit of the known solar BMUs and the aggregates",
        subtitle=[
            (
                "Every BMU gets the same protocol: all-GB mean irradiance, no position, a "
                "free-orientation plant fitted to four-hour changes in output, once with cloud and "
                "once with CAMS's clear-sky irradiance."
            ),
            (
                "A calendar replica is the BMU's mean output by month, half-hour of the day, and "
                "day type: it has the BMU's daily shape and no cloud. Non-solar controls are the "
                "12 largest BMUs of each type."
            ),
        ],
        figure_planning=None,
    )


def real_capacity_figure(*, number: int) -> alt.VConcatChart:
    """Plot the recovered solar capacity of each real aggregate BMU beside the registered one.

    Args:
        number: The figure's number on its page.

    Returns:
        The figure.
    """
    separations = pl.read_parquet(OUTPUT_DIR / "stage3_separations.parquet")
    stage0 = pl.read_parquet(OUTPUT_DIR / "stage0_decision_rule.parquet").select(
        "bmu", "generation_capacity_mw", "envelope_cams_mw"
    )
    frame = separations.join(stage0, on="bmu").filter(
        (pl.col("physical_variance_explained") > 0.5) & (pl.col("physical_ac_mw") >= LARGE_BMU_MW)
    )
    long = pl.concat(
        [
            frame.select("bmu", method=pl.lit(label), mw=pl.col(column))
            for label, column in (
                ("Physical fit to changes", "physical_ac_mw"),
                ("Envelope fit, all-GB CAMS mean", "envelope_cams_mw"),
                ("Registered Generation Capacity", "generation_capacity_mw"),
            )
        ]
    )
    colours = {
        "Physical fit to changes": RECOVERED_COLOUR,
        "Envelope fit, all-GB CAMS mean": ocf.DATA_BLUE,
        "Registered Generation Capacity": ocf.BLACK_1,
    }
    order = frame.sort("physical_ac_mw", descending=True)["bmu"].to_list()
    panel = dot_figure_panel(
        frame=long,
        x="mw",
        y="bmu",
        colour="method",
        x_title="AC capacity (MW)",
        colours=colours,
        y_order=order,
        height=40 * len(order),
        legend=True,
        size=90,
        x_domain=(0, 320),
    )
    return figure(
        panels=[panel],
        number=number,
        title=(
            "Recovered solar capacity of the larger aggregate BMUs, beside the envelope estimate "
            "and the registered capacity"
        ),
        subtitle=[
            (
                f"The aggregate BMUs whose fitted AC capacity is at least {LARGE_BMU_MW:.0f} MW "
                "and whose fitted plant explains more than half of the four-hour changes in "
                "output."
            ),
            (
                "A supplier BMU's registered capacity covers all its technologies and its "
                "demand, so it is not a target. 2__HTGPL000's registered 912.5 MW is off the "
                "scale."
            ),
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write every figure as an SVG under `docs/studies/assets/`."""
    charts = {
        "build": build_figure(number=9),
        "forward_model_summer": forward_model_figure(number=3, bmu_id="T_BURWS-1", season="summer"),
        "fit_weeks": fit_weeks_figure(number=4),
        "method_inputs": method_inputs_figure(number=10),
        "calendar_separation_summer": separation_figure(
            number=11,
            columns=["separation_solar_mw"],
            season="summer",
            title=(
                "The calendar-baseline separation recovers the shape of the solar part but far "
                "too little of it"
            ),
        ),
        "difference_separation_summer": separation_figure(
            number=12,
            columns=["difference_solar_mw"],
            season="summer",
            title="Fitting changes over four hours recovers more of the solar part",
        ),
        "physical_fit_summer": separation_figure(
            number=1,
            columns=["physical_solar_mw"],
            season="summer",
            title="A physical plant fitted to the changes recovers the solar part in a summer week",
        ),
        "decomposition": decomposition_figure(number=13),
        "year": year_figure(number=14),
        "battery_week": separation_figure(
            number=16,
            columns=["difference_solar_mw", "physical_solar_mw"],
            season="summer",
            title="Beside a battery that charges from the sun, no method separates the solar part",
            scenario=BATTERY_SCENARIO,
            show_other=True,
        ),
        "scatter": scatter_figure(number=15),
        "detection": detection_figure(number=17),
        "parameter_errors": parameter_error_figure(number=18),
        "parameter_recovery": parameter_recovery_figure(number=6),
        "site_capacity": site_capacity_figure(number=7),
        "capacity_recovery": capacity_recovery_figure(number=8),
        "heldout": heldout_figure(number=5),
        "problem": problem_figure(number=2),
        "real_controls": real_controls_figure(number=19),
        "real_weeks_summer": real_weeks_figure(number=20, season="summer", bmus=REAL_AGGREGATES),
        "real_weeks_winter": real_weeks_figure(
            number=21, season="winter", bmus=REAL_AGGREGATES[:2]
        ),
        "real_capacity": real_capacity_figure(number=22),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, chart in charts.items():
        chart.save(ASSETS_DIR / f"solar_disaggregation_{name}.svg")
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {ASSETS_DIR / f'solar_disaggregation_{name}.svg'}")


if __name__ == "__main__":
    main()
