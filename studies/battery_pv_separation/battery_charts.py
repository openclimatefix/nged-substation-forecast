"""Draw the battery study's figures for rungs 1 to 4.

Every figure is built from the tables the rung scripts saved, so nothing is refitted here. The
batteries, the prices, and B1610 are public, so the figures name each battery and show calendar
dates and megawatts.

Run after `battery_rung1.py`, `battery_rung2.py`, `battery_rung3.py`, `battery_rung4.py`, and
`battery_rung4_summary.py`:
`uv run python studies/battery_pv_separation/battery_charts.py`.
"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from battery_inputs import (
    BATTERIES,
    NAMES,
    OUTPUT_DIR,
    WEEK_START,
    market_frame,
)
from battery_rung4 import ARM_LABELS, ARMS
from studies.charts import CONTENT_WIDTH_PX, figure

ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
BACKGROUND_ASSETS_DIR: Final[Path] = Path("docs/background/assets")
"""Where the primer figures go: the background page on how GB batteries schedule themselves."""
PLOT_WIDTH_PX: Final[int] = CONTENT_WIDTH_PX - 95
PANEL_HEIGHT_PX: Final[int] = 110
OUTPUT_COLOUR: Final[str] = ocf.BRAND_ORANGE
PRICE_COLOUR: Final[str] = ocf.DATA_BLUE
SECOND_COLOUR: Final[str] = ocf.BLACK_1
Layer = alt.LayerChart
RUNG1_PALETTE: Final[dict[str, str]] = {
    "Lakeside output": OUTPUT_COLOUR,
    "Day-ahead price (N2EX, hourly)": PRICE_COLOUR,
    "System price (half-hourly)": SECOND_COLOUR,
}
RUNG2_PALETTE: Final[dict[str, str]] = {
    "Actual": ocf.BLACK_1,
    "Time-of-day mean only": ocf.DATA_BLUE,
    "Time of day plus day-ahead price rank and level": ocf.BRAND_ORANGE,
}
RUNG3_WEEK_PALETTE: Final[dict[str, str]] = {
    "Output": OUTPUT_COLOUR,
    "State of charge": PRICE_COLOUR,
}


def _week(*, frame: pl.DataFrame, column: str = "time") -> pl.DataFrame:
    """Return the rows of the study's week.

    Args:
        frame: A frame with a UTC time column.
        column: The time column.

    Returns:
        The rows from `WEEK_START` for seven days.
    """
    return frame.filter(
        (pl.col(column) >= WEEK_START) & (pl.col(column) < WEEK_START + timedelta(days=7))
    )


def line_panel(
    *,
    series: Mapping[str, tuple[pl.DataFrame, str, str]],
    title: str,
    y_title: str,
    last: bool,
    interpolate: str = "linear",
    rules: dict[str, float] | None = None,
    time_format: str = "%a %-d %b",
    tick_count: str = "day",
    height: int = PANEL_HEIGHT_PX,
    x_title: str = "Day (UTC)",
    dashes: dict[str, list[int]] | None = None,
    palette: dict[str, str] | None = None,
) -> Layer:
    """Draw one panel of lines over time.

    Args:
        series: For each legend name, the frame with `time` and a value column, the value column's
            name, and the line colour.
        title: The panel's title.
        y_title: The y axis title.
        last: Whether the panel is the bottom one, which carries the time axis labels.
        interpolate: The Vega-Lite line interpolation; `step-after` draws a price held to its
            next change.
        rules: Horizontal reference rules, by label, at the given y values.
        time_format: The time axis label format.
        tick_count: The time axis tick spacing.
        height: The panel height in pixels.
        x_title: The time axis title.
        dashes: Dash patterns by legend name.
        palette: Every series name in the whole figure with its colour. The panels of one figure
            share a colour scale, so each panel passes the figure's full palette.

    Returns:
        The panel.
    """
    long = pl.concat(
        [
            frame.select(time="time", value=pl.col(column), series=pl.lit(name))
            for name, (frame, column, _) in series.items()
        ]
    )
    names = list(series)
    full = palette or {name: colour for name, (_, _, colour) in series.items()}
    dash_values = [(dashes or {}).get(name, [1, 0]) for name in names]
    chart = (
        alt.Chart(long, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
        .mark_line(interpolate=interpolate, aria=False, strokeWidth=1.4)  # ty: ignore[invalid-argument-type]
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T",
                axis=alt.Axis(
                    format=time_format,
                    tickCount=tick_count,  # ty: ignore[invalid-argument-type]
                    labels=last,
                    ticks=last,
                    title=x_title if last else None,
                ),
            ),
            y=alt.Y("value:Q", axis=alt.Axis(tickCount=4, title=y_title)),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=list(full), range=list(full.values())),
                legend=alt.Legend(title=None, columns=3, labelLimit=330, symbolLimit=0),
            ),
            strokeDash=alt.StrokeDash(
                "series:N", scale=alt.Scale(domain=names, range=dash_values), legend=None
            ),
        )
        .properties(width=PLOT_WIDTH_PX, height=height)
    )
    layers = [chart]
    for label, value in (rules or {}).items():
        layers.append(
            alt.Chart(pl.DataFrame({"y": [value], "label": [label]}))
            .mark_rule(color=ocf.BLACK_1, strokeWidth=0.7, strokeDash=[4, 3], aria=False)
            .encode(y="y:Q")  # ty: ignore[unresolved-attribute]
        )
        layers.append(
            alt.Chart(pl.DataFrame({"y": [value], "label": [label]}))
            .mark_text(align="left", dx=3, dy=-5, fontSize=9, color=ocf.BLACK_1, aria=False)
            .encode(x=alt.value(2), y="y:Q", text="label:N")  # ty: ignore[unresolved-attribute]
        )
    return cast(Layer, alt.layer(*layers))


def _battery_week(*, bmu_id: str) -> pl.DataFrame:
    """Return one battery's out-of-fold predictions for the study's week.

    Args:
        bmu_id: The battery.

    Returns:
        Rows of `rung2_out_of_fold.parquet` in the week.
    """
    frame = pl.read_parquet(OUTPUT_DIR / "rung2_out_of_fold.parquet").filter(
        pl.col("bmu_id") == bmu_id
    )
    return _week(frame=frame)


def rung1_week_figure(*, number: int) -> alt.VConcatChart:
    """Draw the week of Lakeside output above the week's prices.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    bmu_id = BATTERIES[0]
    output = _week(
        frame=pl.read_parquet(OUTPUT_DIR / "rung2_out_of_fold.parquet").filter(
            pl.col("bmu_id") == bmu_id
        )
    )
    prices = _week(frame=market_frame())
    top = line_panel(
        series={"Lakeside output": (output, "output_mw", OUTPUT_COLOUR)},
        title="Output (positive: exporting to the grid; negative: charging)",
        y_title="MW",
        last=False,
        interpolate="step-after",
        rules={"zero": 0.0},
        palette=RUNG1_PALETTE,
    )
    bottom = line_panel(
        series={
            "Day-ahead price (N2EX, hourly)": (prices, "day_ahead_gbp_per_mwh", PRICE_COLOUR),
            "System price (half-hourly)": (prices, "system_price_gbp_per_mwh", SECOND_COLOUR),
        },
        title="Prices",
        y_title="£ per MWh",
        last=True,
        interpolate="step-after",
        dashes={"System price (half-hourly)": [3, 2]},
        palette=RUNG1_PALETTE,
    )
    return figure(
        panels=[top, bottom],
        number=number,
        title=(
            "Lakeside battery charged when the day-ahead price was low and exported when it "
            "was high"
        ),
        subtitle=[
            (
                "One week from Monday 15 December 2025, UTC. Output is settled half-hourly "
                "metering (B1610) turned into megawatts."
            )
        ],
        figure_planning=None,
    )


def rung2_figure(*, number: int) -> alt.VConcatChart:
    """Draw predicted against actual output for a week, and the held-out error contrast.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    bmu_id = BATTERIES[0]
    week = _battery_week(bmu_id=bmu_id)
    top = line_panel(
        series={
            "Actual": (week, "output_mw", ocf.BLACK_1),
            "Time-of-day mean only": (week, "time_of_day", ocf.DATA_BLUE),
            "Time of day plus day-ahead price rank and level": (
                week,
                "price_rank",
                ocf.BRAND_ORANGE,
            ),
        },
        title="Lakeside output, predicted by fits that never saw this week's three-month block",
        y_title="MW",
        last=True,
        interpolate="step-after",
        rules={"zero": 0.0},
        height=150,
        palette=RUNG2_PALETTE,
    )
    intervals = pl.read_parquet(OUTPUT_DIR / "rung2_intervals.parquet")
    labels = []
    for key in (*BATTERIES, "pooled"):
        sub = intervals.filter(pl.col("bmu_id") == key)
        base = sub.filter(pl.col("quantity") == "time_of_day")["mean"][0]
        full = sub.filter(pl.col("quantity") == "price_rank")["mean"][0]
        name = "Four batteries pooled" if key == "pooled" else NAMES[key]
        labels.append(
            sub.filter(pl.col("quantity").str.starts_with("price_rank minus")).with_columns(
                label=pl.lit(f"{name}: {base:.1f} to {full:.1f}"),
                kind=pl.lit("pooled" if key == "pooled" else "one battery"),
            )
        )
    table = pl.concat(labels)
    order = table["label"].to_list()
    rule = (
        alt.Chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=0.8, aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    layers = [rule]
    for kind, colour, shape in (
        ("one battery", ocf.BRAND_ORANGE, "circle"),
        ("pooled", ocf.DATA_BLUE, "diamond"),
    ):
        part = alt.Chart(table.filter(pl.col("kind") == kind))
        layers.append(
            part.mark_rule(strokeWidth=2, color=colour, aria=False).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("label:N", sort=order, axis=alt.Axis(title=None, labelLimit=330)),
                x=alt.X(
                    "lower_95:Q",
                    title="Price-aware error minus time-of-day error (points of the battery's "
                    "99th percentile output; more negative is better)",
                ),
                x2="upper_95:Q",
            )
        )
        layers.append(
            part.mark_point(filled=True, size=70, color=colour, shape=shape, aria=False).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("label:N", sort=order), x="mean:Q"
            )
        )
    bottom = cast(Layer, alt.layer(*layers)).properties(width=PLOT_WIDTH_PX - 120, height=130)
    return figure(
        panels=[top, bottom],
        number=number,
        title=(
            "Knowing the day's price ranking lowers a battery's prediction error, but only slightly"
        ),
        subtitle=[
            (
                "Mean absolute error as a percentage of each battery's 99th-percentile absolute "
                "output, on four held-out three-month blocks. Row labels give the error without "
                "and with prices, in that order. Dot: estimate. Line: 95% interval from "
                "resampling whole calendar months. Orange: one battery. Blue: the four batteries "
                "pooled. Zero: no difference."
            )
        ],
        figure_planning="planned",
    )


def rung3_week_figure(*, number: int) -> alt.VConcatChart:
    """Draw Lakeside's output and fitted state of charge for the study's week.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    paths = pl.read_parquet(OUTPUT_DIR / "rung3_soc_paths.parquet")
    sub = paths.filter(pl.col("bmu_id") == BATTERIES[0])
    capacity = float(sub["capacity_mwh"][0])
    eta = float(sub["eta"][0])
    week = _week(frame=sub)
    top = line_panel(
        series={"Output": (week, "output_mw", OUTPUT_COLOUR)},
        title="Output (positive: exporting; negative: charging)",
        y_title="MW",
        last=False,
        interpolate="step-after",
        rules={"zero": 0.0},
        palette=RUNG3_WEEK_PALETTE,
    )
    bottom = line_panel(
        series={"State of charge": (week, "soc_mwh", PRICE_COLOUR)},
        title=f"Implied state of charge, with one-way efficiency {eta:.3f} and capacity "
        f"{capacity:.0f} MWh",
        y_title="MWh",
        last=True,
        rules={"empty": 0.0},
        palette=RUNG3_WEEK_PALETTE,
    )
    return figure(
        panels=[top, bottom],
        number=number,
        title="Adding up a battery's output gives a state of charge that rises and falls with it",
        subtitle=[
            (
                "Lakeside, one week from Monday 15 December 2025, UTC. The path is the year's fit, "
                "which Figure 6 shows in full."
            )
        ],
        figure_planning=None,
    )


SUMMER_WEEK_START: Final[datetime] = datetime(2026, 6, 22, tzinfo=UTC)
"""The Monday of the week containing the summer solstice, fixed by the calendar."""
SHARE_LABELS: Final[dict[float, str]] = {
    0.1: "Solar is 10%",
    0.25: "Solar is 25%",
    0.5: "Solar is 50%",
}
SHARE_COLOURS: Final[dict[float, str]] = {
    0.1: ocf.BRAND_ORANGE,
    0.25: ocf.DATA_BLUE,
    0.5: ocf.DATA_PURPLE,
}
SHARE_SHAPES: Final[dict[float, str]] = {0.1: "circle", 0.25: "diamond", 0.5: "square"}
RUNG4_COLOURS: Final[dict[str, str]] = {
    "True solar": ocf.BRAND_ORANGE,
    "True battery": ocf.DATA_BLUE,
    "Aggregate (what the fit sees)": ocf.BLACK_1,
    "Solar-only fit (A0)": ocf.DATA_PURPLE,
    "Fit with price level and rank (A1)": ocf.DATA_PURPLE,
    "Fit with price level, rank, and spread (A2)": ocf.DATA_GREEN,
    "Fit with the true battery (A3)": ocf.DATA_PURPLE,
}
RUNG4_DASHES: Final[dict[str, list[int]]] = {
    "True solar": [4, 3],
    "True battery": [4, 3],
    "Fit with price level, rank, and spread (A2)": [6, 3],
}


def _rung4_week(*, arm: str) -> pl.DataFrame:
    """Return one arm's fitted series for the summer week.

    Args:
        arm: The arm.

    Returns:
        The rows of `rung4_series.parquet` for the arm in the week, with the true series.
    """
    frame = (
        pl.read_parquet(OUTPUT_DIR / "rung4_series.parquet")
        .filter(pl.col("arm") == arm)
        .with_columns(pl.col("time").dt.replace_time_zone("UTC"))
    )
    return frame.filter(
        (pl.col("time") >= SUMMER_WEEK_START)
        & (pl.col("time") < SUMMER_WEEK_START + timedelta(days=7))
    )


def _rung4_week_figure(
    *, number: int, title: str, subtitle: str, arms: dict[str, str]
) -> alt.VConcatChart:
    """Draw a summer week of the Burwell-plus-Lakeside aggregate and the fits of some arms.

    Args:
        number: The figure number.
        title: The figure title, stating the conclusion.
        subtitle: The subtitle's second sentence onwards.
        arms: For each arm to draw, the legend name of its recovered solar series.

    Returns:
        The figure: truth on top, the recovered solar below, and the fitted battery contribution
        at the bottom.
    """
    first = _rung4_week(arm=next(iter(arms)))
    truth = {
        "True solar": (first, "solar_truth_mw", RUNG4_COLOURS["True solar"]),
        "True battery": (first, "battery_truth_mw", RUNG4_COLOURS["True battery"]),
    }
    recovered = {"True solar": truth["True solar"]}
    contribution = {"True battery": truth["True battery"]}
    for arm, name in arms.items():
        frame = _rung4_week(arm=arm)
        recovered[name] = (frame, "recovered_solar_mw", RUNG4_COLOURS[name])
        if arm != "A0":
            contribution[name] = (frame, "regressor_contribution_mw", RUNG4_COLOURS[name])
    names = [
        "True solar",
        "True battery",
        "Aggregate (what the fit sees)",
        *arms.values(),
    ]
    palette = {name: RUNG4_COLOURS[name] for name in names}
    top = line_panel(
        series={
            **truth,
            "Aggregate (what the fit sees)": (first, "aggregate_mw", ocf.BLACK_1),
        },
        title="The aggregate: Burwell solar (25% of it) plus the Lakeside battery",
        y_title="MW",
        last=False,
        rules={"zero": 0.0},
        palette=palette,
        height=130,
        dashes=RUNG4_DASHES,
    )
    middle = line_panel(
        series=recovered,
        title="Solar recovered by the fit, against the true solar",
        y_title="MW",
        last=len(contribution) == 1,
        rules={"zero": 0.0},
        palette=palette,
        height=130,
        dashes=RUNG4_DASHES,
    )
    panels = [top, middle]
    if len(contribution) > 1:
        panels.append(
            line_panel(
                series=contribution,
                title="Battery output fitted from the regressors, against the true battery",
                y_title="MW",
                last=True,
                rules={"zero": 0.0},
                palette=palette,
                height=130,
                dashes=RUNG4_DASHES,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=title,
        subtitle=[subtitle],
        figure_planning=None,
    )


def rung4_a0_figure(*, number: int) -> alt.VConcatChart:
    """Draw the solar-only fit's recovered solar for a summer week.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    return _rung4_week_figure(
        number=number,
        title=(
            "A solar-only fit to a solar-plus-battery sum recovers only a small part of the solar"
        ),
        subtitle=(
            "One week from Monday 22 June 2026, UTC. The aggregate is 25% Burwell solar and 75% "
            "Lakeside battery by 99th-percentile output, and the 99th percentile of the sum is "
            "100 MW. The fit used the regional CAMS sky and four-hour changes of the whole year."
        ),
        arms={"A0": "Solar-only fit (A0)"},
    )


def rung4_prices_figure(*, number: int) -> alt.VConcatChart:
    """Draw the price-regressor fits' recovered solar and battery contribution.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    return _rung4_week_figure(
        number=number,
        title=(
            "Price regressors recover more of the solar, but the fitted battery output follows "
            "the real battery only loosely"
        ),
        subtitle=(
            "The same week and aggregate as Figure 5. The regressors are the day-ahead price "
            "minus its day's mean and the day's price rank (A1), and the system price minus the "
            "day-ahead price as well (A2)."
        ),
        arms={
            "A1": "Fit with price level and rank (A1)",
            "A2": "Fit with price level, rank, and spread (A2)",
        },
    )


def rung4_oracle_figure(*, number: int) -> alt.VConcatChart:
    """Draw the oracle fit's recovered solar and battery contribution.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    return _rung4_week_figure(
        number=number,
        title="Given the true battery output, the fit recovers the solar closely: the upper bound",
        subtitle=(
            "The same week and aggregate as Figure 5. The oracle fit (A3) is handed the true "
            "battery output as a regressor, which no real analyst has. The remaining gap to the "
            "true solar is the error of the sky and of the plant model."
        ),
        arms={"A3": "Fit with the true battery (A3)"},
    )


def _arm_dot_panel(
    *,
    table: pl.DataFrame,
    title: str,
    x_title: str,
    last: bool,
    reference: float | None = None,
    show_key: bool = False,
    width: int = PLOT_WIDTH_PX - 150,
    arms: tuple[str, ...] = ARMS,
    labels: Mapping[str, str] = ARM_LABELS,
    height: int = 125,
) -> Layer:
    """Draw one dot-and-interval panel with a row per arm and a mark per solar share.

    Args:
        table: Columns `label`, `share`, `mean`, `lower_95`, and `upper_95`.
        title: The panel title.
        x_title: The x axis title.
        last: Whether the panel carries the x axis labels.
        reference: A vertical reference line, such as 1 for a perfect ratio.
        show_key: Whether the panel carries the share legend.
        width: The plot width in pixels.
        arms: The arms to draw, top to bottom.
        labels: Each arm's label in `table`.
        height: The plot height in pixels.

    Returns:
        The panel.
    """
    frame = table.with_columns(
        share_label=pl.col("share").replace_strict(SHARE_LABELS, return_dtype=pl.String)
    )
    order = [labels[arm] for arm in arms]
    share_labels = list(SHARE_LABELS.values())
    colour = alt.Color(
        "share_label:N",
        scale=alt.Scale(domain=share_labels, range=list(SHARE_COLOURS.values())),
        legend=alt.Legend(title=None, orient="top", columns=3) if show_key else None,
    )
    shape = alt.Shape(
        "share_label:N",
        scale=alt.Scale(domain=share_labels, range=list(SHARE_SHAPES.values())),
        legend=alt.Legend(title=None, orient="top", columns=3) if show_key else None,
    )
    y = alt.Y("label:N", sort=order, axis=alt.Axis(title=None, labelLimit=250))
    offset = alt.YOffset("share_label:N", scale=alt.Scale(domain=share_labels))
    base = alt.Chart(frame)
    layers = [
        base.mark_rule(strokeWidth=2, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=y,
            yOffset=offset,
            x=alt.X(
                "lower_95:Q",
                title=x_title if last else None,
                axis=alt.Axis(labels=last, ticks=last),
            ),
            x2="upper_95:Q",
            color=colour,
        ),
        base.mark_point(filled=True, size=55, aria=False).encode(  # ty: ignore[unresolved-attribute]
            y=y, yOffset=offset, x="mean:Q", color=colour, shape=shape
        ),
    ]
    if reference is not None:
        layers.insert(
            0,
            alt.Chart(pl.DataFrame({"x": [reference]}))
            .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], strokeWidth=0.8, aria=False)
            .encode(x="x:Q"),  # ty: ignore[unresolved-attribute]
        )
    return cast(Layer, alt.layer(*layers)).properties(
        width=width,
        height=height,
        title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2),
    )


def _rung4_levels(*, group: str, metric: str) -> pl.DataFrame:
    """Return the regional-sky intervals for one group and metric, one row per arm and share.

    Args:
        group: `pooled` or a battery's BMU id.
        metric: The metric.

    Returns:
        Columns `label`, `share`, `mean`, `lower_95`, and `upper_95`.
    """
    intervals = pl.read_parquet(OUTPUT_DIR / "rung4_intervals.parquet")
    return intervals.filter(
        (pl.col("sky") == "regional")
        & (pl.col("group") == group)
        & (pl.col("metric") == metric)
        & pl.col("quantity").is_in(list(ARMS))
        & (pl.col("share") > 0)
    ).with_columns(label=pl.col("quantity").replace_strict(ARM_LABELS, return_dtype=pl.String))


def rung4_error_figure(*, number: int) -> alt.VConcatChart:
    """Draw the solar series error by arm and share for each battery.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    panels = []
    for index, bmu_id in enumerate(BATTERIES):
        panels.append(
            _arm_dot_panel(
                table=_rung4_levels(group=bmu_id, metric="nmae_of_solar_p99"),
                title=f"With the {NAMES[bmu_id]} battery",
                x_title="Mean absolute error of the recovered solar (% of true solar p99; "
                "smaller is better)",
                last=index == len(BATTERIES) - 1,
                show_key=index == 0,
            )
        )
    return figure(
        panels=panels,
        number=number,
        title=(
            "Price regressors cut the solar error by about 8 points, but prices from another "
            "week cut it by about 5"
        ),
        subtitle=[
            (
                "Regional sky. Dot: mean over the 4 solar sets. Line: 95% interval from "
                "resampling the 4 solar sets, so it is wide and covers only these four. The "
                "true-battery fit (oracle) is the upper bound. The change from the solar-only "
                "fit to the price fit (A1 minus A0) is the planned contrast."
            )
        ],
        figure_planning=None,
    )


def rung4_ratio_figure(*, number: int) -> alt.VConcatChart:
    """Draw the energy ratio and the AC capacity ratio by arm and share.

    Args:
        number: The figure number.

    Returns:
        The figure.
    """
    energy = _arm_dot_panel(
        table=_rung4_levels(group="pooled", metric="energy_ratio"),
        title="Recovered solar energy over true solar energy (1 is exact)",
        x_title="Energy ratio (dot: mean over 16 aggregates; line: 95% interval from "
        "resampling them)",
        last=True,
        reference=1.0,
        show_key=True,
    )
    capacity = _arm_dot_panel(
        table=_rung4_levels(group="pooled", metric="ac_ratio"),
        title="Fitted AC capacity over the direct fit to the solar half (1 is exact)",
        x_title="AC capacity ratio (dot: mean over 16 aggregates; line: 95% interval from "
        "resampling them)",
        last=True,
        reference=1.0,
    )
    return figure(
        panels=[energy, capacity],
        number=number,
        title=(
            "Price regressors lift the recovered solar energy from about a quarter to about "
            "three quarters of the truth"
        ),
        subtitle=[
            (
                "Regional sky, 4 batteries by 4 solar sets. Prices from another week (control) "
                "recover about half. The true-battery fit (oracle) recovers about 94%."
            )
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write every figure as an SVG under `docs/studies/assets/` and a PNG preview."""
    charts = {
        "rung1_week": rung1_week_figure(number=2),
        "rung2_prediction": rung2_figure(number=3),
        "rung3_week": rung3_week_figure(number=4),
        "rung4_a0": rung4_a0_figure(number=5),
        "rung4_prices": rung4_prices_figure(number=6),
        "rung4_oracle": rung4_oracle_figure(number=7),
        "rung4_error": rung4_error_figure(number=8),
        "rung4_ratios": rung4_ratio_figure(number=9),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    previews = OUTPUT_DIR / "previews"
    previews.mkdir(exist_ok=True)
    for name, chart in charts.items():
        target = ASSETS_DIR / f"battery_pv_separation_{name}.svg"
        chart.save(target)
        chart.save(previews / f"{name}.png", scale_factor=1.3)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
