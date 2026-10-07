"""Draw the census page's five charts.

They are the correlation histogram, the map, the summer and winter example weeks, and the capacity
figures.

Run after `report.py`: `uv run python studies/solar_bmu_census/census_charts.py`. The Balancing
Mechanism Unit (BMU) register, B1610, the Installed Generation Capacity per Unit (IGCPU) report, the
Transmission Entry Capacity (TEC) register, and the Renewable Energy Planning Database (REPD) are
public, so the charts name each BMU and show output in megawatts on calendar dates.
"""

import bisect
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from classify import SOLAR_CORRELATION_THRESHOLD, largest_output_mw
from collate import P99_COLUMN, capacity_disparity
from fetch_sources import OUTPUT_DIR, STUDY_DIR, fetch_bmu_reference, recorded_run
from studies.charts import CONTENT_WIDTH_PX, figure

ASSETS_DIR: Final[Path] = Path("docs/studies/assets")
BOUNDARY_PATH: Final[Path] = Path(
    "packages/geo/src/geo/great_britain/england_scotland_wales.geojson"
)
WEEKS: Final[dict[str, datetime]] = {
    "summer": datetime(2026, 6, 15, tzinfo=UTC),
    "winter": datetime(2025, 12, 15, tzinfo=UTC),
}
"""The Monday that starts the week containing each solstice, fixed by the calendar and not chosen by
how the output looks."""
SERIES_COLOURS: Final[dict[str, str]] = {
    "Solar BMU": ocf.BRAND_ORANGE,
    "Storage BMU": ocf.DATA_BLUE,
}
FIGURE_NAMES: Final[dict[str, str]] = {
    "generation_capacity_mw": "Generation Capacity",
    "igcpu_installed_capacity_mw": "IGCPU installed",
    "tec_mw": "TEC",
    "largest_mel_mw": "Largest MEL",
    "repd_installed_capacity_mw": "REPD installed",
    P99_COLUMN: "P99 of output",
}
FIGURE_COLOURS: Final[dict[str, str]] = {
    "Generation Capacity": ocf.DATA_BLUE,
    "IGCPU installed": ocf.DATA_SKY,
    "TEC": ocf.DATA_PURPLE,
    "Largest MEL": ocf.DATA_GREEN,
    "REPD installed": ocf.BRAND_ORANGE,
    "P99 of output": ocf.BLACK_1,
    "Max of output": ocf.BLACK_1,
}
MAX_NAME: Final[str] = "Max of output"
"""The line at the BMU's largest half-hourly output (`classify.largest_output_mw`), which the
figure draws beside the six values `capacity_disparity` ranks, and which takes no part in the
ranking."""
OUTPUT_DASHES: Final[dict[str, list[float]]] = {
    "P99 of output": [5, 3],
    MAX_NAME: [1, 2],
}
"""The dash pattern of the two lines that measure the BMU's own output: dashed for the P99 and
dotted for the maximum. Both are dark, and each line's label names it."""
RULE_WIDTHS: Final[tuple[float, ...]] = (6.0, 5.0, 4.0, 3.0, 2.0, 1.5)
"""Line widths from the first line drawn to the last. Lines that coincide at one height then show as
nested stripes, each in its own colour."""
ONLINE_FIGURES: Final[tuple[str, ...]] = ("Largest MEL", "IGCPU installed", "Generation Capacity")
"""The values that share one text written on their line when two or more of them coincide, in the
order the text names them. TEC, REPD, and the P99 of output always take a label in the margin."""
ONLINE_FONT_PX: Final[int] = 10
ONLINE_CHAR_PX: Final[float] = 5.3
"""A generous estimate of the width of one character of the on-line text, in pixels, used to keep
the text inside the plot and to find where the output series crosses it least."""
ONLINE_INSET_PX: Final[int] = 4
"""The least space between the on-line text and either end of its line."""
ONLINE_LIFT_PX: Final[float] = 3.5
"""How far above its line the bottom of the on-line text sits, clear of the widest stripe."""
ONLINE_HALO_PX: Final[float] = 3.0
"""The width of the background-coloured outline drawn behind the on-line text."""
DISPARITY_EXAMPLES: Final[int] = 3
DISPARITY_ROW_PX: Final[int] = 230
LABEL_MARGIN_PX: Final[int] = 180
"""The width at the right of the plot area that the line labels and arrows occupy."""
LABEL_PADDING_PX: Final[int] = 5
"""The space right of the label margin."""
DISPARITY_FRAME_PX: Final[int] = 110
"""The width of the y axis, its title, and the figure's padding, which the plot area excludes."""
LABEL_GAP_SHARE: Final[float] = 0.09
"""The least gap between two labels, as a share of the y axis."""
LABEL_TEXT_OFFSET_PX: Final[int] = 16
"""How far right of the plot's data area each label's first letter starts."""
ARROW_GAP_PX: Final[int] = 4
"""The space between an arrow's tail and its label's first letter."""
ARROW_HEAD_BACK_PX: Final[float] = 3.0
"""How far behind the arrow's tip the centre of its triangular head sits."""
ARROW_HEAD_SIZE: Final[int] = 28
"""The area of the arrow head, in square pixels."""
TECHNOLOGY_COLOURS: Final[dict[str, str]] = {
    "pure PV": ocf.DATA_BLUE,
    "hybrid": ocf.BRAND_ORANGE,
    "unknown": ocf.GREY_3,
}
ROW_HEIGHT_PX: Final[int] = 115
MAP_HEIGHT_PX: Final[int] = 600
LABEL_CELL_DEGREES: Final[float] = 0.1
"""Points that round to the same grid cell share one label on the map, so labels do not overprint.

The cell is this many degrees wide in longitude and in latitude.
"""
LABEL_STEP_PX: Final[int] = 11
LABEL_RIGHT_ALIGN_LONGITUDE: Final[float] = 0.3
"""Labels of points east of this longitude sit left of their dot, so the labels stay on the page."""
OVERVIEW_SCALE: Final[int] = 2200
ZOOM_SCALE: Final[int] = 8500
ZOOM_HEIGHT_PX: Final[int] = 460
ZOOM_CENTRE_LONGITUDE: Final[float] = -0.7
ZOOM_CENTRE_LATITUDE: Final[float] = 52.1
SCOTLAND_LATITUDE: Final[float] = 55.5


def storage_bmu_at(*, site_bmu_id: str, census: pl.DataFrame) -> str | None:
    """Return the first storage BMU at a solar BMU's site that has B1610 rows in both weeks.

    Args:
        site_bmu_id: A solar BMU.
        census: The census table, whose `storage_bmu_ids` column holds the site's storage BMUs.

    Returns:
        The first storage BMU of the site whose B1610 file has rows in both example weeks, or None.
    """
    ids = census.filter(pl.col("elexon_bmu_id") == site_bmu_id)["storage_bmu_ids"][0] or ""
    for bmu in [bmu for bmu in ids.split(";") if bmu]:
        if all(week_series(bmu_id=bmu, week_start=start).height > 0 for start in WEEKS.values()):
            return bmu
    return None


@dataclass(frozen=True)
class Example:
    """One example site: its solar BMU, the site's kind, and its storage BMU if the site has one."""

    solar_bmu_id: str
    kind: str
    storage_bmu_id: str | None


def choose_examples(*, census: pl.DataFrame) -> list[Example]:
    """Choose the example sites by rule, pure PV first and hybrids after.

    Pure PV: every single-site, pure PV BMU whose output follows the sun, in descending order of
    correlation. Hybrid: the single-site hybrid BMUs whose output follows the sun, ordered by
    correlation, taking the first, the middle, and the last, so the examples span the best to the
    worst fit. Each hybrid example carries the storage BMU at its site, where it has one.

    Args:
        census: The census table.

    Returns:
        The examples in the order the figure draws them.
    """
    following = census.filter(
        (pl.col("scope") == "single-site") & pl.col("basis").str.contains("behaviour")
    ).sort("correlation", descending=True)
    pure = following.filter(pl.col("technology") == "pure PV")["elexon_bmu_id"].to_list()
    hybrid = following.filter(pl.col("technology") == "hybrid")["elexon_bmu_id"].to_list()
    middle = len(hybrid) // 2
    picks = sorted({0, middle, len(hybrid) - 1}) if hybrid else []
    return [
        *[Example(bmu_id, "pure PV", None) for bmu_id in pure],
        *[
            Example(bmu_id, "hybrid", storage_bmu_at(site_bmu_id=bmu_id, census=census))
            for bmu_id in (hybrid[i] for i in picks)
        ],
    ]


def output_series(*, bmu_id: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Return a BMU's output from `start` up to `end`, in megawatts.

    Args:
        bmu_id: The BMU.
        start: Midnight UTC at the start of the period.
        end: Midnight UTC at the end of the period, which is not included.

    Returns:
        Columns `time` (the half-hour's midpoint, UTC) and `megawatts`.

    Raises:
        ValueError: If the period lies outside the study window.
    """
    _, window = recorded_run()
    if not (window.start <= start and end <= window.end):
        raise ValueError("The period lies outside the study window")
    output = pl.read_parquet(OUTPUT_DIR / f"{bmu_id}_{window.label}.parquet")
    midpoint = pl.col("half_hour_end_time").dt.offset_by("-15m")
    return (
        output.filter(midpoint >= start, midpoint < end)
        .select(time=midpoint, megawatts=pl.col("output_mwh") * 2)
        .sort("time")
    )


def week_series(*, bmu_id: str, week_start: datetime) -> pl.DataFrame:
    """Return a BMU's output over one week, in megawatts (see `output_series`)."""
    return output_series(bmu_id=bmu_id, start=week_start, end=week_start + timedelta(days=7))


def week_domain(*, capacity_mw: float, megawatts: pl.Series) -> tuple[float, float]:
    """Return the y-axis limits of a week panel: the data and the capacity, with room to see zero.

    The upper limit is 10% above the larger of the Generation Capacity and the largest output, and
    the lower limit is 10% below the lowest output when that is negative (a battery charging) and
    5% of the upper limit below zero otherwise.
    """
    high = max(capacity_mw, _as_float(megawatts.max())) * 1.1
    lowest = min(0.0, _as_float(megawatts.min()))
    return (lowest * 1.1 if lowest < 0 else -0.05 * high), high


def _as_float(value: object) -> float:
    """Return a Polars aggregate as a float."""
    if not isinstance(value, int | float):
        raise TypeError(f"Expected a number, got {type(value).__name__}")
    return float(value)


def _week_panel(
    *, series: pl.DataFrame, title: str, domain: tuple[float, float], last: bool
) -> alt.LayerChart:
    """Draw a site's week: its solar and storage BMUs as lines, with a rule at zero MW."""
    names = list(SERIES_COLOURS)
    lines = (
        alt.Chart(series, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
        .mark_line(strokeWidth=1.5)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "time:T",
                axis=alt.Axis(
                    format="%a %-d %b",
                    tickCount="day",
                    labels=last,
                    ticks=last,
                    title="Day (UTC)" if last else None,
                ),
            ),
            y=alt.Y(
                "megawatts:Q",
                scale=alt.Scale(domain=list(domain), nice=False),
                axis=alt.Axis(tickCount=4, title="MW"),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=names, range=[SERIES_COLOURS[name] for name in names]),
                legend=None,
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 90, height=ROW_HEIGHT_PX)
    )
    zero = (
        alt.Chart(pl.DataFrame({"zero": [0.0]}))
        .mark_rule(color=ocf.BLACK_1, strokeWidth=0.8, aria=False)
        .encode(y="zero:Q")  # ty: ignore[unresolved-attribute]
    )
    return cast(alt.LayerChart, alt.layer(zero, lines))


def example_week_figure(
    *, census: pl.DataFrame, examples: list[Example], season: str, number: int
) -> alt.VConcatChart:
    """Draw the example sites' output over one week, one row for each site."""
    reference = {str(r["elexonBmUnit"]): r for r in fetch_bmu_reference() if r["elexonBmUnit"]}
    display = dict(zip(census["elexon_bmu_id"], census["display_name"], strict=True))
    start = WEEKS[season]
    panels = []
    for index, example in enumerate(examples):
        row = reference[example.solar_bmu_id]
        capacity = float(row["generationCapacity"])
        name = display.get(example.solar_bmu_id) or str(row["bmUnitName"])
        parts = [
            week_series(bmu_id=example.solar_bmu_id, week_start=start).with_columns(
                series=pl.lit("Solar BMU")
            )
        ]
        storage_note = ""
        if example.storage_bmu_id is not None:
            parts.append(
                week_series(bmu_id=example.storage_bmu_id, week_start=start).with_columns(
                    series=pl.lit("Storage BMU")
                )
            )
            storage_note = f"; storage BMU {example.storage_bmu_id}"
        series = pl.concat(parts)
        panels.append(
            _week_panel(
                series=series,
                title=(
                    f"{example.solar_bmu_id}  {name}  ({example.kind} site, Generation Capacity "
                    f"{capacity:g} MW{storage_note})"
                ),
                domain=week_domain(capacity_mw=capacity, megawatts=series["megawatts"]),
                last=index == len(examples) - 1,
            )
        )
    end = start + timedelta(days=6)
    return figure(
        panels=panels,
        number=number,
        title=(
            f"Solar BMUs at hybrid sites follow the sun like the pure PV BMU in the {season} week; "
            "storage BMUs do not"
        ),
        subtitle=[
            (
                f"Half-hourly output, 00:00 UTC on {start:%-d %B %Y} to 23:30 UTC on "
                f"{end:%-d %B %Y}, in megawatts. Each row is one site, pure PV first."
            ),
            (
                "Orange: solar BMU. Blue: storage BMU at the same site (negative when charging). "
                "The black line marks zero MW."
            ),
            (
                "The three hybrid examples have the highest, median, and lowest correlation "
                "with the sun of the eight hybrid solar BMUs that follow it."
            ),
        ],
        figure_planning=None,
    )


def disparity_examples(*, census: pl.DataFrame) -> pl.DataFrame:
    """Return the three BMUs with the most disparate capacity figures, by `capacity_disparity`."""
    return capacity_disparity(table=census).head(DISPARITY_EXAMPLES)


def coinciding_groups(*, values: dict[str, float]) -> list[list[str]]:
    """Group the `ONLINE_FIGURES` values that are equal at the one decimal place the figure shows.

    Args:
        values: Each figure's value in MW, by its name in `FIGURE_NAMES`. Names outside
            `ONLINE_FIGURES` are ignored.

    Returns:
        Each group of two or more names whose values print the same to one decimal place, with the
        names in `ONLINE_FIGURES` order. A value that coincides with no other is in no group.
    """
    groups: dict[str, list[str]] = {}
    for name in ONLINE_FIGURES:
        if name in values:
            groups.setdefault(f"{values[name]:.1f}", []).append(name)
    return [names for names in groups.values() if len(names) >= 2]


def online_text(*, names: list[str], value: float) -> str:
    """Return the text written on a line that several values share, such as "A = B = 50.0 MW"."""
    return " = ".join([*names, f"{value:.1f} MW"])


def online_text_start(
    *, series: pl.DataFrame, value: float, start: datetime, end: datetime, width_days: float
) -> datetime:
    """Find where along its line the on-line text is crossed by the fewest output readings.

    The rule: slide a window as wide as the text, one day at a time, from `start` to the last day
    the whole text fits before `end`, and count the half-hourly readings inside the window that
    rise above the line, into the text. The window with the fewest such readings wins, and the
    earliest window wins a tie.

    Args:
        series: The BMU's output, with columns `time` and `megawatts`.
        value: The line's height in MW.
        start: The earliest time the text may start.
        end: The latest time the text may end.
        width_days: The text's width, in days of the x axis.

    Returns:
        The time at which the text's left edge sits.
    """
    above = sorted(series.filter(pl.col("megawatts") > value)["time"].to_list())
    width = timedelta(days=width_days)
    best, fewest = start, len(above) + 1
    left = start
    while left + width <= end or left == start:
        crossing = bisect.bisect_right(above, left + width) - bisect.bisect_left(above, left)
        if crossing < fewest:
            best, fewest = left, crossing
        left += timedelta(days=1)
    return best


def label_positions(*, values: list[float], gap: float) -> list[float]:
    """Spread label heights so that no two are closer than `gap`, keeping their order.

    Args:
        values: The heights the labels belong at, in any order.
        gap: The least distance between two label heights, in the same unit.

    Returns:
        The label heights in the order of `values`: each height is its value, or the height above
        the label just below if that value would sit closer than `gap`.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    previous: float | None = None
    for index in order:
        height = values[index] if previous is None else max(values[index], previous + gap)
        placed[index] = height
        previous = height
    return placed


@dataclass(frozen=True)
class LabelGeometry:
    """Where each line's label and arrow lie in one panel, in the order of the lines' values.

    Attributes:
        domain_high: The top of the y axis, in MW, raised if the labels stacked above the data.
        label_mw: The height each label's text is centred at, in MW.
        label_y_px: The same heights, in pixels down from the top of the plot.
        line_y_px: Each line's own height in pixels down from the top of the plot, which is where
            its arrow ends.
    """

    domain_high: float
    label_mw: list[float]
    label_y_px: list[float]
    line_y_px: list[float]


def label_geometry(*, values: list[float], domain_high: float, height_px: float) -> LabelGeometry:
    """Place the labels beside the plot, and the pixel heights their arrows join.

    Args:
        values: Each line's height in MW, in any order.
        domain_high: The top of the y axis in MW before the labels are placed.
        height_px: The plot's height in pixels.

    Returns:
        Positions in the order of `values`. The labels keep the order of the values and sit at
        least `LABEL_GAP_SHARE` of the axis apart, so no two arrows cross.
    """
    gap = domain_high * LABEL_GAP_SHARE
    heights = label_positions(values=values, gap=gap)
    high = max(domain_high, max(heights) + gap)

    def pixels(megawatts: float) -> float:
        return (1 - megawatts / high) * height_px

    return LabelGeometry(
        domain_high=high,
        label_mw=heights,
        label_y_px=[pixels(height) for height in heights],
        line_y_px=[pixels(value) for value in values],
    )


def _arrow_marks(
    *,
    geometry: LabelGeometry,
    names: list[str],
    values: list[float],
    end: datetime,
    days_per_px: float,
    height_px: float,
    colour: alt.Color,
) -> list[alt.Chart]:
    """Draw one thin arrow per line, from just left of its label to the right end of the line."""
    tail_px = LABEL_TEXT_OFFSET_PX - ARROW_GAP_PX
    shaft = []
    heads = []
    for name, value, label_mw, label_px, line_px in zip(
        names, values, geometry.label_mw, geometry.label_y_px, geometry.line_y_px, strict=True
    ):
        shaft.append(
            {
                "figure": name,
                "px": 0,
                "time": end + timedelta(days=tail_px * days_per_px),
                "mw": label_mw,
            }
        )
        shaft.append({"figure": name, "px": 1, "time": end, "mw": value})
        # The marker's angle is clockwise from "up"; the arrow points left and down the page.
        run, drop = -float(tail_px), line_px - label_px
        length = math.hypot(run, drop)
        back = ARROW_HEAD_BACK_PX / length
        heads.append(
            {
                "figure": name,
                "time": end - timedelta(days=run * back * days_per_px),
                "mw": value + drop * back * geometry.domain_high / height_px,
                "angle": math.degrees(math.atan2(run, -drop)),
            }
        )
    return [
        alt.Chart(pl.DataFrame(shaft))
        .mark_line(strokeWidth=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x="time:T", y="mw:Q", detail="figure:N", order="px:Q", color=colour
        ),
        alt.Chart(pl.DataFrame(heads))
        .mark_point(shape="triangle", filled=True, size=ARROW_HEAD_SIZE, opacity=1, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x="time:T", y="mw:Q", angle=alt.Angle("angle:Q", scale=None), color=colour
        ),
    ]


def _disparity_panel(*, row: dict[str, Any], last: bool) -> alt.LayerChart:
    """Draw one BMU's year of output with a line for each capacity value, its P99, and its maximum.

    Where two or more of the `ONLINE_FIGURES` values coincide (`coinciding_groups`), one text such
    as "Largest MEL = IGCPU installed = 50.0 MW" sits just above their shared line, inside the plot.
    The text is a neutral dark colour rather than each name in its own line's colour, because Data
    Sky and Data Green text is hard to read on the cream background, and the stacked stripes of the
    line already show each colour. A background-coloured outline behind the text keeps it legible
    where the output series runs through it, and `online_text_start` places it where the series
    crosses it least.

    Every other value has a label at the right-hand end of the plot, outside the plot area in the
    margin `LABEL_MARGIN_PX`, in the line's own colour and with its value in MW. Those labels are
    stacked, in value order, by `label_geometry`, which keeps every pair at least `LABEL_GAP_SHARE`
    of the y axis apart, and a thin arrow in the line's colour joins each label to the right-hand
    end of its line. The label margin lies inside the x axis domain, so that the arrows are drawn
    in the plot's own coordinates.

    The lines are drawn first and the output series over them, so that the series is never hidden.
    """
    _, window = recorded_run()
    series = output_series(bmu_id=row["elexon_bmu_id"], start=window.start, end=window.end)
    daily = series.group_by_dynamic("time", every="1d").agg(pl.col("megawatts").max())
    output_file = OUTPUT_DIR / f"{row['elexon_bmu_id']}_{window.label}.parquet"
    figures = [
        *[(name, float(row[column])) for column, name in FIGURE_NAMES.items() if row[column]],
        (
            MAX_NAME,
            largest_output_mw(output=pl.read_parquet(output_file)),
        ),
    ]
    groups = coinciding_groups(values=dict(figures))
    grouped = {name for names in groups for name in names}
    margin = [(name, value) for name, value in figures if name not in grouped]
    margin_values = [value for _, value in margin]
    geometry = label_geometry(
        values=margin_values,
        domain_high=max(float(row["highest_mw"]), *(value for _, value in figures)) * 1.1,
        height_px=DISPARITY_ROW_PX,
    )
    high = geometry.domain_high
    rules = pl.DataFrame(
        {
            "figure": [name for name, _ in figures],
            "megawatts": [value for _, value in figures],
            "start": window.start,
            "end": window.end,
        }
    )
    labels = pl.DataFrame(
        {
            "figure": [name for name, _ in margin],
            "time": window.end,
            "label_height": geometry.label_mw,
            "label": [f"{name}: {value:.1f} MW" for name, value in margin],
        }
    )
    plot_px = CONTENT_WIDTH_PX - DISPARITY_FRAME_PX
    data_px = plot_px - LABEL_MARGIN_PX
    days_per_px = (window.end - window.start).total_seconds() / 86400 / data_px
    x_end = window.end + timedelta(days=LABEL_MARGIN_PX * days_per_px)
    names = list(FIGURE_COLOURS)
    colour = alt.Color(
        "figure:N",
        scale=alt.Scale(domain=names, range=[FIGURE_COLOURS[name] for name in names]),
        legend=None,
    )
    title = (
        f"{row['elexon_bmu_id']}  {row['display_name']}  "
        f"(highest value over lowest: {row['ratio']:.2f})"
    )
    rule_marks = []
    for index, (name, _) in enumerate(figures):
        dashes = OUTPUT_DASHES.get(name)
        rule_marks.append(
            alt.Chart(rules.filter(pl.col("figure") == name))
            .mark_rule(
                strokeWidth=RULE_WIDTHS[index] if dashes is None else 1.5,
                strokeDash=[1, 0] if dashes is None else dashes,
                aria=False,
            )
            .encode(  # ty: ignore[unresolved-attribute]
                x=alt.X(
                    "start:T",
                    axis=alt.Axis(
                        labelExpr=(
                            "[timeFormat(datum.value, '%b'), timeFormat(datum.value, '%Y')]"
                        ),
                        labelAlign="center",
                        values=_month_starts(start=window.start, end=window.end),
                        labels=last,
                        ticks=last,
                        title=None,
                        domain=False,
                    ),
                    scale=alt.Scale(domain=[window.start, x_end], nice=False),
                ),
                x2="end:T",
                y=alt.Y(
                    "megawatts:Q",
                    scale=alt.Scale(domain=[0, high], nice=False),
                    axis=alt.Axis(tickCount=4, title="MW"),
                ),
                color=colour,
            )
        )
    half_hourly = (
        alt.Chart(series)
        .mark_line(strokeWidth=0.5, opacity=0.35, color=ocf.BLACK_1, aria=False)
        .encode(x="time:T", y="megawatts:Q")  # ty: ignore[unresolved-attribute]
    )
    daily_line = (
        alt.Chart(daily)
        .mark_line(strokeWidth=1, color=ocf.BLACK_1, aria=False)
        .encode(x="time:T", y="megawatts:Q")  # ty: ignore[unresolved-attribute]
    )
    arrows = _arrow_marks(
        geometry=geometry,
        names=[name for name, _ in margin],
        values=margin_values,
        end=window.end,
        days_per_px=days_per_px,
        height_px=DISPARITY_ROW_PX,
        colour=colour,
    )
    texts = (
        alt.Chart(labels)
        .mark_text(
            align="left", dx=LABEL_TEXT_OFFSET_PX, fontSize=10, fontWeight="bold", aria=False
        )
        .encode(  # ty: ignore[unresolved-attribute]
            x="time:T", y="label_height:Q", text="label:N", color=colour
        )
    )
    return cast(
        alt.LayerChart,
        alt.layer(
            *rule_marks,
            half_hourly,
            daily_line,
            *arrows,
            texts,
            *_online_marks(
                groups=groups,
                values=dict(figures),
                series=series,
                start=window.start,
                end=window.end,
                days_per_px=days_per_px,
                mw_per_px=high / DISPARITY_ROW_PX,
            ),
        ).properties(
            title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2),
            width=plot_px,
            height=DISPARITY_ROW_PX,
        ),
    )


def _online_marks(
    *,
    groups: list[list[str]],
    values: dict[str, float],
    series: pl.DataFrame,
    start: datetime,
    end: datetime,
    days_per_px: float,
    mw_per_px: float,
) -> list[alt.Chart]:
    """Draw each coinciding group's text just above its line.

    Three marks, from the bottom up: a background-coloured backing from the top of the widest
    stripe to the top of the text, which hides any other line running just above the shared line
    (such as the maximum of output) where the text sits; a background-coloured outline of the text;
    and the text itself.
    """
    rows = []
    for names in groups:
        value = values[names[0]]
        text = online_text(names=names, value=value)
        inset = timedelta(days=ONLINE_INSET_PX * days_per_px)
        width_days = len(text) * ONLINE_CHAR_PX * days_per_px
        left = online_text_start(
            series=series, value=value, start=start + inset, end=end - inset, width_days=width_days
        )
        rows.append(
            {
                "time": left,
                "time_end": left + timedelta(days=width_days),
                "megawatts": value,
                "backing_low": value + RULE_WIDTHS[0] / 2 * mw_per_px,
                "backing_high": value + (ONLINE_LIFT_PX + ONLINE_FONT_PX + 2) * mw_per_px,
                "text": text,
            }
        )
    if not rows:
        return []
    frame = pl.DataFrame(rows)
    style: dict[str, Any] = {
        "align": "left",
        "baseline": "bottom",
        "dy": -ONLINE_LIFT_PX,
        "fontSize": ONLINE_FONT_PX,
        "fontWeight": "bold",
        "aria": False,
    }
    return [
        alt.Chart(frame)
        .mark_rect(color=ocf.BACKGROUND, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x="time:T", x2="time_end:T", y="backing_low:Q", y2="backing_high:Q"
        ),
        alt.Chart(frame)
        .mark_text(
            stroke=ocf.BACKGROUND,
            strokeWidth=ONLINE_HALO_PX,
            strokeJoin="round",
            color=ocf.BACKGROUND,
            **style,
        )
        .encode(x="time:T", y="megawatts:Q", text="text:N"),  # ty: ignore[unresolved-attribute]
        alt.Chart(frame)
        .mark_text(color=ocf.BLACK_1, **style)
        .encode(x="time:T", y="megawatts:Q", text="text:N"),  # ty: ignore[unresolved-attribute]
    ]


def _month_starts(*, start: datetime, end: datetime) -> list[alt.DateTime]:
    """Return the first day of each month from the one after `start` to `end`, as axis ticks."""
    months = []
    year, month = start.year, start.month
    while True:
        month += 1
        if month > 12:
            year, month = year + 1, 1
        if datetime(year, month, 1, tzinfo=UTC) > end:
            return months
        months.append(alt.DateTime(year=year, month=month, date=1))


def disparity_figure(*, census: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the year of output of the three BMUs whose capacity figures differ the most."""
    chosen = disparity_examples(census=census)
    rows = chosen.to_dicts()
    panels = [
        _disparity_panel(row=row, last=index == len(rows) - 1) for index, row in enumerate(rows)
    ]
    chart = figure(
        panels=panels,
        number=number,
        title=(
            "For the three BMUs whose capacity values and P99 of output differ most, the highest "
            f"is {rows[-1]['ratio']:.1f} to {rows[0]['ratio']:.1f} times the lowest"
        ),
        subtitle=[
            (
                "Half-hourly output (faint line) and each day's largest half-hour (dark line), in "
                "MW, over the 12-month window."
            ),
            (
                "Horizontal lines: the five published capacity values, and two measures of the "
                "BMU's own output (P99 and maximum)."
            ),
            (
                "Dashed: the 99th percentile of output. Dotted: the largest half-hour, which "
                "played no part in choosing the BMUs."
            ),
            (
                "Text on a line: the values that agree there to one decimal place. An arrow joins "
                "each other value to its label."
            ),
            (
                "TEC and REPD describe the whole Cleve Hill project, which holds both Cleve Hill "
                "BMUs and a battery."
            ),
        ],
        figure_planning=None,
    )
    return chart.properties(padding={"left": 5, "top": 5, "right": LABEL_PADDING_PX, "bottom": 5})


def correlation_figure(*, correlations: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw the histogram of each single-site BMU's correlation with the sun, for BMUs with output.

    The threshold is marked as a dashed line.
    """
    frame = correlations.filter(
        pl.col("scope") == "single-site", pl.col("behaviour") != "no_output"
    ).with_columns(
        census=pl.when(pl.col("behaviour") == "solar")
        .then(pl.lit("Follows the sun"))
        .otherwise(pl.lit("Does not follow the sun"))
    )
    bars = (
        alt.Chart(frame)
        .mark_bar(aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "correlation:Q",
                bin=alt.Bin(extent=[-0.45, 1.0], step=0.05),
                title="Correlation of half-hourly output with the cosine of the solar zenith",
            ),
            y=alt.Y(
                "count():Q",
                scale=alt.Scale(type="symlog", constant=1),
                axis=alt.Axis(values=[0, 1, 10, 100, 400]),
                title="BMUs (log scale)",
            ),
            color=alt.Color(
                "census:N",
                scale=alt.Scale(
                    domain=["Follows the sun", "Does not follow the sun"],
                    range=[ocf.DATA_BLUE, ocf.GREY_3],
                ),
                legend=alt.Legend(title=None),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 80, height=240)
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [SOLAR_CORRELATION_THRESHOLD]}))
        .mark_rule(color=ocf.BLACK_1, strokeDash=[4, 3], aria=False)
        .encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    )
    return figure(
        panels=[cast(alt.LayerChart, alt.layer(bars, rule))],
        number=number,
        title="A wide gap separates single-site BMUs whose output follows the sun from the rest",
        subtitle=[
            (
                "One count per BMU with a T_, E_, or M_ identifier and enough output to judge: "
                f"{frame.height} in all."
            ),
            f"Dashed line: the threshold of {SOLAR_CORRELATION_THRESHOLD}.",
        ],
        figure_planning=None,
    )


def map_labels(*, located: pl.DataFrame) -> pl.DataFrame:
    """Return one label for each group of nearby points, shifted alternately up and down.

    The alternate shifts stop neighbouring labels from overprinting.

    Args:
        located: The census BMUs that have a position, with `display_name`, `longitude`, and
            `latitude`.

    Returns:
        Columns `longitude`, `latitude`, `label` (the group's names, joined with " and "), `offset`
        (the vertical shift in pixels, alternating between two values so neighbouring labels do not
        overprint), and `align` (`left` or `right`, set by `LABEL_RIGHT_ALIGN_LONGITUDE`).
    """
    grouped = (
        located.with_columns(
            cell_x=(pl.col("longitude") / LABEL_CELL_DEGREES).round(),
            cell_y=(pl.col("latitude") / LABEL_CELL_DEGREES).round(),
        )
        .group_by("cell_x", "cell_y")
        .agg(
            pl.col("longitude").first(),
            pl.col("latitude").first(),
            pl.col("display_name").sort().alias("names"),
        )
        .with_columns(
            label=pl.col("names").list.join(" and "),
            align=pl.when(pl.col("longitude") > LABEL_RIGHT_ALIGN_LONGITUDE)
            .then(pl.lit("right"))
            .otherwise(pl.lit("left")),
        )
        .sort("latitude", descending=True)
    )
    return grouped.with_columns(offset=pl.int_range(pl.len()) % 2 * LABEL_STEP_PX - 4).select(
        "longitude", "latitude", "label", "offset", "align"
    )


def _base_map(*, width: int, height: int) -> alt.Chart:
    """Draw the Great Britain outline."""
    boundary = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    return (
        alt.Chart(alt.Data(values=boundary["features"]))
        .mark_geoshape(fill=ocf.GREY_2, stroke=ocf.GREY_3, strokeWidth=0.5, aria=False)
        .properties(width=width, height=height)  # ty: ignore[unresolved-attribute]
    )


def _points(*, located: pl.DataFrame, shown: list[str], legend: bool) -> alt.Chart:
    """Draw one dot for each BMU with a position, coloured by the site's technology."""
    return (
        alt.Chart(located.select("longitude", "latitude", "technology"))
        .mark_circle(size=70, opacity=0.9, aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            longitude="longitude:Q",
            latitude="latitude:Q",
            color=alt.Color(
                "technology:N",
                scale=alt.Scale(domain=shown, range=[TECHNOLOGY_COLOURS[name] for name in shown]),
                legend=alt.Legend(title="Site technology") if legend else None,
            ),
        )
    )


def map_figure(*, census: pl.DataFrame, number: int) -> alt.VConcatChart:
    """Draw two maps of the single-site census BMUs that have a position: Great Britain, and a zoom.

    The zoom shows the BMUs in England.

    Args:
        census: The census table.
        number: The figure's number on the page.

    Returns:
        The figure. On the whole-country map only the BMUs north of `SCOTLAND_LATITUDE` carry
        labels, because the BMUs in England lie too close together to label at that scale. The zoom
        labels every BMU in England.
    """
    single = census.filter(pl.col("scope") == "single-site")
    located = single.filter(pl.col("longitude").is_not_null())
    shown = [name for name in TECHNOLOGY_COLOURS if name in set(located["technology"].to_list())]
    width = CONTENT_WIDTH_PX - 40
    scotland = located.filter(pl.col("latitude") >= SCOTLAND_LATITUDE)
    scotland_labels = (
        alt.Chart(scotland.select("longitude", "latitude", label=pl.col("display_name")))
        .mark_text(align="left", dx=8, fontSize=10, color=ocf.BLACK_1, aria=False)
        .encode(longitude="longitude:Q", latitude="latitude:Q", text="label:N")  # ty: ignore[unresolved-attribute]
    )
    overview = alt.layer(
        _base_map(width=width, height=MAP_HEIGHT_PX),
        _points(located=located, shown=shown, legend=True),
        scotland_labels,
    ).project(
        type="mercator",
        center=[-3.2, 54.4],
        scale=OVERVIEW_SCALE,
        translate=[width / 2, MAP_HEIGHT_PX / 2],
    )
    england = located.filter(pl.col("latitude") < SCOTLAND_LATITUDE)
    label_table = map_labels(located=england)
    labels = [
        alt.Chart(label_table.filter((pl.col("offset") == offset) & (pl.col("align") == align)))
        .mark_text(
            align=align,
            dx=8 if align == "left" else -8,
            dy=offset,
            fontSize=10,
            color=ocf.BLACK_1,
            aria=False,
        )
        .encode(longitude="longitude:Q", latitude="latitude:Q", text="label:N")  # ty: ignore[unresolved-attribute]
        for offset in sorted(set(label_table["offset"].to_list()))
        for align in ("left", "right")
    ]
    zoom = alt.layer(
        _base_map(width=width, height=ZOOM_HEIGHT_PX),
        _points(located=england, shown=shown, legend=False),
        *labels,
    ).project(
        type="mercator",
        center=[ZOOM_CENTRE_LONGITUDE, ZOOM_CENTRE_LATITUDE],
        scale=ZOOM_SCALE,
        translate=[width / 2, ZOOM_HEIGHT_PX / 2],
    )
    return figure(
        panels=[cast(alt.LayerChart, overview), cast(alt.LayerChart, zoom)],
        number=number,
        title="Nine single-site solar BMUs lie in England and one in Scotland",
        subtitle=[
            (
                f"{located.height} of {single.height} single-site solar BMUs have a position, from "
                "the Renewable Energy Planning Database row matched to each BMU."
            ),
            "Top: Great Britain. Bottom: a zoom on the nine BMUs in England, at eight sites.",
            "Blue: pure PV site. Orange: hybrid site, with storage built or planned.",
        ],
        figure_planning=None,
    )


def main() -> None:
    """Write every chart as an SVG under `docs/studies/assets/`."""
    census = pl.read_parquet(STUDY_DIR / "solar_bmus.parquet")
    correlations = pl.read_parquet(STUDY_DIR / "classes.parquet")
    examples = choose_examples(census=census)
    charts = {
        "solar_bmu_census_correlation": correlation_figure(correlations=correlations, number=1),
        "solar_bmu_census_map": map_figure(census=census, number=3),
        "solar_bmu_census_summer_week": example_week_figure(
            census=census, examples=examples, season="summer", number=4
        ),
        "solar_bmu_census_winter_week": example_week_figure(
            census=census, examples=examples, season="winter", number=5
        ),
        "solar_bmu_census_capacity_figures": disparity_figure(census=census, number=2),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in charts.items():
        chart.save(ASSETS_DIR / f"{name}.svg")
        print(f"Wrote {ASSETS_DIR / f'{name}.svg'}")


if __name__ == "__main__":
    main()
