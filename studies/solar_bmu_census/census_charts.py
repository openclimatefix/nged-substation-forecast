"""Draw the census page's five charts.

They are the correlation histogram, the map, the summer and winter example weeks, and the capacity
figures.

Run after `report.py`: `uv run python studies/solar_bmu_census/census_charts.py`. The Balancing
Mechanism Unit (BMU) register, B1610, the Installed Generation Capacity per Unit (IGCPU) report, the
Transmission Entry Capacity (TEC) register, and the Renewable Energy Planning Database (REPD) are
public, so the charts name each BMU and show output in megawatts on calendar dates.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from classify import SOLAR_CORRELATION_THRESHOLD
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
}
RULE_WIDTHS: Final[tuple[float, ...]] = (6.0, 5.0, 4.0, 3.0, 2.0, 1.5)
"""Line widths from the first line drawn to the last. Lines that coincide at one height then show as
nested stripes, each in its own colour."""
DISPARITY_EXAMPLES: Final[int] = 3
DISPARITY_ROW_PX: Final[int] = 230
LABEL_MARGIN_PX: Final[int] = 250
LABEL_PADDING_PX: Final[int] = 40
"""The room right of the plot for the line labels, which Vega leaves out of the figure's width."""
LABEL_GAP_SHARE: Final[float] = 0.09
"""The least gap between two labels, as a share of the y axis."""
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


def _disparity_panel(*, row: dict[str, Any], last: bool) -> alt.LayerChart:
    """Draw one BMU's year of output with a horizontal line for each of its six figures.

    Each line has a label at the right-hand end of the plot, outside the plot area in the margin
    `LABEL_MARGIN_PX`, in the line's own colour and with its value in MW, and the figure has no
    legend. Lines with equal or close values are not combined into one label. Their labels are
    stacked, in value order, by `label_positions`, which keeps every pair at least `LABEL_GAP_SHARE`
    of the y axis apart, and each label has a square in the line's colour beside it.
    """
    _, window = recorded_run()
    series = output_series(bmu_id=row["elexon_bmu_id"], start=window.start, end=window.end)
    daily = series.group_by_dynamic("time", every="1d").agg(pl.col("megawatts").max())
    figures = [
        (name, float(row[column]), column) for column, name in FIGURE_NAMES.items() if row[column]
    ]
    high = float(row["highest_mw"]) * 1.1
    gap = high * LABEL_GAP_SHARE
    heights = label_positions(values=[value for _, value, _ in figures], gap=gap)
    high = max(high, max(heights) + gap)
    rules = pl.DataFrame(
        {
            "figure": [name for name, _, _ in figures],
            "megawatts": [value for _, value, _ in figures],
            "label_height": heights,
            "label": [f"{name}: {value:.1f} MW" for name, value, _ in figures],
        }
    )
    names = list(FIGURE_NAMES.values())
    colour = alt.Color(
        "figure:N",
        scale=alt.Scale(domain=names, range=[FIGURE_COLOURS[name] for name in names]),
        legend=None,
    )
    title = (
        f"{row['elexon_bmu_id']}  {row['display_name']}  "
        f"(highest figure over lowest: {row['ratio']:.2f})"
    )
    base = alt.Chart(series, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
    half_hourly = base.mark_line(
        strokeWidth=0.5, opacity=0.35, color=ocf.BLACK_1, aria=False
    ).encode(  # ty: ignore[unresolved-attribute]
        x=alt.X(
            "time:T",
            axis=alt.Axis(format="%b %Y", tickCount="month", labels=last, ticks=last, title=None),
            scale=alt.Scale(domain=[window.start, window.end]),
        ),
        y=alt.Y(
            "megawatts:Q",
            scale=alt.Scale(domain=[0, high], nice=False),
            axis=alt.Axis(tickCount=4, title="MW"),
        ),
    )
    daily_line = (
        alt.Chart(daily)
        .mark_line(strokeWidth=1, color=ocf.BLACK_1, aria=False)
        .encode(x="time:T", y="megawatts:Q")  # ty: ignore[unresolved-attribute]
    )
    rule_marks = []
    for index, (name, _, column) in enumerate(figures):
        is_p99 = column == P99_COLUMN
        rule_marks.append(
            alt.Chart(rules.filter(pl.col("figure") == name))
            .mark_rule(
                strokeWidth=1.5 if is_p99 else RULE_WIDTHS[index],
                strokeDash=[5, 3] if is_p99 else [1, 0],
                aria=False,
            )
            .encode(y="megawatts:Q", color=colour)  # ty: ignore[unresolved-attribute]
        )
    squares = (
        alt.Chart(rules.with_columns(time=pl.lit(window.end)))
        .mark_square(size=60, dx=8, aria=False)
        .encode(x="time:T", y="label_height:Q", color=colour)  # ty: ignore[unresolved-attribute]
    )
    texts = (
        alt.Chart(rules.with_columns(time=pl.lit(window.end)))
        .mark_text(align="left", dx=16, fontSize=10, fontWeight="bold", aria=False)
        .encode(  # ty: ignore[unresolved-attribute]
            x="time:T", y="label_height:Q", text="label:N", color=colour
        )
    )
    return cast(
        alt.LayerChart,
        alt.layer(half_hourly, daily_line, *rule_marks, squares, texts).properties(
            width=CONTENT_WIDTH_PX - 60 - LABEL_MARGIN_PX, height=DISPARITY_ROW_PX
        ),
    )


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
            "For the three BMUs whose six figures differ most, the highest figure is "
            f"{rows[-1]['ratio']:.1f} to {rows[0]['ratio']:.1f} times the lowest"
        ),
        subtitle=[
            (
                "Half-hourly output (faint line) and each day's largest half-hour (dark line), "
                "in megawatts, over the 12-month study window."
            ),
            (
                "Horizontal lines, each labelled with its figure: the five published capacities, "
                "and the dashed line, the 99th percentile of the BMU's output "
                "(a measure of output, not a capacity)."
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
