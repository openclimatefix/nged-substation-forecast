"""Draw the census page's charts: the correlation histogram, the map, and the example weeks.

Run after `report.py`: `uv run python studies/solar_bmu_census/census_charts.py`. The Balancing
Mechanism Unit (BMU) register, B1610, the Installed Generation Capacity per Unit (IGCPU) report, the
Transmission Entry Capacity (TEC) register, and the Renewable Energy Planning Database (REPD) are
public, so the charts name each BMU and show output in megawatts on calendar dates.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from classify import SOLAR_CORRELATION_THRESHOLD
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
KIND_COLOURS: Final[dict[str, str]] = {
    "pure PV": ocf.DATA_BLUE,
    "hybrid-site solar": ocf.BRAND_ORANGE,
    "storage": ocf.DATA_PURPLE,
}
TECHNOLOGY_COLOURS: Final[dict[str, str]] = {
    "pure PV": ocf.DATA_BLUE,
    "hybrid": ocf.BRAND_ORANGE,
    "unknown": ocf.GREY_3,
}
ROW_HEIGHT_PX: Final[int] = 72
MAP_HEIGHT_PX: Final[int] = 600
LABEL_CELL_DEGREES: Final[float] = 0.05
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


def choose_examples(*, census: pl.DataFrame) -> list[tuple[str, str]]:
    """Choose the example BMUs by rule, and return each with its kind.

    Pure PV: every single-site, pure PV BMU whose output follows the sun, in descending order of
    correlation. Hybrid-site solar: the single-site hybrid BMUs whose output follows the sun,
    ordered by correlation, taking the first, the middle, and the last, so the examples span the
    best to the worst fit. Storage: the storage BMU at each hybrid-site example, where it has one.
    The rows are grouped by kind, in that order.

    Args:
        census: The census table.

    Returns:
        `(bmu_id, kind)` pairs in the order the figure draws them.
    """
    following = census.filter(
        (pl.col("scope") == "single-site") & pl.col("basis").str.contains("behaviour")
    ).sort("correlation", descending=True)
    pure = following.filter(pl.col("technology") == "pure PV")["elexon_bmu_id"].to_list()
    hybrid = following.filter(pl.col("technology") == "hybrid")["elexon_bmu_id"].to_list()
    middle = len(hybrid) // 2
    picks = sorted({0, middle, len(hybrid) - 1}) if hybrid else []
    chosen_hybrid = [hybrid[i] for i in picks]
    storage = [storage_bmu_at(site_bmu_id=bmu_id, census=census) for bmu_id in chosen_hybrid]
    return [
        *[(bmu_id, "pure PV") for bmu_id in pure],
        *[(bmu_id, "hybrid-site solar") for bmu_id in chosen_hybrid],
        *[(bmu_id, "storage") for bmu_id in storage if bmu_id is not None],
    ]


def week_series(*, bmu_id: str, week_start: datetime) -> pl.DataFrame:
    """Return a BMU's output over one week, in megawatts.

    Args:
        bmu_id: The BMU.
        week_start: Midnight UTC at the start of the week.

    Returns:
        Columns `time` (the half-hour's midpoint, UTC) and `megawatts`.

    Raises:
        ValueError: If the week lies outside the study window.
    """
    _, window = recorded_run()
    if not (window.start <= week_start and week_start + timedelta(days=7) <= window.end):
        raise ValueError("The example week lies outside the study window")
    output = pl.read_parquet(OUTPUT_DIR / f"{bmu_id}_{window.label}.parquet")
    midpoint = pl.col("half_hour_end_time").dt.offset_by("-15m")
    return (
        output.filter(midpoint >= week_start, midpoint < week_start + timedelta(days=7))
        .select(time=midpoint, megawatts=pl.col("output_mwh") * 2)
        .sort("time")
    )


def _week_panel(
    *, series: pl.DataFrame, title: str, kind: str, capacity_mw: float, last: bool
) -> alt.LayerChart:
    """Draw one BMU's week as a line in the colour of its kind, titled with its name."""
    low, high = (
        (-capacity_mw * 1.1, capacity_mw * 1.1)
        if kind == "storage"
        else (-capacity_mw * 0.2, capacity_mw * 1.1)
    )
    line = (
        alt.Chart(series, title=alt.TitleParams(title, anchor="start", fontSize=11, offset=2))
        .mark_line(color=KIND_COLOURS[kind], strokeWidth=1.5)
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
                scale=alt.Scale(domain=[low, high], nice=False),
                axis=alt.Axis(tickCount=3, title="MW"),
            ),
        )
        .properties(width=CONTENT_WIDTH_PX - 90, height=ROW_HEIGHT_PX)
    )
    return cast(alt.LayerChart, alt.layer(line))


def example_week_figure(
    *, census: pl.DataFrame, examples: list[tuple[str, str]], season: str, number: int
) -> alt.VConcatChart:
    """Draw the example BMUs' output over one week, one row for each example."""
    reference = {str(r["elexonBmUnit"]): r for r in fetch_bmu_reference() if r["elexonBmUnit"]}
    display = dict(zip(census["elexon_bmu_id"], census["display_name"], strict=True))
    panels = []
    for index, (bmu_id, kind) in enumerate(examples):
        row = reference[bmu_id]
        capacity = float(row["generationCapacity"])
        name = display.get(bmu_id) or str(row["bmUnitName"])
        panels.append(
            _week_panel(
                series=week_series(bmu_id=bmu_id, week_start=WEEKS[season]),
                title=f"{bmu_id}  {name}  ({kind}, Generation Capacity {capacity:g} MW)",
                kind=kind,
                capacity_mw=capacity,
                last=index == len(examples) - 1,
            )
        )
    start = WEEKS[season]
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
                f"{end:%-d %B %Y}, in megawatts."
            ),
            "Blue: pure PV. Orange: solar BMU at a hybrid site. Purple: storage BMU at that site.",
            "Examples span the best to the worst fit to the sun; the page's methods give the rule.",
        ],
        figure_planning=None,
    )


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
        "solar_bmu_census_map": map_figure(census=census, number=2),
        "solar_bmu_census_summer_week": example_week_figure(
            census=census, examples=examples, season="summer", number=3
        ),
        "solar_bmu_census_winter_week": example_week_figure(
            census=census, examples=examples, season="winter", number=4
        ),
    }
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, chart in charts.items():
        chart.save(ASSETS_DIR / f"{name}.svg")
        print(f"Wrote {ASSETS_DIR / f'{name}.svg'}")


if __name__ == "__main__":
    main()
